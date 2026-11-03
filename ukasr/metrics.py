"""Scoring and error analysis. Needs no torch, so it runs (and is tested) anywhere.

The input everywhere is a predictions table with at least these columns:
    model, accent, gender, speaker_id, ref, hyp
`score()` adds normalised text and per-utterance error counts; everything else
works on its output.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from importlib.resources import files

import jiwer
import numpy as np
import pandas as pd
from whisper_normalizer.english import EnglishTextNormalizer


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
class UKTextNormalizer:
    """Whisper's English normaliser, plus fixes for British forms it mis-scores.

    Whisper's normaliser is what most published WER figures use. It copes with
    US money and abbreviations, but not with some everyday UK forms:

        "50p"              -> "50 p"          never matches "fifty pence"
        "10 st 4"          -> "10 saint 4"    (ten stone four)
        "ten pounds fifty" -> "£10 50"        never matches "£10.50"

    Each of these turns a correct transcript into scored errors. The fixes
    below are small and covered by tests. Whisper's normaliser also rewrites
    British spellings as American ones (colour -> color) on both sides, which is
    fair for WER; output spelling is measured separately in `americanisation()`.
    """

    _PENCE = re.compile(r"\b(\d+) ?p\b(?!\.?m\b)")  # 50p, 50 p (but not 3 p.m.)
    _STONE = re.compile(r"\b(\d+) st\b")  # needs the space, so "1st" is left alone
    _MONEY = re.compile(r"([£$€])(\d+) (\d{2})\b")  # £10 50 -> £10.50

    def __init__(self, uk_fixes: bool = True):
        self.base = EnglishTextNormalizer()
        self.uk_fixes = uk_fixes

    def __call__(self, text: str) -> str:
        text = str(text)
        if self.uk_fixes:
            text = self._PENCE.sub(r"\1 pence", text)
            text = self._STONE.sub(r"\1 stone", text)
        text = self.base(text)
        if self.uk_fixes:
            text = self._MONEY.sub(r"\1\2.\3", text)
        return text


# ---------------------------------------------------------------------------
# Per-utterance scoring
# ---------------------------------------------------------------------------
def _counts(ref: str, hyp: str) -> tuple[int, int, int, int]:
    """(hits, substitutions, deletions, insertions) for one utterance."""
    if not hyp.strip():
        return 0, 0, len(ref.split()), 0
    out = jiwer.process_words(ref, hyp)
    return out.hits, out.substitutions, out.deletions, out.insertions


def score(df: pd.DataFrame, normalizer: UKTextNormalizer | None = None) -> pd.DataFrame:
    """Add ref_norm, hyp_norm, hits, subs, dels, ins, n_ref and n_err columns.

    Rows whose reference normalises to nothing are dropped.
    """
    norm = normalizer or UKTextNormalizer()
    out = df.reset_index(drop=True)  # duplicate index labels (e.g. after concat) would break the join below
    out["hyp"] = out["hyp"].fillna("").astype(str)
    out["ref_norm"] = out["ref"].map(norm)
    out["hyp_norm"] = out["hyp"].map(norm)
    out = out[out["ref_norm"].str.strip() != ""].copy()
    counts = pd.DataFrame(
        [_counts(r, h) for r, h in zip(out["ref_norm"], out["hyp_norm"])],
        columns=["hits", "subs", "dels", "ins"],
        index=out.index,
    )
    out = out.join(counts)
    out["n_ref"] = out["hits"] + out["subs"] + out["dels"]
    out["n_err"] = out["subs"] + out["dels"] + out["ins"]
    return out


# ---------------------------------------------------------------------------
# WER with speaker-level confidence intervals
# ---------------------------------------------------------------------------
def _per_speaker(g: pd.DataFrame) -> pd.DataFrame:
    return g.groupby("speaker_id")[["n_err", "n_ref"]].sum()


def _bootstrap(spk: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Corpus WER recomputed on speakers resampled with replacement.

    Resampling speakers rather than utterances gives honest intervals: clips
    from one speaker are not independent, and speaker variation is exactly
    what an accent comparison is about.
    """
    err, words = spk["n_err"].to_numpy(), spk["n_ref"].to_numpy()
    idx = rng.integers(0, len(err), size=(n_boot, len(err)))
    return err[idx].sum(axis=1) / words[idx].sum(axis=1)


def _as_tuple(key) -> tuple:
    return key if isinstance(key, tuple) else (key,)


def wer_table(
    scored: pd.DataFrame,
    by: tuple[str, ...] = ("model", "accent"),
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Corpus WER per group with a 95% speaker-bootstrap interval."""
    rows = []
    for key, g in scored.groupby(list(by)):
        spk = _per_speaker(g)
        samples = _bootstrap(spk, n_boot, np.random.default_rng(seed))
        rows.append(
            {
                **dict(zip(by, _as_tuple(key))),
                "wer": spk["n_err"].sum() / spk["n_ref"].sum(),
                "ci_low": float(np.percentile(samples, 2.5)),
                "ci_high": float(np.percentile(samples, 97.5)),
                "speakers": len(spk),
                "utterances": len(g),
                "words": int(spk["n_ref"].sum()),
            }
        )
    return pd.DataFrame(rows)


def accent_gap(
    scored: pd.DataFrame,
    accent: str,
    reference: str = "southern",
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Per model: WER(accent) minus WER(reference) in percentage points, with a 95% CI.

    If the interval excludes zero, the gap is unlikely to be down to which
    speakers happened to be recorded.
    """
    rows = []
    for model, g in scored.groupby("model"):
        a = _per_speaker(g[g["accent"] == accent])
        b = _per_speaker(g[g["accent"] == reference])
        if a.empty or b.empty:
            continue
        rng = np.random.default_rng(seed)
        diff = _bootstrap(a, n_boot, rng) - _bootstrap(b, n_boot, rng)
        wer_a = a["n_err"].sum() / a["n_ref"].sum()
        wer_b = b["n_err"].sum() / b["n_ref"].sum()
        rows.append(
            {
                "model": model,
                "accent": accent,
                "reference": reference,
                "wer_accent": wer_a,
                "wer_reference": wer_b,
                "gap_points": 100 * (wer_a - wer_b),
                "ci_low": 100 * float(np.percentile(diff, 2.5)),
                "ci_high": 100 * float(np.percentile(diff, 97.5)),
                "relative_gap": wer_a / wer_b - 1 if wer_b > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def matched_text_subset(scored: pd.DataFrame, accents: list[str]) -> pd.DataFrame:
    """Keep only sentences read in every one of `accents`.

    Comparing WER on the same sentences rules out the explanation that one
    group simply read harder prompts.
    """
    texts = [set(scored.loc[scored["accent"] == a, "ref_norm"]) for a in accents]
    common = set.intersection(*texts) if texts else set()
    keep = scored["accent"].isin(accents) & scored["ref_norm"].isin(common)
    return scored[keep]


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------
def substitution_pairs(ref_norm: str, hyp_norm: str) -> list[tuple[str, str]]:
    """Word-level (reference, hypothesis) substitutions, e.g. [('look', 'luck')]."""
    if not hyp_norm.strip():
        return []
    out = jiwer.process_words(ref_norm, hyp_norm)
    ref_words, hyp_words = out.references[0], out.hypotheses[0]
    pairs = []
    for chunk in out.alignments[0]:
        if chunk.type == "substitute":
            for i, j in zip(
                range(chunk.ref_start_idx, chunk.ref_end_idx),
                range(chunk.hyp_start_idx, chunk.hyp_end_idx),
            ):
                pairs.append((ref_words[i], hyp_words[j]))
    return pairs


def _substitution_counts(g: pd.DataFrame) -> Counter:
    counts: Counter = Counter()
    for r, h in zip(g["ref_norm"], g["hyp_norm"]):
        counts.update(substitution_pairs(r, h))
    return counts


def distinctive_substitutions(
    scored: pd.DataFrame,
    model: str,
    accent: str,
    reference: str = "southern",
    min_count: int = 2,
    k: int = 15,
) -> pd.DataFrame:
    """Substitutions that happen relatively more often for `accent` than `reference`.

    Rates are per 10,000 reference words. This is the list to read with a
    phonetician's (or a local's) ear: e.g. look -> luck would point at the
    northern FOOT-STRUT merger.
    """
    g = scored[scored["model"] == model]
    ga, gb = g[g["accent"] == accent], g[g["accent"] == reference]
    if ga.empty or gb.empty:
        return pd.DataFrame()
    ca, cb = _substitution_counts(ga), _substitution_counts(gb)
    wa, wb = ga["n_ref"].sum(), gb["n_ref"].sum()
    rows = []
    for (ref_w, hyp_w), n in ca.items():
        if n < min_count:
            continue
        rate_a = 1e4 * n / wa
        rate_b = 1e4 * (cb[(ref_w, hyp_w)] + 0.5) / wb  # +0.5 so unseen pairs don't divide by zero
        rows.append(
            {
                "ref_word": ref_w,
                "hyp_word": hyp_w,
                f"count_{accent}": n,
                f"count_{reference}": cb[(ref_w, hyp_w)],
                f"per_10k_{accent}": rate_a,
                "ratio": rate_a / rate_b,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["ratio", f"count_{accent}"], ascending=False).head(k)


def worst_utterances(scored: pd.DataFrame, model: str, accent: str, k: int = 25) -> pd.DataFrame:
    """Highest-WER clips for one model and accent: the shortlist to listen to."""
    g = scored[(scored["model"] == model) & (scored["accent"] == accent)].copy()
    g["wer"] = g["n_err"] / g["n_ref"]
    cols = ["config", "speaker_id", "line_id", "ref", "hyp", "wer"]
    return g.sort_values(["wer", "n_err"], ascending=False).head(k)[[c for c in cols if c in g]]


# ---------------------------------------------------------------------------
# en-GB output spelling
# ---------------------------------------------------------------------------
_WORD = re.compile(r"[a-z]+")


def british_to_american() -> dict[str, str]:
    """Whisper's own British -> American spelling map (about 1,700 words)."""
    path = files("whisper_normalizer").joinpath("normalizers/english.json")
    mapping = json.loads(path.read_text())
    return {uk: us for uk, us in mapping.items() if uk != us and uk.isalpha() and us.isalpha()}


def americanisation(scored: pd.DataFrame, by: tuple[str, ...] = ("model",)) -> pd.DataFrame:
    """How often British spellings in the reference come back American.

    Counts reference words such as "colour" where the transcript contains either
    the British form or the American one ("color"), and reports the American
    share. WER hides this because the normaliser maps both to one spelling, but
    in an en-GB product "color" on screen is a visible defect.
    """
    uk2us = british_to_american()
    rows = []
    for key, g in scored.groupby(list(by)):
        kept = americanised = 0
        examples: Counter = Counter()
        for ref, hyp in zip(g["ref"], g["hyp"]):
            hyp_words = set(_WORD.findall(str(hyp).lower()))
            for w in _WORD.findall(str(ref).lower()):
                if w not in uk2us:
                    continue
                if w in hyp_words:
                    kept += 1
                elif uk2us[w] in hyp_words:
                    americanised += 1
                    examples[f"{w}→{uk2us[w]}"] += 1
        total = kept + americanised
        rows.append(
            {
                **dict(zip(by, _as_tuple(key))),
                "british_words_seen": total,
                "american_spellings": americanised,
                "rate": americanised / total if total else float("nan"),
                "examples": ", ".join(e for e, _ in examples.most_common(5)),
            }
        )
    return pd.DataFrame(rows)
