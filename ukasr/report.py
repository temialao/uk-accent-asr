"""Turn saved predictions into tables, a chart and results/summary.md.

    python -m ukasr.report                    # summary.md, wer_by_accent.png, worst/*.csv
    python -m ukasr.report --update-readme    # also refresh the results block in README.md

Options let you change the accent you're focusing on and the one you compare
against (default: northern vs southern).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from . import metrics
from .run_baseline import DEFAULT_MODELS, model_slug

FEW_SPEAKERS = 5  # flag groups with fewer speakers than this


# ---------------------------------------------------------------------------
# Loading and small formatting helpers
# ---------------------------------------------------------------------------
def load_predictions(results: Path) -> pd.DataFrame:
    paths = sorted((results / "predictions").glob("*/*.csv"))
    if not paths:
        raise SystemExit(f"No predictions under {results / 'predictions'}; run ukasr.run_baseline first.")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def model_order(models) -> list[str]:
    """Known Whisper checkpoints smallest first, anything else after."""
    known = [m for m in DEFAULT_MODELS if m in set(models)]
    return known + sorted(set(models) - set(known))


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def pct_ci(row) -> str:
    return f"{pct(row['wer'])} ({100 * row['ci_low']:.1f}–{100 * row['ci_high']:.1f})"


def signed_ci(row) -> str:
    return f"{row['gap_points']:+.1f} ({row['ci_low']:+.1f} to {row['ci_high']:+.1f})"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def wer_by_accent(scored: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Wide table (one row per accent, one column per model) plus accents sorted by best-model WER."""
    table = metrics.wer_table(scored, by=("model", "accent"))
    best = best_model(scored, models)
    order = table[table["model"] == best].sort_values("wer")["accent"].tolist()

    wide = []
    for accent in order:
        sub = table[table["accent"] == accent].set_index("model")
        speakers = int(sub["speakers"].max())
        label = accent.capitalize() + (" †" if speakers < FEW_SPEAKERS else "")
        row = {"Accent": label, "Speakers": speakers, "Clips": int(sub["utterances"].max())}
        for m in models:
            row[model_slug(m)] = pct_ci(sub.loc[m]) if m in sub.index else "–"
        wide.append(row)
    return pd.DataFrame(wide), order


def best_model(scored: pd.DataFrame, models: list[str]) -> str:
    overall = scored.groupby("model")[["n_err", "n_ref"]].sum()
    return (overall["n_err"] / overall["n_ref"]).loc[models].idxmin()


def gap_table(scored, models, accents, reference) -> pd.DataFrame:
    rows = []
    for accent in accents:
        if accent == reference:
            continue
        gaps = metrics.accent_gap(scored, accent, reference).set_index("model")
        row = {"Accent": accent.capitalize()}
        for m in models:
            row[model_slug(m)] = signed_ci(gaps.loc[m]) if m in gaps.index else "–"
        rows.append(row)
    return pd.DataFrame(rows)


def matched_section(scored, models, focus, reference, min_sentences=20) -> str:
    subset = metrics.matched_text_subset(scored, [focus, reference])
    n_sentences = subset["ref_norm"].nunique()
    if n_sentences < min_sentences:
        return (
            f"Only {n_sentences} sentences were read in both {focus.capitalize()} and "
            f"{reference.capitalize()} English in this sample, too few for a same-sentence comparison. "
            "Try a larger --max-per-config."
        )
    table = metrics.wer_table(subset, by=("model", "accent"))
    rows = []
    for m in models:
        sub = table[table["model"] == m].set_index("accent")
        if focus in sub.index and reference in sub.index:
            rows.append({"Model": model_slug(m), focus.capitalize(): pct_ci(sub.loc[focus]),
                         reference.capitalize(): pct_ci(sub.loc[reference])})
    return (
        f"Restricted to the {n_sentences} sentences read in both accents, so neither group "
        "can have had harder prompts:\n\n" + md_table(pd.DataFrame(rows))
    )


def gender_table(scored, model, accents) -> pd.DataFrame:
    table = metrics.wer_table(scored[scored["model"] == model], by=("accent", "gender"))
    rows = []
    for accent in accents:
        sub = table[table["accent"] == accent].set_index("gender")
        rows.append({"Accent": accent.capitalize(),
                     **{g: pct_ci(sub.loc[g]) if g in sub.index else "–" for g in ("female", "male")}})
    return pd.DataFrame(rows)


def spelling_table(scored, models) -> pd.DataFrame:
    table = metrics.americanisation(scored).set_index("model")
    rows = []
    for m in models:
        r = table.loc[m]
        rate = "–" if r["british_words_seen"] == 0 else pct(r["rate"])
        rows.append({"Model": model_slug(m), "British words seen": int(r["british_words_seen"]),
                     "Returned American": int(r["american_spellings"]), "Rate": rate,
                     "Examples": r["examples"] or "–"})
    return pd.DataFrame(rows)


def speed_table(results: Path, models) -> pd.DataFrame | None:
    path = results / "timings.csv"
    if not path.exists():
        return None
    t = pd.read_csv(path).drop_duplicates(["model", "config"], keep="last")
    t = t.groupby("model")[["audio_s", "wall_s"]].sum()
    rows = [{"Model": model_slug(m), "Audio (min)": f"{t.loc[m, 'audio_s'] / 60:.1f}",
             "Compute (s)": f"{t.loc[m, 'wall_s']:.0f}",
             "× real time": f"{t.loc[m, 'audio_s'] / t.loc[m, 'wall_s']:.0f}" if t.loc[m, "wall_s"] >= 1 else "–"}
            for m in models if m in t.index]
    return pd.DataFrame(rows)


def plot_wer(scored, models, accents, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    table = metrics.wer_table(scored, by=("model", "accent"))
    colours = ["#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b", "#252525"]
    x = np.arange(len(accents))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(accents)), 4))
    for i, m in enumerate(models):
        sub = table[table["model"] == m].set_index("accent").reindex(accents)
        y = 100 * sub["wer"].to_numpy()
        err = 100 * np.vstack([sub["wer"] - sub["ci_low"], sub["ci_high"] - sub["wer"]]).astype(float)
        ax.bar(x + (i - (len(models) - 1) / 2) * width, y, width, yerr=err, capsize=2,
               label=model_slug(m), color=colours[i % len(colours)], error_kw={"lw": 0.8})
    speakers = table.groupby("accent")["speakers"].max()
    ax.set_xticks(x, [a.capitalize() + (" †" if speakers[a] < FEW_SPEAKERS else "") for a in accents])
    ax.set_ylabel("Word error rate (%)")
    ax.set_title("Whisper word error rate by accent (OpenSLR 83, 95% speaker-bootstrap CI)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
def build(results: Path, focus: str, reference: str) -> tuple[str, str]:
    """Write chart and worst-clip CSVs; return (full summary, short README block)."""
    scored = metrics.score(load_predictions(results))
    models = model_order(scored["model"].unique())
    best = best_model(scored, models)
    smallest = models[0]

    wide, accents = wer_by_accent(scored, models)
    plot_wer(scored, models, accents, results / "wer_by_accent.png")

    worst_dir = results / "worst"
    worst_dir.mkdir(exist_ok=True)
    for m in models:
        for accent in (focus, reference):
            metrics.worst_utterances(scored, m, accent).to_csv(
                worst_dir / f"{model_slug(m)}__{accent}.csv", index=False)

    clips = scored[scored["model"] == best]
    intro = (
        f"{len(clips):,} clips ({clips['duration_s'].sum() / 3600:.1f} hours) from "
        f"{clips.groupby('accent')['speaker_id'].nunique().sum()} speakers, scored with Whisper's "
        "English normaliser plus UK fixes (see `ukasr/metrics.py`). Brackets are 95% intervals "
        "from resampling speakers. † = fewer than 5 speakers, treat as anecdotal."
    )
    wer_md = md_table(wide)
    gaps_md = md_table(gap_table(scored, models, accents, reference))
    spelling_md = md_table(spelling_table(scored, models))

    parts = [
        "# Results", intro,
        "## Word error rate by accent", wer_md, "![WER by accent](wer_by_accent.png)",
        f"## Gap versus {reference.capitalize()} English (percentage points)",
        "Positive means more errors than for Southern speakers. An interval that excludes 0 is "
        "unlikely to be down to which speakers happened to be recorded.", gaps_md,
        f"## Same sentences only: {focus.capitalize()} vs {reference.capitalize()}",
        matched_section(scored, models, focus, reference),
        f"## By gender ({model_slug(best)})", md_table(gender_table(scored, best, accents)),
        "## British spellings returned in American form",
        "WER can't see this, because the normaliser maps both spellings to one. In an en-GB "
        "product, \"color\" on screen is still a defect.", spelling_md,
    ]
    for m in dict.fromkeys([smallest, best]):
        distinct = metrics.distinctive_substitutions(scored, m, focus, reference, k=12)
        parts += [f"## Substitutions more common for {focus.capitalize()} than "
                  f"{reference.capitalize()} speakers ({model_slug(m)})",
                  md_table(distinct.round(2)) if not distinct.empty else "None with at least 2 occurrences."]
    speed = speed_table(results, models)
    if speed is not None:
        parts += ["## Speed", "Depends on the GPU; useful for comparing models on the same run.", md_table(speed)]
    parts += ["## Clips to listen to",
              f"The 25 worst clips per model for {focus} and {reference} speakers are in `results/worst/`."]
    summary = "\n\n".join(parts) + "\n"

    focus_gap = gap_table(scored, models, [focus], reference)
    readme_block = "\n\n".join([
        intro, wer_md, "![WER by accent](results/wer_by_accent.png)",
        f"**{focus.capitalize()} minus {reference.capitalize()} WER** (percentage points):", md_table(focus_gap),
        "**British spellings returned in American form:**", spelling_md,
        "Full tables, including the same-sentence check and error analysis: "
        "[results/summary.md](results/summary.md)",
    ])
    return summary, readme_block


def update_readme(readme: Path, block: str) -> None:
    text = readme.read_text()
    pattern = re.compile(r"(<!-- RESULTS:START -->)(.*?)(<!-- RESULTS:END -->)", re.S)
    if not pattern.search(text):
        raise SystemExit("README has no <!-- RESULTS:START --> / <!-- RESULTS:END --> markers")
    readme.write_text(pattern.sub(lambda m: f"{m.group(1)}\n{block}\n{m.group(3)}", text))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--focus", default="northern", help="accent to analyse in depth")
    p.add_argument("--reference", default="southern", help="accent to compare against")
    p.add_argument("--update-readme", action="store_true")
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    args = p.parse_args(argv)

    summary, block = build(args.results, args.focus, args.reference)
    (args.results / "summary.md").write_text(summary)
    print(f"wrote {args.results / 'summary.md'} and {args.results / 'wer_by_accent.png'}")
    if args.update_readme:
        update_readme(args.readme, block)
        print(f"updated {args.readme}")


if __name__ == "__main__":
    main()
