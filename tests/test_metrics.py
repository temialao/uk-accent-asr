import io

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from ukasr import data, metrics


# --- UK normaliser fixes -----------------------------------------------------
@pytest.mark.parametrize(
    "ref, hyp",
    [
        ("fifty pence", "50p"),
        ("fifty pence", "50 p"),
        ("ten stone four", "10 st 4"),
        ("ten pounds fifty", "£10.50"),
        ("my favourite colour", "my favorite color"),  # spelling is WER-neutral by design
    ],
)
def test_uk_forms_match_after_normalising(ref, hyp):
    norm = metrics.UKTextNormalizer()
    assert norm(ref) == norm(hyp)


@pytest.mark.parametrize("ref, hyp", [("fifty pence", "50p"), ("ten stone four", "10 st 4"),
                                      ("ten pounds fifty", "£10.50")])
def test_plain_whisper_normaliser_misses_uk_forms(ref, hyp):
    """Documents the bug the UK fixes exist for."""
    plain = metrics.UKTextNormalizer(uk_fixes=False)
    assert plain(ref) != plain(hyp)


@pytest.mark.parametrize("text", ["see you at 3 p.m.", "the 1st of May", "3pm"])
def test_uk_fixes_leave_lookalikes_alone(text):
    assert metrics.UKTextNormalizer()(text) == metrics.UKTextNormalizer(uk_fixes=False)(text)


# --- scoring -----------------------------------------------------------------
def frame(rows):
    return pd.DataFrame(rows, columns=["model", "accent", "gender", "speaker_id", "ref", "hyp"])


def test_score_counts_errors():
    s = metrics.score(frame([["m", "northern", "male", 1, "I would look at the bus", "I would luck at a bus stop"]]))
    row = s.iloc[0]
    assert (row.subs, row.dels, row.ins, row.n_ref, row.n_err) == (2, 0, 1, 6, 3)


def test_empty_hypothesis_is_all_deletions():
    row = metrics.score(frame([["m", "northern", "male", 1, "three words here", None]])).iloc[0]
    assert (row.dels, row.n_err, row.n_ref) == (3, 3, 3)


def test_substitution_pairs():
    assert metrics.substitution_pairs("i would look at the bus", "i would luck at a bus stop") == [
        ("look", "luck"), ("the", "a")]


# --- WER and intervals -------------------------------------------------------
def synthetic(n_speakers=10, clips=10, error_rate=0.1, accent="southern", seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for spk in range(n_speakers):
        for _ in range(clips):
            words = ["word"] * 10
            hyp = [w if rng.random() > error_rate else "other" for w in words]
            rows.append(["m", accent, "male", f"{accent}{spk}", " ".join(words), " ".join(hyp)])
    return frame(rows)


def test_wer_table_interval_brackets_estimate():
    t = metrics.wer_table(metrics.score(synthetic()))
    row = t.iloc[0]
    assert row.ci_low <= row.wer <= row.ci_high
    assert (row.speakers, row.utterances, row.words) == (10, 100, 1000)


def test_single_speaker_interval_has_no_width():
    row = metrics.wer_table(metrics.score(synthetic(n_speakers=1))).iloc[0]
    assert row.ci_low == row.ci_high == row.wer


def test_accent_gap_detects_real_difference():
    df = pd.concat([synthetic(error_rate=0.05, accent="southern"),
                    synthetic(error_rate=0.25, accent="northern", seed=1)])
    gap = metrics.accent_gap(metrics.score(df), "northern", "southern").iloc[0]
    assert gap.gap_points > 0 and gap.ci_low > 0


def test_matched_text_subset_keeps_shared_sentences():
    df = metrics.score(frame([
        ["m", "northern", "male", 1, "shared sentence", "shared sentence"],
        ["m", "northern", "male", 1, "only north", "only north"],
        ["m", "southern", "male", 2, "shared sentence", "shared sentence"],
    ]))
    assert set(metrics.matched_text_subset(df, ["northern", "southern"])["ref_norm"]) == {"shared sentence"}


def test_distinctive_substitutions_ranks_accent_specific_errors_first():
    rows = [["m", "northern", "male", 1, "look at the bath", "luck at the bath"]] * 5
    rows += [["m", "southern", "male", 2, "look at the bath", "look at the bath"]] * 5
    top = metrics.distinctive_substitutions(metrics.score(frame(rows)), "m", "northern").iloc[0]
    assert (top.ref_word, top.hyp_word, top.count_northern) == ("look", "luck", 5)


def test_americanisation_counts_us_spellings():
    df = metrics.score(frame([
        ["m", "southern", "male", 1, "My favourite colour is grey", "My favorite color is grey"],
        ["m", "southern", "male", 1, "The centre", "The centre"],
    ]))
    row = metrics.americanisation(df).iloc[0]
    assert (row.british_words_seen, row.american_spellings) == (4, 2)
    assert "colour→color" in row.examples


# --- data helpers ------------------------------------------------------------
def test_sample_by_speaker_caps_and_spreads():
    speakers = ["a"] * 100 + ["b"] * 5 + ["c"] * 5
    idx = data.sample_by_speaker(speakers, max_utts=12, seed=1)
    picked = pd.Series([speakers[i] for i in idx]).value_counts()
    assert len(idx) == 12 and picked["b"] == 4 and picked["c"] == 4  # round-robin, not 12 from "a"
    assert data.sample_by_speaker(speakers, max_utts=12, seed=1) == idx  # deterministic
    capped = data.sample_by_speaker(speakers, max_per_speaker=3)
    assert len(capped) == 9
    assert len(data.sample_by_speaker(speakers)) == 110


def test_decode_audio_resamples_to_16k_mono():
    sr = 48_000
    t = np.arange(sr) / sr
    stereo = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV")
    audio = data.decode_audio({"bytes": buf.getvalue(), "path": "x.wav"})
    assert audio.ndim == 1 and len(audio) == 16_000 and audio.dtype == np.float32


def test_split_config():
    assert data.split_config("northern_male") == ("northern", "male")
