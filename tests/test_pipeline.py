"""End-to-end run with a fake loader and a fake model: no GPU, no download."""
import numpy as np
import pandas as pd
import pytest

from ukasr import report, run_baseline

SENTENCES = [f"please look at the bath number {i} and pick a favourite colour" for i in range(30)]
SPEAKERS = {"southern_male": 8, "southern_female": 8, "northern_male": 6, "northern_female": 4, "irish_male": 1}

CORPUS = []  # every synthetic clip; the "audio" is just its index into this list
for config, n_spk in SPEAKERS.items():
    for spk in range(n_spk):
        for k in range(12):
            CORPUS.append({"config": config, "speaker_id": hash((config, spk)) % 10_000,
                           "text": SENTENCES[(spk * 7 + k) % len(SENTENCES)]})


def fake_loader(config, max_utts, max_per_speaker, seed, chunk_size):
    rows = [dict(r, line_id=f"{config}_{i}", audio=np.array([i], dtype="float32"), duration_s=4.0)
            for i, r in enumerate(CORPUS) if r["config"] == config][:max_utts]
    for start in range(0, len(rows), chunk_size):
        yield rows[start:start + chunk_size]


class FakeWhisper:
    """Mishears the FOOT vowel for northern speakers; small model also americanises."""
    created = 0

    def __init__(self, model_id, batch_size):
        FakeWhisper.created += 1
        self.small = "tiny" in model_id

    def __call__(self, arrays):
        out = []
        for a in arrays:
            clip = CORPUS[int(a[0])]
            text = clip["text"]
            if clip["config"].startswith("northern") and int(a[0]) % 2:
                text = text.replace("look", "luck")
            if self.small:
                text = text.replace("colour", "color").replace("favourite", "favorite")
            out.append(text)
        return out


def test_run_then_report(tmp_path):
    models = ["openai/whisper-tiny.en", "openai/whisper-small.en"]
    run_baseline.run(models, list(SPEAKERS), tmp_path, max_per_config=200, chunk_size=10,
                     transcriber_factory=FakeWhisper, loader=fake_loader)
    assert FakeWhisper.created == 2
    preds = report.load_predictions(tmp_path)
    assert len(preds) == 2 * len(CORPUS)

    # second run resumes: nothing left to do, no model loaded
    run_baseline.run(models, list(SPEAKERS), tmp_path, max_per_config=200,
                     transcriber_factory=FakeWhisper, loader=fake_loader)
    assert FakeWhisper.created == 2

    # different sampling settings in the same folder are refused
    with pytest.raises(SystemExit):
        run_baseline.run(models, list(SPEAKERS), tmp_path, max_per_config=50,
                         transcriber_factory=FakeWhisper, loader=fake_loader)

    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\noutro\n")
    report.main(["--results", str(tmp_path), "--update-readme", "--readme", str(readme)])

    summary = (tmp_path / "summary.md").read_text()
    assert "| Northern |" in summary and "Irish †" in summary
    assert "| look | luck |" in summary
    assert "colour→color" in summary
    assert (tmp_path / "wer_by_accent.png").stat().st_size > 10_000
    assert (tmp_path / "worst" / "whisper-small.en__northern.csv").exists()
    text = readme.read_text()
    assert "old" not in text and "intro" in text and "outro" in text and "| Northern |" in text

    timings = pd.read_csv(tmp_path / "timings.csv")
    assert len(timings) == 2 * len(SPEAKERS)
