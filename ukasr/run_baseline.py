"""Transcribe OpenSLR 83 with one or more Whisper models and save the predictions.

    python -m ukasr.run_baseline --models openai/whisper-small.en --max-per-config 200

Writes results/predictions/<model>/<config>.csv (one row per clip, no audio) and
appends speed figures to results/timings.csv. It is resumable: a (model, config)
pair that already has a CSV is skipped, so a Colab session that dies part-way
picks up where it left off. Every model sees the same sampled clips.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from . import data

DEFAULT_MODELS = [
    "openai/whisper-tiny.en",
    "openai/whisper-base.en",
    "openai/whisper-small.en",
    "openai/whisper-large-v3-turbo",
]


def model_slug(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def _check_sampling(out_dir: Path, settings: dict) -> None:
    """Refuse to mix samples: resuming only makes sense with the same sampling settings."""
    path = out_dir / "sampling.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != settings:
            raise SystemExit(
                f"{out_dir} was created with {saved}, not {settings}. "
                "Use a different --out folder, or the same settings, so every model sees the same clips."
            )
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings, indent=2))


def _default_transcriber(model_id: str, batch_size: int):
    from .transcribe import WhisperTranscriber

    return WhisperTranscriber(model_id, batch_size=batch_size)


def run(
    models: list[str],
    configs: list[str],
    out_dir: Path,
    max_per_config: int | None = None,
    max_per_speaker: int | None = None,
    batch_size: int = 16,
    seed: int = 0,
    chunk_size: int = 256,
    transcriber_factory: Callable = _default_transcriber,
    loader: Callable = data.iter_config,
) -> None:
    """Transcribe every config with every model. The factory and loader are
    injectable so the pipeline can be tested without a GPU or a download."""
    out_dir = Path(out_dir)
    timings_path = out_dir / "timings.csv"
    _check_sampling(out_dir, {"max_per_config": max_per_config, "max_per_speaker": max_per_speaker, "seed": seed})

    for model_id in models:
        model_dir = out_dir / "predictions" / model_slug(model_id)
        todo = [c for c in configs if not (model_dir / f"{c}.csv").exists()]
        if not todo:
            print(f"[{model_id}] all configs done, skipping")
            continue

        print(f"[{model_id}] loading model")
        asr = transcriber_factory(model_id, batch_size)
        model_dir.mkdir(parents=True, exist_ok=True)

        for config in todo:
            accent, gender = data.split_config(config)
            rows, audio_s, wall_s = [], 0.0, 0.0
            for chunk in loader(config, max_per_config, max_per_speaker, seed, chunk_size):
                start = time.perf_counter()
                hyps = asr([r["audio"] for r in chunk])
                wall_s += time.perf_counter() - start
                for r, hyp in zip(chunk, hyps):
                    audio_s += r["duration_s"]
                    rows.append(
                        {
                            "model": model_id,
                            "config": config,
                            "accent": accent,
                            "gender": gender,
                            "speaker_id": r["speaker_id"],
                            "line_id": r["line_id"],
                            "duration_s": round(r["duration_s"], 3),
                            "ref": r["text"],
                            "hyp": hyp,
                        }
                    )

            # write to a temp file then rename, so a crash never leaves a half-written CSV
            tmp = model_dir / f"{config}.csv.tmp"
            pd.DataFrame(rows).to_csv(tmp, index=False)
            tmp.replace(model_dir / f"{config}.csv")

            timing = pd.DataFrame(
                [{"model": model_id, "config": config, "clips": len(rows),
                  "audio_s": round(audio_s, 1), "wall_s": round(wall_s, 2)}]
            )
            timing.to_csv(timings_path, mode="a", header=not timings_path.exists(), index=False)
            print(f"[{model_id}] {config}: {len(rows)} clips, {audio_s/60:.1f} min audio in {wall_s:.0f}s")

        if hasattr(asr, "close"):
            asr.close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--configs", nargs="+", default=None, help="default: every config in the dataset")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--max-per-config", type=int, default=None, help="clips per accent/gender (default: all)")
    p.add_argument("--max-per-speaker", type=int, default=None, help="cap clips per speaker")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    configs = args.configs or data.list_configs()
    print(f"configs: {', '.join(configs)}")
    run(args.models, configs, args.out, args.max_per_config, args.max_per_speaker, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
