"""Load OpenSLR 83 (UK and Ireland English dialects) from its Hugging Face mirror.

Each config is one accent and gender, e.g. "northern_male". Every clip has a
speaker_id, which matters: pooled WER is easily dominated by a few prolific
speakers, so sampling and confidence intervals are both done per speaker.

Audio is decoded here (soundfile + scipy) rather than by `datasets`, so the
code doesn't depend on which audio backend a given `datasets` version uses.
"""
from __future__ import annotations

import io
from collections.abc import Iterator
from math import gcd

import numpy as np

DATASET_ID = "ylacombe/english_dialects"
TARGET_SR = 16_000


def list_configs() -> list[str]:
    """All accent/gender configs in the dataset, e.g. ['irish_male', 'midlands_female', ...]."""
    from datasets import get_dataset_config_names

    return sorted(get_dataset_config_names(DATASET_ID))


def split_config(config: str) -> tuple[str, str]:
    """'northern_male' -> ('northern', 'male')."""
    accent, _, gender = config.rpartition("_")
    return accent, gender


def sample_by_speaker(
    speaker_ids: list,
    max_utts: int | None = None,
    max_per_speaker: int | None = None,
    seed: int = 0,
) -> list[int]:
    """Choose row indices so that no single speaker dominates the sample.

    Shuffles each speaker's rows, caps them at `max_per_speaker`, then takes rows
    round-robin across speakers until `max_utts` are chosen. With no limits it
    returns every row. The same seed gives the same sample, so every model is
    scored on identical clips.
    """
    rng = np.random.default_rng(seed)
    by_speaker: dict = {}
    for idx in rng.permutation(len(speaker_ids)):
        by_speaker.setdefault(speaker_ids[idx], []).append(int(idx))

    queues = [rows[:max_per_speaker] if max_per_speaker else rows for rows in by_speaker.values()]
    chosen: list[int] = []
    depth = 0
    while any(depth < len(q) for q in queues):
        for q in queues:
            if depth < len(q):
                chosen.append(q[depth])
                if max_utts and len(chosen) >= max_utts:
                    return sorted(chosen)
        depth += 1
    return sorted(chosen)


def decode_audio(audio: dict) -> np.ndarray:
    """Turn an undecoded audio cell ({'bytes', 'path'}) into mono float32 at 16 kHz."""
    import soundfile as sf
    from scipy.signal import resample_poly

    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    samples, sr = sf.read(source, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        samples = resample_poly(samples, TARGET_SR // g, sr // g).astype(np.float32)
    return samples


def iter_config(
    config: str,
    max_utts: int | None = None,
    max_per_speaker: int | None = None,
    seed: int = 0,
    chunk_size: int = 256,
) -> Iterator[list[dict]]:
    """Yield the sampled clips of one config in chunks, with audio decoded.

    Each row: line_id, speaker_id, text, audio (np.ndarray at 16 kHz), duration_s.
    Chunking keeps memory flat on large configs such as southern_male.
    """
    from datasets import Audio, load_dataset

    ds = load_dataset(DATASET_ID, config, split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    indices = sample_by_speaker(list(ds["speaker_id"]), max_utts, max_per_speaker, seed)
    ds = ds.select(indices)

    chunk: list[dict] = []
    for ex in ds:
        audio = decode_audio(ex["audio"])
        chunk.append(
            {
                "line_id": ex["line_id"],
                "speaker_id": int(ex["speaker_id"]),
                "text": ex["text"],
                "audio": audio,
                "duration_s": len(audio) / TARGET_SR,
            }
        )
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
