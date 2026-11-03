"""Run a Whisper checkpoint from Hugging Face over batches of 16 kHz audio."""
from __future__ import annotations

import numpy as np


def pick_device() -> str:
    """CUDA if available (Colab), then Apple Silicon (MPS), then CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class WhisperTranscriber:
    """Callable that maps a list of audio arrays to a list of transcripts.

    English-only checkpoints (names ending ".en") take no language argument;
    multilingual ones are pinned to English so they never drift into
    translation or language ID mistakes.
    """

    def __init__(self, model_id: str, device: str | None = None, batch_size: int = 16):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self._torch = torch
        self.model_id = model_id
        self.device = device or pick_device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.batch_size = batch_size
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(model_id)
            .to(device=self.device, dtype=self.dtype)
            .eval()
        )
        self.generate_kwargs = {} if model_id.endswith(".en") else {"language": "en", "task": "transcribe"}

    def __call__(self, arrays: list[np.ndarray]) -> list[str]:
        texts: list[str] = []
        for start in range(0, len(arrays), self.batch_size):
            batch = arrays[start : start + self.batch_size]
            features = self.processor.feature_extractor(
                batch, sampling_rate=16_000, return_tensors="pt"
            ).input_features.to(device=self.device, dtype=self.dtype)
            with self._torch.inference_mode():
                ids = self.model.generate(features, max_new_tokens=128, **self.generate_kwargs)
            texts += [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]
        return texts

    def close(self) -> None:
        """Free GPU memory before loading the next model."""
        del self.model
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
