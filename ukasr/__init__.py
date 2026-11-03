"""Measure how well speech recognition handles UK and Irish English accents.

Modules:
    data        load OpenSLR 83 from Hugging Face, sample by speaker, decode audio
    transcribe  run a Whisper checkpoint over batches of 16 kHz audio
    metrics     UK-aware normalisation, WER with speaker-level CIs, error analysis
    run_baseline  CLI: transcribe every accent with one or more models
    report      CLI: turn saved predictions into tables, a chart and a summary
"""

__version__ = "0.1.0"
