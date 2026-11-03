# How well does speech recognition handle UK accents?

This project measures how Whisper's word error rate (WER) changes across UK and Irish English accents, focusing on Northern English. It also checks things a British English product needs that a single WER number hides.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/temialao/uk-accent-asr/blob/main/notebooks/01_baseline.ipynb)

## Why this matters

- **Coverage.** Over 15 million people live in the North of England. If an assistant works noticeably better for a London accent than a Manchester or Newcastle one, a large share of UK users get a worse product.
- **Data imbalance.** Even in this dataset, Southern English has about 8,500 clips and Northern English about 2,850. Public training data for most speech models is weighted towards US and southern British speech.
- **On-device models.** Assistants increasingly run speech recognition on the phone, which means smaller models. This compares four model sizes to see whether the accent gap widens as the model shrinks.

## What it measures

1. **WER by accent and gender** for four Whisper sizes: tiny.en (39M parameters), base.en (74M), small.en (244M) and large-v3-turbo (809M).
2. **The gap against Southern English** in percentage points, with 95% confidence intervals from resampling *speakers*, not clips. Clips from one speaker aren't independent, and speaker variation is the point of an accent comparison.
3. **A same-sentence check**: WER only on sentences read in both accents, so the gap can't come from one group reading harder prompts.
4. **British spelling in the output.** How often "colour" comes back as "color". WER can't see this because the normaliser maps both to one spelling, but in an en-GB product it's a visible defect.
5. **Error analysis**: substitutions that are much more common for Northern than Southern speakers (for example look → luck would point at the FOOT–STRUT merger), and the worst clips to listen to.

## Finding so far: the standard normaliser mis-scores British English

Most published WER figures pass both transcripts through Whisper's English normaliser first. It handles US money and abbreviations, but not some everyday British forms:

| Reference | Transcript | After Whisper's normaliser | Scored as |
|---|---|---|---|
| fifty pence | 50p | `50 pence` vs `50 p` | 1 error |
| ten stone four | 10 st 4 | `10 stone 4` vs `10 saint 4` | 1 error |
| ten pounds fifty | £10.50 | `£10 50` vs `£10.50` | 2 errors |

Each one turns a correct transcript into scored errors. `ukasr/metrics.py` adds three small fixes, and `tests/test_metrics.py` documents both the bug and the fix. Read speech in this dataset rarely contains these forms, so the effect on these numbers is small. It matters more for assistant traffic, where prices, weights and times are common.

## Results

<!-- RESULTS:START -->
_Not run yet. Run `notebooks/01_baseline.ipynb`, then `python -m ukasr.report --update-readme`._
<!-- RESULTS:END -->

## Method

- **Data:** [OpenSLR 83](https://openslr.org/83), crowdsourced read speech from Irish, Midlands, Northern, Scottish, Southern and Welsh English speakers, loaded from the Hugging Face mirror [`ylacombe/english_dialects`](https://huggingface.co/datasets/ylacombe/english_dialects) and resampled to 16 kHz mono.
- **Sampling:** up to 400 clips per accent and gender, taken round-robin across speakers so no one speaker dominates. Every model sees the same clips.
- **Scoring:** corpus WER, (substitutions + deletions + insertions) / reference words, after normalisation.
- **Intervals:** 2,000 bootstrap resamples of speakers. Groups with fewer than 5 speakers are flagged.

**Limitations.** This is read speech, not spontaneous requests. "Northern English" is one label covering Manchester, Liverpool, Leeds and Newcastle, which sound very different. Some groups have only a handful of speakers. The dataset has been public since 2020, so it may be in Whisper's training data. And WER weighs every word equally, though getting a contact's name wrong matters more than missing "the".

## Roadmap

- [x] Evaluation harness, UK-aware scoring, tests
- [ ] Baseline on four model sizes (`notebooks/01_baseline.ipynb`)
- [ ] Listen to and annotate the worst Northern clips: vowels (FOOT–STRUT, BATH), place names, dialect words
- [ ] Fine-tune whisper-small.en with LoRA on Northern speakers, test on held-out speakers, and check Southern WER doesn't get worse
- [ ] UK idiom understanding: about 50 requests such as "set an alarm for half seven" or "add a brew to the shopping list" sent to an LLM with tool calling, scored on the arguments it fills in
- [ ] The same clips through Apple's Speech framework (en-GB) on a Mac, in Swift

## Running it

On Colab, open the notebook from the badge above and choose a T4 GPU. Locally:

```bash
git clone https://github.com/temialao/uk-accent-asr.git && cd uk-accent-asr
pip install -r requirements.txt
python -m pytest -q
python -m ukasr.run_baseline --models openai/whisper-small.en --max-per-config 100 --out results_quick
python -m ukasr.report --results results_quick
```

The first run downloads the dataset (several GB). Runs are resumable: a model and accent pair that has already been transcribed is skipped.

```
ukasr/
  data.py          load OpenSLR 83, sample by speaker, decode audio
  transcribe.py    batched Whisper inference (CUDA, Apple Silicon or CPU)
  metrics.py       UK-aware normaliser, WER with speaker-level CIs, error analysis
  run_baseline.py  CLI: transcribe every accent with each model
  report.py        CLI: tables, chart and results/summary.md
notebooks/01_baseline.ipynb
tests/
```

## Data and licence

The code is MIT-licensed. OpenSLR 83 is licensed CC BY-SA 4.0, and the transcripts in `results/` are shared under the same licence.

> Isin Demirsahin, Oddur Kjartansson, Alexander Gutkin and Clara Rivera. 2020. *Open-source Multi-speaker Corpora of the English Accents in the British Isles.* Proceedings of LREC 2020.

Built by Temi Alao, a software engineer from Manchester.
