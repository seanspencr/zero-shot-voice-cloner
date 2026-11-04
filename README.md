# Zero-Shot Voice Conversion (GAN) — Mock-Data Base Architecture

Base architecture + training pipeline for a **zero-shot voice conversion / timbre transfer** GAN.
The final system will take *content* (what is said, from Coqui TTS) and deliver it in the
*identity* of a target speaker (timbre from a single enrollment sample).

The pipeline runs on **real data**: LibriTTS + LibriTTS-R is extracted under `datasets/libritts/`,
the mel target comes from the **real pretrained HiFi-GAN vocoder**, the content features are **real
HuBERT** (layer 9) and the speaker embeddings are **real resemblyzer d-vectors** — all installed and
verified. `python train.py` with no arguments still runs the random-tensor shape test.

## Run it: pick a batch file

| File | Data | Time on this CPU | Notes |
|---|---|---|---|
| `train-short.bat` | 400 utterances × 2 epochs | ~10 min | pipeline check; stops at `--max-minutes 10` |
| `train-medium.bat` | 2,000 utterances × 2 epochs | ~1.3 h | asks for `YES` before starting |
| `train-full.bat` | **all 33,232** utterances × 1 epoch | ~10 h | asks for `YES`; no resume exists |

Double-click them, or run from a terminal. Each one: picks the conda interpreter, uses real
HuBERT + resemblyzer + 4 s segments (batch 4), validates on `dev-clean` every epoch, renders a sample
wav, saves `checkpoints/latest.pt` + `best.pt`, tees everything to `logs/train-*.log`, and prints the
exit code. **Any extra argument overrides the defaults**, e.g.

```powershell
.\train-short.bat --epochs 4 --max-minutes 25
.\train-medium.bat --segment-seconds 2 --epochs 6
.\train-full.bat --max-utterances 8000
```

Edit the `PY=` line at the top of a file if your conda env lives elsewhere.

```
            ┌──────────────────┐
  text ───► │  Coqui TTS       │ ──► reference audio            (existing, testing.py)
            └──────────────────┘
                                          │
  audio ───►┌────────────────────┐         │
            │ HuBERT / wav2vec2  │──► content features [B, T, 768]      ┐
            └────────────────────┘                                      ├─►┌───────────┐
            ┌────────────────────┐                                      │  │ Generator │──► mel [B, 80, T]
  sample ──►│ speaker encoder    │──► speaker embedding [B, 256] ───────┘  └───────────┘         │
            └────────────────────┘                                        AdaIN          ┌───────┴────────┐
                                                                                         ▼                ▼
                                                                       Discriminator/PatchGAN     Vocoder ──► waveform
```

In this repository the HuBERT and speaker-encoder boxes are **mocks** (marked `Mock*`), the
Generator/Discriminator are real (if minimal) architectures, and the **vocoder is the real
pretrained HiFi-GAN**.

## Where do I see the sentence a sample is saying?

There is **no TTS in this pipeline** — the audio you hear is converted *human speech* from LibriTTS
(public-domain audiobook recordings). Coqui XTTS exists in `testing.py` but is not used by training or
evaluation. So the "input text" is the corpus transcript, and it is recorded in four places:

| Where | What you get |
|---|---|
| console, right under each `-> rendered ...` line | `source: <utt_id> "sentence"` + `converted: 4.27s-8.27s of a 11.68s recording` |
| `samples/index.csv` | one row per rendered wav: epoch, utt_id, wav path, mel frames, seconds, crop window, full sentence |
| `logs/train-*.log` | the same `source:` lines, kept after the console closes (`Select-String "source:" logs\train-short.log`) |
| `eval_out/report.csv` / `report.json` | per-utterance `text`, `mcd`, `mel_l1`, `secs_target`, `secs_source`, `wav`, `ref_wav` |
| authoritative source | `datasets/libritts/<split>/<speaker>/<chapter>/<utt_id>.normalized.txt` |

Two details that explain what you actually hear:

* **Training samples are a random crop**, so the printed sentence is the *whole utterance* while only
  the reported window was converted — that is why `converted: 4.27s-8.27s of a 11.68s recording` is
  printed next to the transcript.
* **Short clips are no longer silence-padded.** 46% of LibriTTS clips are shorter than a 4 s segment,
  so `--min-seconds` now defaults to `--segment-seconds` and such clips are skipped (a one-off scan
  cost, cached afterwards). Pass `--min-seconds 0` to restore padding.

## Files

| File | Contents |
|---|---|
| `libritts.py` | `LibriTTSDataset`, `scan_split`, `build_libritts_loader` — real LibriTTS/LibriTTS-R audio → `(content, speaker, mel)` |
| `features.py` | content/speaker extractors: `mel-projection`/`mel-stats` (no weights needed) and `hubert`/`resemblyzer` (real, load from local paths) |
| `dataset.py` | `MockVoiceConversionDataset`, `build_dataloader` — random tensors shaped like extracted speech features (`--data mock`) |
| `encoders.py` | `MockContentEncoder`, `MockSpeakerEncoder` — placeholder front-end models used by the shape check |
| `models.py` | `AdaIN1d`, `AdaINResBlock`, `Generator` (AdaIN timbre injection), `PatchDiscriminator` (PatchGAN) |
| `vocoder.py` | `HiFiGANVocoder` — real pretrained Coqui HiFi-GAN (mel → waveform), mel extraction helper, CLI self-check |
| `train.py` | Training loop (mock or LibriTTS), Hinge/LSGAN adversarial losses, mel-L1 + feature-matching losses, logging, checkpoints, wav rendering, `--val-every` validation |
| `metrics.py` | Objective metrics: `mel_cepstral_distortion` (MCD, no weights needed) and `speaker_cosine_similarity` (SECS, needs resemblyzer) |
| `evaluate.py` | Convert a held-out split with a trained checkpoint and report MCD/SECS; writes wavs + JSON/CSV report |
| `plan.md` | The approved implementation plan (design rationale, contracts, migration path) |
| `todo.md` | Status, the remaining downloads, open decisions |
| `datasets/libritts/` | Extracted corpus: `train-clean-100` (33,232 utts / 247 spk), `dev-clean` (5,736 / 40), `test-clean` (4,837 / 39) |

Unrelated to this phase but kept in the repo: `testing.py` (Coqui XTTS voice-cloning smoke test),
`test.py` (scratch), `coqui-ai-TTS/` (the TTS library checkout), sample audio files.

## Real data (LibriTTS)

```powershell
# 500 utterances, 4 epochs, ~4.5 min on CPU — real audio, real mel target, real HiFi-GAN rendering
python train.py --data libritts --max-utterances 500 --epochs 4 --batch-size 8 --render-every 1
```

**First full real run (verified, 8.9 min on CPU):** 4 s segments, real HuBERT + resemblyzer features,
400 utterances/epoch, batch 4, `--max-minutes 9`:

```powershell
python train.py --data libritts --content-source hubert --speaker-source resemblyzer `
    --segment-seconds 4.0 --batch-size 4 --max-utterances 400 --epochs 2 `
    --val-split dev-clean --val-every 1 --val-utterances 16 --render-every 1 --max-minutes 9
# -> 200 steps in 531.8 s | mel L1 2.0475 -> 1.0202 | val MCD 47.411 -> 46.238 dB
# -> samples/epoch_00{1,2}.wav (4.01 s each) + checkpoints/best.pt
```

`--max-minutes N` stops training gracefully at the budget (saves checkpoints, renders, prints the
summary) instead of being killed — useful for CPU time-boxed experiments.

What that path does differently from `--data mock`:

* the mel target comes from the **installed vocoder's own `AudioProcessor`** (24 kHz LibriTTS → 22.05 kHz,
  80-bin raw log-mel), so the target and the vocoder can never disagree;
* utterances are randomly cropped to `--segment-seconds` (2.0 s → 173 mel frames → exactly
  `[B, 80, 173]`), short clips are zero-padded;
* `content`/`speaker` come from `features.py`. Defaults (`mel-projection`, `mel-stats`) are deterministic
  functions of the real waveform with **no downloads needed**; the real HuBERT / d-vector extractors
  switch on with `--content-source hubert --speaker-source resemblyzer` once their weights are placed
  (URLs and target folders in `todo.md` §1).

Corpus facts: LibriTTS speaker ids are **disjoint** across splits (0 overlap train/test), so
`test-clean` is a genuine unseen-speaker set and `dev-clean` is a clean validation set. Verified run
(500 utts, batch 8, 4 epochs, 248 steps): `mel L1 1.871 → 0.764`, 4 wavs written at 22.05 kHz.

## Evaluation (MCD + SECS)

During training, validate on held-out speakers:

```powershell
python train.py --data libritts --max-utterances 2000 --epochs 20 `
    --val-split dev-clean --val-every 2 --val-utterances 40 --render-every 5
# -> val dev-clean @epoch 2: MCD 46.244 dB | mel L1 1.3719  (40 utterances)
```

After training, run a full conversion pass on the unseen-speaker split:

```powershell
python evaluate.py --checkpoint checkpoints/best.pt --split test-clean `
    --max-utterances 50 --require-enrollment --segment-seconds 4.0 `
    --out-dir eval_out --content-source hubert --speaker-source resemblyzer
```

`--require-enrollment` keeps only speakers with ≥2 clips, so the SECS enrolment clip is always a
*different* utterance from the content (true zero-shot). Without it, the report counts how many items
fell back to their own audio (`enrollment_fallbacks`), which would otherwise flatter SECS.

Result of that command on the 200-step checkpoint above (8 unseen test-clean speakers):

| Metric | Value |
|---|---|
| MCD | 36.884 +/- 7.440 dB |
| mel L1 | 0.9490 |
| SECS vs target | 0.519 +/- 0.063 |
| SECS vs content source | 0.529 |

| Metric | Meaning | Needs |
|---|---|---|
| **MCD** (dB, lower better) | Mel-cepstral distortion over `c1..c13` of the log-mel; classic spectral-distance metric. `c0` (loudness) is excluded by design, so pair it with mel-L1 to catch level errors | nothing (numpy/scipy) |
| **SECS** (higher better) | Cosine similarity between the speaker embedding of the converted audio and the **enrolment** clip. Also reported against the content source, so you can see source-timbre leakage | XTTS v2 speaker encoder (already installed) |
| mel L1 | Per-frame reconstruction error, kept because MCD is level-blind | nothing |

`evaluate.py` defaults to `--enrollment-mode different-utterance`, i.e. the speaker embedding is taken
from a **different clip of the target speaker** than the content — true zero-shot setup, not the
identity shortcut the model sees in training. Wavs, `report.json` and `report.csv` land in `--out-dir`.

Calibration reference: a barely-trained generator (100 steps) scores **MCD ≈ 40 dB** — that is the
"untrained" end of the scale, not a target. Converged VC systems sit in the **4–8 dB** range; SECS
around 0.7–0.9 for a convincing match.

## Quick start

```powershell
# interpreter used by this project (see pyrightconfig.json)
C:/Users/NP25-1/AppData/Local/miniconda3/envs/speech_recog_aol/python.exe train.py
```

Runs 20 epochs of mock training on CPU in ~3 minutes, prints a shape sanity check, a loss line
every 10 steps, an epoch summary, and a final summary table. It also works **before** the vocoder
checkpoint is downloaded: the missing file is reported with its URL and training continues
(rendering disabled).

Useful flags:

```powershell
# later-phase loss (smoothing) instead of the default hinge loss
python train.py --loss lsgan

# longer utterances / bigger batches
python train.py --time-steps 256 --batch-size 16

# no checkpoints, quieter logging
python train.py --save-dir none --log-every 50

# odd frame counts are safe too (both networks are fully convolutional)
python train.py --epochs 2 --time-steps 37

# vocode a sample every 5 epochs with the real HiFi-GAN (needs the checkpoint, see below)
python train.py --render-every 5 --wav-dir samples --vocoder-dir vocoder_models/en/ljspeech/hifigan_v2

# full-speed training with the vocoder switched off entirely
python train.py --no-vocoder
```

`--help` lists every option. Defaults: `epochs=20`, `data=mock`, `batch_size=8`, `time_steps=128`,
`dataset_size=128`, `hidden_dim=256`, `num_blocks=4`, `lr=2e-4` (AdamW, betas 0.5/0.999),
`lambda_mel=45`, `lambda_fm=2`, `loss=hinge`, `save_dir=checkpoints`,
`vocoder_dir=vocoder_models/en/ljspeech/hifigan_v2`, `render_every=0`; for `--data libritts`:
`split=train-clean-100`, `segment_seconds=2.0`, `content_source=mel-projection`,
`speaker_source=mel-stats`, `max_utterances=None`.

## Vocoder (installed)

The vocoder is a **pretrained Coqui HiFi-GAN**, not a mock. It is already installed in this
workspace at `vocoder_models/en/ljspeech/hifigan_v2/` (`config.json` + `model_file.pth`); nothing is
ever downloaded by the code. To reinstall it elsewhere:

| Item | Value |
|---|---|
| Model | `vocoder_models/en/ljspeech/hifigan_v2` |
| Download URL | `https://github.com/coqui-ai/TTS/releases/download/v0.6.1_models/vocoder_models--en--ljspeech--hifigan_v2.zip` |
| Size | **3.6 MB** (the zip holds `config.json` ≈ 7 KB + `model_file.pth` = 3.79 MB) |
| Audio config | 22.05 kHz, 80 mel bins, `hop_length=256`, `n_fft=win=1024`, `mel_fmax=8000`, `log_func=np.log` |
| Mel scaling | **`signal_norm=false`** ⇒ raw natural-log mel, *not* range-normalized (measured on real speech: ≈ −18 … +1, mean ≈ −4.5) |
| Network | small HiFi-GAN: `upsample_initial_channel=128` ⇒ 928,514 generator params (matches the 3.79 MB file), described in its own config as a "universal hifigan trained on LibriTTS, no spectrogram normalization" — i.e. multi-speaker, which is what zero-shot VC wants |
| Where to unzip | `<repo>/vocoder_models/en/ljspeech/hifigan_v2/` |

The folder must end up like this:

```
vocoder_models/en/ljspeech/hifigan_v2/
├── config.json        <- audio config + generator architecture
└── model_file.pth     <- HiFi-GAN weights
```

Then confirm the install (this also round-trips a real audio file so you can *listen* to whether
the checkpoint is correct):

```powershell
python vocoder.py --check jokowi.mp3 --out probe.wav
```

Alternatives, if you prefer a different rate or family (same interface, pass `--vocoder-dir`):

| Model | URL |
|---|---|
| `vocoder_models/en/vctk/hifigan_v2` | `https://github.com/coqui-ai/TTS/releases/download/v0.6.1_models/vocoder_models--en--vctk--hifigan_v2.zip` |
| `vocoder_models/en/ljspeech/multiband-melgan` | `https://github.com/coqui-ai/TTS/releases/download/v0.6.1_models/vocoder_models--en--ljspeech--multiband-melgan.zip` |
| `vocoder_models/universal/libri-tts/fullband-melgan` | `https://github.com/coqui-ai/TTS/releases/download/v0.6.1_models/vocoder_models--universal--libri-tts--fullband-melgan.zip` |

`python vocoder.py --model-dir <dir>` prints the loaded model's `sample_rate`, `hop_length` and
`mel_bins`, so the dataset's mel extraction can be matched to whatever you install. If the
checkpoint declares a different mel-bin count than `--mel-bins`, training stops with an explicit
error instead of producing mismatched audio.

## Tensor contracts

| Name | Shape | Where |
|---|---|---|
| content features | `[B, T, 768]` | `dataset.py`, `MockContentEncoder` |
| speaker embedding | `[B, 256]`, L2-normalized | `dataset.py`, `MockSpeakerEncoder` |
| target mel (ground truth) | `[B, 80, T]` | `dataset.py` |
| predicted mel | `[B, 80, T]` | `Generator` output |
| discriminator logits | `[B, 1, ceil(80/2^n_layers), ceil(T/2^n_layers)]` → `[B, 1, 5, T/8]` by default | `PatchDiscriminator` |
| waveform | `[B, 1, T * hop_length]`, `hop_length=256` for the default checkpoint | `vocoder.py` (`HiFiGANVocoder`) |

**Why there are no shape errors.** The Generator only uses stride-1, same-padded convolutions, so
the frame count `T` is preserved exactly from content features to the predicted mel — the L1 loss
compares `[B, 80, T]` with `[B, 80, T]` by construction, not by luck. The Discriminator
downsamples with `kernel=4, stride=2, padding=1`, whose output length is `floor(L/2)`, so even odd
`T` (e.g. `--time-steps 37`) is handled. `HiFiGANVocoder` trims/pads its output to exactly
`T * hop_length` (Coqui's implementation pads the mel before upsampling, so the raw length can be
off by a few hundred samples). `train.py` runs a startup sanity check that asserts each shape above
before training begins (`--no-sanity-check` to skip).

## Architecture

**Generator** — `content [B,T,768]` → `conv_in` → 4 × `AdaINResBlock` → `conv_out` →
`mel [B,80,T]` (~4M parameters).

Speaker identity is injected with **AdaIN (Adaptive Instance Normalization)**: each block
normalizes its activations over time and then re-scales/re-shifts them per channel with
`(scale, bias) = Linear(speaker_embedding)`. Content features determine the temporal structure
(*what* is said); the speaker embedding determines the timbre (*who* says it). This is the
standard zero-shot VC formulation: one model, any unseen speaker, no per-speaker fine-tuning.

**Discriminator** — a PatchGAN. The mel-spectrogram is treated as a 1-channel image
`[B,1,80,T]`, downsampled 4× by stride-2 convolutions (80 → 40 → 20 → 10 → 5 bins) and collapsed
to one real/fake score per patch. Patch-level supervision pushes the Generator toward locally
coherent spectral detail instead of a single global judgement. `forward_features()` also returns
the intermediate activations, which the feature-matching loss consumes.

## Losses

| Term | Formula | Default weight |
|---|---|---|
| Discriminator adversarial (`hinge`) | `E relu(1 − D(real)) + E relu(1 + D(fake))` | 1.0 (D step) |
| Discriminator adversarial (`lsgan`) | `0.5·E (D(real) − 1)² + 0.5·E D(fake)²` | 1.0 (D step) |
| Generator adversarial | `−E D(fake)` (hinge) / `0.5·E (D(fake) − 1)²` (lsgan) | `--lambda-adv 1.0` |
| Mel reconstruction | `L1(mel_pred, mel_target)` | `--lambda-mel 45.0` |
| Feature matching | mean over D layers of `L1(D_features(fake), D_features(real))` | `--lambda-fm 2.0` |

**Which adversarial loss to use** (`--loss`):

* `hinge` *(default)* — early phase. Chosen while the model is still collapse-prone, which is
  exactly the situation during mock-data/architecture development: it bounds the loss, does not
  saturate as aggressively, and tolerates an over-confident Discriminator.
* `lsgan` — later phase, once the Generator actually produces voice-like audio and the remaining
  problem is roughness/artefacts. Least-squares penalties give a gentler, non-vanishing gradient
  for small spectral discrepancies, which smooths output rather than fighting collapse.

Separate AdamW optimizers are used for G and D. Per batch the Discriminator takes one step on
`(real, G(x).detach())`, then the Generator takes one step through
`λ_adv·adv + λ_mel·L1 + λ_fm·feature_matching`.

## Checkpoints

`--save-dir` (default `checkpoints/`) receives two overwritten files — no unbounded growth:

* `latest.pt` — written every epoch
* `best.pt` — written when the epoch-mean Generator loss improves

Each holds `generator` / `discriminator` state dicts, both optimizer states, the epoch and the
full argument namespace. Pass `--save-dir none` to disable checkpointing (an empty string is
swallowed by PowerShell before it reaches Python).

```python
ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
```

## Mock → real: what has been replaced

| Piece | State | Notes |
|---|---|---|
| mel targets | **real** (LibriTTS) | produced by the installed vocoder's `AudioProcessor` via `HiFiGANVocoder.mel_from_waveform()` — 24 kHz source → 22.05 kHz, 80-bin raw log-mel (no normalization, ≈ −18 … +1), so `--lambda-mel 45` (a TTS norm for normalized mels) may need retuning on real data |
| dataset | **real** (`libritts.py`) | `--data libritts`; fixed-length crops, zero-padded short clips; JSON manifest cache |
| vocoder | **real** (`vocoder.py`) | pretrained HiFi-GAN, see [Vocoder](#vocoder-installed) |
| content features | placeholder `mel-projection` → real `hubert` | `features.py`; drop the three HF files into `pretrained/hubert-base-ls960/` and pass `--content-source hubert`. HuBERT emits a frame every ~20 ms vs the mel's ~11.6 ms, so the extractor interpolates onto mel frames |
| speaker embedding | placeholder `mel-stats` → real `resemblyzer` | `pip install resemblyzer` (the 15 MB wheel bundles `pretrained.pt`) and pass `--speaker-source resemblyzer`; 256-d, L2-normalized, same contract |
| `MockContentEncoder` / `MockSpeakerEncoder` | kept | only used by the startup shape check and the `--data mock` path |
| `hinge` loss | switch to `--loss lsgan` | once output is voice-like but rough |

## Non-goals in this phase

F0/prosody modelling, multi-scale or multi-period discriminators, AMP, distributed training, and
hyper-parameter tuning are deliberately out of scope. `train.py` imports `torch` only; the vocoder's
(`TTS`, `librosa`, `soundfile`) and the extractors' (`transformers`, `resemblyzer`) imports happen
lazily inside `vocoder.py` / `features.py`, so training never depends on them unless you enable those
paths.

## Troubleshooting

* `RuntimeError: shape sanity check failed` — the printed table shows every produced vs expected
  shape; the contract was broken by an edit.
* `SystemExit: Real LibriTTS training needs the vocoder` — `--data libritts` takes its mel target from
  the vocoder's `AudioProcessor`, so the checkpoint must be installed (it is, in this workspace).
* `MissingWeightsError` — you asked for `--content-source hubert` or `--speaker-source resemblyzer`
  without the weights; the message contains the URL and target path, or fall back to the placeholders.
* Losses stay flat on mock data — expected: the "target" is random noise, so there is nothing
  real to learn. The acceptance criterion for this phase is that losses are finite and keep
  *changing* for 10+ epochs without dimension errors, not that they converge.
* After a few epochs the default `hinge` run drives `D` loss toward 0 (`real` logits positive,
  `fake` negative) and `G`'s `adv` term up while `mel` L1 keeps slowly improving. That is the
  Discriminator learning to tell generated spectra from *pure noise*, which is trivial and says
  nothing about real audio. Expect it, don't tune against it; check the `mel` column for actual
  progress, and note LSGAN keeps the two losses balanced for longer.
* `--dataset-size` smaller than `--batch-size` is rejected: incomplete batches are dropped
  (`drop_last=True`), which would otherwise yield zero steps per epoch.
