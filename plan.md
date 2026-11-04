# Zero-Shot Voice Conversion (GAN) — Mock-Data Base Architecture

Refined implementation plan derived from `requirements.md`.

**Status:** approved and implemented. See `README.md` for the user-facing guide.

---

## 1. Goal

Build the *base architecture and training pipeline* for a zero-shot voice conversion
(timbre transfer) GAN, runnable end-to-end on **mock data**, with no tensor-dimension
errors. The real dataset (VCTK) and the real front-end models (HuBERT, speaker encoder) are
replaced by shape-faithful mocks in this phase; the **vocoder is the real pretrained HiFi-GAN**
(`vocoder.py`), as requested after the first review.

Acceptance (from `requirements.md`): `python train.py` prints training logs with losses
changing for at least 10 consecutive epochs on random data, with no
`RuntimeError: size mismatch`.

## 2. Environment

| Item | Value |
|---|---|
| Interpreter | `C:/Users/NP25-1/AppData/Local/miniconda3/envs/speech_recog_aol/python.exe` (repo `pyrightconfig.json` targets this env) |
| Libraries used in this phase | `torch` 2.11.0 + stdlib **only** |
| Libraries present but *not* imported here | `torchaudio`, `transformers` 5.0.0, `librosa`, `soundfile`, `coqui-ai-TTS` (used by `testing.py` for the TTS side) |
| Device | auto: CUDA if available, else CPU |
| Run | `python train.py` (defaults), `python train.py --loss lsgan --save-dir checkpoints` |

## 3. Tensor contracts

Symbols: `B` batch size, `T` time steps (frames), fixed per batch.

| Name | Shape | Produced by | Simulates |
|---|---|---|---|
| `content` | `[B, T, 768]` | `dataset.py` / `MockContentEncoder` | HuBERT / wav2vec 2.0 linguistic features |
| `speaker` | `[B, 256]` | `dataset.py` / `MockSpeakerEncoder` | d-vector of the target user |
| `mel` (target, ground truth) | `[B, 80, T]` | `dataset.py` | 80-bin target Mel-spectrogram |
| `mel` (predicted) | `[B, 80, T]` | `Generator` | output of G |
| D logits | `[B, 1, 10, T/8]` | `PatchDiscriminator` | patch real/fake validity |
| waveform | `[B, 1, T*hop_length]` | `vocoder.py` (`HiFiGANVocoder`) | real HiFi-GAN output (hop 256 default) |

Design rule that makes the acceptance criterion structural rather than lucky:
the Generator uses **stride-1, same-padded** convolutions only, so `T` is preserved
bit-for-bit from content features to mel. The Discriminator's downsampling uses
PyTorch's floor formula, so odd/non-multiple-of-8 `T` remains valid too.

## 4. Files

### 4.1 `dataset.py`

`MockVoiceConversionDataset(size=256, time_steps=128, content_dim=768, speaker_dim=256, mel_bins=80, seed=0)`

- `__getitem__` returns `{"content": [T,768], "speaker": [256], "mel": [80,T]}` from `torch.randn`.
- `speaker` is L2-normalized (d-vector convention).
- Deterministic but epoch-varying sampling: per-item `torch.Generator` seeded with
  `seed + epoch * size + index`; `set_epoch(epoch)` is called by the trainer. Reproducible
  across runs, different data per epoch, `num_workers>0` safe (each item is self-contained).
- `build_dataloader(...)` → `DataLoader(drop_last=True, shuffle=True, generator=<seeded>)`.
- Default `collate_fn` batches the dict, giving the `[B, ...]` shapes above.

### 4.2 `encoders.py`

Mock front-end modules, documented as swap points (the vocoder moved to `vocoder.py` — it is real).

- `MockContentEncoder(in_dim=768, out_dim=768)`: `[B,T,768] → [B,T,768]`, same-padded conv
  so `T` is preserved.
- `MockSpeakerEncoder(in_dim=256, out_dim=256)`: accepts `[B,T,D]` (mean-pooled) or `[B,D]`,
  → `[B,256]`, L2-normalized.

### 4.2b `vocoder.py` (real, not a mock — added after review)

`HiFiGANVocoder` wraps a **pretrained Coqui HiFi-GAN** checkpoint loaded from a local directory
(`--vocoder-dir`, default `vocoder_models/en/ljspeech/hifigan_v2`): 22.05 kHz, 80 mel bins,
`hop_length=256`, `mel_fmax=8000`, `signal_norm=false` (raw natural-log mel), 928,514 generator
params (3.79 MB checkpoint). Loading goes through `TTS.config.load_config` →
`TTS.vocoder.models.setup_model` → `load_checkpoint`, all local-file based, so **nothing is
downloaded by the code**.

- `forward(mel [B,80,T]) → [B,1,T*hop]`, trimmed/padded to exactly `T*hop` (Coqui pads the mel
  before upsampling, so the raw length is off by a few hundred samples).
- `mel_from_waveform(wav, sample_rate)` → `[B,80,T]` using the checkpoint's own `AudioProcessor`,
  so future training targets cannot drift from the vocoder's mel config.
- `render(mel, path)` writes 22.05 kHz mono PCM_16.
- CLI self-check: `python vocoder.py --check <audio> --out <wav>`, plus `--model-dir` to print the
  loaded `sample_rate`/`hop_length`/`mel_bins`.
- Missing checkpoint ⇒ `train.py` prints the download URL + expected layout and continues with
  rendering disabled. No mock fallback exists.

### 4.3 `models.py`

- `AdaIN1d(speaker_dim, channels)`: `Linear(256 → 2C)` → `gamma=1+scale`, `beta`;
  instance normalization computed manually over time (valid at `T=1`, `B=1`);
  `y = gamma * x_norm + beta`.
- `AdaINResBlock(channels, speaker_dim)`: two `Conv1d(k=3, same)` each followed by
  AdaIN + LeakyReLU(0.2); residual connection.
- `Generator(in_dim=768, hidden=256, mel_bins=80, num_blocks=4, speaker_dim=256)`:
  `conv_in → 4 × AdaINResBlock → conv_out`, **no striding**, output `[B,80,T]`. ~4M params
  (keeps a checkpoint at ~81 MB measured, instead of ~200 MB at 512 channels).
- `PatchDiscriminator(mel_bins=80, channels=64, n_layers=4)`:
  `[B,80,T] → [B,1,80,T]`, 4 × `Conv2d(k=4, s=2, p=1)` (80→40→20→10→5 bins) + final
  `Conv2d(512→1, k=3, p=1)` → patch logits. `forward()` returns `(logits, features)` where
  `features` are the intermediate activations used by the feature-matching loss.
- `init_weights` (N(0, 0.02)) and `count_parameters()`.

### 4.4 `train.py`

Defaults: `epochs=20`, `batch_size=8`, `time_steps=128`, `dataset_size=128`, `hidden=256`,
`num_blocks=4`, `loss=hinge`, `lambda_adv=1.0`, `lambda_mel=45.0`, `lambda_fm=2.0`,
`AdamW lr=2e-4 betas=(0.5, 0.999)`, `log_every=10`, `save_dir=checkpoints`, `device=auto`,
`seed=1234`, `num_workers=0`. (`dataset_size=128` keeps a default CPU run at ~3 minutes:
~1.6 steps/s at `batch_size=8`, 16 steps/epoch.)

**Loss choice (per user policy):** both losses implemented and switchable.

| `--loss` | D loss | G adversarial term | When to use |
|---|---|---|---|
| `hinge` (default) | `E relu(1 - D(real)) + E relu(1 + D(fake))` | `-E D(fake)` | early phase / collapse-prone training |
| `lsgan` | `0.5·E (D(real)-1)² + 0.5·E D(fake)²` | `0.5·E (D(fake)-1)²` | later phase, output already voice-like but rough; smoother gradients |

**Startup sanity check** (on by default, `--no-sanity-check` to skip): builds all three mocks
+ G + D, feeds spec-shaped random tensors, asserts each shape against §3, prints the table.

**Step order per batch**

1. `D` step: real logits on `mel_target`; fake `mel = G(content, speaker).detach()`;
   compute D loss; `zero_grad → backward → step`.
2. `G` step: recompute `D(fake)` with graph; `loss = λ_adv·adv + λ_mel·L1(mel, target)
   + λ_fm·feature_matching`; `zero_grad → backward → step`.

**Logging:** every `--log-every` micro-batches,
`epoch e/N step s/S | D <loss> (real x, fake y) | G <loss> (adv a, mel m, fm f)`;
plus an epoch summary with means, steps/s. Console only.

**Checkpoints:** `--save-dir` (default `checkpoints`, `""`/`none` disables). Each epoch writes
`latest.pt` and, when total G loss improves, `best.pt` — both overwritten, so state stays
bounded (~2 files). Contents: G/D `state_dict`, both optimizers, epoch, args.

**Completion banner:** epochs run, first-vs-last mel-L1, and an explicit
"no dimension errors" statement.

### 4.5 `README.md`

Shape table, architecture description, run commands, mock→real swap table, migration notes
(HuBERT layer-9 features at 20 ms hop vs mel frames, real speaker encoder, VCTK variable-length
bucketing, HiFi-GAN vocoder), and the hinge→LSGAN staging advice.

## 5. Verification performed

1. `python -m py_compile dataset.py encoders.py models.py vocoder.py train.py` with the conda interpreter.
2. `python train.py` (default 20 epochs) → ≥10 consecutive epoch logs, finite losses, no
   `RuntimeError`; report the actual mel-L1 trend.
3. Robustness sweep: `--epochs 2 --loss lsgan --batch-size 4 --time-steps 100` and
   `--time-steps 37` (odd) to prove the shape guarantee is not accidental.
4. Vocoder path: a random-weight HiFi-GAN checkpoint was built in a temp dir to exercise
   `vocoder.py` without any download — `python vocoder.py --check <audio>` round-trip plus
   `train.py --render-every 1` wav dumps; and the missing-checkpoint path verified to exit 0 with
   the download URL printed.
5. Clean up `checkpoints/` and `__pycache__` artifacts created by verification runs.

## 5b. Real-data phase (implemented after the plan was approved)

Corpus received and extracted to `datasets/libritts/` (33,232 / 5,736 / 4,837 utterances; 247 / 40 /
39 speakers; 24 kHz mono; **0 speaker overlap** across splits, so `test-clean` is a genuine
unseen-speaker set).

New modules:

* `libritts.py` — `scan_split` (JSON manifest cache, duration probing optional, orphan skipping),
  `LibriTTSDataset` (random/centre crop to `--segment-seconds`, short clips zero-padded, mel via
  `HiFiGANVocoder.mel_from_waveform`), `build_libritts_loader`, `split_summary`.
* `features.py` — content/speaker extractors. `mel-projection` / `mel-stats` are deterministic
  functions of the real waveform (no weights needed); `hubert` / `resemblyzer` load real weights from
  **local paths only** and raise `MissingWeightsError` with the URL + target path otherwise.

`train.py` gained `--data {mock,libritts}` plus `--split`, `--max-utterances`, `--segment-seconds`,
`--content-source`, `--speaker-source`, `--hubert-dir`, `--hubert-layer`, `--resemblyzer-path`. The
mock path stays the default so the original acceptance command is untouched.

Verified: `--data libritts --max-utterances 500 --epochs 4 --batch-size 8` → 248 steps in 274 s,
`mel L1 1.871 → 0.764`, 4 wavs rendered by the real HiFi-GAN at 22.05 kHz (173 frames = 2.01 s each).
Bug caught during verification: the crop length was computed with the vocoder's 22.05 kHz against
24 kHz audio, shortening segments to 1.84 s — now derived from the file's own sample rate.

## 6. Non-goals (this phase)

F0/prosody modelling, multi-scale / multi-period discriminators, AMP, distributed training,
hyper-parameter tuning, and any eager TTS/transformers import inside the training script (the
vocoder's TTS/librosa/soundfile imports live inside `vocoder.py`, the extractors' imports inside
`features.py`, and both are lazy).

## 7. Migration path to real data

| Mock | Real replacement | Watch out for |
|---|---|---|
| `MockVoiceConversionDataset` | VCTK loader + HuBERT feature extraction | variable utterance length ⇒ pad-and-mask or bucketing; content frames (20 ms) ≠ mel frames (hop 256 ≈ 11.6 ms @22.05 kHz) ⇒ resample content or set matching strides |
| `MockContentEncoder` | HuBERT / wav2vec 2.0 (layer 9, 768-d) | keep the 768-d contract, or make the dim configurable |
| `MockSpeakerEncoder` | d-vector / ECAPA-TDNN from one reference utterance | L2 normalization must stay consistent with training |
| `MockVocoder` | HiFi-GAN (hop 256) | **replaced:** `vocoder.py` loads a pretrained Coqui HiFi-GAN; only the checkpoint download is outstanding (`https://github.com/coqui-ai/TTS/releases/download/v0.6.1_models/vocoder_models--en--ljspeech--hifigan_v2.zip`) |
| hinge loss | switch to LSGAN once output is voice-like but rough | see §4.4 table |
