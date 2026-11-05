# TODO — status, data, and the two remaining downloads

Legend: `[x]` done · `[ ]` needed from you · `[~]` decision needed from you

---

## 0. Status at a glance

| Component | State | Detail |
|---|---|---|
| Vocoder | **real, installed** | Coqui HiFi-GAN `vocoder_models/en/ljspeech/hifigan_v2` (3.79 MB), 22.05 kHz / 80 bins / hop 256, raw natural-log mel. Verified by round-tripping `jokowi.mp3` → `jokowi_vocoded.wav`. |
| Corpus | **real, extracted** | LibriTTS + LibriTTS-R in `datasets/libritts/{train-clean-100,dev-clean,test-clean}` — 43,805 utterances, 326 speakers, 24 kHz mono. Speaker splits are disjoint. |
| Mel targets | **real** | Produced by the installed vocoder's own `AudioProcessor` (`vocoder.mel_from_waveform`), so target and vocoder can never disagree. |
| Content features | **real, installed & verified** | `--content-source hubert`: `pretrained/hubert-base-ls960/pytorch_model.bin` (377.6 MB) + configs are in place; loads in ~8 s, returns `[173, 768]` for a 2 s clip, content cosine between different utterances ≈ 0.17. Layer 9 by default. |
| Speaker embedding | **XTTS v2's own encoder (512-d), verified** | `--speaker-source xtts` (now the default): XTTS v2's ResNet/ASP speaker encoder, loaded with `mmap` from the local 1.78 GB checkpoint (only ~32 MB read), 512-d. Measured over 8 speakers: same-speaker cos **+0.536** vs different **+0.241** → margin **+0.295** (resemblyzer managed ~+0.15, the placeholder ~0.04). |
| Trainer | **works on real data, end-to-end verified** | First real run: 4 s segments, HuBERT + resemblyzer, 400 utts/epoch, batch 4 → 200 steps in **531.8 s**, `mel L1 2.0475 → 1.0202`, `val MCD 47.411 → 46.238 dB`, 4.01 s wavs rendered, checkpoints saved. Log: `first_run_10min.log`. |
| Evaluation | **working** | `evaluate.py --require-enrollment` on 8 unseen `test-clean` speakers from that checkpoint: MCD **36.884 dB**, mel L1 0.949, SECS **0.519** vs target / 0.529 vs source. |
| Validation / metrics | **implemented** | `--val-split dev-clean --val-every N` prints MCD + mel-L1 on held-out speakers during training; `evaluate.py` reports MCD/SECS on a held-out split with wavs + JSON/CSV. MCD needs no weights; **SECS activates once `resemblyzer` is installed** (today it prints "unavailable"). |
| Checkpoint resume | **not implemented** | `latest.pt` / `best.pt` now also store the `SpeakerProbe` and its optimizer (so a checkpoint is a complete training state), but nothing loads them yet. |

---

## 1. Downloads — both done (verified working)

### 1.1 HuBERT — real content features (`--content-source hubert`) ✅ installed

```
Target folder : D:\NP\Kuliah\speech-recog\pretrained\hubert-base-ls960\
Files         : config.json               https://huggingface.co/facebook/hubert-base-ls960/resolve/main/config.json
                preprocessor_config.json  https://huggingface.co/facebook/hubert-base-ls960/resolve/main/preprocessor_config.json
                pytorch_model.bin         https://huggingface.co/facebook/hubert-base-ls960/resolve/main/pytorch_model.bin   (~360 MB)
```

- [x] Download those three files into that folder, keeping the filenames. **Done — 377.6 MB
      `pytorch_model.bin` + both configs present, loads in ~8 s.**
- [ ] Verify: `python -c "from features import HubertContentExtractor as H; H()._check_weights(); print('hubert weights OK')"`
- Then: `python train.py --data libritts --content-source hubert`
- Notes: HuBERT expects **16 kHz** audio (LibriTTS is 24 kHz — resampled inside the extractor) and emits
  ~50 frames/s while the mel runs at ~86 frames/s; the extractor interpolates content onto mel frames,
  so the `[T, 768]` contract still holds. `--hubert-layer 9` is the default; 6–12 is the useful range.

### 1.2 Resemblyzer — real speaker embeddings (`--speaker-source resemblyzer`) ✅ installed

```
Install       : pip install resemblyzer       (15 MB wheel — it *bundles* pretrained.pt; there is no separate checkpoint URL)
Lands at      : <your python env>\Lib\site-packages\resemblyzer\pretrained.pt
Override path : --resemblyzer-path <folder containing pretrained.pt>
```

- [ ] Run `pip install resemblyzer` **in the `speech_recog_aol` env** (that 15 MB wheel *is* the download).
- [ ] Verify: `python -c "import resemblyzer, pathlib; print(pathlib.Path(resemblyzer.__file__).parent / 'pretrained.pt')"` → the file must exist.
- Then: `python train.py --data libritts --speaker-source resemblyzer`
- Output is a 256-d L2-normalized d-vector — exactly the `[B, 256]` contract the Generator already has.

Alternative if you prefer ECAPA-TDNN (192-d, needs the contract change):
`https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt` plus
`hyperparams.yaml` from the same folder, and `pip install speechbrain`; then run with `--speaker-dim 192`.

---

## 2. What was done to the data

Extracted with `tar` straight into the final layout. The archives differ (`LibriTTS_R/…` for train,
`./LibriTTS/…` for dev/test) so the strip depth differs — that is why the three roots are consolidated:

```
datasets/libritts/
├── train-clean-100/   33,232 utts   247 speakers   8.65 GB   (LibriTTS-R, restored audio)
├── dev-clean/          5,736 utts    40 speakers   1.45 GB   (LibriTTS)
└── test-clean/         4,837 utts    39 speakers   1.38 GB   (LibriTTS)
datasets/manifests/    JSON scan caches (keyed by split + filters; re-runs take 0.01 s)
```

Measured facts (probed 200 random files per split):

| Property | Value |
|---|---|
| Format | 24 kHz, mono, PCM_16 (LibriTTS and LibriTTS-R are both standardised to this) |
| Duration | median 4.2–5.1 s, min 0.4 s, max 30.1 s → training crops to a fixed segment |
| Speaker overlap | **0** for train/dev, train/test and dev/test → `test-clean` is a genuine unseen-speaker set |
| Transcripts | `<utt>.normalized.txt` for every utterance; train has 1 orphan `.txt` with no `.wav`, skipped by the scanner |
| Id scheme | `<speaker>_<chapter>_<utt>_<group>`, e.g. `103_1241_000000_000001.wav` |

`[~]` **Disk:** the three tarballs (9.9 GB) are still in the repo root and are now redundant — copies
exist in `Downloads`. Say the word and I delete them; I did not remove them on my own.

---

## 3. Commands

```powershell
$py = "C:/Users/NP25-1/AppData/Local/miniconda3/envs/speech_recog_aol/python.exe"

# 1) shape-only smoke test on random data (the original acceptance criterion)
& $py train.py

# 2) real-data smoke run: 500 utterances, 4 epochs, ~4.5 min on CPU  (the run just verified)
& $py train.py --data libritts --max-utterances 500 --epochs 4 --batch-size 8 `
    --segment-seconds 2.0 --log-every 20 --render-every 1

# 3) real data, much more of it (CPU: ~0.9 steps/s at batch 8 → plan accordingly)
& $py train.py --data libritts --epochs 20 --batch-size 8 --segment-seconds 2.0 --render-every 5

# 4) after the two downloads: real content + real speaker features
& $py train.py --data libritts --content-source hubert --speaker-source resemblyzer `
    --max-utterances 2000 --epochs 10 --render-every 2
```

New flags (all in `python train.py --help`): `--data {mock,libritts}`, `--libritts-root`, `--split`,
`--max-utterances`, `--segment-seconds`, `--content-source {mel-projection,hubert}`,
`--speaker-source {mel-stats,resemblyzer}`, `--hubert-dir`, `--hubert-layer`, `--resemblyzer-path`,
plus validation: `--val-split`, `--val-every`, `--val-utterances`.

```powershell
# 5) train with held-out validation (MCD + mel-L1 on dev-clean) — do this on every real run
& $py train.py --data libritts --max-utterances 2000 --epochs 20 --val-split dev-clean `
    --val-every 2 --val-utterances 40 --render-every 5

# 6) evaluate a checkpoint on the unseen-speaker split (wavs + report.json/csv)
& $py evaluate.py --checkpoint checkpoints/best.pt --split test-clean --max-utterances 50 --out-dir eval_out
```

---

## 4. Decisions taken (this session)

1. `[x]` **Weights** — you place HuBERT and install resemblyzer yourself; paths and URLs are in §1.
   *(Update: resemblyzer has since been replaced entirely by XTTS v2's own 512-d speaker encoder.)*
2. `[x]` **Compute** — CPU only, and now **assessed properly**: the machine has an **NVIDIA Quadro
   K620** (Maxwell, sm_50, **2048 MiB VRAM with 1638 MiB already used** by the Windows desktop →
   ~410 MiB free) on driver **452.57** (CUDA 11.0). The pipeline needs ≈850–950 MiB (HuBERT alone is
   380 MB of weights), so it does not fit; the installed `torch 2.11.0+cpu` has no CUDA support anyway.
   Consequence: work in subsets (`--max-utterances`); measured ~3.65 s/step at batch 4 / 4 s segments.
3. `[x]` **Metrics** — MCD + SECS. Implemented in `metrics.py` / `evaluate.py`, with the XTTS speaker
   encoder used both for conditioning and for SECS.

Still open:

4. `[x]` **Segment length** — settled on **4.0 s** (345 mel frames); the first run above used it and
   costs ~4.4 s/step at batch 4 on CPU.
5. `[x]` **Tarball cleanup** — you are deleting the 9.9 GB of archives yourself.
6. `[~]` **Scale of the next run** — 4 s segments on CPU give ~100 steps per 7 min. Options:
   (a) same shape, more epochs (e.g. 10 epochs ≈ 70 min); (b) implement **HuBERT feature caching**
   first (extract once per utterance, then epochs become compute-bound ≈ 2–3× faster and drop the RAM
   pressure); (c) keep runs at ~10 min increments and let the checkpoint accumulate.
7. `[~]` **Repo artifacts** — `checkpoints/` (161 MB), `samples/` (2 wavs), `eval_out/` (8 wavs +
   report), `first_run_10min.log` are from the test run. Delete them, or leave them as the baseline?

---

## 5. Contract (unchanged, now fed by real audio)

```python
{
  "content": Tensor[T, 768],   # HuBERT layer 9 (interpolated to mel frames) or the mel-projection placeholder
  "speaker": Tensor[256],      # L2-normalized d-vector or the mel-stats placeholder
  "mel":     Tensor[80, T],    # vocoder AudioProcessor: 22.05 kHz, n_fft 1024, hop 256, raw log-mel
}
```

`features.py` enforces frame alignment (it raises if content frames ≠ mel frames) and `libritts.py`
raises if the mel geometry disagrees with the loaded vocoder. `T` is exact: 2.0 s → 173 frames →
44,288 samples at 22.05 kHz.

---

## 5b. Regression report + what changed (latest)

Feedback: the XTTS-encoder + `L_spk` pipeline **sounded worse** than the earlier resemblyzer pipeline,
even with more data. Findings:

1. **`L_spk` was the likely culprit.** `SpeakerProbe` was fed *raw* log-mel (mean ≈ −9, range −18…+6), so
   its gradient w.r.t. the generator's mel was large and structured — a generator can satisfy that by
   producing probe-pleasing artifacts, and the single-scale PatchGAN never punishes them. Both `spk` and
   `probe` losses fell together, which is the signature of co-adaptation rather than learned timbre.
2. **Fewer optimizer steps, not fewer data.** Per-step cost rose 2.66 s → 3.65 s, so a 10-minute box gave
   **150 steps vs 200** before. More utterances per epoch does not compensate.
3. **Bigger model, unchanged schedule** (G 4.28M → 5.46M params at the same LR/epochs).
4. **The A/B itself was confounded** — rendered samples were random crops of random utterances.

Changes made:

* `SpeakerProbe` now standardises its input **per sample** (scalar shift/scale, band structure preserved),
  which keeps the gradient sane without discarding the spectral tilt that carries speaker identity.
* `--lambda-spk` now defaults to **0** and the probe step is skipped entirely when it is off (also saves
  CPU time). Enable it later — 0.5–2.0 — once mel-L1 has plateaued.
* New `preview.py` for identical-content A/B listening across checkpoints.
* Logs print `spk off` instead of a misleading 0.000.

Available on request: re-adding `--speaker-source resemblyzer` for a three-way ablation
(old pipeline / XTTS+no-spk / XTTS+spk). It was removed on your instruction, so it is not in the code.

## 6. Still missing (code, not data)

1. Resume from checkpoint (`latest.pt` is written but never loaded).
2. Model upgrades for real data: spectral norm, multi-scale Discriminator, LR decay, AMP, grad clip.
3. Feature caching: HuBERT would run **on the fly** today; a disk cache (`[T, 768]` per utterance)
   makes full-corpus epochs several times faster — the single biggest CPU speedup available.
4. End-to-end demo: text → Coqui TTS → content features + enrollment embedding → G → mel → vocoder → wav.
   `testing.py` does TTS and `train.py` does VC; nothing joins them yet.
5. WER/intelligibility metric — would need an ASR model (~150 MB whisper-tiny); not requested so far.
6. `test.py` is still a dead scratch file; say the word and it goes.
