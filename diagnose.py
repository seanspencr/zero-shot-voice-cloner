"""Diagnose *where* the "zombie" sound comes from, with numbers instead of ears.

Phase 0 of `improvement-plan.md`. It measures the same acoustic properties on four signals so the
culprit can be identified before touching the model:

1. **real**        -- the corpus recording (centre crop)
2. **roundtrip**   -- that recording's mel pushed straight through the vocoder, i.e. the best audio this
                      mel+vocoder configuration can ever produce
3. **meanmel**     -- the average mel of N utterances pushed through the vocoder: the "average voice"
                      that a pure L1 objective converges to
4. **model**       -- the checkpoint's conversion of the same crop (skipped if the checkpoint's speaker
                      embedding size does not match the configured encoder)

Interpretation guide (printed at the end):

* roundtrip already muffled/buzzy  -> the mel+vocoer configuration is the ceiling; fix that first
* meanmel sounds like the model    -> the model has collapsed to the average voice
* model much flatter than real     -> over-smoothing from the reconstruction loss dominates
* model F0 flat while real is not  -> prosody is not being learned (no explicit F0 path)

Writes ``diagnosis.md`` next to the given ``--report`` path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from evaluate import load_generator_from_checkpoint, resolve_device
from features import CONTENT_SOURCES, DEFAULT_HUBERT_DIR, SPEAKER_SOURCES, build_content_extractor, build_speaker_extractor
from libritts import DEFAULT_ROOT, LibriTTSDataset, scan_split
from vocoder import DEFAULT_MODEL_DIR, load_vocoder, missing_checkpoint_help, vocoder_ready

__all__ = ["analyse_audio", "analyse_mel", "main"]

BANDS_HZ = ((0, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 12000))


def _stft_mag(wav: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    spec = np.abs(librosa.stft(np.asarray(wav, dtype=np.float32), n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return spec, freqs


def analyse_audio(wav: np.ndarray, sr: int) -> dict[str, float]:
    """Band energies, brightness and F0 statistics of a waveform."""
    spec, freqs = _stft_mag(wav, sr)
    power = spec**2
    total = float(power.sum()) + 1e-12
    out: dict[str, float] = {}
    for lo, hi in BANDS_HZ:
        mask = (freqs >= lo) & (freqs < hi)
        out[f"band_{lo // 1000}-{hi // 1000}k"] = float(power[mask].sum()) / total

    # spectral centroid (Hz) and high/low energy ratio -> correlates with fricative clarity
    centroid = float((freqs[:, None] * spec).sum() / (spec.sum() + 1e-12))
    high = float(power[(freqs >= 4000)].sum())
    low = float(power[(freqs < 4000)].sum())
    out["centroid_hz"] = centroid
    out["hlr_4k"] = high / (low + 1e-12)

    # F0 statistics (pYIN); slow but this runs on a handful of clips only
    try:
        import librosa

        f0, voiced, _ = librosa.pyin(
            np.asarray(wav, dtype=np.float32), fmin=60, fmax=500, sr=sr, frame_length=1024
        )
        voiced_f0 = f0[~np.isnan(f0)]
        out["f0_mean_hz"] = float(np.nanmean(f0)) if voiced_f0.size else float("nan")
        out["f0_std_semitones"] = (
            float(np.std(12 * np.log2(voiced_f0 / np.median(voiced_f0)))) if voiced_f0.size > 2 else float("nan")
        )
        out["voiced_fraction"] = float(np.mean(~np.isnan(f0))) if f0.size else float("nan")
        out["f0_coverage"] = float(np.mean(voiced)) if voiced.size else float("nan")
    except Exception as exc:  # pragma: no cover
        out["f0_error"] = float("nan")
        print(f"  (F0 analysis skipped: {exc})")
    return out


def analyse_mel(mel: np.ndarray) -> dict[str, float]:
    """Frame-to-frame variation: the cheapest numerical proxy for over-smoothing."""
    m = np.asarray(mel, dtype=np.float32)
    if m.ndim != 2 or m.shape[1] < 2:
        return {"mel_delta_abs": float("nan"), "mel_bin_std": float("nan"), "mel_range": float("nan")}
    delta = np.abs(np.diff(m, axis=1)).mean()
    return {
        "mel_delta_abs": float(delta),  # small => adjacent frames nearly identical => smoothed
        "mel_bin_std": float(m.std(axis=1).mean()),
        "mel_range": float(m.max() - m.min()),
        "mel_mean": float(m.mean()),
    }


def _fmt_table(rows: dict[str, dict[str, float]], keys: list[str]) -> str:
    names = list(rows)
    lines = ["| ukuran | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key in keys:
        cells = []
        for name in names:
            value = rows[name].get(key, float("nan"))
            cells.append("n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.4g}")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Measure spectral/prosody properties of real vs vocoder-roundtrip vs mean-mel vs model output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default="checkpoints/best.pt", help="model to analyse (optional)")
    p.add_argument("--utt-id", default=None, help="utterance to analyse (default: first of the split)")
    p.add_argument("--split", default="test-clean")
    p.add_argument("--libritts-root", default=str(DEFAULT_ROOT))
    p.add_argument("--segment-seconds", type=float, default=4.0)
    p.add_argument("--content-source", choices=CONTENT_SOURCES, default="hubert")
    p.add_argument("--speaker-source", choices=SPEAKER_SOURCES, default="xtts")
    p.add_argument("--hubert-dir", default=str(DEFAULT_HUBERT_DIR))
    p.add_argument("--hubert-layer", type=int, default=9)
    p.add_argument("--xtts-checkpoint", default=None)
    p.add_argument("--mean-mel-utterances", type=int, default=200, help="how many utterances to average for the mean-mel baseline")
    p.add_argument("--vocoder-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="diagnose_out")
    p.add_argument("--report", default="diagnosis.md")
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    args.device = str(device)

    if not vocoder_ready(args.vocoder_dir):
        print(missing_checkpoint_help(args.vocoder_dir))
        return 2
    vocoder = load_vocoder(args.vocoder_dir, device=device)
    if vocoder is None:
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- pick the utterance --------------------------------------------------------
    scanned = scan_split(
        root=args.libritts_root, split=args.split, limit=None,
        min_seconds=args.segment_seconds, shuffle_seed=None, use_cache=True,
    )
    if not scanned:
        raise SystemExit(f"no utterance in '{args.split}' is >= {args.segment_seconds:g}s")
    utt = next((u for u in scanned if u.utt_id == args.utt_id), scanned[0])
    if args.utt_id and utt.utt_id != args.utt_id:
        raise SystemExit(f"utterance '{args.utt_id}' not found in '{args.split}'")

    print("Diagnose")
    print(f"  utterance  : {utt.utt_id} (speaker {utt.speaker_id}, {args.split})")
    print(f'  transcript : "{utt.text[:80]}{"..." if len(utt.text) > 80 else ""}"')

    # ---- real audio + its mel ------------------------------------------------------
    ref_wav, ref_sr = sf.read(utt.wav_path, dtype="float32", always_2d=False)
    ref_wav = np.asarray(ref_wav, dtype=np.float32)
    n = int(round(args.segment_seconds * ref_sr))
    start = max(0, (len(ref_wav) - n) // 2)
    ref_crop = ref_wav[start:start + n]
    if len(ref_crop) < n:
        ref_crop = np.pad(ref_crop, (0, n - len(ref_crop)))

    mel_real = vocoder.mel_from_waveform(ref_crop, sample_rate=ref_sr)[0].cpu().numpy()  # [80, T]
    wav_roundtrip = vocoder.forward(torch.from_numpy(mel_real).unsqueeze(0).to(vocoder.device))[0, 0].cpu().numpy()

    results: dict[str, dict[str, float]] = {}
    results["real"] = {**analyse_audio(ref_crop, ref_sr), **analyse_mel(mel_real)}
    results["roundtrip"] = {**analyse_audio(wav_roundtrip, vocoder.sample_rate), **analyse_mel(mel_real)}
    sf.write(str(out_dir / f"{utt.utt_id}_real.wav"), np.clip(ref_crop, -1, 1), ref_sr, subtype="PCM_16")
    sf.write(str(out_dir / f"{utt.utt_id}_roundtrip.wav"), np.clip(wav_roundtrip, -1, 1),
             vocoder.sample_rate, subtype="PCM_16")

    # ---- mean-mel baseline ("average voice") ---------------------------------------
    print(f"  averaging {args.mean_mel_utterances} utterances for the mean-mel baseline ...")
    average = None
    count = 0
    for other in scanned[: args.mean_mel_utterances]:
        wav, sr = sf.read(other.wav_path, dtype="float32", always_2d=False)
        wav = np.asarray(wav, dtype=np.float32)
        m = int(round(args.segment_seconds * sr))
        s = max(0, (len(wav) - m) // 2)
        seg = wav[s:s + m]
        if len(seg) < m:
            seg = np.pad(seg, (0, m - len(seg)))
        mel = vocoder.mel_from_waveform(seg, sample_rate=sr)[0].cpu().numpy()
        if average is None:
            average = mel
        else:
            frames = min(average.shape[1], mel.shape[1])
            average = average[:, :frames] + mel[:, :frames]
        count += 1
    if average is not None and count:
        mel_mean = average / count
        wav_mean = vocoder.forward(torch.from_numpy(mel_mean).unsqueeze(0).to(vocoder.device))[0, 0].cpu().numpy()
        results["meanmel"] = {**analyse_audio(wav_mean, vocoder.sample_rate), **analyse_mel(mel_mean)}
        sf.write(str(out_dir / "meanmel.wav"), np.clip(wav_mean, -1, 1), vocoder.sample_rate, subtype="PCM_16")

    # ---- model output (optional) ---------------------------------------------------
    model_note = ""
    if Path(args.checkpoint).is_file():
        generator, saved = load_generator_from_checkpoint(args.checkpoint, device)
        saved_dim = int(saved.get("speaker_dim", 256))
        probe = build_speaker_extractor(args.speaker_source, checkpoint=args.xtts_checkpoint, device=str(device))
        if getattr(probe, "out_dim", saved_dim) != saved_dim:
            model_note = (
                f"checkpoint '{args.checkpoint}' expects {saved_dim}-d speaker embeddings, "
                f"--speaker-source {args.speaker_source} gives {getattr(probe, 'out_dim', '?')}-d -> skipped"
            )
            print(f"  model      : skipped ({model_note})")
        else:
            dataset = LibriTTSDataset(
                [utt], vocoder=vocoder, segment_seconds=args.segment_seconds, train=False,
                content_extractor=build_content_extractor(
                    args.content_source, model_dir=args.hubert_dir, layer=args.hubert_layer, device=str(device)
                ),
                speaker_extractor=probe,
            )
            sample = dataset[0]
            with torch.no_grad():
                f0 = sample["f0"].unsqueeze(0).to(device) if "f0" in sample else None
                mel_pred = generator(
                    sample["content"].unsqueeze(0).to(device), sample["speaker"].unsqueeze(0).to(device), f0
                )
            wav_pred = vocoder.forward(mel_pred)[0, 0].detach().cpu().numpy()
            results["model"] = {**analyse_audio(wav_pred, vocoder.sample_rate),
                               **analyse_mel(mel_pred[0].detach().cpu().numpy())}
            sf.write(str(out_dir / f"{utt.utt_id}_model.wav"), np.clip(wav_pred, -1, 1),
                     vocoder.sample_rate, subtype="PCM_16")
    else:
        model_note = f"no checkpoint at '{args.checkpoint}' -> model column skipped"
        print(f"  model      : skipped ({model_note})")

    # ---- report --------------------------------------------------------------------
    keys = (
        [f"band_{lo // 1000}-{hi // 1000}k" for lo, hi in BANDS_HZ]
        + ["centroid_hz", "hlr_4k", "f0_mean_hz", "f0_std_semitones", "voiced_fraction",
           "mel_delta_abs", "mel_bin_std", "mel_range", "mel_mean"]
    )
    table = _fmt_table(results, keys)

    verdict: list[str] = []
    if "roundtrip" in results and "real" in results:
        hlr_real = results["real"]["hlr_4k"]
        hlr_rt = results["roundtrip"]["hlr_4k"]
        verdict.append(
            f"- Round-trip mempertahankan HLR {hlr_rt:.3g} vs asli {hlr_real:.3g}: "
            + ("pita tinggi hilang di jalur mel+vocoder (plafon)" if hlr_rt < 0.6 * hlr_real
               else "pita tinggi relatif terjaga -> plafon bukan di vocoder")
        )
        verdict.append(
            f"- Energi pita 8-12 kHz: asli {results['real'].get('band_8-12k', float('nan')):.3g}, "
            f"round-trip {results['roundtrip'].get('band_8-12k', float('nan')):.3g} "
            "(nilai > 0 hanya mungkin bila mel_fmax > 8000)"
        )
    if "model" in results and "real" in results:
        d_model = results["model"]["mel_delta_abs"]
        d_real = results["real"]["mel_delta_abs"]
        verdict.append(
            f"- Dinamika antar-frame: asli {d_real:.3g} vs model {d_model:.3g} "
            + f"(rasio {d_model / (d_real + 1e-9):.2f}; << 1 berarti model terlalu halus / over-smoothing)"
        )
        if "meanmel" in results:
            verdict.append(
                f"- Model vs mean-mel (jarak spektral rata-rata): "
                f"{abs(results['model'].get('mel_bin_std', 0) - results['meanmel'].get('mel_bin_std', 0)):.3g} "
                "(kecil -> output mendekati 'suara rata-rata')"
            )
        if not np.isnan(results["model"].get("f0_std_semitones", float("nan"))):
            verdict.append(
                f"- Variasi F0: asli {results['real'].get('f0_std_semitones', float('nan')):.3g} semitone vs "
                f"model {results['model'].get('f0_std_semitones', float('nan')):.3g} -> "
                + ("intonasi model jauh lebih datar" if results["model"]["f0_std_semitones"] < 0.6 * results["real"]["f0_std_semitones"]
                   else ("intonasi model LEBIH goyah dari asli" if results["model"]["f0_std_semitones"] > 1.4 * results["real"]["f0_std_semitones"]
                         else "kontur intonasi sebanding dengan asli"))
            )

    body = f"""# Diagnosis (otomatis dari `diagnose.py`)

Utterance: `{utt.utt_id}` (speaker {utt.speaker_id}, {args.split}) - crop {args.segment_seconds:g}s di tengah.
Checkpoint: `{args.checkpoint}`. {model_note}

## Tabel ukuran

{table}

Audio pembanding ada di `{out_dir}/`: `*_real.wav`, `*_roundtrip.wav`, `meanmel.wav`, `*_model.wav`.

## Interpretasi otomatis

{chr(10).join(verdict) if verdict else "- (data tidak cukup untuk interpretasi)"}

## Cara membaca

| Ukuran | Arti | Arah yang diharapkan |
|---|---|---|
| `band_8-11k` | porsi energi 8-11 kHz | harus > 0 untuk konsonan tajam; 0 berarti `mel_fmax` memotong |
| `hlr_4k` | energi > 4 kHz / < 4 kHz | turun tajam = suara teredam |
| `centroid_hz` | "kecerahan" spektral | rendah = muffled |
| `f0_std_semitones` | variasi intonasi | rendah = monoton/zombie |
| `mel_delta_abs` | rata-rata |Δmel| antar frame | kecil = over-smoothing (blur) |
"""
    Path(args.report).write_text(body, encoding="utf-8")
    print()
    print(table)
    print()
    for line in verdict:
        print(line)
    print(f"\n  report: {Path(args.report).resolve()}")
    print(f"  audio : {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
