"""Evaluate a trained generator: convert held-out utterances and report MCD / SECS.

Two modes, same code:

* **zero-shot evaluation** (the default setup): the content comes from one utterance, and the speaker
  embedding comes from a *different* utterance of the same target speaker — the enrolment clip the user
  would provide at inference time. `--enrollment-mode same` falls back to the utterance's own embedding
  (what the model is trained on).
* **quick validation** inside training: :func:`compute_metrics` is reused by `train.py --val-every N`,
  which reports mel-L1 + MCD on a fixed centre-cropped subset (no vocoding, no SECS) so a CPU run is
  measurable while it trains.

Examples::

    # full evaluation on the unseen-speaker test split, with audio written out
    python evaluate.py --checkpoint checkpoints/best.pt --split test-clean --max-utterances 50 \
        --out-dir eval_out

    # only spectral metrics (no speaker encoder available yet)
    python evaluate.py --checkpoint checkpoints/best.pt --max-utterances 20 --no-secs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from features import (
    CONTENT_SOURCES,
    DEFAULT_HUBERT_DIR,
    SPEAKER_SOURCES,
    MissingWeightsError,
    build_content_extractor,
    build_speaker_extractor,
)
from libritts import DEFAULT_ROOT, LibriTTSDataset, Utterance, scan_split
from metrics import EvaluationReport, mel_cepstral_distortion, speaker_cosine_similarity
from models import Generator
from vocoder import DEFAULT_MODEL_DIR, HiFiGANVocoder, load_vocoder, missing_checkpoint_help, vocoder_ready

__all__ = ["build_conversion_loader", "compute_metrics", "parse_args", "main"]


# --------------------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert held-out utterances with a trained generator and report MCD/SECS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="path to latest.pt / best.pt from train.py")
    parser.add_argument("--split", default="test-clean", help="LibriTTS split (test-clean = unseen speakers)")
    parser.add_argument("--libritts-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--max-utterances", type=int, default=50)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=4, help="only affects loader batching, metrics are per item")
    parser.add_argument("--content-source", choices=CONTENT_SOURCES, default="mel-projection")
    parser.add_argument("--speaker-source", choices=SPEAKER_SOURCES, default="xtts")
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=-1.0,
        help=(
            "skip clips shorter than this (negative = use --segment-seconds, so nothing is zero-padded; "
            "0 = keep everything and pad)"
        ),
    )
    parser.add_argument("--hubert-dir", default=str(DEFAULT_HUBERT_DIR))
    parser.add_argument("--hubert-layer", type=int, default=9)
    parser.add_argument("--xtts-checkpoint", default=None, help="path to the XTTS v2 model.pth")
    parser.add_argument("--speaker-dim", type=int, default=512, help="embedding size (512 for xtts)")
    parser.add_argument(
        "--enrollment-mode",
        choices=("different-utterance", "same", "other-speaker"),
        default="different-utterance",
        help=(
            "different-utterance = same speaker, different clip (default); same = the item's own clip; "
            "other-speaker = cross-speaker zero-shot (content from speaker A, embedding from speaker B)"
        ),
    )
    parser.add_argument(
        "--target-speaker",
        default=None,
        help="with --enrollment-mode other-speaker: use this speaker id as the target voice for every item",
    )
    parser.add_argument(
        "--require-enrollment",
        action="store_true",
        help="only evaluate speakers that have >=2 clips, so the enrolment is always a different clip",
    )
    parser.add_argument("--out-dir", default=None, help="write converted wavs here (default: no audio)")
    parser.add_argument("--vocoder-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-secs", action="store_true", help="skip speaker-similarity (needs the XTTS speaker encoder)")
    parser.add_argument("--report", default=None, help="write the JSON report here")
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


# --------------------------------------------------------------------------------------
# setup helpers
# --------------------------------------------------------------------------------------
def load_generator_from_checkpoint(path: str | Path, device: torch.device) -> tuple[Generator, dict[str, Any]]:
    """Rebuild the Generator with the *saved* architecture args, then load its weights."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {}) or {}
    generator = Generator(
        in_dim=saved.get("content_dim", 768),
        hidden_dim=saved.get("hidden_dim", 256),
        mel_bins=saved.get("mel_bins", 80),
        num_blocks=saved.get("num_blocks", 4),
        speaker_dim=saved.get("speaker_dim", 256),
    )
    generator.load_state_dict(ckpt["generator"])
    generator.to(device).eval()
    return generator, saved


def filter_speakers_with_pairs(
    utterances: list[Utterance], max_utterances: int | None = None, per_speaker: int = 2
) -> list[Utterance]:
    """Keep only speakers with >= 2 clips, taking up to ``per_speaker`` clips each.

    This is what makes the SECS number a *zero-shot* number: the enrolment clip is always a different
    utterance from the content clip, never the item itself.
    """
    grouped: dict[str, list[Utterance]] = {}
    for utt in utterances:
        grouped.setdefault(utt.speaker_id, []).append(utt)
    kept: list[Utterance] = []
    for items in grouped.values():
        if len(items) >= 2:
            kept.extend(items[:per_speaker])
    if max_utterances is not None:
        kept = kept[:max_utterances]
    return kept


def build_conversion_loader(
    args: argparse.Namespace,
    vocoder: HiFiGANVocoder,
    utterances: list[Utterance] | None = None,
) -> tuple[LibriTTSDataset, list[Utterance]]:
    """Centre-cropped, unshuffled dataset for evaluation (deterministic, no augmentation)."""
    if utterances is None:
        min_seconds = args.segment_seconds if getattr(args, "min_seconds", -1.0) < 0 else args.min_seconds
        utterances = scan_split(
            root=args.libritts_root,
            split=args.split,
            limit=None if getattr(args, "require_enrollment", False) else args.max_utterances,
            min_seconds=min_seconds,
            shuffle_seed=0,
            use_cache=True,
            probe_duration=None if min_seconds > 0 else False,
        )
    if getattr(args, "require_enrollment", False):
        utterances = filter_speakers_with_pairs(utterances, max_utterances=args.max_utterances)
        if not utterances:
            raise RuntimeError(
                f"no speaker in '{args.split}' has 2 clips; drop --require-enrollment"
            )
    dataset = LibriTTSDataset(
        utterances,
        vocoder=vocoder,
        segment_seconds=args.segment_seconds,
        train=False,
        content_extractor=build_content_extractor(
            args.content_source, model_dir=args.hubert_dir, layer=args.hubert_layer, device=args.device
        ),
        speaker_extractor=build_speaker_extractor(
            args.speaker_source, checkpoint=args.xtts_checkpoint, device=args.device
        ),
    )
    return dataset, utterances


def _enrollment_audio(
    item: Utterance,
    by_speaker: dict[str, list[Utterance]],
    mode: str,
    target_speaker: str | None = None,
) -> tuple[np.ndarray, int, bool, str]:
    """Waveform for the speaker embedding, plus the speaker id it came from.

    Modes:
        ``different-utterance`` (default): another clip of the *same* speaker -- true zero-shot for
            that speaker, and the only mode where MCD is meaningful.
        ``same``: the item's own audio (the identity shortcut; counted in the report).
        ``other-speaker``: a *different* speaker's clip -- cross-speaker timbre transfer. In this mode
            MCD compares against the source speaker's own mel, so it stops being a fidelity measure.

    Returns ``(audio, sample_rate, used_different_clip, speaker_id)``.
    """
    import soundfile as sf

    others = [u for u in by_speaker.get(item.speaker_id, []) if u.utt_id != item.utt_id]

    if mode == "other-speaker":
        if target_speaker:
            candidates = by_speaker.get(str(target_speaker), [])
        else:
            speaker_ids = sorted(by_speaker)
            idx = speaker_ids.index(item.speaker_id)
            candidates = by_speaker[speaker_ids[(idx + 1) % len(speaker_ids)]]
        if candidates:
            chosen = candidates[0]
            wav, sr = sf.read(chosen.wav_path, dtype="float32", always_2d=False)
            return np.asarray(wav, dtype=np.float32), int(sr), True, chosen.speaker_id

    if mode in ("different-utterance", "other-speaker") and others:
        wav, sr = sf.read(others[0].wav_path, dtype="float32", always_2d=False)
        return np.asarray(wav, dtype=np.float32), int(sr), True, others[0].speaker_id

    wav, sr = sf.read(item.wav_path, dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32), int(sr), False, item.speaker_id


# --------------------------------------------------------------------------------------
# core metric computation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def compute_metrics(
    generator: Generator,
    dataset: LibriTTSDataset,
    vocoder: HiFiGANVocoder,
    device: torch.device,
    embedder: Any | None = None,
    render_dir: str | Path | None = None,
    enrollment_mode: str = "same",
    target_speaker: str | None = None,
    progress_every: int = 0,
    checkpoint: str = "",
) -> EvaluationReport:
    """Convert every item in ``dataset`` and aggregate MCD / mel-L1 / (optionally) SECS.

    ``embedder`` is a ``XTTSSpeakerExtractor`` for SECS, or ``None`` to skip it. ``checkpoint`` and the
    start time are recorded in the report so two runs stay comparable after the fact.
    """
    import soundfile as sf

    by_speaker: dict[str, list[Utterance]] = {}
    for utt in dataset.utterances:
        by_speaker.setdefault(utt.speaker_id, []).append(utt)

    if render_dir is not None:
        Path(render_dir).mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mcds: list[float] = []
    mel_l1s: list[float] = []
    secs_target: list[float] = []
    secs_source: list[float] = []
    fallbacks = 0
    per_item: list[dict[str, Any]] = []

    for index in range(len(dataset)):
        utt = dataset.utterances[index]
        sample = dataset[index]
        content = sample["content"].unsqueeze(0).to(device)
        speaker = sample["speaker"].unsqueeze(0).to(device)
        mel_target = sample["mel"].to(device)

        f0 = sample["f0"].unsqueeze(0).to(device) if "f0" in sample else None
        mel_pred = generator(content, speaker, f0)[0]  # [80, T]
        mcd = mel_cepstral_distortion(mel_pred.cpu().numpy(), mel_target.cpu().numpy())
        mel_l1 = float(torch.mean(torch.abs(mel_pred - mel_target)).item())
        mcds.append(mcd)
        mel_l1s.append(mel_l1)

        row: dict[str, Any] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "utt_id": utt.utt_id,
            "speaker_id": utt.speaker_id,
            "text": utt.text,
            "mcd": mcd,
            "mel_l1": mel_l1,
        }

        needs_audio = embedder is not None or render_dir is not None
        if needs_audio:
            wav_gen = vocoder.forward(mel_pred.unsqueeze(0))  # [1, 1, N]
            audio_gen = wav_gen[0, 0].detach().cpu().numpy()
            if render_dir is not None:
                out_path = Path(render_dir) / f"{utt.utt_id}_gen.wav"
                sf.write(str(out_path), np.clip(audio_gen, -1.0, 1.0), vocoder.sample_rate, subtype="PCM_16")
                row["wav"] = str(out_path)
                # also write the reference crop so the two can be compared by ear
                ref_wav, ref_sr = sf.read(utt.wav_path, dtype="float32", always_2d=False)
                ref_crop = dataset.crop_segment(np.asarray(ref_wav, dtype=np.float32), index, int(ref_sr))
                ref_path = Path(render_dir) / f"{utt.utt_id}_ref.wav"
                sf.write(str(ref_path), np.clip(ref_crop, -1.0, 1.0), int(ref_sr), subtype="PCM_16")
                row["ref_wav"] = str(ref_path)

        if embedder is not None:
            enroll_wav, enroll_sr, used_different, enroll_spk = _enrollment_audio(
                utt, by_speaker, enrollment_mode, target_speaker
            )
            if not used_different:
                fallbacks += 1
            src_wav, src_sr = sf.read(utt.wav_path, dtype="float32", always_2d=False)
            secs_t = speaker_cosine_similarity(embedder, audio_gen, vocoder.sample_rate, enroll_wav, enroll_sr)
            secs_s = speaker_cosine_similarity(
                embedder, audio_gen, vocoder.sample_rate, np.asarray(src_wav, dtype=np.float32), int(src_sr)
            )
            secs_target.append(secs_t)
            secs_source.append(secs_s)
            row["secs_target"] = secs_t
            row["secs_source"] = secs_s
            row["enrollment_speaker"] = enroll_spk

        per_item.append(row)
        if progress_every and (index + 1) % progress_every == 0:
            print(f"  ... {index + 1}/{len(dataset)} utterances  MCD {np.mean(mcds):.3f} dB")

    report = EvaluationReport(
        n_items=len(mcds),
        mcd_mean=float(np.mean(mcds)) if mcds else float("nan"),
        mcd_std=float(np.std(mcds)) if mcds else float("nan"),
        mel_l1_mean=float(np.mean(mel_l1s)) if mel_l1s else float("nan"),
        timestamp=started_at,
        checkpoint=checkpoint,
        split=dataset.utterances[0].split if dataset.utterances else "",
        enrollment_mode=enrollment_mode,
        secs_target_mean=float(np.mean(secs_target)) if secs_target else None,
        secs_target_std=float(np.std(secs_target)) if secs_target else None,
        secs_source_mean=float(np.mean(secs_source)) if secs_source else None,
        secs_note="" if secs_target else "SECS skipped (no speaker encoder)",
        enrollment_fallbacks=fallbacks,
        per_item=per_item,
    )
    return report


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    args.device = str(device)  # extractors take a concrete torch device, never "auto"

    if not vocoder_ready(args.vocoder_dir):
        print(missing_checkpoint_help(args.vocoder_dir))
        return 2
    vocoder = load_vocoder(args.vocoder_dir, device=device)
    if vocoder is None:
        return 2

    print("Evaluating converted speech")
    print(vocoder.describe())
    generator, saved = load_generator_from_checkpoint(args.checkpoint, device)
    saved_speaker_dim = int(saved.get("speaker_dim", 256))
    print(f"  checkpoint  : {args.checkpoint}  (trained {saved.get('epochs', '?')} epochs, "
          f"speaker_dim={saved_speaker_dim})")
    print(f"  split       : {args.split}  ({args.max_utterances} utterances, "
          f"enrolment={args.enrollment_mode})")

    embedder = None
    if not args.no_secs:
        try:
            candidate = build_speaker_extractor(
                "xtts", checkpoint=args.xtts_checkpoint, device=str(device)
            )
            candidate.check_ready()  # type: ignore[attr-defined]
            embedder = candidate
        except MissingWeightsError as exc:
            print(f"  SECS        : unavailable -- {str(exc).splitlines()[0]}")
            print("                (the XTTS checkpoint was not found, or pass --no-secs)")

    # A checkpoint is bound to the speaker-embedding size it was trained with; mixing a 256-d model
    # with the 512-d XTTS encoder (or vice versa) fails deep inside AdaIN with a cryptic shape error.
    configured_dim = getattr(
        build_speaker_extractor(args.speaker_source, checkpoint=args.xtts_checkpoint), "out_dim", None
    )
    if configured_dim is not None and configured_dim != saved_speaker_dim:
        raise SystemExit(
            f"Speaker-embedding size mismatch: '{args.checkpoint}' was trained with {saved_speaker_dim}-d "
            f"embeddings but --speaker-source {args.speaker_source} produces {configured_dim}-d.\n"
            f"  Retrain with the new pipeline (train-short.bat) or pick a matching --speaker-source."
        )

    dataset, utterances = build_conversion_loader(args, vocoder)
    print(f"  dataset     : {dataset.describe()}")
    speakers = len({u.speaker_id for u in utterances})
    print(f"  speakers    : {speakers}")
    print()

    report = compute_metrics(
        generator,
        dataset,
        vocoder,
        device,
        embedder=embedder,
        render_dir=args.out_dir,
        enrollment_mode=args.enrollment_mode,
        target_speaker=args.target_speaker,
        progress_every=max(1, len(dataset) // 5),
        checkpoint=args.checkpoint,
    )
    print("Results")
    print(report.summary())
    if args.enrollment_mode == "other-speaker":
        print(
            "  caveat: cross-speaker mode -- MCD compares the output against the SOURCE speaker's own mel, "
            "so read it as a rough content/duration proxy only. The timbre verdict is SECS: it should rise "
            "toward the target and fall for the source."
        )
    if args.out_dir:
        print(f"  audio       : {os.path.abspath(args.out_dir)}")

    report_path = args.report or (os.path.join(args.out_dir, "report.json") if args.out_dir else None)
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        csv_path = str(Path(report_path).with_suffix(".csv"))
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(report.per_item[0].keys()) if report.per_item else [])
            writer.writeheader()
            writer.writerows(report.per_item)
        print(f"  report      : {report_path}  (+ {csv_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
