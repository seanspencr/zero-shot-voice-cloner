"""Render one *named* utterance from a checkpoint, next to its reference audio.

Why this exists: during training, ``--render-every`` writes a random crop of a random utterance from
the last batch, so two runs never produce comparable files. Judging "this one sounds worse" from those
is a coin flip. This tool fixes the content:

    python preview.py --checkpoint checkpoints/best.pt --utt-id 2911_12359_000018_000000

writes, in ``--out-dir`` (default ``preview``):

    <utt>_ref.wav   the real recording's centre crop  (4.0 s by default)
    <utt>_gen.wav   the model's conversion of the same crop
    <utt>_meta.json transcript, crop window, timings, conditioning info

Run it once per checkpoint with the same ``--utt-id`` and the comparison is finally meaningful. With no
``--utt-id`` it picks a deterministic utterance from the split (first one with a >=4 s duration).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from evaluate import load_generator_from_checkpoint, resolve_device
from features import CONTENT_SOURCES, DEFAULT_HUBERT_DIR, SPEAKER_SOURCES, MissingWeightsError
from libritts import DEFAULT_ROOT, LibriTTSDataset, Utterance, scan_split
from vocoder import DEFAULT_MODEL_DIR, load_vocoder, missing_checkpoint_help, vocoder_ready

__all__ = ["parse_args", "find_utterance", "main"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a single named utterance and write ref/gen wavs for A/B listening.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="path to latest.pt / best.pt")
    parser.add_argument("--utt-id", default=None, help="e.g. 2911_12359_000018_000000 (default: first of the split)")
    parser.add_argument("--split", default="test-clean", help="split to search when --utt-id is given or not")
    parser.add_argument("--libritts-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--segment-seconds", type=float, default=4.0, help="centre-crop length used for ref and gen")
    parser.add_argument("--content-source", choices=CONTENT_SOURCES, default="hubert")
    parser.add_argument("--speaker-source", choices=SPEAKER_SOURCES, default="xtts")
    parser.add_argument("--hubert-dir", default=str(DEFAULT_HUBERT_DIR))
    parser.add_argument("--hubert-layer", type=int, default=9)
    parser.add_argument("--xtts-checkpoint", default=None)
    parser.add_argument("--enrollment-utt", default=None, help="use this utterance's voice instead (zero-shot A->B)")
    parser.add_argument("--out-dir", default="preview")
    parser.add_argument("--vocoder-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def find_utterance(args: argparse.Namespace) -> tuple[Utterance, str]:
    """Locate the requested utterance (searching the split), or pick a deterministic default."""
    scanned = scan_split(
        root=args.libritts_root,
        split=args.split,
        limit=None,
        min_seconds=args.segment_seconds,
        shuffle_seed=None,  # keep corpus order => deterministic default choice
        use_cache=True,
    )
    if args.utt_id:
        for utt in scanned:
            if utt.utt_id == args.utt_id:
                return utt, args.split
        raise SystemExit(
            f"utterance '{args.utt_id}' not found in split '{args.split}' (or it is shorter than "
            f"{args.segment_seconds:g}s)"
        )
    if not scanned:
        raise SystemExit(f"no utterance in '{args.split}' is at least {args.segment_seconds:g}s long")
    return scanned[0], args.split


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    args.device = str(device)

    if not vocoder_ready(args.vocoder_dir):
        print(missing_checkpoint_help(args.vocoder_dir))
        return 2
    vocoder = load_vocoder(args.vocoder_dir, device=device)
    if vocoder is None:
        return 2

    generator, saved = load_generator_from_checkpoint(args.checkpoint, device)
    saved_dim = int(saved.get("speaker_dim", 256))

    utt, split = find_utterance(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Preview")
    print(f"  checkpoint : {args.checkpoint}  (speaker_dim={saved_dim})")
    print(f"  utterance  : {utt.utt_id}  (speaker {utt.speaker_id}, {split})")
    print(f'  transcript : "{utt.text[:100]}{"..." if len(utt.text) > 100 else ""}"')

    # ---- conditioning: this utterance's own voice, or another one's (zero-shot A->B)
    enrollment = utt
    if args.enrollment_utt:
        for candidate in scan_split(root=args.libritts_root, split=split, limit=None,
                                    min_seconds=args.segment_seconds, shuffle_seed=None, use_cache=True):
            if candidate.utt_id == args.enrollment_utt:
                enrollment = candidate
                break
        else:
            raise SystemExit(f"enrollment utterance '{args.enrollment_utt}' not found in '{split}'")

    from features import build_content_extractor, build_speaker_extractor

    speaker_extractor = build_speaker_extractor(
        args.speaker_source, checkpoint=args.xtts_checkpoint, device=str(device)
    )
    produced = getattr(speaker_extractor, "out_dim", saved_dim)
    if produced != saved_dim:
        raise SystemExit(
            f"Speaker-embedding size mismatch: checkpoint expects {saved_dim}-d, "
            f"--speaker-source {args.speaker_source} gives {produced}-d."
        )

    dataset = LibriTTSDataset(
        [utt],
        vocoder=vocoder,
        segment_seconds=args.segment_seconds,
        train=False,  # centre crop: identical every time
        content_extractor=build_content_extractor(
            args.content_source, model_dir=args.hubert_dir, layer=args.hubert_layer, device=str(device)
        ),
        speaker_extractor=speaker_extractor,
    )
    try:
        dataset.validate()
    except MissingWeightsError as exc:
        raise SystemExit(str(exc)) from exc

    # ---- conditional sample, but with the ENROLMENT voice (may differ from the item's own)
    sample = dataset[0]
    content = sample["content"].unsqueeze(0).to(device)
    mel_target = sample["mel"]
    if enrollment.utt_id == utt.utt_id:
        speaker_vec = sample["speaker"].unsqueeze(0).to(device)
        enroll_note = "this utterance (identity / reconstruction)"
    else:
        wav_e, sr_e = sf.read(enrollment.wav_path, dtype="float32", always_2d=False)
        mel_e = vocoder.mel_from_waveform(np.asarray(wav_e, dtype=np.float32), sample_rate=sr_e)[0].cpu().numpy()
        speaker_vec = torch.from_numpy(
            speaker_extractor.extract(np.asarray(wav_e, dtype=np.float32), sr_e, mel_e)
        ).unsqueeze(0).to(device)
        enroll_note = f"{enrollment.utt_id} (speaker {enrollment.speaker_id}) -- zero-shot A->B"

    started = time.perf_counter()
    with torch.no_grad():
        f0 = sample["f0"].unsqueeze(0).to(device) if "f0" in sample else None
        mel_pred = generator(content, speaker_vec, f0)
    gen_seconds = time.perf_counter() - started
    wav_gen = vocoder.forward(mel_pred)[0, 0].detach().cpu().numpy()

    # ---- reference crop, produced exactly the way the dataset cropped it
    ref_wav, ref_sr = sf.read(utt.wav_path, dtype="float32", always_2d=False)
    ref_crop = dataset.crop_segment(np.asarray(ref_wav, dtype=np.float32), 0, int(ref_sr))

    gen_path = out_dir / f"{utt.utt_id}_gen.wav"
    ref_path = out_dir / f"{utt.utt_id}_ref.wav"
    sf.write(str(gen_path), np.clip(wav_gen, -1.0, 1.0), vocoder.sample_rate, subtype="PCM_16")
    sf.write(str(ref_path), np.clip(ref_crop, -1.0, 1.0), int(ref_sr), subtype="PCM_16")

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": saved.get("epoch"),
        "utterance": utt.utt_id,
        "speaker_id": utt.speaker_id,
        "split": split,
        "text": utt.text,
        "crop_start_s": float(sample.get("crop_start", 0.0)),
        "crop_end_s": float(sample.get("crop_end", 0.0)),
        "utterance_s": float(sample.get("duration", 0.0)),
        "enrollment": enroll_note,
        "mel_frames": int(mel_pred.shape[-1]),
        "gen_wav": str(gen_path.resolve()),
        "ref_wav": str(ref_path.resolve()),
        "content_source": args.content_source,
        "speaker_source": args.speaker_source,
    }
    meta_path = out_dir / f"{utt.utt_id}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"  crop       : {meta['crop_start_s']:.2f}s-{meta['crop_end_s']:.2f}s of {meta['utterance_s']:.2f}s")
    print(f"  voice from : {enroll_note}")
    print(f"  mel frames : {meta['mel_frames']}  ({gen_seconds * 1000:.0f} ms to generate)")
    print(f"  ref        : {ref_path}")
    print(f"  gen        : {gen_path}")
    print(f"  meta       : {meta_path}")
    print("\nListen to the two side by side; the words are identical, only the voice should differ.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
