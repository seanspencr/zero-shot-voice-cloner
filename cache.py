"""On-disk feature cache: extract HuBERT / speaker / F0 **once** per utterance.

Why this exists
---------------
Per training step on CPU, ~85 % of the wall clock is data extraction, not the GAN:

    HuBERT (4 s clip)  ~700 ms   +   XTTS speaker encoder ~250 ms   +   F0 ~90 ms

With `--max-minutes 10` that caps you at ~170 steps per session — two orders of magnitude short of
what converging a VC GAN needs. Caching the extracted features turns those milliseconds into a
memory-mapped `.npz` read, so a step costs only the GAN forward/backward (~1.3 s at batch 4).

What is stored per utterance (full length, not cropped)
-------------------------------------------------------
``mel``      ``[80, T]``      float32   vocoder AudioProcessor at 22.05 kHz
``content``  ``[T, 768]``     float16   HuBERT layer N, already interpolated to mel frames
``f0``       ``[T, 2]``       float32   log-F0 contour + voiced flag
``speaker``  ``[512]``        float32   XTTS speaker encoder (computed on the whole clip, so it is
                                        stabler than a 4 s-crop embedding)

Cropping happens at *read* time in frame space, so every epoch can still use a different 4 s window
without re-extraction. float16 for content halves the disk cost with no measurable effect on features
that already sat 768-dim wide (~0.8 MB per utterance at the median 5 s duration).

Layout::

    datasets/cache/<content_source>-L<layer>/<split>/<speaker>/<chapter>/<utt>.npz

Usage::

    python cache.py --split train-clean-100 --max-utterances 5000      # build (resumable)
    python train.py --data libritts --cache-dir datasets/cache/hubert-L9
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from features import (
    CONTENT_SOURCES,
    DEFAULT_HUBERT_DIR,
    SPEAKER_SOURCES,
    build_content_extractor,
    build_speaker_extractor,
    f0_from_waveform,
)
from libritts import DEFAULT_ROOT, Utterance, scan_split
from vocoder import DEFAULT_MODEL_DIR, load_vocoder, missing_checkpoint_help, vocoder_ready

__all__ = ["CACHE_ROOT", "cache_dir_for", "cache_file", "build_cache"]

CACHE_ROOT = DEFAULT_ROOT.parent / "cache"


def cache_dir_for(root: str | Path, content_source: str, hubert_layer: int) -> Path:
    """Directory holding the cache for a given feature configuration."""
    tag = f"{content_source}" + (f"-L{hubert_layer}" if content_source == "hubert" else "")
    return Path(root).parent / "cache" / tag


def cache_file(
    cache_dir: str | Path, split: str, utt: Utterance, content_source: str = "hubert", hubert_layer: int = 9
) -> Path:
    """Path of one utterance's cached features inside ``cache_dir``."""
    return Path(cache_dir) / split / utt.speaker_id / utt.chapter_id / f"{utt.utt_id}.npz"


def build_cache(
    root: str | Path = DEFAULT_ROOT,
    split: str = "train-clean-100",
    max_utterances: int | None = None,
    content_source: str = "hubert",
    speaker_source: str = "xtts",
    hubert_dir: str | Path = DEFAULT_HUBERT_DIR,
    hubert_layer: int = 9,
    xtts_checkpoint: str | None = None,
    vocoder_dir: str | Path = DEFAULT_MODEL_DIR,
    device: str = "cpu",
    min_seconds: float = 0.0,
    progress_every: int = 200,
) -> Path:
    """Extract and store features for ``split``. Resumable: existing files are skipped."""
    import soundfile as sf

    if not vocoder_ready(vocoder_dir):
        raise SystemExit(missing_checkpoint_help(vocoder_dir))
    vocoder = load_vocoder(vocoder_dir, device=device)
    if vocoder is None:
        raise SystemExit("vocoder failed to load")

    utterances = scan_split(
        root=root,
        split=split,
        limit=max_utterances,
        min_seconds=min_seconds,
        shuffle_seed=0,
        use_cache=True,
    )
    out_dir = cache_dir_for(root, content_source, hubert_layer)
    content_extractor = build_content_extractor(
        content_source, model_dir=hubert_dir, layer=hubert_layer, device=device
    )
    speaker_extractor = build_speaker_extractor(speaker_source, checkpoint=xtts_checkpoint, device=device)
    for extractor in (content_extractor, speaker_extractor):
        check = getattr(extractor, "check_ready", None)
        if callable(check):
            check()

    print(f"Building cache -> {out_dir}")
    print(f"  {len(utterances)} utterances from '{split}' "
          f"(content={content_source}, speaker={speaker_source}, min_seconds={min_seconds:g})")

    started = time.perf_counter()
    written = skipped = failed = 0
    total_bytes = 0
    for index, utt in enumerate(utterances, start=1):
        target = cache_file(out_dir, split, utt, content_source, hubert_layer)
        if target.is_file():
            skipped += 1
            continue
        try:
            wav, sr = sf.read(utt.wav_path, dtype="float32", always_2d=False)
            wav = np.asarray(wav, dtype=np.float32)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)

            mel = vocoder.mel_from_waveform(wav, sample_rate=sr)[0].cpu().numpy()  # [80, T]
            frames = mel.shape[1]
            content = content_extractor.extract(wav, sr, mel, frames)  # [T, 768]
            f0 = f0_from_waveform(wav, sr, hop_length=vocoder.hop_length)  # [T, 2]
            if f0.shape[0] >= frames:
                f0 = f0[:frames]
            else:
                f0 = np.pad(f0, ((0, frames - f0.shape[0]), (0, 0)))
            speaker = speaker_extractor.extract(wav, sr, mel)  # [D]

            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                target,
                mel=mel.astype(np.float32),
                content=content.astype(np.float16),
                f0=f0.astype(np.float32),
                speaker=speaker.astype(np.float32),
                utt_id=np.array(utt.utt_id),
                speaker_id=np.array(utt.speaker_id),
            )
            written += 1
            total_bytes += target.stat().st_size
        except Exception as exc:  # keep going: one bad file must not kill an 8-hour job
            failed += 1
            print(f"  [skip] {utt.utt_id}: {type(exc).__name__}: {exc}")

        if progress_every and index % progress_every == 0:
            elapsed = time.perf_counter() - started
            rate = (written + skipped) / max(elapsed, 1e-6)
            remaining = len(utterances) - index
            print(f"  {index}/{len(utterances)}  {rate:.2f} utt/s  "
                  f"eta {remaining / max(rate, 1e-6) / 60:.1f} min  "
                  f"({total_bytes / 1e9:.2f} GB written)")

    elapsed = time.perf_counter() - started
    print(f"Done in {elapsed / 60:.1f} min: {written} written, {skipped} already cached, {failed} failed")
    print(f"  cache size (this run): {total_bytes / 1e9:.2f} GB   location: {out_dir}")
    if written:
        print(f"  ~{total_bytes / written / 1e6:.2f} MB per utterance")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Extract HuBERT / speaker / F0 features once and store them for fast training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--split", default="train-clean-100")
    p.add_argument("--libritts-root", default=str(DEFAULT_ROOT))
    p.add_argument("--max-utterances", type=int, default=None, help="how many to cache (default: all)")
    p.add_argument("--content-source", choices=CONTENT_SOURCES, default="hubert")
    p.add_argument("--speaker-source", choices=SPEAKER_SOURCES, default="xtts")
    p.add_argument("--hubert-dir", default=str(DEFAULT_HUBERT_DIR))
    p.add_argument("--hubert-layer", type=int, default=9)
    p.add_argument("--xtts-checkpoint", default=None)
    p.add_argument("--vocoder-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--device", default="cpu")
    p.add_argument("--min-seconds", type=float, default=4.0,
                   help="skip clips shorter than this (match --segment-seconds to avoid padding)")
    p.add_argument("--progress-every", type=int, default=200)
    args = p.parse_args(argv)

    build_cache(
        root=args.libritts_root,
        split=args.split,
        max_utterances=args.max_utterances,
        content_source=args.content_source,
        speaker_source=args.speaker_source,
        hubert_dir=args.hubert_dir,
        hubert_layer=args.hubert_layer,
        xtts_checkpoint=args.xtts_checkpoint,
        vocoder_dir=args.vocoder_dir,
        device=args.device,
        min_seconds=args.min_seconds,
        progress_every=args.progress_every,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
