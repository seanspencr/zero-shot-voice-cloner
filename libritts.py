"""LibriTTS / LibriTTS-R dataset: real audio -> ``(content, speaker, mel)``.

Expected tree (this is what ``datasets/libritts/`` holds)::

    datasets/libritts/
    ├── train-clean-100/          <- LibriTTS-R (restored audio, same layout/ids)
    │   └── <speaker_id>/<chapter_id>/<utt_id>.wav
    │       <utt_id>.normalized.txt / <utt_id>.original.txt
    ├── dev-clean/                <- LibriTTS
    └── test-clean/               <- LibriTTS

Facts about this corpus (measured on the extracted copy):

* 24 kHz mono PCM_16 wav; 33,232 / 5,736 / 4,837 utterances for
  train-clean-100 / dev-clean / test-clean; 247 / 40 / 39 speakers.
* The speaker ids of the three splits are **disjoint**, so `dev-clean` works as a validation set and
  `test-clean` is a genuine *unseen speaker* set -- which is what makes zero-shot VC evaluation
  meaningful.
* Median utterance is ~4–5 s, the longest ~30 s, so training crops to a fixed segment.

The mel target is produced by the **installed HiFi-GAN vocoder's own ``AudioProcessor``** (see
``vocoder.py``), so the training target and the vocoder can never disagree: 24 kHz audio is
resampled to the vocoder's 22.05 kHz and turned into 80-bin raw log-mel.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from features import (
    ContentExtractor,
    SpeakerExtractor,
    build_content_extractor,
    build_speaker_extractor,
    f0_from_waveform,
)

__all__ = [
    "DEFAULT_ROOT",
    "SPLITS",
    "Utterance",
    "LibriTTSSample",
    "LibriTTSDataset",
    "scan_split",
    "build_libritts_loader",
    "split_summary",
]

DEFAULT_ROOT = Path(__file__).resolve().parent / "datasets" / "libritts"
SPLITS = ("train-clean-100", "dev-clean", "test-clean")

_UTT_RE = re.compile(r"^(\d+)_(\d+)_(\d+)_(\d+)$")


@dataclass(frozen=True)
class Utterance:
    """One corpus item."""

    utt_id: str
    speaker_id: str
    chapter_id: str
    wav_path: str
    text: str
    split: str
    duration: float


class LibriTTSSample(TypedDict):
    content: torch.Tensor  # [T, 768]
    speaker: torch.Tensor  # [D]
    mel: torch.Tensor  # [80, T]
    f0: torch.Tensor  # [T, 2] log-F0 contour + voiced flag, frames aligned with the mel
    utt_id: str  # source utterance, for traceability
    text: str  # transcript of the source utterance
    crop_start: float  # seconds into the utterance where the converted segment begins
    crop_end: float  # seconds where it ends
    duration: float  # full utterance length in seconds


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def scan_split(
    root: str | Path = DEFAULT_ROOT,
    split: str = "train-clean-100",
    limit: int | None = None,
    min_seconds: float = 0.0,
    max_seconds: float | None = None,
    shuffle_seed: int | None = 0,
    use_cache: bool = True,
    probe_duration: bool | None = None,
) -> list[Utterance]:
    """Scan ``<root>/<split>`` into a deterministic list of utterances.

    Args:
        limit: keep at most this many utterances (the smoke-run switch).
        min_seconds / max_seconds: duration filter; requires reading every wav header
            (``probe_duration`` is then forced on).
        shuffle_seed: ``None`` keeps speaker/chapter order; an int shuffles with that seed so a
            ``limit`` produces a speaker-diverse subset instead of the first few folders.
        probe_duration: read each wav's duration with ``soundfile.info``. ``None`` means "only when a
            duration filter is set", which keeps a full-corpus scan at a few seconds instead of
            opening 33k files.

    A JSON manifest cache is written to ``datasets/manifests/<split>_<filters>.json`` so repeated runs
    are instant.
    """
    import soundfile as sf

    root = Path(root)
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"LibriTTS split not found: {split_dir}")

    if probe_duration is None:
        probe_duration = bool(min_seconds > 0.0 or max_seconds is not None)

    cache_dir = root.parent / "manifests"
    cache_key = (
        f"{split}_limit{limit}_min{min_seconds}_max{max_seconds}_seed{shuffle_seed}"
        f"_dur{int(bool(probe_duration))}"
    )
    cache_path = cache_dir / f"{cache_key}.json"

    if use_cache and cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return [Utterance(**item) for item in payload]

    items: list[Utterance] = []
    skipped_orphans = 0
    for wav_path in sorted(split_dir.rglob("*.wav")):
        utt_id = wav_path.stem
        match = _UTT_RE.match(utt_id)
        if match is None:
            continue
        speaker_id, chapter_id = match.group(1), match.group(2)
        text_path = wav_path.with_name(f"{utt_id}.normalized.txt")
        if not text_path.is_file():
            skipped_orphans += 1
            continue
        duration = 0.0
        if probe_duration:
            try:
                info = sf.info(str(wav_path))
            except Exception:  # unreadable file: skip rather than crash a long scan
                skipped_orphans += 1
                continue
            duration = info.frames / info.samplerate if info.samplerate else 0.0
            if duration < min_seconds:
                continue
            if max_seconds is not None and duration > max_seconds:
                continue
        items.append(
            Utterance(
                utt_id=utt_id,
                speaker_id=speaker_id,
                chapter_id=chapter_id,
                wav_path=str(wav_path),
                text=_read_text(text_path),
                split=split,
                duration=float(duration),
            )
        )

    if shuffle_seed is not None:
        # shuffle so that a small `limit` still covers many speakers
        random.Random(shuffle_seed).shuffle(items)
    if limit is not None:
        items = items[:limit]

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps([asdict(i) for i in items]), encoding="utf-8")

    if skipped_orphans:
        print(f"  [scan] {split}: skipped {skipped_orphans} wav file(s) without a .normalized.txt")
    return items


class LibriTTSDataset(Dataset[LibriTTSSample]):
    """Serves ``(content [T,768], speaker [D], mel [80,T])`` from real LibriTTS audio.

    Args:
        utterances: output of :func:`scan_split`.
        vocoder: loaded :class:`vocoder.HiFiGANVocoder`; its ``AudioProcessor`` builds the mel
            target, so the spectrogram geometry always matches the vocoder.
        segment_seconds: crop length. ``None`` keeps whole utterances (they then differ in length,
            so batching needs ``batch_size=1``).
        train: random crop + no fixed seed progression in eval mode (eval uses a centre crop).
        content_extractor / speaker_extractor: see ``features.py``. Defaults are the
            no-weights-needed placeholders.
    """

    def __init__(
        self,
        utterances: list[Utterance],
        vocoder,
        segment_seconds: float | None = 2.0,
        train: bool = True,
        seed: int = 1234,
        content_extractor: ContentExtractor | None = None,
        speaker_extractor: SpeakerExtractor | None = None,
        cache_dir: str | Path | None = None,
        content_source: str = "mel-projection",
        hubert_layer: int = 9,
    ) -> None:
        if not utterances:
            raise ValueError("no utterances: check the root/split/limit arguments")
        if segment_seconds is not None and segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive or None")
        if vocoder is None:
            raise ValueError("a loaded vocoder is required: the mel target comes from its AudioProcessor")

        self.utterances = utterances
        self.vocoder = vocoder
        self.segment_seconds = segment_seconds
        self.train = train
        self.seed = seed
        self.epoch = 0
        self.content_extractor = content_extractor or build_content_extractor("mel-projection")
        self.speaker_extractor = speaker_extractor or build_speaker_extractor("mel-stats")
        # when a cache is available the extractors are never called at training time
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.content_source = content_source
        self.hubert_layer = hubert_layer

    def set_epoch(self, epoch: int) -> None:
        """Changes the random crop offsets between epochs."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.utterances)

    def cache_path(self, item: Utterance) -> Path | None:
        """Path of this utterance's cached features, or ``None`` when caching is off/missing."""
        if self.cache_dir is None:
            return None
        from cache import cache_file

        path = cache_file(self.cache_dir, item.split, item, self.content_source, self.hubert_layer)
        return path if path.is_file() else None

    # ------------------------------------------------------------------ helpers
    def crop_bounds(self, n_samples: int, index: int, sr: int) -> tuple[int, int]:
        """Sample range ``[start, end)`` actually used for this item.

        Random for training (changes with ``set_epoch``), centre crop for evaluation, and the whole
        clip when it is shorter than the segment (the caller then zero-pads). Exposed so the trainer
        can report *which part* of the utterance was converted -- the transcript is per-utterance, so
        without this the printed sentence and the audible words disagree.
        """
        if self.segment_seconds is None:
            return 0, n_samples
        n = int(round(self.segment_seconds * sr))
        if n_samples < n:
            return 0, n_samples
        if self.train:
            rng = np.random.default_rng(self.seed + self.epoch * 100003 + index)
            start = int(rng.integers(0, n_samples - n + 1))
        else:
            start = (n_samples - n) // 2
        return start, start + n

    def crop_segment(self, wav: np.ndarray, index: int, sr: int) -> np.ndarray:
        """Random (train) / centre (eval) crop to ``segment_seconds``; short clips are zero-padded.

        The crop length is computed from the **file's** sample rate, not the vocoder's: LibriTTS is
        24 kHz while the vocoder is 22.05 kHz, so using the wrong one shortens every segment.

        Public so evaluation can reproduce the exact same crop when it writes a reference wav.
        """
        start, end = self.crop_bounds(len(wav), index, sr)
        cropped = wav[start:end]
        if self.segment_seconds is None:
            return cropped
        n = int(round(self.segment_seconds * sr))
        if len(cropped) < n:
            cropped = np.pad(cropped, (0, n - len(cropped)), mode="constant")
        return cropped

    def __getitem__(self, index: int) -> LibriTTSSample:
        import soundfile as sf

        item = self.utterances[index]

        cached = self.cache_path(item)
        if cached is not None:
            return self._from_cache(cached, item, index)

        wav, sr = sf.read(item.wav_path, dtype="float32", always_2d=False)
        if wav.ndim > 1:  # mono safety: LibriTTS is mono, but be forgiving
            wav = wav.mean(axis=1)
        wav = np.asarray(wav, dtype=np.float32)
        total_seconds = len(wav) / sr
        start, end = self.crop_bounds(len(wav), index, sr)
        wav = self.crop_segment(wav, index, sr)

        mel = self.vocoder.mel_from_waveform(wav, sample_rate=sr)  # [1, 80, T]
        mel_np = mel[0].detach().cpu().numpy()
        n_frames = mel_np.shape[-1]

        f0_np = f0_from_waveform(wav, sr, hop_length=self.vocoder.hop_length)  # [T', 2]
        if f0_np.shape[0] >= n_frames:
            f0_np = f0_np[:n_frames]
        else:  # pad the tail so content / f0 / mel all share the same frame count
            f0_np = np.pad(f0_np, ((0, n_frames - f0_np.shape[0]), (0, 0)))

        content = self.content_extractor.extract(wav, sr, mel_np, n_frames)  # [T, 768]
        speaker = self.speaker_extractor.extract(wav, sr, mel_np)  # [D]

        if content.shape[0] != n_frames:
            raise RuntimeError(
                f"content extractor '{getattr(self.content_extractor, 'name', '?')}' returned "
                f"{content.shape[0]} frames for {n_frames} mel frames -- the Generator is frame-aligned"
            )

        return LibriTTSSample(
            content=torch.from_numpy(np.ascontiguousarray(content, dtype=np.float32)),
            speaker=torch.from_numpy(np.ascontiguousarray(speaker, dtype=np.float32)),
            mel=torch.from_numpy(np.ascontiguousarray(mel_np, dtype=np.float32)),
            f0=torch.from_numpy(np.ascontiguousarray(f0_np, dtype=np.float32)),
            utt_id=item.utt_id,
            text=item.text,
            crop_start=start / sr,
            crop_end=end / sr,
            duration=total_seconds,
        )


    # ------------------------------------------------------------------ cache reader
    def _from_cache(self, path: Path, item: Utterance, index: int) -> LibriTTSSample:
        """Serve a sample from the pre-extracted feature cache (no HuBERT / XTTS / F0 at runtime).

        Cropping happens in mel-frame space: the cached tensors cover the whole utterance, and the
        same random (train) / centre (eval) crop is applied to mel, content and f0 alike, so the frame
        alignment the Generator relies on is preserved by construction.
        """
        import soundfile as sf

        data = np.load(path)
        mel_full = data["mel"]  # [80, T_full]
        frames_total = mel_full.shape[1]

        info = sf.info(item.wav_path)
        sr = info.samplerate
        seconds = info.frames / sr if sr else 0.0
        start_sample, end_sample = self.crop_bounds(info.frames, index, sr)

        hop = self.vocoder.hop_length
        frame_start = int(round(start_sample * self.vocoder.sample_rate / (sr * hop)))
        if self.segment_seconds is None:
            n_frames = frames_total
        else:
            n_frames = int(round(self.segment_seconds * self.vocoder.sample_rate / hop))
        frame_start = max(0, min(frame_start, max(0, frames_total - 1)))
        frame_end = min(frame_start + n_frames, frames_total)
        take = frame_end - frame_start

        mel = mel_full[:, frame_start:frame_end]
        content = data["content"][frame_start:frame_end].astype(np.float32)
        f0 = data["f0"][frame_start:frame_end]
        speaker = data["speaker"].astype(np.float32)

        # pad to a full segment (only possible for clips shorter than the segment)
        if take < n_frames:
            pad = n_frames - take
            mel = np.pad(mel, ((0, 0), (0, pad)))
            content = np.pad(content, ((0, pad), (0, 0)))
            f0 = np.pad(f0, ((0, pad), (0, 0)))

        return LibriTTSSample(
            content=torch.from_numpy(np.ascontiguousarray(content, dtype=np.float32)),
            speaker=torch.from_numpy(np.ascontiguousarray(speaker, dtype=np.float32)),
            mel=torch.from_numpy(np.ascontiguousarray(mel, dtype=np.float32)),
            f0=torch.from_numpy(np.ascontiguousarray(f0, dtype=np.float32)),
            utt_id=item.utt_id,
            text=item.text,
            crop_start=start_sample / sr if sr else 0.0,
            crop_end=end_sample / sr if sr else seconds,
            duration=seconds,
        )

    # ------------------------------------------------------------------ info
    def describe(self) -> str:
        speakers = len({u.speaker_id for u in self.utterances})
        return (
            f"{len(self.utterances)} utterances, {speakers} speakers "
            f"(content: {getattr(self.content_extractor, 'name', '?')}, "
            f"speaker: {getattr(self.speaker_extractor, 'name', '?')}, "
            f"segment: {'full' if self.segment_seconds is None else f'{self.segment_seconds:g}s'}"
            + (", cache: on" if self.cache_dir is not None else ", cache: off")
            + ")"
        )

    def validate(self) -> None:
        """Load the extractors' weights eagerly so failures happen before training, not mid-epoch."""
        for extractor in (self.content_extractor, self.speaker_extractor):
            check = getattr(extractor, "check_ready", None)
            if callable(check):
                check()


def build_libritts_loader(
    root: str | Path = DEFAULT_ROOT,
    split: str = "train-clean-100",
    vocoder=None,
    limit: int | None = None,
    batch_size: int = 8,
    segment_seconds: float | None = 2.0,
    num_workers: int = 0,
    seed: int = 1234,
    train: bool = True,
    content_source: str = "mel-projection",
    speaker_source: str = "mel-stats",
    content_kwargs: dict | None = None,
    speaker_kwargs: dict | None = None,
    shuffle_seed: int | None = 0,
    min_seconds: float = 0.0,
    cache_dir: str | Path | None = None,
) -> tuple[LibriTTSDataset, DataLoader[LibriTTSSample]]:
    """Scan a split, build the extractors and wrap everything in a DataLoader.

    ``min_seconds`` drops clips shorter than the segment length, which removes the zero padding that
    would otherwise silence-pad ~46% of LibriTTS clips (median 4.3 s against a 4 s segment). It costs
    one slow scan (reading every wav header), then the result is cached in ``datasets/manifests``.
    """
    utterances = scan_split(
        root=root,
        split=split,
        limit=limit,
        min_seconds=min_seconds,
        shuffle_seed=shuffle_seed,
        use_cache=True,
        probe_duration=None if min_seconds > 0 else False,
    )
    if not utterances:
        raise RuntimeError(f"no utterances in '{split}' matched the filters")

    dataset = LibriTTSDataset(
        utterances,
        vocoder=vocoder,
        segment_seconds=segment_seconds,
        train=train,
        seed=seed,
        content_extractor=build_content_extractor(content_source, **(content_kwargs or {})),
        speaker_extractor=build_speaker_extractor(speaker_source, **(speaker_kwargs or {})),
        cache_dir=cache_dir,
        content_source=content_source,
        hubert_layer=int((content_kwargs or {}).get("layer", 9)),
    )

    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)
    loader: DataLoader[LibriTTSSample] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=True,
        generator=shuffle_generator if train else None,
    )
    return dataset, loader


def split_summary(root: str | Path = DEFAULT_ROOT) -> str:
    """Speaker/utterance counts and cross-split speaker overlap (zero-shot sanity)."""
    root = Path(root)
    lines: list[str] = []
    speakers: dict[str, set[str]] = {}
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            lines.append(f"  {split:<16} (missing)")
            continue
        speakers[split] = {d.name for d in split_dir.iterdir() if d.is_dir()}
        n_wav = sum(1 for _ in split_dir.rglob("*.wav"))
        lines.append(f"  {split:<16} utterances={n_wav:>6}  speakers={len(speakers[split]):>4}")
    if "train-clean-100" in speakers and "test-clean" in speakers:
        overlap = speakers["train-clean-100"] & speakers["test-clean"]
        lines.append(f"  train/test speaker overlap: {len(overlap)} (0 => true zero-shot set)")
    return "\n".join(lines)
