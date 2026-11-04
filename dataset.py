"""Mock dataset + dataloader for the zero-shot voice conversion GAN.

The real project will read VCTK audio and extract:
  * content features  -- HuBERT / wav2vec 2.0 hidden states      [time_steps, 768]
  * speaker embedding -- d-vector of the target user             [256]
  * target mel        -- 80-bin log Mel-spectrogram of the audio  [80, time_steps]

Until that data exists, this module fabricates tensors with *exactly* those shapes so
the model + training loop can be developed and shape-tested end to end.

Swap plan: replace :class:`MockVoiceConversionDataset` with a VCTK loader that returns the
same dictionary keys and shapes (see ``README.md``).
"""

from __future__ import annotations

from typing import TypedDict

import torch
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "VCBatch",
    "VCSample",
    "MockVoiceConversionDataset",
    "build_dataloader",
    "CONTENT_DIM",
    "SPEAKER_DIM",
    "MEL_BINS",
]


CONTENT_DIM = 768  # HuBERT / wav2vec 2.0 hidden size
SPEAKER_DIM = 256  # d-vector size
MEL_BINS = 80  # Mel filterbank bins


class VCSample(TypedDict):
    """One un-batched mock training example."""

    content: torch.Tensor  # [time_steps, 768]  float32
    speaker: torch.Tensor  # [256]              float32, L2-normalized
    mel: torch.Tensor  # [80, time_steps] float32


class VCBatch(TypedDict):
    """One batch, as produced by ``default_collate``."""

    content: torch.Tensor  # [batch_size, time_steps, 768]
    speaker: torch.Tensor  # [batch_size, 256]
    mel: torch.Tensor  # [batch_size, 80, time_steps]


class MockVoiceConversionDataset(Dataset[VCSample]):
    """Generates random tensors that stand in for extracted speech features.

    Sampling is deterministic *and* epoch-dependent: item ``i`` in epoch ``e`` is drawn from a
    generator seeded with ``seed + e * len(dataset) + i``. Consequences:

    * two identical runs produce identical batches (reproducible experiments);
    * consecutive epochs see different data (the model cannot memorise one fixed set);
    * every item is computed from its own seed, so ``num_workers > 0`` is safe.

    Call :meth:`set_epoch` from the training loop to advance the epoch counter.
    """

    def __init__(
        self,
        size: int = 256,
        time_steps: int = 128,
        content_dim: int = CONTENT_DIM,
        speaker_dim: int = SPEAKER_DIM,
        mel_bins: int = MEL_BINS,
        seed: int = 0,
    ) -> None:
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        if time_steps <= 0:
            raise ValueError(f"time_steps must be positive, got {time_steps}")

        self.size = size
        self.time_steps = time_steps
        self.content_dim = content_dim
        self.speaker_dim = speaker_dim
        self.mel_bins = mel_bins
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Advance the sampling epoch (changes the random data seen by the loader)."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> VCSample:
        if not 0 <= index < self.size:
            raise IndexError(f"index {index} out of range for dataset of size {self.size}")

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * self.size + index)

        content = torch.randn(self.time_steps, self.content_dim, generator=generator)
        speaker = torch.randn(self.speaker_dim, generator=generator)
        # d-vectors are conventionally unit length; AdaIN learns the scale anyway.
        speaker = torch.nn.functional.normalize(speaker, dim=0)
        mel = torch.randn(self.mel_bins, self.time_steps, generator=generator)

        return VCSample(content=content, speaker=speaker, mel=mel)


def build_dataloader(
    batch_size: int = 8,
    dataset_size: int = 256,
    time_steps: int = 128,
    content_dim: int = CONTENT_DIM,
    speaker_dim: int = SPEAKER_DIM,
    mel_bins: int = MEL_BINS,
    num_workers: int = 0,
    seed: int = 0,
    drop_last: bool = True,
) -> tuple[MockVoiceConversionDataset, DataLoader[VCSample]]:
    """Create the mock dataset and its dataloader.

    ``drop_last=True`` keeps every batch at exactly ``batch_size`` so the log lines and the
    PatchGAN output shape stay identical from step to step.
    """
    dataset = MockVoiceConversionDataset(
        size=dataset_size,
        time_steps=time_steps,
        content_dim=content_dim,
        speaker_dim=speaker_dim,
        mel_bins=mel_bins,
        seed=seed,
    )
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)
    loader: DataLoader[VCSample] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=shuffle_generator,
    )
    return dataset, loader
