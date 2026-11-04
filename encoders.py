"""Mock front-end modules for the zero-shot voice conversion pipeline.

These are placeholders with the same *interface and output shapes* as the real models that
will be plugged in later. Nothing here is trained to do anything useful -- the point is that
``MockContentEncoder`` / ``MockSpeakerEncoder`` can be swapped for HuBERT / a d-vector encoder
without touching the rest of the codebase.

| Mock                  | Real replacement                       | Contract                              |
|-----------------------|----------------------------------------|---------------------------------------|
| ``MockContentEncoder``| HuBERT / wav2vec 2.0 (layer 9)         | ``[B, T, 768] -> [B, T, 768]``        |
| ``MockSpeakerEncoder``| d-vector / ECAPA-TDNN from enrollment  | ``[B, ...] -> [B, 256]``, unit length  |

The vocoder is **not** mocked: ``vocoder.py`` loads a pretrained Coqui HiFi-GAN checkpoint
(``mel [B, 80, T] -> waveform [B, 1, T * hop_length]``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import CONTENT_DIM, SPEAKER_DIM

__all__ = [
    "MockContentEncoder",
    "MockSpeakerEncoder",
]


class MockContentEncoder(nn.Module):
    """Stand-in for HuBERT / wav2vec 2.0 feature extraction.

    Time resolution is preserved (same-padded ``Conv1d``, stride 1), which is the property the
    Generator relies on: content frames and mel frames stay aligned 1:1 within this mock setup.

    Shapes:
        input  ``[B, T, in_dim]``  (or ``[B, in_dim, T]``)
        output ``[B, T, out_dim]``

    TODO(real-data): replace with HuBERT layer-9 features. HuBERT emits one frame every ~20 ms,
    while a mel spectrogram at ``hop_length=256`` / 22.05 kHz emits one frame every ~11.6 ms, so
    the real pipeline must resample or re-stride content frames (see ``README.md``).
    """

    def __init__(self, in_dim: int = CONTENT_DIM, out_dim: int = CONTENT_DIM, hidden_dim: int = CONTENT_DIM) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm = nn.LayerNorm(in_dim)
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, out_dim, kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"MockContentEncoder expects a 3D tensor, got shape {tuple(x.shape)}")
        # Accept both [B, T, D] and [B, D, T]; length sits on dim 1 for Conv1d.
        if x.shape[-1] != self.in_dim:
            if x.shape[1] != self.in_dim:
                raise ValueError(
                    f"MockContentEncoder expects {self.in_dim}-dim features on the last axis, "
                    f"got shape {tuple(x.shape)}"
                )
            x = x.transpose(1, 2)  # [B, T, D]
        h = self.norm(x).transpose(1, 2)  # [B, D, T]
        h = F.gelu(self.conv1(h))
        h = self.conv2(h)
        return h.transpose(1, 2)  # [B, T, out_dim]


class MockSpeakerEncoder(nn.Module):
    """Stand-in for a speaker verification network producing a d-vector.

    Shapes:
        input  ``[B, T, in_dim]`` (time-pooled), ``[B, in_dim]``, or ``[B, 1, N]`` waveform
        output ``[B, out_dim]``, L2-normalized

    TODO(real-data): replace with a pretrained speaker encoder (e.g. ECAPA-TDNN) that consumes a
    single enrollment utterance of the target user. Keep the L2 normalization -- the Generator's
    AdaIN layer is trained against whichever convention is used here.
    """

    def __init__(self, in_dim: int = SPEAKER_DIM, out_dim: int = SPEAKER_DIM) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)  # [B, D] -- cheap stand-in for temporal statistics pooling
        elif x.dim() == 2:
            pass
        else:
            raise ValueError(f"MockSpeakerEncoder expects a 2D or 3D tensor, got shape {tuple(x.shape)}")

        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"MockSpeakerEncoder expects {self.in_dim}-dim input, got shape {tuple(x.shape)}"
            )

        z = self.norm(self.proj(x))
        return F.normalize(z, dim=-1)

