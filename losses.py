"""Losses that fight over-smoothing, in the mel domain.

The plan (`improvement-plan.md` 1.1) called for a multi-resolution STFT loss. That loss needs
**waveforms**, and this converter only produces mel-spectrograms — running the vocoder inside the
training loop is far too expensive on CPU. So the same idea is applied in the mel domain, where it is
cheap and fully differentiable:

``multi_scale_mel_loss``
    L1 at several temporal resolutions (average-pooling by 1×, 2×, 4×). The coarse scales pin the
    overall envelope, the fine scale pins fast transitions (consonant bursts, onsets). A single-scale
    L1 has a trivial minimiser — the conditional mean, i.e. a blurred mel — which is exactly the
    "buzzy" artefact that motivated this loss.

``global_variance_loss``
    L1 between the **per-band temporal standard deviation** of prediction and target. Plain L1/L2 is
    happy to shrink a band's dynamics to its mean; the classic remedy in statistical voice conversion
    is a global-variance (GV) criterion (Hwang et al., Interspeech 2013). This term makes "flat band"
    expensive, so the generator cannot buy a lower L1 by smearing detail away.

Both are cheap: a few mean/std reductions over `[B, 80, T]`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["multi_scale_mel_loss", "global_variance_loss", "SpectralDetailLoss"]


def multi_scale_mel_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    scales: tuple[int, ...] = (1, 2, 4),
    weights: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """L1 mel loss evaluated at several temporal resolutions.

    Args:
        pred, target: ``[B, n_mels, T]`` mel spectrograms.
        scales: pooling factors; 1 = full resolution. ``T`` need not be divisible — sizes match because
            both tensors are pooled identically.
        weights: per-scale weights (default: uniform).
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    total = pred.new_zeros(())
    weight_sum = 0.0
    for index, scale in enumerate(scales):
        w = 1.0 if weights is None else weights[index]
        if scale == 1:
            p, t = pred, target
        else:
            p = F.avg_pool1d(pred, kernel_size=scale, stride=scale, ceil_mode=True)
            t = F.avg_pool1d(target, kernel_size=scale, stride=scale, ceil_mode=True)
        total = total + w * F.l1_loss(p, t)
        weight_sum += w
    return total / max(weight_sum, 1e-8)


def global_variance_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """L1 distance between per-band temporal standard deviations (anti-flattening term)."""
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    std_pred = pred.std(dim=2, unbiased=False)
    std_target = target.std(dim=2, unbiased=False)
    return F.l1_loss(std_pred, std_target + eps)


class SpectralDetailLoss(torch.nn.Module):
    """``multi_scale_mel_loss + global_variance_loss`` as one module, for logging separately.

    Returns ``(total, multi_scale_value, global_variance_value)`` so the trainer can print both.
    """

    def __init__(self, scales: tuple[int, ...] = (1, 2, 4), gv_weight: float = 1.0) -> None:
        super().__init__()
        self.scales = scales
        self.gv_weight = gv_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ms = multi_scale_mel_loss(pred, target, self.scales)
        gv = global_variance_loss(pred, target)
        return ms + self.gv_weight * gv, ms, gv
