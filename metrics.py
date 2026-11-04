"""Objective metrics for converted speech: MCD and SECS.

**MCD — Mel-Cepstral Distortion** (spectral distance, lower is better, in dB)
    Classic formulation (Kubichek): DCT the log-mel spectrogram, keep cepstra ``c1..c13`` (drop ``c0``,
    which is loudness), and take::

        MCD = (10 / ln 10) * sqrt(2 * mean_t sum_{d=1..13} (c_d^gen - c_d^ref)^2)

    Needs only ``numpy`` + ``scipy`` — no model weights.

**SECS — Speaker Embedding Cosine Similarity** (timbre similarity, higher is better)
    Cosine between speaker embeddings of the converted audio and the *target speaker's enrollment*
    audio; also computed against the content source so you can see whether the output is drifting
    back toward the wrong voice. Uses whichever speaker extractor is configured -- by default XTTS v2's own
    speaker encoder (`features.XTTSSpeakerExtractor`). If it is absent the metric reports ``None`` with a note instead of failing the run.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "MCD_CEPSTRA",
    "mel_cepstral_distortion",
    "speaker_cosine_similarity",
    "EvaluationReport",
]

#: Number of cepstral coefficients compared (c1..cN). 13 is the common choice for 80-bin mels.
MCD_CEPSTRA = 13


def _mel_to_cepstra(mel: np.ndarray, n_cepstra: int = MCD_CEPSTRA + 1) -> np.ndarray:
    """``[n_mels, T]`` log-mel -> ``[n_cepstra, T]`` cepstra (c0 dropped)."""
    from scipy.fftpack import dct

    m = np.asarray(mel, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"expected mel of shape [n_mels, T], got {m.shape}")
    # type-2 DCT with ortho norm == the usual MFCC transform; keep c0..cN then drop c0
    cepstra = dct(m, type=2, axis=0, norm="ortho")
    return cepstra[1:n_cepstra]


def mel_cepstral_distortion(
    mel_gen: np.ndarray,
    mel_ref: np.ndarray,
    n_cepstra: int = MCD_CEPSTRA,
) -> float:
    """Mean Mel-Cepstral Distortion (dB) between two ``[n_mels, T]`` log-mels.

    Frames are aligned 1:1; if the two have different lengths they are cut to the shorter one, which
    is what you want when comparing a converted utterance with its reference.
    """
    gen = np.asarray(mel_gen, dtype=np.float32)
    ref = np.asarray(mel_ref, dtype=np.float32)
    if gen.shape[0] != ref.shape[0]:
        raise ValueError(f"mel bin mismatch: {gen.shape[0]} vs {ref.shape[0]}")
    frames = min(gen.shape[1], ref.shape[1])
    if frames == 0:
        return float("nan")

    c_gen = _mel_to_cepstra(gen[:, :frames], n_cepstra + 1)
    c_ref = _mel_to_cepstra(ref[:, :frames], n_cepstra + 1)
    diff = c_gen - c_ref
    mcd = (10.0 / math.log(10.0)) * math.sqrt(2.0 * float(np.mean(np.sum(diff**2, axis=0))))
    return float(mcd)


def speaker_cosine_similarity(
    embedder: Any,
    audio_a: np.ndarray,
    sr_a: int,
    audio_b: np.ndarray,
    sr_b: int,
) -> float:
    """Cosine similarity of two waveforms' speaker embeddings (SECS)."""
    if not hasattr(embedder, "extract"):
        raise TypeError("speaker_cosine_similarity needs an extractor with .extract(wav, sr, mel)")
    # the mel argument is unused by the extractor; pass a dummy so the interface stays uniform
    dummy = np.zeros((80, 1), dtype=np.float32)
    emb_a = embedder.extract(np.asarray(audio_a, dtype=np.float32), sr_a, dummy)
    emb_b = embedder.extract(np.asarray(audio_b, dtype=np.float32), sr_b, dummy)
    return float(np.clip(emb_a @ emb_b, -1.0, 1.0))


@dataclass
class EvaluationReport:
    """Aggregated conversion metrics over a set of utterances."""

    n_items: int = 0
    mcd_mean: float = float("nan")
    mcd_std: float = float("nan")
    mel_l1_mean: float = float("nan")
    secs_target_mean: float | None = None  # vs. the enrollment audio of the target speaker
    secs_target_std: float | None = None
    secs_source_mean: float | None = None  # vs. the content source (leakage indicator)
    secs_note: str = ""
    enrollment_fallbacks: int = 0  # items where no other clip of that speaker existed
    per_item: list[dict[str, Any]] = field(default_factory=list)
    n_cepstra: int = MCD_CEPSTRA

    def summary(self) -> str:
        lines = [
            f"  utterances           : {self.n_items}",
            f"  MCD (dB, lower better): {self.mcd_mean:.3f} +/- {self.mcd_std:.3f}   (c1..c{self.n_cepstra})",
            f"  mel L1 (lower better): {self.mel_l1_mean:.4f}",
        ]
        if self.secs_target_mean is not None:
            lines.append(
                f"  SECS vs target (higher better): {self.secs_target_mean:.3f} +/- {self.secs_target_std:.3f}"
            )
        if self.secs_source_mean is not None:
            lines.append(f"  SECS vs content source        : {self.secs_source_mean:.3f}  (lower = less source leakage)")
        if self.secs_note:
            lines.append(f"  note: {self.secs_note}")
        if self.enrollment_fallbacks:
            lines.append(
                f"  note: {self.enrollment_fallbacks}/{self.n_items} items had no second clip of the target "
                f"speaker, so SECS used the item's own audio as the enrolment (identity shortcut)"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("per_item", None)
        payload["per_item"] = self.per_item
        return payload
