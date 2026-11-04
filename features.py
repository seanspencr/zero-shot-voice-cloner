"""Content (linguistic) and speaker (timbre) feature extractors for real audio.

The pipeline needs two side inputs per utterance:

* ``content`` -- ``[T, 768]``: what is said (HuBERT / wav2vec 2.0 hidden states in the real system)
* ``speaker`` -- ``[D]`` (default 256): who says it (a d-vector in the real system)

There are two families here:

**Placeholder extractors** (``mel-projection`` / ``mel-stats``)
    Deterministic functions of the *actual waveform* -- not random noise, so the GAN really does
    learn content -> mel on real LibriTTS audio -- but they are not HuBERT or d-vectors. They exist
    so the real-data pipeline can run without downloading any model weights. Swap them out when the
    weights are in place.

**Real extractors** (``hubert`` / ``resemblyzer``)
    Load pretrained weights from a **local path**. Nothing is downloaded here: if the weights are
    missing they raise :class:`MissingWeightsError` with the download URL and expected location.

Both families keep the frame-count contract: ``content`` has exactly as many frames as the mel
spectrogram (``T``), because the Generator is frame-aligned with its target.
"""

from __future__ import annotations

import glob
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np

__all__ = [
    "CONTENT_DIM",
    "SPEAKER_DIM",
    "HUBERT_SR",
    "XTTS_EMBED_DIM",
    "DEFAULT_HUBERT_DIR",
    "HUBERT_FILES",
    "MissingWeightsError",
    "MelProjectionContentExtractor",
    "HubertContentExtractor",
    "MelStatsSpeakerExtractor",
    "XTTSSpeakerExtractor",
    "build_content_extractor",
    "build_speaker_extractor",
    "CONTENT_SOURCES",
    "SPEAKER_SOURCES",
    "xtts_help",
]

CONTENT_DIM = 768  # HuBERT base / wav2vec 2.0 hidden size
SPEAKER_DIM = 256  # d-vector size
HUBERT_SR = 16000  # HuBERT is trained on 16 kHz audio (LibriTTS ships 24 kHz)

DEFAULT_HUBERT_DIR = Path(__file__).resolve().parent / "pretrained" / "hubert-base-ls960"

#: Where to get each HuBERT file, and the exact filenames `from_pretrained` expects.
HUBERT_FILES: dict[str, str] = {
    "config.json": "https://huggingface.co/facebook/hubert-base-ls960/resolve/main/config.json",
    "preprocessor_config.json": (
        "https://huggingface.co/facebook/hubert-base-ls960/resolve/main/preprocessor_config.json"
    ),
    "pytorch_model.bin": "https://huggingface.co/facebook/hubert-base-ls960/resolve/main/pytorch_model.bin",
}

CONTENT_SOURCES = ("mel-projection", "hubert")
SPEAKER_SOURCES = ("mel-stats", "xtts")

#: XTTS v2's speaker encoder (ResNet/ASP + SE blocks) sits inside the TTS checkpoint.
XTTS_EMBED_DIM = 512
XTTS_SR = 16000
XTTS_CHECKPOINT_GLOB = "~/.cache/huggingface/hub/models--tts-hub--XTTS-v2/snapshots/*/model.pth"
XTTS_WEIGHT_PREFIX = "hifigan_decoder.speaker_encoder."
#: The exact construction XTTS uses (TTS/tts/layers/xtts/hifigan_decoder.py). The mel settings must
#: match or the embedding is meaningless.
XTTS_SPEAKER_AUDIO_CONFIG = {
    "fft_size": 512,
    "win_length": 400,
    "hop_length": 160,
    "sample_rate": XTTS_SR,
    "preemphasis": 0.97,
    "num_mels": 64,
}


class MissingWeightsError(RuntimeError):
    """Raised when a real extractor's weights are not installed locally."""


def _resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav.astype(np.float32, copy=False)
    import librosa

    return librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=target_sr)


def _standardize(x: np.ndarray) -> np.ndarray:
    """Global z-score.

    Raw log-mel values sit around -9, so projecting them directly drives ``tanh`` into saturation,
    where every utterance collapses onto the same ±1 pattern and the features carry no information.
    Centring and scaling first keeps the projection in its responsive range.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    return (x - x.mean()) / (x.std() + 1e-5)


class ContentExtractor(Protocol):
    """``wav [N]`` + mel ``[80, T]`` -> content features ``[T, 768]``."""

    name: str

    def extract(self, wav: np.ndarray, sr: int, mel: np.ndarray, n_frames: int) -> np.ndarray: ...


class SpeakerExtractor(Protocol):
    """``wav [N]`` + mel ``[80, T]`` -> one unit-length speaker embedding ``[D]``."""

    name: str

    def extract(self, wav: np.ndarray, sr: int, mel: np.ndarray) -> np.ndarray: ...


# --------------------------------------------------------------------------------------
# placeholder content extractors (no weights needed, but derived from real audio)
# --------------------------------------------------------------------------------------
class MelProjectionContentExtractor:
    """Placeholder for HuBERT: fixed projection of the *real* log-mel + its temporal context.

    Per frame the input is ``[mel_t (80), local mean over ±context (80), delta (80)]`` = 240 dims,
    projected to 768 by a seeded matrix and squashed with ``tanh``. Deterministic, cheap, and it
    carries genuine acoustic content -- enough for the Generator to learn a real mapping -- but it
    is a stand-in for HuBERT, not a substitute.

    Replace with: :class:`HubertContentExtractor`.
    """

    name = "mel-projection"

    def __init__(self, out_dim: int = CONTENT_DIM, seed: int = 1234, context: int = 2) -> None:
        self.out_dim = out_dim
        self.context = context
        rng = np.random.default_rng(seed)
        in_dim = 3 * 80
        self._proj = rng.standard_normal((in_dim, out_dim)).astype(np.float32) / math.sqrt(in_dim)

    def extract(self, wav: np.ndarray, sr: int, mel: np.ndarray, n_frames: int) -> np.ndarray:
        m = np.ascontiguousarray(mel[:, :n_frames], dtype=np.float32)  # [80, T]
        if m.shape[1] == 0:
            return np.zeros((0, self.out_dim), dtype=np.float32)
        kernel = np.ones(2 * self.context + 1, dtype=np.float32) / (2 * self.context + 1)
        local = np.stack(
            [np.convolve(row, kernel, mode="same") for row in m]  # smoothing along time
        )
        delta = np.gradient(m, axis=1)
        feats = np.concatenate([m, local, delta], axis=0).T  # [T, 240]
        return np.tanh(_standardize(feats) @ self._proj).astype(np.float32)


class HubertContentExtractor:
    """Real content features: HuBERT hidden states, resampled to the mel frame rate.

    Loads ``transformers.HubertModel`` from a **local directory** (default
    ``pretrained/hubert-base-ls960``) containing ``config.json`` + ``pytorch_model.bin``.
    Nothing is downloaded; call :func:`hubert_help` (or read the error) for the URLs.

    Args:
        model_dir: local folder with the HF files.
        layer: which hidden state to take. Layer 9 (0 = input embeddings) is the usual
            choice for voice conversion; ``--hubert-layer`` exposes it.
        device: torch device string.
    """

    name = "hubert"

    def __init__(self, model_dir: str | Path = DEFAULT_HUBERT_DIR, layer: int = 9, device: str = "cpu") -> None:
        self.model_dir = Path(model_dir)
        self.layer = layer
        self.device = device
        self._model: Any = None
        self._torch: Any = None

    # -- weights ---------------------------------------------------------------------
    def _check_weights(self) -> None:
        if not self.model_dir.is_dir():
            raise MissingWeightsError(hubert_help(self.model_dir))
        missing = [name for name in ("config.json",) if not (self.model_dir / name).is_file()]
        has_weights = any(
            (self.model_dir / name).is_file()
            for name in ("pytorch_model.bin", "model.safetensors", "pytorch_model.bin.index.json")
        )
        if missing or not has_weights:
            raise MissingWeightsError(hubert_help(self.model_dir))

    def _load(self) -> None:
        if self._model is not None:
            return
        self._check_weights()
        import torch
        from transformers import HubertModel

        model = HubertModel.from_pretrained(str(self.model_dir))
        model.eval().to(self.device)
        self._model = model
        self._torch = torch

    def check_ready(self) -> None:
        """Validate (and load) eagerly so a missing checkpoint fails before training starts."""
        self._load()

    # -- inference -------------------------------------------------------------------
    def extract(self, wav: np.ndarray, sr: int, mel: np.ndarray, n_frames: int) -> np.ndarray:
        self._load()
        torch = self._torch
        wav16 = _resample(wav, sr, HUBERT_SR)
        with torch.no_grad():
            inputs = torch.from_numpy(wav16).float().unsqueeze(0).to(self.device)
            out = self._model(inputs, output_hidden_states=True)
            hidden = out.hidden_states[self.layer]  # [1, T16, 768]
        if n_frames <= 0:
            return np.zeros((0, hidden.shape[-1]), dtype=np.float32)
        # HuBERT emits ~50 frames/s, the mel ~86 frames/s: interpolate content onto mel frames.
        resized = torch.nn.functional.interpolate(
            hidden.transpose(1, 2), size=n_frames, mode="linear", align_corners=False
        )
        return resized.transpose(1, 2)[0].cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------------------
# placeholder speaker extractors (no weights needed, but derived from real audio)
# --------------------------------------------------------------------------------------
class MelStatsSpeakerExtractor:
    """Placeholder for a d-vector: long-term log-mel statistics, a classic timbre descriptor.

    Uses the mean, standard deviation and delta statistics of the log-mel over the whole utterance,
    each block mean-removed (so the shared spectral tilt does not dominate), projected to
    ``out_dim`` with a seeded matrix and L2-normalized.

    **Measured on LibriTTS**: different speakers land at cosine ≈ 0.91 and the same speaker's two
    utterances at ≈ 0.95, i.e. a margin of ~0.04 and per-coordinate std ~0.02. That is a *weak* but
    non-degenerate conditioning signal (AdaIN amplifies it with learned weights), nothing like a real
    speaker encoder.

    Replace with: :class:`XTTSSpeakerExtractor`.
    """

    name = "mel-stats"

    def __init__(self, out_dim: int = SPEAKER_DIM, seed: int = 4321) -> None:
        self.out_dim = out_dim
        rng = np.random.default_rng(seed)
        in_dim = 3 * 80
        self._proj = rng.standard_normal((in_dim, out_dim)).astype(np.float32) / math.sqrt(in_dim)

    def extract(self, wav: np.ndarray, sr: int, mel: np.ndarray) -> np.ndarray:
        m = np.asarray(mel, dtype=np.float32)
        if m.shape[1] == 0:
            return np.zeros(self.out_dim, dtype=np.float32)
        # Spectral *shape* rather than absolute spectra: all speech shares a 1/f tilt, so leaving it
        # in makes every utterance project to almost the same vector. Removing each block's mean keeps
        # the speaker-specific envelope/dynamics, which is what AdaIN needs to tell voices apart.
        envelope = m.mean(axis=1)
        envelope = envelope - envelope.mean()
        dynamics = m.std(axis=1)
        dynamics = dynamics - dynamics.mean()
        delta_std = np.gradient(m, axis=1).std(axis=1)
        delta_std = delta_std - delta_std.mean()
        stats = np.concatenate([envelope, dynamics, delta_std])
        emb = np.tanh(_standardize(stats) @ self._proj).astype(np.float32)
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 0 else emb


class XTTSSpeakerExtractor:
    """Real speaker embedding (512-d) from **XTTS v2's own speaker encoder**.

    That encoder is the ResNet/ASP network inside ``HifiDecoder``
    (``TTS.encoder.models.resnet.ResNetSpeakerEncoder``), and its weights are already on this machine
    inside the XTTS v2 checkpoint (``hifigan_decoder.speaker_encoder.*``, ~32 MB of the 1.78 GB file).

    Why prefer it over a generic d-vector: it was trained **jointly with the XTTS generator**, i.e. its
    vector is exactly the kind of conditioning a speech generator consumes, and it is the same voice
    description your TTS front-end uses — so the VC stage and the TTS stage share one voice space.

    Loading is memory-friendly: the checkpoint is a zip archive, so ``torch.load(..., mmap=True)`` +
    key filtering pulls in only the ~32 MB we need instead of all 1.78 GB.

    Args:
        checkpoint: explicit path to ``model.pth``; by default the Hugging Face cache of the installed
            XTTS v2 model is searched.
        device: torch device string.
    """

    name = "xtts"
    out_dim = XTTS_EMBED_DIM

    def __init__(self, checkpoint: str | Path | None = None, device: str = "cpu") -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.device = device
        self._encoder: Any = None
        self._torch: Any = None

    # -- weights ---------------------------------------------------------------------
    def resolve_checkpoint(self) -> Path:
        if self.checkpoint is not None:
            if not self.checkpoint.is_file():
                raise MissingWeightsError(xtts_help(self.checkpoint))
            return self.checkpoint
        import os

        pattern = os.path.expanduser(XTTS_CHECKPOINT_GLOB)
        candidates = sorted(Path(p) for p in glob.glob(pattern))
        if not candidates:
            raise MissingWeightsError(xtts_help())
        return candidates[-1]

    def _load(self) -> None:
        if self._encoder is not None:
            return
        import torch
        from TTS.encoder.models.resnet import ResNetSpeakerEncoder

        path = self.resolve_checkpoint()
        try:
            payload = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
        except Exception:  # older/odd checkpoints may carry non-tensor objects
            payload = torch.load(str(path), map_location="cpu", weights_only=False)
        state = payload.get("model", payload) if isinstance(payload, dict) else payload

        subset = {
            key[len(XTTS_WEIGHT_PREFIX):]: value
            for key, value in state.items()
            if key.startswith(XTTS_WEIGHT_PREFIX)
        }
        if not subset:
            raise MissingWeightsError(
                f"'{path}' has no '{XTTS_WEIGHT_PREFIX}*' weights -- is this really the XTTS v2 checkpoint?"
            )

        encoder = ResNetSpeakerEncoder(
            input_dim=64,
            proj_dim=XTTS_EMBED_DIM,
            log_input=True,
            use_torch_spec=True,
            audio_config=dict(XTTS_SPEAKER_AUDIO_CONFIG),
        )
        encoder.load_state_dict(subset)
        encoder.eval().to(self.device)
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        self._encoder = encoder
        self._torch = torch
        self.checkpoint = path

    def check_ready(self) -> None:
        """Load eagerly so a missing/broken checkpoint fails before training starts."""
        self._load()

    # -- inference -------------------------------------------------------------------
    def extract(self, wav: np.ndarray, sr: int, mel: np.ndarray) -> np.ndarray:
        self._load()
        torch = self._torch
        wav16 = _resample(wav, sr, XTTS_SR)  # XTTS's speaker encoder works on 16 kHz audio
        with torch.no_grad():
            audio = torch.from_numpy(wav16).float().unsqueeze(0).to(self.device)
            emb = self._encoder(audio, l2_norm=True)  # already L2-normalized
        return emb[0].detach().cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------------------
# help text + factories
# --------------------------------------------------------------------------------------
def hubert_help(model_dir: str | Path = DEFAULT_HUBERT_DIR) -> str:
    lines = [
        f"HuBERT weights not found in '{model_dir}'.",
        "  Download these three files (the .bin is ~360 MB):",
    ]
    lines += [f"    {url}" for url in HUBERT_FILES.values()]
    lines += [
        f"  and place them in '{model_dir}' (keep the filenames as shown).",
        "  Nothing was downloaded automatically.",
        "  Without them, run with --content-source mel-projection (real audio, placeholder features).",
    ]
    return "\n".join(lines)


def xtts_help(checkpoint: str | Path | None = None) -> str:
    target = checkpoint or "~/.cache/huggingface/hub/models--tts-hub--XTTS-v2/snapshots/*/model.pth"
    return "\n".join(
        [
            "XTTS v2's speaker encoder was not found.",
            f"  Expected checkpoint: {target}",
            "  It is the same 1.78 GB model.pth that Coqui TTS uses; if XTTS is installed, it is already",
            "  there (~/.cache/huggingface/hub or your TTS models dir). Only ~32 MB of it is loaded.",
            "  Point at a specific file with --xtts-checkpoint <.../model.pth>.",
            "  Nothing was downloaded automatically.",
            "  Without it, run with --speaker-source mel-stats (real audio, placeholder features).",
        ]
    )


def build_content_extractor(kind: str, **kwargs: Any) -> ContentExtractor:
    if kind == "mel-projection":
        return MelProjectionContentExtractor(out_dim=kwargs.get("out_dim", CONTENT_DIM))
    if kind == "hubert":
        return HubertContentExtractor(
            model_dir=kwargs.get("model_dir", DEFAULT_HUBERT_DIR),
            layer=kwargs.get("layer", 9),
            device=kwargs.get("device", "cpu"),
        )
    raise ValueError(f"unknown content source {kind!r}; choose from {CONTENT_SOURCES}")


def build_speaker_extractor(kind: str, **kwargs: Any) -> SpeakerExtractor:
    if kind == "mel-stats":
        return MelStatsSpeakerExtractor(out_dim=kwargs.get("out_dim", SPEAKER_DIM))
    if kind == "xtts":
        return XTTSSpeakerExtractor(
            checkpoint=kwargs.get("checkpoint"), device=kwargs.get("device", "cpu")
        )
    raise ValueError(f"unknown speaker source {kind!r}; choose from {SPEAKER_SOURCES}")
