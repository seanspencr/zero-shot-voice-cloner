"""Real neural vocoder: mel-spectrogram ``[B, 80, T]`` -> waveform ``[B, 1, N]``.

This replaces ``MockVocoder``. It wraps a **pretrained Coqui HiFi-GAN** checkpoint and runs it
through the same `TTS` package the project already uses for the text-to-speech front-end, so the
mel preprocessing and the vocoder can never drift apart: the vocabulary of `audio_config`
(sample rate, ``n_fft``, ``hop_length``, ``n_mels``, mel scaling) comes from the vocoder's own
``config.json``.

**Nothing is downloaded automatically.** ``--model-dir`` must point at a local directory holding
the vocoder ``config.json`` + checkpoint. If it is missing, :func:`missing_checkpoint_help` prints
the exact URL and the expected folder layout (see ``README.md`` § "Vocoder").

Default model: ``vocoder_models/en/ljspeech/hifigan_v2`` -- 22.05 kHz, 80 mel bins,
``hop_length=256``. That is the configuration the project was specified against
(``requirements.md``: "assuming hop_length = 256"), and 22.05 kHz is the standard rate for
voice-conversion research.

Two properties of that checkpoint are worth knowing (read from its ``config.json``, which is the
authority -- never hard-code these):

* ``signal_norm=false``, ``log_func=np.log``, ``spec_gain=1.0``, ``mel_fmax=8000`` ⇒ the mel input
  is a **raw natural-log mel spectrogram, not range-normalized and not clipped to ±4**. Use
  :meth:`HiFiGANVocoder.mel_from_waveform` so your training targets match it exactly.
* ``upsample_initial_channel=128`` (a small/narrow HiFi-GAN, ~1M parameters, 3.8 MB checkpoint) and
  the archive's own description reads "universal hifigan trained on LibriTTS with no spectrogram
  normalization" -- i.e. multi-speaker, which is what zero-shot VC wants.

CLI (self-check, no training required)::

    # inspect the checkpoint and round-trip a real audio file through mel -> vocoder -> wav
    python vocoder.py --check jokowi.mp3 --out probe.wav

    # just print the audio config of a checkpoint
    python vocoder.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "DEFAULT_MODEL_DIR",
    "MODEL_URLS",
    "HiFiGANVocoder",
    "resolve_model_files",
    "vocoder_ready",
    "missing_checkpoint_help",
    "load_vocoder",
]

DEFAULT_MODEL_DIR = "vocoder_models/en/ljspeech/hifigan_v2"

#: Where to download each supported checkpoint (Coqui release assets). Nothing here is fetched
#: automatically -- the URLs exist so you can put the files in place yourself.
MODEL_URLS: dict[str, str] = {
    "vocoder_models/en/ljspeech/hifigan_v2": (
        "https://github.com/coqui-ai/TTS/releases/download/v0.6.1_models/"
        "vocoder_models--en--ljspeech--hifigan_v2.zip"
    ),
    "vocoder_models/en/vctk/hifigan_v2": (
        "https://github.com/coqui-ai/TTS/releases/download/v0.6.1_models/"
        "vocoder_models--en--vctk--hifigan_v2.zip"
    ),
    "vocoder_models/en/ljspeech/multiband-melgan": (
        "https://github.com/coqui-ai/TTS/releases/download/v0.6.1_models/"
        "vocoder_models--en--ljspeech--multiband-melgan.zip"
    ),
    "vocoder_models/universal/libri-tts/fullband-melgan": (
        "https://github.com/coqui-ai/TTS/releases/download/v0.6.1_models/"
        "vocoder_models--universal--libri-tts--fullband-melgan.zip"
    ),
}

_CHECKPOINT_NAMES = ("model_file.pth", "model_file.pth.tar", "model.pth", "checkpoint.pth")


def resolve_model_files(model_dir: str | Path) -> tuple[Path, Path]:
    """Locate ``(checkpoint, config)`` inside ``model_dir``.

    Accepts the extracted model folder (the normal case) and is forgiving about one level of
    nesting, mirroring Coqui's own ``ModelManager`` discovery rules.

    Raises:
        FileNotFoundError: with a message that includes the download URL and expected layout.
    """
    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(missing_checkpoint_help(model_dir))

    if path.is_file():  # a checkpoint was passed directly; config must sit next to it
        config_candidates = [path.parent / "config.json"]
        checkpoint_candidates = [path]
    else:
        named = [path / name for name in _CHECKPOINT_NAMES if (path / name).is_file()]
        if named:
            checkpoint_candidates = named
        else:
            checkpoint_candidates = sorted(
                p for p in path.rglob("*") if p.suffix in (".pth", ".pt", ".tar")
            )
        direct_config = path / "config.json"
        config_candidates = [direct_config] if direct_config.is_file() else sorted(path.rglob("config.json"))

    if not checkpoint_candidates or not config_candidates:
        raise FileNotFoundError(missing_checkpoint_help(model_dir))

    return checkpoint_candidates[0], config_candidates[0]


def vocoder_ready(model_dir: str | Path) -> bool:
    """True when ``model_dir`` contains a usable checkpoint + config pair."""
    try:
        resolve_model_files(model_dir)
    except FileNotFoundError:
        return False
    return True


def missing_checkpoint_help(model_dir: str | Path) -> str:
    """Human-readable instructions for the case where the checkpoint is absent."""
    name = str(model_dir).replace("\\", "/").rstrip("/")
    key = DEFAULT_MODEL_DIR
    for candidate in MODEL_URLS:
        if name.endswith(candidate):
            key = candidate
            break
    url = MODEL_URLS.get(key, MODEL_URLS[DEFAULT_MODEL_DIR])
    return (
        f"No vocoder checkpoint found at '{model_dir}'.\n"
        f"  Download: {url}\n"
        f"  Then unzip it so the folder contains 'config.json' and 'model_file.pth', e.g.\n"
        f"    {DEFAULT_MODEL_DIR}/config.json\n"
        f"    {DEFAULT_MODEL_DIR}/model_file.pth\n"
        f"  and point --vocoder-dir at that folder (default: '{DEFAULT_MODEL_DIR}').\n"
        f"  Verify the install with: python vocoder.py --check <any.wav> --out probe.wav\n"
        f"  Nothing was downloaded automatically."
    )


class HiFiGANVocoder:
    """Thin wrapper around a pretrained Coqui vocoder checkpoint.

    The mel contract is ``[B, n_mels, T]`` -- exactly what :class:`models.Generator` outputs, so no
    transposes are needed anywhere in the pipeline.

    Shapes:
        input  ``[B, n_mels, T]``  (float32, in the mel scale the checkpoint expects --
                                    see :attr:`mel_normalization`)
        output ``[B, 1, T * hop_length]``
    """

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, device: str | torch.device = "cpu") -> None:
        checkpoint_path, config_path = resolve_model_files(model_dir)
        self.model_dir = Path(model_dir)
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = torch.device(device)

        # Imported lazily: keeps `import vocoder` cheap and avoids a hard TTS dependency for
        # callers that only want the helpers above.
        from TTS.config import load_config
        from TTS.utils.audio import AudioProcessor
        from TTS.vocoder.models import setup_model

        self.config: Any = load_config(str(config_path))
        self.audio_config: dict[str, Any] = dict(self.config.audio)
        self.ap = AudioProcessor(self.config.audio)  # mel extraction matching this checkpoint

        self.model = setup_model(self.config)
        self.model.load_checkpoint(self.config, str(checkpoint_path), eval=True)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------ audio contract
    @property
    def sample_rate(self) -> int:
        return int(self.audio_config["sample_rate"])

    @property
    def hop_length(self) -> int:
        return int(self.audio_config["hop_length"])

    @property
    def mel_bins(self) -> int:
        return int(self.audio_config["num_mels"])

    @property
    def mel_normalization(self) -> str:
        """How the checkpoint expects its mel input scaled (from its own config)."""
        if self.audio_config.get("signal_norm"):
            return (
                "normalized (signal_norm=true, "
                f"log_func={self.audio_config.get('log_func')}, "
                f"max_norm={self.audio_config.get('max_norm')})"
            )
        return (
            "raw log-mel, NOT range-normalized (signal_norm=false, "
            f"log_func={self.audio_config.get('log_func')}, spec_gain={self.audio_config.get('spec_gain')})"
        )

    def describe(self) -> str:
        return (
            f"vocoder: {self.model_dir.name}  "
            f"sample_rate={self.sample_rate}  hop_length={self.hop_length}  "
            f"mel_bins={self.mel_bins}  n_fft={self.audio_config.get('fft_size')}  "
            f"mel_fmax={self.audio_config.get('mel_fmax')}  "
            f"checkpoint={self.checkpoint_path.name}"
        )

    def describe_mel(self) -> str:
        return (
            f"mel scaling: {self.mel_normalization}  "
            f"mel_fmin={self.audio_config.get('mel_fmin')} mel_fmax={self.audio_config.get('mel_fmax')}"
        )

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel ``[B, mel_bins, T]`` -> waveform ``[B, 1, T * hop_length]``."""
        if mel.dim() != 3:
            raise ValueError(f"vocoder expects mel of shape [B, {self.mel_bins}, T], got {tuple(mel.shape)}")
        if mel.shape[1] != self.mel_bins:
            raise ValueError(
                f"vocoder checkpoint was trained on {self.mel_bins} mel bins but received "
                f"{mel.shape[1]} (shape {tuple(mel.shape)})"
            )

        mel = mel.to(self.device, dtype=torch.float32)
        time_steps = mel.shape[-1]
        wav = self.model.inference(mel)
        if wav.dim() == 2:  # some vocoders return [B, N]
            wav = wav.unsqueeze(1)

        # Coqui's HiFi-GAN pads the mel before upsampling, so the raw output can be off by a few
        # hundred samples. Trim/pad to the exact T * hop_length the pipeline expects.
        expected = time_steps * self.hop_length
        if wav.shape[-1] > expected:
            wav = wav[..., :expected]
        elif wav.shape[-1] < expected:
            wav = torch.nn.functional.pad(wav, (0, expected - wav.shape[-1]))
        return wav

    __call__ = forward

    # ------------------------------------------------------------------ helpers
    def mel_from_waveform(self, waveform: np.ndarray | torch.Tensor, sample_rate: int | None = None) -> torch.Tensor:
        """Extract the mel this vocoder expects -- use this for real training targets too.

        Args:
            waveform: ``[N]`` or ``[B, N]`` float waveform.
            sample_rate: set it when the input is not already at ``self.sample_rate``; the waveform
                is then resampled with the vocoder's own AudioProcessor settings.

        Returns:
            ``[B, mel_bins, T]`` tensor on the vocoder's device.
        """
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        waveform = np.asarray(waveform, dtype=np.float32)

        if sample_rate is not None and sample_rate != self.sample_rate:
            import librosa

            if waveform.ndim == 2:
                waveform = np.stack(
                    [librosa.resample(w, orig_sr=sample_rate, target_sr=self.sample_rate) for w in waveform]
                )
            else:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=self.sample_rate)

        if waveform.ndim == 2:
            mel = np.stack([self.ap.melspectrogram(w) for w in waveform])
        elif waveform.ndim == 1:
            mel = self.ap.melspectrogram(waveform)[None, ...]
        else:
            raise ValueError(f"expected a waveform of shape [N] or [B, N], got {waveform.shape}")
        return torch.from_numpy(np.ascontiguousarray(mel)).to(self.device, dtype=torch.float32)

    def render(self, mel: torch.Tensor, path: str | Path) -> Path:
        """Vocode ``mel [B, mel_bins, T]`` and write the first item of the batch to ``path``."""
        import soundfile as sf

        wav = self.forward(mel)
        audio = wav[0, 0].detach().cpu().numpy()
        audio = np.clip(audio, -1.0, 1.0)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), audio, self.sample_rate, subtype="PCM_16")
        return out


def load_vocoder(
    model_dir: str | Path = DEFAULT_MODEL_DIR, device: str | torch.device = "cpu"
) -> HiFiGANVocoder | None:
    """Load the vocoder, or return ``None`` (after printing help) when the checkpoint is absent.

    Returning ``None`` instead of raising keeps ``python train.py`` runnable before the checkpoint
    has been downloaded -- but it never substitutes a fake vocoder: audio rendering is simply
    disabled and the reason is printed.
    """
    try:
        return HiFiGANVocoder(model_dir, device=device)
    except FileNotFoundError as exc:
        print(str(exc))
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect / smoke-test the real neural vocoder (no download is performed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="folder with config.json + checkpoint")
    parser.add_argument("--check", default=None, help="audio file to round-trip: waveform -> mel -> vocoder -> wav")
    parser.add_argument("--out", default=None, help="output wav path for --check (default: <check>_vocoded.wav)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seconds", type=float, default=5.0, help="max audio length to use from --check")
    args = parser.parse_args(argv)

    if not vocoder_ready(args.model_dir):
        print(missing_checkpoint_help(args.model_dir))
        return 2

    vocoder = HiFiGANVocoder(args.model_dir, device=args.device)
    print(vocoder.describe())
    print(f"  {vocoder.describe_mel()}")
    print(f"  checkpoint: {vocoder.checkpoint_path}")
    print(f"  config    : {vocoder.config_path}")

    # shape check with a random mel in the vocoder's own value range
    random_mel = torch.randn(1, vocoder.mel_bins, 50, device=vocoder.device)
    wav = vocoder.forward(random_mel)
    print(f"  random check: mel {tuple(random_mel.shape)} -> wav {tuple(wav.shape)} "
          f"(expected last dim {50 * vocoder.hop_length})")
    if wav.shape[-1] != 50 * vocoder.hop_length:
        print("  [!] unexpected waveform length")
        return 1

    if args.check:
        import librosa

        audio, sr = librosa.load(args.check, sr=None, mono=True)
        if args.seconds and len(audio) > int(args.seconds * sr):
            audio = audio[: int(args.seconds * sr)]
        mel = vocoder.mel_from_waveform(audio, sample_rate=sr)
        out = Path(args.out) if args.out else Path(args.check).with_name(Path(args.check).stem + "_vocoded.wav")
        vocoder.render(mel, out)
        duration = mel.shape[-1] * vocoder.hop_length / vocoder.sample_rate
        print(f"  mel values : min {mel.min().item():+.2f}  max {mel.max().item():+.2f}  "
              f"mean {mel.mean().item():+.2f}  <- the training targets must live in this range")
        print(f"  round-trip : {args.check} ({sr} Hz) -> mel {tuple(mel.shape)} "
              f"({mel.shape[-1]} frames ~ {duration:.2f}s) -> {out}")
        print("  wrote the vocoded file; listen to it to confirm the checkpoint is correct.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
