"""Generator + Discriminator for zero-shot voice conversion (timbre transfer).

Generator
    ``content_features [B, T, 768]`` + ``speaker_embedding [B, 256]``
    -> ``predicted_mel [B, 80, T]``

    Speaker identity is injected with **AdaIN**: the speaker embedding is projected to a
    per-channel ``(scale, bias)`` pair that modulates every convolution block processing the
    content features. Content carries *what* is said, the embedding carries *who* says it.

Discriminator
    PatchGAN over ``[B, 80, T]`` mel-spectrograms -> per-patch real/fake logits.

Both networks are fully convolutional with stride-1 (G) / floor-stride (D) operations, so no
input length ever causes a shape mismatch between the prediction and the target mel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "AdaIN1d",
    "AdaINResBlock",
    "Generator",
    "PatchDiscriminator",
    "SpeakerProbe",
    "init_weights",
    "count_parameters",
]


class AdaIN1d(nn.Module):
    """Adaptive Instance Normalization (1-D, over time).

    Normalizes each channel of the content features over the time axis (per sample), then
    applies a speaker-dependent affine transform::

        y = (1 + scale(z)) * (x - mean_t(x)) / std_t(x) + bias(z)

    Instance statistics are computed manually instead of with ``nn.InstanceNorm1d`` so that
    ``T == 1`` and ``batch_size == 1`` are handled without errors.

    Shapes:
        x: ``[B, C, T]``, z: ``[B, speaker_dim]`` -> ``[B, C, T]``
    """

    def __init__(self, channels: int, speaker_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.to_scale_bias = nn.Linear(speaker_dim, 2 * channels)
        nn.init.zeros_(self.to_scale_bias.weight)  # start as plain instance norm
        nn.init.zeros_(self.to_scale_bias.bias)

    def forward(self, x: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"AdaIN1d expects x of shape [B, C, T], got {tuple(x.shape)}")
        if speaker.dim() != 2:
            raise ValueError(f"AdaIN1d expects speaker of shape [B, D], got {tuple(speaker.shape)}")

        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True, unbiased=False) + self.eps
        x_norm = (x - mean) / std

        scale_bias = self.to_scale_bias(speaker)  # [B, 2C]
        scale, bias = scale_bias.chunk(2, dim=1)
        scale = scale.unsqueeze(-1)  # [B, C, 1]
        bias = bias.unsqueeze(-1)  # [B, C, 1]
        return (1.0 + scale) * x_norm + bias


class AdaINResBlock(nn.Module):
    """Residual block: conv -> AdaIN(speaker) -> LeakyReLU, twice, plus a skip connection.

    Kernel 3 with same padding keeps the time axis length unchanged, so ``T`` survives the whole
    stack and the Generator output matches the target mel frame count exactly.
    """

    def __init__(self, channels: int, speaker_dim: int, kernel_size: int = 3, slope: float = 0.2) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.adain1 = AdaIN1d(channels, speaker_dim)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.adain2 = AdaIN1d(channels, speaker_dim)
        self.slope = slope

    def forward(self, x: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = F.leaky_relu(self.adain1(h, speaker), self.slope)
        h = self.conv2(h)
        h = self.adain2(h, speaker)
        return F.leaky_relu(h + x, self.slope)


class Generator(nn.Module):
    """Content + speaker -> predicted mel-spectrogram.

    Shapes:
        content: ``[B, T, 768]`` (or ``[B, 768, T]``)
        speaker: ``[B, 256]``
        output:  ``[B, 80, T]``  -- identical ``T`` to the input content, by construction

    ~4M parameters, small enough that a 20-epoch mock run is quick on CPU and checkpoints stay
    around ~70 MB.
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        mel_bins: int = 80,
        num_blocks: int = 4,
        speaker_dim: int = 256,
        slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.mel_bins = mel_bins
        self.slope = slope

        self.conv_in = nn.Conv1d(in_dim, hidden_dim, kernel_size=7, padding=3)
        self.adain_in = AdaIN1d(hidden_dim, speaker_dim)
        self.blocks = nn.ModuleList(
            AdaINResBlock(hidden_dim, speaker_dim) for _ in range(num_blocks)
        )
        self.conv_out = nn.Conv1d(hidden_dim, mel_bins, kernel_size=7, padding=3)

    def forward(self, content: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        if content.dim() != 3:
            raise ValueError(
                f"Generator expects content of shape [B, T, {self.in_dim}], got {tuple(content.shape)}"
            )
        if content.shape[-1] == self.in_dim:
            x = content.transpose(1, 2)  # [B, 768, T]
        elif content.shape[1] == self.in_dim:
            x = content  # already [B, in_dim, T]
        else:
            raise ValueError(
                f"Generator expects {self.in_dim}-dim content features, got shape {tuple(content.shape)}"
            )

        h = F.leaky_relu(self.adain_in(self.conv_in(x), speaker), self.slope)
        for block in self.blocks:
            h = block(h, speaker)
        return self.conv_out(h)  # [B, mel_bins, T]


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator over mel-spectrograms.

    Treats the mel-spectrogram as a 1-channel image ``[B, 1, 80, T]`` and classifies patches
    instead of the whole utterance, which is what gives the Generator a local, spectral-detail
    driven gradient signal.

    Shapes:
        input:  ``[B, 80, T]``
        output: logits ``[B, 1, n_bins, n_frames]`` (+ list of intermediate feature maps used by
                the feature-matching loss)

    Downsampling uses ``kernel_size=4, stride=2, padding=1`` whose output length is
    ``floor((L + 2*1 - 4) / 2) + 1 = floor(L / 2)``, so odd ``T`` -- or ``T`` not divisible by
    8 -- is handled without any shape error.
    """

    def __init__(
        self,
        mel_bins: int = 80,
        channels: int = 64,
        n_layers: int = 4,
        in_channels: int = 1,
        slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.mel_bins = mel_bins
        self.slope = slope

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(slope, inplace=True),
        ]
        ch = channels
        for i in range(1, n_layers):
            next_ch = min(ch * 2, 512)
            layers += [
                nn.Conv2d(ch, next_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(next_ch, affine=True),
                nn.LeakyReLU(slope, inplace=True),
            ]
            ch = next_ch
        # collapse the patch grid to a single validity score per patch
        layers.append(nn.Conv2d(ch, 1, kernel_size=3, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward_features(self, mel: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return ``(patch_logits, intermediate feature maps)``."""
        if mel.dim() != 3 or mel.shape[1] != self.mel_bins:
            raise ValueError(
                f"PatchDiscriminator expects mel of shape [B, {self.mel_bins}, T], got {tuple(mel.shape)}"
            )
        h = mel.unsqueeze(1)  # [B, 1, 80, T]
        features: list[torch.Tensor] = []
        for layer in self.net:
            h = layer(h)
            if isinstance(layer, nn.Conv2d) and h.shape[1] > 1:
                features.append(h)
        return h, features

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_features(mel)
        return logits


class SpeakerProbe(nn.Module):
    """Small mel -> speaker-embedding probe, used for the speaker-consistency loss ``L_spk``.

    The generator is otherwise free to ignore its speaker conditioning: nothing in the adversarial or
    mel-reconstruction terms mentions the voice. This head closes that gap. It is trained on **real**
    mel spectrograms to predict the speaker embedding of that utterance (a supervised auxiliary task),
    and then applied to the **generated** mel: the generator is rewarded when the probe recovers the
    *requested* embedding from its own output, which only happens if the output actually carries that
    voice.

    Using an embedding-prediction head (rather than a speaker classifier) means the loss works for
    **unseen** speakers too, which is what zero-shot needs.

    Shapes:
        input  ``[B, mel_bins, T]``
        output ``[B, embed_dim]``, L2-normalized
    """

    def __init__(self, mel_bins: int = 80, embed_dim: int = 512, hidden: int = 128, kernel: int = 5) -> None:
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(mel_bins, hidden, kernel_size=kernel, padding=pad),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=pad),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=pad),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.proj = nn.Linear(2 * hidden, embed_dim)  # mean + std pooling over time

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() != 3:
            raise ValueError(f"SpeakerProbe expects mel of shape [B, C, T], got {tuple(mel.shape)}")
        h = self.net(mel)
        pooled = torch.cat([h.mean(dim=2), h.std(dim=2, unbiased=False)], dim=1)
        return torch.nn.functional.normalize(self.proj(pooled), dim=-1)


def init_weights(module: nn.Module, mean: float = 0.0, std: float = 0.02) -> None:
    """DCGAN-style initialization: normal weights, zero bias, unit scale for norm layers."""
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.normal_(module.weight, mean=mean, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.LayerNorm)):
        if getattr(module, "weight", None) is not None:
            nn.init.ones_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters (for logging / sanity checks)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
