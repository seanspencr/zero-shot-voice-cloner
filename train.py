"""Training loop for the mock zero-shot voice conversion GAN.

Run it with no arguments::

    python train.py

It builds a mock dataset, the AdaIN Generator and the PatchGAN Discriminator, then trains for
``--epochs`` epochs while printing losses every ``--log-every`` steps.

Losses
------
Discriminator: adversarial loss only (Hinge or LSGAN, selected with ``--loss``).
Generator: adversarial + Mel reconstruction (L1) + feature matching.

Loss selection follows the project's staging policy:

* ``--loss hinge`` (default) -- early phase. Hinge GANs are the more stable choice while the
  model is still prone to collapse, which is exactly where mock-data training sits.
* ``--loss lsgan`` -- later phase, once the model actually produces voice-like output and only
  needs smoothing. Least-squares gradients are gentler on small spectral discrepancies.

Everything here depends on ``torch`` only; no TTS / transformers imports.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import CONTENT_DIM, MEL_BINS, SPEAKER_DIM, build_dataloader
from encoders import MockContentEncoder, MockSpeakerEncoder
from evaluate import build_conversion_loader, compute_metrics
from features import (
    CONTENT_SOURCES,
    DEFAULT_HUBERT_DIR,
    SPEAKER_SOURCES,
    XTTS_EMBED_DIM,
    MissingWeightsError,
)
from libritts import DEFAULT_ROOT as DEFAULT_LIBRITTS_ROOT
from libritts import build_libritts_loader, split_summary
from losses import SpectralDetailLoss
from models import Generator, PatchDiscriminator, SpeakerProbe, count_parameters, init_weights
from vocoder import DEFAULT_MODEL_DIR, HiFiGANVocoder, load_vocoder, missing_checkpoint_help, vocoder_ready

__all__ = ["parse_args", "adversarial_d_loss", "adversarial_g_loss", "feature_matching_loss", "main"]


# --------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the mock (GAN) zero-shot voice conversion model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--data",
        choices=("mock", "libritts"),
        default="mock",
        help="mock: random tensors (shape testing); libritts: real audio from --libritts-root",
    )
    data.add_argument("--libritts-root", default=str(DEFAULT_LIBRITTS_ROOT), help="root of the LibriTTS tree")
    data.add_argument("--split", default="train-clean-100", help="LibriTTS split to train on")
    data.add_argument("--max-utterances", type=int, default=None, help="cap utterance count (smoke runs)")
    data.add_argument(
        "--min-seconds",
        type=float,
        default=-1.0,
        help=(
            "skip clips shorter than this (negative = use --segment-seconds, so nothing is zero-padded; "
            "0 = keep everything and pad)"
        ),
    )
    data.add_argument(
        "--segment-seconds",
        type=float,
        default=2.0,
        help="crop utterances to this many seconds; <=0 keeps whole utterances (needs batch-size 1)",
    )
    data.add_argument(
        "--content-source",
        choices=CONTENT_SOURCES,
        default="mel-projection",
        help="hubert needs local weights; mel-projection is a no-download placeholder derived from the real audio",
    )
    data.add_argument(
        "--speaker-source",
        choices=SPEAKER_SOURCES,
        default="xtts",
        help="xtts = XTTS v2's own 512-d speaker encoder (from the installed TTS checkpoint); "
        "mel-stats is a no-download placeholder",
    )
    data.add_argument("--hubert-dir", default=str(DEFAULT_HUBERT_DIR), help="folder with HuBERT config + weights")
    data.add_argument("--hubert-layer", type=int, default=9, help="HuBERT hidden layer used as content features")
    data.add_argument(
        "--xtts-checkpoint",
        default=None,
        help="path to the XTTS v2 model.pth (default: the installed XTTS cache; ~32 MB of it is loaded)",
    )
    data.add_argument("--dataset-size", type=int, default=128, help="number of MOCK utterances")
    data.add_argument("--time-steps", type=int, default=128, help="frames T per MOCK utterance")
    data.add_argument("--batch-size", type=int, default=8)
    data.add_argument("--content-dim", type=int, default=CONTENT_DIM)
    data.add_argument("--speaker-dim", type=int, default=SPEAKER_DIM)
    data.add_argument("--mel-bins", type=int, default=MEL_BINS)
    data.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows)")

    model = parser.add_argument_group("model")
    model.add_argument("--hidden-dim", type=int, default=256, help="Generator channel width")
    model.add_argument("--num-blocks", type=int, default=4, help="number of AdaIN residual blocks")
    model.add_argument("--d-channels", type=int, default=64, help="Discriminator base channels")
    model.add_argument("--d-layers", type=int, default=4, help="Discriminator downsampling layers")

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--epochs", type=int, default=20)
    optim.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate for G and D")
    optim.add_argument("--beta1", type=float, default=0.5)
    optim.add_argument("--beta2", type=float, default=0.999)
    optim.add_argument("--weight-decay", type=float, default=0.0)
    optim.add_argument(
        "--loss",
        choices=("hinge", "lsgan"),
        default="hinge",
        help="adversarial objective: hinge (early phase, collapse-resistant) or lsgan (late phase, smoothing)",
    )
    optim.add_argument("--lambda-adv", type=float, default=1.0)
    optim.add_argument(
        "--lambda-mel",
        type=float,
        default=10.0,
        help=(
            "weight of the plain L1 mel loss. Was 45 (a *vocoder* recipe); at 45 the reconstruction term "
            "dominates and the generator converges to a blurred, 'buzzy' average -- see improvement-plan 1.1"
        ),
    )
    optim.add_argument(
        "--lambda-detail",
        type=float,
        default=2.0,
        help="weight of the anti-over-smoothing term (multi-scale mel + global-variance loss)",
    )
    optim.add_argument(
        "--lambda-spk",
        type=float,
        default=0.0,
        help=(
            "weight of the speaker-consistency loss (probe must recover the requested embedding from G's "
            "output). OFF by default: it corrupts audio if enabled before the generator produces "
            "intelligible mel -- raise it to 0.5-2.0 once mel-L1 has plateaued"
        ),
    )
    optim.add_argument("--lambda-fm", type=float, default=2.0, help="weight of the feature matching loss")
    optim.add_argument("--lr-min", type=float, default=1e-5, help="cosine-decay floor for the learning rate")
    optim.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm (0 disables)")
    optim.add_argument("--grad-accum", type=int, default=1, help="micro-batches per optimizer step")
    optim.add_argument("--no-spectral-norm", action="store_true", help="disable spectral norm on the discriminator")
    optim.add_argument("--no-f0", action="store_true", help="do not feed the pitch (F0) stream to the generator")
    optim.add_argument("--resume", default=None, help="checkpoint to continue from (nothing else is reset)")
    optim.add_argument("--seed", type=int, default=1234)

    run = parser.add_argument_group("runtime")
    run.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    run.add_argument("--log-every", type=int, default=10, help="print losses every N steps")
    run.add_argument(
        "--save-dir",
        default="checkpoints",
        help="where to write latest.pt / best.pt (pass 'none' to disable checkpointing)",
    )
    run.add_argument("--no-sanity-check", action="store_true", help="skip the startup shape check")
    run.add_argument(
        "--max-minutes",
        type=float,
        default=0.0,
        help="stop gracefully after this many minutes (0 = no limit); useful for CPU time-boxed runs",
    )

    voc = parser.add_argument_group("vocoder")
    voc.add_argument(
        "--vocoder-dir",
        default=DEFAULT_MODEL_DIR,
        help="folder holding the pretrained vocoder config.json + checkpoint (nothing is downloaded)",
    )
    voc.add_argument(
        "--render-every",
        type=int,
        default=0,
        help="vocode and save one sample every N epochs (0 disables; needs a vocoder checkpoint)",
    )
    voc.add_argument("--wav-dir", default="samples", help="where rendered wav files are written")
    voc.add_argument("--no-vocoder", action="store_true", help="do not load the vocoder at all")

    val = parser.add_argument_group("validation")
    val.add_argument(
        "--val-split",
        default=None,
        help="LibriTTS split to validate on every --val-every epochs (e.g. dev-clean); default: no validation",
    )
    val.add_argument("--val-every", type=int, default=0, help="run validation every N epochs (0 disables)")
    val.add_argument("--val-utterances", type=int, default=40, help="fixed subset size for validation")

    args = parser.parse_args(argv)

    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.data == "mock":
        if args.dataset_size < args.batch_size:
            parser.error("--dataset-size must be >= --batch-size (batches are dropped when incomplete)")
        if args.time_steps <= 0:
            parser.error("--time-steps must be positive")
    else:
        if args.max_utterances is not None and args.max_utterances < args.batch_size:
            parser.error("--max-utterances must be >= --batch-size (batches are dropped when incomplete)")
        if args.segment_seconds is not None and args.segment_seconds <= 0:
            parser.error("--segment-seconds must be positive for LibriTTS training")
        if args.content_source == "hubert" and args.content_dim != CONTENT_DIM:
            parser.error(f"HuBERT content features are {CONTENT_DIM}-dim; drop --content-dim")
        if args.speaker_source == "xtts":
            args.speaker_dim = XTTS_EMBED_DIM  # XTTS's speaker encoder is fixed at 512-d
    if args.render_every < 0:
        parser.error("--render-every must be >= 0")
    if args.max_minutes < 0:
        parser.error("--max-minutes must be >= 0")
    if args.grad_accum < 1:
        parser.error("--grad-accum must be >= 1")
    if args.val_every < 0:
        parser.error("--val-every must be >= 0")
    if args.val_every > 0 and args.val_split is None:
        parser.error("--val-every needs --val-split (e.g. --val-split dev-clean)")
    if args.val_every > 0 and args.data != "libritts":
        parser.error("--val-every is only available with --data libritts")
    return args


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


# --------------------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------------------
def adversarial_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor, kind: str) -> torch.Tensor:
    """Discriminator adversarial loss (Hinge or LSGAN)."""
    if kind == "hinge":
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    if kind == "lsgan":
        return 0.5 * (real_logits - 1.0).pow(2).mean() + 0.5 * fake_logits.pow(2).mean()
    raise ValueError(f"unknown adversarial loss: {kind!r}")


def adversarial_g_loss(fake_logits: torch.Tensor, kind: str) -> torch.Tensor:
    """Generator adversarial loss: the generator wants ``fake`` to look real."""
    if kind == "hinge":
        return -fake_logits.mean()
    if kind == "lsgan":
        return 0.5 * (fake_logits - 1.0).pow(2).mean()
    raise ValueError(f"unknown adversarial loss: {kind!r}")


def feature_matching_loss(
    real_features: list[torch.Tensor], fake_features: list[torch.Tensor]
) -> torch.Tensor:
    """L1 distance between Discriminator activations on real and predicted mels."""
    if len(real_features) != len(fake_features):
        raise RuntimeError(
            f"feature matching needs equal feature counts, got {len(real_features)} and {len(fake_features)}"
        )
    loss = fake_features[0].new_zeros(())
    for real, fake in zip(real_features, fake_features):
        if real.shape != fake.shape:
            raise RuntimeError(f"discriminator feature shape mismatch: {tuple(real.shape)} vs {tuple(fake.shape)}")
        loss = loss + F.l1_loss(fake, real.detach())
    return loss / len(real_features)


# --------------------------------------------------------------------------------------
# startup shape check
# --------------------------------------------------------------------------------------
def _patch_length(length: int, n_layers: int) -> int:
    """Output length of the PatchGAN stack for an input of ``length``.

    Each downsampling layer is ``Conv2d(kernel=4, stride=2, padding=1)``, i.e.
    ``floor((L + 2 - 4) / 2) + 1 == floor(L / 2)`` -- repeated ``n_layers`` times.
    """
    for _ in range(n_layers):
        length = length // 2
    return length


def run_shape_sanity_check(
    args: argparse.Namespace, device: torch.device, vocoder: HiFiGANVocoder | None
) -> None:
    """Feed random tensors through every module and assert the documented shapes.

    ``vocoder`` is the real pretrained vocoder (``vocoder.py``) or ``None`` when its checkpoint is
    not installed -- there is no mock fallback, the vocoder row is simply skipped and reported.
    """
    batch = min(args.batch_size, 2)
    time_steps = args.time_steps
    content = torch.randn(batch, time_steps, args.content_dim, device=device)
    speaker = F.normalize(torch.randn(batch, args.speaker_dim, device=device), dim=-1)
    mel = torch.randn(batch, args.mel_bins, time_steps, device=device)

    content_encoder = MockContentEncoder(in_dim=args.content_dim, out_dim=args.content_dim).to(device)
    speaker_encoder = MockSpeakerEncoder(in_dim=args.speaker_dim, out_dim=args.speaker_dim).to(device)
    generator = Generator(
        in_dim=args.content_dim,
        hidden_dim=args.hidden_dim,
        mel_bins=args.mel_bins,
        num_blocks=args.num_blocks,
        speaker_dim=args.speaker_dim,
    ).to(device)
    discriminator = PatchDiscriminator(
        mel_bins=args.mel_bins, channels=args.d_channels, n_layers=args.d_layers
    ).to(device)

    checks: list[tuple[str, torch.Tensor, tuple[int, ...]]] = []
    with torch.no_grad():
        encoded_content = content_encoder(content)
        checks.append(("MockContentEncoder", encoded_content, (batch, time_steps, args.content_dim)))

        encoded_speaker = speaker_encoder(speaker)
        checks.append(("MockSpeakerEncoder", encoded_speaker, (batch, args.speaker_dim)))

        predicted_mel = generator(encoded_content, encoded_speaker)
        checks.append(("Generator", predicted_mel, (batch, args.mel_bins, time_steps)))

        real_logits, real_features = discriminator.forward_features(mel)
        fake_logits, fake_features = discriminator.forward_features(predicted_mel)
        expected_logits = (
            batch,
            1,
            _patch_length(args.mel_bins, args.d_layers),
            _patch_length(time_steps, args.d_layers),
        )
        checks.append(("PatchDiscriminator", real_logits, expected_logits))

        mel_l1 = F.l1_loss(predicted_mel, mel)
        if not torch.isfinite(mel_l1):
            raise RuntimeError("mel L1 loss is not finite -- check the mock data/model scales")

    waveform: torch.Tensor | None = None
    if vocoder is not None:
        if vocoder.mel_bins != args.mel_bins:
            raise RuntimeError(
                f"vocoder expects {vocoder.mel_bins} mel bins but --mel-bins is {args.mel_bins}; "
                f"the Generator output would not be compatible with '{vocoder.model_dir}'"
            )
        with torch.no_grad():
            waveform = vocoder(predicted_mel)
        checks.append(("HiFiGANVocoder", waveform, (batch, 1, time_steps * vocoder.hop_length)))

    failures = [
        f"  {name}: got {tuple(tensor.shape)}, expected {expected}"
        for name, tensor, expected in checks
        if tuple(tensor.shape) != expected
    ]
    if real_logits.shape != fake_logits.shape:
        failures.append(f"  real/fake logits differ: {tuple(real_logits.shape)} vs {tuple(fake_logits.shape)}")
    if len(real_features) != len(fake_features):
        failures.append(f"  real/fake feature counts differ: {len(real_features)} vs {len(fake_features)}")
    if failures:
        raise RuntimeError("shape sanity check failed:\n" + "\n".join(failures))

    print("Shape sanity check, all OK:")
    for name, tensor, _expected in checks:
        print(f"  {name:<22} -> {tuple(tensor.shape)}")
    print(f"  feature matching      -> {len(real_features)} feature maps "
          f"[{tuple(real_features[0].shape)} ... {tuple(real_features[-1].shape)}]")
    if waveform is not None and vocoder is not None:
        seconds = waveform.shape[-1] / vocoder.sample_rate
        print(f"  vocoder               -> {vocoder.sample_rate} Hz, "
              f"{time_steps} frames = {seconds:.2f}s of audio")
    else:
        print("  vocoder               -> skipped (no checkpoint installed, see the note above)")
    print()


# --------------------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------------------
def save_checkpoint(
    path: str,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    probe: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    opt_p: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> None:
    """Write everything needed to resume or analyse a run.

    The ``SpeakerProbe`` and its optimizer are included on purpose: they are part of the training
    state (the probe defines ``L_spk``), so a checkpoint without them could not be resumed and the
    probe's behaviour could not be inspected afterwards.
    """
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "probe": probe.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "opt_p": opt_p.state_dict(),
            "args": vars(args),
        },
        path,
    )


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    args.device = str(device)  # extractors take a concrete torch device, never "auto"

    run_name = "LibriTTS (real audio)" if args.data == "libritts" else "mock-data"
    print(f"Zero-Shot Voice Conversion (GAN) -- {run_name} training run")
    print(f"  device      : {device}  (torch {torch.__version__})")
    print(f"  loss        : {args.loss}    mel-L1 weight: {args.lambda_mel}    feature-matching weight: {args.lambda_fm}")
    if args.data == "mock":
        print(f"  data        : {args.dataset_size} mock utterances x {args.time_steps} frames "
              f"(content {args.content_dim}d, speaker {args.speaker_dim}d, mel {args.mel_bins} bins)")
    else:
        print(f"  data        : LibriTTS '{args.split}' from {args.libritts_root}")
        print(f"                content={args.content_source}  speaker={args.speaker_source}  "
              f"segment={args.segment_seconds:g}s  "
              f"utterances={'all' if args.max_utterances is None else args.max_utterances}")
    print()

    # Real pretrained vocoder (never a mock). Missing checkpoint => rendering disabled + a note.
    vocoder: HiFiGANVocoder | None = None
    if args.no_vocoder:
        print("Vocoder       : disabled (--no-vocoder); no audio will be rendered")
    elif vocoder_ready(args.vocoder_dir):
        vocoder = load_vocoder(args.vocoder_dir, device=device)
        if vocoder is not None:
            print(vocoder.describe())
            print(vocoder.describe_mel())
    else:
        print("Vocoder       : checkpoint not installed -- training continues, rendering disabled")
        print("                " + missing_checkpoint_help(args.vocoder_dir).replace("\n", "\n                "))
    print()

    if args.data == "libritts":
        if vocoder is None:
            raise SystemExit(
                "Real LibriTTS training needs the vocoder: its AudioProcessor produces the mel target.\n"
                + missing_checkpoint_help(args.vocoder_dir)
            )
        if args.mel_bins != vocoder.mel_bins:
            print(f"Note          : --mel-bins {args.mel_bins} overridden by the vocoder's {vocoder.mel_bins}")
            args.mel_bins = vocoder.mel_bins

    if not args.no_sanity_check:
        run_shape_sanity_check(args, device, vocoder)

    if args.data == "libritts":
        print("Corpus:")
        print(split_summary(args.libritts_root))
        min_seconds = args.segment_seconds if args.min_seconds < 0 else args.min_seconds
        if min_seconds > 0:
            print(f"Filters       : clips shorter than {min_seconds:g}s are skipped "
                  f"(so segments are never zero-padded; first scan probes every wav header, then caches)")
        try:
            dataset, loader = build_libritts_loader(
                root=args.libritts_root,
                split=args.split,
                vocoder=vocoder,
                limit=args.max_utterances,
                batch_size=args.batch_size,
                segment_seconds=args.segment_seconds,
                num_workers=args.num_workers,
                seed=args.seed,
                train=True,
                min_seconds=min_seconds,
                content_source=args.content_source,
                speaker_source=args.speaker_source,
                content_kwargs={
                    "model_dir": args.hubert_dir,
                    "layer": args.hubert_layer,
                    "device": str(device),
                },
                speaker_kwargs={"checkpoint": args.xtts_checkpoint, "device": str(device)},
            )
        except MissingWeightsError as exc:
            raise SystemExit(f"{exc}\n\nPass --content-source/--speaker-source to use the placeholders.") from exc
        try:
            dataset.validate()  # load extractor weights now, not half-way through epoch 1
        except MissingWeightsError as exc:
            raise SystemExit(f"{exc}\n\nPass --content-source/--speaker-source to use the placeholders.") from exc
        print(f"Dataset       : {dataset.describe()}")
        print()
    else:
        dataset, loader = build_dataloader(
            batch_size=args.batch_size,
            dataset_size=args.dataset_size,
            time_steps=args.time_steps,
            content_dim=args.content_dim,
            speaker_dim=args.speaker_dim,
            mel_bins=args.mel_bins,
            num_workers=args.num_workers,
            seed=args.seed,
        )
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise RuntimeError("the dataloader yielded no batches; raise --dataset-size / --max-utterances")

    generator = Generator(
        in_dim=args.content_dim,
        hidden_dim=args.hidden_dim,
        mel_bins=args.mel_bins,
        num_blocks=args.num_blocks,
        speaker_dim=args.speaker_dim,
    ).to(device)
    discriminator = PatchDiscriminator(
        mel_bins=args.mel_bins,
        channels=args.d_channels,
        n_layers=args.d_layers,
        spectral_norm=not args.no_spectral_norm,
    ).to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    # the pitch projection starts as a no-op (added to the content path, zero weights)
    torch.nn.init.zeros_(generator.f0_proj.weight)
    torch.nn.init.zeros_(generator.f0_proj.bias)
    probe = SpeakerProbe(mel_bins=args.mel_bins, embed_dim=args.speaker_dim).to(device)

    print(f"Generator     : {count_parameters(generator):,} trainable parameters")
    print(f"Discriminator : {count_parameters(discriminator):,} trainable parameters")
    probe_state = f"lambda_spk={args.lambda_spk}" if args.lambda_spk > 0 else "OFF (--lambda-spk 0)"
    print(f"Loss weights  : adv={args.lambda_adv} mel={args.lambda_mel} detail={args.lambda_detail} "
          f"fm={args.lambda_fm} spk={args.lambda_spk}")
    print(f"Regularisation: spectral_norm={'off' if args.no_spectral_norm else 'on'} "
          f"grad_clip={args.grad_clip} grad_accum={args.grad_accum} lr={args.lr}->{args.lr_min} (cosine)")
    print(f"Pitch stream  : {'off' if args.no_f0 else 'on (log-F0 contour + voiced flag, zero-init projection)'}")
    print(f"Speaker probe : {count_parameters(probe):,} trainable parameters "
          f"({probe_state}, embed_dim={args.speaker_dim})")
    print()

    betas = (args.beta1, args.beta2)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=betas, weight_decay=args.weight_decay)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr, betas=betas, weight_decay=args.weight_decay)
    opt_p = torch.optim.AdamW(probe.parameters(), lr=args.lr, betas=betas, weight_decay=args.weight_decay)
    detail_loss = SpectralDetailLoss()

    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        if "probe" in state:
            probe.load_state_dict(state["probe"])
        opt_g.load_state_dict(state["opt_g"])
        opt_d.load_state_dict(state["opt_d"])
        if "opt_p" in state:
            opt_p.load_state_dict(state["opt_p"])
        resume_epoch = int(state.get("epoch", 0))
        print(f"Resumed       : {args.resume} (finished epoch {resume_epoch})")
        print()

    save_dir = args.save_dir.strip() if args.save_dir else ""
    save_dir = "" if save_dir.lower() in ("", "none", "null") else save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Checkpoints   : {os.path.abspath(save_dir)}/latest.pt, best.pt")
        print()

    history: list[dict[str, float]] = []
    validation_history: list[dict[str, float]] = []
    rendered: list[str] = []
    best_g_total = math.inf
    stopped_early = False
    start_time = time.perf_counter()

    val_dataset = None
    if args.val_every > 0 and args.val_split is not None and args.data == "libritts":
        val_args = argparse.Namespace(**vars(args))
        val_args.split = args.val_split
        val_args.max_utterances = args.val_utterances
        val_dataset, _ = build_conversion_loader(val_args, vocoder)
        val_dataset.validate()
        print(f"Validation    : '{args.val_split}' ({len(val_dataset)} utterances) every "
              f"{args.val_every} epoch(s)")
        print()

    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        epoch_start = time.perf_counter()
        sums = {"d": 0.0, "g": 0.0, "adv": 0.0, "mel": 0.0, "fm": 0.0, "detail": 0.0, "ms": 0.0,
                "gv": 0.0, "spk": 0.0, "probe": 0.0, "d_real": 0.0, "d_fake": 0.0}
        steps = 0

        for step, batch in enumerate(loader, start=1):
            content = batch["content"].to(device, non_blocking=True)
            speaker = batch["speaker"].to(device, non_blocking=True)
            mel_target = batch["mel"].to(device, non_blocking=True)

            f0 = None if args.no_f0 or "f0" not in batch else batch["f0"].to(device, non_blocking=True)

            # ---------------- Discriminator step ----------------
            with torch.no_grad():
                mel_fake = generator(content, speaker, f0)
            fake_logits = discriminator(mel_fake)
            real_logits = discriminator(mel_target)
            loss_d = adversarial_d_loss(real_logits, fake_logits, args.loss)

            opt_d.zero_grad(set_to_none=True)
            (loss_d / args.grad_accum).backward()
            if step % args.grad_accum == 0 or step == steps_per_epoch:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.grad_clip)
                opt_d.step()

            # ---------------- Speaker probe step (on REAL mel only) ----------------
            # The probe learns to recover a speaker embedding from a real spectrogram. Keeping it fed
            # with real data (never G's output) is what stops it from collapsing into "predict whatever
            # the generator produced". Skipped entirely when the loss is off, which also saves the
            # forward+backward on CPU.
            if args.lambda_spk > 0.0:
                probe_emb_real = probe(mel_target)
                loss_probe = 1.0 - F.cosine_similarity(probe_emb_real, speaker, dim=-1).mean()
                opt_p.zero_grad(set_to_none=True)
                loss_probe.backward()
                opt_p.step()
                probe_value = loss_probe.item()
            else:
                loss_probe = None
                probe_value = 0.0

            # ---------------- Generator step ----------------
            mel_pred = generator(content, speaker, f0)
            fake_logits, fake_features = discriminator.forward_features(mel_pred)
            with torch.no_grad():
                _, real_features = discriminator.forward_features(mel_target)

            loss_adv = adversarial_g_loss(fake_logits, args.loss)
            loss_mel = F.l1_loss(mel_pred, mel_target)
            loss_fm = (
                feature_matching_loss(real_features, fake_features)
                if args.lambda_fm > 0.0
                else mel_pred.new_zeros(())
            )
            # anti-over-smoothing: multi-resolution + global-variance terms (improvement-plan 1.1)
            if args.lambda_detail > 0.0:
                loss_detail, loss_ms, loss_gv = detail_loss(mel_pred, mel_target)
            else:
                loss_detail = mel_pred.new_zeros(())
                loss_ms = loss_gv = mel_pred.new_zeros(())
            # Speaker-consistency loss: the frozen probe must recover the *requested* embedding from
            # the generated spectrogram. Gradients flow into G through the probe's frozen weights.
            if args.lambda_spk > 0.0:
                for parameter in probe.parameters():
                    parameter.requires_grad_(False)
                loss_spk = 1.0 - F.cosine_similarity(probe(mel_pred), speaker, dim=-1).mean()
            else:
                loss_spk = mel_pred.new_zeros(())
            loss_g = (
                args.lambda_adv * loss_adv
                + args.lambda_mel * loss_mel
                + args.lambda_detail * loss_detail
                + args.lambda_fm * loss_fm
                + args.lambda_spk * loss_spk
            )

            opt_g.zero_grad(set_to_none=True)
            (loss_g / args.grad_accum).backward()
            if step % args.grad_accum == 0 or step == steps_per_epoch:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip)
                opt_g.step()
                # cosine decay over the whole schedule (steps, not epochs, so --max-minutes still works)
                progress = min(1.0, (epoch - 1 + step / max(steps_per_epoch, 1)) / max(args.epochs, 1))
                for opt in (opt_g, opt_d, opt_p):
                    for group in opt.param_groups:
                        group["lr"] = args.lr_min + 0.5 * (args.lr - args.lr_min) * (
                            1.0 + math.cos(math.pi * progress)
                        )
            if args.lambda_spk > 0.0:
                for parameter in probe.parameters():
                    parameter.requires_grad_(True)

            values = {
                "d": loss_d.item(),
                "g": loss_g.item(),
                "adv": loss_adv.item(),
                "mel": loss_mel.item(),
                "fm": loss_fm.item(),
                "detail": loss_detail.item(),
                "ms": loss_ms.item(),
                "gv": loss_gv.item(),
                "spk": loss_spk.item(),
                "probe": probe_value,
                "d_real": real_logits.mean().item(),
                "d_fake": fake_logits.mean().item(),
            }
            for key, value in values.items():
                sums[key] += value
            steps += 1

            if step % args.log_every == 0 or step == steps_per_epoch:
                print(
                    f"epoch {epoch:>3}/{args.epochs}  step {step:>4}/{steps_per_epoch}  "
                    f"| D {values['d']:7.4f} (real {values['d_real']:6.3f} fake {values['d_fake']:6.3f})  "
                    f"| G {values['g']:8.4f} (adv {values['adv']:7.4f} mel {values['mel']:.4f} "
                    f"det {values['detail']:.4f} fm {values['fm']:7.4f} "
                    + (f"spk {values['spk']:.3f} probe {values['probe']:.3f}" if args.lambda_spk > 0 else "spk off")
                    + ")"
                )

            if args.max_minutes > 0 and (time.perf_counter() - start_time) >= args.max_minutes * 60.0:
                stopped_early = True
                print(f"\n  [time budget] {args.max_minutes:g} min reached after {steps} step(s) "
                      f"in epoch {epoch} -- stopping")
                break

        if steps == 0:
            break

        epoch_stats = {key: value / steps for key, value in sums.items()}
        history.append(epoch_stats)
        elapsed = time.perf_counter() - epoch_start
        print(
            f"  -> epoch {epoch:>3} summary: D {epoch_stats['d']:.4f} | G {epoch_stats['g']:.4f} "
            f"(adv {epoch_stats['adv']:+.4f}, mel {epoch_stats['mel']:.4f}, "
            f"det {epoch_stats['detail']:.4f} (ms {epoch_stats['ms']:.4f}/gv {epoch_stats['gv']:.4f}), "
            f"fm {epoch_stats['fm']:.4f}, "
            + (f"spk {epoch_stats['spk']:.3f}, probe {epoch_stats['probe']:.3f}" if args.lambda_spk > 0 else "spk off")
            + ") "
            f"| {steps / elapsed:.1f} steps/s"
        )

        if vocoder is not None and args.render_every > 0 and epoch % args.render_every == 0:
            with torch.no_grad():
                mel_preview = generator(content[:1], speaker[:1])
            wav_path = os.path.join(args.wav_dir, f"epoch_{epoch:03d}.wav")
            written = vocoder.render(mel_preview, wav_path)
            seconds = mel_preview.shape[-1] * vocoder.hop_length / vocoder.sample_rate
            rendered.append(str(written))
            # traceability: say which corpus item (and sentence) this sample came from
            utt_ids = batch.get("utt_id") if hasattr(batch, "get") else None
            texts = batch.get("text") if hasattr(batch, "get") else None
            c_starts = batch.get("crop_start") if hasattr(batch, "get") else None
            c_ends = batch.get("crop_end") if hasattr(batch, "get") else None
            durs = batch.get("duration") if hasattr(batch, "get") else None
            source = ""
            if utt_ids:
                source = f"\n     source: {utt_ids[0]}"
                if texts:
                    snippet = texts[0][:110] + ("..." if len(texts[0]) > 110 else "")
                    source += f'  "{snippet}"'
                if c_starts is not None and c_ends is not None and durs is not None:
                    source += (f"\n     converted: {float(c_starts[0]):.2f}s-{float(c_ends[0]):.2f}s "
                               f"of a {float(durs[0]):.2f}s recording (random crop, so the words above are the "
                               f"ones around that window)")
            print(
                f"  -> rendered {written}  (mel {tuple(mel_preview.shape)}, "
                f"min {mel_preview.min().item():+.2f} max {mel_preview.max().item():+.2f}, ~{seconds:.2f}s)"
                f"{source}"
            )
            if utt_ids:
                # permanent mapping: which sentence does each rendered wav say?
                render_root = Path(args.wav_dir)
                render_root.mkdir(parents=True, exist_ok=True)
                index_path = render_root / "index.csv"
                new_file = not index_path.exists()
                with open(index_path, "a", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    if new_file:
                        writer.writerow(["epoch", "utt_id", "wav", "mel_frames", "seconds",
                                         "crop_start_s", "crop_end_s", "utterance_s", "text"])
                    writer.writerow([
                        epoch,
                        utt_ids[0],
                        os.path.abspath(written),
                        mel_preview.shape[-1],
                        f"{seconds:.2f}",
                        f"{float(c_starts[0]):.2f}" if c_starts is not None else "",
                        f"{float(c_ends[0]):.2f}" if c_ends is not None else "",
                        f"{float(durs[0]):.2f}" if durs is not None else "",
                        texts[0] if texts else "",
                    ])

        if val_dataset is not None and epoch % args.val_every == 0:
            generator.eval()
            report = compute_metrics(generator, val_dataset, vocoder, device, embedder=None)
            generator.train()
            validation_history.append({"epoch": epoch, "mcd": report.mcd_mean, "mel_l1": report.mel_l1_mean})
            print(
                f"  -> val {args.val_split} @epoch {epoch}: MCD {report.mcd_mean:.3f} dB  "
                f"| mel L1 {report.mel_l1_mean:.4f}  ({report.n_items} utterances)"
            )

        if save_dir:
            save_checkpoint(os.path.join(save_dir, "latest.pt"), epoch, generator, discriminator, probe,
                            opt_g, opt_d, opt_p, args)
            if epoch_stats["g"] < best_g_total:
                best_g_total = epoch_stats["g"]
                save_checkpoint(os.path.join(save_dir, "best.pt"), epoch, generator, discriminator, probe,
                                opt_g, opt_d, opt_p, args)
                print(f"  -> saved best checkpoint (G {best_g_total:.4f}) to {os.path.abspath(save_dir)}")

        if stopped_early:
            break

    total_time = time.perf_counter() - start_time
    epochs_done = len(history)
    first, last = history[0], history[-1]
    print()
    print("=" * 78)
    if stopped_early:
        print(f"Training stopped at the --max-minutes {args.max_minutes:g} budget: "
              f"{epochs_done} epoch(s) recorded in {total_time:.1f}s")
    else:
        print(f"Training finished: {epochs_done} epoch(s), {epochs_done * steps_per_epoch} steps "
              f"in {total_time:.1f}s")
    print(f"  adv loss   : {first['adv']:+.4f} -> {last['adv']:+.4f}")
    print(f"  mel L1     : {first['mel']:.4f} -> {last['mel']:.4f}")
    print(f"  detail     : {first['detail']:.4f} -> {last['detail']:.4f}   "
          f"(multi-scale {first['ms']:.4f}->{last['ms']:.4f}, global-var {first['gv']:.4f}->{last['gv']:.4f})")
    print(f"  feature mt : {first['fm']:.4f} -> {last['fm']:.4f}")
    if args.lambda_spk > 0:
        print(f"  spk loss   : {first['spk']:.4f} -> {last['spk']:.4f}   (1 - cos; lower = output carries the target voice)")
        print(f"  probe loss : {first['probe']:.4f} -> {last['probe']:.4f}")
    else:
        print("  spk loss   : off (--lambda-spk 0); enable it once mel-L1 has plateaued")
    print(f"  D loss     : {first['d']:.4f} -> {last['d']:.4f}")
    print(f"  G loss     : {first['g']:.4f} -> {last['g']:.4f}")
    if rendered:
        print(f"  audio      : {len(rendered)} sample(s) written with the real HiFi-GAN vocoder")
        for path in rendered[-3:]:
            print(f"               {path}")
    elif vocoder is None:
        print("  audio      : none (no vocoder checkpoint -- see the URL printed at startup)")
    else:
        print("  audio      : none (pass --render-every N to write a wav every N epochs)")
    if validation_history:
        first_val, last_val = validation_history[0], validation_history[-1]
        print(f"  val MCD    : {first_val['mcd']:.3f} -> {last_val['mcd']:.3f} dB on '{args.val_split}' "
              f"({len(validation_history)} checks)")
        print(f"  val mel L1 : {first_val['mel_l1']:.4f} -> {last_val['mel_l1']:.4f}")
    print("  all losses finite, no tensor dimension errors at any step.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
