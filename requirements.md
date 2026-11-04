Development Instructions for Zero-Shot Voice Conversion (GAN) System with Mock Data

Project Context

You are an AI Deep Learning Engineer. Your task is to build the base architecture and training pipeline for a Zero-Shot Voice Conversion (Timbre Transfer) project based on a Generative Adversarial Network (GAN).

This system will eventually take content (what is spoken) from Text-to-Speech (Coqui TTS) output and speaker identity (timbre/voice) from a single user audio sample.

Currently, the real dataset (like VCTK) is being prepared. Therefore, your main task right now is to build data placeholders (mock dataset), model architectures (Generator & Discriminator), and a training loop that can be executed without dimension errors (shape mismatch).

Task Specifications

1. Create Mock Dataset & Dataloader (dataset.py)

Create a PyTorch Dataset class that generates random tensors (torch.randn) to simulate extracted data. Use the following dimension specifications:

content_features (Simulated HuBERT/wav2vec 2.0 output):

Shape: [batch_size, time_steps, 768]

Description: Represents the linguistic representation.

speaker_embedding (Simulated d-vector extraction from target user):

Shape: [batch_size, 256]

Description: Represents the user's voice identity.

target_mel_spectrogram (Ground truth target audio):

Shape: [batch_size, 80, time_steps]

Description: 80-bin Mel-spectrogram.

2. Create Mock Encoders (encoders.py)

Create dummy functions/modules that will be replaced by the real models (HuBERT/Speaker Encoder) in the future:

MockContentEncoder: Receives dummy tensor and returns shape [batch_size, time_steps, 768].

MockSpeakerEncoder: Receives dummy tensor and returns shape [batch_size, 256].

MockVocoder: Receives mel-spectrogram [batch_size, 80, time_steps] and returns 1D waveform [batch_size, 1, time_steps * 256] (assuming hop_length = 256).

3. GAN Network Architecture (models.py)

Design the main PyTorch model architecture with the following structure:

Generator (G):

Input: content_features and speaker_embedding.

Mechanism: Use Adaptive Instance Normalization (AdaIN) to inject the speaker_embedding into the convolutional/residual layers processing the content_features.

Output: predicted_mel_spectrogram with dimensions [batch_size, 80, time_steps].

Discriminator (D):

Architecture: Use a PatchGAN style (Convolutional 2D/1D).

Input: Mel-spectrogram [batch_size, 80, time_steps].

Output: Probability/validity tensor (Real vs Fake).

4. Training Loop & Loss Function (train.py)

Create a standard PyTorch training script that combines all the components above.

Use separate optimizers for the Generator and Discriminator (e.g., AdamW).

Implement Loss calculation functions:

Discriminator Loss: Adversarial loss (Hinge or LSGAN) to distinguish real vs. predicted target mel.

Generator Loss: Adversarial loss (to fool D), Mel-Reconstruction Loss (L1 distance between prediction and target mel), and optional Feature Matching Loss.

Create a training loop for X epochs that prints the loss values (G_loss and D_loss) to the console every few steps.

The backward pass and optimizer step must run smoothly without Out of Memory or Shape Error.

Acceptance Criteria

Your work is considered complete and successful IF you can run the command python train.py, and the script prints training logs showing the loss decreasing/changing for at least 10 consecutive epochs using random data, without any tensor dimension errors (RuntimeError: size mismatch).