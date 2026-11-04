import torch
from TTS.api import TTS
import os

# os.add_dll_directory(r"C:/Program Files/ffmpeg/bin")

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Initialize TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# List speakers
print(tts.speakers)

# Run TTS
# ❗ XTTS supports both, but many models allow only one of the `speaker` and
# `speaker_wav` arguments

# TTS with list of amplitude values as output, clone the voice from `speaker_wav`
# wav = tts.tts(
#   text="Hello world!",
#   speaker_wav="my/cloning/audio.wav",
#   language="en"
# )

# TTS to a file, use a preset speaker
tts.tts_to_file(
  text="Hello world! The quick brown fox jumps over the lazy dog.",
  speaker_wav="sample.mp3",
  language="en",
  file_path="output-2.wav"
)
