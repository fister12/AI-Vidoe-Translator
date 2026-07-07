import sys
from pathlib import Path
try:
    from TTS.api import TTS
    print("TTS imported successfully.")
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    print(f"Loading {model_name}...")
    tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
    print("Model loaded.")
    print("TTS class type:", type(tts))
    print("Attributes and methods of tts:")
    for attr in dir(tts):
        if not attr.startswith("_"):
            print(f"  - {attr}")
except Exception as exc:
    print(f"Error: {exc}")
