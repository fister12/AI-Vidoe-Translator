import sys
from pathlib import Path

# Add DLL search paths before importing anything else
try:
    from src.tts import _prepare_windows_dll_search_paths
    _prepare_windows_dll_search_paths()
    print("DLL search paths prepared.")
except Exception as e:
    print("Failed to prepare DLL search paths:", e)

from TTS.api import TTS
import inspect
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
model = tts.synthesizer.tts_model

print("Signature of get_conditioning_latents:")
print(inspect.signature(model.get_conditioning_latents))

audio_path = "temp/speaker_sample.wav"
try:
    res = model.get_conditioning_latents(audio_path=[audio_path])
    print("Returned type:", type(res))
    if isinstance(res, tuple):
        print("Tuple length:", len(res))
        for i, val in enumerate(res):
            if hasattr(val, "shape"):
                print(f"  - Element {i} shape: {val.shape}, type: {type(val)}")
            else:
                print(f"  - Element {i} type: {type(val)}")
except Exception as exc:
    print(f"Failed to get_conditioning_latents with audio_path list: {exc}")
