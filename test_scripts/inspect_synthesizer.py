from TTS.api import TTS
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
print("Type of tts.synthesizer:", type(tts.synthesizer))
print("Attributes of tts.synthesizer:")
for attr in dir(tts.synthesizer):
    if not attr.startswith("_"):
        print(f"  - {attr}")

if hasattr(tts.synthesizer, "tts_model"):
    tts_model = tts.synthesizer.tts_model
    print("Type of tts.synthesizer.tts_model:", type(tts_model))
    print("Attributes of tts.synthesizer.tts_model:")
    for attr in dir(tts_model):
        if not attr.startswith("_"):
            print(f"  - {attr}")
