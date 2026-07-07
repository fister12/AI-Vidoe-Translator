import torchaudio
print("Torchaudio version:", torchaudio.__version__)

try:
    print("Available backends:", torchaudio.list_audio_backends())
except AttributeError:
    print("torchaudio.list_audio_backends() not available.")

# Let's try loading with soundfile directly or using a backend setter if available
try:
    import torchaudio.backend.soundfile_backend
    print("soundfile_backend is importable.")
except ImportError:
    print("soundfile_backend is NOT importable.")

import torch
print("Trying to load audio with torchaudio.load...")
try:
    info = torchaudio.info("temp/speaker_sample.wav")
    print("Info success:", info)
    waveform, sample_rate = torchaudio.load("temp/speaker_sample.wav")
    print("Load success. Shape:", waveform.shape, "Sample rate:", sample_rate)
except Exception as e:
    print("Load failed:", e)

# Let's see if we can mock torchaudio.load to use soundfile / librosa
print("Testing custom soundfile load fallback...")
import soundfile as sf
def load_fallback(filepath):
    data, sr = sf.read(filepath)
    # convert to channel-first tensor
    if len(data.shape) == 1:
        tensor = torch.FloatTensor(data).unsqueeze(0)
    else:
        tensor = torch.FloatTensor(data.T)
    return tensor, sr

try:
    w, sr = load_fallback("temp/speaker_sample.wav")
    print("Fallback success. Shape:", w.shape, "Sample rate:", sr)
except Exception as e:
    print("Fallback failed:", e)
