import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

try:
    import sklearn
    print("scikit-learn is available.")
except ImportError:
    print("scikit-learn is NOT available.")

try:
    import scipy
    print("scipy is available.")
except ImportError:
    print("scipy is NOT available.")

try:
    import torch
    print("torch is available. Version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
except ImportError:
    print("torch is NOT available.")

try:
    import TTS
    print("TTS is available.")
except ImportError:
    print("TTS is NOT available.")
