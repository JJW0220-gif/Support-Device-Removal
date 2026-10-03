"""Check environment for cxr-lines-and-tubes inference."""
import sys

print("Python:", sys.version)

try:
    import torch
    print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("torch: NOT INSTALLED")

for pkg in ["timm", "cv2", "numpy", "pycocotools", "einops"]:
    try:
        mod = __import__(pkg)
        version = getattr(mod, "__version__", "unknown")
        print(f"{pkg}: {version}")
    except ImportError:
        print(f"{pkg}: NOT INSTALLED")
