"""Debug mask resize issue."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor

REPO_ROOT = Path(__file__).resolve().parent.parent

# Test with one sample
sample_dir = REPO_ROOT / "outputs" / "cxr_seg" / "train_patient00067_study2_view1_frontal"
mask_path = sample_dir / "mask.png"

print(f"Testing: {mask_path}")

# Method 1: PIL resize (current implementation)
mask_pil = Image.open(mask_path).convert("L")
print(f"\n1. Original (PIL): size={mask_pil.size}, mode={mask_pil.mode}")
mask_array = np.array(mask_pil)
print(f"   Unique values: {np.unique(mask_array)}")
print(f"   Positive pixels: {(mask_array > 127).sum()} / {mask_array.size} ({100*(mask_array > 127).mean():.2f}%)")

mask_pil_resized = mask_pil.resize((256, 256), Image.NEAREST)
mask_pil_resized_array = np.array(mask_pil_resized)
print(f"\n2. After PIL resize to 256x256:")
print(f"   Unique values: {np.unique(mask_pil_resized_array)}")
print(f"   Positive pixels: {(mask_pil_resized_array > 127).sum()} / {mask_pil_resized_array.size} ({100*(mask_pil_resized_array > 127).mean():.2f}%)")

mask_tensor = to_tensor(mask_pil_resized)
print(f"\n3. After to_tensor:")
print(f"   Shape: {mask_tensor.shape}")
print(f"   Range: [{mask_tensor.min():.4f}, {mask_tensor.max():.4f}]")
print(f"   Positive pixels (>0.5): {(mask_tensor > 0.5).sum().item()} / {mask_tensor.numel()}")

# Method 2: OpenCV resize (alternative)
mask_cv = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
print(f"\n4. Original (OpenCV): shape={mask_cv.shape}")
print(f"   Unique values: {np.unique(mask_cv)}")
print(f"   Positive pixels: {(mask_cv > 127).sum()} / {mask_cv.size} ({100*(mask_cv > 127).mean():.2f}%)")

mask_cv_resized = cv2.resize(mask_cv, (256, 256), interpolation=cv2.INTER_NEAREST)
print(f"\n5. After OpenCV resize to 256x256:")
print(f"   Unique values: {np.unique(mask_cv_resized)}")
print(f"   Positive pixels: {(mask_cv_resized > 127).sum()} / {mask_cv_resized.size} ({100*(mask_cv_resized > 127).mean():.2f}%)")

# Save for visual inspection
cv2.imwrite(str(REPO_ROOT / "outputs" / "debug_mask_original.png"), mask_cv)
cv2.imwrite(str(REPO_ROOT / "outputs" / "debug_mask_pil_resized.png"), mask_pil_resized_array)
cv2.imwrite(str(REPO_ROOT / "outputs" / "debug_mask_cv_resized.png"), mask_cv_resized)
print(f"\nSaved debug images to outputs/")
