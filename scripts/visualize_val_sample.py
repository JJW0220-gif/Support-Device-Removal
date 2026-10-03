"""Visualize a validation sample (image + mask)."""

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from other_device_dataset import CheXlocalizeOtherDataset  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "outputs" / "val_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Load validation dataset (no augmentation)
    dataset = CheXlocalizeOtherDataset(split="val", augment=False)

    # Get first sample
    idx = 0
    sample = dataset[idx]

    # Convert tensors to numpy
    image = sample["x"][0].numpy().astype(np.uint8)  # (1024, 1024)
    mask = sample["other_mask"][0].numpy().astype(np.uint8)  # (1024, 1024)
    key = sample["key"]

    print(f"Sample: {key}")
    print(f"Image shape: {image.shape}, range: [{image.min()}, {image.max()}]")
    print(f"Mask shape: {mask.shape}, positive pixels: {mask.sum()} ({100*mask.mean():.2f}%)")

    # Create side-by-side visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    mask_rgb = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2RGB)

    # Overlay
    overlay = image_rgb.copy()
    overlay[mask > 0] = [0, 255, 0]  # Green for mask

    # Combine: image | mask | overlay
    panel = np.concatenate([image_rgb, mask_rgb, overlay], axis=1)

    # Save
    out_path = OUTPUT_DIR / f"{idx:03d}_{key}_sample.png"
    cv2.imwrite(str(out_path), panel)
    print(f"Saved to: {out_path}")

    # Also save individual files
    cv2.imwrite(str(OUTPUT_DIR / f"{idx:03d}_{key}_image.png"), image)
    cv2.imwrite(str(OUTPUT_DIR / f"{idx:03d}_{key}_mask.png"), mask * 255)
    cv2.imwrite(str(OUTPUT_DIR / f"{idx:03d}_{key}_overlay.png"), overlay)


if __name__ == "__main__":
    main()
