"""Debug XOR label quality."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

REPO_ROOT = Path(__file__).resolve().parent.parent
CHEX_ROOT = REPO_ROOT / "cheXlocalize_dataset" / "chexlocalize"
CACHE_DIR = REPO_ROOT / "outputs" / "family_pred_cache"

TASK = "Support Devices"
DILATE_KERNEL = 7

# Load annotations
with open(CHEX_ROOT / "CheXlocalize" / "gt_segmentations_val.json") as f:
    segs = json.load(f)
with open(CHEX_ROOT / "CheXlocalize" / "gt_annotations_val.json") as f:
    anns = json.load(f)

keys = [k for k, v in anns.items() if TASK in v][:5]

for key in keys:
    # Find image
    parts = key.split("_", 2)
    img_path = CHEX_ROOT / "CheXpert" / "val" / parts[0] / parts[1] / f"{parts[2]}.jpg"
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    # GT
    rle = segs[key][TASK]
    rles = rle if isinstance(rle, list) else [rle]
    gt = np.zeros(image.shape, dtype=np.uint8)
    for r in rles:
        if isinstance(r, dict) and "counts" in r:
            gt |= mask_utils.decode(r)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL, DILATE_KERNEL))
    gt_dilated = cv2.dilate(gt, kernel, iterations=1)

    # Family pred
    cache_path = CACHE_DIR / f"val_{key}_family.npy"
    family_pred = np.load(cache_path)
    family_resized = cv2.resize(family_pred, (image.shape[1], image.shape[0]))
    family_bin = (family_resized > 0.5).astype(np.uint8)

    # XOR
    other = gt_dilated & (~family_bin.astype(bool)).astype(np.uint8)

    print(f"\n{key}:")
    print(f"  Image shape: {image.shape}")
    print(f"  GT pixels: {gt_dilated.sum():,} ({100*gt_dilated.mean():.2f}%)")
    print(f"  Family pixels: {family_bin.sum():,} ({100*family_bin.mean():.2f}%)")
    print(f"  Overlap: {(gt_dilated & family_bin).sum():,}")
    print(f"  Other (XOR): {other.sum():,} ({100*other.mean():.2f}%)")
    print(f"  Other/GT ratio: {other.sum() / max(gt_dilated.sum(), 1):.2%}")

    # Visualize
    h, w = image.shape
    vis = np.zeros((h, w * 3, 3), dtype=np.uint8)
    vis[:, :w] = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    vis[:, w:w*2] = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    vis[:, w*2:] = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    vis[gt_dilated > 0] = [0, 255, 0]  # Green: GT
    vis[family_bin > 0, 1] = 255  # Blue channel in middle panel
    vis[other > 0, 2] = 255  # Red in right panel

    scale = 800 / h
    vis_small = cv2.resize(vis, (int(w * 3 * scale), 800))
    out_path = REPO_ROOT / "outputs" / f"debug_{key}.png"
    cv2.imwrite(str(out_path), vis_small)
    print(f"  Saved: {out_path.name}")
