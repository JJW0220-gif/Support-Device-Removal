"""Generate XOR labels for "Other Support Devices" from CheXlocalize.

Process:
1. Load CheXlocalize val images with Support Devices annotations
2. Run cxr-lines-and-tubes to get 4-class family predictions
3. Compute: Other_GT = SupportDevices_GT XOR Family_Union_Pred
4. Save visualizations + statistics
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

REPO_ROOT = Path(__file__).resolve().parent.parent
CXR_MODEL_ROOT = REPO_ROOT / "cxr-lines-and-tubes"
CHEX_ROOT = REPO_ROOT / "cheXlocalize_dataset" / "chexlocalize"
OUT_DIR = REPO_ROOT / "outputs" / "other_device_labels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CXR_MODEL_ROOT))

from local_lines_tubes_runtime import LocalLinesTubesRuntime  # noqa: E402

TASK = "Support Devices"
DILATE_KERNEL = 7  # dilate thin CheXlocalize lines to match cxr model's thick masks
N_VISUALIZE = 10


def find_image(split: str, name: str) -> Path | None:
    folder = CHEX_ROOT / "CheXpert" / split
    parts = name.split("_", 2)
    if len(parts) == 3:
        patient, study, view = parts
        candidate = folder / patient / study / f"{view}.jpg"
        if candidate.is_file():
            return candidate
    return None


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    runtime = LocalLinesTubesRuntime(CXR_MODEL_ROOT, device)

    with open(CHEX_ROOT / "CheXlocalize" / "gt_segmentations_val.json") as f:
        segs = json.load(f)
    with open(CHEX_ROOT / "CheXlocalize" / "gt_annotations_val.json") as f:
        anns = json.load(f)

    candidates = [k for k, v in anns.items() if TASK in v]
    print(f"Found {len(candidates)} images with '{TASK}' annotations")

    stats = []
    for idx, key in enumerate(candidates):
        img_path = find_image("val", key)
        if img_path is None:
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        # Ground truth: CheXlocalize Support Devices (thin lines -> dilate)
        rle = segs[key][TASK]
        rles = rle if isinstance(rle, list) else [rle]
        gt_support = np.zeros(image.shape, dtype=np.uint8)
        for r in rles:
            if isinstance(r, dict) and "counts" in r:
                gt_support |= mask_utils.decode(r)
        gt_support_dilated = dilate_mask(gt_support, DILATE_KERNEL)

        # Prediction: cxr-lines-and-tubes family (4 classes union)
        agnostic, family, raw_family, _ = runtime._device(image, count=1)
        # family: (8, 1024, 1024) -> lines 0-3, tips 4-7
        family_lines = family[:4]  # ETT, NGT, CVC, Swan lines
        family_union = np.max(family_lines, axis=0)  # (1024, 1024)
        family_union = cv2.resize(family_union, (image.shape[1], image.shape[0]))
        family_union_bin = (family_union > 0.5).astype(np.uint8)

        # XOR: Other = GT AND NOT Family
        other_mask = gt_support_dilated & (~family_union_bin.astype(bool)).astype(np.uint8)

        # Statistics
        gt_pixels = gt_support_dilated.sum()
        family_pixels = family_union_bin.sum()
        overlap_pixels = (gt_support_dilated & family_union_bin).sum()
        other_pixels = other_mask.sum()

        stat = {
            "key": key,
            "image_shape": list(image.shape),
            "gt_support_pixels": int(gt_pixels),
            "family_pred_pixels": int(family_pixels),
            "overlap": int(overlap_pixels),
            "other_pixels": int(other_pixels),
            "other_ratio": float(other_pixels / max(gt_pixels, 1)),
        }
        stats.append(stat)

        # Visualize first N samples
        if idx < N_VISUALIZE:
            h, w = image.shape
            panel = np.zeros((h, w * 4, 3), dtype=np.uint8)

            # Original
            panel[:, :w] = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # GT Support Devices (green)
            gt_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).copy()
            gt_vis[gt_support_dilated > 0] = [0, 255, 0]
            panel[:, w : w * 2] = gt_vis

            # Family Prediction (blue)
            family_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).copy()
            family_vis[family_union_bin > 0] = [255, 0, 0]
            panel[:, w * 2 : w * 3] = family_vis

            # Other = XOR (red)
            other_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).copy()
            other_vis[other_mask > 0] = [0, 0, 255]
            panel[:, w * 3 :] = other_vis

            # Downscale for viewing
            target_h = 800
            scale = target_h / h
            target_w = int(w * scale)
            panel_small = cv2.resize(panel, (target_w * 4, target_h))

            out_path = OUT_DIR / f"{idx:03d}_{key}_xor.png"
            cv2.imwrite(str(out_path), panel_small)

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(candidates)}")

    # Save stats
    stats_path = OUT_DIR / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    total_gt = sum(s["gt_support_pixels"] for s in stats)
    total_other = sum(s["other_pixels"] for s in stats)
    avg_ratio = np.mean([s["other_ratio"] for s in stats])

    print(f"\n=== Summary ===")
    print(f"Total images: {len(stats)}")
    print(f"Total GT pixels: {total_gt:,}")
    print(f"Total Other pixels: {total_other:,}")
    print(f"Average Other/GT ratio: {avg_ratio:.2%}")
    print(f"Visualizations saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
