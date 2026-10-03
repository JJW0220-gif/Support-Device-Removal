"""Decode and visualize Support Devices masks from CheXlocalize."""

import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parent.parent / "cheXlocalize_dataset" / "chexlocalize"
ANN_ROOT = ROOT / "CheXlocalize"
IMG_ROOT = ROOT / "CheXpert"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "support_devices_vis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK = "Support Devices"
N_SAMPLES = 8


def find_image(split: str, name: str) -> Path | None:
    """CheXlocalize keys look like 'patient64622_study1_view1_frontal'.

    Actual layout is ``val/patient64622/study1/view1_frontal.jpg``.
    """
    folder = IMG_ROOT / split
    parts = name.split("_", 2)  # patient64622, study1, view1_frontal
    if len(parts) == 3:
        patient, study, view = parts
        for ext in (".jpg", ".png", ".jpeg"):
            candidate = folder / patient / study / f"{view}{ext}"
            if candidate.is_file():
                return candidate
    matches = list(folder.rglob(f"{parts[-1]}.*"))
    return matches[0] if matches else None


def main() -> None:
    with open(ANN_ROOT / "gt_segmentations_val.json") as f:
        segs = json.load(f)
    with open(ANN_ROOT / "gt_annotations_val.json") as f:
        anns = json.load(f)

    # pick images that actually have Support Devices annotations
    candidates = [k for k, v in anns.items() if TASK in v]
    print(f"{len(candidates)} val images have '{TASK}' annotations")
    candidates = candidates[:N_SAMPLES]

    for key in candidates:
        img_path = find_image("val", key)
        if img_path is None:
            print(f"  [skip] image not found: {key}")
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  [skip] cannot read: {img_path}")
            continue

        rle = segs[key][TASK]
        # gt masks may store multiple annotators; take list of RLEs or single RLE
        rles = rle if isinstance(rle, list) else [rle]
        mask = np.zeros(image.shape, dtype=np.uint8)
        for r in rles:
            if isinstance(r, dict) and "counts" in r:
                mask |= mask_utils.decode(r)

        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
        color = np.zeros_like(overlay)
        color[..., 1] = 255  # green
        alpha = (mask > 0)[..., None] * 0.6
        overlay = np.rint(overlay * (1 - alpha) + color * alpha).astype(np.uint8)
        # draw contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        out_path = OUT_DIR / f"{key}_support_devices.png"
        cv2.imwrite(str(out_path), overlay[..., ::-1])
        cv2.imwrite(str(OUT_DIR / f"{key}_mask.png"), mask * 255)
        cv2.imwrite(str(OUT_DIR / f"{key}_original.png"), image)

        # side-by-side: original | mask | overlay (downscaled for viewing)
        mask_rgb = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2RGB)
        h = 1024
        w = int(image.shape[1] * h / image.shape[0])
        panel = np.concatenate(
            [
                cv2.resize(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), (w, h)),
                cv2.resize(mask_rgb, (w, h)),
                cv2.resize(overlay[..., ::-1], (w, h)),
            ],
            axis=1,
        )
        cv2.imwrite(str(OUT_DIR / f"{key}_panel.png"), panel)
        coverage = 100.0 * mask.mean()
        print(f"  saved {out_path.name}  (mask coverage: {coverage:.2f}%)")

    print(f"\nOutput -> {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
