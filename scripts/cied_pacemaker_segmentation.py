"""Pacemaker segmentation using CIED model.

Download models from Zenodo and run segmentation on CXR images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
CIED_ROOT = REPO_ROOT / "cied"
MODEL_DIR = CIED_ROOT / "models"

# Model URLs from Zenodo
ZENODO_BASE = "https://zenodo.org/record/10955502/files"
MODELS = {
    "segmentation.pkl": f"{ZENODO_BASE}/segmentation.pkl",
    "classification_manuf.pkl": f"{ZENODO_BASE}/classification_manuf.pkl",
    "classification_model.pkl": f"{ZENODO_BASE}/classification_model.pkl",
}


def download_models():
    """Download model files from Zenodo if not present."""
    import urllib.request

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in MODELS.items():
        filepath = MODEL_DIR / filename
        if filepath.exists():
            print(f"✓ {filename} already exists")
            continue

        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            return False

    return True


def resize_img(img: Image.Image, small_ax: int = 1024) -> Image.Image:
    """Resize preserving aspect ratio, smallest dimension = small_ax."""
    scale_f = small_ax / min(img.size)
    new_size = (
        int(np.floor(img.size[0] * scale_f)),
        int(np.floor(img.size[1] * scale_f)),
    )
    return img.resize(new_size)


def center_crop(img: Image.Image, size: int = 1024) -> Image.Image:
    """Center crop to square."""
    width, height = img.size
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2
    return img.crop((int(left), int(top), int(right), int(bottom)))


def segment_pacemaker(model, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Run segmentation and return mask + bbox."""
    from fastai.vision.all import load_learner
    import skimage

    # Preprocess
    img = ImageOps.exif_transpose(image)
    img_resized = resize_img(img, small_ax=1024)
    img_cropped = center_crop(img_resized, size=1024)
    img_array = np.array(img_cropped)

    # Predict
    res = model.predict(img_array, with_input=True)
    mask = np.array(res[1])
    mask[mask != 1] = 0  # Keep only pacemaker class

    # Find largest connected component
    labeled_image, count = skimage.measure.label(mask, return_num=True)
    objects = skimage.measure.regionprops(labeled_image)

    if len(objects) == 0:
        return mask, None

    # Get largest object
    object_areas = [obj["area"] for obj in objects]
    max_idx = np.argmax(object_areas)
    obj = objects[max_idx]
    bbox = obj["bbox"]  # (min_row, min_col, max_row, max_col)

    return mask, bbox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input CXR image")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "cied_seg")
    parser.add_argument("--download-models", action="store_true", help="Download models from Zenodo")
    args = parser.parse_args()

    # Download models if requested
    if args.download_models:
        print("=== Downloading CIED models from Zenodo ===")
        if not download_models():
            print("Failed to download models. Please download manually from:")
            print("https://zenodo.org/record/10955502")
            return

    # Check if models exist
    seg_model_path = MODEL_DIR / "segmentation.pkl"
    if not seg_model_path.exists():
        print(f"Error: {seg_model_path} not found.")
        print("Run with --download-models to download from Zenodo.")
        return

    # Load model
    print(f"Loading segmentation model from {seg_model_path}...")
    from fastai.vision.all import load_learner
    model_seg = load_learner(seg_model_path)

    # Load image
    print(f"Loading image: {args.image}")
    image = Image.open(args.image)

    # Segment
    print("Running segmentation...")
    mask, bbox = segment_pacemaker(model_seg, image)

    # Create subdirectory based on input image filename (like cxr_seg)
    image_stem = args.image.stem  # filename without extension
    output_dir = args.out / image_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy original image
    img_array = np.array(image.convert("L"))
    cv2.imwrite(str(output_dir / "original.png"), img_array)

    # Save mask (original size)
    cv2.imwrite(str(output_dir / "mask.png"), mask * 255)

    # === Generate 256x256 versions for inpainting ===
    # Resize original to 256x256
    img_256 = cv2.resize(img_array, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(output_dir / "original_256.png"), img_256)

    # Resize mask to 256x256
    mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(output_dir / "mask_256.png"), mask_256 * 255)

    # Save overlay
    if len(img_array.shape) == 2:
        overlay = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        overlay = img_array.copy()

    # Resize mask to match original image size
    mask_resized = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay[mask_resized > 0] = [255, 0, 0]  # Red for pacemaker

    cv2.imwrite(str(output_dir / "overlay.png"), overlay[..., ::-1])

    # Save metadata
    import json
    metadata = {
        "image": str(args.image),
        "pacemaker_detected": bbox is not None,
    }
    if bbox is not None:
        minr, minc, maxr, maxc = bbox
        metadata["bbox"] = {"min_row": int(minr), "min_col": int(minc), "max_row": int(maxr), "max_col": int(maxc)}
        metadata["size"] = {"height": int(maxr - minr), "width": int(maxc - minc)}
        metadata["area_pixels"] = int(mask.sum())
        metadata["area_percent"] = float(100 * mask.mean())

    with open(output_dir / "result.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Print results
    print(f"\n=== Results ===")
    if bbox is not None:
        minr, minc, maxr, maxc = bbox
        print(f"Pacemaker detected!")
        print(f"  Bounding box: ({minr}, {minc}) to ({maxr}, {maxc})")
        print(f"  Size: {maxr - minr} x {maxc - minc} pixels")
        print(f"  Area: {mask.sum()} pixels ({100 * mask.mean():.2f}%)")
    else:
        print("No pacemaker detected.")

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
