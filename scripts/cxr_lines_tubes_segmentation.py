"""Run CXR lines/tubes segmentation with the local ianpan/cxr-lines-and-tubes snapshot.

Outputs per-device line / distal-tip probability masks, the device-agnostic
union masks, the carina heatmap + landmark, and a color overlay for review.

Example:
    python scripts/cxr_lines_tubes_segmentation.py path/to/cxr.png \
        --out outputs/cxr_seg --device-folds 1 --carina-folds 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_ROOT = REPO_ROOT / "cxr-lines-and-tubes"
sys.path.insert(0, str(SNAPSHOT_ROOT))

import torch  # noqa: E402

from local_lines_tubes_runtime import (  # noqa: E402
    PRESENCE_NAMES,
    LocalLinesTubesRuntime,
    _device_overlay,
)


def load_grayscale(path: Path) -> np.ndarray:
    """Load any common image (or DICOM if pydicom is installed) as uint8 gray."""
    if path.suffix.lower() == ".dcm":
        import pydicom

        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        arr -= arr.min()
        arr /= max(arr.max(), 1e-6)
        return (arr * 255.0).round().astype(np.uint8)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def save_mask(probability: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.clip(probability, 0, 1) * 255).round().astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Grayscale chest radiograph")
    parser.add_argument("--out", type=Path, default=Path("outputs/cxr_seg"))
    parser.add_argument("--device-folds", type=int, default=1, choices=range(0, 6))
    parser.add_argument("--carina-folds", type=int, default=1, choices=range(0, 6))
    parser.add_argument("--presence-folds", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--gpu", action="store_true", help="Use CUDA if available")
    args = parser.parse_args()

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    image = load_grayscale(args.image)
    runtime = LocalLinesTubesRuntime(SNAPSHOT_ROOT, device)

    presence, presence_folds = runtime._presence(image, args.presence_folds)
    print("Device presence (raw scores):", json.dumps(presence, indent=2))

    # Create subdirectory based on input image filename
    image_stem = args.image.stem  # filename without extension
    output_dir = args.out / image_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy original image to output folder
    cv2.imwrite(str(output_dir / "original.png"), image)

    result = {"presence": presence, "image": str(args.image)}

    if args.device_folds:
        # agnostic: (2, 1024, 1024) -> [line union, distal-tip union]
        # family / raw_family: (8, 1024, 1024) -> lines 0-3, tips 4-7
        #   channel order: ETT, NGT, CVC, Swan Ganz
        agnostic, family, raw_family, device_folds = runtime._device(
            image, args.device_folds
        )
        save_mask(agnostic[0], output_dir / "agnostic_line.png")
        save_mask(agnostic[1], output_dir / "agnostic_tip.png")

        # Binary mask with threshold 0.1 (at 1024x1024)
        binary_mask = (agnostic[0] > 0.1).astype(np.uint8) * 255

        # Resize mask to original image size for saving
        mask_resized = cv2.resize(binary_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(output_dir / "mask.png"), mask_resized)

        # Original image with mask removed (inpainted effect)
        image_with_mask_removed = image.copy()
        image_with_mask_removed[mask_resized > 0] = 0  # Set masked pixels to black
        cv2.imwrite(str(output_dir / "now.png"), image_with_mask_removed)

        # === For inpainting: JPEG conversion + resize to 256x256 ===
        import io
        from PIL import Image as PILImage

        # Process original image
        pil_img = PILImage.fromarray(image)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        img_jpeg = np.array(PILImage.open(buffer).convert('L'))
        img_256 = cv2.resize(img_jpeg, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(output_dir / "original_256.png"), img_256)

        # Process mask (same JPEG conversion + resize)
        pil_mask = PILImage.fromarray(mask_resized)
        buffer_mask = io.BytesIO()
        pil_mask.save(buffer_mask, format='JPEG', quality=95)
        buffer_mask.seek(0)
        mask_jpeg = np.array(PILImage.open(buffer_mask).convert('L'))
        mask_256 = cv2.resize(mask_jpeg, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(output_dir / "mask_256.png"), mask_256)

        for index, name in enumerate(PRESENCE_NAMES):
            tag = name.lower().replace(" ", "_")
            save_mask(raw_family[index], output_dir / f"{tag}_line.png")
            save_mask(raw_family[index + 4], output_dir / f"{tag}_tip.png")

        overlay = _device_overlay(image, family, presence)
        cv2.imwrite(str(output_dir / "overlay_devices.png"), overlay[..., ::-1])
        result["device_folds"] = device_folds

    if args.carina_folds:
        heatmap, carina_xy, carina_folds = runtime._carina(image, args.carina_folds)
        save_mask(heatmap, output_dir / "carina_heatmap.png")
        result["carina_xy"] = carina_xy
        result["carina_folds"] = carina_folds
        print(f"Carina (x, y) in original pixels: {carina_xy}")

    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"Saved masks and overlay to {output_dir.resolve()}")
    print(f"Saved masks and overlay to {args.out.resolve()}")


if __name__ == "__main__":
    main()
