"""Combine CIED pacemaker masks and CXR lines/tubes masks.

Merges masks from both segmentation approaches and copies to a new directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def combine_single(cied_dir: Path, cxr_dir: Path, output_dir: Path, image_name: str) -> None:
    """Combine masks for a single image."""
    cied_mask_path = cied_dir / image_name / "mask.png"
    cxr_mask_path = cxr_dir / image_name / "mask.png"
    original_path = cxr_dir / image_name / "original.png"

    # Check if all required files exist
    if not cied_mask_path.exists():
        print(f"  [skip] {image_name}: CIED mask not found")
        return
    if not cxr_mask_path.exists():
        print(f"  [skip] {image_name}: CXR mask not found")
        return
    if not original_path.exists():
        print(f"  [skip] {image_name}: Original image not found")
        return

    # Load masks
    cied_mask = cv2.imread(str(cied_mask_path), cv2.IMREAD_GRAYSCALE)
    cxr_mask = cv2.imread(str(cxr_mask_path), cv2.IMREAD_GRAYSCALE)
    original = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)

    # Ensure same size
    if cied_mask.shape != cxr_mask.shape:
        cied_mask = cv2.resize(cied_mask, (cxr_mask.shape[1], cxr_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Combine masks (union: any white pixel becomes white)
    combined_mask = np.maximum(cied_mask, cxr_mask)

    # Create output directory
    output_subdir = output_dir / image_name
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Save files
    cv2.imwrite(str(output_subdir / "original.png"), original)
    cv2.imwrite(str(output_subdir / "mask_cied.png"), cied_mask)
    cv2.imwrite(str(output_subdir / "mask_cxr.png"), cxr_mask)
    cv2.imwrite(str(output_subdir / "mask.png"), combined_mask)

    # Create overlay
    overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    overlay[cied_mask > 0] = [255, 0, 0]  # Red for CIED (pacemaker)
    overlay[cxr_mask > 0] = [0, 255, 0]   # Green for CXR (tubes/lines)
    # Overlap becomes yellow
    overlap = (cied_mask > 0) & (cxr_mask > 0)
    overlay[overlap] = [255, 255, 0]
    cv2.imwrite(str(output_subdir / "overlay.png"), overlay[..., ::-1])

    # Generate 256x256 versions
    original_256 = cv2.resize(original, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    mask_256 = cv2.resize(combined_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(output_subdir / "original_256.png"), original_256)
    cv2.imwrite(str(output_subdir / "mask_256.png"), mask_256)

    # Save metadata
    metadata = {
        "image": image_name,
        "cied_pixels": int(cied_mask.sum() / 255),
        "cxr_pixels": int(cxr_mask.sum() / 255),
        "combined_pixels": int(combined_mask.sum() / 255),
        "overlap_pixels": int(overlap.sum()),
    }
    with open(output_subdir / "result.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ {image_name}: CIED={metadata['cied_pixels']}, CXR={metadata['cxr_pixels']}, Combined={metadata['combined_pixels']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cied-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "cied_seg",
        help="Directory with CIED segmentation results"
    )
    parser.add_argument(
        "--cxr-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "cxr_seg",
        help="Directory with CXR segmentation results"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "combined_seg",
        help="Output directory for combined results"
    )
    args = parser.parse_args()

    # Check directories
    if not args.cied_dir.exists():
        print(f"Error: {args.cied_dir} not found. Run cied_pacemaker_segmentation.py first.")
        return
    if not args.cxr_dir.exists():
        print(f"Error: {args.cxr_dir} not found. Run cxr_lines_tubes_segmentation.py first.")
        return

    # Find all images that have both CIED and CXR results
    cied_samples = {d.name for d in args.cied_dir.iterdir() if d.is_dir()}
    cxr_samples = {d.name for d in args.cxr_dir.iterdir() if d.is_dir()}
    common_samples = cied_samples & cxr_samples

    if not common_samples:
        print("Error: No common images found in both directories.")
        print(f"CIED samples: {len(cied_samples)}")
        print(f"CXR samples: {len(cxr_samples)}")
        return

    print(f"Found {len(common_samples)} images with both CIED and CXR results")
    print(f"Combining masks...\n")

    # Process each image
    for image_name in sorted(common_samples):
        combine_single(args.cied_dir, args.cxr_dir, args.output_dir, image_name)

    print(f"\n{'='*60}")
    print(f"Done! Combined results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
