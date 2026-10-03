"""Prepare images and masks for Cheff inpainting notebook.

Converts cxr_seg outputs (original size) to 256x256 format expected by notebook.
"""

from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
CXR_SEG_DIR = REPO_ROOT / "outputs" / "cxr_seg"
OUTPUT_DIR = REPO_ROOT / "outputs" / "inpainting_ready"
IMG_SIZE = 256


def prepare_single(image_folder: Path, output_dir: Path) -> None:
    """Convert one sample to 256x256."""
    img_path = image_folder / "original.png"
    mask_path = image_folder / "mask.png"

    if not img_path.exists() or not mask_path.exists():
        print(f"  [skip] Missing files in {image_folder.name}")
        return

    # Load
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"  [skip] Cannot read files in {image_folder.name}")
        return

    # Resize to 256x256
    img_256 = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    mask_256 = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "image.png"), img_256)
    cv2.imwrite(str(output_dir / "mask.png"), mask_256)

    print(f"  ✓ {image_folder.name} -> {output_dir.name}")


def main():
    if not CXR_SEG_DIR.exists():
        print(f"Error: {CXR_SEG_DIR} not found. Run cxr_lines_tubes_segmentation.py first.")
        return

    # Process all subdirectories in cxr_seg
    subdirs = [d for d in CXR_SEG_DIR.iterdir() if d.is_dir()]

    if not subdirs:
        print(f"Error: No subdirectories found in {CXR_SEG_DIR}")
        return

    print(f"Found {len(subdirs)} samples in {CXR_SEG_DIR}")
    print(f"Converting to {IMG_SIZE}x{IMG_SIZE} and saving to {OUTPUT_DIR}\n")

    for image_folder in subdirs:
        output_subdir = OUTPUT_DIR / image_folder.name
        prepare_single(image_folder, output_subdir)

    print(f"\nDone! Prepared data saved to: {OUTPUT_DIR}")
    print(f"\nTo use in notebook, update paths:")
    print(f"  img_path = '{OUTPUT_DIR.relative_to(REPO_ROOT)}/SAMPLE_NAME/image.png'")
    print(f"  seg_mask_path = '{OUTPUT_DIR.relative_to(REPO_ROOT)}/SAMPLE_NAME/mask.png'")


if __name__ == "__main__":
    main()
