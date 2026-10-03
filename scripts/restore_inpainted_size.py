"""Restore inpainted_0 images to their combined_seg original image size."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inpainted-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "inpainted",
        help="Directory containing inpainted_0.png outputs",
    )
    parser.add_argument(
        "--combined-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "combined_seg",
        help="Directory containing original.png images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "result",
        help="Directory for restored full-size images",
    )
    args = parser.parse_args()

    restored = 0
    for sample_dir in sorted(path for path in args.inpainted_dir.iterdir() if path.is_dir()):
        source_path = sample_dir / "inpainted_0.png"
        original_path = args.combined_dir / sample_dir.name / "original.png"
        if not source_path.is_file() or not original_path.is_file():
            print(f"[skip] {sample_dir.name}: missing inpainted_0.png or original.png")
            continue

        inpainted = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        original = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
        if inpainted is None or original is None:
            print(f"[skip] {sample_dir.name}: unable to read an image")
            continue

        height, width = original.shape
        restored_image = cv2.resize(
            inpainted,
            (width, height),
            interpolation=cv2.INTER_LANCZOS4,
        )
        output_path = args.output_dir / sample_dir.name / "inpainted_0.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), restored_image)
        restored += 1
        print(f"[saved] {output_path}")

    print(f"Restored {restored} image(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
