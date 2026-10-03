"""Batch inpainting script for removing support devices from CXR images.

Processes all images in outputs/inpainting_ready/ using Cheff LDM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import to_pil_image, to_tensor, resize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cheff import CheffLDM, CheffLDMT2I  # noqa: E402

IMG_SIZE = 256
LATENT_SIZE = 64
DILATION = 5


def prepare_mask(mask_path: Path, device: str) -> torch.Tensor:
    """Load 256x256 mask and convert to 64x64 latent-space mask."""
    mask = Image.open(mask_path).convert("L")
    mask_tensor = to_tensor(mask)
    mask_tensor = (mask_tensor > 0.5).float()

    # Dilate to cover thin lines
    pad = DILATION // 2
    mask_tensor = F.max_pool2d(
        mask_tensor.unsqueeze(0),
        kernel_size=DILATION,
        stride=1,
        padding=pad
    ).squeeze(0)
    mask_tensor = (mask_tensor > 0.0).float()

    # Resize to latent space
    mask_64 = resize(mask_tensor, [LATENT_SIZE, LATENT_SIZE], interpolation=InterpolationMode.NEAREST)
    mask_64 = (mask_64 > 0.5).float()

    # Cheff: 1 = keep original, 0 = regenerate
    inpaint_mask = 1.0 - mask_64
    return inpaint_mask.to(device)


def inpaint_single(
    cheff_ldm: CheffLDM,
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    device: str,
    n_samples: int = 4,
    sampling_steps: int = 100,
    prompt: str = "",
) -> None:
    """Run inpainting on a single image."""
    # Load pre-processed 256x256 image
    img = Image.open(image_path).convert("L")
    img_tensor = to_tensor(img).to(device)
    img_tensor = img_tensor * 2 - 1  # Normalize to [-1, 1]

    if img_tensor.shape[0] == 1:
        img_tensor = img_tensor.expand(3, -1, -1)

    # Prepare mask (already 256x256)
    inpaint_mask = prepare_mask(mask_path, device)

    # Check if text-guided
    is_t2i = isinstance(cheff_ldm, CheffLDMT2I)
    if is_t2i and not prompt:
        print("    ⚠️  Warning: Using T2I model but no prompt provided, using empty string")
        prompt = ""

    # Expand to batch
    imgs = img_tensor.unsqueeze(0).expand(n_samples, -1, -1, -1)
    masks = inpaint_mask.unsqueeze(0).expand(n_samples, -1, -1, -1)

    # Run inpainting
    print(f"    Running inpainting ({n_samples} samples, {sampling_steps} steps)...")
    if is_t2i:
        print(f"    Using prompt: '{prompt}'")
        # T2I now supports batch (modified CheffLDMT2I)
        imgs_edited = cheff_ldm.sample_inpaint(
            target_img=imgs,
            mask=masks,
            sampling_steps=sampling_steps,
            conditioning=prompt,
        )
    else:
        # Unconditional supports batch
        imgs_edited = cheff_ldm.sample_inpaint(
            target_img=imgs,
            mask=masks,
            sampling_steps=sampling_steps,
        )

    # Post-process
    imgs_edited.clamp_(-1, 1)
    imgs_edited = (imgs_edited + 1) / 2  # Back to [0, 1]

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resized original (256x256, used for inpainting)
    original_np = (img_tensor[0].cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / "original.png"), original_np)

    # Save resized mask (256x256, used for inpainting)
    mask_np = (1.0 - inpaint_mask[0].cpu().numpy() * 255).astype(np.uint8)
    mask_np = cv2.resize(mask_np, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(output_dir / "mask.png"), mask_np)

    # Also save original size versions for reference
    original_full = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask_full = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(output_dir / "original_fullsize.png"), original_full)
    cv2.imwrite(str(output_dir / "mask_fullsize.png"), mask_full)

    # Save all samples
    for i in range(n_samples):
        edited_np = (imgs_edited[i, 0].cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"inpainted_{i}.png"), edited_np)

    # Save grid comparison
    from torchvision.utils import make_grid
    comparison = torch.cat([
        (img_tensor.unsqueeze(0).cpu() + 1) / 2,
        imgs_edited.cpu()
    ], dim=0)
    grid = make_grid(comparison, nrow=n_samples + 1, padding=2)
    grid_np = (grid[0].numpy() * 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / "comparison.png"), grid_np)

    print(f"    ✓ Saved to {output_dir.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
            default=REPO_ROOT / "outputs" / "combined_seg",
            help="Directory with combined CIED and CXR segmentation outputs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "inpainted",
        help="Output directory for inpainted results"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=4,
        help="Number of inpainting samples per image"
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=100,
        help="Diffusion sampling steps"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--text-guided",
        action="store_true",
        help="Use text-guided model (cheff_diff_t2i.pt)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt for T2I model (e.g., 'chest x-ray without medical devices')"
    )
    args = parser.parse_args()

    # Check input directory
    if not args.input_dir.exists():
        print(f"Error: {args.input_dir} not found.")
        print("Run scripts/combine_masks.py first.")
        return

    # Find all samples
    sample_dirs = [d for d in args.input_dir.iterdir() if d.is_dir()]
    if not sample_dirs:
        print(f"Error: No sample directories found in {args.input_dir}")
        return

    print(f"Found {len(sample_dirs)} samples")

    # Load model
    if args.text_guided:
        print(f"Loading Cheff LDM (Text-Guided)...")
        sdm_path = REPO_ROOT / "trained_models" / "cheff_diff_t2i.pt"
        ae_path = REPO_ROOT / "trained_models" / "cheff_autoencoder.pt"
        cheff_ldm = CheffLDMT2I(model_path=str(sdm_path), ae_path=str(ae_path), device=args.device)
        print(f"Text-guided mode enabled")
    else:
        print(f"Loading Cheff LDM (Unconditional)...")
        sdm_path = REPO_ROOT / "trained_models" / "cheff_diff_uncond.pt"
        ae_path = REPO_ROOT / "trained_models" / "cheff_autoencoder.pt"
        cheff_ldm = CheffLDM(model_path=str(sdm_path), ae_path=str(ae_path), device=args.device)
        print(f"Unconditional mode")

    print(f"Processing {len(sample_dirs)} images...\n")

    # Process each sample
    for i, sample_dir in enumerate(sample_dirs, 1):
        # Use pre-processed 256x256 versions (JPEG + resize)
        image_path = sample_dir / "original_256.png"
        mask_path = sample_dir / "mask_256.png"

        if not image_path.exists() or not mask_path.exists():
            print(f"[{i}/{len(sample_dirs)}] {sample_dir.name}: Missing original_256.png or mask_256.png, skipping")
            continue

        print(f"[{i}/{len(sample_dirs)}] {sample_dir.name}")
        output_subdir = args.output_dir / sample_dir.name

        try:
            inpaint_single(
                cheff_ldm,
                image_path,
                mask_path,
                output_subdir,
                args.device,
                args.n_samples,
                args.sampling_steps,
                args.prompt,
            )
        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Done! Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
