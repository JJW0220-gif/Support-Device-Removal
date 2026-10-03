"""Test the trained Other Device decoder on a single image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "cxr-lines-and-tubes"))

from other_device_model import DeviceNetWithOther
from skp.configs.ranzcr_device_specific_distal_b1_unet_1024_swan10_ablation import (
    cfg as device_cfg,
)


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "outputs/other_decoder_training/best_model.pt")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/test_other")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.checkpoint}")

    import copy
    cfg = copy.deepcopy(device_cfg)
    cfg.pretrained = False
    cfg.load_pretrained_model = None

    model = DeviceNetWithOther(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Load full state dict (includes other_decoder and other_head)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(device)
    model.eval()

    print(f"Loading image: {args.image}")
    image = load_grayscale(args.image)
    h, w = image.shape

    # Preprocess
    resized = cv2.resize(image, (1024, 1024))
    tensor = torch.from_numpy(resized).float()[None, None].to(device)  # (1, 1, 1024, 1024)

    # Inference
    with torch.no_grad():
        output = model({"x": tensor})
        prob = torch.sigmoid(output["other_logits"])[0, 0].cpu().numpy()  # (1024, 1024)
        prob = cv2.resize(prob, (w, h))
        mask = (prob > args.threshold).astype(np.uint8)

    # Stats
    print(f"\nPrediction stats:")
    print(f"  Prob range: [{prob.min():.4f}, {prob.max():.4f}]")
    print(f"  Prob mean: {prob.mean():.4f}")
    print(f"  Mask coverage (threshold={args.threshold}): {mask.mean():.2%}")

    # Save
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    # Save probability as grayscale (0-255)
    cv2.imwrite(str(args.out / f"{stem}_prob.png"), (prob * 255).astype(np.uint8))
    # Save binary mask
    cv2.imwrite(str(args.out / f"{stem}_mask.png"), mask * 255)

    # Overlay (use probability for smooth blending)
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
    color = np.zeros_like(overlay)
    color[..., 1] = 255  # green
    # Use probability as alpha for smooth overlay
    alpha = np.clip(prob, 0, 1)[..., None] * 0.65
    overlay = np.rint(overlay * (1 - alpha) + color * alpha).astype(np.uint8)
    cv2.imwrite(str(args.out / f"{stem}_overlay.png"), overlay[..., ::-1])

    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
