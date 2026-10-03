"""Train Other Devices decoder using cxr-lines-and-tubes methodology.

Follows the exact training procedure from cxr-lines-and-tubes:
- BCE + low-weight Dice loss
- Direct resize to 1024x1024
- Frozen encoder, trainable decoder
- AdamW optimizer with cosine annealing
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
CXR_MODEL_ROOT = REPO_ROOT / "cxr-lines-and-tubes"
CHEX_ROOT = REPO_ROOT / "cheXlocalize_dataset" / "chexlocalize"
sys.path.insert(0, str(CXR_MODEL_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from other_device_model import DeviceNetWithOther  # noqa: E402
from skp.configs.ranzcr_device_specific_distal_b1_unet_1024_swan10_ablation import (  # noqa: E402
    cfg as device_cfg,
)

TASK = "Support Devices"
DILATE_KERNEL = 15
IMG_SIZE = 1024


class CheXlocalizeDataset(Dataset):
    """CheXlocalize dataset with preprocessing matching cxr-lines-and-tubes."""

    def __init__(self, split: str = "val", augment: bool = False):
        self.split = split
        self.augment = augment

        # Load annotations
        seg_file = CHEX_ROOT / "CheXlocalize" / f"gt_segmentations_{split}.json"
        ann_file = CHEX_ROOT / "CheXlocalize" / f"gt_annotations_{split}.json"

        with open(seg_file) as f:
            self.segs = json.load(f)
        with open(ann_file) as f:
            self.anns = json.load(f)

        # Filter images with Support Devices annotations
        self.keys = [k for k, v in self.anns.items() if TASK in v]
        print(f"Loaded {len(self.keys)} images from {split} (augment={augment})")

        # Augmentation pipeline (similar to RANZCR training)
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=10,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5,
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            ])
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.keys)

    def _find_image(self, key: str) -> Path | None:
        folder = CHEX_ROOT / "CheXpert" / self.split
        parts = key.split("_", 2)
        if len(parts) == 3:
            patient, study, view = parts
            candidate = folder / patient / study / f"{view}.jpg"
            if candidate.is_file():
                return candidate
        return None

    def _load_mask(self, key: str, target_size: int) -> np.ndarray:
        """Load, dilate, and resize mask to target size."""
        rle = self.segs[key][TASK]
        rles = rle if isinstance(rle, list) else [rle]

        # Get original image size from annotations
        img_size = self.anns[key]["img_size"]
        orig_h, orig_w = img_size

        # Decode RLE at original size
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for r in rles:
            if isinstance(r, dict) and "counts" in r:
                mask |= mask_utils.decode(r)

        # Dilate to create thicker masks
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (DILATE_KERNEL, DILATE_KERNEL)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Resize to target size (same as image preprocessing)
        mask_resized = cv2.resize(
            mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST
        )
        return mask_resized

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        key = self.keys[idx]
        img_path = self._find_image(key)
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {key}")

        # Load and resize image (cxr-lines-and-tubes style: direct resize, no padding)
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        # Load and resize mask to same size
        mask_resized = self._load_mask(key, IMG_SIZE)

        # Apply augmentation (to both image and mask)
        if self.transform is not None:
            augmented = self.transform(image=image_resized, mask=mask_resized)
            image_resized = augmented["image"]
            mask_resized = augmented["mask"]

        # Convert to tensors (normalize to [0, 1])
        image_tensor = torch.from_numpy(image_resized[None].copy()).float()
        mask_tensor = torch.from_numpy(mask_resized[None].copy()).float()

        return {
            "x": image_tensor,
            "mask": mask_tensor,
            "key": key,
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "key": [b["key"] for b in batch],
    }


class CXRLoss(nn.Module):
    """Loss function: Weighted BCE + soft Dice.

    Balanced weighting for severe class imbalance.
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, pos_weight: float = 20.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Weighted BCE to handle class imbalance
        pos_weight_tensor = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight_tensor
        )

        # Soft Dice
        prob = torch.sigmoid(logits)
        smooth = 1.0
        intersection = (prob * target).sum(dim=(2, 3))
        union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def dice_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Hard Dice score for evaluation."""
    pred = (torch.sigmoid(logits) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (2.0 * intersection / union).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for decoder training")
    parser.add_argument("--pos-weight", type=float, default=20.0, help="Positive class weight for BCE")
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Dice loss weight (balanced with BCE)")
    parser.add_argument("--device-fold", type=int, default=0)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "other_decoder_cxr_style")
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"Loading device fold {args.device_fold}...")
    cfg = copy.deepcopy(device_cfg)
    cfg.pretrained = False
    cfg.load_pretrained_model = None

    model = DeviceNetWithOther(cfg).to(device)

    # Load pretrained weights
    checkpoint_path = CXR_MODEL_ROOT / "weights" / "device_family" / f"fold{args.device_fold}" / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("state_dict", checkpoint)
    state = {
        key.removeprefix("model."): value
        for key, value in raw.items()
        if key.startswith("model.") and not key.startswith("model.criterion.")
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # Freeze original branches
    model.freeze_original_branches()
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Dataset
    full_dataset = CheXlocalizeDataset(split="val", augment=False)

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # Wrap train with augmentation
    train_dataset = CheXlocalizeDataset(split="val", augment=True)
    train_dataset.keys = [full_dataset.keys[i] for i in train_subset.indices]

    val_dataset = val_subset

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Loss and optimizer (weighted BCE + low-weight Dice)
    criterion = CXRLoss(bce_weight=1.0, dice_weight=args.dice_weight, pos_weight=args.pos_weight)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_dice = 0.0
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["mask"].to(device)

            logits = model({"x": x})["other_logits"]
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validate
        model.eval()
        val_losses = []
        val_dices = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                x = batch["x"].to(device)
                y = batch["mask"].to(device)

                logits = model({"x": x})["other_logits"]
                loss = criterion(logits, y)
                dice = dice_score(logits, y)

                val_losses.append(loss.item())
                val_dices.append(dice)

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_dice = np.mean(val_dices)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_dice={val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_path = args.output / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_dice": val_dice,
            }, save_path)
            print(f"  -> Saved best model (dice={val_dice:.4f})")

        scheduler.step()

        with open(args.output / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val_dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
