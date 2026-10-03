"""Train the Other Devices decoder on CheXlocalize XOR labels."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
CXR_MODEL_ROOT = REPO_ROOT / "cxr-lines-and-tubes"
sys.path.insert(0, str(CXR_MODEL_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from other_device_dataset import CheXlocalizeOtherDataset, collate_fn  # noqa: E402
from other_device_model import DeviceNetWithOther  # noqa: E402
from skp.configs.ranzcr_device_specific_distal_b1_unet_1024_swan10_ablation import (  # noqa: E402
    cfg as device_cfg,
)


class BCEDiceLoss(nn.Module):
    """Combined BCE + soft Dice loss with class imbalance handling.

    Loss = BCE_loss × 1.0 + Dice_loss × 1.0
    - BCE_loss = weighted BCE (positive class 權重 20 倍)
    - Dice_loss = soft Dice coefficient
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, pos_weight: float = 20.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Weighted BCE: positive pixels weighted by pos_weight
        pos_weight = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW optimizer")
    parser.add_argument("--pos-weight", type=float, default=20.0, help="Positive class weight in BCE (for class imbalance)")
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Weight for Dice loss (balanced with BCE)")
    parser.add_argument("--device-fold", type=int, default=0, help="Which device fold to load")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "other_decoder_training")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--frontal-only", action="store_true", help="Use only frontal view images")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load device model checkpoint
    print(f"Loading device fold {args.device_fold}...")
    cfg = copy.deepcopy(device_cfg)
    cfg.pretrained = False
    cfg.load_pretrained_model = None

    model = DeviceNetWithOther(cfg).to(device)

    # Load pretrained weights
    checkpoint_path = CXR_MODEL_ROOT / "weights" / "device_family" / f"fold{args.device_fold}" / "model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("state_dict", checkpoint)
    state = {
        key.removeprefix("model."): value
        for key, value in raw.items()
        if key.startswith("model.") and not key.startswith("model.criterion.")
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Expect missing keys for other_decoder and other_head
    print(f"Loaded checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    assert all("other_" in k for k in missing), "Only other_* keys should be missing"

    # Freeze original branches
    model.freeze_original_branches()
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Dataset - direct supervision with Support Devices masks
    # Create two separate datasets: train with augmentation, val without
    full_dataset = CheXlocalizeOtherDataset(split="val", augment=False, frontal_only=args.frontal_only)

    # Split train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # Wrap train subset with augmentation
    train_dataset = CheXlocalizeOtherDataset(split="val", augment=True, frontal_only=args.frontal_only)
    train_dataset.keys = [full_dataset.keys[i] for i in train_subset.indices]
    print(f"Train dataset with augmentation: {len(train_dataset)} images")

    val_dataset = val_subset  # No augmentation for validation

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Loss and optimizer
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=args.dice_weight, pos_weight=args.pos_weight)
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
            y = batch["other_mask"].to(device)

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
                y = batch["other_mask"].to(device)

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

        # Save best model
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

        # Save history
        with open(args.output / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val Dice: {best_val_dice:.4f}")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
