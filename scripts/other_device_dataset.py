"""CheXlocalize dataset for training the Other Devices decoder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
CHEX_ROOT = REPO_ROOT / "cheXlocalize_dataset" / "chexlocalize"

TASK = "Support Devices"
DILATE_KERNEL = 7  # Dilate thin lines to 7px
IMG_SIZE = 1024


class CheXlocalizeOtherDataset(Dataset):
    """Load CheXlocalize images with Support Devices masks (direct supervision)."""

    def __init__(self, split: str = "val", augment: bool = False, frontal_only: bool = False):
        self.split = split
        self.augment = augment
        self.frontal_only = frontal_only

        # Load annotations
        seg_file = CHEX_ROOT / "CheXlocalize" / f"gt_segmentations_{split}.json"
        ann_file = CHEX_ROOT / "CheXlocalize" / f"gt_annotations_{split}.json"

        with open(seg_file) as f:
            self.segs = json.load(f)
        with open(ann_file) as f:
            self.anns = json.load(f)

        # Filter images with Support Devices annotations
        keys = [k for k, v in self.anns.items() if TASK in v]

        # Optionally filter to frontal views only
        if frontal_only:
            keys = [k for k in keys if "frontal" in k.lower()]

        self.keys = keys
        print(f"Loaded {len(self.keys)} images with '{TASK}' from {split} split "
              f"(augment={augment}, frontal_only={frontal_only})")

        # Data augmentation pipeline
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
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
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

    def _load_gt_support(self, key: str, shape: tuple[int, int]) -> np.ndarray:
        """Load and dilate CheXlocalize Support Devices mask."""
        rle = self.segs[key][TASK]
        rles = rle if isinstance(rle, list) else [rle]
        mask = np.zeros(shape, dtype=np.uint8)
        for r in rles:
            if isinstance(r, dict) and "counts" in r:
                mask |= mask_utils.decode(r)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL, DILATE_KERNEL))
        return cv2.dilate(mask, kernel, iterations=1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        key = self.keys[idx]
        img_path = self._find_image(key)
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {key}")

        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        orig_shape = image.shape

        # GT Support Devices (dilated)
        gt_support = self._load_gt_support(key, orig_shape)

        # Resize to model input size
        image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask_resized = cv2.resize(gt_support, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # Apply augmentation
        if self.transform is not None:
            augmented = self.transform(image=image_resized, mask=mask_resized)
            image_resized = augmented["image"]
            mask_resized = augmented["mask"]

        # Convert to tensors
        image_tensor = torch.from_numpy(image_resized[None].copy()).float()
        mask_tensor = torch.from_numpy(mask_resized[None].copy()).float()

        return {
            "x": image_tensor,
            "other_mask": mask_tensor,
            "key": key,
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "other_mask": torch.stack([b["other_mask"] for b in batch]),
        "key": [b["key"] for b in batch],
    }
