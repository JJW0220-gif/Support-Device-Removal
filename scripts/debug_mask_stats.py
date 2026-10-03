"""Quick check: Support Devices mask statistics."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from other_device_dataset import CheXlocalizeOtherDataset

dataset = CheXlocalizeOtherDataset(split="val")

print(f"Dataset size: {len(dataset)}")
print("\n=== First 5 samples ===")

for i in range(min(5, len(dataset))):
    sample = dataset[i]
    mask = sample["other_mask"].numpy()[0]  # Remove channel dim
    print(f"\nSample {i}: {sample['key']}")
    print(f"  Shape: {mask.shape}")
    print(f"  Positive pixels: {mask.sum():.0f} / {mask.size}")
    print(f"  Positive ratio: {mask.mean():.4%}")
    print(f"  Range: [{mask.min():.1f}, {mask.max():.1f}]")
