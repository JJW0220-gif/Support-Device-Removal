"""Count available data for training."""

import json
from pathlib import Path

CHEX_ROOT = Path(__file__).resolve().parent.parent / "cheXlocalize_dataset" / "chexlocalize"

# Load annotations
with open(CHEX_ROOT / "CheXlocalize" / "gt_annotations_val.json") as f:
    anns = json.load(f)

# Filter Support Devices
keys_all = [k for k, v in anns.items() if "Support Devices" in v]
keys_frontal = [k for k in keys_all if "frontal" in k.lower()]
keys_lateral = [k for k in keys_all if "lateral" in k.lower()]

print("=== CheXlocalize val split ===")
print(f"Total images: {len(anns)}")
print(f"With Support Devices: {len(keys_all)}")
print(f"  - Frontal: {len(keys_frontal)}")
print(f"  - Lateral: {len(keys_lateral)}")
print()
print("Train/Val split (80/20):")
train_all = int(len(keys_all) * 0.8)
val_all = len(keys_all) - train_all
train_frontal = int(len(keys_frontal) * 0.8)
val_frontal = len(keys_frontal) - train_frontal
print(f"  All views:     {train_all} train / {val_all} val")
print(f"  Frontal only:  {train_frontal} train / {val_frontal} val")
print()
print("Recommendation:")
if train_frontal >= 60:
    print("  ✅ Frontal-only is sufficient (>= 60 train images)")
elif train_frontal >= 40:
    print("  ⚠️  Frontal-only is marginal (40-60 train images)")
else:
    print("  ❌ Frontal-only is too few (< 40 train images), use all views")
