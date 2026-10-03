"""Debug: Check what the model is actually predicting."""

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
CXR_MODEL_ROOT = REPO_ROOT / "cxr-lines-and-tubes"
sys.path.insert(0, str(CXR_MODEL_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import copy
from other_device_dataset import CheXlocalizeOtherDataset, collate_fn
from other_device_model import DeviceNetWithOther
from skp.configs.ranzcr_device_specific_distal_b1_unet_1024_swan10_ablation import cfg as device_cfg
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
print("=== Loading model ===")
cfg = copy.deepcopy(device_cfg)
cfg.pretrained = False
cfg.load_pretrained_model = None
model = DeviceNetWithOther(cfg).to(device)

checkpoint_path = CXR_MODEL_ROOT / "weights" / "device_family" / "fold0" / "model.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
raw = checkpoint.get("state_dict", checkpoint)
state = {k.removeprefix("model."): v for k, v in raw.items() if k.startswith("model.") and not k.startswith("model.criterion.")}
model.load_state_dict(state, strict=False)
model.freeze_original_branches()
model.eval()

# Load one batch
print("\n=== Loading dataset ===")
dataset = CheXlocalizeOtherDataset(split="val", augment=False)
loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

batch = next(iter(loader))
x = batch["x"].to(device)
y_true = batch["other_mask"].to(device)

print(f"\n=== Model input/output ===")
print(f"Input shape: {x.shape}, range: [{x.min():.1f}, {x.max():.1f}]")
print(f"GT mask shape: {y_true.shape}, positive: {y_true.sum().item():.0f}")

# Forward pass
with torch.no_grad():
    output = model({"x": x})
    logits = output["other_logits"]
    probs = torch.sigmoid(logits)

print(f"\nLogits shape: {logits.shape}")
print(f"Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
print(f"Logits mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")

print(f"\nProbabilities shape: {probs.shape}")
print(f"Probabilities range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
print(f"Probabilities mean: {probs.mean().item():.4f}")

# Check predictions at different thresholds
for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
    pred = (probs > thresh).float()
    tp = (pred * y_true).sum().item()
    fp = (pred * (1 - y_true)).sum().item()
    fn = ((1 - pred) * y_true).sum().item()

    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    print(f"\nThreshold {thresh:.1f}:")
    print(f"  Predicted positive: {pred.sum().item():.0f}")
    print(f"  Dice: {dice:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# Check if model is just predicting all zeros
print(f"\n=== Diagnosis ===")
if probs.max().item() < 0.5:
    print("⚠️  Model NEVER predicts positive (max prob < 0.5)")
    print("   This means the model is stuck predicting background only.")
    print("   Possible causes:")
    print("   1. Learning rate too high -> gradient explosion -> stuck at all-zero")
    print("   2. Loss function issue -> BCE dominating, Dice too weak")
    print("   3. Initialization issue -> decoder weights too small")
elif probs.mean().item() < 0.01:
    print("⚠️  Model predictions are EXTREMELY low (mean prob < 0.01)")
    print("   The model is barely activating. This suggests:")
    print("   1. Encoder features are not useful for this task")
    print("   2. Decoder is not learning from frozen features")
else:
    print(f"✓ Model is producing predictions (max prob: {probs.max().item():.4f})")
