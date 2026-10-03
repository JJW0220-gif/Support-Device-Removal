"""Inspect checkpoint keys."""
import sys
import torch

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("Keys:", list(ckpt.keys()))
if "state_dict" in ckpt:
    keys = list(ckpt["state_dict"].keys())
    other_keys = [k for k in keys if "other" in k.lower()]
    print(f"\nOther-related keys ({len(other_keys)}):")
    for k in other_keys[:10]:
        print(f"  {k}")
    if not other_keys:
        print("  (none found)")
        print("\nAll unique prefixes:")
        prefixes = set(k.split(".")[0] for k in keys)
        for p in sorted(prefixes):
            print(f"  {p}")
