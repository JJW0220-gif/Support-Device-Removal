"""Inspect CheXlocalize dataset structure."""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "cheXlocalize_dataset" / "chexlocalize" / "CheXlocalize"

for name in ["gt_segmentations_val.json", "gt_annotations_val.json"]:
    print(f"\n=== {name} ===")
    with open(ROOT / name) as f:
        data = json.load(f)
    keys = list(data.keys())
    print(f"Total entries: {len(keys)}")
    print(f"First key: {keys[0]}")
    first = data[keys[0]]
    print(f"Value type: {type(first).__name__}")
    if isinstance(first, dict):
        print(f"Sub-keys (first 5): {list(first.keys())[:5]}")
        # count tasks across dataset
        task_counter = Counter()
        for v in data.values():
            if isinstance(v, dict):
                task_counter.update(v.keys())
        print(f"Task distribution: {dict(task_counter)}")
        # sample one annotation
        for task, ann in first.items():
            print(f"\nSample task '{task}':")
            if isinstance(ann, list):
                print(f"  type: list, len={len(ann)}, first elem type: {type(ann[0]).__name__ if ann else 'empty'}")
                if ann and isinstance(ann[0], list):
                    print(f"  nested list, first sublist len={len(ann[0])}, sample: {str(ann[0][:2])[:200]}")
            else:
                print(f"  {str(ann)[:300]}")
            break
