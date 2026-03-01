#!/usr/bin/env python3
"""Find dog clips from Stage 1 that haven't been processed by Stage 2."""

import csv
from pathlib import Path

# Load all dog-detected clips from Stage 1
with open("shortlist.csv", newline="") as f:
    stage1 = [r["path"] for r in csv.DictReader(f)
              if r["dog_detected"].strip().upper() == "TRUE"]

# Load all clips already processed in Stage 2
with open("pad_clips1.csv", newline="") as f:
    stage2 = {r["path"] for r in csv.DictReader(f)}

# Find what's missing
missing = [p for p in stage1 if p not in stage2]

print(f"Stage 1 dog clips:     {len(stage1)}")
print(f"Stage 2 processed:     {len(stage2)}")
print(f"Missing from Stage 2:  {len(missing)}")

if missing:
    print("\nMissing clips:")
    for p in missing[:20]:
        print(f"  {Path(p).name}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")

    # Optionally write to file for Stage 2 to re-process
    with open("missing_stage2.txt", "w") as f:
        for p in missing:
            f.write(p + "\n")
    print(f"\nFull list saved to missing_stage2.txt")