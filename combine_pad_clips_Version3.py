#!/usr/bin/env python3
"""Combine all pad_clips CSVs into one deduplicated file."""
import csv
from pathlib import Path

INPUT_FILES = [
    r"C:\Users\jaysh\ToiletTraining\pad_clips1.csv",
    r"C:\Users\jaysh\ToiletTraining\pad_clips.csv",
    r"C:\Users\jaysh\ToiletTraining\pad_clips_new.csv",
]
OUTPUT = r"C:\Users\jaysh\ToiletTraining\pad_clips_combined.csv"

seen = {}
for f in INPUT_FILES:
    if not Path(f).exists():
        print(f"Skipping (not found): {f}")
        continue
    with open(f, newline="") as fh:
        for row in csv.DictReader(fh):
            seen[row["path"]] = row  # last one wins (dedup by path)

with open(OUTPUT, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(next(iter(seen.values())).keys()))
    writer.writeheader()
    writer.writerows(seen.values())

on_pad = sum(1 for r in seen.values() if r["on_pad"].strip() == "True")
print(f"Combined: {len(seen)} clips ({on_pad} on pad) → {OUTPUT}")