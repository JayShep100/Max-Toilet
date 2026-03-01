#!/usr/bin/env python3
"""
Cleanup — Delete No-Dog Clips
==============================
Reads the completed shortlist.csv from stage1_miner.py and permanently
deletes any video clip where dog_detected = False.

All clips where YOLO detected a dog — even at low confidence — are kept,
since low confidence detections are still genuine sightings (e.g. unusual
poses like eating with head down).

Run this AFTER stage1_miner.py has fully finished.

Usage:
    python cleanup_no_dog.py --shortlist shortlist.csv
    python cleanup_no_dog.py --shortlist shortlist.csv --dry-run
"""

import argparse
import csv
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Delete clips with no dog detected")
    parser.add_argument("--shortlist",      required=True, help="shortlist.csv from stage1_miner.py")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Show what would be deleted without actually deleting")
    args = parser.parse_args()

    if not Path(args.shortlist).exists():
        print(f"Shortlist not found: {args.shortlist}")
        return

    with open(args.shortlist, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Shortlist is empty.")
        return

    to_delete = []
    to_keep   = []

    for row in rows:
        path       = row.get("path", "")
        detected   = row.get("dog_detected", "False").strip().lower() == "true"
        if not detected:
            to_delete.append((path, "no dog detected"))
        else:
            to_keep.append(path)

    print(f"Shortlist:    {len(rows)} clips")
    print(f"To delete:    {len(to_delete)}")
    print(f"To keep:      {len(to_keep)}")

    if not to_delete:
        print("\nNothing to delete.")
        return

    if args.dry_run:
        print("\n--- DRY RUN — nothing will be deleted ---\n")
        for path, reason in to_delete[:20]:
            print(f"  WOULD DELETE  {Path(path).name}  ({reason})")
        if len(to_delete) > 20:
            print(f"  ... and {len(to_delete) - 20} more")
        return

    # Confirm before deleting
    print(f"\n⚠  This will permanently delete {len(to_delete)} video files.")
    answer = input("Type YES to confirm: ").strip()
    if answer != "YES":
        print("Aborted — nothing deleted.")
        return

    deleted  = 0
    missing  = 0
    errors   = 0

    for path, reason in to_delete:
        p = Path(path)
        if not p.exists():
            missing += 1
            continue
        try:
            p.unlink()
            deleted += 1
            print(f"  DELETED  {p.name}  ({reason})")
        except Exception as e:
            errors += 1
            print(f"  ERROR    {p.name}  — {e}")

    print(f"\nDone.")
    print(f"  Deleted:  {deleted}")
    print(f"  Missing:  {missing}  (already gone)")
    print(f"  Errors:   {errors}")


if __name__ == "__main__":
    main()
