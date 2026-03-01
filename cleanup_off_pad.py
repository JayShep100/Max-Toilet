#!/usr/bin/env python3
"""
Cleanup — Delete Off-Pad Clips
================================
Reads pad_clips.csv from Stage 2 and deletes any clip where
on_pad = False (Max was not detected on the toilet pad).

Usage:
    python cleanup_off_pad.py --pad-clips pad_clips.csv
    python cleanup_off_pad.py --pad-clips pad_clips.csv --dry-run
"""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Delete clips where Max wasn't on the pad")
    parser.add_argument("--pad-clips", required=True, help="pad_clips.csv from Stage 2")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Show what would be deleted without actually deleting")
    args = parser.parse_args()

    if not Path(args.pad_clips).exists():
        print(f"File not found: {args.pad_clips}")
        return

    with open(args.pad_clips, newline="") as f:
        rows = list(csv.DictReader(f))

    to_delete = [r["path"] for r in rows if r["on_pad"].strip() == "False"]
    to_keep   = [r["path"] for r in rows if r["on_pad"].strip() == "True"]

    print(f"Total clips:  {len(rows)}")
    print(f"To keep:      {len(to_keep)}")
    print(f"To delete:    {len(to_delete)}")

    if not to_delete:
        print("\nNothing to delete.")
        return

    if args.dry_run:
        print("\n--- DRY RUN — nothing will be deleted ---\n")
        for path in to_delete[:20]:
            print(f"  WOULD DELETE  {Path(path).name}")
        if len(to_delete) > 20:
            print(f"  ... and {len(to_delete) - 20} more")
        return

    print(f"\n⚠  This will permanently delete {len(to_delete)} video files.")
    answer = input("Type YES to confirm: ").strip()
    if answer != "YES":
        print("Aborted — nothing deleted.")
        return

    deleted = 0
    missing = 0
    errors  = 0

    for path in to_delete:
        p = Path(path)
        if not p.exists():
            missing += 1
            continue
        try:
            p.unlink()
            deleted += 1
            print(f"  DELETED  {p.name}")
        except Exception as e:
            errors += 1
            print(f"  ERROR    {p.name}  — {e}")

    print(f"\nDone.")
    print(f"  Deleted:  {deleted}")
    print(f"  Missing:  {missing}  (already gone)")
    print(f"  Errors:   {errors}")
    print(f"\n{len(to_keep)} clips remaining — Max on pad events ready to label.")


if __name__ == "__main__":
    main()
