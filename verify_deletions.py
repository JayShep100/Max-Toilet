#!/usr/bin/env python3
"""
Verify Deletions
=================
Checks shortlist.csv and reports which files actually exist on disk
and which have been deleted.

Usage:
    python verify_deletions.py --shortlist shortlist.csv
"""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Verify clip deletions")
    parser.add_argument("--shortlist", required=True)
    args = parser.parse_args()

    if not Path(args.shortlist).exists():
        print(f"Shortlist not found: {args.shortlist}")
        return

    with open(args.shortlist, newline="") as f:
        rows = list(csv.DictReader(f))

    should_keep   = []
    should_delete = []

    for row in rows:
        path     = row.get("path", "")
        detected = row.get("dog_detected", "False").strip().lower() == "true"
        exists   = Path(path).exists()

        if detected:
            should_keep.append((path, exists))
        else:
            should_delete.append((path, exists))

    # No-dog clips
    deleted_ok  = [(p, e) for p, e in should_delete if not e]
    still_there = [(p, e) for p, e in should_delete if e]

    # Dog clips
    kept_ok    = [(p, e) for p, e in should_keep if e]
    missing    = [(p, e) for p, e in should_keep if not e]

    print(f"── No-dog clips (should be deleted) ───────────────────────")
    print(f"   Confirmed deleted:  {len(deleted_ok)}")
    print(f"   Still on disk:      {len(still_there)}")

    print(f"\n── Dog clips (should be kept) ──────────────────────────────")
    print(f"   Confirmed present:  {len(kept_ok)}")
    print(f"   Unexpectedly gone:  {len(missing)}")

    if still_there:
        print(f"\n⚠  These no-dog clips were NOT deleted:")
        for path, _ in still_there:
            print(f"   {Path(path).name}")

    if missing:
        print(f"\n⚠  These dog clips are missing (deleted by mistake?):")
        for path, _ in missing:
            print(f"   {Path(path).name}")

    if not still_there and not missing:
        print(f"\n✅ All good — {len(deleted_ok)} no-dog clips deleted, "
              f"{len(kept_ok)} dog clips intact.")


if __name__ == "__main__":
    main()
