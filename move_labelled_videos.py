import csv
import os
import shutil
from pathlib import Path

LABELS_CSV = Path(r"C:\Users\jaysh\ToiletTraining\dog_labels.csv")

DESTS = {
    "neither": Path(r"C:\Users\jaysh\ToiletTraining\Neither"),
    "poo": Path(r"C:\Users\jaysh\ToiletTraining\Poo"),
    "wee": Path(r"C:\Users\jaysh\ToiletTraining\Wee"),
}

def normalise_label(label: str) -> str:
    return (label or "").strip().lower()

def safe_move(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    # Avoid overwriting by adding suffix if needed
    if dst.exists():
        stem = dst.stem
        suffix = dst.suffix
        i = 1
        while True:
            candidate = dst_dir / f"{stem}__dup{i}{suffix}"
            if not candidate.exists():
                dst = candidate
                break
            i += 1

    shutil.move(str(src), str(dst))
    return dst

def main() -> None:
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Labels CSV not found: {LABELS_CSV}")

    for d in DESTS.values():
        d.mkdir(parents=True, exist_ok=True)

    moved = 0
    missing = 0
    unknown = 0
    errors = 0

    with LABELS_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"CSV must contain 'path' and 'label' columns. Found: {reader.fieldnames}")

        for row in reader:
            src_str = (row.get("path") or "").strip()
            label = normalise_label(row.get("label", ""))

            if not src_str:
                continue

            if label not in DESTS:
                unknown += 1
                continue

            src = Path(src_str)

            # If already moved, treat it as done (skip)
            if not src.exists():
                # Try to find it in any destination folder by filename
                name = Path(src_str).name
                already = any((dest / name).exists() for dest in DESTS.values())
                if already:
                    continue
                missing += 1
                continue

            try:
                dst = safe_move(src, DESTS[label])
                moved += 1
            except Exception:
                errors += 1

    print("Done.")
    print(f"Moved:   {moved}")
    print(f"Missing: {missing}")
    print(f"Unknown label rows: {unknown}")
    print(f"Errors:  {errors}")

if __name__ == "__main__":
    main()