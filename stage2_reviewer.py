#!/usr/bin/env python3
# stage2_reviewer.py
# Review Stage 1 dog clips and label into wee / poo / neither folders.
#
# Usage example (PowerShell):
# python stage2_reviewer.py --shortlist "C:/Users/jaysh/ToiletTraining/shortlist.csv" --only-dog-true

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".ts", ".flv", ".webm"}


def as_bool(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


@dataclass
class ReviewRow:
    path: Path
    meta: dict


def ensure_label_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "wee": root / "wee",
        "poo": root / "poo",
        "neither": root / "neither",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def load_csv_rows(shortlist_csv: Path) -> List[dict]:
    with shortlist_csv.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in CSV: {shortlist_csv}")
    if "path" not in rows[0]:
        raise ValueError("shortlist CSV must contain a 'path' column.")
    return rows


def filter_rows(rows: List[dict], only_dog_true: bool, allowed_sources: List[str]) -> List[ReviewRow]:
    allowed_l = [s.lower() for s in allowed_sources]
    out: List[ReviewRow] = []

    for r in rows:
        p = Path((r.get("path") or "").strip())
        if not str(p):
            continue

        # Filter to dog_detected=true if requested and the column exists
        if only_dog_true and "dog_detected" in r:
            if not as_bool(r.get("dog_detected", "")):
                continue

        # Restrict to specified Tapo source folders if requested
        if allowed_l:
            pl = str(p).lower()
            ok = False
            for src in allowed_l:
                if f"tapovideos\\{src}\\" in pl or f"tapovideos/{src}/" in pl:
                    ok = True
                    break
            if not ok:
                continue

        # Only review existing video files
        if not p.exists():
            continue
        if p.suffix.lower() not in VIDEO_EXTS:
            continue

        out.append(ReviewRow(path=p, meta=r))

    return out


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def safe_put_file(src: Path, dst_dir: Path, mode: str) -> Path:
    dst = dst_dir / src.name
    if mode == "move":
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
    else:
        if not dst.exists():
            shutil.copy2(str(src), str(dst))
    return dst


def open_video(path: Path) -> Tuple[cv2.VideoCapture, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, fps, total


def append_label(labels_csv: Path, rr: ReviewRow, label: str) -> None:
    exists = labels_csv.exists()
    fieldnames = ["path", "label", "ts", "best_confidence", "dog_frame_count", "total_frames_checked"]

    with labels_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(
            {
                "path": str(rr.path),
                "label": label,
                "ts": datetime.now(timezone.utc).isoformat(),
                "best_confidence": rr.meta.get("best_confidence", ""),
                "dog_frame_count": rr.meta.get("dog_frame_count", ""),
                "total_frames_checked": rr.meta.get("total_frames_checked", ""),
            }
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Review Stage 1 dog clips and label as wee/poo/neither.")
    ap.add_argument("--shortlist", required=True, help="Path to Stage 1 shortlist CSV (must contain 'path').")
    ap.add_argument("--labelled", default=r"C:\Users\jaysh\ToiletTraining", help="Root folder for wee/poo/neither.")
    ap.add_argument("--labels-csv", default=r"C:\Users\jaysh\ToiletTraining\dog_labels.csv", help="Label log CSV.")
    ap.add_argument("--mode", choices=["copy", "move"], default="copy", help="Copy or move clips into label folders.")
    ap.add_argument("--only-dog-true", action="store_true", help="Only review dog_detected=true rows (if column exists).")
    ap.add_argument("--allowed-source", action="append", default=[], help="Restrict to TapoVideos source folders. Repeatable.")
    ap.add_argument("--state-file", default=None, help="Resume state JSON (default: <labelled>/review_state.json).")
    ap.add_argument("--seek-seconds", type=float, default=2.0, help="Seek step seconds for A/D.")
    args = ap.parse_args()

    shortlist_csv = Path(args.shortlist)
    labelled_root = Path(args.labelled)
    labels_csv = Path(args.labels_csv)
    state_path = Path(args.state_file) if args.state_file else (labelled_root / "review_state.json")

    if not shortlist_csv.exists():
        sys.exit(f"Shortlist CSV not found: {shortlist_csv}")

    ensure_label_dirs(labelled_root)

    rows = load_csv_rows(shortlist_csv)
    review_rows = filter_rows(rows, only_dog_true=args.only_dog_true, allowed_sources=args.allowed_source)

    if not review_rows:
        sys.exit("No reviewable clips found after filtering (check --only-dog-true and --allowed-source).")

    state = load_state(state_path)
    pos = int(state.get("position", 0))
    history = state.get("history", [])

    win = "Stage2 Reviewer"
    paused = False

    print(f"Reviewable clips: {len(review_rows)}")
    print(f"Resume position: {pos}")
    print("Keys: W=wee, P=poo, N=neither, S=skip, B=back, Space=pause, A/D=seek, Q/Esc=quit")

    while 0 <= pos < len(review_rows):
        rr = review_rows[pos]

        try:
            cap, fps, total_frames = open_video(rr.path)
        except Exception as e:
            print(f"[{pos+1}/{len(review_rows)}] SKIP unreadable: {rr.path.name} ({e})")
            pos += 1
            state["position"] = pos
            save_state(state_path, state)
            continue

        seek_step = max(1, int(round(args.seek_seconds * fps)))
        frame_pos = 0

        if state.get("current_path") == str(rr.path):
            frame_pos = int(state.get("frame_pos", 0))
            if frame_pos > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

        paused = False

        print(f"\n[{pos+1}/{len(review_rows)}] {rr.path.name}")
        if "best_confidence" in rr.meta or "dog_frame_count" in rr.meta:
            print(
                f"  Conf: {rr.meta.get('best_confidence','?')}  "
                f"dog_frames: {rr.meta.get('dog_frame_count','?')}/{rr.meta.get('total_frames_checked','?')}"
            )

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)

                overlay = f"[{pos+1}/{len(review_rows)}] {rr.path.name}  frame {frame_pos}/{total_frames}  fps~{fps:.1f}"
                cv2.putText(frame, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.imshow(win, frame)

            key = cv2.waitKey(30 if not paused else 0) & 0xFF

            if key in (ord("q"), 27):
                state["position"] = pos
                state["current_path"] = str(rr.path)
                state["frame_pos"] = frame_pos
                state["history"] = history
                save_state(state_path, state)
                cap.release()
                cv2.destroyAllWindows()
                print(f"\nSaved state: {state_path}")
                print(f"Labels CSV:  {labels_csv}")
                return

            if key == ord(" "):
                paused = not paused
                continue

            if key in (ord("a"), ord("A")):
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_pos - seek_step))
                continue

            if key in (ord("d"), ord("D")):
                new_pos = frame_pos + seek_step
                if total_frames > 0:
                    new_pos = min(total_frames - 1, new_pos)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                continue

            if key in (ord("s"), ord("S")):
                cap.release()
                pos += 1
                state["position"] = pos
                state["current_path"] = ""
                state["frame_pos"] = 0
                save_state(state_path, state)
                break

            if key in (ord("b"), ord("B")):
                cap.release()
                if history:
                    last = history.pop()
                    pos = max(0, int(last.get("pos", pos)) - 1)
                else:
                    pos = max(0, pos - 1)
                state["position"] = pos
                state["history"] = history
                state["current_path"] = ""
                state["frame_pos"] = 0
                save_state(state_path, state)
                break

            label = None
            if key in (ord("w"), ord("W")):
                label = "wee"
            elif key in (ord("p"), ord("P")):
                label = "poo"
            elif key in (ord("n"), ord("N")):
                label = "neither"

            if label:
                try:
                    safe_put_file(rr.path, labelled_root / label, args.mode)
                except Exception as e:
                    print(f"WARNING: {args.mode} failed: {e}")

                append_label(labels_csv, rr, label)
                history.append({"pos": pos + 1, "path": str(rr.path), "label": label})

                cap.release()
                pos += 1
                state["position"] = pos
                state["current_path"] = ""
                state["frame_pos"] = 0
                state["history"] = history
                save_state(state_path, state)
                break

    cv2.destroyAllWindows()
    state["position"] = pos
    state["current_path"] = ""
    state["frame_pos"] = 0
    state["history"] = history
    save_state(state_path, state)
    print("\nDone.")
    print(f"Saved state: {state_path}")
    print(f"Labels CSV:  {labels_csv}")


if __name__ == "__main__":
    main()