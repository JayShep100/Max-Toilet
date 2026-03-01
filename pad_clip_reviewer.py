#!/usr/bin/env python3
"""
Review clips listed in a CSV and label each one as wee / poo / neither.

Expected CSV columns:
- path (required)
- on_pad (optional bool)

Usage example:
python pad_clip_reviewer.py \
  --csv "C:/Users/jaysh/ToiletTraining/pad_clips.csv" \
  --dest-root "C:/Users/jaysh/ToiletTraining" \
  --move
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

try:
    import cv2
except ImportError:
    sys.exit("Missing dependency: pip install opencv-python")

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".ts", ".wmv", ".m4v"}
LABELS = {"w": "wee", "p": "poo", "n": "neither"}


@dataclass
class ReviewItem:
    index: int
    path: str
    on_pad: Optional[bool] = None


@dataclass
class Decision:
    index: int
    label: str
    original: str
    current: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Review clip paths from a CSV and label them.")
    p.add_argument("--csv", required=True, help="CSV containing a 'path' column")
    p.add_argument("--dest-root", required=True, help="Root folder containing wee/poo/neither folders")
    p.add_argument("--state", default=None, help="Optional JSON state file path")
    p.add_argument("--labels-csv", default=None, help="Optional CSV export of labels")
    p.add_argument("--copy", action="store_true", help="Copy clips instead of moving them")
    p.add_argument("--move", action="store_true", help="Move clips into label folders")
    p.add_argument("--only-on-pad", action="store_true", help="If CSV has on_pad column, only include rows where on_pad is true")
    p.add_argument("--start-index", type=int, default=0, help="Start reviewing from this CSV index if no state file exists")
    p.add_argument("--fps", type=float, default=24.0, help="Playback FPS in the review window")
    p.add_argument("--max-width", type=int, default=1280, help="Resize display frame to at most this width")
    return p.parse_args()


def parse_bool(value: str) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def load_items(csv_path: Path, only_on_pad: bool) -> List[ReviewItem]:
    items: List[ReviewItem] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "path" not in (reader.fieldnames or []):
            raise ValueError("CSV must contain a 'path' column")
        for idx, row in enumerate(reader):
            clip_path = (row.get("path") or "").strip()
            if not clip_path:
                continue
            p = Path(clip_path)
            if p.suffix.lower() not in VIDEO_EXTS:
                continue
            on_pad = parse_bool(row.get("on_pad")) if "on_pad" in row else None
            if only_on_pad and on_pad is False:
                continue
            items.append(ReviewItem(index=idx, path=str(p), on_pad=on_pad))
    return items


def ensure_dirs(dest_root: Path) -> None:
    for name in ["wee", "poo", "neither"]:
        (dest_root / name).mkdir(parents=True, exist_ok=True)


def default_state_path(dest_root: Path) -> Path:
    return dest_root / "pad_review_state.json"


def default_labels_csv(dest_root: Path) -> Path:
    return dest_root / "pad_labels.csv"


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"position": 0, "history": []}
    with state_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("position", 0)
    data.setdefault("history", [])
    return data


def save_state(state_path: Path, position: int, history: List[Decision]) -> None:
    payload = {
        "position": position,
        "history": [asdict(h) for h in history],
    }
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_labels_csv(labels_csv: Path, history: List[Decision]) -> None:
    with labels_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "label", "original", "current"])
        writer.writeheader()
        for item in history:
            writer.writerow(asdict(item))


def fit_frame(frame, max_width: int):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(frame, new_size)


def play_clip(clip_path: Path, fps: float, max_width: int) -> str:
    window = "Pad Clip Reviewer"
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return "skip"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 24.0
    effective_fps = fps if fps > 0 else native_fps
    delay = max(1, int(1000 / max(effective_fps, 1)))
    paused = False
    frame_idx = 0
    current_frame = None

    help_text = "W=wee  P=poo  N=neither  S=skip  B=back  Space=pause  A/D=seek  Q/Esc=quit"

    while True:
        if not paused or current_frame is None:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                ret, frame = cap.read()
                if not ret:
                    break
            current_frame = frame
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or frame_idx + 1)

        display = fit_frame(current_frame.copy(), max_width)
        title = f"{clip_path.name}  [{frame_idx}/{max(total_frames, 1)}]"
        cv2.putText(display, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display, help_text, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(window, display)

        key = cv2.waitKey(0 if paused else delay) & 0xFF
        if key in (ord('w'), ord('W')):
            cap.release(); cv2.destroyWindow(window); return 'wee'
        if key in (ord('p'), ord('P')):
            cap.release(); cv2.destroyWindow(window); return 'poo'
        if key in (ord('n'), ord('N')):
            cap.release(); cv2.destroyWindow(window); return 'neither'
        if key in (ord('s'), ord('S')):
            cap.release(); cv2.destroyWindow(window); return 'skip'
        if key in (ord('b'), ord('B')):
            cap.release(); cv2.destroyWindow(window); return 'back'
        if key in (ord('q'), ord('Q'), 27):
            cap.release(); cv2.destroyWindow(window); return 'quit'
        if key == 32:
            paused = not paused
            continue
        if key in (ord('a'), ord('A')):
            new_frame = max(0, frame_idx - int(native_fps * 2))
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            current_frame = None
            paused = True
            continue
        if key in (ord('d'), ord('D')):
            new_frame = min(max(total_frames - 1, 0), frame_idx + int(native_fps * 2))
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            current_frame = None
            paused = True
            continue

    cap.release()
    cv2.destroyWindow(window)
    return 'skip'


def unique_dest(dest_dir: Path, src_name: str) -> Path:
    candidate = dest_dir / src_name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    i = 2
    while True:
        test = dest_dir / f"{stem}__{i}{suffix}"
        if not test.exists():
            return test
        i += 1


def place_clip(src: Path, dest_root: Path, label: str, move_files: bool) -> Path:
    dest_dir = dest_root / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = unique_dest(dest_dir, src.name)
    if move_files:
        shutil.move(str(src), str(dest))
    else:
        shutil.copy2(str(src), str(dest))
    return dest


def main() -> int:
    args = parse_args()
    if args.copy and args.move:
        print("Use either --copy or --move, not both.")
        return 2

    move_files = bool(args.move)
    csv_path = Path(args.csv)
    dest_root = Path(args.dest_root)
    state_path = Path(args.state) if args.state else default_state_path(dest_root)
    labels_csv = Path(args.labels_csv) if args.labels_csv else default_labels_csv(dest_root)

    ensure_dirs(dest_root)
    items = load_items(csv_path, only_on_pad=args.only_on_pad)
    if not items:
        print("No valid video rows found in CSV.")
        return 1

    state = load_state(state_path)
    history: List[Decision] = [Decision(**x) for x in state.get("history", [])]
    position = int(state.get("position", args.start_index)) if state_path.exists() else args.start_index
    position = max(0, min(position, len(items)))

    print(f"Loaded {len(items)} clip(s) from {csv_path}")
    print(f"State file: {state_path}")
    print(f"Destination root: {dest_root}")
    print("Controls: W=wee, P=poo, N=neither, S=skip, B=back, Space=pause, A/D=seek, Q/Esc=quit")

    while position < len(items):
        item = items[position]
        clip = Path(item.path)
        if not clip.exists():
            print(f"[{position+1}/{len(items)}] Missing file, skipping: {clip}")
            position += 1
            save_state(state_path, position, history)
            export_labels_csv(labels_csv, history)
            continue

        print(f"[{position+1}/{len(items)}] Reviewing: {clip}")
        action = play_clip(clip, fps=args.fps, max_width=args.max_width)

        if action == 'quit':
            save_state(state_path, position, history)
            export_labels_csv(labels_csv, history)
            print("Saved progress and quit.")
            return 0

        if action == 'back':
            if history:
                last = history.pop()
                current = Path(last.current)
                original = Path(last.original)
                try:
                    if current.exists() and move_files:
                        original.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(current), str(original))
                    elif current.exists() and not move_files:
                        current.unlink(missing_ok=True)
                except Exception as e:
                    print(f"Could not undo previous action: {e}")
                    history.append(last)
                position = max(0, int(last.index))
            else:
                position = max(0, position - 1)
            save_state(state_path, position, history)
            export_labels_csv(labels_csv, history)
            continue

        if action == 'skip':
            position += 1
            save_state(state_path, position, history)
            export_labels_csv(labels_csv, history)
            continue

        if action in {'wee', 'poo', 'neither'}:
            dest = place_clip(clip, dest_root, action, move_files=move_files)
            history.append(Decision(index=position, label=action, original=str(clip), current=str(dest)))
            position += 1
            save_state(state_path, position, history)
            export_labels_csv(labels_csv, history)
            continue

    save_state(state_path, position, history)
    export_labels_csv(labels_csv, history)
    print("All clips reviewed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
