#!/usr/bin/env python3
"""
Stage 1 — Dog Detector
=======================
Scans all clips and keeps only those where a dog is detected.
That's it. No heuristics, no guessing, no false poo/wee labels.

Uses YOLOv8n — a fast, lightweight object detector that knows what a dog
looks like. Samples a few frames per clip and flags the clip if a dog
appears in any of them.

You then review the dog clips manually in Stage 2 and label them yourself.

Requirements:
    pip install ultralytics opencv-python

Usage:
    python stage1_miner.py --clips "C:\\path\\to\\clips" --out shortlist.csv
    python stage1_miner.py --clips "C:\\path\\to\\clips" --out shortlist.csv --workers 4
    python stage1_miner.py --clips "C:\\path\\to\\clips" --out shortlist.csv --confidence 0.4
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

try:
    import cv2
except ImportError:
    sys.exit("Missing: pip install opencv-python")

try:
    import numpy as np
except ImportError:
    sys.exit("Missing: pip install numpy")

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("Missing: pip install ultralytics")


# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_FPS   = 2     # frames per second to check
DOG_CLASS    = 16    # COCO class ID for 'dog'
MIN_CONF     = 0.35  # YOLO confidence threshold

FIELDNAMES   = ["path", "dog_detected", "best_confidence", "dog_frame_count", "total_frames_checked"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_clips(root, extensions=(".mp4", ".avi", ".mov", ".mkv", ".ts")):
    root = Path(root)
    seen = set()
    for ext in extensions:
        for p in list(root.rglob(f"*{ext}")) + list(root.rglob(f"*{ext.upper()}")):
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield p


def load_existing(out_path):
    done = {}
    if Path(out_path).exists():
        with open(out_path, newline="") as f:
            for row in csv.DictReader(f):
                done[row["path"]] = row
    return done


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dog detector — Stage 1")
    parser.add_argument("--clips",      required=True, help="Root folder of clips")
    parser.add_argument("--out",        default="shortlist.csv")
    parser.add_argument("--confidence", type=float, default=MIN_CONF,
                        help=f"YOLO confidence threshold (default {MIN_CONF})")
    args = parser.parse_args()

    # Load YOLO
    print("Loading YOLOv8n (downloads ~6 MB on first run)…")
    model = YOLO("yolov8n.pt")
    print("Model ready.\n")

    # Resume support
    existing = load_existing(args.out)
    if existing:
        print(f"Resuming — {len(existing)} clips already processed.")

    clips = sorted(find_clips(args.clips))
    print(f"Found {len(clips)} unique clips")

    to_process = [c for c in clips if str(c) not in existing]
    if len(clips) - len(to_process):
        print(f"Skipping {len(clips) - len(to_process)} already done.")
    print(f"Processing {len(to_process)} clips…\n")

    if not to_process:
        print("Nothing to do.")
        return

    # Open output CSV
    out_exists = Path(args.out).exists()
    out_f  = open(args.out, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES, extrasaction="ignore")
    if not out_exists:
        writer.writeheader()

    start      = time.time()
    total      = len(to_process)
    dog_count  = 0

    try:
        for i, clip in enumerate(to_process, 1):
            elapsed = time.time() - start
            rate    = i / max(elapsed, 0.1)
            eta     = (total - i) / rate

            cap = cv2.VideoCapture(str(clip))
            if not cap.isOpened():
                print(f"  [{i}/{total}] {clip.name}  ETA~{eta/60:.0f}m  SKIP (unreadable)", flush=True)
                continue

            video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25
            interval     = max(1, int(video_fps / SAMPLE_FPS))
            frame_idx    = 0
            frames_checked = 0
            dog_frames   = 0
            best_conf    = 0.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % interval == 0:
                    frames_checked += 1
                    results = model(frame, verbose=False, classes=[DOG_CLASS],
                                    conf=args.confidence)
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes):
                        dog_frames += 1
                        conf = float(boxes.conf.max())
                        best_conf = max(best_conf, conf)
                frame_idx += 1

            cap.release()

            dog_detected = dog_frames > 0

            row = {
                "path":                str(clip),
                "dog_detected":        dog_detected,
                "best_confidence":     round(best_conf, 3),
                "dog_frame_count":     dog_frames,
                "total_frames_checked": frames_checked,
            }
            writer.writerow(row)
            out_f.flush()

            if dog_detected:
                dog_count += 1
                tag = f"DOG FOUND  conf={best_conf:.0%}  ({dog_frames}/{frames_checked} frames)"
            else:
                tag = f"no dog     ({frames_checked} frames checked)"

            print(f"  [{i}/{total}] {clip.name}  ETA~{eta/60:.0f}m  {tag}", flush=True)

    except KeyboardInterrupt:
        print("\n\nInterrupted — progress saved.")
    finally:
        out_f.close()

    elapsed = time.time() - start
    print(f"\nFinished in {elapsed/60:.1f} min")
    print(f"Clips with dog detected: {dog_count} / {total}")
    print(f"\nNext step:")
    print(f"  python stage2_reviewer.py --shortlist {args.out}")


if __name__ == "__main__":
    main()
