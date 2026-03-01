#!/usr/bin/env python3
"""
Stage 1 — Dog Detector (Parallel YOLO11n)
=========================================

Purpose
-------
Scans video clips recursively and writes a CSV containing clips where a dog
was detected. This version is built for CPU batch processing:

- sequential frame reading
- parallel processing across CPU cores
- leaves one CPU core free by default
- 2 fps sampling by default
- YOLO11n enabled by default
- resume support by clip path
- incremental CSV writing so Ctrl+C is safe

Output CSV columns
------------------
path,dog_detected,best_confidence,dog_frame_count,total_frames_checked

Usage examples
--------------
python stage1_miner.py --clips "C:/Users/jaysh/TapoVideos/Computer Camera" --out "C:/Users/jaysh/ToiletTraining/shortlist.csv"
python stage1_miner.py --clips "C:/Users/jaysh/TapoVideos/Computer Camera" --out "C:/Users/jaysh/ToiletTraining/shortlist.csv" --workers 4
python stage1_miner.py --clips "C:/Users/jaysh/TapoVideos/Computer Camera" --out "C:/Users/jaysh/ToiletTraining/shortlist.csv" --sample-fps 2 --confidence 0.35

Notes
-----
- Each worker loads its own YOLO11n model instance. This is required on Windows
  because worker processes are spawned separately.
- This is a Stage 1 dog-presence miner, not a wee/poo classifier.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

try:
    import cv2
except ImportError:
    sys.exit("Missing dependency: pip install opencv-python")

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("Missing dependency: pip install ultralytics")


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".ts", ".wmv", ".m4v")
DOG_CLASS_ID = 16  # COCO 'dog'
DEFAULT_SAMPLE_FPS = 2.0
DEFAULT_CONFIDENCE = 0.35
DEFAULT_MODEL = "yolo11n.pt"
FIELDNAMES = [
    "path",
    "dog_detected",
    "best_confidence",
    "dog_frame_count",
    "total_frames_checked",
]

# Worker globals, set by _worker_init()
_WORKER_MODEL = None
_WORKER_CONF = DEFAULT_CONFIDENCE
_WORKER_SAMPLE_FPS = DEFAULT_SAMPLE_FPS
_WORKER_IMG_SIZE = 640
_WORKER_DEVICE = None


def find_clips(root: str | Path, extensions=VIDEO_EXTS):
    root = Path(root)
    seen = set()
    for ext in extensions:
        for p in list(root.rglob(f"*{ext}")) + list(root.rglob(f"*{ext.upper()}")):
            try:
                resolved = p.resolve()
            except Exception:
                resolved = p
            if resolved in seen:
                continue
            seen.add(resolved)
            yield p


def load_existing(out_path: str | Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    p = Path(out_path)
    if not p.exists():
        return done
    with p.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            clip_path = (row.get("path") or "").strip()
            if clip_path:
                done[clip_path] = row
    return done


def _worker_init(model_name: str, confidence: float, sample_fps: float, imgsz: int, device: str | None):
    global _WORKER_MODEL, _WORKER_CONF, _WORKER_SAMPLE_FPS, _WORKER_IMG_SIZE, _WORKER_DEVICE
    _WORKER_CONF = float(confidence)
    _WORKER_SAMPLE_FPS = float(sample_fps)
    _WORKER_IMG_SIZE = int(imgsz)
    _WORKER_DEVICE = device
    _WORKER_MODEL = YOLO(model_name)


def analyse_clip(path_str: str) -> dict:
    """
    Sequentially decode the clip, keep only sampled frames, then run YOLO11n on
    those frames in one batched call. This preserves sequential reading while
    reducing per-frame inference overhead.
    """
    global _WORKER_MODEL, _WORKER_CONF, _WORKER_SAMPLE_FPS, _WORKER_IMG_SIZE, _WORKER_DEVICE

    row = {
        "path": path_str,
        "dog_detected": False,
        "best_confidence": 0.0,
        "dog_frame_count": 0,
        "total_frames_checked": 0,
        "_error": "",
    }

    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        row["_error"] = "unreadable"
        return row

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(video_fps / max(_WORKER_SAMPLE_FPS, 0.1))))

    sampled_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            sampled_frames.append(frame)
        frame_idx += 1

    cap.release()

    if not sampled_frames:
        row["_error"] = "no_frames"
        return row

    row["total_frames_checked"] = len(sampled_frames)

    try:
        results = _WORKER_MODEL.predict(
            sampled_frames,
            classes=[DOG_CLASS_ID],
            conf=_WORKER_CONF,
            imgsz=_WORKER_IMG_SIZE,
            device=_WORKER_DEVICE,
            verbose=False,
            stream=False,
        )
    except Exception as e:
        row["_error"] = f"inference_error: {e}"
        return row

    best_conf = 0.0
    dog_frames = 0

    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        dog_frames += 1
        try:
            frame_best = max(float(c) for c in boxes.conf.tolist())
        except Exception:
            try:
                frame_best = float(boxes.conf[0])
            except Exception:
                frame_best = 0.0
        if frame_best > best_conf:
            best_conf = frame_best

    row["dog_frame_count"] = dog_frames
    row["best_confidence"] = round(best_conf, 3)
    row["dog_detected"] = bool(dog_frames > 0)
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel Stage 1 dog detector using YOLO11n.")
    p.add_argument("--clips", required=True, help="Root folder of clips")
    p.add_argument("--out", default="shortlist.csv", help="Output CSV path")
    p.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                   help=f"YOLO confidence threshold (default {DEFAULT_CONFIDENCE})")
    p.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS,
                   help=f"Frames per second to sample from each clip (default {DEFAULT_SAMPLE_FPS})")
    p.add_argument("--workers", type=int, default=None,
                   help="Worker process count (default: CPU cores minus one)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"YOLO model name/path (default {DEFAULT_MODEL})")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size (default 640)")
    p.add_argument("--device", default=None,
                   help="Optional Ultralytics device string passed to predict(), e.g. cpu or 0")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    clips_root = Path(args.clips)
    if not clips_root.exists():
        sys.exit(f"Clips folder not found: {clips_root}")

    n_workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    existing = load_existing(args.out)
    if existing:
        print(f"Resuming — {len(existing)} clips already processed.")

    clips = sorted(find_clips(clips_root))
    print(f"Found {len(clips)} unique clips under {clips_root}")

    to_process = [c for c in clips if str(c) not in existing]
    skipped_existing = len(clips) - len(to_process)
    if skipped_existing:
        print(f"Skipping {skipped_existing} already-processed clips.")
    print(f"Processing {len(to_process)} clips with {n_workers} worker(s)…\n")

    if not to_process:
        print("Nothing to do.")
        return

    out_path = Path(args.out)
    out_exists = out_path.exists()

    start = time.time()
    processed = 0
    dog_count = 0

    with out_path.open("a", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not out_exists:
            writer.writeheader()

        try:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_worker_init,
                initargs=(args.model, args.confidence, args.sample_fps, args.imgsz, args.device),
            ) as executor:
                futures = {executor.submit(analyse_clip, str(clip)): clip for clip in to_process}

                total = len(futures)
                for future in as_completed(futures):
                    processed += 1
                    clip = futures[future]

                    try:
                        row = future.result()
                    except Exception as e:
                        row = {
                            "path": str(clip),
                            "dog_detected": False,
                            "best_confidence": 0.0,
                            "dog_frame_count": 0,
                            "total_frames_checked": 0,
                            "_error": f"worker_error: {e}",
                        }

                    if row.get("dog_detected"):
                        dog_count += 1

                    writer.writerow({k: row[k] for k in FIELDNAMES})
                    out_f.flush()

                    elapsed = time.time() - start
                    rate = processed / max(elapsed, 0.1)
                    eta = (total - processed) / max(rate, 1e-9)
                    name = Path(row["path"]).name

                    if row.get("_error"):
                        tag = f"SKIP ({row['_error']})"
                    else:
                        tag = (
                            f"dog={str(row['dog_detected']).lower()}  "
                            f"best_conf={row['best_confidence']}  "
                            f"dog_frames={row['dog_frame_count']}/{row['total_frames_checked']}"
                        )

                    print(f"  [{processed}/{total}] {name}  ETA~{eta/60:.0f}m  {tag}", flush=True)

        except KeyboardInterrupt:
            print("\n\nInterrupted — progress saved.")
            return

    elapsed = time.time() - start
    print(f"\nFinished in {elapsed/60:.1f} min")
    print(f"Dog clips detected: {dog_count} / {len(to_process)}")
    print(f"CSV written to: {out_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
