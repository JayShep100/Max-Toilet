#!/usr/bin/env python3
"""
Stage 2 — Pad Presence Detector
=================================
Before running detection, shows you the first frame of your first clip
and lets you draw the pad ROI with your mouse. The coordinates are saved
to roi.txt so you only need to do this once.

Controls in ROI setup window:
  Click + drag  — draw the pad rectangle
  Enter         — confirm and start detection
  R             — redraw
  Q             — quit

Requirements:
    pip install ultralytics opencv-python

Usage:
    python stage2_pad_detector.py --shortlist shortlist.csv --out pad_clips.csv
    python stage2_pad_detector.py --shortlist shortlist.csv --out pad_clips.csv --reset-roi
"""

import argparse
import csv
import time
from pathlib import Path

try:
    import cv2
except ImportError:
    import sys; sys.exit("Missing: pip install opencv-python")

try:
    from ultralytics import YOLO
except ImportError:
    import sys; sys.exit("Missing: pip install ultralytics")


# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_FPS = 2
DOG_CLASS  = 16
DOG_CONF   = 0.30
ROI_FILE   = "roi.txt"

FIELDNAMES = ["path", "on_pad", "best_overlap", "pad_frame_count",
              "dog_frame_count", "total_frames_checked", "best_confidence"]


# ─────────────────────────────────────────────────────────────────────────────
# Interactive ROI picker
# ─────────────────────────────────────────────────────────────────────────────

def pick_roi(frame):
    """
    Opens an OpenCV window showing the frame.
    User clicks and drags to draw a rectangle.
    Returns (x1, y1, x2, y2) or None if cancelled.
    """
    clone    = frame.copy()
    drawing  = False
    start    = (0, 0)
    end      = (0, 0)
    rect     = [None]   # use list so mouse callback can write to it

    INSTRUCTIONS = [
        "Draw the toilet pad area",
        "Click + drag to draw",
        "ENTER to confirm  |  R to redraw  |  Q to quit",
    ]

    def draw_overlay(img, s, e, confirmed=False):
        out = img.copy()
        # Dim the whole frame slightly
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
        # Draw rectangle
        if s != e:
            color = (0, 220, 0) if confirmed else (0, 200, 255)
            x1, y1 = min(s[0], e[0]), min(s[1], e[1])
            x2, y2 = max(s[0], e[0]), max(s[1], e[1])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(out, (x1, y1), (x2, y2), (*color[::-1], 40), -1)
            size_text = f"{x2-x1} x {y2-y1}  ({x1},{y1}) → ({x2},{y2})"
            cv2.putText(out, size_text, (x1, max(y1-8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        # Instructions bar
        bar_h = 28 * len(INSTRUCTIONS) + 12
        cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (0, 0, 0), -1)
        for i, line in enumerate(INSTRUCTIONS):
            cv2.putText(out, line, (10, 22 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1, cv2.LINE_AA)
        return out

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, start, end
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start   = (x, y)
            end     = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            end = (x, y)
            cv2.imshow("Set Pad ROI", draw_overlay(clone, start, end))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end     = (x, y)
            rect[0] = (min(start[0], end[0]), min(start[1], end[1]),
                       max(start[0], end[0]), max(start[1], end[1]))
            cv2.imshow("Set Pad ROI", draw_overlay(clone, start, end))

    cv2.namedWindow("Set Pad ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Set Pad ROI", min(frame.shape[1], 1280),
                                    min(frame.shape[0], 720))
    cv2.setMouseCallback("Set Pad ROI", on_mouse)
    cv2.imshow("Set Pad ROI", draw_overlay(clone, (0, 0), (0, 0)))

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13 or key == ord('\r'):   # Enter — confirm
            if rect[0] and rect[0][2] - rect[0][0] > 10 and rect[0][3] - rect[0][1] > 10:
                cv2.destroyAllWindows()
                return rect[0]
            else:
                print("  Draw a box first, then press Enter.")
        elif key == ord('r') or key == ord('R'):  # Redraw
            rect[0] = None
            start = end = (0, 0)
            cv2.imshow("Set Pad ROI", draw_overlay(clone, (0, 0), (0, 0)))
        elif key == ord('q') or key == ord('Q'):  # Quit
            cv2.destroyAllWindows()
            return None


def setup_roi(first_clip_path):
    """Open the first frame of the first clip and let the user draw the ROI."""
    cap = cv2.VideoCapture(first_clip_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Could not read frame from {first_clip_path}")
        return None

    print("\n── ROI Setup ────────────────────────────────────────────────")
    print("  A window will open showing a frame from your first clip.")
    print("  Draw a rectangle around the toilet pad, then press ENTER.")
    print("─────────────────────────────────────────────────────────────\n")

    roi = pick_roi(frame)
    if roi is None:
        return None

    x1, y1, x2, y2 = roi
    print(f"  Pad ROI set: ({x1}, {y1}) → ({x2}, {y2})  "
          f"[{x2-x1} x {y2-y1} px]")

    # Save for future runs
    with open(ROI_FILE, "w") as f:
        f.write(f"{x1},{y1},{x2},{y2}\n")
    print(f"  Saved to {ROI_FILE} — won't ask again next run.\n")

    return (x1, y1, x2, y2)


def load_roi():
    if not Path(ROI_FILE).exists():
        return None
    try:
        parts = Path(ROI_FILE).read_text().strip().split(",")
        return tuple(int(p) for p in parts)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Overlap calculation
# ─────────────────────────────────────────────────────────────────────────────

def box_overlap_fraction(bx1, by1, bx2, by2, pad):
    px1, py1, px2, py2 = pad
    ix1 = max(bx1, px1)
    iy1 = max(by1, py1)
    ix2 = min(bx2, px2)
    iy2 = min(by2, py2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    pad_area     = (px2 - px1) * (py2 - py1)
    return intersection / pad_area


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    parser = argparse.ArgumentParser(description="Pad presence detector — Stage 2")
    parser.add_argument("--shortlist",  required=True,
                        help="shortlist.csv from Stage 1 (dog-detected clips)")
    parser.add_argument("--out",        default="pad_clips.csv")
    parser.add_argument("--overlap",    type=float, default=0.05,
                        help="Min fraction of pad covered by dog box (default 0.05)")
    parser.add_argument("--reset-roi",  action="store_true",
                        help="Ignore saved ROI and draw a new one")
    args = parser.parse_args()

    # Load dog clips from Stage 1
    if not Path(args.shortlist).exists():
        print(f"Shortlist not found: {args.shortlist}")
        return

    with open(args.shortlist, newline="") as f:
        all_rows = list(csv.DictReader(f))

    dog_clips = [r["path"] for r in all_rows
                 if r.get("dog_detected", "").strip().lower() == "true"
                 and Path(r["path"]).exists()]

    print(f"Dog clips from Stage 1: {len(dog_clips)}")

    if not dog_clips:
        print("No clips to process.")
        return

    # ROI setup
    pad = None
    if not args.reset_roi:
        pad = load_roi()
        if pad:
            print(f"Using saved ROI: {pad}  (run with --reset-roi to change)\n")

    if pad is None:
        pad = setup_roi(dog_clips[0])
        if pad is None:
            print("ROI setup cancelled.")
            return

    # Resume support
    existing = load_existing(args.out)
    if existing:
        print(f"Resuming — {len(existing)} already processed.")

    to_process = [p for p in dog_clips if p not in existing]
    print(f"Processing {len(to_process)} clips…\n")

    if not to_process:
        print("Nothing to do.")
        return

    print("Loading YOLOv8n…")
    model = YOLO("yolov8n.pt")
    print("Model ready.\n")

    out_exists = Path(args.out).exists()
    out_f  = open(args.out, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES, extrasaction="ignore")
    if not out_exists:
        writer.writeheader()

    start     = time.time()
    total     = len(to_process)
    pad_count = 0

    try:
        for i, clip_path in enumerate(to_process, 1):
            elapsed = time.time() - start
            rate    = i / max(elapsed, 0.1)
            eta     = (total - i) / rate

            cap = cv2.VideoCapture(clip_path)
            if not cap.isOpened():
                print(f"  [{i}/{total}] {Path(clip_path).name}  SKIP (unreadable)",
                      flush=True)
                continue

            video_fps      = cap.get(cv2.CAP_PROP_FPS) or 25
            interval       = max(1, int(video_fps / SAMPLE_FPS))
            frame_idx      = 0
            frames_checked = 0
            dog_frames     = 0
            pad_frames     = 0
            best_overlap   = 0.0
            best_conf      = 0.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % interval == 0:
                    frames_checked += 1
                    results = model(frame, verbose=False,
                                    classes=[DOG_CLASS], conf=DOG_CONF)
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes):
                        dog_frames += 1
                        for box in boxes:
                            conf = float(box.conf[0])
                            best_conf = max(best_conf, conf)
                            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                            overlap = box_overlap_fraction(bx1, by1, bx2, by2, pad)
                            best_overlap = max(best_overlap, overlap)
                            if overlap >= args.overlap:
                                pad_frames += 1
                                break
                frame_idx += 1

            cap.release()

            on_pad = pad_frames > 0

            row = {
                "path":                 clip_path,
                "on_pad":               on_pad,
                "best_overlap":         round(best_overlap, 3),
                "pad_frame_count":      pad_frames,
                "dog_frame_count":      dog_frames,
                "total_frames_checked": frames_checked,
                "best_confidence":      round(best_conf, 3),
            }
            writer.writerow(row)
            out_f.flush()

            if on_pad:
                pad_count += 1
                tag = (f"ON PAD  overlap={best_overlap:.0%}  "
                       f"({pad_frames} pad frames / {dog_frames} dog frames)")
            else:
                tag = f"not on pad  (dog in {dog_frames}/{frames_checked} frames)"

            print(f"  [{i}/{total}] {Path(clip_path).name}  "
                  f"ETA~{eta/60:.0f}m  {tag}", flush=True)

    except KeyboardInterrupt:
        print("\n\nInterrupted — progress saved.")
    finally:
        out_f.close()

    elapsed = time.time() - start
    print(f"\nFinished in {elapsed/60:.1f} min")
    print(f"Clips with Max on pad: {pad_count} / {total}")
    print(f"\nNext step:")
    print(f"  python stage3_labeller.py --pad-clips {args.out}")


if __name__ == "__main__":
    main()
