#!/usr/bin/env python3
"""
Stage 4 — Classifier
=====================
Given a trained model from Stage 3, classify new incoming clips.
Outputs: label (wee / poo / neither), confidence, and timestamp of the event.

Usage:
    # Single clip
    python3 stage4_classifier.py --model dog_toilet_model.joblib --clip new_clip.mp4

    # Whole folder (watch mode)
    python3 stage4_classifier.py --model dog_toilet_model.joblib --watch /path/to/tapo/clips --out results.csv

    # Batch
    python3 stage4_classifier.py --model dog_toilet_model.joblib --clips /path/to/clips --out results.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import cv2
    from ultralytics import YOLO
    import joblib
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\npip install ultralytics scikit-learn joblib opencv-python numpy")

# Import feature extraction from Stage 3
# (If running standalone, the function is duplicated below for convenience)
try:
    from stage3_trainer import extract_clip_features, FEATURE_NAMES
except ImportError:
    # Inline fallback — same logic as stage3_trainer.py
    SAMPLE_FPS = 3
    DOG_CLASS = 16
    KP_FRONT_LEFT_PAW    = 0
    KP_FRONT_LEFT_KNEE   = 1
    KP_FRONT_LEFT_ELBOW  = 2
    KP_REAR_LEFT_PAW     = 3
    KP_REAR_LEFT_KNEE    = 4
    KP_REAR_LEFT_ELBOW   = 5
    KP_FRONT_RIGHT_PAW   = 6
    KP_FRONT_RIGHT_KNEE  = 7
    KP_FRONT_RIGHT_ELBOW = 8
    KP_REAR_RIGHT_PAW    = 9
    KP_REAR_RIGHT_KNEE   = 10
    KP_REAR_RIGHT_ELBOW  = 11
    KP_TAIL_START        = 12
    KP_TAIL_END          = 13
    KP_WITHERS           = 22

    def extract_clip_features(clip_path, pose_model, det_model):
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened(): return None
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, int(video_fps / SAMPLE_FPS))
        h_frame = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480
        w_frame = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640
        frames = []
        idx = 0
        while idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
            idx += interval
        cap.release()
        if not frames: return None

        rear_hip_heights, spine_angles, tail_angles, rear_paw_spreads = [], [], [], []
        bbox_aspects, dog_present, per_frame_motion = [], [], []
        prev_gray = None
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion = cv2.absdiff(prev_gray, gray).mean() / 255.0 if prev_gray is not None else 0.0
            per_frame_motion.append(motion)
            prev_gray = gray
            det_results = det_model(frame, verbose=False, classes=[DOG_CLASS], conf=0.3)
            boxes = det_results[0].boxes
            if boxes is None or len(boxes) == 0:
                dog_present.append(False); continue
            dog_present.append(True)
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes.xyxy.tolist()]
            best_i = int(np.argmax(areas))
            x1, y1, x2, y2 = boxes.xyxy.tolist()[best_i]
            bw = (x2-x1)/w_frame; bh = (y2-y1)/h_frame
            if bh > 0: bbox_aspects.append(bw/bh)
            pose_results = pose_model(frame, verbose=False, conf=0.25)
            kps = pose_results[0].keypoints
            if kps is not None and len(kps.xy) > 0 and best_i < len(kps.xy):
                kp = kps.xy[best_i].cpu().numpy()
                if len(kp) >= 24:
                    def _kv(i): return kp[i][0] != 0 or kp[i][1] != 0
                    rk = [kp[KP_REAR_LEFT_KNEE], kp[KP_REAR_RIGHT_KNEE]]
                    rk_v = [p for p in rk if p[0] != 0 or p[1] != 0]
                    if rk_v:
                        rear_hip_heights.append(float(np.mean([p[1]/h_frame for p in rk_v])))
                    if _kv(KP_WITHERS):
                        re_pts = [kp[KP_REAR_LEFT_ELBOW], kp[KP_REAR_RIGHT_ELBOW]]
                        re_v = [p for p in re_pts if p[0] != 0 or p[1] != 0]
                        if re_v:
                            mid = np.mean(re_v, axis=0)
                            dy = mid[1] - kp[KP_WITHERS][1]; dx = mid[0] - kp[KP_WITHERS][0]
                            spine_angles.append(float(np.degrees(np.arctan2(abs(dy), abs(dx)+1e-6))))
                    if _kv(KP_TAIL_START) and _kv(KP_TAIL_END):
                        dy = kp[KP_TAIL_END][1]-kp[KP_TAIL_START][1]
                        dx = kp[KP_TAIL_END][0]-kp[KP_TAIL_START][0]
                        tail_angles.append(float(np.degrees(np.arctan2(dy, dx+1e-6))))
                    if _kv(KP_REAR_LEFT_PAW) and _kv(KP_REAR_RIGHT_PAW):
                        rear_paw_spreads.append(abs(kp[KP_REAR_LEFT_PAW][0]-kp[KP_REAR_RIGHT_PAW][0])/w_frame)

        def safe_mean(lst, d=0.0): return float(np.mean(lst)) if lst else d
        def safe_std(lst): return float(np.std(lst)) if lst else 0.0
        dog_frac = safe_mean([float(v) for v in dog_present])
        still_thresh = 0.003
        dwell_count = sum(1 for m,d in zip(per_frame_motion, dog_present) if d and m < still_thresh)
        dwell_frac = dwell_count / max(len(frames), 1)
        n = len(per_frame_motion)
        first_q = safe_mean(per_frame_motion[:n//4]); middle = safe_mean(per_frame_motion[n//4:3*n//4]); last_q = safe_mean(per_frame_motion[3*n//4:])
        pattern = 1.0 if (middle < first_q*0.6 and middle < last_q*0.6) else 0.0
        return np.array([
            dog_frac,
            safe_mean(rear_hip_heights), max(rear_hip_heights) if rear_hip_heights else 0.0,
            safe_mean(spine_angles, 90.0), min(spine_angles) if spine_angles else 90.0,
            safe_mean(tail_angles), safe_mean(rear_paw_spreads),
            safe_mean(bbox_aspects, 1.0), min(bbox_aspects) if bbox_aspects else 0.0,
            dwell_frac, pattern, safe_mean(per_frame_motion), safe_std(per_frame_motion),
        ], dtype=np.float32)

    FEATURE_NAMES = [
        "dog_frac",
        "rear_hip_height_mean", "rear_hip_height_max",
        "spine_angle_mean", "spine_angle_min",
        "tail_angle_mean",
        "rear_paw_spread",
        "bbox_aspect_mean", "bbox_aspect_min",
        "dwell_frac", "motion_pattern", "motion_mean", "motion_std",
    ]


def find_event_timestamp(clip_path, det_model, pose_model):
    """
    Return approximate timestamp (seconds) of the toilet event within the clip.
    Heuristic: first frame where dog is present AND posture is low AND motion is low.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened(): return None
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h_frame = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480
    interval = max(1, int(video_fps / 3))
    prev_gray = None
    idx = 0
    best_ts = None
    while idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = cv2.absdiff(prev_gray, gray).mean() / 255.0 if prev_gray is not None else 1.0
        prev_gray = gray
        if motion < 0.005:
            det = det_model(frame, verbose=False, classes=[16], conf=0.3)
            boxes = det[0].boxes
            if boxes is not None and len(boxes):
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes.xyxy.tolist()]
                x1, y1, x2, y2 = boxes.xyxy.tolist()[int(np.argmax(areas))]
                if y2 / h_frame > 0.55:   # dog low in frame
                    best_ts = idx / video_fps
                    break
        idx += interval
    cap.release()
    return best_ts


def classify_clip(clip_path, model_data, det_model, pose_model, find_ts=True):
    clf = model_data["model"]
    le  = model_data["label_encoder"]
    feats = extract_clip_features(clip_path, pose_model, det_model)
    if feats is None:
        return {"label": "error", "confidence": 0.0, "timestamp_s": None}
    proba = clf.predict_proba(feats.reshape(1, -1))[0]
    idx   = int(np.argmax(proba))
    label = le.inverse_transform([idx])[0]
    conf  = float(proba[idx])
    ts    = None
    if find_ts and label in ("wee", "poo"):
        ts = find_event_timestamp(clip_path, det_model, pose_model)
    return {"label": label, "confidence": round(conf, 3), "timestamp_s": ts}


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".ts"}


def main():
    parser = argparse.ArgumentParser(description="Tapo toilet-event classifier — Stage 4")
    parser.add_argument("--model", required=True, help="Model file from Stage 3")
    parser.add_argument("--clip", help="Single clip to classify")
    parser.add_argument("--clips", help="Folder of clips to classify (batch)")
    parser.add_argument("--watch", help="Folder to watch for new clips (continuous)")
    parser.add_argument("--out", default="results.csv", help="Output CSV for batch/watch mode")
    parser.add_argument("--no-timestamp", action="store_true", help="Skip event timestamp detection (faster)")
    args = parser.parse_args()

    print("Loading models…")
    model_data = joblib.load(args.model)
    det_model  = YOLO("yolov8n.pt")
    pose_model = YOLO("dog_pose_model.pt")
    print("Ready.\n")

    find_ts = not args.no_timestamp

    if args.clip:
        # Single clip mode
        r = classify_clip(Path(args.clip), model_data, det_model, pose_model, find_ts)
        print(f"Result: {r['label'].upper()}  (confidence: {r['confidence']:.1%})")
        if r["timestamp_s"] is not None:
            m, s = divmod(int(r["timestamp_s"]), 60)
            print(f"Event at: {m:02d}:{s:02d}")

    elif args.clips or args.watch:
        root = Path(args.clips or args.watch)
        watch_mode = bool(args.watch)
        processed = set()

        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["clip", "label", "confidence", "timestamp_s"])
            writer.writeheader()

            def process_clip(clip):
                if str(clip) in processed:
                    return
                processed.add(str(clip))
                print(f"  Classifying: {clip.name}", end="  ", flush=True)
                r = classify_clip(clip, model_data, det_model, pose_model, find_ts)
                print(f"→ {r['label'].upper()} ({r['confidence']:.0%})")
                writer.writerow({"clip": str(clip), **r})
                f.flush()

            if watch_mode:
                print(f"Watching: {root}  (Ctrl+C to stop)\n")
                while True:
                    for clip in root.rglob("*"):
                        if clip.suffix.lower() in VIDEO_EXTENSIONS:
                            process_clip(clip)
                    time.sleep(10)
            else:
                clips = [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]
                print(f"Found {len(clips)} clips in {root}\n")
                for clip in sorted(clips):
                    process_clip(clip)

        print(f"\n✅ Results saved to: {args.out}")
    else:
        parser.error("Specify --clip, --clips, or --watch")


if __name__ == "__main__":
    main()
