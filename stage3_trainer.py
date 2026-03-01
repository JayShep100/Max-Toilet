#!/usr/bin/env python3
"""
Stage 3 — Pose-Based Classifier Trainer
=========================================
Uses your labelled clips from Stage 2 to train a classifier that understands
Max's posture rather than relying on colour heuristics.

Pipeline:
  1. For each labelled clip, extract YOLOv8-pose keypoints per frame
  2. Compute clip-level features from those keypoints
  3. Train a gradient-boosted classifier (XGBoost / sklearn)
  4. Evaluate with cross-validation
  5. Save the model for Stage 4

Key features used:
  - Normalised spine angle (horizontal = squatting)
  - Hip keypoint height relative to frame (low = squat)
  - Tail-base y-position (if detectable)
  - Duration of squatting frames in clip
  - Stillness during squat phase
  - Bounding box aspect ratio during still phase

Requirements:
    pip install ultralytics scikit-learn joblib numpy opencv-python

Usage:
    python3 stage3_trainer.py --labelled ./labelled --labels labels.csv --model dog_toilet_model.joblib
"""

import argparse
import sys
import csv
import json
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("pip install numpy")

try:
    import cv2
except ImportError:
    sys.exit("pip install opencv-python")

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("pip install ultralytics")

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report
    import joblib
except ImportError:
    sys.exit("pip install scikit-learn joblib")


SAMPLE_FPS = 3          # frames per second to sample
DOG_CLASS = 16          # COCO dog
POSE_MODEL = "yolov8n-pose.pt"   # downloads ~6 MB on first run

# YOLOv8 pose keypoint indices (COCO 17-point skeleton)
# We care about: 5,6=shoulders, 11,12=hips, 13,14=knees, 15,16=ankles
# For a dog the mapping isn't perfect but relative positions still carry signal.
KP_SHOULDER_L, KP_SHOULDER_R = 5, 6
KP_HIP_L, KP_HIP_R = 11, 12
KP_KNEE_L, KP_KNEE_R = 13, 14


def extract_clip_features(clip_path, pose_model, det_model):
    """
    Returns a feature vector (1D numpy array) for this clip, or None on failure.
    Features are clip-level summaries over sampled frames.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None

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
        if not ret:
            break
        frames.append(frame)
        idx += interval
    cap.release()

    if not frames:
        return None

    # Per-frame metrics
    hip_heights = []          # normalised (0=top, 1=bottom)
    spine_angles = []         # degrees from horizontal (0=flat/squat, 90=upright)
    bbox_aspects = []         # width/height of dog bbox
    dog_present = []          # bool per frame
    per_frame_motion = []

    prev_gray = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Motion
        if prev_gray is not None:
            motion = cv2.absdiff(prev_gray, gray).mean() / 255.0
        else:
            motion = 0.0
        per_frame_motion.append(motion)
        prev_gray = gray

        # Dog detection (bounding box)
        det_results = det_model(frame, verbose=False, classes=[DOG_CLASS], conf=0.3)
        boxes = det_results[0].boxes
        if boxes is None or len(boxes) == 0:
            dog_present.append(False)
            continue
        dog_present.append(True)

        # Largest dog box
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes.xyxy.tolist()]
        best_box = boxes.xyxy.tolist()[int(np.argmax(areas))]
        x1, y1, x2, y2 = best_box
        bw = (x2 - x1) / w_frame
        bh = (y2 - y1) / h_frame
        if bh > 0:
            bbox_aspects.append(bw / bh)
        hip_heights.append(y2 / h_frame)   # bottom of bbox as proxy for hip height

        # Pose keypoints
        pose_results = pose_model(frame, verbose=False, conf=0.25)
        kps = pose_results[0].keypoints
        if kps is not None and len(kps.xy) > 0:
            kp = kps.xy[0].cpu().numpy()  # shape (17, 2)
            # Spine angle: shoulder midpoint → hip midpoint
            sh = (kp[KP_SHOULDER_L] + kp[KP_SHOULDER_R]) / 2
            hp = (kp[KP_HIP_L] + kp[KP_HIP_R]) / 2
            if sh[0] != 0 and hp[0] != 0:
                dy = hp[1] - sh[1]
                dx = hp[0] - sh[0]
                angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
                spine_angles.append(angle)

    # ── Clip-level features ───────────────────────────────────────────────────
    def safe_mean(lst): return float(np.mean(lst)) if lst else 0.0
    def safe_std(lst):  return float(np.std(lst))  if lst else 0.0

    # Fraction of frames dog was present
    dog_frac = safe_mean([float(v) for v in dog_present])

    # Hip height stats (high value = bbox bottom low in frame = squatting)
    hip_mean = safe_mean(hip_heights)
    hip_max  = float(max(hip_heights)) if hip_heights else 0.0

    # Spine angle stats (low angle = horizontal = squatting)
    spine_mean = safe_mean(spine_angles)
    spine_min  = float(min(spine_angles)) if spine_angles else 90.0

    # Bbox aspect ratio (wide+short = squat; narrow+tall = standing)
    aspect_mean = safe_mean(bbox_aspects)
    aspect_min  = float(min(bbox_aspects)) if bbox_aspects else 0.0

    # Dwell time — frames where motion is low AND dog is present
    still_thresh = 0.003
    dwell_count = sum(
        1 for m, d in zip(per_frame_motion, dog_present)
        if d and m < still_thresh
    )
    dwell_frac = dwell_count / max(len(frames), 1)

    # Motion-then-still pattern
    n = len(per_frame_motion)
    first_q  = safe_mean(per_frame_motion[:n//4])
    middle   = safe_mean(per_frame_motion[n//4: 3*n//4])
    last_q   = safe_mean(per_frame_motion[3*n//4:])
    pattern  = 1.0 if (middle < first_q*0.6 and middle < last_q*0.6) else 0.0

    feature_vector = np.array([
        dog_frac,
        hip_mean, hip_max,
        spine_mean, spine_min,
        aspect_mean, aspect_min,
        dwell_frac,
        pattern,
        safe_mean(per_frame_motion),
        safe_std(per_frame_motion),
    ], dtype=np.float32)

    return feature_vector


FEATURE_NAMES = [
    "dog_frac",
    "hip_height_mean", "hip_height_max",
    "spine_angle_mean", "spine_angle_min",
    "bbox_aspect_mean", "bbox_aspect_min",
    "dwell_frac",
    "motion_pattern",
    "motion_mean", "motion_std",
]


def main():
    parser = argparse.ArgumentParser(description="Tapo classifier trainer — Stage 3")
    parser.add_argument("--labelled", required=True, help="Root folder with wee/poo/neither subfolders")
    parser.add_argument("--labels", required=True, help="labels.csv from Stage 2")
    parser.add_argument("--model", default="dog_toilet_model.joblib", help="Output model path")
    parser.add_argument("--features-cache", default="features_cache.json", help="Cache extracted features")
    args = parser.parse_args()

    print("Loading models…")
    det_model  = YOLO("yolov8n.pt")
    pose_model = YOLO(POSE_MODEL)

    # Load labels
    labels_map = {}
    with open(args.labels, newline="") as f:
        for row in csv.DictReader(f):
            if row["label"] in ("wee", "poo", "neither"):
                labels_map[row["path"]] = row["label"]

    # Also scan labelled folder directly
    labelled_root = Path(args.labelled)
    for label in ("wee", "poo", "neither"):
        for clip in (labelled_root / label).rglob("*"):
            if clip.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".ts"):
                labels_map.setdefault(str(clip), label)

    if not labels_map:
        sys.exit("No labelled clips found. Run Stage 2 first.")

    print(f"Found {len(labels_map)} labelled clips.")

    # Load feature cache
    cache = {}
    if Path(args.features_cache).exists():
        with open(args.features_cache) as f:
            cache = json.load(f)

    X, y = [], []
    for path_str, label in labels_map.items():
        clip = Path(path_str)
        if not clip.exists():
            # Try labelled folder
            clip = labelled_root / label / Path(path_str).name
            if not clip.exists():
                print(f"  MISSING: {path_str}")
                continue

        key = str(clip)
        if key in cache:
            feats = np.array(cache[key], dtype=np.float32)
        else:
            print(f"  Extracting features: {clip.name}")
            feats = extract_clip_features(clip, pose_model, det_model)
            if feats is None:
                print(f"    → failed, skipping")
                continue
            cache[key] = feats.tolist()
            with open(args.features_cache, "w") as f:
                json.dump(cache, f)

        X.append(feats)
        y.append(label)

    if not X:
        sys.exit("No features could be extracted.")

    X = np.array(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"\nTraining on {len(X)} clips, classes: {list(le.classes_)}")
    print(f"Class counts: { {c: int((y_enc==i).sum()) for i,c in enumerate(le.classes_)} }")

    # ── Train ─────────────────────────────────────────────────────────────────
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)

    if len(X) >= 10:
        cv = StratifiedKFold(n_splits=min(5, len(X)//3), shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y_enc, cv=cv, scoring="f1_macro")
        print(f"\nCross-val F1 (macro): {scores.mean():.3f} ± {scores.std():.3f}")

    clf.fit(X, y_enc)

    # Feature importance
    print("\nFeature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {name:<25} {imp:.3f}  {bar}")

    # Save
    model_data = {"model": clf, "label_encoder": le, "feature_names": FEATURE_NAMES}
    joblib.dump(model_data, args.model)
    print(f"\n✅ Model saved to: {args.model}")
    print(f"   Next step: python3 stage4_classifier.py --model {args.model} --clip <path>")


if __name__ == "__main__":
    main()
