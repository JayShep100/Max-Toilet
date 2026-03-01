#!/usr/bin/env python3
"""
smart_reviewer.py – ML-assisted review of dog toilet pad video clips.

Plays unreviewed clips, overlays an ML prediction, and records the user's
confirmed label.  After each review session the accuracy of the model is
printed so the user can see how well the classifier is performing.

Usage examples::

    # First run – trains from existing labels and starts review
    python smart_reviewer.py --pad-clips pad_clips_combined.csv \\
        --shortlist shortlist.csv \\
        --dest-root C:\\Users\\jaysh\\ToiletTraining

    # Resume where you left off (skips already-reviewed clips)
    python smart_reviewer.py --pad-clips pad_clips_combined.csv \\
        --shortlist shortlist.csv \\
        --dest-root C:\\Users\\jaysh\\ToiletTraining

    # Force-retrain the model after adding more labels
    python smart_reviewer.py --pad-clips pad_clips_combined.csv \\
        --shortlist shortlist.csv \\
        --dest-root C:\\Users\\jaysh\\ToiletTraining --retrain

    # Only show clips where the dog was on the pad
    python smart_reviewer.py --pad-clips pad_clips_combined.csv \\
        --shortlist shortlist.csv \\
        --dest-root C:\\Users\\jaysh\\ToiletTraining --only-on-pad

    # Copy files to label folders instead of moving them
    python smart_reviewer.py --pad-clips pad_clips_combined.csv \\
        --shortlist shortlist.csv \\
        --dest-root C:\\Users\\jaysh\\ToiletTraining --copy

    # Dry-run: show predictions without touching any files
    python smart_reviewer.py --pad-clips pad_clips_combined.csv \\
        --shortlist shortlist.csv \\
        --dest-root C:\\Users\\jaysh\\ToiletTraining --dry-run

Dependencies
------------
Required: scikit-learn, opencv-python (GUI build), numpy, joblib
Optional: ultralytics (enables YOLO-based dog bounding-box features)

Note: Use ``opencv-python`` (not ``opencv-python-headless``) when running
this script locally so that ``cv2.imshow`` is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_SAMPLES_FOR_TRAINING = 10
MODEL_FILENAME = "smart_reviewer_model.joblib"
SCALER_FILENAME = "smart_reviewer_scaler.joblib"
STATE_FILENAME = "smart_review_state.json"
LOG_FILENAME = "smart_review_log.csv"

#: Valid label strings used throughout the module.
LABELS = ("wee", "poo", "neither")

#: Video file extensions considered when scanning label folders.
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# BGR colour used for the on-screen prediction overlay.
_COLOUR_HIGH = (0, 200, 0)    # green  – confidence ≥ 70 %
_COLOUR_MED  = (0, 200, 200)  # yellow – confidence ≥ 40 %
_COLOUR_LOW  = (0, 0, 200)    # red    – confidence <  40 %

# Regex matching filenames like "2026-02-22 14-21-36" or "2026-02-22T14-21-36"
_CLIP_FILENAME_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[\sT](\d{2})-(\d{2})-(\d{2})"
)

# ---------------------------------------------------------------------------
# Dog-Pose 24-keypoint skeleton constants (Ultralytics dog-pose dataset)
# ---------------------------------------------------------------------------
_DOG_CLASS = 16  # COCO class index for "dog" (used in fallback detection)

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
KP_LEFT_EAR_BASE     = 14
KP_RIGHT_EAR_BASE    = 15
KP_NOSE              = 16
KP_CHIN              = 17
KP_LEFT_EAR_TIP      = 18
KP_RIGHT_EAR_TIP     = 19
KP_LEFT_EYE          = 20
KP_RIGHT_EYE         = 21
KP_WITHERS           = 22
KP_THROAT            = 23

# Feature vector column names (used for documentation / debugging).
FEATURE_NAMES = [
    # CSV features
    "best_confidence",
    "dog_frame_ratio",
    "best_overlap",
    "pad_frame_count",
    # Time features
    "hour_of_day",
    "time_since_last_s",
    # Video duration
    "duration_seconds",
    # Dog-pose features
    "rear_hip_height_mean",
    "rear_hip_height_max",
    "spine_angle_mean",
    "spine_angle_min",
    "tail_angle_mean",
    "tail_height_mean",
    "rear_paw_spread",
    "front_rear_height_diff",
    "dwell_frac",
    "motion_pattern",
    "bbox_aspect_mean",
    # Bounding-box features (also populated in fallback mode)
    "bbox_cx",
    "bbox_cy",
    "bbox_size",
    "bbox_movement",
]

# CSV columns for the accuracy log.
_LOG_COLUMNS = [
    "path",
    "predicted_label",
    "predicted_confidence",
    "actual_label",
    "was_correct",
    "timestamp",
]


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def parse_clip_timestamp(filename: str) -> Optional[datetime]:
    """Parse a UTC ``datetime`` from a clip filename.

    Handles filenames of the form ``2026-02-22 14-21-36.mp4`` as well as the
    ISO-8601-ish variant ``2026-02-22T14-21-36.mp4``.

    Returns ``None`` when no match is found.
    """
    stem = Path(filename).stem
    m = _CLIP_FILENAME_RE.search(stem)
    if m:
        try:
            return datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4)), int(m.group(5)), int(m.group(6)),
                tzinfo=timezone.utc,
            )
        except ValueError:
            pass
    return None


def extract_csv_features(row: dict) -> dict:
    """Extract numeric features from a CSV metadata row.

    Missing or non-numeric values fall back to ``0.0`` so the returned dict
    always contains all four keys.
    """

    def _float(val, default: float = 0.0) -> float:
        try:
            return float(val) if val not in (None, "") else default
        except (ValueError, TypeError):
            return default

    dog_count = _float(row.get("dog_frame_count"))
    total = _float(row.get("total_frames_checked"), default=1.0)

    return {
        "best_confidence": _float(row.get("best_confidence")),
        "dog_frame_ratio": dog_count / max(total, 1.0),
        "best_overlap":    _float(row.get("best_overlap")),
        "pad_frame_count": _float(row.get("pad_frame_count")),
    }


def extract_time_features(
    filename: str,
    all_timestamps: list[datetime],
) -> dict:
    """Extract time-based features from a clip filename.

    Parameters
    ----------
    filename:
        Basename of the clip (e.g. ``"2026-02-22 14-21-36.mp4"``).
    all_timestamps:
        Sorted list of UTC datetimes for all clips in the dataset – used to
        compute *time since last clip*.

    Returns a dict with ``hour_of_day`` and ``time_since_last_s`` keys.
    """
    features = {"hour_of_day": 12.0, "time_since_last_s": 0.0}
    ts = parse_clip_timestamp(filename)
    if ts is None:
        return features

    features["hour_of_day"] = float(ts.hour)
    prev = [t for t in all_timestamps if t < ts]
    if prev:
        diff = (ts - max(prev)).total_seconds()
        features["time_since_last_s"] = min(diff, 86400.0)  # cap at 24 h
    return features


def _extract_dog_pose_features(
    cap: "cv2.VideoCapture",
    total_frames: int,
    features: dict,
    pose_model=None,
) -> None:
    """Extract dog bbox and pose-keypoint statistics from a sparse frame sample.

    When *pose_model* is a loaded YOLO dog-pose model (24 keypoints) it also
    computes anatomy-based pose features.  When *pose_model* is ``None`` the
    function falls back to ``yolov8n.pt`` for bounding-box-only detection.

    Modifies *features* in-place.
    """
    try:
        from ultralytics import YOLO  # optional dependency
    except ImportError:
        return

    has_pose = pose_model is not None

    if not has_pose:
        try:
            det_model = YOLO("yolov8n.pt")
        except Exception:  # noqa: BLE001
            return
    else:
        det_model = None

    step = max(1, total_frames // 20)  # sample ≤ 20 frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cx_list: list[float] = []
    cy_list: list[float] = []
    sz_list: list[float] = []

    rear_hip_heights: list[float] = []
    spine_angles:     list[float] = []
    tail_angles:      list[float] = []
    tail_heights:     list[float] = []
    rear_paw_spreads: list[float] = []
    front_rear_diffs: list[float] = []
    bbox_aspects:     list[float] = []

    per_frame_motion: list[float] = []
    dog_present:      list[bool]  = []
    prev_gray: Optional[np.ndarray] = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            motion = (
                float(cv2.absdiff(gray, prev_gray).mean() / 255.0)
                if prev_gray is not None else 0.0
            )
            per_frame_motion.append(motion)
            prev_gray = gray

            if has_pose:
                results  = pose_model(frame, verbose=False, conf=0.25)
                boxes    = results[0].boxes
                kps_obj  = results[0].keypoints
            else:
                results  = det_model(frame, verbose=False, classes=[_DOG_CLASS])
                boxes    = results[0].boxes
                kps_obj  = None

            if boxes is None or len(boxes) == 0:
                dog_present.append(False)
                frame_idx += 1
                continue
            dog_present.append(True)

            # Largest detection
            areas   = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes.xyxy.tolist()]
            best_i  = int(np.argmax(areas))
            x1, y1, x2, y2 = boxes.xyxy.tolist()[best_i]

            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            sz = (x2 - x1) * (y2 - y1) / (w * h)
            cx_list.append(cx)
            cy_list.append(cy)
            sz_list.append(sz)

            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            if bh > 0:
                bbox_aspects.append(bw / bh)

            # ── Pose keypoints (dog-pose 24-kp model only) ────────────────
            if (
                has_pose
                and kps_obj is not None
                and len(kps_obj.xy) > 0
                and best_i < len(kps_obj.xy)
            ):
                kp = kps_obj.xy[best_i].cpu().numpy()  # (24, 2)
                if len(kp) >= 24:
                    def _kv(i: int) -> bool:  # keypoint visible?
                        return bool(kp[i][0] != 0 or kp[i][1] != 0)

                    # 1. Rear hip height — y of rear knees (4, 10)
                    rk = [kp[KP_REAR_LEFT_KNEE], kp[KP_REAR_RIGHT_KNEE]]
                    rk_v = [p for p in rk if p[0] != 0 or p[1] != 0]
                    if rk_v:
                        rear_hip_heights.append(
                            float(np.mean([p[1] / h for p in rk_v]))
                        )

                    # 2. Spine angle — withers (22) → midpoint of rear elbows (5,11)
                    if _kv(KP_WITHERS):
                        re_pts = [kp[KP_REAR_LEFT_ELBOW], kp[KP_REAR_RIGHT_ELBOW]]
                        re_v   = [p for p in re_pts if p[0] != 0 or p[1] != 0]
                        if re_v:
                            mid = np.mean(re_v, axis=0)
                            dy  = mid[1] - kp[KP_WITHERS][1]
                            dx  = mid[0] - kp[KP_WITHERS][0]
                            spine_angles.append(
                                float(np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-6)))
                            )

                    # 3+4. Tail angle and tail height
                    if _kv(KP_TAIL_START) and _kv(KP_TAIL_END):
                        dy = kp[KP_TAIL_END][1] - kp[KP_TAIL_START][1]
                        dx = kp[KP_TAIL_END][0] - kp[KP_TAIL_START][0]
                        tail_angles.append(
                            float(np.degrees(np.arctan2(dy, dx + 1e-6)))
                        )
                        tail_heights.append(float(kp[KP_TAIL_END][1] / h))

                    # 5. Rear paw spread — horizontal distance between rear paws (3, 9)
                    if _kv(KP_REAR_LEFT_PAW) and _kv(KP_REAR_RIGHT_PAW):
                        rear_paw_spreads.append(
                            abs(kp[KP_REAR_LEFT_PAW][0] - kp[KP_REAR_RIGHT_PAW][0]) / w
                        )

                    # 6. Front-rear height diff — positive = rear lower (squatting)
                    fp_v = [p for p in [kp[KP_FRONT_LEFT_PAW], kp[KP_FRONT_RIGHT_PAW]]
                            if p[0] != 0 or p[1] != 0]
                    rp_v = [p for p in [kp[KP_REAR_LEFT_PAW],  kp[KP_REAR_RIGHT_PAW]]
                            if p[0] != 0 or p[1] != 0]
                    if fp_v and rp_v:
                        front_rear_diffs.append(
                            float(np.mean([p[1]/h for p in rp_v]))
                            - float(np.mean([p[1]/h for p in fp_v]))
                        )

        frame_idx += 1

    # ── Dwell fraction ────────────────────────────────────────────────────────
    still_thresh = 0.003
    dwell_count  = sum(
        1 for m, d in zip(per_frame_motion, dog_present) if d and m < still_thresh
    )
    dwell_frac = dwell_count / max(len(per_frame_motion), 1)

    # ── Motion pattern (walk-still-walk) ────────────────────────────────────
    n = len(per_frame_motion)
    if n >= 4:
        first_q = float(np.mean(per_frame_motion[:n//4]))
        middle  = float(np.mean(per_frame_motion[n//4: 3*n//4]))
        last_q  = float(np.mean(per_frame_motion[3*n//4:]))
        motion_pattern = 1.0 if (middle < first_q * 0.6 and middle < last_q * 0.6) else 0.0
    else:
        motion_pattern = 0.0

    def _sm(lst: list, default: float = 0.0) -> float:
        return float(np.mean(lst)) if lst else default

    # Bbox features
    if cx_list:
        features["bbox_cx"]       = _sm(cx_list)
        features["bbox_cy"]       = _sm(cy_list)
        features["bbox_size"]     = _sm(sz_list)
        features["bbox_movement"] = float(np.var(cx_list) + np.var(cy_list))

    # Pose features (populated regardless; defaults remain when lists are empty)
    features["bbox_aspect_mean"]       = _sm(bbox_aspects, default=1.0)
    features["dwell_frac"]             = dwell_frac
    features["motion_pattern"]         = motion_pattern
    features["rear_hip_height_mean"]   = _sm(rear_hip_heights)
    features["rear_hip_height_max"]    = float(max(rear_hip_heights)) if rear_hip_heights else 0.0
    features["spine_angle_mean"]       = _sm(spine_angles, default=45.0)
    features["spine_angle_min"]        = float(min(spine_angles)) if spine_angles else 45.0
    features["tail_angle_mean"]        = _sm(tail_angles)
    features["tail_height_mean"]       = _sm(tail_heights, default=0.5)
    features["rear_paw_spread"]        = _sm(rear_paw_spreads)
    features["front_rear_height_diff"] = _sm(front_rear_diffs)


def _extract_motion_features(
    cap: "cv2.VideoCapture",
    total_frames: int,
    features: dict,
) -> None:
    """Approximate movement using inter-frame pixel differences.

    Modifies *features* in-place.  Used as a fallback when YOLO is not
    available.
    """
    step = max(1, total_frames // 20)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    prev_gray: Optional[np.ndarray] = None
    motion_vals: list[float] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_vals.append(float(np.mean(diff)))
            prev_gray = gray
        frame_idx += 1

    if motion_vals:
        features["bbox_movement"] = float(np.mean(motion_vals))


def extract_video_features(video_path: Path, pose_model=None) -> dict:
    """Extract duration and bounding-box / pose statistics from a video file.

    When *pose_model* is a loaded YOLO dog-pose model it extracts 24-keypoint
    anatomy features.  Falls back to ``yolov8n.pt`` bbox detection (and then
    to motion analysis) when *pose_model* is ``None``.

    Returns a dict with safe defaults so callers never need to guard against
    missing keys.
    """
    defaults: dict = {
        "duration_seconds":     0.0,
        # Bbox features
        "bbox_cx":              0.5,
        "bbox_cy":              0.5,
        "bbox_size":            0.1,
        "bbox_movement":        0.0,
        # Pose features
        "rear_hip_height_mean":   0.0,
        "rear_hip_height_max":    0.0,
        "spine_angle_mean":       45.0,
        "spine_angle_min":        45.0,
        "tail_angle_mean":        0.0,
        "tail_height_mean":       0.5,
        "rear_paw_spread":        0.0,
        "front_rear_height_diff": 0.0,
        "dwell_frac":             0.0,
        "motion_pattern":         0.0,
        "bbox_aspect_mean":       1.0,
    }

    if not video_path.exists():
        return defaults

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return defaults

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        features = dict(defaults)
        features["duration_seconds"] = total_frames / fps if fps > 0 else 0.0

        try:
            _extract_dog_pose_features(cap, total_frames, features, pose_model)
        except Exception:  # noqa: BLE001
            _extract_motion_features(cap, total_frames, features)

        cap.release()
        return features

    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not extract video features from %s: %s", video_path, exc)
        return defaults


def build_feature_vector(
    row: dict,
    filename: str,
    all_timestamps: list[datetime],
    video_path: Optional[Path] = None,
    pose_model=None,
) -> list[float]:
    """Combine all feature sources into a flat numeric vector.

    The order matches :data:`FEATURE_NAMES`.
    """
    csv_f  = extract_csv_features(row)
    time_f = extract_time_features(filename, all_timestamps)
    vid_f  = (
        extract_video_features(video_path, pose_model)
        if video_path else {
            "duration_seconds":     0.0,
            "bbox_cx":              0.5,
            "bbox_cy":              0.5,
            "bbox_size":            0.1,
            "bbox_movement":        0.0,
            "rear_hip_height_mean":   0.0,
            "rear_hip_height_max":    0.0,
            "spine_angle_mean":       45.0,
            "spine_angle_min":        45.0,
            "tail_angle_mean":        0.0,
            "tail_height_mean":       0.5,
            "rear_paw_spread":        0.0,
            "front_rear_height_diff": 0.0,
            "dwell_frac":             0.0,
            "motion_pattern":         0.0,
            "bbox_aspect_mean":       1.0,
        }
    )

    return [
        csv_f["best_confidence"],
        csv_f["dog_frame_ratio"],
        csv_f["best_overlap"],
        csv_f["pad_frame_count"],
        time_f["hour_of_day"],
        time_f["time_since_last_s"],
        vid_f["duration_seconds"],
        vid_f["rear_hip_height_mean"],
        vid_f["rear_hip_height_max"],
        vid_f["spine_angle_mean"],
        vid_f["spine_angle_min"],
        vid_f["tail_angle_mean"],
        vid_f["tail_height_mean"],
        vid_f["rear_paw_spread"],
        vid_f["front_rear_height_diff"],
        vid_f["dwell_frac"],
        vid_f["motion_pattern"],
        vid_f["bbox_aspect_mean"],
        vid_f["bbox_cx"],
        vid_f["bbox_cy"],
        vid_f["bbox_size"],
        vid_f["bbox_movement"],
    ]


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------


def _basename(path_str: str) -> str:
    """Return the bare filename from a path string, handling both separators."""
    # pathlib on Linux does not split on backslashes, so do it manually.
    return Path(path_str.replace("\\", "/")).name


def load_labels(dest_root: Path) -> list[tuple[str, str]]:
    """Collect all existing labels from the destination root folder.

    Labels are sourced from (in priority order):

    1. ``<dest_root>/dog_labels.csv`` – columns ``path`` and ``label``.
    2. Files inside ``<dest_root>/Wee/`` → label ``"wee"``.
    3. Files inside ``<dest_root>/Poo/`` → label ``"poo"``.
    4. Files inside ``<dest_root>/Neither/`` → label ``"neither"``.

    Folder-based labels override CSV labels when filenames conflict.

    Returns
    -------
    list of (filename, label) tuples
        *filename* is the bare filename (no directory component).
    """
    labeled: dict[str, str] = {}

    # 1. CSV
    csv_path = dest_root / "dog_labels.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                path_val  = (row.get("path")  or "").strip()
                label_val = (row.get("label") or "").strip().lower()
                if path_val and label_val in LABELS:
                    labeled[_basename(path_val)] = label_val

    # 2–4. Subdirectories
    for label in LABELS:
        folder = dest_root / label.capitalize()
        if folder.exists():
            for f in folder.iterdir():
                if f.is_file() and f.suffix.lower() in _VIDEO_EXTENSIONS:
                    labeled[f.name] = label

    return list(labeled.items())


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------


def load_metadata(
    pad_clips_csv: Path,
    shortlist_csv: Optional[Path],
) -> dict[str, dict]:
    """Load clip metadata from one or two CSV files.

    Rows from *shortlist_csv* augment (but do not override) rows already found
    in *pad_clips_csv*.

    Returns
    -------
    dict mapping bare filename → CSV row dict
    """
    metadata: dict[str, dict] = {}

    def _ingest(path: Path) -> None:
        if not path.exists():
            logger.warning("Metadata CSV not found: %s", path)
            return
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                for col in ("path", "clip_path", "file_path", "filename"):
                    raw = (row.get(col) or "").strip()
                    if raw:
                        fname = _basename(raw)
                        metadata.setdefault(fname, dict(row))
                        break

    _ingest(pad_clips_csv)
    if shortlist_csv:
        _ingest(shortlist_csv)

    return metadata


def get_all_timestamps(metadata: dict[str, dict]) -> list[datetime]:
    """Return sorted UTC timestamps parsed from every key in *metadata*."""
    timestamps = []
    for filename in metadata:
        ts = parse_clip_timestamp(filename)
        if ts is not None:
            timestamps.append(ts)
    return sorted(timestamps)


# ---------------------------------------------------------------------------
# ML model helpers
# ---------------------------------------------------------------------------


def train_model(
    labels: list[tuple[str, str]],
    metadata: dict[str, dict],
    all_timestamps: list[datetime],
    video_root: Optional[Path] = None,
    verbose: bool = False,
    pose_model=None,
    extra_X: Optional[list[list[float]]] = None,
    extra_y: Optional[list[str]] = None,
):
    """Train a :class:`~sklearn.ensemble.RandomForestClassifier`.

    Parameters
    ----------
    labels:
        List of ``(filename, label)`` pairs from :func:`load_labels`.
    metadata:
        Dict mapping filename → CSV row, from :func:`load_metadata`.
    all_timestamps:
        Sorted timestamps for time-feature computation.
    video_root:
        Root path used to locate video files for feature extraction.  When
        *None*, video features are set to their defaults.
    verbose:
        When ``True``, print per-sample progress and training statistics.

    Returns
    -------
    (model, scaler) or (None, None)
        Returns ``(None, None)`` when fewer than
        :data:`MIN_SAMPLES_FOR_TRAINING` labeled samples are available.
    """
    import time

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    X: list[list[float]] = []
    y: list[str] = []

    total = len(labels)
    succeeded = 0
    for i, (filename, label) in enumerate(labels, 1):
        row = metadata.get(filename, {})
        video_path: Optional[Path] = None

        if video_root is not None:
            # Look in every known label subfolder and the root itself.
            candidates = [
                video_root / lbl.capitalize() / filename
                for lbl in LABELS
            ] + [video_root / filename]
            for c in candidates:
                if c.exists():
                    video_path = c
                    break

        if verbose:
            disp = f"{label}/{filename}" if video_root is None else (
                str(video_path.relative_to(video_root)) if video_path else filename
            )
            print(f"  [{i}/{total}] Processing {disp} ...", end=" ", flush=True)
            t0 = time.monotonic()

        X.append(build_feature_vector(row, filename, all_timestamps, video_path, pose_model))
        y.append(label)
        succeeded += 1

        if verbose:
            print(f"done ({time.monotonic() - t0:.1f}s)")

    if verbose:
        print(f"  → Feature extraction complete. {succeeded}/{total} succeeded.")

    # Append pre-computed features from the current review session (online learning).
    if extra_X:
        X.extend(extra_X)
        y.extend(extra_y or [])

    if len(X) < MIN_SAMPLES_FOR_TRAINING:
        msg = (
            f"Too few labeled samples ({len(X)}) to train a model "
            f"(minimum required: {MIN_SAMPLES_FOR_TRAINING})."
        )
        logger.info(msg)
        if verbose:
            print(f"  ⚠  {msg}")
        return None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
    )

    if verbose:
        print("  → Using Random Forest classifier")
        n_classes = len(set(y))
        print(f"  → Training on {len(X)} samples ({n_classes} classes)")

    model.fit(X_scaled, y)

    if verbose:
        try:
            cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(X)), scoring="accuracy")
            cv_acc = float(cv_scores.mean()) * 100
            print(f"  → Cross-validation accuracy: {cv_acc:.1f}%")
        except Exception:  # noqa: BLE001
            print("  → Cross-validation skipped (dataset too small or single-class fold)")

    class_counts = {lbl: y.count(lbl) for lbl in set(y)}
    logger.info("Trained model on %d samples: %s", len(X), class_counts)
    return model, scaler


def predict(model, scaler, feature_vector: list[float]) -> tuple[str, float]:
    """Return ``(predicted_label, confidence)`` for a single feature vector."""
    X = scaler.transform([feature_vector])
    label: str = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    idx = list(model.classes_).index(label)
    return label, float(proba[idx])


def save_model(model, scaler, model_path: Path, scaler_path: Path) -> None:
    """Persist *model* and *scaler* to disk using :mod:`joblib`."""
    import joblib

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info("Model saved to %s", model_path)


def load_model(model_path: Path, scaler_path: Path):
    """Load a previously saved model and scaler.

    Returns ``(None, None)`` when the files do not exist or cannot be read.
    """
    import joblib

    if model_path.exists() and scaler_path.exists():
        try:
            model  = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            logger.info("Loaded model from %s", model_path)
            return model, scaler
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load saved model: %s", exc)
    return None, None


# ---------------------------------------------------------------------------
# Accuracy log helpers
# ---------------------------------------------------------------------------


def append_log_entry(
    log_path: Path,
    clip_path: str,
    predicted_label: str,
    predicted_confidence: float,
    actual_label: str,
) -> None:
    """Append a single review result to the accuracy log CSV."""
    write_header = not log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "path":                  clip_path,
            "predicted_label":       predicted_label,
            "predicted_confidence":  f"{predicted_confidence:.4f}",
            "actual_label":          actual_label,
            "was_correct":           str(predicted_label == actual_label),
            "timestamp":             datetime.now(timezone.utc).isoformat(),
        })


def load_log(log_path: Path) -> list[dict]:
    """Return all rows from the accuracy log, or ``[]`` if not found."""
    if not log_path.exists():
        return []
    with open(log_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def print_accuracy_stats(log_entries: list[dict]) -> None:
    """Print a human-readable accuracy summary to stdout.

    Example output::

        Model accuracy so far: 15/20 (75%) — Poo: 3/5, Neither: 4/5, Wee: 8/10
    """
    if not log_entries:
        return

    total   = len(log_entries)
    correct = sum(1 for e in log_entries if e.get("was_correct") == "True")
    pct     = correct / total * 100

    per_label: dict[str, list[int]] = {}
    for e in log_entries:
        lbl = e.get("actual_label", "")
        if lbl not in per_label:
            per_label[lbl] = [0, 0]  # [correct, total]
        per_label[lbl][1] += 1
        if e.get("was_correct") == "True":
            per_label[lbl][0] += 1

    label_parts = [
        f"{lbl.capitalize()}: {c}/{t}"
        for lbl, (c, t) in sorted(per_label.items())
    ]
    print(
        f"\nModel accuracy so far: {correct}/{total} ({pct:.0f}%) — "
        + ", ".join(label_parts)
    )


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def load_state(state_path: Path) -> dict:
    """Load review progress from *state_path*.

    Returns an empty state dict when the file is absent or malformed.
    """
    if state_path.exists():
        try:
            with open(state_path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load state file %s: %s", state_path, exc)
    return {"reviewed": [], "skipped": []}


def save_state(state_path: Path, state: dict) -> None:
    """Persist *state* to *state_path* as JSON."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


# ---------------------------------------------------------------------------
# Video playback
# ---------------------------------------------------------------------------


def _prediction_colour(confidence: float) -> tuple[int, int, int]:
    """Return the BGR overlay colour matching a confidence level."""
    if confidence >= 0.7:
        return _COLOUR_HIGH
    if confidence >= 0.4:
        return _COLOUR_MED
    return _COLOUR_LOW


def play_clip_and_get_label(
    video_path: Path,
    predicted_label: Optional[str],
    predicted_confidence: float,
    clip_index: int,
    total_clips: int,
) -> Optional[str]:
    """Play a video clip in a loop and wait for a keyboard label.

    The clip loops continuously until the user presses one of:

    * **W** – wee
    * **P** – poo
    * **N** – neither
    * **S** – skip
    * **B** – go back to the previous clip
    * **Q** – quit the session

    Returns the key string (``"wee"``, ``"poo"``, ``"neither"``,
    ``"skip"``, ``"back"``, ``"quit"``), or ``None`` when the video
    cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay_ms = max(1, int(1000 / fps))
    window_name = "Smart Reviewer"
    result: Optional[str] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            h, w = frame.shape[:2]

            # Top bar – filename and keyboard hints
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"Clip {clip_index}/{total_clips}: {video_path.name}",
                (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "W=wee  P=poo  N=neither  S=skip  B=back  Q=quit",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
                cv2.LINE_AA,
            )

            # Bottom bar – ML prediction
            cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
            if predicted_label is not None:
                colour = _prediction_colour(predicted_confidence)
                pct    = int(predicted_confidence * 100)
                text   = f"PREDICTED: {predicted_label.upper()} ({pct}%)"
            else:
                colour = (120, 120, 120)
                text   = "NOT ENOUGH DATA"
            cv2.putText(
                frame, text,
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(delay_ms) & 0xFF

            if key in (ord("w"), ord("W")):
                result = "wee"
                break
            elif key in (ord("p"), ord("P")):
                result = "poo"
                break
            elif key in (ord("n"), ord("N")):
                result = "neither"
                break
            elif key in (ord("s"), ord("S")):
                result = "skip"
                break
            elif key in (ord("b"), ord("B")):
                result = "back"
                break
            elif key in (ord("q"), ord("Q")):
                result = "quit"
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return result


# ---------------------------------------------------------------------------
# File management
# ---------------------------------------------------------------------------


def move_or_copy_clip(
    src: Path,
    dest_root: Path,
    label: str,
    copy: bool = False,
    dry_run: bool = False,
) -> Path:
    """Move or copy *src* into ``<dest_root>/<Label>/``.

    When *dry_run* is ``True`` the operation is only logged, not performed.
    """
    label_folder = dest_root / label.capitalize()
    dest = label_folder / src.name

    if dry_run:
        action = "copy" if copy else "move"
        logger.info("[DRY RUN] Would %s %s → %s", action, src, dest)
        return dest

    label_folder.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dest)
    else:
        shutil.move(str(src), dest)
    return dest


# ---------------------------------------------------------------------------
# Unreviewed clip discovery
# ---------------------------------------------------------------------------


def find_unreviewed_clips(
    pad_clips_csv: Path,
    shortlist_csv: Optional[Path],
    dest_root: Path,
    only_on_pad: bool,
    state: dict,
) -> list[tuple[Path, dict]]:
    """Return ``(video_path, metadata_row)`` pairs for clips not yet reviewed.

    A clip is considered *already done* if:

    * its filename appears in ``state["reviewed"]`` or ``state["skipped"]``, or
    * a file with that name already exists in one of the label subfolders.

    When *only_on_pad* is ``True``, clips that have no pad presence signal
    (``on_pad`` field or non-zero ``pad_frame_count``) are excluded.
    """
    metadata = load_metadata(pad_clips_csv, shortlist_csv)

    # Build the set of already-handled filenames.
    done: set[str] = set()
    for key in ("reviewed", "skipped"):
        for p in state.get(key, []):
            done.add(_basename(p))
    for label in LABELS:
        folder = dest_root / label.capitalize()
        if folder.exists():
            for f in folder.iterdir():
                done.add(f.name)

    clips: list[tuple[Path, dict]] = []

    for filename, row in sorted(metadata.items()):
        if filename in done:
            continue

        if only_on_pad:
            on_pad_str = str(row.get("on_pad", "")).strip().lower()
            pad_count  = 0.0
            try:
                pad_count = float(row.get("pad_frame_count") or 0)
            except (ValueError, TypeError):
                pass
            if on_pad_str not in ("true", "1", "yes") and pad_count <= 0:
                continue

        # Resolve video file path.
        video_path: Optional[Path] = None
        raw_path = (
            row.get("path") or row.get("clip_path") or row.get("file_path") or ""
        ).strip()
        if raw_path:
            candidate = Path(raw_path.replace("\\", "/"))
            if candidate.exists():
                video_path = candidate
        if video_path is None:
            candidate = dest_root / filename
            if candidate.exists():
                video_path = candidate

        if video_path is None:
            logger.debug("Video file not found for '%s', skipping.", filename)
            continue

        clips.append((video_path, row))

    return clips


# ---------------------------------------------------------------------------
# Main review loop
# ---------------------------------------------------------------------------


_BANNER = """\
============================================================
  SMART REVIEWER — ML-Assisted Video Clip Labeller
============================================================
"""


def _print_retrain_box(n_total: int, n_new: int, session_y: list[str]) -> None:
    """Print a framed notice after the model is retrained online."""
    wee     = session_y.count("wee")
    poo     = session_y.count("poo")
    neither = session_y.count("neither")
    line1 = f"  MODEL RETRAINED — {n_total} total samples ({n_new} new this session)  "
    line2 = f"  Classes: wee={wee}, poo={poo}, neither={neither}  "
    width = max(len(line1), len(line2)) + 2
    bar   = "═" * width
    print(f"\n  ╔{bar}╗")
    print(f"  ║{line1:<{width}}║")
    print(f"  ║{line2:<{width}}║")
    print(f"  ╚{bar}╝\n")


def run_review(
    pad_clips_csv: Path,
    shortlist_csv: Optional[Path],
    dest_root: Path,
    only_on_pad: bool = False,
    copy: bool = False,
    dry_run: bool = False,
    retrain: bool = False,
    model_path: Path = Path(MODEL_FILENAME),
    scaler_path: Path = Path(SCALER_FILENAME),
    state_path: Path = Path(STATE_FILENAME),
    log_path: Path = Path(LOG_FILENAME),
    pose_model_path: Path = Path("dog_pose_model.pt"),
    retrain_interval: int = 10,
) -> None:
    """Orchestrate the full review session.

    1. Loads (or trains) the ML model.
    2. Discovers unreviewed clips.
    3. Plays each clip and records the user's label.
    4. Moves/copies the clip to the appropriate label folder.
    5. Persists state and logs accuracy after every decision.
    6. Periodically retrains the model with newly labelled clips (online
       learning) when *retrain_interval* > 0.
    """
    print(_BANNER)
    state = load_state(state_path)

    # Build metadata index and timestamp list for feature engineering.
    metadata       = load_metadata(pad_clips_csv, shortlist_csv)
    all_timestamps = get_all_timestamps(metadata)

    # ------------------------------------------------------------------
    # STEP 1: Scan labelled folders
    # ------------------------------------------------------------------
    print("[STEP 1/5] Scanning labelled folders...")
    labels = load_labels(dest_root)
    label_counts: dict[str, int] = {lbl: 0 for lbl in LABELS}
    for _, lbl in labels:
        if lbl in label_counts:
            label_counts[lbl] += 1
    for lbl in LABELS:
        print(f"  \u2192 Found {label_counts[lbl]} videos in {lbl}/")
    print(f"  \u2192 Total labelled: {len(labels)} videos")

    # ------------------------------------------------------------------
    # Load dog-pose model once (shared across all feature extractions)
    # ------------------------------------------------------------------
    pose_model = None
    if pose_model_path.exists():
        try:
            from ultralytics import YOLO as _YOLO  # optional
            pose_model = _YOLO(str(pose_model_path))
            print(f"  → Dog-pose model loaded from {pose_model_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠  Could not load dog-pose model ({exc}). Falling back to yolov8n.pt bbox detection (no pose features).")
    else:
        print(
            f"  ⚠  Dog-pose model not found ({pose_model_path}) "
            "— falling back to yolov8n.pt bbox detection."
        )

    # ------------------------------------------------------------------
    # STEP 2 + 3: Extract features and train (or load) the model
    # ------------------------------------------------------------------
    model, scaler = None, None
    if not retrain:
        model, scaler = load_model(model_path, scaler_path)
        if model is not None:
            print("[STEP 2/5] Feature extraction skipped (loaded saved model).")
            print(f"[STEP 3/5] ML model loaded from {model_path}")

    if model is None:
        print("[STEP 2/5] Extracting features from labelled videos...")
        if labels:
            print("[STEP 3/5] Training ML model...")
            model, scaler = train_model(
                labels, metadata, all_timestamps, dest_root,
                verbose=True, pose_model=pose_model,
            )
            if model is not None:
                save_model(model, scaler, model_path, scaler_path)
                print(f"  \u2192 Model saved to {model_path}")
            else:
                print(
                    f"  \u26a0  Not enough labelled data "
                    f"(need \u2265 {MIN_SAMPLES_FOR_TRAINING} samples)."
                )
        else:
            print("  \u26a0  No labelled videos found \u2014 skipping training.")

    # ------------------------------------------------------------------
    # STEP 4: Find unlabelled clips
    # ------------------------------------------------------------------
    print("[STEP 4/5] Finding unlabelled clips...")
    if pad_clips_csv.exists():
        print(f"  \u2192 Scanning {pad_clips_csv.name} ...")
    if shortlist_csv and shortlist_csv.exists():
        print(f"  \u2192 Scanning {shortlist_csv.name} ...")

    clips = find_unreviewed_clips(
        pad_clips_csv, shortlist_csv, dest_root, only_on_pad, state
    )
    print(f"  \u2192 Found {len(clips)} unlabelled clips")

    if not clips:
        print("\nNo unreviewed clips found.")
        return

    # ------------------------------------------------------------------
    # STEP 5: Review / classify clips
    # ------------------------------------------------------------------
    if model:
        print("[STEP 5/5] Reviewing clips (ML predictions active)...")
    else:
        print(
            f"[STEP 5/5] Reviewing clips "
            f"(no ML model \u2014 need \u2265 {MIN_SAMPLES_FOR_TRAINING} labeled samples)..."
        )

    log_entries = load_log(log_path)
    idx = 0

    # Online learning: track features + labels from this session.
    session_X: list[list[float]] = []
    session_y: list[str]         = []

    while idx < len(clips):
        video_path, row = clips[idx]
        filename = video_path.name

        predicted_label: Optional[str] = None
        predicted_confidence            = 0.0
        cached_fv: Optional[list[float]] = None
        if model is not None and scaler is not None:
            try:
                cached_fv = build_feature_vector(
                    row, filename, all_timestamps, video_path, pose_model
                )
                predicted_label, predicted_confidence = predict(model, scaler, cached_fv)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Prediction failed for %s: %s", filename, exc)

        if predicted_label is not None:
            pct = int(predicted_confidence * 100)
            print(
                f"  [{idx + 1}/{len(clips)}] {filename} "
                f"\u2192 PREDICTED: {predicted_label.upper()} (confidence: {pct}%)",
                end=" ",
                flush=True,
            )
        else:
            print(f"  [{idx + 1}/{len(clips)}] {filename}", end=" ", flush=True)

        result = play_clip_and_get_label(
            video_path,
            predicted_label,
            predicted_confidence,
            idx + 1,
            len(clips),
        )

        if result == "quit":
            print("Quitting review.")
            break
        elif result == "back":
            idx = max(0, idx - 1)
            continue
        elif result == "skip":
            print("\u2192 skipped")
            state.setdefault("skipped", []).append(str(video_path))
            save_state(state_path, state)
            idx += 1
            continue
        elif result in LABELS:
            action = "copied" if copy else "moved"
            if dry_run:
                action = "dry-run"
            print(f"\u2192 labelled: {result.upper()} ... {action} to {result}/")

            if not dry_run:
                try:
                    move_or_copy_clip(video_path, dest_root, result, copy, dry_run)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to %s file: %s",
                        "copy" if copy else "move", exc,
                    )

            append_log_entry(
                log_path,
                str(video_path),
                predicted_label or "none",
                predicted_confidence,
                result,
            )
            log_entries.append({
                "actual_label":    result,
                "predicted_label": predicted_label or "none",
                "was_correct":     str((predicted_label or "none") == result),
            })

            # Store feature vector for online learning.
            if cached_fv is not None:
                session_X.append(cached_fv)
                session_y.append(result)

            state.setdefault("reviewed", []).append(str(video_path))
            save_state(state_path, state)
            print_accuracy_stats(log_entries)

            # ── Online learning: periodic retrain ─────────────────────────
            n_labelled = len(session_y)
            if (
                retrain_interval > 0
                and n_labelled > 0
                and n_labelled % retrain_interval == 0
                and session_X
            ):
                new_model, new_scaler = train_model(
                    labels, metadata, all_timestamps, dest_root,
                    verbose=False, pose_model=pose_model,
                    extra_X=session_X, extra_y=session_y,
                )
                if new_model is not None:
                    model  = new_model
                    scaler = new_scaler
                    save_model(model, scaler, model_path, scaler_path)
                    _print_retrain_box(
                        n_total=len(labels) + len(session_X),
                        n_new=len(session_X),
                        session_y=session_y,
                    )

            idx += 1
        else:
            # None → video could not be opened; advance anyway.
            print("\u2192 could not open video, skipping.")
            idx += 1

    print(f"\nReview complete. Processed {idx} clip(s).")
    print_accuracy_stats(log_entries)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart_reviewer.py",
        description="ML-assisted review of dog toilet pad video clips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pad-clips", required=True, type=Path,
        help="CSV file with pad clip metadata (e.g. pad_clips_combined.csv).",
    )
    parser.add_argument(
        "--shortlist", default=None, type=Path,
        help="CSV file from stage-1 YOLO mining (e.g. shortlist.csv).",
    )
    parser.add_argument(
        "--dest-root", required=True, type=Path,
        help="Root folder containing Wee/, Poo/, Neither/ label subfolders.",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force retraining even if a saved model already exists.",
    )
    parser.add_argument(
        "--only-on-pad", action="store_true",
        help="Only review clips where the dog was detected on the pad.",
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy clips to label folders instead of moving them.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show predictions without moving or copying any files.",
    )
    parser.add_argument(
        "--model-path", type=Path, default=Path(MODEL_FILENAME),
        help="Path to save/load the trained model.",
    )
    parser.add_argument(
        "--scaler-path", type=Path, default=Path(SCALER_FILENAME),
        help="Path to save/load the feature scaler.",
    )
    parser.add_argument(
        "--state-path", type=Path, default=Path(STATE_FILENAME),
        help="Path to save/load review progress.",
    )
    parser.add_argument(
        "--log-path", type=Path, default=Path(LOG_FILENAME),
        help="Path to the accuracy log CSV.",
    )
    parser.add_argument(
        "--pose-model", type=Path, default=Path("dog_pose_model.pt"),
        dest="pose_model",
        help="Path to the trained dog-pose YOLO model weights (default: dog_pose_model.pt). "
             "If the file does not exist, falls back to yolov8n.pt bbox detection.",
    )
    parser.add_argument(
        "--retrain-interval", type=int, default=10,
        dest="retrain_interval",
        help="Retrain the model every N newly-labelled clips (online learning). "
             "Set to 0 to disable (default: 10).",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args()
    run_review(
        pad_clips_csv=args.pad_clips,
        shortlist_csv=args.shortlist,
        dest_root=args.dest_root,
        only_on_pad=args.only_on_pad,
        copy=args.copy,
        dry_run=args.dry_run,
        retrain=args.retrain,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        state_path=args.state_path,
        log_path=args.log_path,
        pose_model_path=args.pose_model,
        retrain_interval=args.retrain_interval,
    )


if __name__ == "__main__":
    main()
