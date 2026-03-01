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

# Feature vector column names (used for documentation / debugging).
FEATURE_NAMES = [
    "best_confidence",
    "dog_frame_ratio",
    "best_overlap",
    "pad_frame_count",
    "hour_of_day",
    "time_since_last_s",
    "duration_seconds",
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


def _extract_yolo_features(
    cap: "cv2.VideoCapture",
    total_frames: int,
    features: dict,
) -> None:
    """Run YOLOv8n on a sparse sample of frames to get dog bbox statistics.

    Modifies *features* in-place.  Class 16 is "dog" in the COCO dataset.
    """
    try:
        from ultralytics import YOLO  # optional dependency
    except ImportError:
        return

    model = YOLO("yolov8n.pt")
    step = max(1, total_frames // 20)  # sample ≤ 20 frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cx_list: list[float] = []
    cy_list: list[float] = []
    sz_list: list[float] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            h, w = frame.shape[:2]
            results = model(frame, verbose=False, classes=[16])
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx_list.append((x1 + x2) / 2 / w)
                cy_list.append((y1 + y2) / 2 / h)
                sz_list.append((x2 - x1) * (y2 - y1) / (w * h))
        frame_idx += 1

    if cx_list:
        features["bbox_cx"]       = float(np.mean(cx_list))
        features["bbox_cy"]       = float(np.mean(cy_list))
        features["bbox_size"]     = float(np.mean(sz_list))
        features["bbox_movement"] = float(np.var(cx_list) + np.var(cy_list))


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


def extract_video_features(video_path: Path) -> dict:
    """Extract duration and bounding-box statistics from a video file.

    Tries to use YOLOv8 for accurate dog detection; falls back to motion
    analysis when ``ultralytics`` is not installed or fails.

    Returns a dict with safe defaults so callers never need to guard against
    missing keys.
    """
    defaults: dict = {
        "duration_seconds": 0.0,
        "bbox_cx":          0.5,
        "bbox_cy":          0.5,
        "bbox_size":        0.1,
        "bbox_movement":    0.0,
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
            _extract_yolo_features(cap, total_frames, features)
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
) -> list[float]:
    """Combine all feature sources into a flat numeric vector.

    The order matches :data:`FEATURE_NAMES`.
    """
    csv_f  = extract_csv_features(row)
    time_f = extract_time_features(filename, all_timestamps)
    vid_f  = extract_video_features(video_path) if video_path else {
        "duration_seconds": 0.0,
        "bbox_cx":          0.5,
        "bbox_cy":          0.5,
        "bbox_size":        0.1,
        "bbox_movement":    0.0,
    }

    return [
        csv_f["best_confidence"],
        csv_f["dog_frame_ratio"],
        csv_f["best_overlap"],
        csv_f["pad_frame_count"],
        time_f["hour_of_day"],
        time_f["time_since_last_s"],
        vid_f["duration_seconds"],
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

    Returns
    -------
    (model, scaler) or (None, None)
        Returns ``(None, None)`` when fewer than
        :data:`MIN_SAMPLES_FOR_TRAINING` labeled samples are available.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    X: list[list[float]] = []
    y: list[str] = []

    for filename, label in labels:
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

        X.append(build_feature_vector(row, filename, all_timestamps, video_path))
        y.append(label)

    if len(X) < MIN_SAMPLES_FOR_TRAINING:
        logger.info(
            "Too few labeled samples (%d) to train a model "
            "(minimum required: %d).",
            len(X), MIN_SAMPLES_FOR_TRAINING,
        )
        return None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_scaled, y)

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
) -> None:
    """Orchestrate the full review session.

    1. Loads (or trains) the ML model.
    2. Discovers unreviewed clips.
    3. Plays each clip and records the user's label.
    4. Moves/copies the clip to the appropriate label folder.
    5. Persists state and logs accuracy after every decision.
    """
    state = load_state(state_path)

    # Build metadata index and timestamp list for feature engineering.
    metadata       = load_metadata(pad_clips_csv, shortlist_csv)
    all_timestamps = get_all_timestamps(metadata)

    # Load or train model.
    model, scaler = None, None
    if not retrain:
        model, scaler = load_model(model_path, scaler_path)

    if model is None:
        labels = load_labels(dest_root)
        if labels:
            model, scaler = train_model(labels, metadata, all_timestamps, dest_root)
            if model is not None:
                save_model(model, scaler, model_path, scaler_path)

    clips = find_unreviewed_clips(
        pad_clips_csv, shortlist_csv, dest_root, only_on_pad, state
    )

    if not clips:
        print("No unreviewed clips found.")
        return

    print(f"Found {len(clips)} clip(s) to review.")
    if model:
        print("ML model active — predictions will be shown.")
    else:
        print(
            f"ML model not available "
            f"(need ≥ {MIN_SAMPLES_FOR_TRAINING} labeled samples)."
        )

    log_entries = load_log(log_path)
    idx = 0

    while idx < len(clips):
        video_path, row = clips[idx]
        filename = video_path.name

        predicted_label: Optional[str] = None
        predicted_confidence            = 0.0
        if model is not None and scaler is not None:
            try:
                fv = build_feature_vector(row, filename, all_timestamps, video_path)
                predicted_label, predicted_confidence = predict(model, scaler, fv)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Prediction failed for %s: %s", filename, exc)

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
            state.setdefault("skipped", []).append(str(video_path))
            save_state(state_path, state)
            idx += 1
            continue
        elif result in LABELS:
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

            state.setdefault("reviewed", []).append(str(video_path))
            save_state(state_path, state)
            print_accuracy_stats(log_entries)
            idx += 1
        else:
            # None → video could not be opened; advance anyway.
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
    )


if __name__ == "__main__":
    main()
