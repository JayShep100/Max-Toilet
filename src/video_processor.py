"""
video_processor.py – Process a local video file through the detection pipeline.

When historical recordings are downloaded from the camera, this module feeds
every frame through :class:`~detector.PadDetector` and logs any detected
toilet events via :class:`~logger.EventLogger`.

The timestamp attached to each logged event is derived from the recording's
known start time plus the frame's position within the video, so the log
accurately reflects *when* the event happened rather than when it was processed.

When no explicit start time is provided, the module attempts to extract one
from the video file's metadata, filename, or filesystem modification time.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import cv2

from .detector import DetectorConfig, PadDetector
from .logger import EventLogger

logger = logging.getLogger(__name__)

# Common timestamp patterns found in camera-generated filenames
_FILENAME_PATTERNS = [
    # YYYYMMDD_HHMMSS or YYYYMMDD-HHMMSS (e.g. 20250601_143022.mp4)
    (re.compile(r"(\d{4})(\d{2})(\d{2})[_\-](\d{2})(\d{2})(\d{2})"), None),
    # YYYY-MM-DD_HH-MM-SS or YYYY-MM-DDTHH:MM:SS (ISO-ish)
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})[T_](\d{2})[:\-](\d{2})[:\-](\d{2})"), None),
    # epoch-epoch.mp4 (cloud downloader naming: 1748736000-1748739600.mp4)
    (re.compile(r"^(\d{9,10})-\d{9,10}"), "epoch"),
    # Single epoch timestamp (1748736000.mp4)
    (re.compile(r"^(\d{9,10})$"), "epoch_single"),
    # YYYYMMDD only – treat as midnight UTC
    (re.compile(r"(\d{4})(\d{2})(\d{2})"), "date_only"),
]


def extract_video_timestamp(video_path: str) -> Optional[datetime]:
    """
    Try to determine the recording start time of a local video file.

    The function attempts the following sources in order:

    1. **Filename** – parse common date/time patterns embedded in the filename.
    2. **File modification time** – use the file-system ``mtime`` as a last
       resort.

    Returns
    -------
    datetime (UTC-aware) or None
        The best-effort recording start time, or *None* if no timestamp could
        be determined.
    """
    path = Path(video_path)
    stem = path.stem  # filename without extension

    # 1. Filename parsing
    for pattern, kind in _FILENAME_PATTERNS:
        m = pattern.search(stem)
        if m:
            try:
                if kind == "epoch":
                    ts = datetime.fromtimestamp(int(m.group(1)), tz=timezone.utc)
                    logger.debug("Timestamp from filename (epoch): %s", ts)
                    return ts
                if kind == "epoch_single":
                    ts = datetime.fromtimestamp(int(m.group(1)), tz=timezone.utc)
                    logger.debug("Timestamp from filename (epoch single): %s", ts)
                    return ts
                if kind == "date_only":
                    ts = datetime(
                        int(m.group(1)), int(m.group(2)), int(m.group(3)),
                        tzinfo=timezone.utc,
                    )
                    logger.debug("Timestamp from filename (date only): %s", ts)
                    return ts
                # Full datetime groups
                ts = datetime(
                    int(m.group(1)), int(m.group(2)), int(m.group(3)),
                    int(m.group(4)), int(m.group(5)), int(m.group(6)),
                    tzinfo=timezone.utc,
                )
                logger.debug("Timestamp from filename: %s", ts)
                return ts
            except (ValueError, OSError):
                continue

    # 2. File modification time
    try:
        mtime = os.path.getmtime(video_path)
        ts = datetime.fromtimestamp(mtime, tz=timezone.utc)
        logger.debug("Timestamp from file mtime: %s", ts)
        return ts
    except OSError:
        pass

    return None


def process_video_file(
    video_path: str,
    detector: PadDetector,
    event_logger: EventLogger,
    recording_start: Optional[datetime] = None,
    source_label: str = "",
) -> int:
    """
    Read every frame of a local video file and run it through *detector*.
    Any :class:`~detector.DetectionEvent` produced is written to *event_logger*.

    Parameters
    ----------
    video_path:
        Absolute or relative path to the MP4 / video file.
    detector:
        A (typically fresh) :class:`~detector.PadDetector` instance.
    event_logger:
        The :class:`~logger.EventLogger` to write events to.
    recording_start:
        UTC datetime for the first frame of the recording.  If provided,
        each event's timestamp is calculated as
        ``recording_start + elapsed_seconds``.  If *None*, the current
        clock time is used for every event.
    source_label:
        Optional label included in log messages (e.g. the original filename).

    Returns
    -------
    int
        Number of events detected and logged.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    label = source_label or path.name
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video file: {video_path}")

    # Auto-extract timestamp from the video file when none is provided
    if recording_start is None:
        recording_start = extract_video_timestamp(video_path)
        if recording_start is not None:
            logger.info(
                "Extracted recording start time from '%s': %s",
                label, recording_start.isoformat(),
            )

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        "Processing video '%s': %d frames @ %.1f fps", label, total_frames, fps
    )

    events_logged = 0
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            event = detector.process_frame(frame)
            if event is not None:
                if recording_start is not None:
                    elapsed_seconds = frame_index / fps
                    ts = recording_start + timedelta(seconds=elapsed_seconds)
                else:
                    ts = datetime.now(timezone.utc)

                event_logger.log_event(event, timestamp=ts)
                events_logged += 1
                logger.info(
                    "  [%s] Event %d: %s", label, events_logged, event
                )

            frame_index += 1
    finally:
        cap.release()

    logger.info(
        "Finished processing '%s': %d event(s) detected.", label, events_logged
    )
    return events_logged
