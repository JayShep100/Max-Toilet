"""
video_processor.py – Process a local video file through the detection pipeline.

When historical recordings are downloaded from the camera, this module feeds
every frame through :class:`~detector.PadDetector` and logs any detected
toilet events via :class:`~logger.EventLogger`.

The timestamp attached to each logged event is derived from the recording's
known start time plus the frame's position within the video, so the log
accurately reflects *when* the event happened rather than when it was processed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import cv2

from .detector import DetectorConfig, PadDetector
from .logger import EventLogger

logger = logging.getLogger(__name__)


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
