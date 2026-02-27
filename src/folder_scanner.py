"""
folder_scanner.py – Scan a local folder of video files through the detection pipeline.

Given a directory of video clips (e.g. MP4s saved from a Tapo camera), this
module processes every supported video file through :class:`~detector.PadDetector`
and writes a human-readable ``toilet_events_summary.txt`` to the same folder.

Supported extensions: ``.mp4``, ``.avi``, ``.mov``, ``.mkv``, ``.ts``
(case-insensitive), discovered in alphabetical order.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .detector import DetectorConfig, PadDetector
from .logger import FolderScanTxtLogger
from .video_processor import process_video_file

logger = logging.getLogger(__name__)

# Video file extensions to discover (lower-case)
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".ts"}

# Regex patterns for timestamp inference from filenames
_EPOCH_RE = re.compile(r"^(\d{10})")  # Unix epoch at start, e.g. 1737000754-...
_DATETIME_RE = re.compile(
    r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})[_T](\d{2})[-_]?(\d{2})[-_]?(\d{2})"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _EventCollector:
    """
    Lightweight :class:`~logger.EventLogger`-compatible sink that accumulates
    events in memory without writing any files.
    """

    def __init__(self) -> None:
        self.events: list[tuple] = []  # list of (DetectionEvent, datetime)

    def log_event(self, event, timestamp: Optional[datetime] = None) -> None:  # type: ignore[override]
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        self.events.append((event, timestamp))


def _infer_recording_start(video_path: Path) -> datetime:
    """
    Try to derive the recording start time from *video_path*.

    Priority
    --------
    1. Filename starts with a 10-digit Unix epoch (e.g. ``1737000754-…``).
    2. Filename contains ``YYYYMMDD_HHMMSS`` or ``YYYY-MM-DD_HH-MM-SS``.
    3. Fall back to ``os.path.getmtime()``.

    Returns
    -------
    datetime
        A timezone-aware UTC :class:`datetime`.
    """
    stem = video_path.stem

    # 1. Unix epoch prefix (Tapo-style filenames)
    m = _EPOCH_RE.match(stem)
    if m:
        try:
            return datetime.fromtimestamp(int(m.group(1)), tz=timezone.utc)
        except (ValueError, OSError):
            pass

    # 2. YYYYMMDD_HHMMSS or YYYY-MM-DD_HH-MM-SS embedded in filename
    m2 = _DATETIME_RE.search(stem)
    if m2:
        try:
            year, month, day, hour, minute, second = (int(v) for v in m2.groups())
            return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
        except ValueError:
            pass

    # 3. File modification time
    mtime = os.path.getmtime(video_path)
    return datetime.fromtimestamp(mtime, tz=timezone.utc)


def _discover_videos(folder_path: Path, recursive: bool) -> list[Path]:
    """
    Return a sorted list of video files inside *folder_path*.

    Parameters
    ----------
    folder_path:
        Directory to search.
    recursive:
        When *True*, subdirectories are also searched.
    """
    if recursive:
        candidates = sorted(folder_path.rglob("*"))
    else:
        candidates = sorted(folder_path.iterdir())

    return [
        p for p in candidates if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_folder(
    folder_path: str,
    detector_config: DetectorConfig,
    recursive: bool = False,
) -> dict:
    """
    Process every video file in *folder_path* through the detection pipeline.

    Parameters
    ----------
    folder_path:
        Path to the local folder containing video files.
    detector_config:
        Detector configuration shared across all clips.
    recursive:
        When *True*, subdirectories are also scanned.  The output
        ``toilet_events_summary.txt`` is always written to the top-level
        *folder_path*.

    Returns
    -------
    dict
        A dictionary with keys:
        ``videos_processed``, ``total_events``, ``wee_count``,
        ``poo_count``, ``unknown_count``, ``txt_path``.
    """
    root = Path(folder_path)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder_path}")

    videos = _discover_videos(root, recursive)
    total_videos = len(videos)
    txt_logger = FolderScanTxtLogger(folder_path=str(root))
    scan_time = datetime.now(timezone.utc)

    logger.info(
        "Folder scan starting: %d video(s) found in '%s'", total_videos, root
    )

    videos_processed = 0
    for idx, video_path in enumerate(videos, start=1):
        logger.info("[%d/%d] Processing %s …", idx, total_videos, video_path.name)

        recording_start = _infer_recording_start(video_path)
        detector = PadDetector(config=detector_config)
        collector = _EventCollector()

        try:
            process_video_file(
                video_path=str(video_path),
                detector=detector,
                event_logger=collector,  # type: ignore[arg-type]
                recording_start=recording_start,
                source_label=video_path.name,
            )
        except (FileNotFoundError, OSError) as exc:
            logger.warning("Skipping '%s': %s", video_path.name, exc)
            continue

        for event, ts in collector.events:
            txt_logger.add_event(event, ts, source=video_path.name)

        videos_processed += 1

    txt_path = txt_logger.write_summary(
        videos_processed=videos_processed,
        scan_time=scan_time,
    )

    total_events = len(txt_logger.records)
    wee_count = sum(1 for r in txt_logger.records if r["event_type"] == "wee")
    poo_count = sum(1 for r in txt_logger.records if r["event_type"] == "poo")
    unknown_count = sum(
        1 for r in txt_logger.records if r["event_type"] == "unknown"
    )

    logger.info(
        "Folder scan complete: %d video(s), %d event(s) (wee=%d, poo=%d, unknown=%d).",
        videos_processed,
        total_events,
        wee_count,
        poo_count,
        unknown_count,
    )

    return {
        "videos_processed": videos_processed,
        "total_events": total_events,
        "wee_count": wee_count,
        "poo_count": poo_count,
        "unknown_count": unknown_count,
        "txt_path": str(txt_path),
    }
