"""
test_video_processor.py – Tests for process_video_file().
"""

from __future__ import annotations

import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detector import DetectionEvent, DetectorConfig, EventType, PadDetector
from src.logger import EventLogger
from src.video_processor import process_video_file, extract_video_timestamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_minimal_video(path: Path, frames: int = 10, h: int = 64, w: int = 64) -> None:
    """Write a tiny solid-grey MP4 using OpenCV's VideoWriter."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, (w, h))
    frame = np.full((h, w, 3), 180, dtype=np.uint8)
    for _ in range(frames):
        out.write(frame)
    out.release()


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestProcessVideoFileErrors:
    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        detector = PadDetector()
        event_logger = MagicMock(spec=EventLogger)

        with pytest.raises(FileNotFoundError):
            process_video_file(
                str(tmp_path / "nonexistent.mp4"),
                detector,
                event_logger,
            )

    def test_raises_on_unreadable_file(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.mp4"
        bad_file.write_bytes(b"not a video")
        detector = PadDetector()
        event_logger = MagicMock(spec=EventLogger)

        with pytest.raises(OSError):
            process_video_file(str(bad_file), detector, event_logger)


# ---------------------------------------------------------------------------
# Normal processing
# ---------------------------------------------------------------------------


class TestProcessVideoFileNormal:
    def test_returns_zero_events_for_static_video(self, tmp_path: Path) -> None:
        """A static grey video should produce no detection events."""
        video_path = tmp_path / "static.mp4"
        _write_minimal_video(video_path, frames=15)

        detector = PadDetector(DetectorConfig(presence_frames_required=5))
        event_logger = MagicMock(spec=EventLogger)

        count = process_video_file(str(video_path), detector, event_logger)

        assert count == 0
        event_logger.log_event.assert_not_called()

    def test_returns_int(self, tmp_path: Path) -> None:
        video_path = tmp_path / "vid.mp4"
        _write_minimal_video(video_path, frames=5)
        count = process_video_file(
            str(video_path),
            PadDetector(),
            MagicMock(spec=EventLogger),
        )
        assert isinstance(count, int)

    def test_logs_event_with_correct_timestamp(self, tmp_path: Path) -> None:
        """
        When a DetectionEvent is produced, the timestamp passed to log_event
        should be recording_start + elapsed time.
        """
        video_path = tmp_path / "vid.mp4"
        # 10 frames at 10 fps → 1 second of footage
        _write_minimal_video(video_path, frames=10)

        recording_start = datetime(2025, 6, 1, 8, 0, 0, tzinfo=timezone.utc)

        event_logger = MagicMock(spec=EventLogger)
        detector = MagicMock(spec=PadDetector)

        fake_event = DetectionEvent(
            event_type=EventType.WEE,
            confidence=0.9,
            color_pixel_counts={"wee_pixels": 100, "poo_pixels": 0},
        )
        # Return an event on the 5th call (frame index 4), None otherwise
        detector.process_frame.side_effect = (
            [None] * 4 + [fake_event] + [None] * 5
        )

        count = process_video_file(
            str(video_path),
            detector,
            event_logger,
            recording_start=recording_start,
        )

        assert count == 1
        event_logger.log_event.assert_called_once()
        _, kwargs = event_logger.log_event.call_args
        ts = kwargs.get("timestamp") or event_logger.log_event.call_args[0][1]
        # Frame 4 at 10 fps → 0.4 s after start
        expected_ts = recording_start + timedelta(seconds=4 / 10.0)
        assert abs((ts - expected_ts).total_seconds()) < 0.1

    def test_uses_current_time_when_no_recording_start(self, tmp_path: Path) -> None:
        """When recording_start is None, timestamp should be derived from the
        video file (e.g. file mtime) and be close to now."""
        video_path = tmp_path / "vid.mp4"
        _write_minimal_video(video_path, frames=5)

        event_logger = MagicMock(spec=EventLogger)
        detector = MagicMock(spec=PadDetector)
        fake_event = DetectionEvent(
            event_type=EventType.POO, confidence=0.8,
            color_pixel_counts={"wee_pixels": 0, "poo_pixels": 200},
        )
        detector.process_frame.side_effect = [fake_event] + [None] * 4

        now = datetime.now(timezone.utc)
        process_video_file(str(video_path), detector, event_logger)

        event_logger.log_event.assert_called_once()
        _, kwargs = event_logger.log_event.call_args
        ts = kwargs.get("timestamp") or event_logger.log_event.call_args[0][1]
        # Timestamp should be within a few seconds of now (from mtime or clock)
        assert abs((ts - now).total_seconds()) < 5

    def test_source_label_does_not_affect_result(self, tmp_path: Path) -> None:
        video_path = tmp_path / "labelled.mp4"
        _write_minimal_video(video_path, frames=5)
        count = process_video_file(
            str(video_path),
            PadDetector(),
            MagicMock(spec=EventLogger),
            source_label="my_custom_label",
        )
        assert isinstance(count, int)


# ---------------------------------------------------------------------------
# extract_video_timestamp
# ---------------------------------------------------------------------------


class TestExtractVideoTimestamp:
    def test_epoch_dash_epoch_filename(self, tmp_path: Path) -> None:
        """Cloud downloader naming: 1748736000-1748739600.mp4"""
        p = tmp_path / "1748736000-1748739600.mp4"
        p.write_bytes(b"fake")
        ts = extract_video_timestamp(str(p))
        assert ts == datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_yyyymmdd_hhmmss_filename(self, tmp_path: Path) -> None:
        p = tmp_path / "20250601_143022.mp4"
        p.write_bytes(b"fake")
        ts = extract_video_timestamp(str(p))
        assert ts == datetime(2025, 6, 1, 14, 30, 22, tzinfo=timezone.utc)

    def test_iso_ish_filename(self, tmp_path: Path) -> None:
        p = tmp_path / "2025-06-01T14-30-22.mp4"
        p.write_bytes(b"fake")
        ts = extract_video_timestamp(str(p))
        assert ts == datetime(2025, 6, 1, 14, 30, 22, tzinfo=timezone.utc)

    def test_date_only_filename(self, tmp_path: Path) -> None:
        p = tmp_path / "20250601.mp4"
        p.write_bytes(b"fake")
        ts = extract_video_timestamp(str(p))
        assert ts == datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_falls_back_to_mtime(self, tmp_path: Path) -> None:
        """File with no timestamp pattern in name should use mtime."""
        p = tmp_path / "random_clip.mp4"
        p.write_bytes(b"fake")
        ts = extract_video_timestamp(str(p))
        assert ts is not None
        now = datetime.now(timezone.utc)
        assert abs((ts - now).total_seconds()) < 5

    def test_nonexistent_file_returns_none(self) -> None:
        ts = extract_video_timestamp("/nonexistent/path/video.mp4")
        assert ts is None
