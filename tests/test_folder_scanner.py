"""
test_folder_scanner.py – Tests for FolderScanner.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from src.detector import DetectorConfig, PadDetector
from src.folder_scanner import FolderScanner
from src.logger import EventLogger


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


class TestFolderScannerErrors:
    def test_raises_when_folder_missing(self, tmp_path: Path) -> None:
        config = DetectorConfig()
        event_logger = MagicMock(spec=EventLogger)
        scanner = FolderScanner(
            str(tmp_path / "nonexistent"),
            config,
            event_logger,
        )
        with pytest.raises(FileNotFoundError):
            scanner.scan()


# ---------------------------------------------------------------------------
# Default summary output path
# ---------------------------------------------------------------------------


class TestFolderScannerDefaults:
    def test_default_summary_path(self, tmp_path: Path) -> None:
        config = DetectorConfig()
        event_logger = MagicMock(spec=EventLogger)
        scanner = FolderScanner(str(tmp_path), config, event_logger)
        assert scanner.summary_output_path == str(tmp_path / "scan_summary.txt")

    def test_custom_summary_path(self, tmp_path: Path) -> None:
        config = DetectorConfig()
        event_logger = MagicMock(spec=EventLogger)
        custom = str(tmp_path / "out" / "report.txt")
        scanner = FolderScanner(str(tmp_path), config, event_logger, summary_output_path=custom)
        assert scanner.summary_output_path == custom


# ---------------------------------------------------------------------------
# Empty folder
# ---------------------------------------------------------------------------


class TestFolderScannerEmptyFolder:
    def test_empty_folder_returns_zero(self, tmp_path: Path) -> None:
        config = DetectorConfig()
        event_logger = MagicMock(spec=EventLogger)
        scanner = FolderScanner(str(tmp_path), config, event_logger)
        total = scanner.scan()
        assert total == 0

    def test_empty_folder_writes_summary(self, tmp_path: Path) -> None:
        config = DetectorConfig()
        event_logger = MagicMock(spec=EventLogger)
        scanner = FolderScanner(str(tmp_path), config, event_logger)
        scanner.scan()
        summary = Path(scanner.summary_output_path)
        assert summary.exists()
        content = summary.read_text(encoding="utf-8")
        assert "Total events: 0" in content

    def test_summary_contains_folder_path(self, tmp_path: Path) -> None:
        config = DetectorConfig()
        event_logger = MagicMock(spec=EventLogger)
        scanner = FolderScanner(str(tmp_path), config, event_logger)
        scanner.scan()
        content = Path(scanner.summary_output_path).read_text(encoding="utf-8")
        assert str(tmp_path) in content

    def test_summary_contains_timestamp(self, tmp_path: Path) -> None:
        config = DetectorConfig()
        event_logger = MagicMock(spec=EventLogger)
        scanner = FolderScanner(str(tmp_path), config, event_logger)
        scanner.scan()
        content = Path(scanner.summary_output_path).read_text(encoding="utf-8")
        assert "Scan timestamp (UTC):" in content


# ---------------------------------------------------------------------------
# Non-video files are ignored
# ---------------------------------------------------------------------------


class TestFolderScannerFiltering:
    def test_ignores_non_video_files(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "data.csv").write_text("a,b")
        config = DetectorConfig()
        event_logger = MagicMock(spec=EventLogger)
        scanner = FolderScanner(str(tmp_path), config, event_logger)
        with patch("src.folder_scanner.process_video_file") as mock_pvf:
            scanner.scan()
            mock_pvf.assert_not_called()

    def test_recognises_all_supported_extensions(self, tmp_path: Path) -> None:
        """process_video_file is called for each recognised video extension."""
        for ext in (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"):
            (tmp_path / f"clip{ext}").write_bytes(b"fake")
        config = DetectorConfig()
        event_logger = MagicMock(spec=EventLogger)
        scanner = FolderScanner(str(tmp_path), config, event_logger)
        with patch("src.folder_scanner.process_video_file", return_value=0) as mock_pvf:
            scanner.scan()
            assert mock_pvf.call_count == 7

    def test_extension_matching_is_case_insensitive(self, tmp_path: Path) -> None:
        (tmp_path / "CLIP.MP4").write_bytes(b"fake")
        (tmp_path / "clip.Avi").write_bytes(b"fake")
        config = DetectorConfig()
        event_logger = MagicMock(spec=EventLogger)
        scanner = FolderScanner(str(tmp_path), config, event_logger)
        with patch("src.folder_scanner.process_video_file", return_value=0) as mock_pvf:
            scanner.scan()
            assert mock_pvf.call_count == 2


# ---------------------------------------------------------------------------
# Processing and event counts
# ---------------------------------------------------------------------------


class TestFolderScannerProcessing:
    def test_scan_returns_total_event_count(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"clip{i}.mp4").write_bytes(b"fake")
        config = DetectorConfig()
        event_logger = MagicMock(spec=EventLogger)
        scanner = FolderScanner(str(tmp_path), config, event_logger)
        with patch("src.folder_scanner.process_video_file", return_value=2):
            total = scanner.scan()
        assert total == 6  # 3 files × 2 events each

    def test_fresh_detector_created_per_file(self, tmp_path: Path) -> None:
        """A new PadDetector must be instantiated for each video file."""
        for i in range(2):
            (tmp_path / f"clip{i}.mp4").write_bytes(b"fake")
        config = DetectorConfig()
        event_logger = MagicMock(spec=EventLogger)
        scanner = FolderScanner(str(tmp_path), config, event_logger)
        with patch("src.folder_scanner.PadDetector") as MockDetector, \
             patch("src.folder_scanner.process_video_file", return_value=0):
            scanner.scan()
            assert MockDetector.call_count == 2

    def test_process_video_file_called_with_correct_args(self, tmp_path: Path) -> None:
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")
        config = DetectorConfig()
        event_logger = MagicMock(spec=EventLogger)
        scanner = FolderScanner(str(tmp_path), config, event_logger)
        with patch("src.folder_scanner.PadDetector") as MockDetector, \
             patch("src.folder_scanner.process_video_file", return_value=1) as mock_pvf:
            scanner.scan()
            mock_pvf.assert_called_once_with(
                str(video_file),
                MockDetector.return_value,
                event_logger,
                recording_start=None,
                source_label="test.mp4",
            )

    def test_files_are_processed_in_sorted_order(self, tmp_path: Path) -> None:
        for name in ("c.mp4", "a.mp4", "b.mp4"):
            (tmp_path / name).write_bytes(b"fake")
        config = DetectorConfig()
        event_logger = MagicMock(spec=EventLogger)
        scanner = FolderScanner(str(tmp_path), config, event_logger)
        processed = []
        with patch("src.folder_scanner.process_video_file", side_effect=lambda p, *a, **kw: processed.append(Path(p).name) or 0):
            scanner.scan()
        assert processed == ["a.mp4", "b.mp4", "c.mp4"]

    def test_non_recursive_scan(self, tmp_path: Path) -> None:
        """Video files in subdirectories should NOT be processed."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.mp4").write_bytes(b"fake")
        config = DetectorConfig()
        event_logger = MagicMock(spec=EventLogger)
        scanner = FolderScanner(str(tmp_path), config, event_logger)
        with patch("src.folder_scanner.process_video_file", return_value=0) as mock_pvf:
            scanner.scan()
            mock_pvf.assert_not_called()


# ---------------------------------------------------------------------------
# Summary report content
# ---------------------------------------------------------------------------


class TestFolderScannerSummaryContent:
    def test_summary_lists_per_file_counts(self, tmp_path: Path) -> None:
        for name in ("alpha.mp4", "beta.mp4"):
            (tmp_path / name).write_bytes(b"fake")
        config = DetectorConfig()
        event_logger = MagicMock(spec=EventLogger)
        scanner = FolderScanner(str(tmp_path), config, event_logger)
        with patch("src.folder_scanner.process_video_file", return_value=3):
            scanner.scan()
        content = Path(scanner.summary_output_path).read_text(encoding="utf-8")
        assert "alpha.mp4: 3 event(s)" in content
        assert "beta.mp4: 3 event(s)" in content

    def test_summary_total_events_line(self, tmp_path: Path) -> None:
        (tmp_path / "vid.mp4").write_bytes(b"fake")
        config = DetectorConfig()
        event_logger = MagicMock(spec=EventLogger)
        scanner = FolderScanner(str(tmp_path), config, event_logger)
        with patch("src.folder_scanner.process_video_file", return_value=5):
            scanner.scan()
        content = Path(scanner.summary_output_path).read_text(encoding="utf-8")
        assert "Total events: 5" in content

    def test_custom_summary_path_is_used(self, tmp_path: Path) -> None:
        custom_path = tmp_path / "reports" / "my_summary.txt"
        config = DetectorConfig()
        event_logger = MagicMock(spec=EventLogger)
        scanner = FolderScanner(
            str(tmp_path), config, event_logger,
            summary_output_path=str(custom_path),
        )
        scanner.scan()
        assert custom_path.exists()

    def test_scan_with_real_video_file(self, tmp_path: Path) -> None:
        """End-to-end: a real static video produces 0 events and a valid summary."""
        video_path = tmp_path / "static.mp4"
        _write_minimal_video(video_path, frames=15)
        config = DetectorConfig(presence_frames_required=5)
        event_logger = MagicMock(spec=EventLogger)
        scanner = FolderScanner(str(tmp_path), config, event_logger)
        total = scanner.scan()
        assert total == 0
        content = Path(scanner.summary_output_path).read_text(encoding="utf-8")
        assert "static.mp4: 0 event(s)" in content
        assert "Total events: 0" in content
