"""
test_folder_scanner.py – Tests for scan_folder() and FolderScanTxtLogger.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detector import DetectionEvent, DetectorConfig, EventType, PadDetector
from src.folder_scanner import _infer_recording_start, scan_folder
from src.logger import FolderScanTxtLogger


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_video_processor.py)
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


def _default_config() -> DetectorConfig:
    return DetectorConfig(presence_frames_required=5)


# ---------------------------------------------------------------------------
# FolderScanTxtLogger unit tests
# ---------------------------------------------------------------------------


class TestFolderScanTxtLogger:
    def test_write_summary_creates_file(self, tmp_path: Path) -> None:
        log = FolderScanTxtLogger(folder_path=str(tmp_path))
        txt_path = log.write_summary(videos_processed=0)
        assert txt_path.exists()

    def test_txt_written_inside_folder(self, tmp_path: Path) -> None:
        log = FolderScanTxtLogger(folder_path=str(tmp_path))
        txt_path = log.write_summary(videos_processed=0)
        assert txt_path.parent == tmp_path

    def test_txt_filename(self, tmp_path: Path) -> None:
        log = FolderScanTxtLogger(folder_path=str(tmp_path))
        txt_path = log.write_summary(videos_processed=0)
        assert txt_path.name == "toilet_events_summary.txt"

    def test_header_present(self, tmp_path: Path) -> None:
        log = FolderScanTxtLogger(folder_path=str(tmp_path))
        txt_path = log.write_summary(videos_processed=3)
        content = txt_path.read_text(encoding="utf-8")
        assert "Max-Toilet Event Summary" in content
        assert "Videos processed: 3" in content
        assert "Total events: 0" in content

    def test_summary_footer_present(self, tmp_path: Path) -> None:
        log = FolderScanTxtLogger(folder_path=str(tmp_path))
        txt_path = log.write_summary(videos_processed=0)
        content = txt_path.read_text(encoding="utf-8")
        assert "wee event(s)" in content
        assert "poo event(s)" in content
        assert "unknown event(s)" in content

    def test_event_line_format(self, tmp_path: Path) -> None:
        log = FolderScanTxtLogger(folder_path=str(tmp_path))
        ts = datetime(2026, 1, 15, 8, 12, 34, tzinfo=timezone.utc)
        event = DetectionEvent(
            event_type=EventType.WEE,
            confidence=0.9,
            color_pixel_counts={"wee_pixels": 500, "poo_pixels": 0},
        )
        log.add_event(event, ts, source="clip.mp4")
        txt_path = log.write_summary(videos_processed=1)
        content = txt_path.read_text(encoding="utf-8")
        assert "2026-01-15 08:12:34 UTC" in content
        assert "WEE" in content
        assert "source: clip.mp4" in content

    def test_counts_in_footer(self, tmp_path: Path) -> None:
        log = FolderScanTxtLogger(folder_path=str(tmp_path))
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        wee = DetectionEvent(
            event_type=EventType.WEE,
            confidence=0.9,
            color_pixel_counts={"wee_pixels": 500, "poo_pixels": 0},
        )
        poo = DetectionEvent(
            event_type=EventType.POO,
            confidence=0.8,
            color_pixel_counts={"wee_pixels": 0, "poo_pixels": 400},
        )
        log.add_event(wee, ts, source="a.mp4")
        log.add_event(wee, ts, source="a.mp4")
        log.add_event(poo, ts, source="b.mp4")
        txt_path = log.write_summary(videos_processed=2)
        content = txt_path.read_text(encoding="utf-8")
        assert " 2 wee event(s)" in content
        assert " 1 poo event(s)" in content
        assert " 0 unknown event(s)" in content


# ---------------------------------------------------------------------------
# Timestamp inference
# ---------------------------------------------------------------------------


class TestTimestampInference:
    def test_tapo_epoch_filename(self, tmp_path: Path) -> None:
        """1737000754-1737001354.mp4 → UTC datetime from epoch 1737000754."""
        p = tmp_path / "1737000754-1737001354.mp4"
        p.touch()
        result = _infer_recording_start(p)
        expected = datetime.fromtimestamp(1737000754, tz=timezone.utc)
        assert result == expected

    def test_datetime_embedded_filename(self, tmp_path: Path) -> None:
        """Filename containing YYYYMMDD_HHMMSS."""
        p = tmp_path / "cam_20260215_143022.mp4"
        p.touch()
        result = _infer_recording_start(p)
        expected = datetime(2026, 2, 15, 14, 30, 22, tzinfo=timezone.utc)
        assert result == expected

    def test_fallback_to_mtime(self, tmp_path: Path) -> None:
        """Unrecognised filename falls back to mtime."""
        import os
        import time

        p = tmp_path / "random_clip.mp4"
        p.touch()
        before = datetime.now(timezone.utc).replace(microsecond=0)
        time.sleep(0.05)
        os.utime(p, None)
        result = _infer_recording_start(p)
        after = datetime.now(timezone.utc)
        assert before <= result <= after


# ---------------------------------------------------------------------------
# scan_folder integration tests
# ---------------------------------------------------------------------------


class TestScanFolder:
    def test_empty_folder_produces_txt_with_zero_events(self, tmp_path: Path) -> None:
        result = scan_folder(str(tmp_path), _default_config())
        txt = Path(result["txt_path"])
        assert txt.exists()
        content = txt.read_text(encoding="utf-8")
        assert "Total events: 0" in content
        assert result["videos_processed"] == 0
        assert result["total_events"] == 0

    def test_txt_written_to_correct_location(self, tmp_path: Path) -> None:
        result = scan_folder(str(tmp_path), _default_config())
        txt = Path(result["txt_path"])
        assert txt.parent == tmp_path
        assert txt.name == "toilet_events_summary.txt"

    def test_static_video_produces_zero_events(self, tmp_path: Path) -> None:
        video = tmp_path / "clip.mp4"
        _write_minimal_video(video, frames=15)
        result = scan_folder(str(tmp_path), _default_config())
        assert result["videos_processed"] == 1
        assert result["total_events"] == 0
        txt = Path(result["txt_path"])
        content = txt.read_text(encoding="utf-8")
        assert "Total events: 0" in content

    def test_txt_contains_correct_header_and_footer(self, tmp_path: Path) -> None:
        result = scan_folder(str(tmp_path), _default_config())
        content = Path(result["txt_path"]).read_text(encoding="utf-8")
        assert "Max-Toilet Event Summary" in content
        assert "Scanned:" in content
        assert f"Folder:  {tmp_path}" in content
        assert "Videos processed:" in content
        assert "Total events:" in content
        assert "wee event(s)" in content
        assert "poo event(s)" in content
        assert "unknown event(s)" in content
        assert "-" * 40 in content

    def test_return_dict_keys(self, tmp_path: Path) -> None:
        result = scan_folder(str(tmp_path), _default_config())
        expected_keys = {
            "videos_processed",
            "total_events",
            "wee_count",
            "poo_count",
            "unknown_count",
            "txt_path",
        }
        assert expected_keys == set(result.keys())

    def test_non_video_files_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("hello")
        (tmp_path / "image.jpg").write_bytes(b"\xff\xd8")
        result = scan_folder(str(tmp_path), _default_config())
        assert result["videos_processed"] == 0

    def test_multiple_video_files_counted(self, tmp_path: Path) -> None:
        for name in ("clip_a.mp4", "clip_b.mp4", "clip_c.avi"):
            _write_minimal_video(tmp_path / name, frames=5)
        result = scan_folder(str(tmp_path), _default_config())
        assert result["videos_processed"] == 3

    def test_recursive_finds_videos_in_subfolders(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_minimal_video(sub / "deep.mp4", frames=5)
        # Non-recursive should not find it
        result_flat = scan_folder(str(tmp_path), _default_config(), recursive=False)
        assert result_flat["videos_processed"] == 0
        # Recursive should find it
        result_rec = scan_folder(str(tmp_path), _default_config(), recursive=True)
        assert result_rec["videos_processed"] == 1

    def test_raises_on_invalid_folder(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            scan_folder(str(tmp_path / "nonexistent"), _default_config())

    def test_txt_always_in_top_level_folder_when_recursive(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_minimal_video(sub / "deep.mp4", frames=5)
        result = scan_folder(str(tmp_path), _default_config(), recursive=True)
        txt = Path(result["txt_path"])
        assert txt.parent == tmp_path
