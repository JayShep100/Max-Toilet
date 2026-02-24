"""
test_logger.py – Tests for EventLogger (CSV and JSON Lines output).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.detector import DetectionEvent, EventType
from src.logger import EventLogger, _CSV_HEADERS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_logger(tmp_path: Path) -> EventLogger:
    """Return an EventLogger writing to a temporary directory."""
    return EventLogger(
        log_dir=str(tmp_path),
        csv_filename="test_events.csv",
        json_filename="test_events.json",
    )


def _wee_event() -> DetectionEvent:
    return DetectionEvent(
        event_type=EventType.WEE,
        confidence=0.85,
        motion_pixel_count=120,
        color_pixel_counts={"wee_pixels": 500, "poo_pixels": 20},
    )


def _poo_event() -> DetectionEvent:
    return DetectionEvent(
        event_type=EventType.POO,
        confidence=0.72,
        motion_pixel_count=200,
        color_pixel_counts={"wee_pixels": 10, "poo_pixels": 800},
    )


# ---------------------------------------------------------------------------
# Directory / file creation
# ---------------------------------------------------------------------------


class TestSetup:
    def test_log_dir_created(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        EventLogger(log_dir=str(nested))
        assert nested.exists()

    def test_csv_header_written(self, tmp_logger: EventLogger) -> None:
        with open(tmp_logger.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == _CSV_HEADERS

    def test_json_file_created(self, tmp_logger: EventLogger) -> None:
        assert tmp_logger.json_path.exists()


# ---------------------------------------------------------------------------
# log_event – CSV output
# ---------------------------------------------------------------------------


class TestCSVOutput:
    def test_wee_event_csv_row(self, tmp_logger: EventLogger, tmp_path: Path) -> None:
        ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        tmp_logger.log_event(_wee_event(), timestamp=ts)

        with open(tmp_logger.csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        assert rows[0]["event_type"] == "wee"
        assert rows[0]["timestamp_utc"] == "2025-06-01T12:00:00.000Z"
        assert float(rows[0]["confidence"]) == pytest.approx(0.85)
        assert int(rows[0]["wee_pixels"]) == 500
        assert int(rows[0]["poo_pixels"]) == 20

    def test_poo_event_csv_row(self, tmp_logger: EventLogger) -> None:
        ts = datetime(2025, 6, 2, 8, 30, 0, tzinfo=timezone.utc)
        tmp_logger.log_event(_poo_event(), timestamp=ts)

        with open(tmp_logger.csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["event_type"] == "poo"
        assert int(rows[0]["poo_pixels"]) == 800

    def test_multiple_events_appended(self, tmp_logger: EventLogger) -> None:
        for _ in range(5):
            tmp_logger.log_event(_wee_event())

        with open(tmp_logger.csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 5

    def test_csv_not_duplicated_on_second_init(self, tmp_path: Path) -> None:
        """Creating a second EventLogger for the same path must not re-write the header."""
        el1 = EventLogger(log_dir=str(tmp_path), csv_filename="dup.csv")
        el1.log_event(_wee_event())
        el2 = EventLogger(log_dir=str(tmp_path), csv_filename="dup.csv")
        el2.log_event(_poo_event())

        with open(el2.csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2


# ---------------------------------------------------------------------------
# log_event – JSON output
# ---------------------------------------------------------------------------


class TestJSONOutput:
    def test_json_line_structure(self, tmp_logger: EventLogger) -> None:
        ts = datetime(2025, 7, 4, 10, 0, 0, tzinfo=timezone.utc)
        tmp_logger.log_event(_wee_event(), timestamp=ts)

        with open(tmp_logger.json_path, encoding="utf-8") as f:
            record = json.loads(f.readline())

        assert record["event_type"] == "wee"
        assert record["timestamp_utc"] == "2025-07-04T10:00:00.000Z"
        assert record["confidence"] == pytest.approx(0.85)
        assert record["wee_pixels"] == 500

    def test_multiple_json_lines(self, tmp_logger: EventLogger) -> None:
        tmp_logger.log_event(_wee_event())
        tmp_logger.log_event(_poo_event())

        with open(tmp_logger.json_path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2
        assert lines[0]["event_type"] == "wee"
        assert lines[1]["event_type"] == "poo"

    def test_json_default_timestamp_is_utc(self, tmp_logger: EventLogger) -> None:
        """When no timestamp is provided, the logged time should end in 'Z'."""
        tmp_logger.log_event(_wee_event())

        with open(tmp_logger.json_path, encoding="utf-8") as f:
            record = json.loads(f.readline())

        assert record["timestamp_utc"].endswith("Z")


# ---------------------------------------------------------------------------
# _build_record (unit)
# ---------------------------------------------------------------------------


class TestBuildRecord:
    def test_record_keys(self) -> None:
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        record = EventLogger._build_record(_wee_event(), ts)
        for key in _CSV_HEADERS:
            assert key in record

    def test_unknown_event_type(self) -> None:
        event = DetectionEvent(
            event_type=EventType.UNKNOWN,
            confidence=0.5,
            color_pixel_counts={},
        )
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        record = EventLogger._build_record(event, ts)
        assert record["event_type"] == "unknown"
        assert record["wee_pixels"] == 0
        assert record["poo_pixels"] == 0
