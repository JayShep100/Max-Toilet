"""
logger.py – Event logging for toilet pad detections.

Each :class:`DetectionEvent` is persisted to:
  * A CSV file (append-only, human-readable).
  * A JSON Lines file (one JSON object per line, machine-readable).

Both files are created automatically in the configured log directory.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .detector import DetectionEvent

logger = logging.getLogger(__name__)

# CSV column headers
_CSV_HEADERS = [
    "timestamp_utc",
    "event_type",
    "confidence",
    "motion_pixel_count",
    "wee_pixels",
    "poo_pixels",
]


class EventLogger:
    """
    Persists :class:`~detector.DetectionEvent` records with timestamps to CSV
    and JSON Lines files.

    Parameters
    ----------
    log_dir:
        Directory where log files are written.  Created if it does not exist.
    csv_filename:
        Name of the CSV log file (default: ``toilet_events.csv``).
    json_filename:
        Name of the JSON Lines log file (default: ``toilet_events.json``).
    """

    def __init__(
        self,
        log_dir: str = "logs",
        csv_filename: str = "toilet_events.csv",
        json_filename: str = "toilet_events.json",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.csv_path = self.log_dir / csv_filename
        self.json_path = self.log_dir / json_filename
        self._setup()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def log_event(
        self,
        event: DetectionEvent,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Write the event to both log files.

        Parameters
        ----------
        event:
            The detection event to record.
        timestamp:
            Explicit UTC timestamp.  Defaults to *now* (UTC) if not provided.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        record = self._build_record(event, timestamp)
        self._write_csv(record)
        self._write_json(record)
        logger.info(
            "[%s] Event logged: %s (confidence=%.2f)",
            record["timestamp_utc"],
            record["event_type"],
            event.confidence,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """Ensure log directory, CSV header, and JSON file exist."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_HEADERS)
                writer.writeheader()
            logger.debug("Created CSV log file: %s", self.csv_path)
        if not self.json_path.exists():
            self.json_path.touch()
            logger.debug("Created JSON log file: %s", self.json_path)

    @staticmethod
    def _build_record(event: DetectionEvent, timestamp: datetime) -> dict:
        """Convert a :class:`DetectionEvent` + timestamp into a flat dict."""
        return {
            "timestamp_utc": timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "event_type": event.event_type.value,
            "confidence": round(event.confidence, 4),
            "motion_pixel_count": event.motion_pixel_count,
            "wee_pixels": event.color_pixel_counts.get("wee_pixels", 0),
            "poo_pixels": event.color_pixel_counts.get("poo_pixels", 0),
        }

    def _write_csv(self, record: dict) -> None:
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_HEADERS)
            writer.writerow(record)

    def _write_json(self, record: dict) -> None:
        with open(self.json_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class FolderScanTxtLogger:
    """
    Accumulates toilet event records from a folder scan and writes a
    human-readable ``toilet_events_summary.txt`` to the scanned folder.

    Parameters
    ----------
    folder_path:
        The folder being scanned; the output file is written here.
    """

    _TXT_FILENAME = "toilet_events_summary.txt"
    _SEPARATOR = "-" * 40

    def __init__(self, folder_path: str) -> None:
        self.folder_path = Path(folder_path)
        self.txt_path = self.folder_path / self._TXT_FILENAME
        self.records: list[dict] = []

    def add_event(
        self,
        event: DetectionEvent,
        timestamp: datetime,
        source: str = "",
    ) -> None:
        """Accumulate a single event record."""
        self.records.append(
            {
                "timestamp": timestamp,
                "event_type": event.event_type.value,
                "source": source,
            }
        )

    def write_summary(
        self,
        videos_processed: int,
        scan_time: Optional[datetime] = None,
    ) -> Path:
        """
        Write the ``toilet_events_summary.txt`` file and return its path.

        Parameters
        ----------
        videos_processed:
            Total number of video files that were processed.
        scan_time:
            UTC datetime of the scan.  Defaults to *now* if not provided.
        """
        if scan_time is None:
            scan_time = datetime.now(timezone.utc)

        wee_count = sum(1 for r in self.records if r["event_type"] == "wee")
        poo_count = sum(1 for r in self.records if r["event_type"] == "poo")
        unknown_count = sum(1 for r in self.records if r["event_type"] == "unknown")
        total_events = len(self.records)

        lines = [
            "Max-Toilet Event Summary",
            f"Scanned: {scan_time.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            f"Folder:  {self.folder_path}",
            f"Videos processed: {videos_processed}",
            f"Total events: {total_events}",
            self._SEPARATOR,
        ]
        for r in self.records:
            ts_str = r["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
            etype = r["event_type"].upper().ljust(7)
            lines.append(f"{ts_str} | {etype} | source: {r['source']}")
        lines.append(self._SEPARATOR)
        lines.append(
            f"Summary: {wee_count:2d} wee event(s), {poo_count:2d} poo event(s),"
            f" {unknown_count:2d} unknown event(s)"
        )

        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        logger.info("Folder scan summary written to: %s", self.txt_path)
        return self.txt_path
