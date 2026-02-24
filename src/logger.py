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
