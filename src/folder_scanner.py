"""
folder_scanner.py – Scan a local folder of video files and write a summary report.

For each video file found in the target directory (non-recursive), a fresh
:class:`~detector.PadDetector` is created and the file is fed through the
:func:`~video_processor.process_video_file` pipeline.  After all files have
been processed a plain-text ``.txt`` summary is written to disk.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .detector import DetectorConfig, PadDetector
from .logger import EventLogger
from .video_processor import process_video_file

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


class FolderScanner:
    """
    Scan a local directory for video files and process each through the
    detection pipeline.

    Parameters
    ----------
    folder_path:
        Path to the directory containing video files to scan.
    detector_config:
        :class:`~detector.DetectorConfig` used to construct a fresh
        :class:`~detector.PadDetector` for each video file.
    event_logger:
        :class:`~logger.EventLogger` instance used to persist detected events.
    summary_output_path:
        Path for the ``.txt`` summary report.  Defaults to
        ``<folder_path>/scan_summary.txt``.
    """

    def __init__(
        self,
        folder_path: str,
        detector_config: DetectorConfig,
        event_logger: EventLogger,
        summary_output_path: Optional[str] = None,
    ) -> None:
        self.folder_path = folder_path
        self.detector_config = detector_config
        self.event_logger = event_logger
        self.summary_output_path = summary_output_path or str(
            Path(folder_path) / "scan_summary.txt"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scan(self) -> int:
        """
        Scan the folder, process each video file, and write the summary report.

        Returns
        -------
        int
            Total number of events detected across all video files.

        Raises
        ------
        FileNotFoundError
            If :attr:`folder_path` does not exist.
        """
        folder = Path(self.folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")

        video_files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS
        )

        logger.info(
            "FolderScanner: found %d video file(s) in '%s'",
            len(video_files),
            self.folder_path,
        )

        per_file_counts: dict[str, int] = {}
        total_events = 0

        for video_path in video_files:
            filename = video_path.name
            detector = PadDetector(config=self.detector_config)
            count = process_video_file(
                str(video_path),
                detector,
                self.event_logger,
                recording_start=None,
                source_label=filename,
            )
            per_file_counts[filename] = count
            total_events += count
            logger.info("  %s: %d event(s)", filename, count)

        self._write_summary(per_file_counts, total_events)
        return total_events

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_summary(self, per_file_counts: dict[str, int], total_events: int) -> None:
        """Write the plain-text summary report to :attr:`summary_output_path`."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        lines = [
            f"Scan folder: {self.folder_path}",
            f"Scan timestamp (UTC): {timestamp}",
            "",
        ]
        for filename, count in per_file_counts.items():
            lines.append(f"{filename}: {count} event(s)")
        lines.append("")
        lines.append(f"Total events: {total_events}")

        output_path = Path(self.summary_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        logger.info("Summary written to '%s'", self.summary_output_path)
