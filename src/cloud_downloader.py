"""
cloud_downloader.py – Download previously saved recordings from a Tapo camera.

Tapo cameras store footage on their local SD card and, when cloud-enabled,
sync that footage to Tapo Cloud.  This module uses the pytapo library to:

  1. List all dates that have recordings within the last N days (up to 30).
  2. Retrieve the individual recording segments (start/end UTC timestamps)
     for each date.
  3. Download each segment as an MP4 file using pytapo's async Downloader.

Downloaded files are placed in a configurable output directory and named
``<start_utc>-<end_utc>.mp4`` (pytapo default naming).

Usage example
-------------
    downloader = TapoCloudDownloader(
        host="192.168.1.100",
        username="admin",
        password="localpass",
        cloud_password="cloudpass",
        output_dir="/tmp/tapo_recordings",
        days_back=30,
    )
    for path, info in downloader.download_recordings():
        process_video_file(path, detector, event_logger)
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Maximum days of backfill that will ever be requested
MAX_DAYS_BACK = 30


@dataclass
class RecordingSegment:
    """One continuous recording segment on the camera."""

    start_time: int  # UTC epoch seconds
    end_time: int    # UTC epoch seconds

    @property
    def start_dt(self) -> datetime:
        return datetime.fromtimestamp(self.start_time, tz=timezone.utc)

    @property
    def end_dt(self) -> datetime:
        return datetime.fromtimestamp(self.end_time, tz=timezone.utc)

    def __str__(self) -> str:
        return (
            f"RecordingSegment({self.start_dt.isoformat()} – {self.end_dt.isoformat()})"
        )


class TapoCloudDownloader:
    """
    Lists and downloads camera recordings for the last *days_back* days.

    Parameters
    ----------
    host:
        IP address or hostname of the Tapo camera on the local network.
    username:
        Camera admin username (default ``admin``).
    password:
        Local camera password.
    cloud_password:
        Tapo Cloud / app account password (used by pytapo for authentication).
    output_dir:
        Directory where downloaded MP4 files are saved.  Created if absent.
    days_back:
        How many days of history to fetch (capped at :data:`MAX_DAYS_BACK`).
    """

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        cloud_password: str,
        output_dir: str,
        days_back: int = 30,
    ) -> None:
        self.host = host
        self.username = username
        self.password = password
        self.cloud_password = cloud_password
        self.output_dir = Path(output_dir)
        self.days_back = min(int(days_back), MAX_DAYS_BACK)
        self._tapo = None  # lazily initialised

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def list_recording_dates(self) -> List[str]:
        """
        Return a list of ``YYYYMMDD`` strings that have recordings within the
        configured backfill window.
        """
        tapo = self._get_tapo()
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=self.days_back)

        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        logger.info(
            "Querying recording dates from %s to %s on %s",
            start_str,
            end_str,
            self.host,
        )

        raw = tapo.getRecordingsList(start_date=start_str, end_date=end_str)
        # raw is a list of date strings like ["20240601", "20240602", ...]
        dates = [entry for entry in raw if isinstance(entry, str)]
        logger.info("Found %d date(s) with recordings.", len(dates))
        return dates

    def list_segments_for_date(self, date: str) -> List[RecordingSegment]:
        """
        Return :class:`RecordingSegment` objects for all recordings on *date*
        (``YYYYMMDD`` format).
        """
        tapo = self._get_tapo()
        logger.debug("Listing recording segments for date %s", date)
        raw = tapo.getRecordings(date)
        segments: List[RecordingSegment] = []
        for entry in raw:
            # Each entry has keys "startTime" / "endTime" (UTC epoch seconds)
            start = entry.get("startTime") or entry.get("start_time")
            end = entry.get("endTime") or entry.get("end_time")
            if start is not None and end is not None:
                segments.append(RecordingSegment(start_time=int(start), end_time=int(end)))
        logger.debug("  → %d segment(s) on %s", len(segments), date)
        return segments

    def download_segment(self, segment: RecordingSegment) -> Optional[str]:
        """
        Download one recording segment and return the path to the saved MP4,
        or *None* if the download failed.
        """
        from pytapo.media_stream.downloader import Downloader

        self.output_dir.mkdir(parents=True, exist_ok=True)
        tapo = self._get_tapo()
        time_correction = tapo.getTimeCorrection()

        logger.info("Downloading %s …", segment)
        try:
            downloader = Downloader(
                tapo=tapo,
                startTime=segment.start_time,
                endTime=segment.end_time,
                timeCorrection=time_correction,
                outputDirectory=str(self.output_dir) + os.sep,
                overwriteFiles=False,
            )
            result = asyncio.run(downloader.downloadFile())
            if result and isinstance(result, dict):
                file_path = result.get("fileName")
            else:
                # pytapo Downloader may yield status dicts; locate the file ourselves
                start_str = str(segment.start_time)
                end_str = str(segment.end_time)
                file_path = str(self.output_dir / f"{start_str}-{end_str}.mp4")

            if file_path and Path(file_path).exists():
                logger.info("Download complete: %s", file_path)
                return file_path
            logger.warning("Download finished but output file not found for %s", segment)
            return None
        except Exception as exc:
            logger.error("Failed to download %s: %s", segment, exc)
            return None

    def download_recordings(self) -> Iterator[Tuple[str, RecordingSegment]]:
        """
        Yield ``(file_path, segment)`` pairs for every successfully downloaded
        recording within the backfill window, in chronological order.
        """
        dates = self.list_recording_dates()
        for date in sorted(dates):
            try:
                segments = self.list_segments_for_date(date)
            except Exception as exc:
                logger.error("Could not list segments for %s: %s", date, exc)
                continue
            for segment in segments:
                path = self.download_segment(segment)
                if path is not None:
                    yield path, segment

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_tapo(self):
        """Return a cached :class:`pytapo.Tapo` instance."""
        if self._tapo is None:
            from pytapo import Tapo

            logger.info("Connecting to Tapo camera API at %s", self.host)
            self._tapo = Tapo(
                self.host,
                self.username,
                self.password,
                self.cloud_password,
            )
        return self._tapo
