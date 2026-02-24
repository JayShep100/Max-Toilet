"""
cloud_downloader.py – Download previously saved recordings from Tapo Cloud.

Tapo cameras can store footage to Tapo Cloud (TP-Link's cloud servers).
This module authenticates against Tapo Cloud and:

  1. Lists all dates that have recordings within the last N days (up to 30).
  2. Retrieves individual recording segments (start/end UTC timestamps).
  3. Downloads each segment as an MP4 file via HTTP streaming.

Downloaded files are placed in a configurable output directory and named
``<start_utc>-<end_utc>.mp4``.

Usage example
-------------
    downloader = TapoCloudDownloader(
        host="192.168.1.100",
        username="user@example.com",   # Tapo account email
        password="localpass",          # local camera admin password (RTSP only)
        cloud_password="cloudpass",    # Tapo app / TP-Link account password
        output_dir="/tmp/tapo_recordings",
        days_back=30,
    )
    for path, info in downloader.download_recordings():
        process_video_file(path, detector, event_logger)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Maximum days of backfill that will ever be requested
MAX_DAYS_BACK = 30

_CLOUD_URL = "https://n-eap-gl.tplinkcloud.com"


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


class TapoCloud:
    """Minimal TP-Link cloud API client for listing and downloading recordings."""

    def __init__(self, username: str, cloud_password: str) -> None:
        self.username = username
        self.cloud_password = cloud_password
        self._session = requests.Session()
        self._token: Optional[str] = None

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self) -> None:
        """Authenticate with Tapo Cloud and store the session token."""
        resp = self._session.post(
            _CLOUD_URL,
            json={
                "method": "login",
                "params": {
                    "appType": "Tapo_Ios",
                    "cloudUserName": self.username,
                    "cloudPassword": self.cloud_password,
                    "terminalUUID": str(uuid.uuid4()),
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error_code", -1) != 0:
            raise RuntimeError(
                f"Cloud authentication failed (error {data.get('error_code')}): "
                f"{data.get('msg', data)}"
            )
        self._token = data["result"]["token"]
        logger.debug("Authenticated with Tapo Cloud")

    def _ensure_auth(self) -> None:
        if self._token is None:
            self.authenticate()

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def get_devices(self) -> List[dict]:
        """Return the list of devices registered on this cloud account."""
        self._ensure_auth()
        resp = self._session.post(
            _CLOUD_URL,
            json={
                "method": "getDeviceListByPage",
                "params": {
                    "token": self._token,
                    "pageSize": 100,
                    "pageIndex": 1,
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error_code", -1) != 0:
            raise RuntimeError(
                f"Failed to list devices (error {data.get('error_code')}): "
                f"{data.get('msg', data)}"
            )
        return data.get("result", {}).get("deviceList", [])

    # ------------------------------------------------------------------
    # Recording management
    # ------------------------------------------------------------------

    def get_recordings_list(
        self, device_id: str, start_date: str, end_date: str
    ) -> List[str]:
        """Return a list of YYYYMMDD strings with recordings for *device_id*."""
        self._ensure_auth()
        resp = self._session.post(
            _CLOUD_URL,
            json={
                "method": "getCloudStorageRecordingsList",
                "params": {
                    "token": self._token,
                    "deviceId": device_id,
                    "startDate": start_date,
                    "endDate": end_date,
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error_code", -1) != 0:
            raise RuntimeError(
                f"Failed to list recordings (error {data.get('error_code')}): "
                f"{data.get('msg', data)}"
            )
        return data.get("result", {}).get("dates", [])

    def get_recordings(self, device_id: str, date: str) -> List[dict]:
        """Return recording segment dicts for *device_id* on *date* (YYYYMMDD)."""
        self._ensure_auth()
        resp = self._session.post(
            _CLOUD_URL,
            json={
                "method": "getCloudStorageRecordings",
                "params": {
                    "token": self._token,
                    "deviceId": device_id,
                    "date": date,
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error_code", -1) != 0:
            raise RuntimeError(
                f"Failed to get recordings for {date} (error {data.get('error_code')}): "
                f"{data.get('msg', data)}"
            )
        return data.get("result", {}).get("recordings", [])

    def get_download_url(
        self, device_id: str, start_time: int, end_time: int
    ) -> str:
        """Return a presigned download URL for the MP4 between *start_time* and *end_time*."""
        self._ensure_auth()
        resp = self._session.post(
            _CLOUD_URL,
            json={
                "method": "getCloudStorageDownloadUrl",
                "params": {
                    "token": self._token,
                    "deviceId": device_id,
                    "startTime": start_time,
                    "endTime": end_time,
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error_code", -1) != 0:
            raise RuntimeError(
                f"Failed to get download URL (error {data.get('error_code')}): "
                f"{data.get('msg', data)}"
            )
        return data["result"]["downloadUrl"]


class TapoCloudDownloader:
    """
    Lists and downloads camera recordings from Tapo Cloud for the last *days_back* days.

    Parameters
    ----------
    host:
        IP address or hostname of the Tapo camera (used to identify the device
        on the cloud account).
    username:
        Tapo account **email address** (used for cloud login).
    password:
        Local camera admin password (only needed for live RTSP streaming,
        not for cloud backfill).
    cloud_password:
        Tapo app / TP-Link account password (used for cloud login).
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
        self._cloud: Optional[TapoCloud] = None   # lazily initialised
        self._device_id: Optional[str] = None     # cached cloud device ID

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def list_recording_dates(self) -> List[str]:
        """
        Return a list of ``YYYYMMDD`` strings that have recordings within the
        configured backfill window.
        """
        cloud = self._get_cloud()
        device_id = self._get_device_id()
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=self.days_back)

        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        logger.info(
            "Querying cloud recording dates from %s to %s",
            start_str,
            end_str,
        )

        raw = cloud.get_recordings_list(device_id, start_str, end_str)
        # raw is a list of date strings like ["20240601", "20240602", ...]
        dates = [entry for entry in raw if isinstance(entry, str)]
        logger.info("Found %d date(s) with recordings.", len(dates))
        return dates

    def list_segments_for_date(self, date: str) -> List[RecordingSegment]:
        """
        Return :class:`RecordingSegment` objects for all recordings on *date*
        (``YYYYMMDD`` format).
        """
        cloud = self._get_cloud()
        device_id = self._get_device_id()
        logger.debug("Listing cloud recording segments for date %s", date)
        raw = cloud.get_recordings(device_id, date)
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
        Download one recording segment from Tapo Cloud and return the path to
        the saved MP4, or *None* if the download failed.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        file_path = str(
            self.output_dir / f"{segment.start_time}-{segment.end_time}.mp4"
        )

        existing = Path(file_path)
        if existing.exists() and existing.stat().st_size > 0:
            logger.info("Already downloaded: %s", file_path)
            return file_path

        logger.info("Downloading %s …", segment)
        try:
            cloud = self._get_cloud()
            device_id = self._get_device_id()
            url = cloud.get_download_url(device_id, segment.start_time, segment.end_time)
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(file_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info("Download complete: %s", file_path)
            return file_path
        except Exception as exc:
            logger.error("Failed to download %s: %s", segment, exc)
            # Remove any partial download
            try:
                Path(file_path).unlink(missing_ok=True)
            except OSError:
                pass
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

    def _get_cloud(self) -> TapoCloud:
        """Return a cached, authenticated :class:`TapoCloud` instance."""
        if self._cloud is None:
            logger.info("Authenticating with Tapo Cloud for account '%s'", self.username)
            self._cloud = TapoCloud(self.username, self.cloud_password)
            self._cloud.authenticate()
        return self._cloud

    def _get_device_id(self) -> str:
        """Return the cloud device ID that matches the configured *host*."""
        if self._device_id is not None:
            return self._device_id
        cloud = self._get_cloud()
        devices = cloud.get_devices()
        for device in devices:
            if (
                device.get("deviceIp") == self.host
                or device.get("alias") == self.host
            ):
                self._device_id = device["deviceId"]
                logger.info(
                    "Found cloud device '%s' (ID: %s)",
                    device.get("alias", self.host),
                    self._device_id,
                )
                return self._device_id
        raise RuntimeError(
            f"No cloud device found matching host '{self.host}'. "
            f"Available devices: "
            f"{[d.get('alias', d.get('deviceIp', '?')) for d in devices]}"
        )
