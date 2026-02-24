"""
cloud_downloader.py – Download previously saved recordings from Tapo Cloud.

Tapo cameras can sync footage to Tapo Cloud (TP-Link's servers).  This module
uses the pytapo library's cloud client to:

  1. Authenticate with Tapo Cloud using the account email and password.
  2. Locate the camera device on the account by alias name or local IP address.
  3. List all recording segments available in the last N days.
  4. Download each segment as an MP4 file via HTTP streaming.

Downloaded files are placed in a configurable output directory and named
``<start_epoch>-<end_epoch>.mp4``.

Config notes
............
- ``username``       : Your Tapo/TP-Link **account email address**
- ``cloud_password`` : Your Tapo/TP-Link **account password**
- ``password``       : Local camera admin password (only needed for live stream)
- ``host``           : Camera LAN IP, used as fallback to identify the device
- ``camera_alias``   : Camera name as shown in the Tapo app (e.g. "Hall")

Usage example
-------------
    downloader = TapoCloudDownloader(
        host="192.168.1.100",
        username="you@example.com",
        password="local_camera_pass",
        cloud_password="your_tapo_account_password",
        output_dir="downloads",
        days_back=30,
        camera_alias="Hall",
    )
    for path, info in downloader.download_recordings():
        process_video_file(path, detector, event_logger)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Maximum days of backfill that will ever be requested
MAX_DAYS_BACK = 30

# Tapo Cloud API endpoint
TAPO_CLOUD_URL = "https://eu-wap.tplinkcloud.com"


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
            f"RecordingSegment({self.start_dt.isoformat()} \u2013 {self.end_dt.isoformat()})"
        )


class TapoCloudDownloader:
    """
    Lists and downloads camera recordings for the last *days_back* days
    from Tapo Cloud (no SD card required).

    Parameters
    ----------
    host:
        LAN IP address of the Tapo camera, used as a fallback to identify
        the device on your cloud account.
    username:
        Your Tapo/TP-Link **account email address**.
    password:
        Local camera admin password (used only for live RTSP stream, not
        required for cloud backfill).
    cloud_password:
        Your Tapo/TP-Link **account password** (used for cloud login).
    output_dir:
        Directory where downloaded MP4 files are saved.  Created if absent.
    days_back:
        How many days of history to fetch (capped at :data:`MAX_DAYS_BACK`).
    camera_alias:
        The camera name exactly as shown in the Tapo app (e.g. ``"Hall"``).
        When supplied this takes priority over IP-based matching.
    """

    def __init__(self,
        host: str,
        username: str,
        password: str,
        cloud_password: str,
        output_dir: str,
        days_back: int = 30,
        camera_alias: Optional[str] = None,
    ) -> None:
        self.host = host
        self.username = username
        self.password = password
        self.cloud_password = cloud_password
        self.output_dir = Path(output_dir)
        self.days_back = min(int(days_back), MAX_DAYS_BACK)
        self.camera_alias = camera_alias
        self._cloud_token: Optional[str] = None
        self._device_id: Optional[str] = None
        self._device_server_url: Optional[str] = None
        self._tapo = None  # lazily initialised local connection

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def list_recording_dates(self) -> List[str]:
        """
        Return a list of ``YYYYMMDD`` strings that have recordings within the
        configured backfill window.
        """
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

        tapo = self._get_tapo()
        try:
            if tapo is None:
                raise RuntimeError("No local connection available")
            raw = tapo.getRecordingsList(start_date=start_str, end_date=end_str)
            dates = [entry for entry in raw if isinstance(entry, str)]
        except Exception as exc:
            logger.warning(
                "Local SD card query failed (%s) \u2013 falling back to cloud API.", exc
            )
            dates = self._list_dates_from_cloud(start_date, end_date)

        logger.info("Found %d date(s) with recordings.", len(dates))
        return dates

    def list_segments_for_date(self, date: str) -> List[RecordingSegment]:
        """
        Return :class:`RecordingSegment` objects for all recordings on *date*
        (``YYYYMMDD`` format).
        """
        tapo = self._get_tapo()
        logger.debug("Listing recording segments for date %s", date)
        try:
            if tapo is None:
                raise RuntimeError("No local connection available")
            raw = tapo.getRecordings(date)
        except Exception as exc:
            logger.warning(
                "Local segment query failed for %s (%s) \u2013 falling back to cloud API.",
                date, exc
            )
            return self._list_segments_from_cloud(date)

        segments: List[RecordingSegment] = []
        for entry in raw:
            start = entry.get("startTime") or entry.get("start_time")
            end = entry.get("endTime") or entry.get("end_time")
            if start is not None and end is not None:
                segments.append(RecordingSegment(start_time=int(start), end_time=int(end)))
        logger.debug("  \u2192 %d segment(s) on %s", len(segments), date)
        return segments

    def download_segment(self, segment: RecordingSegment) -> Optional[str]:
        """
        Download one recording segment and return the path to the saved MP4,
        or *None* if the download failed.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.output_dir / f"{segment.start_time}-{segment.end_time}.mp4"

        if file_path.exists():
            logger.info("Already downloaded, skipping: %s", file_path)
            return str(file_path)

        logger.info("Downloading %s \u2026", segment)

        # Try cloud download first (no SD card needed)
        result = self._download_segment_from_cloud(segment, file_path)
        if result:
            return result

        # Fallback: try local pytapo downloader (requires SD card)
        return self._download_segment_local(segment, file_path)

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
    # Cloud API helpers
    # ------------------------------------------------------------------

    def _get_cloud_token(self) -> str:
        """Authenticate with Tapo Cloud and return the auth token."""
        if self._cloud_token:
            return self._cloud_token

        logger.info("Authenticating with Tapo Cloud as %s", self.username)
        payload = {
            "method": "login",
            "params": {
                "appType": "Tapo_Ios",
                "cloudPassword": self.cloud_password,
                "cloudUserName": self.username,
                "terminalUUID": "Max-Toilet-Backfill",
            },
        }
        resp = requests.post(TAPO_CLOUD_URL, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        error_code = data.get("error_code", -1)
        if error_code != 0:
            raise RuntimeError(
                f"Tapo Cloud login failed (error_code={error_code}): "
                f"{data.get('msg', data)}"
            )
        self._cloud_token = data["result"]["token"]
        logger.info("Tapo Cloud authentication successful.")
        return self._cloud_token

    def _get_device_id(self) -> str:
        """Find this camera's deviceId on the Tapo Cloud account."""
        if self._device_id:
            return self._device_id

        token = self._get_cloud_token()
        logger.info(
            "Fetching device list from Tapo Cloud to find camera '%s' / %s",
            self.camera_alias or "(by IP)",
            self.host,
        )
        payload = {"method": "getDeviceList"}
        resp = requests.post(
            f"{TAPO_CLOUD_URL}?token={token}", json=payload, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error_code", -1) != 0:
            raise RuntimeError(
                f"Failed to fetch device list: {data.get('msg', data)}"
            )

        devices = data["result"].get("deviceList", [])
        logger.debug("Found %d device(s) on Tapo Cloud account.", len(devices))

        # Filter to camera-type devices only
        camera_devices = [
            d for d in devices
            if "camera" in d.get("deviceType", "").lower()
            or d.get("deviceModel", "").upper().startswith("C")
        ]

        # 1. Match by camera_alias (exact, case-insensitive)
        if self.camera_alias:
            for device in camera_devices:
                alias = device.get("alias", "")
                if alias.lower() == self.camera_alias.lower():
                    self._device_id = device["deviceId"]
                    self._device_server_url = device.get("deviceServerUrl") or TAPO_CLOUD_URL
                    logger.info(
                        "Matched camera by alias '%s' (deviceId: %s)",
                        alias, self._device_id
                    )
                    return self._device_id
            logger.warning(
                "Could not find camera with alias '%s'. Available cameras: %s",
                self.camera_alias,
                [d.get("alias") for d in camera_devices],
            )

        # 2. Match by IP in alias or remark
        for device in camera_devices:
            alias = device.get("alias", "")
            remark = device.get("deviceRemark", "")
            if self.host in alias or self.host in remark:
                self._device_id = device["deviceId"]
                self._device_server_url = device.get("deviceServerUrl") or TAPO_CLOUD_URL
                logger.info("Matched camera by IP in alias/remark: %s", alias)
                return self._device_id

        # 3. Fall back to first camera device
        if camera_devices:
            self._device_id = camera_devices[0]["deviceId"]
            self._device_server_url = camera_devices[0].get("deviceServerUrl") or TAPO_CLOUD_URL
            logger.warning(
                "Could not match camera by alias or IP; using first camera on account: %s",
                camera_devices[0].get("alias", self._device_id),
            )
            return self._device_id

        # 4. Last resort: first device
        if devices:
            self._device_id = devices[0]["deviceId"]
            self._device_server_url = devices[0].get("deviceServerUrl") or TAPO_CLOUD_URL
            logger.warning(
                "No camera-type devices found; using first device: %s",
                devices[0].get("alias", self._device_id),
            )
            return self._device_id

        raise RuntimeError(
            f"No devices found on Tapo Cloud account for {self.username}. "
            "Ensure the camera is registered in the Tapo app."
        )

    def _list_dates_from_cloud(
        self, start_date: datetime, end_date: datetime
    ) -> List[str]:
        """List recording dates via the Tapo Cloud API."""
        token = self._get_cloud_token()
        device_id = self._get_device_id()
        server_url = self._device_server_url or TAPO_CLOUD_URL

        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        payload = {
            "method": "searchDateWithVideo",
            "params": {
                "deviceId": device_id,
                "start_date": start_str,
                "end_date": end_str,
                "start_index": 0,
                "end_index": 99,
            },
        }
        resp = requests.post(
            f"{server_url}?token={token}", json=payload, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("error_code", -1) != 0:
            logger.warning("Cloud date search returned error: %s", data)
            return []

        date_list = data.get("result", {}).get("date_list", [])
        if isinstance(date_list, list):
            return [str(d) for d in date_list if d]
        return []

    def _list_segments_from_cloud(self, date: str) -> List[RecordingSegment]:
        """List recording segments for a specific date via the Tapo Cloud API."""
        token = self._get_cloud_token()
        device_id = self._get_device_id()
        server_url = self._device_server_url or TAPO_CLOUD_URL

        payload = {
            "method": "searchVideoWithPage",
            "params": {
                "deviceId": device_id,
                "date": date,
                "start_index": 0,
                "end_index": 999,
            },
        }
        resp = requests.post(
            f"{server_url}?token={token}", json=payload, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("error_code", -1) != 0:
            logger.warning("Cloud segment search returned error for %s: %s", date, data)
            return []

        video_list = data.get("result", {}).get("video_list", [])
        segments: List[RecordingSegment] = []
        for entry in video_list:
            start = entry.get("startTime") or entry.get("start_time")
            end = entry.get("endTime") or entry.get("end_time")
            if start is not None and end is not None:
                segments.append(
                    RecordingSegment(start_time=int(start), end_time=int(end))
                )
        logger.debug("  \u2192 %d cloud segment(s) on %s", len(segments), date)
        return segments

    def _download_segment_from_cloud(
        self, segment: RecordingSegment, file_path: Path
    ) -> Optional[str]:
        """Download a segment via Tapo Cloud HTTP URL. Returns file path or None."""
        try:
            token = self._get_cloud_token()
            device_id = self._get_device_id()
            server_url = self._device_server_url or TAPO_CLOUD_URL

            payload = {
                "method": "getVideoDownloadUrl",
                "params": {
                    "deviceId": device_id,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                },
            }
            resp = requests.post(
                f"{server_url}?token={token}", json=payload, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("error_code", -1) != 0:
                logger.debug(
                    "Cloud download URL request failed for %s: %s", segment, data
                )
                return None

            result = data.get("result", {})
            url = result.get("url") or result.get("download_url")

            if not url:
                logger.debug("No download URL in cloud response for %s", segment)
                return None

            logger.info("Streaming download from cloud for %s", segment)
            with requests.get(url, stream=True, timeout=120) as dl_resp:
                dl_resp.raise_for_status()
                with open(file_path, "wb") as f:
                    for chunk in dl_resp.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)

            if file_path.exists() and file_path.stat().st_size > 0:
                logger.info("Cloud download complete: %s", file_path)
                return str(file_path)

            logger.warning("Cloud download produced empty file for %s", segment)
            if file_path.exists():
                file_path.unlink()
            return None

        except Exception as exc:
            logger.debug("Cloud download failed for %s: %s", segment, exc)
            return None

    def _download_segment_local(
        self, segment: RecordingSegment, file_path: Path
    ) -> Optional[str]:
        """Fallback: download via local pytapo Downloader (requires SD card)."""
        import asyncio
        from pytapo.media_stream.downloader import Downloader

        tapo = self._get_tapo()
        if tapo is None:
            logger.warning("No local camera connection available for %s", segment)
            return None
        try:
            time_correction = tapo.getTimeCorrection()
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
                fp = result.get("fileName")
            else:
                fp = str(file_path)

            if fp and Path(fp).exists():
                logger.info("Local download complete: %s", fp)
                return fp
            logger.warning("Local download produced no file for %s", segment)
            return None
        except Exception as exc:
            logger.error("Local download failed for %s: %s", segment, exc)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_tapo(self):
        """Return a cached :class:`pytapo.Tapo` instance (local LAN connection)."""
        if self._tapo is None:
            from pytapo import Tapo

            logger.info("Connecting to Tapo camera API at %s", self.host)
            try:
                self._tapo = Tapo(
                    self.host,
                    self.username,
                    self.password,
                    self.cloud_password,
                )
            except Exception as exc:
                logger.warning(
                    "Could not connect to local camera at %s (%s). "
                    "Cloud-only mode will be used.",
                    self.host, exc
                )
                self._tapo = None
        return self._tapo
