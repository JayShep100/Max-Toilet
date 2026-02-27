"""
camera.py – Tapo camera connection and frame capture.

Connects to a Tapo camera via its RTSP stream using OpenCV, with optional
PyTapo-based API access for camera metadata/events.
"""

from __future__ import annotations

import logging
import time
from typing import Optional
from urllib.parse import quote, urlparse, urlunparse

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraConnectionError(Exception):
    """Raised when the camera cannot be reached or the stream cannot be opened."""


def _inject_credentials(stream_url: str, username: str, password: str) -> str:
    """Insert *username*:*password* into an RTSP URL that has no credentials.

    If the URL already contains a ``@`` (i.e. credentials are embedded),
    the original URL is returned unchanged.
    """
    parsed = urlparse(stream_url)
    if parsed.username is not None:
        return stream_url
    netloc = f"{quote(username, safe='')}:{quote(password, safe='')}@{parsed.hostname}"
    if parsed.port is not None:
        netloc += f":{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


class TapoCamera:
    """
    Connects to a Tapo camera RTSP stream and provides frame-by-frame access.

    Parameters
    ----------
    stream_url:
        Full RTSP URL, e.g. ``rtsp://admin:pass@192.168.1.100:554/stream1``.
        If *username* and *password* are supplied separately the URL may omit
        the credentials (e.g. ``rtsp://192.168.1.100:554/stream1``).
    username:
        RTSP username.  When provided together with *password*, credentials
        are injected into *stream_url* unless the URL already contains them.
    password:
        RTSP password.
    reconnect_attempts:
        How many times to retry opening the stream before raising.
    reconnect_delay:
        Seconds to wait between reconnection attempts.
    """

    def __init__(
        self,
        stream_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        reconnect_attempts: int = 5,
        reconnect_delay: float = 3.0,
    ) -> None:
        if username and password and urlparse(stream_url).username is None:
            stream_url = _inject_credentials(stream_url, username, password)
        self.stream_url = stream_url
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the RTSP stream.  Raises :class:`CameraConnectionError` on failure."""
        for attempt in range(1, self.reconnect_attempts + 1):
            logger.info("Connecting to camera (attempt %d/%d)…", attempt, self.reconnect_attempts)
            cap = cv2.VideoCapture(self.stream_url)
            if cap.isOpened():
                self._cap = cap
                logger.info("Camera stream opened successfully.")
                return
            cap.release()
            if attempt < self.reconnect_attempts:
                time.sleep(self.reconnect_delay)

        raise CameraConnectionError(
            f"Could not open camera stream after {self.reconnect_attempts} attempts: {self.stream_url}"
        )

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read the next frame from the stream.

        Returns
        -------
        numpy.ndarray or None
            The BGR frame, or *None* if the frame could not be read (e.g. stream
            hiccup).  Callers should handle *None* gracefully (skip the frame).
        """
        if self._cap is None:
            raise CameraConnectionError("Camera is not connected.  Call connect() first.")
        ret, frame = self._cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera stream.")
            return None
        return frame

    def release(self) -> None:
        """Release the underlying VideoCapture resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera stream released.")

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "TapoCamera":
        self.connect()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """``True`` if the stream is currently open."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_width(self) -> int:
        """Width of the video frame in pixels (0 if not connected)."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def frame_height(self) -> int:
        """Height of the video frame in pixels (0 if not connected)."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
