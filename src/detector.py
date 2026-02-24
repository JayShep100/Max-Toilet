"""
detector.py – Motion detection and wee/poo classification.

The detector operates on a configurable Region of Interest (ROI) within each
camera frame.  It uses a two-phase approach:

1. **Motion phase** – Background subtraction (MOG2) detects when the dog is
   present on the pad.
2. **Classification phase** – Once the dog has left (motion stops), a colour
   analysis of the pad ROI is used to decide whether a *wee*, a *poo*, or
   *nothing* occurred.

Colour heuristics
-----------------
- **Wee**: yellowish pixels in HSV space (hue ≈ 20-35°).
- **Poo**: dark brownish pixels in HSV space (hue ≈ 5-20°, low value/saturation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """Type of toilet event detected on the pad."""

    WEE = "wee"
    POO = "poo"
    UNKNOWN = "unknown"


@dataclass
class DetectionEvent:
    """A single detected toilet event."""

    event_type: EventType
    confidence: float  # 0.0 – 1.0
    motion_pixel_count: int = 0
    color_pixel_counts: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"DetectionEvent(type={self.event_type.value}, confidence={self.confidence:.2f})"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class DetectorConfig:
    """All tunable parameters for the :class:`PadDetector`."""

    # Region of Interest – (x, y, w, h).  None = full frame.
    pad_roi: Optional[Tuple[int, int, int, int]] = None

    # MOG2 background subtractor sensitivity
    motion_threshold: int = 500
    motion_min_area: int = 800

    # How many consecutive frames with motion = dog is considered "present"
    presence_frames_required: int = 10

    # Seconds (expressed as frame counts) after motion stops before classifying
    post_event_delay_frames: int = 25

    # HSV ranges for wee detection (yellowish)
    wee_hue_lower: Tuple[int, int, int] = (20, 40, 40)
    wee_hue_upper: Tuple[int, int, int] = (35, 255, 255)

    # HSV ranges for poo detection (dark brown)
    poo_hue_lower: Tuple[int, int, int] = (5, 40, 20)
    poo_hue_upper: Tuple[int, int, int] = (20, 200, 130)

    # Minimum number of matching pixels to count as a colour hit
    color_change_pixel_threshold: int = 300


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class PadDetector:
    """
    Stateful detector that processes a stream of frames and emits
    :class:`DetectionEvent` objects when a toilet event is identified.

    Parameters
    ----------
    config:
        Detector configuration.  Defaults are suitable for a typical indoor
        camera at 720p.
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        self._motion_frame_count: int = 0
        self._post_motion_countdown: int = 0
        self._dog_was_present: bool = False
        # Snapshot of pad ROI just before dog arrived (background reference)
        self._background_snapshot: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> Optional[DetectionEvent]:
        """
        Analyse a single frame and return a :class:`DetectionEvent` if a
        toilet event has been fully classified, otherwise return *None*.

        Parameters
        ----------
        frame:
            Full BGR camera frame.

        Returns
        -------
        DetectionEvent or None
        """
        roi = self._extract_roi(frame)
        motion_pixels = self._detect_motion(roi)
        motion_detected = motion_pixels >= self.config.motion_min_area

        if motion_detected:
            # Capture a background snapshot just before the dog arrives
            if self._motion_frame_count == 0 and self._background_snapshot is None:
                self._background_snapshot = roi.copy()
            self._motion_frame_count += 1
            if self._motion_frame_count >= self.config.presence_frames_required:
                self._dog_was_present = True
            self._post_motion_countdown = self.config.post_event_delay_frames
            return None

        # No motion in this frame
        if self._post_motion_countdown > 0:
            self._post_motion_countdown -= 1
            return None

        # Post-event countdown has finished – classify if dog was present
        if self._dog_was_present:
            event = self._classify(roi)
            logger.info("Event classified: %s", event)
            self._reset_state()
            return event

        # Idle: update background reference occasionally
        if self._motion_frame_count == 0:
            self._background_snapshot = roi.copy()

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """Crop frame to the configured pad ROI, or return the full frame."""
        if self.config.pad_roi is None:
            return frame
        x, y, w, h = self.config.pad_roi
        return frame[y : y + h, x : x + w]

    def _detect_motion(self, roi: np.ndarray) -> int:
        """Return total foreground pixel count in the ROI."""
        fg_mask = self._bg_subtractor.apply(roi)
        # Threshold and clean up noise
        _, thresh = cv2.threshold(fg_mask, self.config.motion_threshold // 10, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return int(np.sum(cleaned > 0))

    def _classify(self, roi: np.ndarray) -> DetectionEvent:
        """Classify the event using HSV colour analysis of the pad ROI."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        wee_lower = np.array(self.config.wee_hue_lower, dtype=np.uint8)
        wee_upper = np.array(self.config.wee_hue_upper, dtype=np.uint8)
        poo_lower = np.array(self.config.poo_hue_lower, dtype=np.uint8)
        poo_upper = np.array(self.config.poo_hue_upper, dtype=np.uint8)

        wee_mask = cv2.inRange(hsv, wee_lower, wee_upper)
        poo_mask = cv2.inRange(hsv, poo_lower, poo_upper)

        # Subtract background snapshot contribution when available
        if self._background_snapshot is not None:
            bg_hsv = cv2.cvtColor(self._background_snapshot, cv2.COLOR_BGR2HSV)
            bg_wee_mask = cv2.inRange(bg_hsv, wee_lower, wee_upper)
            bg_poo_mask = cv2.inRange(bg_hsv, poo_lower, poo_upper)
            wee_mask = cv2.subtract(wee_mask, bg_wee_mask)
            poo_mask = cv2.subtract(poo_mask, bg_poo_mask)

        wee_pixels = int(np.sum(wee_mask > 0))
        poo_pixels = int(np.sum(poo_mask > 0))
        threshold = self.config.color_change_pixel_threshold

        color_counts = {"wee_pixels": wee_pixels, "poo_pixels": poo_pixels}
        logger.debug("Colour analysis – wee: %d px, poo: %d px", wee_pixels, poo_pixels)

        if wee_pixels < threshold and poo_pixels < threshold:
            # Dog was on pad but no clear colour signature – log as unknown
            return DetectionEvent(
                event_type=EventType.UNKNOWN,
                confidence=0.5,
                motion_pixel_count=self._motion_frame_count,
                color_pixel_counts=color_counts,
            )

        if wee_pixels >= poo_pixels:
            total = max(wee_pixels + poo_pixels, 1)
            confidence = min(wee_pixels / total, 1.0)
            return DetectionEvent(
                event_type=EventType.WEE,
                confidence=confidence,
                motion_pixel_count=self._motion_frame_count,
                color_pixel_counts=color_counts,
            )

        total = max(wee_pixels + poo_pixels, 1)
        confidence = min(poo_pixels / total, 1.0)
        return DetectionEvent(
            event_type=EventType.POO,
            confidence=confidence,
            motion_pixel_count=self._motion_frame_count,
            color_pixel_counts=color_counts,
        )

    def _reset_state(self) -> None:
        """Reset per-event state after classification."""
        self._motion_frame_count = 0
        self._post_motion_countdown = 0
        self._dog_was_present = False
        self._background_snapshot = None
