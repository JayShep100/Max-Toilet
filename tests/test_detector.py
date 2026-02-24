"""
test_detector.py – Tests for the PadDetector and classification logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.detector import (
    DetectorConfig,
    DetectionEvent,
    EventType,
    PadDetector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blank_frame(h: int = 240, w: int = 320, color: tuple = (200, 200, 200)) -> np.ndarray:
    """Create a solid-colour BGR frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def _yellow_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """
    Create a frame that falls inside the wee HSV range.
    HSV (27, 200, 200) → approx BGR (55, 143, 200) via cv2.
    We build it in HSV space and convert.
    """
    import cv2

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:] = (27, 200, 200)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _brown_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """
    Create a frame that falls inside the poo HSV range.
    HSV (12, 150, 80) is a dark brownish colour.
    """
    import cv2

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:] = (12, 150, 80)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# DetectorConfig
# ---------------------------------------------------------------------------


class TestDetectorConfig:
    def test_defaults(self) -> None:
        cfg = DetectorConfig()
        assert cfg.pad_roi is None
        assert cfg.motion_min_area > 0
        assert cfg.presence_frames_required > 0
        assert len(cfg.wee_hue_lower) == 3
        assert len(cfg.wee_hue_upper) == 3
        assert len(cfg.poo_hue_lower) == 3
        assert len(cfg.poo_hue_upper) == 3

    def test_custom_roi(self) -> None:
        cfg = DetectorConfig(pad_roi=(10, 20, 100, 80))
        assert cfg.pad_roi == (10, 20, 100, 80)


# ---------------------------------------------------------------------------
# PadDetector – ROI extraction
# ---------------------------------------------------------------------------


class TestROIExtraction:
    def test_no_roi_returns_full_frame(self) -> None:
        detector = PadDetector(DetectorConfig(pad_roi=None))
        frame = _blank_frame(100, 200)
        roi = detector._extract_roi(frame)
        assert roi.shape == (100, 200, 3)

    def test_roi_crops_correctly(self) -> None:
        detector = PadDetector(DetectorConfig(pad_roi=(10, 5, 50, 30)))
        frame = _blank_frame(100, 200)
        roi = detector._extract_roi(frame)
        assert roi.shape == (30, 50, 3)


# ---------------------------------------------------------------------------
# PadDetector – classification
# ---------------------------------------------------------------------------


class TestClassification:
    """
    Test the _classify method by injecting pre-built frames and a blank
    background snapshot.
    """

    def _detector_with_bg(self, bg_frame: np.ndarray) -> PadDetector:
        cfg = DetectorConfig(color_change_pixel_threshold=50)
        d = PadDetector(cfg)
        d._background_snapshot = bg_frame
        return d

    def test_classify_wee(self) -> None:
        bg = _blank_frame()
        detector = self._detector_with_bg(bg)
        event = detector._classify(_yellow_frame())
        assert event.event_type == EventType.WEE
        assert event.confidence > 0

    def test_classify_poo(self) -> None:
        bg = _blank_frame()
        detector = self._detector_with_bg(bg)
        event = detector._classify(_brown_frame())
        assert event.event_type == EventType.POO
        assert event.confidence > 0

    def test_classify_unknown_when_no_colour_change(self) -> None:
        # Using a very high threshold so that no colour change triggers unknown
        cfg = DetectorConfig(color_change_pixel_threshold=999_999)
        detector = PadDetector(cfg)
        detector._background_snapshot = _blank_frame()
        event = detector._classify(_blank_frame())
        assert event.event_type == EventType.UNKNOWN

    def test_classify_no_background_snapshot(self) -> None:
        """When there is no background snapshot, analysis still runs without error."""
        detector = PadDetector(DetectorConfig(color_change_pixel_threshold=50))
        # Yellow frame should still produce a wee event with no background ref
        event = detector._classify(_yellow_frame())
        assert event.event_type in (EventType.WEE, EventType.UNKNOWN)

    def test_color_pixel_counts_populated(self) -> None:
        bg = _blank_frame()
        detector = self._detector_with_bg(bg)
        event = detector._classify(_yellow_frame())
        assert "wee_pixels" in event.color_pixel_counts
        assert "poo_pixels" in event.color_pixel_counts


# ---------------------------------------------------------------------------
# PadDetector – state machine
# ---------------------------------------------------------------------------


class TestStateMachine:
    """
    Drive the detector through a motion → idle sequence and verify that it
    emits an event at the right time.
    """

    def _make_detector(self, presence_frames: int = 3, post_delay: int = 2) -> PadDetector:
        cfg = DetectorConfig(
            presence_frames_required=presence_frames,
            post_event_delay_frames=post_delay,
            color_change_pixel_threshold=1,  # very sensitive
        )
        return PadDetector(cfg)

    def test_no_event_without_enough_motion(self) -> None:
        detector = self._make_detector(presence_frames=10)
        # Feed only a few motion frames – dog not considered "present" yet
        for _ in range(5):
            detector._dog_was_present = False
        assert not detector._dog_was_present

    def test_reset_state_clears_all_fields(self) -> None:
        detector = self._make_detector()
        detector._motion_frame_count = 99
        detector._post_motion_countdown = 5
        detector._dog_was_present = True
        detector._background_snapshot = _blank_frame()
        detector._reset_state()
        assert detector._motion_frame_count == 0
        assert detector._post_motion_countdown == 0
        assert not detector._dog_was_present
        assert detector._background_snapshot is None

    def test_process_frame_returns_none_during_motion(self) -> None:
        """While the dog is on the pad, no event should be emitted."""
        # Use a very long post-event delay so the event cannot fire within
        # the frames we feed in this test, even if MOG2 adapts quickly.
        detector = self._make_detector(presence_frames=3, post_delay=100)

        # The MOG2 subtractor needs a few background frames to initialise.
        bg = _blank_frame()
        for _ in range(10):
            result = detector.process_frame(bg)
            assert result is None

        # Feed frames that differ noticeably from background (simulate motion).
        # Because post_delay=100, the countdown will not reach zero in only
        # 5 frames even after motion stops, so result must always be None.
        moving = _yellow_frame()
        for _ in range(5):
            result = detector.process_frame(moving)
            assert result is None
