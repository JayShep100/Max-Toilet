"""
main.py – Entry point for the Max-Toilet detection logger.

Usage
-----
    python -m src.main --config config.json

The application connects to a Tapo camera, continuously reads frames, and
passes them through the :class:`~detector.PadDetector`.  Any detected toilet
events are written to the configured log files via :class:`~logger.EventLogger`.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict

from .camera import TapoCamera, CameraConnectionError
from .cloud_downloader import TapoCloudDownloader
from .detector import DetectorConfig, PadDetector
from .folder_scanner import FolderScanner
from .logger import EventLogger
from .video_processor import process_video_file

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s \u2013 %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

def _load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _build_detector_config(cfg: Dict[str, Any]) -> DetectorConfig:
    det = cfg.get("detection", {})
    roi_cfg = det.get("pad_roi")
    pad_roi = None
    if roi_cfg and isinstance(roi_cfg, dict):
        pad_roi = (roi_cfg["x"], roi_cfg["y"], roi_cfg["width"], roi_cfg["height"])

    def _triple(key: str, default: tuple) -> tuple:
        val = det.get(key)
        if val and len(val) == 3:
            return tuple(int(v) for v in val)
        return default

    return DetectorConfig(
        pad_roi=pad_roi,
        motion_threshold=int(det.get("motion_threshold", 500)),
        motion_min_area=int(det.get("motion_min_area", 800)),
        presence_frames_required=int(det.get("presence_frames_required", 10)),
        post_event_delay_frames=int(det.get("post_event_delay_seconds", 5)) * 5,
        wee_hue_lower=_triple("wee_hue_lower", (20, 40, 40)),
        wee_hue_upper=_triple("wee_hue_upper", (35, 255, 255)),
        poo_hue_lower=_triple("poo_hue_lower", (5, 40, 20)),
        poo_hue_upper=_triple("poo_hue_upper", (20, 200, 130)),
        color_change_pixel_threshold=int(det.get("color_change_pixel_threshold", 300)),
        color_change_ratio_threshold=float(det.get("color_change_ratio_threshold", 0.005)),
        warmup_frames=int(det.get("warmup_frames", 30)),
        dominance_ratio=float(det.get("dominance_ratio", 1.5)),
        cooldown_frames=int(det.get("cooldown_frames", 50)),
    )

def run_backfill(config_path: str, days_back: int) -> None:
    """Download and process up to *days_back* days of cloud recordings."""
    cfg = _load_config(config_path)
    _setup_logging(cfg.get("logging", {}).get("log_level", "INFO"))

    log = logging.getLogger(__name__)
    log.info("Max-Toilet backfill starting (last %d day(s))\u2026", days_back)

    camera_cfg = cfg.get("camera", {})
    backfill_cfg = cfg.get("cloud_backfill", {})

    downloader = TapoCloudDownloader(
        host=camera_cfg.get("host", ""),
        username=camera_cfg.get("username", "admin"),
        password=camera_cfg.get("password", ""),
        cloud_password=camera_cfg.get("cloud_password", ""),
        output_dir=backfill_cfg.get("download_dir", "downloads"),
        days_back=days_back,
        camera_alias=camera_cfg.get("camera_alias"),
    )

    detector_config = _build_detector_config(cfg)
    log_cfg = cfg.get("logging", {})
    event_logger = EventLogger(
        log_dir=log_cfg.get("log_dir", "logs"),
        csv_filename=log_cfg.get("csv_filename", "toilet_events.csv"),
        json_filename=log_cfg.get("json_filename", "toilet_events.json"),
    )

    total_events = 0
    total_files = 0
    for file_path, segment in downloader.download_recordings():
        total_files += 1
        # Each video file gets a fresh detector so MOG2 initialises on that clip
        detector = PadDetector(config=detector_config)
        n = process_video_file(
            video_path=file_path,
            detector=detector,
            event_logger=event_logger,
            recording_start=segment.start_dt,
            source_label=f"{segment.start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        )
        total_events += n

    log.info(
        "Backfill complete: %d file(s) processed, %d event(s) logged.",
        total_files,
        total_events,
    )

def run_video_folder(config_path: str, video_folder: str) -> None:
    """Scan a local folder of video files and process each through the detection pipeline."""
    cfg = _load_config(config_path)
    _setup_logging(cfg.get("logging", {}).get("log_level", "INFO"))

    log = logging.getLogger(__name__)
    log.info("Max-Toilet video-folder scan starting: '%s'", video_folder)

    detector_config = _build_detector_config(cfg)
    log_cfg = cfg.get("logging", {})
    event_logger = EventLogger(
        log_dir=log_cfg.get("log_dir", "logs"),
        csv_filename=log_cfg.get("csv_filename", "toilet_events.csv"),
        json_filename=log_cfg.get("json_filename", "toilet_events.json"),
    )

    scanner = FolderScanner(
        folder_path=video_folder,
        detector_config=detector_config,
        event_logger=event_logger,
    )

    try:
        total_events = scanner.scan()
    except FileNotFoundError as exc:
        log.error("Video folder not found: %s", exc)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover
        log.error("Error processing video folder: %s", exc)
        sys.exit(1)

    log.info("Video-folder scan complete: %d event(s) logged.", total_events)


def run(config_path: str) -> None:
    """Main application loop."""
    cfg = _load_config(config_path)
    _setup_logging(cfg.get("logging", {}).get("log_level", "INFO"))

    log = logging.getLogger(__name__)
    log.info("Max-Toilet detection logger starting\u2026")

    # Build components
    camera_cfg = cfg.get("camera", {})
    stream_url = camera_cfg.get("stream_url", "")
    camera = TapoCamera(
        stream_url=stream_url,
        username=camera_cfg.get("username"),
        password=camera_cfg.get("password"),
    )

    detector_config = _build_detector_config(cfg)
    detector = PadDetector(config=detector_config)

    log_cfg = cfg.get("logging", {})
    event_logger = EventLogger(
        log_dir=log_cfg.get("log_dir", "logs"),
        csv_filename=log_cfg.get("csv_filename", "toilet_events.csv"),
        json_filename=log_cfg.get("json_filename", "toilet_events.json"),
    )

    # Graceful shutdown on SIGINT / SIGTERM
    _running = [True]

    def _shutdown(signum: int, _frame: object) -> None:
        log.info("Shutdown signal received (%d).  Stopping\u2026", signum)
        _running[0] = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Connect & run
    try:
        camera.connect()
    except CameraConnectionError as exc:
        log.error("Failed to connect to camera: %s", exc)
        sys.exit(1)

    log.info("Connected to camera stream.  Monitoring pad\u2026")
    consecutive_read_failures = 0
    max_consecutive_failures = 30

    try:
        while _running[0]:
            frame = camera.read_frame()
            if frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures >= max_consecutive_failures:
                    log.error("Too many consecutive frame-read failures.  Exiting.")
                    break
                time.sleep(0.1)
                continue

            consecutive_read_failures = 0
            event = detector.process_frame(frame)
            if event is not None:
                event_logger.log_event(event)
    finally:
        camera.release()
        log.info("Max-Toilet detection logger stopped.")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Max-Toilet \u2013 Dog toilet pad detection logger"
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the JSON configuration file (default: config.json)",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help=(
            "Download previously saved recordings from the Tapo camera and "
            "run them through the detector instead of monitoring the live stream."
        ),
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        metavar="N",
        help="Number of days of history to backfill (default: 30, max: 30)",
    )
    parser.add_argument(
        "--video-folder",
        metavar="PATH",
        help=(
            "Path to a local directory of video files to process. "
            "Supported formats: .mp4 .avi .mkv .mov .wmv .flv .webm. "
            "When provided, the script processes the folder and exits."
        ),
    )
    args = parser.parse_args()
    if args.video_folder:
        run_video_folder(args.config, args.video_folder)
    elif args.backfill:
        run_backfill(args.config, args.days)
    else:
        run(args.config)

if __name__ == "__main__":
    main()