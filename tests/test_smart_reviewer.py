"""
test_smart_reviewer.py – Tests for smart_reviewer.py
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import smart_reviewer as sr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_video(path: Path, frames: int = 10, h: int = 64, w: int = 64) -> None:
    """Write a tiny solid-grey MP4."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(path), fourcc, 10.0, (w, h))
    frame  = np.full((h, w, 3), 128, dtype=np.uint8)
    for _ in range(frames):
        out.write(frame)
    out.release()


def _write_pad_clips_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# parse_clip_timestamp
# ---------------------------------------------------------------------------


class TestParseClipTimestamp:
    def test_space_separator(self) -> None:
        ts = sr.parse_clip_timestamp("2026-02-22 14-21-36.mp4")
        assert ts == datetime(2026, 2, 22, 14, 21, 36, tzinfo=timezone.utc)

    def test_T_separator(self) -> None:
        ts = sr.parse_clip_timestamp("2026-02-22T14-21-36.mp4")
        assert ts == datetime(2026, 2, 22, 14, 21, 36, tzinfo=timezone.utc)

    def test_no_match_returns_none(self) -> None:
        assert sr.parse_clip_timestamp("random_clip.mp4") is None

    def test_embedded_in_longer_stem(self) -> None:
        ts = sr.parse_clip_timestamp("cam1_2026-02-22 14-21-36_end.mp4")
        assert ts is not None
        assert ts.hour == 14


# ---------------------------------------------------------------------------
# extract_csv_features
# ---------------------------------------------------------------------------


class TestExtractCsvFeatures:
    def test_full_row(self) -> None:
        row = {
            "best_confidence":     "0.92",
            "dog_frame_count":     "30",
            "total_frames_checked": "100",
            "best_overlap":        "0.45",
            "pad_frame_count":     "12",
        }
        f = sr.extract_csv_features(row)
        assert f["best_confidence"] == pytest.approx(0.92)
        assert f["dog_frame_ratio"] == pytest.approx(0.30)
        assert f["best_overlap"]    == pytest.approx(0.45)
        assert f["pad_frame_count"] == pytest.approx(12.0)

    def test_missing_values_default_to_zero(self) -> None:
        f = sr.extract_csv_features({})
        assert f["best_confidence"] == 0.0
        assert f["dog_frame_ratio"] == 0.0

    def test_non_numeric_values_default_to_zero(self) -> None:
        row = {"best_confidence": "N/A", "dog_frame_count": ""}
        f   = sr.extract_csv_features(row)
        assert f["best_confidence"] == 0.0

    def test_zero_total_frames_does_not_divide_by_zero(self) -> None:
        row = {"dog_frame_count": "5", "total_frames_checked": "0"}
        f   = sr.extract_csv_features(row)
        assert f["dog_frame_ratio"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# extract_time_features
# ---------------------------------------------------------------------------


class TestExtractTimeFeatures:
    def _ts(self, hour: int) -> datetime:
        return datetime(2026, 2, 22, hour, 0, 0, tzinfo=timezone.utc)

    def test_hour_extracted(self) -> None:
        f = sr.extract_time_features("2026-02-22 09-00-00.mp4", [])
        assert f["hour_of_day"] == 9.0

    def test_time_since_last(self) -> None:
        earlier = self._ts(8)
        f = sr.extract_time_features(
            "2026-02-22 09-00-00.mp4", [earlier]
        )
        assert f["time_since_last_s"] == pytest.approx(3600.0)

    def test_no_previous_clip(self) -> None:
        f = sr.extract_time_features("2026-02-22 09-00-00.mp4", [])
        assert f["time_since_last_s"] == 0.0

    def test_gap_capped_at_one_day(self) -> None:
        two_days_ago = datetime(2026, 2, 20, 9, 0, 0, tzinfo=timezone.utc)
        f = sr.extract_time_features(
            "2026-02-22 09-00-00.mp4", [two_days_ago]
        )
        assert f["time_since_last_s"] == pytest.approx(86400.0)

    def test_unrecognised_filename_returns_defaults(self) -> None:
        f = sr.extract_time_features("clip_no_date.mp4", [])
        assert f["hour_of_day"]      == 12.0
        assert f["time_since_last_s"] == 0.0


# ---------------------------------------------------------------------------
# extract_video_features
# ---------------------------------------------------------------------------


class TestExtractVideoFeatures:
    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        f = sr.extract_video_features(tmp_path / "nonexistent.mp4")
        assert f["duration_seconds"] == 0.0
        assert f["bbox_cx"]          == 0.5

    def test_valid_video_returns_duration(self, tmp_path: Path) -> None:
        p = tmp_path / "vid.mp4"
        _write_video(p, frames=20)
        with patch("smart_reviewer._extract_dog_pose_features") as mock_pose:
            mock_pose.side_effect = ImportError("no ultralytics")
            f = sr.extract_video_features(p)
        assert f["duration_seconds"] > 0

    def test_corrupt_video_returns_defaults(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.mp4"
        p.write_bytes(b"not a video")
        f = sr.extract_video_features(p)
        assert f["duration_seconds"] == 0.0


# ---------------------------------------------------------------------------
# build_feature_vector
# ---------------------------------------------------------------------------


class TestBuildFeatureVector:
    def test_returns_correct_length(self) -> None:
        fv = sr.build_feature_vector({}, "2026-02-22 09-00-00.mp4", [])
        assert len(fv) == len(sr.FEATURE_NAMES)

    def test_all_numeric(self) -> None:
        fv = sr.build_feature_vector(
            {"best_confidence": "0.8", "pad_frame_count": "5"},
            "2026-02-22 09-00-00.mp4",
            [],
        )
        assert all(isinstance(v, float) for v in fv)


# ---------------------------------------------------------------------------
# load_labels
# ---------------------------------------------------------------------------


class TestLoadLabels:
    def test_loads_from_subdirectories(self, tmp_path: Path) -> None:
        (tmp_path / "Wee").mkdir()
        (tmp_path / "Poo").mkdir()
        (tmp_path / "Wee" / "clip1.mp4").write_bytes(b"")
        (tmp_path / "Poo" / "clip2.mp4").write_bytes(b"")

        labels = dict(sr.load_labels(tmp_path))
        assert labels.get("clip1.mp4") == "wee"
        assert labels.get("clip2.mp4") == "poo"

    def test_loads_from_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dog_labels.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["path", "label"])
            writer.writeheader()
            writer.writerow({"path": r"C:\clips\clip3.mp4", "label": "neither"})

        labels = dict(sr.load_labels(tmp_path))
        assert labels.get("clip3.mp4") == "neither"

    def test_folder_overrides_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dog_labels.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["path", "label"])
            writer.writeheader()
            writer.writerow({"path": "clip1.mp4", "label": "wee"})

        (tmp_path / "Poo").mkdir()
        (tmp_path / "Poo" / "clip1.mp4").write_bytes(b"")

        labels = dict(sr.load_labels(tmp_path))
        assert labels.get("clip1.mp4") == "poo"

    def test_empty_dest_root_returns_empty(self, tmp_path: Path) -> None:
        assert sr.load_labels(tmp_path) == []

    def test_invalid_label_in_csv_is_ignored(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dog_labels.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["path", "label"])
            writer.writeheader()
            writer.writerow({"path": "clip.mp4", "label": "unknown_label"})
        labels = dict(sr.load_labels(tmp_path))
        assert "clip.mp4" not in labels


# ---------------------------------------------------------------------------
# load_metadata
# ---------------------------------------------------------------------------


class TestLoadMetadata:
    def test_loads_from_pad_clips_csv(self, tmp_path: Path) -> None:
        p = tmp_path / "pad_clips.csv"
        _write_pad_clips_csv(p, [
            {"path": "/clips/2026-02-22 14-21-36.mp4", "best_confidence": "0.9"},
        ])
        meta = sr.load_metadata(p, None)
        assert "2026-02-22 14-21-36.mp4" in meta

    def test_merges_shortlist(self, tmp_path: Path) -> None:
        pad = tmp_path / "pad_clips.csv"
        sl  = tmp_path / "shortlist.csv"
        _write_pad_clips_csv(pad, [{"path": "clip_a.mp4", "best_confidence": "0.9"}])
        _write_pad_clips_csv(sl,  [{"path": "clip_b.mp4", "best_confidence": "0.7"}])
        meta = sr.load_metadata(pad, sl)
        assert "clip_a.mp4" in meta
        assert "clip_b.mp4" in meta

    def test_missing_csv_returns_empty(self, tmp_path: Path) -> None:
        meta = sr.load_metadata(tmp_path / "missing.csv", None)
        assert meta == {}

    def test_shortlist_does_not_override_pad_clips(self, tmp_path: Path) -> None:
        pad = tmp_path / "pad_clips.csv"
        sl  = tmp_path / "shortlist.csv"
        _write_pad_clips_csv(pad, [{"path": "clip_a.mp4", "best_confidence": "0.9"}])
        _write_pad_clips_csv(sl,  [{"path": "clip_a.mp4", "best_confidence": "0.5"}])
        meta = sr.load_metadata(pad, sl)
        assert meta["clip_a.mp4"]["best_confidence"] == "0.9"


# ---------------------------------------------------------------------------
# get_all_timestamps
# ---------------------------------------------------------------------------


class TestGetAllTimestamps:
    def test_returns_sorted_list(self) -> None:
        meta = {
            "2026-02-22 14-21-36.mp4": {},
            "2026-02-22 09-00-00.mp4": {},
        }
        ts = sr.get_all_timestamps(meta)
        assert ts == sorted(ts)
        assert len(ts) == 2

    def test_ignores_unrecognised_filenames(self) -> None:
        meta = {"random_clip.mp4": {}, "another.mp4": {}}
        ts = sr.get_all_timestamps(meta)
        assert ts == []


# ---------------------------------------------------------------------------
# train_model / predict
# ---------------------------------------------------------------------------


class TestTrainModel:
    def _make_labels(self, n_wee: int = 5, n_poo: int = 3, n_neither: int = 5):
        labels: list[tuple[str, str]] = []
        for i in range(n_wee):
            labels.append((f"wee_{i}.mp4", "wee"))
        for i in range(n_poo):
            labels.append((f"poo_{i}.mp4", "poo"))
        for i in range(n_neither):
            labels.append((f"neither_{i}.mp4", "neither"))
        return labels

    def test_returns_none_when_too_few_samples(self) -> None:
        labels = [("a.mp4", "wee"), ("b.mp4", "poo")]
        model, scaler = sr.train_model(labels, {}, [])
        assert model is None
        assert scaler is None

    def test_trains_with_enough_samples(self) -> None:
        labels = self._make_labels()
        model, scaler = sr.train_model(labels, {}, [])
        assert model is not None
        assert scaler is not None

    def test_predict_returns_valid_label(self) -> None:
        labels = self._make_labels()
        model, scaler = sr.train_model(labels, {}, [])
        fv = [0.0] * len(sr.FEATURE_NAMES)
        label, conf = sr.predict(model, scaler, fv)
        assert label in sr.LABELS
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------


class TestModelPersistence:
    def test_round_trip(self, tmp_path: Path) -> None:
        labels = [
            (f"clip_{i}.mp4", sr.LABELS[i % 3])
            for i in range(sr.MIN_SAMPLES_FOR_TRAINING + 2)
        ]
        model, scaler = sr.train_model(labels, {}, [])
        assert model is not None

        mp = tmp_path / "model.joblib"
        sp = tmp_path / "scaler.joblib"
        sr.save_model(model, scaler, mp, sp)

        loaded_model, loaded_scaler = sr.load_model(mp, sp)
        assert loaded_model is not None

        fv    = [0.0] * len(sr.FEATURE_NAMES)
        label, conf = sr.predict(loaded_model, loaded_scaler, fv)
        assert label in sr.LABELS

    def test_load_missing_files_returns_none(self, tmp_path: Path) -> None:
        model, scaler = sr.load_model(
            tmp_path / "nope.joblib",
            tmp_path / "nope2.joblib",
        )
        assert model  is None
        assert scaler is None


# ---------------------------------------------------------------------------
# append_log_entry / load_log / print_accuracy_stats
# ---------------------------------------------------------------------------


class TestAccuracyLog:
    def test_creates_csv_with_header(self, tmp_path: Path) -> None:
        log = tmp_path / "log.csv"
        sr.append_log_entry(log, "clip.mp4", "wee", 0.80, "wee")
        rows = sr.load_log(log)
        assert len(rows) == 1
        assert set(sr._LOG_COLUMNS).issubset(rows[0].keys())

    def test_appends_multiple_entries(self, tmp_path: Path) -> None:
        log = tmp_path / "log.csv"
        sr.append_log_entry(log, "a.mp4", "wee", 0.90, "wee")
        sr.append_log_entry(log, "b.mp4", "poo", 0.70, "neither")
        rows = sr.load_log(log)
        assert len(rows) == 2

    def test_was_correct_flag(self, tmp_path: Path) -> None:
        log = tmp_path / "log.csv"
        sr.append_log_entry(log, "c.mp4", "wee", 0.85, "wee")
        sr.append_log_entry(log, "d.mp4", "poo", 0.60, "neither")
        rows = sr.load_log(log)
        assert rows[0]["was_correct"] == "True"
        assert rows[1]["was_correct"] == "False"

    def test_load_missing_log_returns_empty(self, tmp_path: Path) -> None:
        assert sr.load_log(tmp_path / "missing.csv") == []

    def test_print_accuracy_stats_no_crash_on_empty(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        sr.print_accuracy_stats([])
        assert capsys.readouterr().out == ""

    def test_print_accuracy_stats_output(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        entries = [
            {"actual_label": "wee",     "was_correct": "True"},
            {"actual_label": "wee",     "was_correct": "False"},
            {"actual_label": "neither", "was_correct": "True"},
        ]
        sr.print_accuracy_stats(entries)
        out = capsys.readouterr().out
        assert "2/3" in out
        assert "Wee" in out


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


class TestStateManagement:
    def test_load_returns_empty_on_missing_file(self, tmp_path: Path) -> None:
        s = sr.load_state(tmp_path / "missing.json")
        assert s == {"reviewed": [], "skipped": []}

    def test_round_trip(self, tmp_path: Path) -> None:
        p     = tmp_path / "state.json"
        state = {"reviewed": ["a.mp4"], "skipped": ["b.mp4"]}
        sr.save_state(p, state)
        loaded = sr.load_state(p)
        assert loaded == state

    def test_load_handles_corrupt_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad_state.json"
        p.write_text("not valid json", encoding="utf-8")
        s = sr.load_state(p)
        assert s == {"reviewed": [], "skipped": []}


# ---------------------------------------------------------------------------
# move_or_copy_clip
# ---------------------------------------------------------------------------


class TestMoveOrCopyClip:
    def test_move_creates_dest_folder(self, tmp_path: Path) -> None:
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"data")
        dest_root = tmp_path / "labels"

        dest = sr.move_or_copy_clip(src, dest_root, "wee")
        assert dest.exists()
        assert dest.parent == dest_root / "Wee"
        assert not src.exists()

    def test_copy_keeps_source(self, tmp_path: Path) -> None:
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"data")
        dest_root = tmp_path / "labels"

        sr.move_or_copy_clip(src, dest_root, "poo", copy=True)
        assert src.exists()

    def test_dry_run_does_not_move(self, tmp_path: Path) -> None:
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"data")
        dest_root = tmp_path / "labels"

        sr.move_or_copy_clip(src, dest_root, "wee", dry_run=True)
        assert src.exists()
        assert not (dest_root / "Wee" / "clip.mp4").exists()


# ---------------------------------------------------------------------------
# find_unreviewed_clips
# ---------------------------------------------------------------------------


class TestFindUnreviewedClips:
    def _setup(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        """Create a minimal pad_clips.csv and two video files."""
        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()

        # Two clips
        clip_a = clips_dir / "2026-02-22 09-00-00.mp4"
        clip_b = clips_dir / "2026-02-22 10-00-00.mp4"
        _write_video(clip_a)
        _write_video(clip_b)

        pad_csv = tmp_path / "pad_clips.csv"
        _write_pad_clips_csv(pad_csv, [
            {"path": str(clip_a), "best_confidence": "0.9", "pad_frame_count": "5"},
            {"path": str(clip_b), "best_confidence": "0.8", "pad_frame_count": "3"},
        ])
        dest = tmp_path / "labels"
        return pad_csv, clips_dir, dest

    def test_returns_all_when_nothing_reviewed(self, tmp_path: Path) -> None:
        pad_csv, _, dest = self._setup(tmp_path)
        clips = sr.find_unreviewed_clips(pad_csv, None, dest, False, {})
        assert len(clips) == 2

    def test_excludes_reviewed_clips(self, tmp_path: Path) -> None:
        pad_csv, clips_dir, dest = self._setup(tmp_path)
        reviewed = str(clips_dir / "2026-02-22 09-00-00.mp4")
        state = {"reviewed": [reviewed], "skipped": []}
        clips = sr.find_unreviewed_clips(pad_csv, None, dest, False, state)
        assert len(clips) == 1

    def test_excludes_clips_in_label_folders(self, tmp_path: Path) -> None:
        pad_csv, clips_dir, dest = self._setup(tmp_path)
        wee_folder = dest / "Wee"
        wee_folder.mkdir(parents=True)
        (wee_folder / "2026-02-22 09-00-00.mp4").write_bytes(b"")
        clips = sr.find_unreviewed_clips(pad_csv, None, dest, False, {})
        assert len(clips) == 1

    def test_only_on_pad_filter(self, tmp_path: Path) -> None:
        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        clip_on  = clips_dir / "2026-02-22 09-00-00.mp4"
        clip_off = clips_dir / "2026-02-22 10-00-00.mp4"
        _write_video(clip_on)
        _write_video(clip_off)

        pad_csv = tmp_path / "pad_clips.csv"
        _write_pad_clips_csv(pad_csv, [
            {"path": str(clip_on),  "best_confidence": "0.9", "pad_frame_count": "5"},
            {"path": str(clip_off), "best_confidence": "0.8", "pad_frame_count": "0"},
        ])
        dest   = tmp_path / "labels"
        clips  = sr.find_unreviewed_clips(pad_csv, None, dest, True, {})
        fnames = [c[0].name for c in clips]
        assert "2026-02-22 09-00-00.mp4" in fnames
        assert "2026-02-22 10-00-00.mp4" not in fnames


# ---------------------------------------------------------------------------
# _prediction_colour
# ---------------------------------------------------------------------------


class TestPredictionColour:
    def test_high_confidence_is_green(self) -> None:
        assert sr._prediction_colour(0.9) == sr._COLOUR_HIGH

    def test_medium_confidence_is_yellow(self) -> None:
        assert sr._prediction_colour(0.5) == sr._COLOUR_MED

    def test_low_confidence_is_red(self) -> None:
        assert sr._prediction_colour(0.1) == sr._COLOUR_LOW


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


class TestCLIParser:
    def test_required_args(self) -> None:
        parser = sr._build_parser()
        args   = parser.parse_args([
            "--pad-clips", "pad.csv",
            "--dest-root", "/labels",
        ])
        assert args.pad_clips == Path("pad.csv")
        assert args.dest_root == Path("/labels")
        assert args.shortlist  is None
        assert not args.retrain
        assert not args.only_on_pad
        assert not args.copy
        assert not args.dry_run

    def test_all_optional_flags(self) -> None:
        parser = sr._build_parser()
        args   = parser.parse_args([
            "--pad-clips",    "pad.csv",
            "--shortlist",    "sl.csv",
            "--dest-root",    "/labels",
            "--retrain",
            "--only-on-pad",
            "--copy",
            "--dry-run",
            "--model-path",   "model.joblib",
            "--scaler-path",  "scaler.joblib",
            "--state-path",   "state.json",
            "--log-path",     "log.csv",
        ])
        assert args.shortlist    == Path("sl.csv")
        assert args.retrain
        assert args.only_on_pad
        assert args.copy
        assert args.dry_run
        assert args.model_path   == Path("model.joblib")


# ---------------------------------------------------------------------------
# train_model verbose output
# ---------------------------------------------------------------------------


class TestTrainModelVerbose:
    def _make_labels(self, n_wee: int = 5, n_poo: int = 3, n_neither: int = 5):
        labels: list[tuple[str, str]] = []
        for i in range(n_wee):
            labels.append((f"wee_{i}.mp4", "wee"))
        for i in range(n_poo):
            labels.append((f"poo_{i}.mp4", "poo"))
        for i in range(n_neither):
            labels.append((f"neither_{i}.mp4", "neither"))
        return labels

    def test_verbose_prints_classifier_info(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        labels = self._make_labels()
        sr.train_model(labels, {}, [], verbose=True)
        out = capsys.readouterr().out
        assert "Random Forest" in out
        assert "Training on" in out

    def test_verbose_prints_per_sample_progress(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        labels = self._make_labels(n_wee=2, n_poo=2, n_neither=6)
        sr.train_model(labels, {}, [], verbose=True)
        out = capsys.readouterr().out
        assert "Processing" in out
        assert "done" in out

    def test_verbose_prints_feature_extraction_summary(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        labels = self._make_labels()
        sr.train_model(labels, {}, [], verbose=True)
        out = capsys.readouterr().out
        assert "Feature extraction complete" in out

    def test_verbose_too_few_samples_prints_warning(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        labels = [("a.mp4", "wee"), ("b.mp4", "poo")]
        model, scaler = sr.train_model(labels, {}, [], verbose=True)
        out = capsys.readouterr().out
        assert model is None
        assert "Too few labeled samples" in out

    def test_silent_by_default(self, capsys: pytest.CaptureFixture) -> None:
        labels = self._make_labels()
        sr.train_model(labels, {}, [])
        out = capsys.readouterr().out
        assert out == ""


# ---------------------------------------------------------------------------
# run_review banner / step output
# ---------------------------------------------------------------------------


class TestRunReviewVerboseOutput:
    """Verify that run_review prints the step banner without hanging."""

    def _setup(self, tmp_path: Path):
        """Create minimal filesystem layout for run_review."""
        dest = tmp_path / "labels"
        dest.mkdir()

        pad_csv = tmp_path / "pad_clips.csv"
        pad_csv.write_text("path,best_confidence\n", encoding="utf-8")

        model_path  = tmp_path / "model.joblib"
        scaler_path = tmp_path / "scaler.joblib"
        state_path  = tmp_path / "state.json"
        log_path    = tmp_path / "log.csv"

        return dest, pad_csv, model_path, scaler_path, state_path, log_path

    def test_banner_is_printed(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        dest, pad_csv, mp, sp, stp, lp = self._setup(tmp_path)
        sr.run_review(
            pad_clips_csv=pad_csv,
            shortlist_csv=None,
            dest_root=dest,
            model_path=mp,
            scaler_path=sp,
            state_path=stp,
            log_path=lp,
        )
        out = capsys.readouterr().out
        assert "SMART REVIEWER" in out

    def test_step1_output_printed(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        dest, pad_csv, mp, sp, stp, lp = self._setup(tmp_path)
        sr.run_review(
            pad_clips_csv=pad_csv,
            shortlist_csv=None,
            dest_root=dest,
            model_path=mp,
            scaler_path=sp,
            state_path=stp,
            log_path=lp,
        )
        out = capsys.readouterr().out
        assert "STEP 1" in out
        assert "Scanning labelled folders" in out

    def test_step4_output_printed(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        dest, pad_csv, mp, sp, stp, lp = self._setup(tmp_path)
        sr.run_review(
            pad_clips_csv=pad_csv,
            shortlist_csv=None,
            dest_root=dest,
            model_path=mp,
            scaler_path=sp,
            state_path=stp,
            log_path=lp,
        )
        out = capsys.readouterr().out
        assert "STEP 4" in out
        assert "Finding unlabelled clips" in out

    def test_no_clips_message_when_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        dest, pad_csv, mp, sp, stp, lp = self._setup(tmp_path)
        sr.run_review(
            pad_clips_csv=pad_csv,
            shortlist_csv=None,
            dest_root=dest,
            model_path=mp,
            scaler_path=sp,
            state_path=stp,
            log_path=lp,
        )
        out = capsys.readouterr().out
        assert "No unreviewed clips found" in out


# ---------------------------------------------------------------------------
# Dog-pose feature defaults and new FEATURE_NAMES
# ---------------------------------------------------------------------------


class TestDogPoseFeatureDefaults:
    def test_missing_video_has_all_pose_defaults(self, tmp_path: Path) -> None:
        """extract_video_features returns pose defaults for a missing file."""
        f = sr.extract_video_features(tmp_path / "nonexistent.mp4")
        assert f["rear_hip_height_mean"]   == 0.0
        assert f["rear_hip_height_max"]    == 0.0
        assert f["spine_angle_mean"]       == 45.0
        assert f["spine_angle_min"]        == 45.0
        assert f["tail_angle_mean"]        == 0.0
        assert f["tail_height_mean"]       == 0.5
        assert f["rear_paw_spread"]        == 0.0
        assert f["front_rear_height_diff"] == 0.0
        assert f["dwell_frac"]             == 0.0
        assert f["motion_pattern"]         == 0.0
        assert f["bbox_aspect_mean"]       == 1.0

    def test_feature_names_length(self) -> None:
        """FEATURE_NAMES has 22 entries."""
        assert len(sr.FEATURE_NAMES) == 22

    def test_feature_vector_matches_feature_names(self) -> None:
        """build_feature_vector returns a vector whose length equals FEATURE_NAMES."""
        fv = sr.build_feature_vector({}, "2026-02-22 09-00-00.mp4", [])
        assert len(fv) == len(sr.FEATURE_NAMES)

    def test_pose_model_passed_to_extract_video_features(self, tmp_path: Path) -> None:
        """pose_model argument is forwarded from build_feature_vector."""
        p = tmp_path / "vid.mp4"
        _write_video(p)
        sentinel = object()
        calls: list = []
        original = sr.extract_video_features

        def _mock(video_path, pose_model=None):
            calls.append(pose_model)
            return original(video_path, pose_model=None)

        with patch("smart_reviewer.extract_video_features", side_effect=_mock):
            sr.build_feature_vector({}, "clip.mp4", [], video_path=p, pose_model=sentinel)

        assert calls and calls[0] is sentinel

    def test_new_pose_feature_names_present(self) -> None:
        """The new dog-pose feature names are all in FEATURE_NAMES."""
        expected = [
            "rear_hip_height_mean", "rear_hip_height_max",
            "spine_angle_mean", "spine_angle_min",
            "tail_angle_mean", "tail_height_mean",
            "rear_paw_spread", "front_rear_height_diff",
            "dwell_frac", "motion_pattern", "bbox_aspect_mean",
        ]
        for name in expected:
            assert name in sr.FEATURE_NAMES, f"{name!r} missing from FEATURE_NAMES"


# ---------------------------------------------------------------------------
# Online learning
# ---------------------------------------------------------------------------


class TestOnlineLearning:
    def _make_labels(self, n_wee: int = 5, n_poo: int = 3, n_neither: int = 5):
        labels: list[tuple[str, str]] = []
        for i in range(n_wee):
            labels.append((f"wee_{i}.mp4", "wee"))
        for i in range(n_poo):
            labels.append((f"poo_{i}.mp4", "poo"))
        for i in range(n_neither):
            labels.append((f"neither_{i}.mp4", "neither"))
        return labels

    def test_extra_samples_appended(self) -> None:
        """train_model incorporates extra_X / extra_y without error."""
        labels = self._make_labels()
        extra_X = [[float(i) for i in range(len(sr.FEATURE_NAMES))]]
        extra_y = ["wee"]
        model, scaler = sr.train_model(
            labels, {}, [], extra_X=extra_X, extra_y=extra_y
        )
        assert model is not None
        assert scaler is not None

    def test_extra_samples_increase_training_size(self) -> None:
        """With extra samples the model can train even when base labels are below threshold."""
        labels = [("wee_0.mp4", "wee"), ("poo_0.mp4", "poo")]  # only 2 = too few alone
        n_extra = sr.MIN_SAMPLES_FOR_TRAINING
        extra_X = [[0.0] * len(sr.FEATURE_NAMES)] * n_extra
        extra_y = ["neither"] * n_extra
        model, scaler = sr.train_model(
            labels, {}, [], extra_X=extra_X, extra_y=extra_y
        )
        # 2 base + n_extra >= MIN_SAMPLES_FOR_TRAINING  → should train
        assert model is not None

    def test_run_review_accepts_retrain_interval(self, tmp_path: Path) -> None:
        """run_review accepts retrain_interval without raising."""
        import inspect
        sig = inspect.signature(sr.run_review)
        assert "retrain_interval" in sig.parameters
        default = sig.parameters["retrain_interval"].default
        assert default == 10

    def test_run_review_accepts_pose_model_path(self) -> None:
        """run_review accepts pose_model_path parameter with a sensible default."""
        import inspect
        sig = inspect.signature(sr.run_review)
        assert "pose_model_path" in sig.parameters


# ---------------------------------------------------------------------------
# New CLI arguments
# ---------------------------------------------------------------------------


class TestCLIParserNewArgs:
    def test_pose_model_default(self) -> None:
        parser = sr._build_parser()
        args = parser.parse_args(["--pad-clips", "p.csv", "--dest-root", "/l"])
        assert args.pose_model == Path("dog_pose_model.pt")

    def test_pose_model_custom_path(self) -> None:
        parser = sr._build_parser()
        args = parser.parse_args([
            "--pad-clips", "p.csv", "--dest-root", "/l",
            "--pose-model", "custom_pose.pt",
        ])
        assert args.pose_model == Path("custom_pose.pt")

    def test_retrain_interval_default(self) -> None:
        parser = sr._build_parser()
        args = parser.parse_args(["--pad-clips", "p.csv", "--dest-root", "/l"])
        assert args.retrain_interval == 10

    def test_retrain_interval_zero_disables(self) -> None:
        parser = sr._build_parser()
        args = parser.parse_args([
            "--pad-clips", "p.csv", "--dest-root", "/l",
            "--retrain-interval", "0",
        ])
        assert args.retrain_interval == 0

    def test_retrain_interval_custom(self) -> None:
        parser = sr._build_parser()
        args = parser.parse_args([
            "--pad-clips", "p.csv", "--dest-root", "/l",
            "--retrain-interval", "5",
        ])
        assert args.retrain_interval == 5


# ---------------------------------------------------------------------------
# _print_retrain_box
# ---------------------------------------------------------------------------


class TestPrintRetrainBox:
    def test_output_contains_retrained_message(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        sr._print_retrain_box(n_total=60, n_new=10, session_y=["wee"] * 6 + ["poo"] * 2 + ["neither"] * 2)
        out = capsys.readouterr().out
        assert "MODEL RETRAINED" in out
        assert "60" in out
        assert "10" in out

    def test_output_contains_class_counts(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        sr._print_retrain_box(n_total=15, n_new=5, session_y=["wee", "poo", "neither", "wee", "poo"])
        out = capsys.readouterr().out
        assert "wee=2" in out
        assert "poo=2" in out
        assert "neither=1" in out
