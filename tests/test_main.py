"""
test_main.py – Tests for the main() CLI entry point, focused on --video-folder.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.main import main, run_video_folder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG = {
    "camera": {},
    "detection": {},
    "logging": {"log_level": "WARNING"},
    "cloud_backfill": {},
}


def _write_config(path: Path) -> str:
    import json

    config_file = path / "config.json"
    config_file.write_text(json.dumps(_MINIMAL_CONFIG), encoding="utf-8")
    return str(config_file)


# ---------------------------------------------------------------------------
# run_video_folder
# ---------------------------------------------------------------------------


class TestRunVideoFolder:
    def test_calls_folder_scanner_scan(self, tmp_path: Path) -> None:
        config_path = _write_config(tmp_path)
        video_folder = str(tmp_path / "videos")
        Path(video_folder).mkdir()

        with patch("src.main.FolderScanner") as MockScanner:
            mock_instance = MockScanner.return_value
            mock_instance.scan.return_value = 0
            run_video_folder(config_path, video_folder)
            MockScanner.assert_called_once()
            mock_instance.scan.assert_called_once()

    def test_folder_scanner_receives_correct_folder_path(self, tmp_path: Path) -> None:
        config_path = _write_config(tmp_path)
        video_folder = str(tmp_path / "videos")
        Path(video_folder).mkdir()

        with patch("src.main.FolderScanner") as MockScanner:
            MockScanner.return_value.scan.return_value = 0
            run_video_folder(config_path, video_folder)
            call_kwargs = MockScanner.call_args
            folder_arg = call_kwargs.kwargs.get("folder_path") or call_kwargs.args[0]
            assert folder_arg == video_folder

    def test_exits_when_folder_not_found(self, tmp_path: Path) -> None:
        config_path = _write_config(tmp_path)
        missing_folder = str(tmp_path / "nonexistent")

        with patch("src.main.FolderScanner") as MockScanner:
            MockScanner.return_value.scan.side_effect = FileNotFoundError("not found")
            with pytest.raises(SystemExit) as exc_info:
                run_video_folder(config_path, missing_folder)
            assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# main() CLI routing
# ---------------------------------------------------------------------------


class TestMainCliRouting:
    def test_video_folder_flag_calls_run_video_folder(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = _write_config(tmp_path)
        video_folder = str(tmp_path / "videos")
        Path(video_folder).mkdir()

        monkeypatch.setattr(
            sys, "argv", ["prog", "--config", config_path, "--video-folder", video_folder]
        )
        with patch("src.main.run_video_folder") as mock_rvf:
            main()
            mock_rvf.assert_called_once_with(config_path, video_folder)

    def test_video_folder_does_not_call_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = _write_config(tmp_path)
        video_folder = str(tmp_path / "videos")
        Path(video_folder).mkdir()

        monkeypatch.setattr(
            sys, "argv", ["prog", "--config", config_path, "--video-folder", video_folder]
        )
        with patch("src.main.run_video_folder"), patch("src.main.run") as mock_run:
            main()
            mock_run.assert_not_called()

    def test_video_folder_does_not_call_run_backfill(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = _write_config(tmp_path)
        video_folder = str(tmp_path / "videos")
        Path(video_folder).mkdir()

        monkeypatch.setattr(
            sys, "argv", ["prog", "--config", config_path, "--video-folder", video_folder]
        )
        with patch("src.main.run_video_folder"), patch("src.main.run_backfill") as mock_backfill:
            main()
            mock_backfill.assert_not_called()

    def test_no_video_folder_calls_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = _write_config(tmp_path)

        monkeypatch.setattr(sys, "argv", ["prog", "--config", config_path])
        with patch("src.main.run") as mock_run:
            main()
            mock_run.assert_called_once_with(config_path)

    def test_backfill_flag_calls_run_backfill(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = _write_config(tmp_path)

        monkeypatch.setattr(sys, "argv", ["prog", "--config", config_path, "--backfill"])
        with patch("src.main.run_backfill") as mock_backfill:
            main()
            mock_backfill.assert_called_once_with(config_path, 30)
