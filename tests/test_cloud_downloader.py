"""
test_cloud_downloader.py – Tests for TapoCloudDownloader.

All pytapo network calls are mocked so the tests run without a real camera.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from src.cloud_downloader import (
    MAX_DAYS_BACK,
    RecordingSegment,
    TapoCloudDownloader,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def downloader(tmp_path: Path) -> TapoCloudDownloader:
    """Return a downloader pointing to a temp directory (no live camera)."""
    d = TapoCloudDownloader(
        host="192.168.1.10",
        username="admin",
        password="localpass",
        cloud_password="cloudpass",
        output_dir=str(tmp_path / "downloads"),
        days_back=7,
    )
    return d


def _make_tapo_mock(dates=None, segments=None):
    """Return a mock Tapo object with configurable responses."""
    mock = MagicMock()
    mock.getRecordingsList.return_value = dates or ["20250601", "20250602"]
    mock.getRecordings.return_value = segments or [
        {"startTime": 1748736000, "endTime": 1748739600},
    ]
    mock.getTimeCorrection.return_value = 0
    return mock


# ---------------------------------------------------------------------------
# RecordingSegment
# ---------------------------------------------------------------------------


class TestRecordingSegment:
    def test_start_dt(self) -> None:
        seg = RecordingSegment(start_time=0, end_time=3600)
        assert seg.start_dt == datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_end_dt(self) -> None:
        seg = RecordingSegment(start_time=0, end_time=3600)
        assert seg.end_dt == datetime(1970, 1, 1, 1, 0, 0, tzinfo=timezone.utc)

    def test_str(self) -> None:
        seg = RecordingSegment(start_time=0, end_time=3600)
        text = str(seg)
        assert "RecordingSegment" in text
        assert "1970" in text


# ---------------------------------------------------------------------------
# TapoCloudDownloader – initialisation
# ---------------------------------------------------------------------------


class TestDownloaderInit:
    def test_days_back_capped_at_max(self, tmp_path: Path) -> None:
        d = TapoCloudDownloader(
            host="h", username="u", password="p", cloud_password="c",
            output_dir=str(tmp_path), days_back=999,
        )
        assert d.days_back == MAX_DAYS_BACK

    def test_days_back_within_limit(self, tmp_path: Path) -> None:
        d = TapoCloudDownloader(
            host="h", username="u", password="p", cloud_password="c",
            output_dir=str(tmp_path), days_back=10,
        )
        assert d.days_back == 10

    def test_output_dir_stored_as_path(self, tmp_path: Path, downloader: TapoCloudDownloader) -> None:
        assert isinstance(downloader.output_dir, Path)


# ---------------------------------------------------------------------------
# TapoCloudDownloader – list_recording_dates
# ---------------------------------------------------------------------------


class TestListRecordingDates:
    def test_returns_dates_from_tapo(self, downloader: TapoCloudDownloader) -> None:
        mock_tapo = _make_tapo_mock(dates=["20250601", "20250602", "20250603"])
        downloader._tapo = mock_tapo

        dates = downloader.list_recording_dates()

        assert dates == ["20250601", "20250602", "20250603"]
        mock_tapo.getRecordingsList.assert_called_once()

    def test_date_range_limited_to_days_back(self, downloader: TapoCloudDownloader) -> None:
        mock_tapo = _make_tapo_mock(dates=[])
        downloader._tapo = mock_tapo
        downloader.list_recording_dates()

        call_args = mock_tapo.getRecordingsList.call_args
        positional = call_args[0]
        keyword = call_args[1]
        start = positional[0] if positional else keyword.get("start_date")

        # start_date should be 7 days before today (downloader.days_back=7)
        from datetime import date, timedelta
        expected_start = (date.today() - timedelta(days=7)).strftime("%Y%m%d")
        assert start == expected_start

    def test_non_string_entries_filtered(self, downloader: TapoCloudDownloader) -> None:
        mock_tapo = _make_tapo_mock(dates=["20250601", None, 12345, "20250602"])
        downloader._tapo = mock_tapo
        dates = downloader.list_recording_dates()
        assert dates == ["20250601", "20250602"]


# ---------------------------------------------------------------------------
# TapoCloudDownloader – list_segments_for_date
# ---------------------------------------------------------------------------


class TestListSegmentsForDate:
    def test_parses_camel_case_keys(self, downloader: TapoCloudDownloader) -> None:
        mock_tapo = _make_tapo_mock(segments=[
            {"startTime": 1748736000, "endTime": 1748739600},
            {"startTime": 1748739600, "endTime": 1748743200},
        ])
        downloader._tapo = mock_tapo
        segs = downloader.list_segments_for_date("20250601")

        assert len(segs) == 2
        assert segs[0].start_time == 1748736000
        assert segs[0].end_time == 1748739600

    def test_parses_snake_case_keys(self, downloader: TapoCloudDownloader) -> None:
        mock_tapo = _make_tapo_mock(segments=[
            {"start_time": 1748736000, "end_time": 1748739600},
        ])
        downloader._tapo = mock_tapo
        segs = downloader.list_segments_for_date("20250601")

        assert len(segs) == 1
        assert segs[0].start_time == 1748736000

    def test_skips_entries_with_missing_times(self, downloader: TapoCloudDownloader) -> None:
        mock_tapo = _make_tapo_mock(segments=[
            {"startTime": 1748736000},  # missing endTime
            {"startTime": 1748740000, "endTime": 1748743600},
        ])
        downloader._tapo = mock_tapo
        segs = downloader.list_segments_for_date("20250601")
        assert len(segs) == 1
        assert segs[0].start_time == 1748740000


# ---------------------------------------------------------------------------
# TapoCloudDownloader – download_segment
# ---------------------------------------------------------------------------


class TestDownloadSegment:
    def test_returns_none_when_download_raises(
        self, downloader: TapoCloudDownloader, tmp_path: Path
    ) -> None:
        mock_tapo = _make_tapo_mock()
        downloader._tapo = mock_tapo

        seg = RecordingSegment(start_time=1748736000, end_time=1748739600)

        with patch(
            "pytapo.media_stream.downloader.Downloader",
            side_effect=Exception("network error"),
        ):
            result = downloader.download_segment(seg)

        assert result is None

    def test_output_dir_created(
        self, downloader: TapoCloudDownloader, tmp_path: Path
    ) -> None:
        """The output directory is created even if download fails."""
        mock_tapo = _make_tapo_mock()
        downloader._tapo = mock_tapo
        seg = RecordingSegment(start_time=1748736000, end_time=1748739600)

        with patch("pytapo.media_stream.downloader.Downloader", side_effect=Exception("err")):
            downloader.download_segment(seg)

        assert downloader.output_dir.exists()

    def test_returns_path_when_file_exists(
        self, downloader: TapoCloudDownloader, tmp_path: Path
    ) -> None:
        """If the downloader produces a file, the path is returned."""
        mock_tapo = _make_tapo_mock()
        downloader._tapo = mock_tapo
        downloader.output_dir.mkdir(parents=True, exist_ok=True)

        seg = RecordingSegment(start_time=1748736000, end_time=1748739600)
        expected_path = downloader.output_dir / f"{seg.start_time}-{seg.end_time}.mp4"
        expected_path.touch()  # simulate Downloader having created the file

        mock_dl = MagicMock()
        mock_dl.downloadFile = MagicMock(return_value=MagicMock())

        async def fake_download(_cb=None):
            return {"fileName": str(expected_path)}

        mock_dl.downloadFile = fake_download

        with patch("pytapo.media_stream.downloader.Downloader", return_value=mock_dl):
            result = downloader.download_segment(seg)

        assert result == str(expected_path)


# ---------------------------------------------------------------------------
# TapoCloudDownloader – download_recordings (integration of list + download)
# ---------------------------------------------------------------------------


class TestDownloadRecordings:
    def test_yields_nothing_when_no_dates(self, downloader: TapoCloudDownloader) -> None:
        mock_tapo = _make_tapo_mock(dates=[])
        downloader._tapo = mock_tapo

        results = list(downloader.download_recordings())
        assert results == []

    def test_skips_failed_downloads(
        self, downloader: TapoCloudDownloader, tmp_path: Path
    ) -> None:
        mock_tapo = _make_tapo_mock(
            dates=["20250601"],
            segments=[{"startTime": 1748736000, "endTime": 1748739600}],
        )
        downloader._tapo = mock_tapo

        with patch.object(downloader, "download_segment", return_value=None):
            results = list(downloader.download_recordings())

        assert results == []

    def test_yields_path_and_segment(
        self, downloader: TapoCloudDownloader, tmp_path: Path
    ) -> None:
        mock_tapo = _make_tapo_mock(
            dates=["20250601"],
            segments=[{"startTime": 1748736000, "endTime": 1748739600}],
        )
        downloader._tapo = mock_tapo
        expected_path = str(tmp_path / "fake.mp4")

        with patch.object(downloader, "download_segment", return_value=expected_path):
            results = list(downloader.download_recordings())

        assert len(results) == 1
        path, seg = results[0]
        assert path == expected_path
        assert seg.start_time == 1748736000

    def test_dates_processed_in_sorted_order(
        self, downloader: TapoCloudDownloader
    ) -> None:
        """Dates should be processed chronologically regardless of API order."""
        mock_tapo = _make_tapo_mock(dates=["20250603", "20250601", "20250602"])
        mock_tapo.getRecordings.return_value = []
        downloader._tapo = mock_tapo

        list(downloader.download_recordings())

        calls = [c[0][0] for c in mock_tapo.getRecordings.call_args_list]
        assert calls == ["20250601", "20250602", "20250603"]

    def test_list_segments_error_skips_date(
        self, downloader: TapoCloudDownloader
    ) -> None:
        mock_tapo = _make_tapo_mock(dates=["20250601", "20250602"])
        mock_tapo.getRecordings.side_effect = [Exception("API error"), []]
        downloader._tapo = mock_tapo

        # Should not raise; bad date is skipped
        results = list(downloader.download_recordings())
        assert results == []
