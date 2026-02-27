"""Tests for src.camera – credential injection and TapoCamera construction."""

from __future__ import annotations

import pytest

from src.camera import TapoCamera, _inject_credentials


# ------------------------------------------------------------------
# _inject_credentials helper
# ------------------------------------------------------------------

class TestInjectCredentials:
    def test_inserts_creds_into_plain_url(self):
        result = _inject_credentials(
            "rtsp://192.168.1.100:554/stream1", "admin", "secret"
        )
        assert result == "rtsp://admin:secret@192.168.1.100:554/stream1"

    def test_preserves_existing_credentials(self):
        url = "rtsp://user:pass@192.168.1.100:554/stream1"
        assert _inject_credentials(url, "other", "other") == url

    def test_url_without_port(self):
        result = _inject_credentials(
            "rtsp://10.0.0.1/live", "cam", "pw"
        )
        assert result == "rtsp://cam:pw@10.0.0.1/live"

    def test_url_with_path_and_query(self):
        result = _inject_credentials(
            "rtsp://10.0.0.1:554/stream?channel=1", "admin", "pass"
        )
        assert result == "rtsp://admin:pass@10.0.0.1:554/stream?channel=1"


# ------------------------------------------------------------------
# TapoCamera.__init__ credential handling
# ------------------------------------------------------------------

class TestTapoCameraInit:
    def test_url_unchanged_when_no_creds_supplied(self):
        url = "rtsp://192.168.1.100:554/stream1"
        cam = TapoCamera(stream_url=url)
        assert cam.stream_url == url

    def test_url_unchanged_when_creds_already_embedded(self):
        url = "rtsp://admin:pass@192.168.1.100:554/stream1"
        cam = TapoCamera(stream_url=url, username="other", password="other")
        assert cam.stream_url == url

    def test_creds_injected_when_url_has_none(self):
        cam = TapoCamera(
            stream_url="rtsp://192.168.1.100:554/stream1",
            username="admin",
            password="secret",
        )
        assert cam.stream_url == "rtsp://admin:secret@192.168.1.100:554/stream1"

    def test_no_injection_when_only_username(self):
        url = "rtsp://192.168.1.100:554/stream1"
        cam = TapoCamera(stream_url=url, username="admin")
        assert cam.stream_url == url

    def test_no_injection_when_only_password(self):
        url = "rtsp://192.168.1.100:554/stream1"
        cam = TapoCamera(stream_url=url, password="secret")
        assert cam.stream_url == url

    def test_no_injection_when_username_empty_string(self):
        url = "rtsp://192.168.1.100:554/stream1"
        cam = TapoCamera(stream_url=url, username="", password="secret")
        assert cam.stream_url == url
