"""Tests for phone_server.py — /set_destination + /process_frame contract."""

from __future__ import annotations

import io
from unittest.mock import patch

import cv2
import numpy as np
import pytest

# Importing phone_server triggers the heavy module-level model loading. We
# can't avoid it; the test runs once per session.
import phone_server  # noqa: E402


@pytest.fixture
def client():
    phone_server.app.config["TESTING"] = True
    with phone_server.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# /set_destination
# ---------------------------------------------------------------------------


def test_set_destination_geocodes(client):
    """Happy path: address resolves, server stores destination, returns 200."""
    with patch.object(phone_server, "geocode_address", return_value=(40.7484, -73.9857)):
        rv = client.post("/set_destination", data={"address": "Empire State Building"})
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["ok"] is True
    assert data["lat"] == pytest.approx(40.7484)
    assert data["lon"] == pytest.approx(-73.9857)
    # And the interpreter's settings now carry the destination.
    assert phone_server.interpreter.settings.dest_lat == pytest.approx(40.7484)
    assert phone_server.interpreter.settings.dest_lon == pytest.approx(-73.9857)
    assert phone_server.interpreter.settings.use_map_guidance is True


def test_set_destination_400_on_missing(client):
    rv = client.post("/set_destination", data={})
    assert rv.status_code == 400
    assert rv.get_json()["error"] == "missing_address"


def test_set_destination_400_on_empty(client):
    rv = client.post("/set_destination", data={"address": "   "})
    assert rv.status_code == 400
    assert rv.get_json()["error"] == "missing_address"


def test_set_destination_422_on_no_geocode(client):
    with patch.object(phone_server, "geocode_address", side_effect=ValueError("no result")):
        rv = client.post("/set_destination", data={"address": "asdkjfasdkfjasdf"})
    assert rv.status_code == 422
    assert rv.get_json()["error"] == "address_not_found"


def test_set_destination_without_gps_defers_route_fetch(client):
    """When no /process_frame has carried GPS, /set_destination still succeeds."""
    with patch.object(phone_server, "geocode_address", return_value=(40.0, -74.0)):
        rv = client.post("/set_destination", data={"address": "anywhere"})
    assert rv.status_code == 200
    # MapGuidance instance is invalidated; next /process_frame with GPS will fetch.
    assert phone_server.interpreter._map_guidance is None
    assert phone_server.interpreter._map_route_attempted is False


def test_set_destination_posts_address_form_field(client):
    """Verify the form field name is exactly 'address' (case-sensitive)."""
    with patch.object(phone_server, "geocode_address", return_value=(0.0, 0.0)):
        rv = client.post("/set_destination", data={"Address": "wrong field name"})
    assert rv.status_code == 400


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_endpoint(client):
    rv = client.get("/health")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True


# ---------------------------------------------------------------------------
# /process_frame
# ---------------------------------------------------------------------------


def _jpeg_bytes(h: int = 240, w: int = 320) -> bytes:
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def test_process_frame_returns_json_with_command(client):
    data = {
        "frame": (io.BytesIO(_jpeg_bytes()), "frame.jpg"),
    }
    rv = client.post("/process_frame", data=data, content_type="multipart/form-data")
    assert rv.status_code == 200
    body = rv.get_json()
    assert "command" in body
    assert "phrase" in body


def test_process_frame_400_when_missing_frame(client):
    rv = client.post("/process_frame", data={}, content_type="multipart/form-data")
    assert rv.status_code == 400


def test_process_frame_without_depth_uses_proxy(client):
    data = {"frame": (io.BytesIO(_jpeg_bytes()), "frame.jpg")}
    rv = client.post("/process_frame", data=data, content_type="multipart/form-data")
    assert rv.status_code == 200
    assert rv.get_json()["depth_source"] == "proxy"


def test_process_frame_with_depth_m_uses_client(client):
    """A posted depth_m value flips the depth source to the on-device path."""
    data = {
        "frame": (io.BytesIO(_jpeg_bytes()), "frame.jpg"),
        "depth_m": "1.8",
    }
    rv = client.post("/process_frame", data=data, content_type="multipart/form-data")
    assert rv.status_code == 200
    assert rv.get_json()["depth_source"] == "client"


def test_process_frame_ignores_blank_depth_m(client):
    data = {
        "frame": (io.BytesIO(_jpeg_bytes()), "frame.jpg"),
        "depth_m": "",
    }
    rv = client.post("/process_frame", data=data, content_type="multipart/form-data")
    assert rv.status_code == 200
    assert rv.get_json()["depth_source"] == "proxy"
