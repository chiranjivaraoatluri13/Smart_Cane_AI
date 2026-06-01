"""HUD overlay tests."""

import numpy as np

from navigation.output.hud import (
    build_live_detail_text,
    draw_navigation_hud,
    draw_sparse_navigation_hud,
    format_command_banner,
)


def test_draw_navigation_hud():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = draw_navigation_hud(
        frame,
        phrase="Go forward",
        command="go_forward",
        speak=True,
        confidence=0.9,
        rationale="Clear path",
    )
    assert out.shape == frame.shape


def test_draw_sparse_navigation_hud():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_sparse_navigation_hud(
        frame,
        phrase="Clear path ahead",
        visible=True,
        rationale="on route (12 m remaining)",
        facts={"route_cue": {"turn": "forward", "meters_to_turn": 12.0}},
    )
    assert out.shape == frame.shape
    assert out.sum() > 0


def test_build_live_detail_text_route():
    details, route = build_live_detail_text(
        rationale="",
        facts={"route_cue": {"turn": "left", "meters_to_turn": 10.0}},
    )
    assert "Turn left" in details
    assert route == details


def test_format_command_banner():
    line = format_command_banner(
        {"phrase": "Stop", "command": "stop", "speak": True, "confidence": 0.85}
    )
    assert "STOP" in line
