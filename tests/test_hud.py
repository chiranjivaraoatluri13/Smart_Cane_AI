"""HUD overlay tests."""

import numpy as np

from navigation.output.hud import draw_navigation_hud, format_command_banner


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


def test_format_command_banner():
    line = format_command_banner(
        {"phrase": "Stop", "command": "stop", "speak": True, "confidence": 0.85}
    )
    assert "STOP" in line
