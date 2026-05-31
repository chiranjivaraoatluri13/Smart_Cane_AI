"""Tests for CARE heuristic (center-lane gating)."""

from __future__ import annotations

from navigation.config import Settings
from navigation.models import DepthResult, SegmentationResult
from navigation.reasoning.care import CareNavigator


def _seg(**kwargs) -> SegmentationResult:
    defaults = {
        "walkable_ratio": 0.5,
        "metadata": {"shape": [48, 64]},
        "per_side_walkable_ratio": {"left": 0.2, "center": 0.6, "right": 0.2},
        "per_side_class_pixels": {"left": {}, "center": {}, "right": {}},
    }
    defaults.update(kwargs)
    return SegmentationResult(**defaults)


def test_care_no_hazard_on_clear_center():
    care = CareNavigator(Settings(min_center_walkable_for_forward=0.18))
    seg = _seg(
        per_side_class_pixels={
            "left": {"plant": 900.0},
            "center": {"sidewalk": 500.0},
            "right": {},
        },
        obstacle_pixels_weighted=900.0,
    )
    out = care._heuristic(seg, DepthResult(obstacle_depth_m=3.0))
    assert out.hazard_detected is False
    assert out.safety_score >= 0.75
