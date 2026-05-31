"""Tests for mask-resolution obstacle ratio helpers."""

from __future__ import annotations

from navigation.reasoning.mask_metrics import (
    center_band_pixel_area,
    center_obstacle_ratio,
)


def test_center_ratio_uses_mask_area_not_full_frame():
    """Ratio must use center-third area at mask resolution (48×64), not 640×480."""
    from navigation.models import SegmentationResult

    seg = SegmentationResult(
        per_side_class_pixels={
            "left": {},
            "center": {"person": 50.0},
            "right": {},
        },
        per_side_walkable_ratio={"left": 0.0, "center": 0.1, "right": 0.0},
        metadata={"shape": [48, 64]},
    )
    center_area = center_band_pixel_area(seg)
    assert center_area == 48 * (64 // 3 + (64 % 3))
    ratio = center_obstacle_ratio(seg, {"person"})
    assert ratio == 50.0 / center_area
    # Would be ~50 / (640*480/3) if full-frame area were used — far too small.
    assert ratio > 0.04
