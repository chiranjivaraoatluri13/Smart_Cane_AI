"""Mask-resolution helpers for vision_stop and CARE thresholds.

Segmentation stats are computed on the class map at inference resolution
(e.g. 48×64), with ``metadata["shape"]`` matching that map. Thresholds must
use the center third's pixel area at the same resolution, not the camera
frame size, or obstacle ratios are inflated and clear paths read as blocked.
"""

from __future__ import annotations

from navigation.models import SIDES, SegmentationResult, Side


def analysis_shape(seg: SegmentationResult) -> tuple[int, int]:
    shape = (seg.metadata or {}).get("shape", [480, 640])
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        return 480, 640
    return int(shape[0]), int(shape[1])


def center_band_pixel_area(seg: SegmentationResult) -> float:
    """Pixel count of the center vertical third at mask resolution."""
    h, w = analysis_shape(seg)
    if h <= 0 or w <= 0:
        return 1.0
    third = w // 3
    rem = w - third * 3
    center_w = third + rem
    return max(float(h) * float(center_w), 1.0)


def center_obstacle_weighted(
    seg: SegmentationResult, obstacle_classes: set[str]
) -> float:
    per_side = seg.per_side_class_pixels or {}
    if not isinstance(per_side, dict):
        return 0.0
    center = per_side.get("center", {}) or {}
    return sum(
        float(w) for cls, w in center.items() if cls in obstacle_classes
    )


def center_obstacle_ratio(
    seg: SegmentationResult, obstacle_classes: set[str]
) -> float:
    """Region-weighted obstacle mass in the center third / center band area."""
    return center_obstacle_weighted(seg, obstacle_classes) / center_band_pixel_area(
        seg
    )


def walkable_by_side(seg: SegmentationResult) -> dict[Side, float]:
    per = seg.per_side_walkable_ratio
    if per:
        return dict(per)
    g = float(seg.walkable_ratio)
    return {"left": g, "center": g, "right": g}


__all__ = [
    "analysis_shape",
    "center_band_pixel_area",
    "center_obstacle_ratio",
    "center_obstacle_weighted",
    "walkable_by_side",
]
