"""Per-side spatial helpers for the segmentation class map.

The frame is split into left/center/right thirds (Requirement 1.5). Any
remainder columns from `width % 3` are assigned to the center side so the
left and right sides have equal width — that's the rule the reasoner
relies on when comparing walkable ratios.

These helpers run inside the same one-pass `np.unique(class_map)` loop in
`YoloSegmenter._parse_semantic`, so no extra full-frame scan is added.
"""

from __future__ import annotations

import numpy as np

from navigation.models import SIDES, Side


def _side_slices(width: int) -> dict[Side, slice]:
    """Split a frame width into left/center/right thirds.

    Left and right thirds always have equal width. Any remainder columns
    from `width % 3` go to the center side (Requirement 1.5). This is the
    invariant the reasoner relies on when weighing per-side hazards.

    Examples:
        width=300 -> left[0:100], center[100:200], right[200:300]
        width=320 -> left[0:106], center[106:214], right[214:320]
                     (left=106, center=108, right=106)
    """
    if width <= 0:
        return {s: slice(0, 0) for s in SIDES}
    third = width // 3
    rem = width - third * 3  # 0, 1 or 2
    left_end = third
    right_start = third + (third + rem)  # center grows by remainder
    return {
        "left": slice(0, left_end),
        "center": slice(left_end, right_start),
        "right": slice(right_start, width),
    }


def _per_side_class_pixels(
    class_map: np.ndarray,
    id_to_name: dict[int, str],
    weight_map: np.ndarray,
) -> dict[Side, dict[str, float]]:
    """Region-weighted pixel count for every class, per side.

    The weight map (from `_region_weight_map`) is reused so a tall building
    in the upper periphery still contributes near-zero per-side weight,
    just like it does in the global obstacle count.

    Sum of the three side counts equals the global region-weighted count
    for that class (Requirement 1.4 — verified by Property 1).
    """
    h, w = class_map.shape[:2]
    if h == 0 or w == 0:
        return {s: {} for s in SIDES}
    if weight_map.shape[:2] != (h, w):
        raise ValueError(
            f"weight_map shape {weight_map.shape[:2]} != class_map shape {(h, w)}"
        )

    slices = _side_slices(w)
    out: dict[Side, dict[str, float]] = {s: {} for s in SIDES}
    for side in SIDES:
        sl = slices[side]
        cm_side = class_map[:, sl]
        wt_side = weight_map[:, sl]
        if cm_side.size == 0:
            continue
        for cls_id in np.unique(cm_side):
            name = id_to_name.get(int(cls_id), str(int(cls_id)))
            mask = cm_side == cls_id
            out[side][name] = float(wt_side[mask].sum())
    return out


def _per_side_walkable_ratio(
    class_map: np.ndarray,
    id_to_name: dict[int, str],
    walkable_classes: set[str],
) -> dict[Side, float]:
    """Walkable-pixel fraction per side, in [0, 1] (Requirement 2.2).

    Empty sides (zero columns) report 0.0 walkable. Unknown classes are
    treated as not-walkable.
    """
    h, w = class_map.shape[:2]
    if h == 0 or w == 0:
        return {s: 0.0 for s in SIDES}

    slices = _side_slices(w)
    out: dict[Side, float] = {}
    for side in SIDES:
        sl = slices[side]
        cm_side = class_map[:, sl]
        if cm_side.size == 0:
            out[side] = 0.0
            continue
        walkable_pixels = 0
        for cls_id in np.unique(cm_side):
            name = id_to_name.get(int(cls_id), str(int(cls_id)))
            if name in walkable_classes:
                walkable_pixels += int((cm_side == cls_id).sum())
        out[side] = float(walkable_pixels) / float(cm_side.size)
    return out


def empty_per_side_counts() -> dict[Side, dict[str, float]]:
    """Zeroed per-side dict for mocks (Requirement 1.6)."""
    return {s: {} for s in SIDES}


def empty_per_side_walkable() -> dict[Side, float]:
    """Zeroed per-side walkable dict for mocks (Requirement 1.6)."""
    return {s: 0.0 for s in SIDES}


__all__ = [
    "_side_slices",
    "_per_side_class_pixels",
    "_per_side_walkable_ratio",
    "empty_per_side_counts",
    "empty_per_side_walkable",
]
