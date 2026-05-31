"""Depth estimation adapter.

Produces an approximate obstacle distance from one of two real sources:
a client-measured value (on-device Depth Anything V2, posted as ``depth_m``)
or a geometric proxy derived from the segmentation class map. No synthetic
brightness fallback — if neither source is available, depth is undefined.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from navigation.config import Settings
from navigation.models import DepthResult, SegmentationResult

logger = logging.getLogger(__name__)


class DepthEstimator:
    """Derives obstacle depth from client depth or the segmentation proxy."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any = None

    def predict(
        self,
        frame: np.ndarray,
        *,
        segmentation: SegmentationResult | None = None,
        external_depth_m: float | None = None,
    ) -> DepthResult:
        # Depth sources, in priority order:
        # 1. Real depth measured on the client (Depth Anything V2 on the phone),
        #    posted as ``depth_m`` — wins when present.
        # 2. Geometric proxy derived from the segmentation class map (real
        #    computation from the live segmenter, not fabricated).
        if external_depth_m is not None:
            return self._from_external(external_depth_m)
        proxy = self._proxy_from_segmentation(frame, segmentation)
        if proxy is not None:
            return proxy
        # No segmentation class map available — depth cannot be derived.
        raise ValueError(
            "Depth requires a segmentation class map or a client external_depth_m; "
            "neither was provided."
        )

    @staticmethod
    def _from_external(depth_m: float) -> DepthResult:
        """Build a DepthResult from a client-measured obstacle distance.

        The phone computes an approximate nearest-obstacle distance from an
        on-device depth network and posts it as ``depth_m``. We clamp it to a
        sane walking range and treat it as both the center and obstacle depth.
        """
        try:
            d = float(depth_m)
        except (TypeError, ValueError):
            d = 0.0
        # NaN is unknown -> treat as "close" (safest for a walking aid).
        # Infinities clamp naturally below (+inf -> far, -inf -> close).
        if np.isnan(d):
            d = 0.0
        d = max(0.3, min(15.0, d))
        return DepthResult(
            depth_map=None,
            min_depth_m=max(0.3, d - 0.5),
            center_depth_m=d,
            obstacle_depth_m=d,
            metadata={"mock": False, "source": "client_depth_anything"},
        )

    def _proxy_from_segmentation(
        self,
        frame: np.ndarray,
        segmentation: SegmentationResult | None,
    ) -> DepthResult | None:
        """Derive a coarse depth from the segmentation class map.

        Two-stage strategy, both O(H × W) and folded into the existing
        per-frame scan:

        1. **Vertical position (preferred when an obstacle is present)**.
           In a perspective camera, the *base* of an object sits lower in
           the frame the closer it is. We find the lowest row containing
           any obstacle-class pixel (within the center 60% of width — the
           walking path) and convert that row's normalized y-position to
           an approximate depth via an inverse-perspective curve. This is
           monotonic in real distance: closer objects always produce a
           smaller `obstacle_depth_m`. It's not metric — for that we'd
           need camera intrinsics and one calibration sample — but the
           bucket order (immediate → near → mid → far) tracks reality.

        2. **Walkable/obstacle ratio (fallback when no in-path obstacle)**.
           If the bottom-center band is mostly walkable, depth is "far"
           (clear path). If mostly obstacle pixels but no clear bottom
           edge, depth is "near" (something is blocking the lane). This
           is the original three-bucket logic, kept as the safety net.

        When real metric depth lands later (UniDepthV2 / Depth Anything),
        only this method is replaced. Phrase templates and the bucketizer
        remain untouched.
        """
        if segmentation is None or segmentation.class_map is None:
            return None

        cm = np.asarray(segmentation.class_map)
        if cm.ndim == 3:
            cm = cm[0]
        if cm.size == 0:
            return None

        h, w = cm.shape[:2]

        id_to_name = (segmentation.metadata or {}).get("id_to_name") or {}
        id_to_name = {int(k): str(v) for k, v in id_to_name.items()}

        try:
            seg_cfg = self.settings.seg_class_config()
        except Exception:
            seg_cfg = {}
        seg_cfg = seg_cfg if isinstance(seg_cfg, dict) else {}
        walkable_set = set(seg_cfg.get("walkable_classes", []))
        obstacle_set = set(seg_cfg.get("obstacle_classes", []))

        # ----- Stage 1: vertical position of the closest obstacle ---------
        obstacle_ids = [
            cid for cid, name in id_to_name.items() if name in obstacle_set
        ]
        if obstacle_ids:
            obstacle_mask = np.isin(cm, obstacle_ids)
            # Restrict to the center 60% of width — peripheral objects on
            # the far edges of the frame should not drive the in-path
            # distance phrase.
            x0 = int(w * 0.20)
            x1 = int(w * 0.80)
            if x1 > x0:
                center_band = obstacle_mask[:, x0:x1]
                rows_with_obstacle = np.where(center_band.any(axis=1))[0]
                if rows_with_obstacle.size > 0:
                    lowest_row = int(rows_with_obstacle.max())
                    y_frac = lowest_row / max(h - 1, 1)  # 0 (top) → 1 (bottom)
                    depth_m = _row_to_depth_m(y_frac)
                    return DepthResult(
                        depth_map=None,
                        min_depth_m=max(0.5, depth_m - 0.5),
                        center_depth_m=depth_m,
                        obstacle_depth_m=depth_m,
                        metadata={
                            "mock": True,
                            "source": "vertical_position",
                            "lowest_obstacle_row_frac": float(y_frac),
                        },
                    )

        # ----- Stage 2: walkable/obstacle ratio fallback ------------------
        y0 = int(h * 0.6)
        x0, x1 = int(w * 0.3), int(w * 0.7)
        patch = cm[y0:, x0:x1]
        if patch.size == 0:
            return None

        total = patch.size
        walkable_count = 0
        obstacle_count = 0
        for cls_id in np.unique(patch):
            name = id_to_name.get(int(cls_id), str(int(cls_id)))
            count = int((patch == cls_id).sum())
            if name in walkable_set:
                walkable_count += count
            elif name in obstacle_set:
                obstacle_count += count

        walkable_frac = walkable_count / total
        obstacle_frac = obstacle_count / total

        if obstacle_frac >= 0.4:
            depth = 1.0  # near — something is in your way
        elif walkable_frac >= 0.4:
            depth = 3.0  # far — clear path
        else:
            depth = 2.0  # mid — mixed scene

        return DepthResult(
            depth_map=None,
            min_depth_m=max(0.5, depth - 0.5),
            center_depth_m=depth,
            obstacle_depth_m=depth,
            metadata={
                "mock": True,
                "source": "segmentation_proxy",
                "walkable_frac": float(walkable_frac),
                "obstacle_frac": float(obstacle_frac),
            },
        )


def _row_to_depth_m(y_frac: float) -> float:
    """Map normalized vertical position to an approximate depth in meters.

    Inverse-perspective approximation. Assumes a chest-height forward
    camera with the horizon near `y_frac = 0.5`. Below the horizon, depth
    scales as `~k / (y_frac - 0.5)` — closer to the bottom edge → smaller
    depth. Above the horizon, depth saturates at "far" (5.0 m).

    Calibrated so the bucket boundaries land where the user expects:
      y_frac = 0.85  →  ~1.1 m  (immediate, "right in front of you")
      y_frac = 0.70  →  ~2.0 m  (near, "about 6 feet ahead")
      y_frac = 0.65  →  ~2.7 m  (mid, "about 10 feet ahead")
      y_frac = 0.55  →  ~8 m    (far, "about 30 feet ahead")
    """
    horizon = 0.5
    if y_frac <= horizon:
        return 5.0
    below = y_frac - horizon  # 0 (at horizon) → 0.5 (at frame bottom)
    return max(0.5, min(10.0, 0.4 / max(below, 0.04)))


# Backwards-compatible alias: the class was historically named
# ``UniDepthEstimator``. Depth no longer depends on UniDepth, but the alias
# keeps existing imports working.
UniDepthEstimator = DepthEstimator


__all__ = ["DepthEstimator", "UniDepthEstimator"]
