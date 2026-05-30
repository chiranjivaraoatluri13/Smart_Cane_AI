"""UniDepthV2 monocular depth adapter."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from navigation.config import Settings
from navigation.models import DepthResult, SegmentationResult

logger = logging.getLogger(__name__)


class UniDepthEstimator:
    """UniDepthV2 integration point; mock depth until weights are wired."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any = None
        self._live_warned = False

    def predict(
        self,
        frame: np.ndarray,
        *,
        dry_run: bool = False,
        segmentation: SegmentationResult | None = None,
    ) -> DepthResult:
        if dry_run or not self.settings.unidepth_model_path:
            return self._mock(frame, segmentation=segmentation)
        return self._predict_live(frame, segmentation=segmentation)

    def _predict_live(
        self,
        frame: np.ndarray,
        *,
        segmentation: SegmentationResult | None = None,
    ) -> DepthResult:
        # UniDepthV2 isn't wired yet. Instead of crashing the pipeline when
        # UNIDEPTH_MODEL_PATH is set, warn once and fall back to the mock so
        # the rest of the system keeps running.
        if not self._live_warned:
            logger.warning(
                "UNIDEPTH_MODEL_PATH=%r is set but live UniDepthV2 inference is "
                "not wired in navigation/perception/depth.py. Falling back to "
                "synthetic depth (mock).",
                self.settings.unidepth_model_path,
            )
            self._live_warned = True
        return self._mock(frame, segmentation=segmentation)

    def _mock(
        self,
        frame: np.ndarray,
        *,
        segmentation: SegmentationResult | None = None,
    ) -> DepthResult:
        # Prefer a segmentation-derived proxy when available. This is a far
        # better signal than image brightness because it correlates with the
        # actual safety question: "is the path ahead walkable or blocked?"
        proxy = self._proxy_from_segmentation(frame, segmentation)
        if proxy is not None:
            return proxy

        # Fallback: synthetic depth from the brightness of the central patch.
        # Used only when no segmentation result is provided (e.g. unit tests).
        h, w = frame.shape[:2]
        gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame
        center = gray[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
        center_depth = float(3.0 - (center.mean() / 255.0) * 2.0)
        return DepthResult(
            depth_map=None,
            min_depth_m=max(0.5, center_depth - 0.5),
            center_depth_m=center_depth,
            obstacle_depth_m=center_depth,
            metadata={"mock": True, "source": "brightness"},
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
            from navigation.config import load_yaml_config

            cfg = load_yaml_config(self.settings.config_path)
        except Exception:
            cfg = {}
        seg_cfg = cfg.get("segmentation", {}) if isinstance(cfg, dict) else {}
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
