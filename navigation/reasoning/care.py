"""CARE safety-aware navigation reasoning."""

from __future__ import annotations

import json
from typing import Any

import httpx
import numpy as np

from navigation.config import Settings
from navigation.models import CareResult, DepthResult, SegmentationResult
from navigation.reasoning.mask_metrics import (
    center_band_pixel_area,
    center_obstacle_weighted,
    walkable_by_side,
)


class CareNavigator:
    """Calls CARE endpoint or uses heuristic fallback."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def predict(
        self,
        frame: np.ndarray,
        segmentation: SegmentationResult,
        depth: DepthResult,
    ) -> CareResult:
        if not self.settings.use_care_http:
            return self._heuristic(segmentation, depth)
        return self._call_remote(frame, segmentation, depth)

    def _heuristic(
        self, segmentation: SegmentationResult, depth: DepthResult
    ) -> CareResult:
        cfg = self.settings.seg_class_config()
        obstacle_set = set(cfg.get("obstacle_classes", []))
        min_center_walk = float(
            getattr(self.settings, "min_center_walkable_for_forward", 0.18)
        )
        walkable = walkable_by_side(segmentation)
        center_walk = walkable.get("center", 0.0)
        path_clear = (
            center_walk >= min_center_walk
            or float(segmentation.walkable_ratio) >= min_center_walk
        )

        # Center-lane obstacles only — plants/cars on the sides must not STOP.
        weighted = center_obstacle_weighted(segmentation, obstacle_set)
        if weighted <= 0:
            weighted = float(segmentation.obstacle_pixels_weighted)
        center_area = center_band_pixel_area(segmentation)
        ratio = self.settings.hazard_obstacle_ratio
        min_obstacle = center_area * ratio
        hazard = (not path_clear) and weighted >= max(min_obstacle, 1.0)

        score = 0.9 if not hazard else 0.4
        direction = 0.0
        if (
            depth.obstacle_depth_m is not None
            and depth.obstacle_depth_m < 1.2
            and not path_clear
        ):
            hazard = True
            score = 0.2
            direction = -15.0 if center_walk >= walkable.get("left", 0.0) else 15.0
        return CareResult(
            safe_direction_deg=direction,
            safety_score=score,
            hazard_detected=hazard,
            raw={
                "mode": "heuristic",
                "obstacle_pixels": int(segmentation.obstacle_pixels),
                "obstacle_weighted": float(weighted),
                "center_walkable": float(center_walk),
                "path_clear": path_clear,
                "min_obstacle_threshold": float(min_obstacle),
            },
        )

    def _call_remote(
        self,
        frame: np.ndarray,
        segmentation: SegmentationResult,
        depth: DepthResult,
    ) -> CareResult:
        payload = {
            "obstacle_pixels": segmentation.obstacle_pixels,
            "walkable_ratio": segmentation.walkable_ratio,
            "center_depth_m": depth.center_depth_m,
            "shape": list(frame.shape),
        }
        try:
            with httpx.Client(timeout=self.settings.care_timeout_sec) as client:
                resp = client.post(
                    self.settings.care_endpoint,
                    json=payload,
                )
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            return self._heuristic(segmentation, depth)

        return CareResult(
            safe_direction_deg=data.get("safe_direction_deg"),
            safety_score=float(data.get("safety_score", 0.5)),
            hazard_detected=bool(data.get("hazard_detected", False)),
            raw=data,
        )
