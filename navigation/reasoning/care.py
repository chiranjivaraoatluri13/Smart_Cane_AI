"""CARE safety-aware navigation reasoning."""

from __future__ import annotations

import json
from typing import Any

import httpx
import numpy as np

from navigation.config import Settings
from navigation.models import CareResult, DepthResult, SegmentationResult


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
        shape = segmentation.metadata.get("shape", [480, 640])
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            frame_area = int(shape[0]) * int(shape[1])
        else:
            frame_area = 640 * 480

        # Region-weighted count is preferred: a person 1m dead-center weighs
        # full, a building filling the upper periphery weighs ~0. Falls back
        # to the raw obstacle count when no weighted count is available.
        weighted = segmentation.obstacle_pixels_weighted
        obstacle_score = weighted if weighted > 0 else float(segmentation.obstacle_pixels)

        # Effective area: weighted counts live in [0, ~half the frame] because
        # weights average <1, so we compare them against the same area * ratio.
        # We use a slightly tighter ratio for weighted counts so a small object
        # in the path still trips, while a class that only fills the periphery
        # does not.
        ratio = self.settings.hazard_obstacle_ratio
        if weighted > 0:
            # For weighted obstacles, use a higher threshold (2x) to reduce false positives
            # on plain footpaths where the model might detect minor artifacts
            min_obstacle = frame_area * ratio * 1.0  # Changed from 0.5 to 1.0
        else:
            min_obstacle = frame_area * ratio
        hazard = obstacle_score >= max(min_obstacle, 1.0)

        score = 0.9 if not hazard else 0.4
        direction = 0.0
        if depth.obstacle_depth_m is not None and depth.obstacle_depth_m < 1.2:
            hazard = True
            score = 0.2
            direction = -15.0 if segmentation.walkable_ratio > 0.5 else 15.0
        return CareResult(
            safe_direction_deg=direction,
            safety_score=score,
            hazard_detected=hazard,
            raw={
                "mode": "heuristic",
                "obstacle_pixels": int(segmentation.obstacle_pixels),
                "obstacle_weighted": float(weighted),
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
