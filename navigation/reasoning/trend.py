"""Per-category centroid trend tracker (Requirement 4).

Reads `SegmentationResult.per_side_class_pixels`, maps each class to a
spoken category via `CLASS_TO_CATEGORY`, and labels each category's
recent motion as one of: static, crossing_left_to_right,
crossing_right_to_left, closing_in, receding.

The work is `O(buffer_size × tracked_categories)` per frame — bounded by
constants. `AlertTracker` continues to use its own growth detector for
proximity alerts; this tracker is a separate consumer of the same
per-side weighted counts so no behavior in alerts.py changes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from navigation.config import Settings
from navigation.models import SIDES, Side
from navigation.reasoning.hazard_labels import hazard_key_for_class
from navigation.reasoning.facts import ApproachDirection

# Normalized horizontal centers per side. Used to compute a centroid x
# coordinate from per-side weights, so a centroid swing from "all weight
# on left" to "all weight on right" travels from 0.17 to 0.83.
_SIDE_X: dict[Side, float] = {"left": 0.17, "center": 0.50, "right": 0.83}


@dataclass
class _CategoryHistory:
    samples: deque = field(default_factory=lambda: deque(maxlen=6))


def per_side_to_per_category(
    per_side_class_pixels: dict[Side, dict[str, float]] | None,
    *,
    obstacle_classes: set[str] | None = None,
    hazard_classes: set[str] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Aggregate `class -> weight` dicts (per side) into `category -> (l,c,r)`.

    Classes that are not configured obstacles/hazards are dropped.
    """
    out: dict[str, tuple[float, float, float]] = {}
    if not per_side_class_pixels:
        return out

    obstacle = obstacle_classes or set()
    hazard = hazard_classes or set()

    for side in SIDES:
        side_dict = per_side_class_pixels.get(side, {}) or {}
        for cls_name, weight in side_dict.items():
            cat = hazard_key_for_class(cls_name, obstacle, hazard)
            if cat is None:
                continue
            triple = list(out.get(cat, (0.0, 0.0, 0.0)))
            idx = SIDES.index(side)
            triple[idx] += float(weight)
            out[cat] = (triple[0], triple[1], triple[2])
    return out


def centroid_x_norm(left: float, center: float, right: float) -> float:
    """Normalized centroid x of a per-side weight triple."""
    total = left + center + right
    if total <= 0:
        return _SIDE_X["center"]
    return (
        _SIDE_X["left"] * left
        + _SIDE_X["center"] * center
        + _SIDE_X["right"] * right
    ) / total


class TrendTracker:
    """Per-category centroid + total-weight ring buffer (size 6)."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings
        self.buffer_size = 6
        self.cross_threshold = 0.15
        self.static_threshold = 0.05
        self.growth_factor = 1.3
        self.recede_factor = 0.5
        self._history: dict[str, _CategoryHistory] = {}

    def update(
        self,
        per_side_class_pixels: dict[Side, dict[str, float]] | None,
    ) -> None:
        """Push the current frame's per-side counts into each category's buffer.

        Categories not seen this frame still get a sample (with zeros) so
        their trends can decay rather than being frozen at the last seen
        value. This prevents "ghost" classifications from objects that
        have left the frame.
        """
        obstacle: set[str] = set()
        hazard: set[str] = set()
        if self.settings is not None:
            cfg = self.settings.seg_class_config() or {}
            obstacle = set(cfg.get("obstacle_classes", []))
            hazard = set(cfg.get("hazard_classes", []))
        per_category = per_side_to_per_category(
            per_side_class_pixels,
            obstacle_classes=obstacle,
            hazard_classes=hazard,
        )
        # Push known categories with the new sample; push zeros for any
        # category we've previously seen but isn't in this frame.
        seen = set(per_category.keys())
        for cat in seen | set(self._history.keys()):
            triple = per_category.get(cat, (0.0, 0.0, 0.0))
            cx = centroid_x_norm(*triple)
            total = triple[0] + triple[1] + triple[2]
            self._history.setdefault(cat, _CategoryHistory()).samples.append(
                (cx, total)
            )

    def classify(self, category: str) -> ApproachDirection:
        """Label a single category's recent motion."""
        hist = self._history.get(category)
        if hist is None or len(hist.samples) < 3:
            return "static"
        centroids = [s[0] for s in hist.samples]
        weights = [s[1] for s in hist.samples]
        first_w = next((w for w in weights if w > 0), 0.0)
        last_w = weights[-1]

        # closing_in / receding short-circuit: weight trend dominates the
        # centroid trend when motion is mostly toward/away from the camera.
        if first_w > 0 and last_w > 0:
            if last_w >= first_w * self.growth_factor:
                return "closing_in"
            if last_w <= first_w * self.recede_factor:
                return "receding"

        dx_total = centroids[-1] - centroids[0]
        if abs(dx_total) < self.static_threshold:
            return "static"

        deltas = [centroids[i + 1] - centroids[i] for i in range(len(centroids) - 1)]
        # Allow tiny non-monotone bumps (1e-3) due to fp rounding.
        if dx_total >= self.cross_threshold and all(d >= -1e-3 for d in deltas):
            return "crossing_left_to_right"
        if dx_total <= -self.cross_threshold and all(d <= 1e-3 for d in deltas):
            return "crossing_right_to_left"
        return "static"

    def classify_all(self) -> dict[str, ApproachDirection]:
        """Label every category currently being tracked."""
        return {cat: self.classify(cat) for cat in self._history}

    def reset(self) -> None:
        self._history.clear()


__all__ = [
    "TrendTracker",
    "centroid_x_norm",
    "per_side_to_per_category",
]
