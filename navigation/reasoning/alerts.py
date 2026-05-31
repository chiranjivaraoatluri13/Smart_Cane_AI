"""Per-class proximity alerts ("Person approaching", "Car approaching", ...).

This sits *alongside* the navigation command (stop / go_forward / ...). The
command tells the user what to **do**; the alert tells what is **there**.

IMPORTANT — false-positive guardrails:
The default segmentation model (yolo26n-sem.pt) is trained on Cityscapes
*street* scenes. Pointed indoors, at a wall, or at an empty path, it
hallucinates street classes (car / bicycle / truck / person) in noise.
Without guardrails the tracker then announces every category it knows,
one after another. To prevent that, alerts require:

  1. A large, sustained presence (high ``min_weighted_pixels``), not a
     transient blob.
  2. Real growth across the buffer (the object is genuinely approaching),
     not a one-frame pop-in.
  3. Scene coherence — if too many distinct categories are "present" in
     one frame, that's the signature of a hallucinating model, so we stay
     silent that frame.
  4. A global cooldown — at most one alert every ``global_cooldown_sec``
     regardless of category, so even noisy detection can't list everything
     rapid-fire.

Alerts can be disabled entirely with ``ALERTS_ENABLED=false`` in .env.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from navigation.models import SegmentationResult


# Spoken category → priority (higher number wins when several fire at once).
CATEGORY_PRIORITY: dict[str, int] = {
    "heavy_vehicle": 100,
    "car": 80,
    "cyclist": 60,
    "scooter": 55,
    "bicycle": 50,
    "person": 40,
    "stairs": 90,
}

CLASS_TO_CATEGORY: dict[str, str] = {
    "person": "person",
    "rider": "cyclist",
    "bicycle": "bicycle",
    "motorcycle": "scooter",
    "car": "car",
    "truck": "heavy_vehicle",
    "bus": "heavy_vehicle",
    "train": "heavy_vehicle",
}

CATEGORY_PHRASE: dict[str, str] = {
    "heavy_vehicle": "Heavy vehicle approaching",
    "car": "Car approaching",
    "cyclist": "Cyclist approaching",
    "scooter": "Scooter approaching",
    "bicycle": "Bicycle approaching",
    "person": "Person approaching",
    "stairs": "Steps ahead",
}


@dataclass
class ProximityAlert:
    """A single per-class alert worth speaking on top of the command."""

    category: str
    phrase: str
    confidence: float
    priority: int
    rationale: str = ""

    def __lt__(self, other: "ProximityAlert") -> bool:
        return self.priority < other.priority


@dataclass
class _ClassHistory:
    """Tracks a single category's weighted pixel count over recent frames."""

    counts: deque = field(default_factory=lambda: deque(maxlen=6))

    def push(self, weighted_count: float) -> None:
        self.counts.append(float(weighted_count))

    def is_growing(self, growth_factor: float, min_recent: float) -> bool:
        """True only when the class is large AND consistently getting larger.

        Both conditions are required, so a class that merely "pops in" with a
        big count (the classic hallucination pattern) does NOT trigger — it
        has to actually grow across the buffer, which a real approaching
        object does and a flicker does not.
        """
        if len(self.counts) < 4:
            return False
        recent = self.counts[-1]
        if recent < min_recent:
            return False
        # Require the object to be present (above half-threshold) for most of
        # the buffer — a real approach is sustained, noise is not.
        present_frames = sum(1 for c in self.counts if c > min_recent * 0.5)
        if present_frames < 3:
            return False
        earlier = next((c for c in self.counts if c > min_recent * 0.5), None)
        if earlier is None or earlier <= 0:
            return False
        return recent >= earlier * growth_factor


class AlertTracker:
    """Per-frame state for proximity alerts, with false-positive guardrails."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        cooldown_sec: float = 5.0,
        global_cooldown_sec: float = 4.0,
        min_weighted_pixels: float = 1500.0,
        growth_factor: float = 1.5,
        max_simultaneous_categories: int = 2,
    ) -> None:
        self.enabled = bool(enabled)
        self.cooldown_sec = float(cooldown_sec)
        self.global_cooldown_sec = float(global_cooldown_sec)
        self.min_weighted_pixels = float(min_weighted_pixels)
        self.growth_factor = float(growth_factor)
        self.max_simultaneous_categories = int(max_simultaneous_categories)
        self._history: dict[str, _ClassHistory] = {}
        self._last_spoken_at: dict[str, float] = {}
        self._last_any_alert_at: float = -1e9

    @classmethod
    def from_settings(cls, settings) -> "AlertTracker":
        """Build from Settings, honoring the alert-tuning fields if present."""
        return cls(
            enabled=getattr(settings, "alerts_enabled", True),
            cooldown_sec=getattr(settings, "alert_cooldown_sec", 5.0),
            global_cooldown_sec=getattr(settings, "alert_global_cooldown_sec", 4.0),
            min_weighted_pixels=getattr(settings, "alert_min_weighted_pixels", 1500.0),
            growth_factor=getattr(settings, "alert_growth_factor", 1.5),
            max_simultaneous_categories=getattr(
                settings, "alert_max_simultaneous_categories", 2
            ),
        )

    def update(self, segmentation: SegmentationResult) -> list[ProximityAlert]:
        """Push the latest frame, return at most one alert that fired."""
        if not self.enabled:
            return []

        now = time.monotonic()
        per_category = self._weighted_counts_per_category(segmentation)

        # Push every known category (zero counts kept so trends decay).
        for category in CATEGORY_PRIORITY:
            count = per_category.get(category, 0.0)
            self._history.setdefault(category, _ClassHistory()).push(count)

        # --- Scene-coherence gate ---------------------------------------
        # Count how many distinct categories are *meaningfully* present this
        # frame. A real walking scene has 0-2. If more than the cap are all
        # present at once, the model is almost certainly hallucinating
        # (common indoors / on empty paths) — stay silent.
        present_now = [
            c
            for c, v in per_category.items()
            if v >= self.min_weighted_pixels * 0.5
        ]
        if len(present_now) > self.max_simultaneous_categories:
            return []

        # --- Global cooldown --------------------------------------------
        if now - self._last_any_alert_at < self.global_cooldown_sec:
            return []

        alerts: list[ProximityAlert] = []
        for category, history in self._history.items():
            if not history.is_growing(
                growth_factor=self.growth_factor,
                min_recent=self.min_weighted_pixels,
            ):
                continue
            last = self._last_spoken_at.get(category, 0.0)
            if now - last < self.cooldown_sec:
                continue
            phrase = CATEGORY_PHRASE.get(category, f"{category} approaching")
            priority = CATEGORY_PRIORITY.get(category, 0)
            recent = history.counts[-1] if history.counts else 0.0
            alerts.append(
                ProximityAlert(
                    category=category,
                    phrase=phrase,
                    confidence=min(1.0, recent / (self.min_weighted_pixels * 4)),
                    priority=priority,
                    rationale=f"weighted={recent:.0f}",
                )
            )

        # Speak only the single highest-priority alert; the global cooldown
        # means the next one (if still valid) waits its turn seconds later.
        alerts.sort(reverse=True)
        if alerts:
            self._last_spoken_at[alerts[0].category] = now
            self._last_any_alert_at = now
            return [alerts[0]]
        return []

    @staticmethod
    def _weighted_counts_per_category(
        segmentation: SegmentationResult,
    ) -> dict[str, float]:
        """Convert per-class pixel info into per-category weighted counts.

        The only segmentation backend is semantic (ADE20K SegFormer), so this
        always reads the dense ``class_map``.
        """
        meta = segmentation.metadata or {}
        class_map = segmentation.class_map
        id_to_name = {int(k): str(v) for k, v in (meta.get("id_to_name") or {}).items()}
        if class_map is None or not id_to_name:
            return {}

        from navigation.perception.segmentation_base import _region_weight_map

        cm = np.asarray(class_map)
        if cm.ndim == 3:
            cm = cm[0]
        weights = _region_weight_map(cm.shape[:2])

        # One weighted ``bincount`` pass tallies the region-weighted pixel mass
        # for every class id; only the few ids mapped to an alert category are
        # then folded in. Avoids a full-frame boolean mask per distinct class.
        flat = cm.ravel()
        length = int(flat.max()) + 1
        pixels_by_id = np.bincount(flat, minlength=length)
        weighted_by_id = np.bincount(
            flat, weights=weights.ravel(), minlength=length
        )
        per_category: dict[str, float] = {}
        for cls_id in np.nonzero(pixels_by_id)[0]:
            name = id_to_name.get(int(cls_id), str(int(cls_id)))
            category = CLASS_TO_CATEGORY.get(name)
            if category is None:
                continue
            per_category[category] = (
                per_category.get(category, 0.0) + float(weighted_by_id[cls_id])
            )
        return per_category


__all__ = [
    "AlertTracker",
    "ProximityAlert",
    "CATEGORY_PHRASE",
    "CATEGORY_PRIORITY",
    "CLASS_TO_CATEGORY",
]
