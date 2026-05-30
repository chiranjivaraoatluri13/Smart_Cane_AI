"""Tests for the proximity alert tracker (with false-positive guardrails)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from navigation.models import SegmentationResult
from navigation.reasoning.alerts import (
    CATEGORY_PHRASE,
    CATEGORY_PRIORITY,
    CLASS_TO_CATEGORY,
    AlertTracker,
)


def _seg_with_class_in_path(class_name: str, *, fraction: float = 0.5) -> SegmentationResult:
    """Build a SegmentationResult where ``class_name`` covers ``fraction`` of
    the bottom-center walking path."""
    h, w = 240, 320
    cls_id = 99
    class_map = np.zeros((h, w), dtype=np.int32)
    y0 = int(h * (1 - fraction))
    x0, x1 = int(w * 0.3), int(w * 0.7)
    class_map[y0:, x0:x1] = cls_id
    return SegmentationResult(
        class_map=class_map,
        metadata={
            "semantic": True,
            "id_to_name": {0: "road", cls_id: class_name},
            "shape": [h, w],
        },
    )


def _tracker(**overrides) -> AlertTracker:
    """A tracker tuned low + no cooldowns so growth logic is testable."""
    base = dict(
        enabled=True,
        cooldown_sec=0.0,
        global_cooldown_sec=0.0,
        min_weighted_pixels=100.0,
        growth_factor=1.5,
        max_simultaneous_categories=2,
    )
    base.update(overrides)
    return AlertTracker(**base)


def _push_growing(tracker: AlertTracker, class_name: str, fractions: list[float]):
    """Push frames of the class growing; return the FIRST non-empty alert
    list seen across the sequence (so a later cooldown-suppressed frame
    doesn't mask an alert that already fired)."""
    fired: list = []
    for f in fractions:
        alerts = tracker.update(_seg_with_class_in_path(class_name, fraction=f))
        if alerts and not fired:
            fired = alerts
    return fired


# ---------------------------------------------------------------------------
# No false positives
# ---------------------------------------------------------------------------


def test_no_alert_on_empty_frame():
    tracker = _tracker()
    seg = SegmentationResult(metadata={"semantic": True})
    assert tracker.update(seg) == []


def test_no_alert_on_single_frame():
    """Approaching is temporal; a single frame never alerts."""
    tracker = _tracker()
    alerts = tracker.update(_seg_with_class_in_path("person", fraction=0.4))
    assert alerts == []


def test_no_alert_on_sudden_popup_without_growth():
    """A class that pops in big but doesn't grow is treated as noise."""
    tracker = _tracker()
    # Same large size every frame — present but NOT growing.
    alerts = _push_growing(tracker, "person", [0.5, 0.5, 0.5, 0.5, 0.5])
    assert alerts == []


def test_scene_coherence_gate_silences_hallucinated_multiclass():
    """If too many distinct categories are present at once (the signature of
    a hallucinating model on an indoor/empty scene), stay silent."""
    tracker = _tracker(max_simultaneous_categories=2)
    h, w = 240, 320
    # Four different street categories all present in one frame — impossible
    # in a real walking scene, classic indoor-hallucination pattern.
    cm = np.zeros((h, w), dtype=np.int32)
    cm[120:, 0:80] = 1     # car
    cm[120:, 80:160] = 2   # truck
    cm[120:, 160:240] = 3  # bicycle
    cm[120:, 240:320] = 4  # person
    seg = SegmentationResult(
        class_map=cm,
        metadata={
            "semantic": True,
            "id_to_name": {0: "road", 1: "car", 2: "truck", 3: "bicycle", 4: "person"},
            "shape": [h, w],
        },
    )
    # Even after several frames of this, no alert — the coherence gate trips.
    for _ in range(6):
        alerts = tracker.update(seg)
    assert alerts == []


def test_disabled_tracker_never_alerts():
    tracker = _tracker(enabled=False)
    alerts = _push_growing(tracker, "person", [0.05, 0.2, 0.4, 0.6, 0.8])
    assert alerts == []


# ---------------------------------------------------------------------------
# True positives
# ---------------------------------------------------------------------------


def test_alert_fires_when_class_grows_over_time():
    """A genuinely approaching (growing) person → 'Person approaching'."""
    tracker = _tracker()
    alerts = _push_growing(tracker, "person", [0.05, 0.15, 0.3, 0.5, 0.7])
    assert len(alerts) == 1
    assert alerts[0].category == "person"
    assert alerts[0].phrase == "Person approaching"


def test_motorcycle_maps_to_scooter_phrase():
    tracker = _tracker()
    alerts = _push_growing(tracker, "motorcycle", [0.05, 0.15, 0.3, 0.5, 0.7])
    assert len(alerts) == 1
    assert alerts[0].category == "scooter"
    assert alerts[0].phrase == "Scooter approaching"


def test_heavy_vehicle_outranks_person_when_both_grow():
    """Truck outranks person when both grow in the same frames."""
    tracker = _tracker()
    h, w = 240, 320
    truck_id, person_id = 10, 11

    def with_both(truck_frac, person_frac):
        cm = np.zeros((h, w), dtype=np.int32)
        cm[int(h * (1 - truck_frac)):, int(w * 0.5):int(w * 0.7)] = truck_id
        cm[int(h * (1 - person_frac)):, int(w * 0.3):int(w * 0.5)] = person_id
        return SegmentationResult(
            class_map=cm,
            metadata={
                "semantic": True,
                "id_to_name": {truck_id: "truck", person_id: "person"},
                "shape": [h, w],
            },
        )

    alerts = []
    for tf, pf in [(0.05, 0.05), (0.15, 0.15), (0.3, 0.3), (0.5, 0.5), (0.7, 0.7)]:
        alerts = tracker.update(with_both(tf, pf))
    assert len(alerts) == 1
    assert alerts[0].category == "heavy_vehicle"


def test_unknown_classes_ignored():
    """A class with no category mapping (e.g. 'building') never alerts."""
    tracker = _tracker()
    alerts = _push_growing(tracker, "building", [0.05, 0.2, 0.4, 0.6, 0.9])
    assert alerts == []


# ---------------------------------------------------------------------------
# Cooldowns
# ---------------------------------------------------------------------------


def test_global_cooldown_limits_rapid_alerts(monkeypatch):
    """At most one alert per global_cooldown_sec, regardless of category."""
    fake = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: fake[0])

    tracker = _tracker(global_cooldown_sec=4.0, cooldown_sec=0.0)
    a1 = _push_growing(tracker, "person", [0.05, 0.15, 0.3, 0.5, 0.7])
    assert len(a1) == 1

    # Immediately try a growing car — blocked by the global cooldown even
    # though the car is genuinely growing.
    fake[0] += 1.0
    blocked = []
    for f in [0.05, 0.15, 0.3, 0.5]:
        out = tracker.update(_seg_with_class_in_path("car", fraction=f))
        if out:
            blocked = out
    assert blocked == []

    # After the global window elapses, a fresh growing car is allowed.
    fake[0] += 6.0
    a3 = _push_growing(tracker, "car", [0.05, 0.15, 0.3, 0.5, 0.7])
    assert len(a3) == 1


def test_per_category_cooldown_prevents_repeat(monkeypatch):
    fake = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: fake[0])

    tracker = _tracker(cooldown_sec=10.0, global_cooldown_sec=0.0)
    a1 = _push_growing(tracker, "person", [0.05, 0.15, 0.3, 0.5, 0.7])
    assert len(a1) == 1
    fake[0] += 2.0
    a2 = tracker.update(_seg_with_class_in_path("person", fraction=0.9))
    assert a2 == []


# ---------------------------------------------------------------------------
# Table integrity
# ---------------------------------------------------------------------------


def test_phrase_table_covers_every_priority_category():
    for category in CATEGORY_PRIORITY:
        assert category in CATEGORY_PHRASE
        assert CATEGORY_PHRASE[category]


def test_class_map_only_contains_known_categories():
    for cls_name, cat in CLASS_TO_CATEGORY.items():
        assert cat in CATEGORY_PRIORITY, (
            f"class {cls_name!r} mapped to unknown category {cat!r}"
        )
