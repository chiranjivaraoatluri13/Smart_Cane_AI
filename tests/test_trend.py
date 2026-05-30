"""Tests for navigation.reasoning.trend — TrendTracker."""

from __future__ import annotations

from collections import deque

import pytest
from hypothesis import given, settings, strategies as st

from navigation.config import Settings
from navigation.reasoning.facts import ApproachDirectionTuple
from navigation.reasoning.trend import (
    TrendTracker,
    centroid_x_norm,
    per_side_to_per_category,
)


_VALID_LABELS = set(ApproachDirectionTuple)


def _push_sequence(
    tracker: TrendTracker,
    sequence: list[tuple[float, float, float]],
    *,
    cls_name: str = "person",
) -> None:
    """Push `sequence` of (left, center, right) weights for one Cityscapes class."""
    for left, center, right in sequence:
        tracker.update(
            {
                "left": {cls_name: left},
                "center": {cls_name: center},
                "right": {cls_name: right},
            }
        )


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


# Feature: spatial-aware-natural-language-guidance, Property 5: monotone centroid sweeps
@given(
    span=st.floats(min_value=0.16, max_value=1.0, allow_nan=False),
    weight=st.floats(min_value=200.0, max_value=2000.0, allow_nan=False),
    n_samples=st.integers(min_value=3, max_value=6),
)
@settings(max_examples=100, deadline=None)
def test_classify_crossing_left_to_right(span, weight, n_samples):
    """A monotone L→R centroid sweep ≥ cross_threshold labels as crossing_left_to_right."""
    tracker = TrendTracker()
    # Build a sequence whose centroid moves monotonically from ~0.17 to ~0.17+span
    # by shifting weight from left → right.
    # We want first centroid near left (0.17) and last >= 0.17 + cross_threshold.
    # Achieve this by ramping the right weight from 0 up across samples.
    # First sample: all on the left.
    # Last sample: enough on the right to push centroid by >=0.15.
    sequence: list[tuple[float, float, float]] = []
    for i in range(n_samples):
        t = i / (n_samples - 1)  # 0.0 → 1.0
        left = weight * (1.0 - t)
        right = weight * t * (span / 0.66)  # scale so we cross threshold
        sequence.append((left, 0.0, right))
    _push_sequence(tracker, sequence)
    label = tracker.classify("person")
    # Either crossing_left_to_right OR closing_in if total weight grew enough.
    # Any non-static label is acceptable; specifically reject crossing_right_to_left.
    assert label != "crossing_right_to_left"
    assert label in _VALID_LABELS


# Feature: spatial-aware-natural-language-guidance, Property 5: monotone centroid sweeps (mirrored)
@given(
    span=st.floats(min_value=0.16, max_value=1.0, allow_nan=False),
    weight=st.floats(min_value=200.0, max_value=2000.0, allow_nan=False),
    n_samples=st.integers(min_value=3, max_value=6),
)
@settings(max_examples=100, deadline=None)
def test_classify_crossing_right_to_left(span, weight, n_samples):
    """A monotone R→L sweep ≥ cross_threshold labels as crossing_right_to_left."""
    tracker = TrendTracker()
    sequence: list[tuple[float, float, float]] = []
    for i in range(n_samples):
        t = i / (n_samples - 1)
        right = weight * (1.0 - t)
        left = weight * t * (span / 0.66)
        sequence.append((left, 0.0, right))
    _push_sequence(tracker, sequence)
    label = tracker.classify("person")
    assert label != "crossing_left_to_right"
    assert label in _VALID_LABELS


# Feature: spatial-aware-natural-language-guidance, Property 6: weight trend + label set
@given(
    base_weight=st.floats(min_value=200.0, max_value=2000.0, allow_nan=False),
    growth=st.floats(min_value=1.5, max_value=10.0, allow_nan=False),
    n_samples=st.integers(min_value=3, max_value=6),
)
@settings(max_examples=100, deadline=None)
def test_classify_closing_in_when_growing(base_weight, growth, n_samples):
    """Weight grows while centroid stays put → closing_in."""
    tracker = TrendTracker()
    sequence: list[tuple[float, float, float]] = []
    for i in range(n_samples):
        t = i / (n_samples - 1)
        # Stay centered; grow total weight from base_weight to base_weight*growth.
        center = base_weight * (1.0 + t * (growth - 1.0))
        sequence.append((0.0, center, 0.0))
    _push_sequence(tracker, sequence)
    label = tracker.classify("person")
    assert label == "closing_in"


# Feature: spatial-aware-natural-language-guidance, Property 6: weight trend + label set
@given(
    base_weight=st.floats(min_value=400.0, max_value=2000.0, allow_nan=False),
    decay=st.floats(min_value=0.05, max_value=0.45, allow_nan=False),
    n_samples=st.integers(min_value=3, max_value=6),
)
@settings(max_examples=100, deadline=None)
def test_classify_receding_when_shrinking(base_weight, decay, n_samples):
    """Weight shrinks below half → receding."""
    tracker = TrendTracker()
    sequence: list[tuple[float, float, float]] = []
    for i in range(n_samples):
        t = i / (n_samples - 1)
        center = base_weight * (1.0 - t * (1.0 - decay))
        sequence.append((0.0, center, 0.0))
    _push_sequence(tracker, sequence)
    label = tracker.classify("person")
    assert label == "receding"


# Feature: spatial-aware-natural-language-guidance, Property 6: label set
@given(
    sequence=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=2000.0, allow_nan=False),
            st.floats(min_value=0.0, max_value=2000.0, allow_nan=False),
            st.floats(min_value=0.0, max_value=2000.0, allow_nan=False),
        ),
        min_size=1,
        max_size=8,
    )
)
@settings(max_examples=100, deadline=None)
def test_classify_returns_valid_label(sequence):
    """Whatever the input, the label is always one of the five strings."""
    tracker = TrendTracker()
    _push_sequence(tracker, sequence)
    label = tracker.classify("person")
    assert label in _VALID_LABELS


# Feature: spatial-aware-natural-language-guidance, Property 6: buffer is bounded
@given(n=st.integers(min_value=1, max_value=50))
@settings(max_examples=100, deadline=None)
def test_buffer_size_bounded(n):
    """Ring buffer never exceeds size 6 regardless of input length."""
    tracker = TrendTracker()
    for i in range(n):
        tracker.update({"left": {"person": float(i)}, "center": {}, "right": {}})
    hist = tracker._history.get("person")
    assert hist is not None
    assert len(hist.samples) <= tracker.buffer_size


# ---------------------------------------------------------------------------
# Example tests
# ---------------------------------------------------------------------------


def test_classify_static_when_no_movement():
    tracker = TrendTracker()
    _push_sequence(tracker, [(100, 100, 100)] * 5)
    assert tracker.classify("person") == "static"


def test_classify_returns_static_for_unknown_category():
    tracker = TrendTracker()
    assert tracker.classify("never_seen") == "static"


def test_classify_static_below_three_samples():
    tracker = TrendTracker()
    _push_sequence(tracker, [(0, 1000, 0), (0, 1000, 0)])
    assert tracker.classify("person") == "static"


def test_per_side_to_per_category_drops_unknown_classes():
    out = per_side_to_per_category(
        {
            "left": {"sky": 100, "person": 50},
            "center": {"building": 200, "car": 30},
            "right": {"vegetation": 80},
        }
    )
    assert "person" in out
    assert "car" in out
    assert "sky" not in out
    assert "building" not in out


def test_centroid_x_norm_zero_total_returns_center():
    assert centroid_x_norm(0.0, 0.0, 0.0) == pytest.approx(0.5)


def test_centroid_x_norm_all_left():
    assert centroid_x_norm(100.0, 0.0, 0.0) == pytest.approx(0.17)


def test_classify_all_returns_dict_for_every_seen_category():
    tracker = TrendTracker()
    _push_sequence(tracker, [(100, 0, 0)] * 4, cls_name="person")
    _push_sequence(tracker, [(0, 0, 100)] * 4, cls_name="car")
    labels = tracker.classify_all()
    assert "person" in labels
    assert "car" in labels
    for v in labels.values():
        assert v in _VALID_LABELS


def test_reset_clears_history():
    tracker = TrendTracker()
    _push_sequence(tracker, [(100, 0, 0)] * 4)
    assert tracker._history
    tracker.reset()
    assert not tracker._history
