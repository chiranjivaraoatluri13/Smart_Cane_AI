"""Tests for navigation.perception.spatial — per-side helpers."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from navigation.config import Settings
from navigation.models import SIDES
from navigation.perception.segmentation_base import _region_weight_map
from navigation.perception.segmentation_segformer import SegformerSegmenter
from navigation.perception.spatial import (
    _per_side_class_pixels,
    _per_side_walkable_ratio,
    _side_slices,
    empty_per_side_counts,
    empty_per_side_walkable,
)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


# Feature: spatial-aware-natural-language-guidance, Property 1: Per-side conservation
@given(
    h=st.integers(min_value=4, max_value=64),
    w=st.integers(min_value=4, max_value=64),
    n_classes=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=100, deadline=None)
def test_per_side_counts_round_trip_to_global(h, w, n_classes, seed):
    """Sum of per-side weighted counts equals the global weighted count
    for every class within tolerance of 1, AND left and right have equal
    width (any remainder columns assigned to center)."""
    rng = np.random.default_rng(seed)
    class_map = rng.integers(0, n_classes, size=(h, w), dtype=np.int32)
    id_to_name = {i: f"cls_{i}" for i in range(n_classes)}
    weight_map = _region_weight_map((h, w))

    per_side = _per_side_class_pixels(class_map, id_to_name, weight_map)

    # Side widths: left == right, remainder in center.
    slices = _side_slices(w)
    left_w = slices["left"].stop - slices["left"].start
    right_w = slices["right"].stop - slices["right"].start
    assert left_w == right_w, f"left={left_w} != right={right_w} for w={w}"

    # Sum of per-side equals the global region-weighted count per class.
    for cls_id in np.unique(class_map):
        name = id_to_name[int(cls_id)]
        mask = class_map == cls_id
        global_weighted = float(weight_map[mask].sum())
        side_total = sum(per_side[s].get(name, 0.0) for s in SIDES)
        assert side_total == pytest.approx(global_weighted, abs=1.0)


# Feature: spatial-aware-natural-language-guidance, Property 2: Per-side walkable ratio is a probability
@given(
    h=st.integers(min_value=4, max_value=64),
    w=st.integers(min_value=4, max_value=64),
    n_classes=st.integers(min_value=1, max_value=5),
    walkable_set_seed=st.integers(min_value=0, max_value=31),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=100, deadline=None)
def test_per_side_walkable_ratio_in_unit_interval(
    h, w, n_classes, walkable_set_seed, seed
):
    """Each side's walkable ratio is in [0, 1] for any random class map
    and walkable subset."""
    rng = np.random.default_rng(seed)
    class_map = rng.integers(0, n_classes, size=(h, w), dtype=np.int32)
    id_to_name = {i: f"cls_{i}" for i in range(n_classes)}
    walkable_set = {
        f"cls_{i}" for i in range(n_classes) if (walkable_set_seed >> i) & 1
    }

    per_side = _per_side_walkable_ratio(class_map, id_to_name, walkable_set)
    assert set(per_side.keys()) == set(SIDES)
    for side, ratio in per_side.items():
        assert 0.0 <= ratio <= 1.0, f"{side}: {ratio}"


# ---------------------------------------------------------------------------
# Example tests
# ---------------------------------------------------------------------------


def test_per_side_counts_split_evenly():
    """Width 30 → left[0:10], center[10:20], right[20:30]; uniform class
    map produces equal per-side weighted counts."""
    h, w = 12, 30
    class_map = np.full((h, w), 0, dtype=np.int32)
    id_to_name = {0: "road"}
    weights = _region_weight_map((h, w))
    per_side = _per_side_class_pixels(class_map, id_to_name, weights)
    # Width split is 10/10/10 — center has the highest weights because the
    # horizontal weight map peaks at the center, so center > left == right.
    assert per_side["left"]["road"] == pytest.approx(per_side["right"]["road"])
    assert per_side["center"]["road"] >= per_side["left"]["road"]


def test_side_slices_remainder_goes_to_center():
    """w=320 => left=106, center=108, right=106 (remainder of 2 in center)."""
    slices = _side_slices(320)
    assert slices["left"].stop - slices["left"].start == 106
    assert slices["center"].stop - slices["center"].start == 108
    assert slices["right"].stop - slices["right"].start == 106


def test_side_slices_zero_width_is_empty():
    slices = _side_slices(0)
    for s in SIDES:
        assert slices[s].start == slices[s].stop == 0


def test_empty_helpers_return_correct_shapes():
    counts = empty_per_side_counts()
    walk = empty_per_side_walkable()
    assert set(counts.keys()) == set(SIDES)
    assert set(walk.keys()) == set(SIDES)
    for s in SIDES:
        assert counts[s] == {}
        assert walk[s] == 0.0


def test_parser_populates_per_side_dicts():
    """Requirement 1.6 — the segmenter parser populates per-side fields."""
    segmenter = SegformerSegmenter(Settings())
    segmenter._id_to_name = {0: "floor", 1: "person"}
    class_map = np.zeros((40, 60), dtype=np.int32)
    class_map[20:, :] = 0
    class_map[:20, 20:40] = 1
    seg = segmenter._parse_class_map(class_map, 40, 60)
    assert seg.per_side_class_pixels is not None
    assert seg.per_side_walkable_ratio is not None
    assert set(seg.per_side_class_pixels.keys()) == set(SIDES)
    assert set(seg.per_side_walkable_ratio.keys()) == set(SIDES)
    for s in SIDES:
        assert 0.0 <= seg.per_side_walkable_ratio[s] <= 1.0


def test_weight_map_shape_mismatch_raises():
    cm = np.zeros((4, 6), dtype=np.int32)
    wrong_weights = np.ones((10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        _per_side_class_pixels(cm, {0: "x"}, wrong_weights)
