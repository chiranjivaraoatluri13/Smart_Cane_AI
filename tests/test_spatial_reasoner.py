"""Tests for navigation.reasoning.spatial_reasoner."""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from navigation.config import Settings
from navigation.models import (
    CareResult,
    DepthResult,
    NavigationCommand,
    SegmentationResult,
)
from navigation.reasoning.facts import RouteCue, StairsResult
from navigation.reasoning.spatial_reasoner import SpatialReasoner


def _settings(**kwargs) -> Settings:
    base = dict(
        hazard_obstacle_ratio=0.05,
        min_lane_walkable_ratio=0.10,
    )
    base.update(kwargs)
    return Settings(**base)


def _seg(
    *,
    per_side_class_pixels=None,
    per_side_walkable_ratio=None,
    walkable_ratio=0.4,
    shape=(240, 320),
) -> SegmentationResult:
    return SegmentationResult(
        per_side_class_pixels=per_side_class_pixels,
        per_side_walkable_ratio=per_side_walkable_ratio,
        walkable_ratio=walkable_ratio,
        metadata={"semantic": True, "shape": list(shape)},
    )


def _no_stairs() -> StairsResult:
    return StairsResult(False, 0.0, "")


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


_ROUTE_CUE_CHOICES = st.one_of(
    st.none(),
    st.builds(
        RouteCue,
        turn=st.sampled_from(["left", "right", "forward", "stop"]),
        meters_to_turn=st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
        target_bearing_deg=st.floats(min_value=0.0, max_value=359.0, allow_nan=False),
    ),
)


# Feature: spatial-aware-natural-language-guidance, Property 4: Vision STOP overrides everything
@given(
    hazard=st.booleans(),
    route_cue=_ROUTE_CUE_CHOICES,
    walk_left=st.floats(min_value=0.0, max_value=1.0),
    walk_center=st.floats(min_value=0.0, max_value=1.0),
    walk_right=st.floats(min_value=0.0, max_value=1.0),
    safety_score=st.floats(min_value=0.0, max_value=1.0),
    safe_dir=st.floats(min_value=-90.0, max_value=90.0),
)
@settings(max_examples=100, deadline=None)
def test_vision_stop_overrides_all(
    hazard, route_cue, walk_left, walk_center, walk_right, safety_score, safe_dir
):
    """When CARE flags hazard, decision is STOP; route_cue may remain for HUD."""
    if not hazard:
        return  # only assert on the hazard branch
    reasoner = SpatialReasoner(_settings())
    seg = _seg(
        per_side_walkable_ratio={"left": walk_left, "center": walk_center, "right": walk_right},
    )
    care = CareResult(
        hazard_detected=True, safety_score=safety_score, safe_direction_deg=safe_dir
    )
    decision, facts = reasoner.decide(
        seg, DepthResult(), care, route_cue, stairs=_no_stairs()
    )
    assert decision.command == NavigationCommand.STOP
    assert facts.vision_stop is True
    # route_cue kept for on-screen turn info; composer still picks vision_stop.
    if route_cue is not None:
        assert facts.route_cue == route_cue


# Feature: spatial-aware-natural-language-guidance, Property 3: directional move only when target walkable
@given(
    walk_left=st.floats(min_value=0.0, max_value=1.0),
    walk_center=st.floats(min_value=0.0, max_value=1.0),
    walk_right=st.floats(min_value=0.0, max_value=1.0),
    safe_dir=st.floats(min_value=-90.0, max_value=90.0),
    safety_score=st.floats(min_value=0.5, max_value=1.0),
)
@settings(max_examples=100, deadline=None)
def test_directional_move_only_when_target_walkable_at_least_center(
    walk_left, walk_center, walk_right, safe_dir, safety_score
):
    """A MOVE_LEFT decision implies left walkable >= center walkable; mirror for right."""
    reasoner = SpatialReasoner(_settings())
    # Force at least one lane above min_lane_walkable_ratio so we don't hit the
    # all-lanes-blocked SLOW_DOWN branch.
    walk_center = max(walk_center, 0.11)
    seg = _seg(
        per_side_walkable_ratio={"left": walk_left, "center": walk_center, "right": walk_right},
    )
    care = CareResult(
        hazard_detected=False, safety_score=safety_score, safe_direction_deg=safe_dir
    )
    decision, _ = reasoner.decide(
        seg, DepthResult(), care, None, stairs=_no_stairs()
    )
    if decision.command == NavigationCommand.MOVE_LEFT:
        assert walk_left >= walk_center
    if decision.command == NavigationCommand.MOVE_RIGHT:
        assert walk_right >= walk_center


# ---------------------------------------------------------------------------
# Example tests
# ---------------------------------------------------------------------------


def test_combines_left_and_right_hazards():
    """Per-side hazards include classes from both sides simultaneously."""
    reasoner = SpatialReasoner(_settings())
    seg = _seg(
        per_side_class_pixels={
            "left": {"car": 800.0},
            "center": {},
            "right": {"bicycle": 600.0},
        },
        per_side_walkable_ratio={"left": 0.1, "center": 0.5, "right": 0.1},
    )
    care = CareResult(hazard_detected=False, safety_score=0.9, safe_direction_deg=0.0)
    _, facts = reasoner.decide(seg, DepthResult(), care, None, stairs=_no_stairs())
    cats_left = {h.category for h in facts.hazards_by_side["left"]}
    cats_right = {h.category for h in facts.hazards_by_side["right"]}
    assert "car" in cats_left
    assert "bicycle" in cats_right


def test_route_cue_merged_into_facts():
    reasoner = SpatialReasoner(_settings())
    seg = _seg(
        per_side_walkable_ratio={"left": 0.3, "center": 0.6, "right": 0.6},
    )
    care = CareResult(hazard_detected=False, safety_score=0.9, safe_direction_deg=0.0)
    cue = RouteCue(turn="right", meters_to_turn=20.0, target_bearing_deg=90.0)
    decision, facts = reasoner.decide(
        seg, DepthResult(), care, cue, stairs=_no_stairs()
    )
    assert facts.route_cue is not None
    assert facts.route_cue.turn == "right"
    assert decision.command == NavigationCommand.MOVE_RIGHT


def test_map_and_vision_blend_when_clear():
    """Route says right, right side is walkable, command becomes MOVE_RIGHT."""
    reasoner = SpatialReasoner(_settings())
    seg = _seg(
        per_side_walkable_ratio={"left": 0.2, "center": 0.4, "right": 0.7},
    )
    care = CareResult(hazard_detected=False, safety_score=0.9, safe_direction_deg=0.0)
    cue = RouteCue(turn="right", meters_to_turn=15.0, target_bearing_deg=90.0)
    decision, _ = reasoner.decide(seg, DepthResult(), care, cue, stairs=_no_stairs())
    assert decision.command == NavigationCommand.MOVE_RIGHT


def test_vision_stop_drops_route():
    reasoner = SpatialReasoner(_settings())
    seg = _seg(
        per_side_walkable_ratio={"left": 0.5, "center": 0.5, "right": 0.5},
    )
    care = CareResult(hazard_detected=True, safety_score=0.2, safe_direction_deg=0.0)
    cue = RouteCue(turn="left", meters_to_turn=10.0, target_bearing_deg=270.0)
    decision, facts = reasoner.decide(seg, DepthResult(), care, cue, stairs=_no_stairs())
    assert decision.command == NavigationCommand.STOP
    assert facts.route_cue == cue


def test_slow_down_when_no_lane_walkable():
    """Req 2.5 — every side below min_lane_walkable_ratio → SLOW_DOWN."""
    reasoner = SpatialReasoner(_settings(min_lane_walkable_ratio=0.20))
    seg = _seg(
        per_side_walkable_ratio={"left": 0.05, "center": 0.05, "right": 0.05},
    )
    care = CareResult(hazard_detected=False, safety_score=0.8, safe_direction_deg=0.0)
    decision, _ = reasoner.decide(seg, DepthResult(), care, None, stairs=_no_stairs())
    assert decision.command == NavigationCommand.SLOW_DOWN


def test_route_cue_stop_at_destination():
    reasoner = SpatialReasoner(_settings())
    seg = _seg(
        per_side_walkable_ratio={"left": 0.5, "center": 0.5, "right": 0.5},
    )
    care = CareResult(hazard_detected=False, safety_score=0.9)
    cue = RouteCue(turn="stop", meters_to_turn=2.0, target_bearing_deg=0.0)
    decision, facts = reasoner.decide(seg, DepthResult(), care, cue, stairs=_no_stairs())
    assert decision.command == NavigationCommand.STOP
    assert facts.vision_stop is False  # destination, not hazard
    assert facts.route_cue is not None  # destination cue stays


def test_facts_carries_distance_bucket_and_phrase():
    reasoner = SpatialReasoner(_settings())
    seg = _seg(
        per_side_walkable_ratio={"left": 0.4, "center": 0.4, "right": 0.4},
    )
    care = CareResult(hazard_detected=False, safety_score=0.9)
    depth = DepthResult(obstacle_depth_m=2.5)
    _, facts = reasoner.decide(seg, depth, care, None, stairs=_no_stairs())
    assert facts.distance_bucket in {"immediate", "near", "mid", "far"}
    assert facts.distance_phrase
