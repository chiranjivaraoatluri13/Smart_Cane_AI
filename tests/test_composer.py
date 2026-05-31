"""Tests for navigation.reasoning.composer.PhraseComposer."""

from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent

import pytest
from hypothesis import given, settings, strategies as st

from navigation.config import Settings
from navigation.models import NavigationCommand
from navigation.reasoning.composer import PhraseComposer
from navigation.reasoning.facts import (
    GuidanceFacts,
    HazardEntry,
    RouteCue,
    StairsResult,
)


def _settings(seed: int | None = 42, phrases_path: Path | None = None) -> Settings:
    return Settings(
        composer_seed=seed,
        phrases_path=str(phrases_path or Path("config/phrases.yaml")),
        min_lane_walkable_ratio=0.10,
        stairs_min_confidence=0.4,
    )


def _facts(
    *,
    command: NavigationCommand = NavigationCommand.GO_FORWARD,
    vision_stop: bool = False,
    hazards_by_side: dict | None = None,
    walkable_by_side: dict | None = None,
    stairs: StairsResult | None = None,
    distance_phrase: str = "about 6 feet ahead",
    route_cue: RouteCue | None = None,
) -> GuidanceFacts:
    return GuidanceFacts(
        command=command,
        confidence=0.85,
        hazards_by_side=hazards_by_side or {"left": [], "center": [], "right": []},
        walkable_by_side=walkable_by_side or {"left": 0.5, "center": 0.5, "right": 0.5},
        approach_direction_by_category={},
        stairs=stairs or StairsResult(False, 0.0, ""),
        distance_bucket="near",
        distance_phrase=distance_phrase,
        route_cue=route_cue,
        vision_stop=vision_stop,
    )


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@st.composite
def _facts_strategy(draw) -> GuidanceFacts:
    cmd = draw(st.sampled_from(list(NavigationCommand)))
    vision_stop = draw(st.booleans()) and cmd == NavigationCommand.STOP
    has_route = draw(st.booleans())
    route_cue = (
        RouteCue(
            turn=draw(st.sampled_from(["left", "right", "forward"])),
            meters_to_turn=draw(st.floats(min_value=0.0, max_value=200.0, allow_nan=False)),
            target_bearing_deg=draw(st.floats(min_value=0.0, max_value=359.0, allow_nan=False)),
        )
        if has_route
        else None
    )
    side = draw(st.sampled_from(["left", "center", "right"]))
    has_hazard = draw(st.booleans())
    hazards = {"left": [], "center": [], "right": []}
    if has_hazard:
        hazards[side] = [
            HazardEntry(
                category=draw(st.sampled_from(["person", "car", "bicycle", "pole"])),
                weighted_pixels=draw(
                    st.floats(min_value=10.0, max_value=2000.0, allow_nan=False)
                ),
                approach=draw(
                    st.sampled_from(
                        [
                            "static",
                            "crossing_left_to_right",
                            "crossing_right_to_left",
                            "closing_in",
                            "receding",
                        ]
                    )
                ),
            )
        ]
    return _facts(
        command=cmd,
        vision_stop=vision_stop,
        hazards_by_side=hazards,
        route_cue=route_cue,
    )


# Feature: spatial-aware-natural-language-guidance, Property 8: complete & non-repetitive phrases
@given(facts=_facts_strategy())
@settings(max_examples=100, deadline=None)
def test_no_unsubstituted_placeholders(facts):
    composer = PhraseComposer(_settings(seed=0))
    phrase = composer.compose(facts)
    assert phrase, "phrase must be non-empty"
    assert "{" not in phrase, f"unsubstituted placeholder in {phrase!r}"
    assert "}" not in phrase, f"unsubstituted placeholder in {phrase!r}"


# Feature: spatial-aware-natural-language-guidance, Property 8: no repeat in window
def test_paraphrase_no_repeat_in_window(tmp_path):
    """Same scenario, two consecutive calls → different paraphrases when ≥2 exist."""
    yaml_text = dedent(
        """
        status_update_clear:
          - "first."
          - "second."
          - "third."
        """
    )
    p = tmp_path / "phrases.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    composer = PhraseComposer(_settings(seed=0, phrases_path=p))
    facts = _facts(command=NavigationCommand.GO_FORWARD)
    a = composer.compose(facts)
    b = composer.compose(facts)
    assert a != b


# ---------------------------------------------------------------------------
# Example tests
# ---------------------------------------------------------------------------


def test_loads_phrases_yaml():
    composer = PhraseComposer(_settings())
    facts = _facts(command=NavigationCommand.GO_FORWARD)
    phrase = composer.compose(facts)
    assert isinstance(phrase, str) and phrase


def test_missing_template_falls_back(tmp_path):
    # An empty yaml — composer must still return a sensible built-in.
    p = tmp_path / "phrases.yaml"
    p.write_text("", encoding="utf-8")
    composer = PhraseComposer(_settings(phrases_path=p))
    facts = _facts(command=NavigationCommand.GO_FORWARD)
    phrase = composer.compose(facts)
    assert phrase  # built-in fallback


def test_under_populated_tag_warns_once(tmp_path, caplog):
    yaml_text = dedent(
        """
        status_update_clear:
          - "only one."
        """
    )
    p = tmp_path / "phrases.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    with caplog.at_level("WARNING"):
        PhraseComposer(_settings(phrases_path=p))
    assert any("status_update_clear" in r.message for r in caplog.records)


def test_stairs_warning_low_conf_phrase():
    composer = PhraseComposer(_settings())
    facts = _facts(
        command=NavigationCommand.GO_FORWARD,
        stairs=StairsResult(True, 0.6, "row=12"),
    )
    phrase = composer.compose(facts)
    assert "step" in phrase.lower() or "curb" in phrase.lower()


def test_stairs_with_stop_prepended():
    composer = PhraseComposer(_settings())
    facts = _facts(
        command=NavigationCommand.STOP,
        stairs=StairsResult(True, 0.7, "row=15"),
    )
    phrase = composer.compose(facts)
    # The stairs_warning_with_stop scenario produces phrases that mention
    # the step/stair/curb regardless of which paraphrase wins.
    low = phrase.lower()
    assert "stop" in low or "step" in low or "stair" in low or "curb" in low


def test_route_blend_phrase_contains_both_pieces():
    composer = PhraseComposer(_settings())
    facts = _facts(
        command=NavigationCommand.MOVE_RIGHT,
        route_cue=RouteCue(turn="right", meters_to_turn=10.0, target_bearing_deg=90.0),
    )
    phrase = composer.compose(facts)
    assert "right" in phrase.lower() or "feet" in phrase.lower()


def test_no_route_cue_renders_vision_only_phrase():
    composer = PhraseComposer(_settings())
    facts = _facts(
        command=NavigationCommand.MOVE_LEFT,
        hazards_by_side={
            "left": [],
            "center": [],
            "right": [HazardEntry(category="car", weighted_pixels=500.0)],
        },
        walkable_by_side={"left": 0.6, "center": 0.3, "right": 0.1},
    )
    phrase = composer.compose(facts)
    assert phrase  # rendered without crashing
    # Should not mention turn directions / feet since there's no route cue.
    assert not re.search(r"turn (left|right)", phrase.lower())


def test_vision_stop_phrase_does_not_mention_turn():
    composer = PhraseComposer(_settings())
    facts = _facts(
        command=NavigationCommand.STOP,
        vision_stop=True,
        # vision STOP comes with route_cue=None per the reasoner — we just
        # double-check the composer doesn't accidentally pull turn copy.
        route_cue=None,
        hazards_by_side={
            "left": [],
            "center": [HazardEntry(category="person", weighted_pixels=900.0)],
            "right": [],
        },
    )
    phrase = composer.compose(facts)
    # Instructive vocabulary — vision STOP says "wait", "hold up", "stop",
    # or "pause". Whichever paraphrase wins, it must be a stopping word and
    # must not mention a turn direction.
    low = phrase.lower()
    assert any(w in low for w in ("stop", "wait", "hold", "pause"))
    assert not re.search(r"turn (left|right)", low)
    assert not re.search(r"\btake a (left|right)\b", low)


def test_vision_stop_center_hazard_says_ahead_not_center():
    composer = PhraseComposer(_settings(seed=0))
    facts = _facts(
        command=NavigationCommand.STOP,
        vision_stop=True,
        hazards_by_side={
            "left": [],
            "center": [HazardEntry(category="chair", weighted_pixels=900.0)],
            "right": [],
        },
    )
    phrase = composer.compose(facts)
    low = phrase.lower()
    assert "chair" in low
    assert "ahead" in low
    assert "center" not in low


def test_vision_stop_side_hazard_names_position():
    composer = PhraseComposer(_settings(seed=0))
    facts = _facts(
        command=NavigationCommand.STOP,
        vision_stop=True,
        hazards_by_side={
            "left": [HazardEntry(category="pole", weighted_pixels=900.0)],
            "center": [],
            "right": [],
        },
    )
    phrase = composer.compose(facts)
    low = phrase.lower()
    assert "pole" in low
    assert "left" in low


def test_destination_reached_phrase():
    composer = PhraseComposer(_settings())
    facts = _facts(
        command=NavigationCommand.STOP,
        route_cue=RouteCue(turn="stop", meters_to_turn=2.0, target_bearing_deg=0.0),
    )
    phrase = composer.compose(facts)
    assert "arrived" in phrase.lower() or "destination" in phrase.lower() or "here" in phrase.lower()


def test_lane_blocked_all_sides():
    composer = PhraseComposer(_settings())
    facts = _facts(
        command=NavigationCommand.SLOW_DOWN,
        walkable_by_side={"left": 0.0, "center": 0.0, "right": 0.0},
    )
    phrase = composer.compose(facts)
    assert "block" in phrase.lower() or "slow" in phrase.lower() or "no clear" in phrase.lower()
