"""Tests for navigation.reasoning.hazard_labels."""

from navigation.reasoning.hazard_labels import (
    hazard_key_for_class,
    hazard_priority,
    spoken_label,
)

_OBS = {"person", "car", "chair", "wall", "van", "minibike"}
_HAZ = {"stairs", "stairway", "step", "escalator"}


def test_spoken_label_furniture_and_hazards():
    assert spoken_label("chair") == "chair"
    assert spoken_label("stairs") == "steps"
    assert spoken_label("stairway") == "steps"
    assert spoken_label("step") == "steps"
    assert spoken_label("minibike") == "scooter"
    assert spoken_label("streetlight") == "street light"


def test_hazard_key_includes_all_configured_obstacles():
    for cls in ("chair", "van", "person", "wall"):
        assert hazard_key_for_class(cls, _OBS, _HAZ) == spoken_label(cls)


def test_hazard_key_drops_non_obstacles():
    assert hazard_key_for_class("sky", _OBS, _HAZ) is None
    assert hazard_key_for_class("tree", _OBS, _HAZ) is None


def test_steps_outranks_furniture():
    assert hazard_priority("steps") > hazard_priority("chair")
