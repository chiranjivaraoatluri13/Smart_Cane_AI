"""Tests for command cooldown and anti-jitter logic.

The validator is the single authority on whether a command is spoken.
Semantics: a command speaks when it first becomes active (speak-on-change),
then stays silent until either the command changes or the cooldown
"reminder interval" elapses.
"""

import pytest

from navigation.config import Settings
from navigation.models import NavigationCommand, NavigationDecision
from navigation.output.validator import CommandValidator


def _settings(**overrides) -> Settings:
    base = dict(
        command_cooldown_sec=10.0,
        repeat_command_suppress=True,
        command_dwell_frames=1,
        min_speech_gap_sec=0.0,
    )
    base.update(overrides)
    return Settings(**base)


def _decide(cmd: NavigationCommand, speak: bool = True) -> NavigationDecision:
    return NavigationDecision(command=cmd, speak=speak)


# ---------------------------------------------------------------------------
# Speak-on-change behavior
# ---------------------------------------------------------------------------


def test_first_command_speaks():
    v = CommandValidator(_settings())
    assert v.approve(_decide(NavigationCommand.STOP)).speak is True


def test_same_command_within_cooldown_is_silent():
    """A continuing command is not re-announced every frame."""
    v = CommandValidator(_settings(command_cooldown_sec=10.0))
    assert v.approve(_decide(NavigationCommand.STOP)).speak is True
    assert v.approve(_decide(NavigationCommand.STOP)).speak is False
    assert v.approve(_decide(NavigationCommand.STOP)).speak is False


def test_command_change_speaks():
    """A different command is announced immediately (dwell=1, gap=0)."""
    v = CommandValidator(_settings())
    assert v.approve(_decide(NavigationCommand.GO_FORWARD)).speak is True
    assert v.approve(_decide(NavigationCommand.MOVE_LEFT)).speak is True


def test_stop_speaks_on_change_even_during_other_cooldown():
    v = CommandValidator(_settings())
    v.approve(_decide(NavigationCommand.GO_FORWARD))
    # STOP is a change → speaks.
    assert v.approve(_decide(NavigationCommand.STOP)).speak is True


# ---------------------------------------------------------------------------
# Dwell filter
# ---------------------------------------------------------------------------


def test_dwell_suppresses_first_appearances_of_non_stop():
    v = CommandValidator(_settings(command_dwell_frames=3))
    assert v.approve(_decide(NavigationCommand.GO_FORWARD)).speak is False
    assert v.approve(_decide(NavigationCommand.GO_FORWARD)).speak is False
    # Third consecutive frame: dwell satisfied → speaks (it's a change).
    assert v.approve(_decide(NavigationCommand.GO_FORWARD)).speak is True


def test_dwell_never_clears_under_jitter():
    v = CommandValidator(_settings(command_dwell_frames=3))
    for _ in range(10):
        assert v.approve(_decide(NavigationCommand.MOVE_LEFT)).speak is False
        assert v.approve(_decide(NavigationCommand.MOVE_RIGHT)).speak is False


def test_stop_bypasses_dwell():
    v = CommandValidator(_settings(command_dwell_frames=5))
    assert v.approve(_decide(NavigationCommand.STOP)).speak is True


# ---------------------------------------------------------------------------
# Min-gap floor
# ---------------------------------------------------------------------------


def test_min_gap_throttles_back_to_back_distinct_commands():
    v = CommandValidator(
        _settings(command_dwell_frames=1, min_speech_gap_sec=10.0)
    )
    assert v.approve(_decide(NavigationCommand.GO_FORWARD)).speak is True
    # Different command within the min-gap window → suppressed.
    assert v.approve(_decide(NavigationCommand.MOVE_LEFT)).speak is False


def test_min_gap_does_not_throttle_stop():
    v = CommandValidator(
        _settings(command_dwell_frames=1, min_speech_gap_sec=10.0)
    )
    assert v.approve(_decide(NavigationCommand.GO_FORWARD)).speak is True
    # STOP is exempt from the min-gap floor.
    assert v.approve(_decide(NavigationCommand.STOP)).speak is True


# ---------------------------------------------------------------------------
# Reminder interval (cooldown) elapsing
# ---------------------------------------------------------------------------


def test_continuing_command_reannounced_after_cooldown(monkeypatch):
    fake_now = [1000.0]

    import navigation.output.validator as validator_module

    monkeypatch.setattr(validator_module.time, "monotonic", lambda: fake_now[0])
    v = CommandValidator(
        _settings(command_cooldown_sec=5.0, command_dwell_frames=1)
    )
    assert v.approve(_decide(NavigationCommand.STOP)).speak is True
    # Within cooldown → silent reminder suppressed.
    fake_now[0] += 2.0
    assert v.approve(_decide(NavigationCommand.STOP)).speak is False
    # Past cooldown → gentle reminder speaks again.
    fake_now[0] += 4.0
    assert v.approve(_decide(NavigationCommand.STOP)).speak is True


def test_stationary_hazard_speaks_once_then_quiet(monkeypatch):
    """The core fix: STOP on a stationary hazard speaks once, not every frame."""
    fake_now = [1000.0]

    import navigation.output.validator as validator_module

    monkeypatch.setattr(validator_module.time, "monotonic", lambda: fake_now[0])
    v = CommandValidator(
        _settings(command_cooldown_sec=8.0, command_dwell_frames=1)
    )
    spoken = 0
    # Simulate 40 frames of continuous STOP over ~6 seconds.
    for _ in range(40):
        if v.approve(_decide(NavigationCommand.STOP)).speak:
            spoken += 1
        fake_now[0] += 0.15  # ~6.6 fps
    # Should speak at most twice in ~6s with an 8s reminder interval:
    # once at the start. (6s < 8s so no reminder yet.)
    assert spoken == 1
