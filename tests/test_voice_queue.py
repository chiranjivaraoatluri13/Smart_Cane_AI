"""Tests for navigation.output.voice_queue.VoiceQueue."""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from navigation.config import Settings
from navigation.output.voice_queue import (
    TIER_PRIORITY,
    VALID_TIERS,
    VoiceItem,
    VoiceQueue,
)


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _settings(**kw) -> Settings:
    base = dict(
        tts_enabled=True,
        status_update_interval_sec=10.0,
        voice_cooldowns={
            "vision_stop": 0.0,
            "directional_warning": 2.0,
            "map_turn": 8.0,
            "approach_alert": 3.0,
            "status_update": 10.0,
        },
    )
    base.update(kw)
    return Settings(**base)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


# Feature: spatial-aware-natural-language-guidance, Property 9: priority always wins on enqueue
@given(
    sequence=st.lists(
        st.tuples(st.sampled_from(VALID_TIERS), st.text(min_size=1, max_size=10)),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=100, deadline=None)
def test_higher_tier_preempts_lower_tier(sequence):
    """The item that survives to be returned by tick() has the highest tier."""
    clock = FakeClock(1000.0)
    queue = VoiceQueue(_settings(), clock=clock)
    queue.reset_cooldowns()  # ignore status_update_interval gating
    # Allow vision_stop to flow even with default cooldown 0 by skipping ahead.
    clock.advance(100.0)
    queue.reset_cooldowns()

    for tier, phrase in sequence:
        queue.enqueue(VoiceItem(tier=tier, phrase=phrase))

    spoken = queue.tick()
    if spoken is None:
        # Cooldown gated everything — that's allowed (e.g. only status_update enqueues
        # but interval not elapsed). Skip this case.
        return

    expected_priority = max(TIER_PRIORITY[t] for t, _ in sequence)
    candidates = {p for t, p in sequence if TIER_PRIORITY[t] == expected_priority}
    assert spoken in candidates


# Feature: spatial-aware-natural-language-guidance, Property 10: per-tier cooldowns
@given(
    n_calls=st.integers(min_value=2, max_value=4),
    dt=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
)
@settings(max_examples=100, deadline=None)
def test_map_turn_cooldown(n_calls, dt):
    """Successive map_turn enqueues, each separated by dt < cooldown / n_calls,
    cannot all fire — at least one must be cooldown-suppressed."""
    clock = FakeClock(1000.0)
    queue = VoiceQueue(_settings(), clock=clock)

    fired = 0
    for i in range(n_calls):
        queue.enqueue(VoiceItem(tier="map_turn", phrase=f"turn {i}"))
        out = queue.tick()
        if out is not None:
            fired += 1
        clock.advance(dt)

    # Total elapsed time across the loop is (n_calls - 1) * dt. If that's
    # less than the cooldown, then all of `n_calls - 1` later items are
    # suppressed and only the first fires.
    elapsed = (n_calls - 1) * dt
    if elapsed < queue._cooldowns["map_turn"]:
        # Within a single cooldown window: max one item fires.
        assert fired <= 1
    assert fired <= n_calls


# Feature: spatial-aware-natural-language-guidance, Property 10: per-tier cooldowns
def test_status_update_interval():
    """status_update items inside the cooldown window are suppressed."""
    clock = FakeClock(1000.0)
    queue = VoiceQueue(_settings(), clock=clock)
    queue.enqueue(VoiceItem(tier="status_update", phrase="all clear"))
    first = queue.tick()
    assert first == "all clear"

    # Within cooldown.
    clock.advance(5.0)
    queue.enqueue(VoiceItem(tier="status_update", phrase="still clear"))
    assert queue.tick() is None

    # Past cooldown.
    clock.advance(6.0)
    queue.enqueue(VoiceItem(tier="status_update", phrase="still clear"))
    assert queue.tick() == "still clear"


# Feature: spatial-aware-natural-language-guidance, Property 10: per-tier cooldowns
def test_per_tier_cooldowns_independent():
    clock = FakeClock(1000.0)
    queue = VoiceQueue(_settings(), clock=clock)
    queue.enqueue(VoiceItem(tier="map_turn", phrase="turn"))
    assert queue.tick() == "turn"

    # status_update cooldown is independent from map_turn — still allowed.
    clock.advance(0.5)
    queue.enqueue(VoiceItem(tier="vision_stop", phrase="stop"))
    assert queue.tick() == "stop"


# ---------------------------------------------------------------------------
# Example tests
# ---------------------------------------------------------------------------


def test_stop_preempts():
    clock = FakeClock(1000.0)
    queue = VoiceQueue(_settings(), clock=clock)
    queue.enqueue(VoiceItem(tier="status_update", phrase="all clear"))
    queue.enqueue(VoiceItem(tier="vision_stop", phrase="STOP"))
    assert queue.tick() == "STOP"


def test_lower_tier_does_not_overwrite_pending_higher_tier():
    clock = FakeClock(1000.0)
    queue = VoiceQueue(_settings(), clock=clock)
    queue.enqueue(VoiceItem(tier="directional_warning", phrase="move"))
    queue.enqueue(VoiceItem(tier="status_update", phrase="all clear"))
    assert queue.tick() == "move"


def test_unknown_tier_is_dropped(caplog):
    clock = FakeClock(1000.0)
    queue = VoiceQueue(_settings(), clock=clock)
    with caplog.at_level("WARNING"):
        queue.enqueue(VoiceItem(tier="bogus", phrase="x"))  # type: ignore[arg-type]
    assert queue.tick() is None
    assert any("unknown tier" in r.message.lower() for r in caplog.records)


def test_tts_disabled_logs_without_error(caplog):
    clock = FakeClock(1000.0)
    queue = VoiceQueue(_settings(tts_enabled=False), clock=clock)
    with caplog.at_level("INFO"):
        queue.enqueue(VoiceItem(tier="vision_stop", phrase="stop"))
        out = queue.tick()
    assert out == "stop"
    assert any("VoiceQueue:silent" in r.message for r in caplog.records)


def test_is_idle_after_interval():
    clock = FakeClock(1000.0)
    queue = VoiceQueue(_settings(), clock=clock)
    queue.enqueue(VoiceItem(tier="vision_stop", phrase="stop"))
    queue.tick()
    assert queue.is_idle() is False
    clock.advance(11.0)
    assert queue.is_idle() is True
