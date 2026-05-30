"""Priority-aware speech scheduler.

Implements Requirement 9 — five Voice_Tiers with per-tier cooldowns,
preemption rules, and an idle status update interval. The phone path
calls `enqueue()` and `tick()` and speaks whatever phrase `tick()` returns
via the browser's Web Speech API. `_say()` on the laptop path is wired in
the runner.

The queue itself is in-memory only. No I/O. `O(1)` `enqueue()` and `tick()`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from navigation.config import Settings

logger = logging.getLogger(__name__)

VoiceTier = Literal[
    "vision_stop",
    "directional_warning",
    "map_turn",
    "approach_alert",
    "status_update",
]

TIER_PRIORITY: dict[VoiceTier, int] = {
    "vision_stop": 5,
    "directional_warning": 4,
    "map_turn": 3,
    "approach_alert": 2,
    "status_update": 1,
}

VALID_TIERS: tuple[VoiceTier, ...] = (
    "vision_stop",
    "directional_warning",
    "map_turn",
    "approach_alert",
    "status_update",
)


@dataclass
class VoiceItem:
    tier: VoiceTier
    phrase: str
    enqueued_at: float = 0.0
    frame_id: int = 0


class VoiceQueue:
    """Holds at most one pending item; preemption keeps the highest priority."""

    def __init__(
        self,
        settings: Settings,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.settings = settings
        self._clock = clock
        self._pending: VoiceItem | None = None
        self._last_spoken_at: dict[VoiceTier, float] = {t: -1e9 for t in VALID_TIERS}
        self._cooldowns: dict[VoiceTier, float] = self._load_cooldowns(settings)
        self._status_interval = float(settings.status_update_interval_sec)

    @staticmethod
    def _load_cooldowns(settings: Settings) -> dict[VoiceTier, float]:
        cd = dict(settings.voice_cooldowns or {})
        defaults = {
            "vision_stop": 0.0,
            "directional_warning": 2.0,
            "map_turn": 8.0,
            "approach_alert": 3.0,
            "status_update": 10.0,
        }
        return {t: float(cd.get(t, defaults[t])) for t in VALID_TIERS}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, item: VoiceItem) -> None:
        """Put an item in the queue. Higher-priority items preempt lower ones."""
        if item.tier not in TIER_PRIORITY:
            logger.warning("VoiceQueue: dropping item with unknown tier %r", item.tier)
            return
        item.enqueued_at = self._clock()
        if self._pending is None or TIER_PRIORITY[item.tier] >= TIER_PRIORITY[self._pending.tier]:
            self._pending = item

    def tick(self) -> Optional[str]:
        """Decide what to speak right now. Returns the phrase, or None."""
        now = self._clock()
        item = self._pending
        if item is None:
            return None
        cooldown = self._cooldowns.get(item.tier, 0.0)
        if now - self._last_spoken_at[item.tier] < cooldown:
            return None
        # Speak it.
        self._pending = None
        self._last_spoken_at[item.tier] = now
        if not self.settings.tts_enabled:
            logger.info("[VoiceQueue:silent] %s: %s", item.tier, item.phrase)
        return item.phrase

    def is_idle(self, now: float | None = None) -> bool:
        """True when no tier has spoken recently (drives status updates)."""
        ts = self._clock() if now is None else now
        last_any = max(self._last_spoken_at.values())
        return (ts - last_any) >= self._status_interval

    def reset_cooldowns(self) -> None:
        for t in VALID_TIERS:
            self._last_spoken_at[t] = -1e9


__all__ = ["TIER_PRIORITY", "VALID_TIERS", "VoiceItem", "VoiceQueue", "VoiceTier"]
