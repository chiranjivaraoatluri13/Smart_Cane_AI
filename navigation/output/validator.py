"""Safety validator and command cooldown."""

from __future__ import annotations

import time

from navigation.config import Settings
from navigation.models import NavigationCommand, NavigationDecision
from navigation.reasoning.alerts import ProximityAlert


class CommandValidator:
    """The single authority on whether a command is spoken.

    Layers, in order:

    1. **Dwell filter** — a non-STOP command must be the reasoner's choice
       for at least ``command_dwell_frames`` consecutive inference frames
       before it's allowed to speak. Kills frame-to-frame perception jitter.

    2. **Speak-on-change** — a command is announced when it *first* becomes
       the active command. While the same command persists, it is only
       re-announced after ``command_cooldown_sec`` has elapsed (a gentle
       reminder), not every frame. This is what stops "Wait. Wait. Wait."
       from looping while a hazard sits in front of a stationary user.

    3. **Min-gap floor** — any two spoken utterances are at least
       ``min_speech_gap_sec`` apart. STOP is exempt from dwell and min-gap
       (safety speaks immediately) but still obeys its own change/cooldown
       rule so it doesn't repeat every frame.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._active_command: NavigationCommand | None = None
        self._active_spoken_at: float = 0.0
        self._last_any_spoken_at: float = 0.0
        # Dwell tracking: how many consecutive frames has the *current*
        # candidate been the reasoner's pick?
        self._candidate: NavigationCommand | None = None
        self._candidate_streak: int = 0
        # Per-category proximity alert cooldowns.
        self._last_alert_at: dict[str, float] = {}

    def approve(self, decision: NavigationDecision) -> NavigationDecision:
        now = time.monotonic()
        cooldown = float(self.settings.command_cooldown_sec)
        dwell_frames = max(1, int(self.settings.command_dwell_frames))
        min_gap = float(self.settings.min_speech_gap_sec)
        is_stop = decision.command == NavigationCommand.STOP

        # Dwell tracking — how long has this exact command been the pick?
        if decision.command == self._candidate:
            self._candidate_streak += 1
        else:
            self._candidate = decision.command
            self._candidate_streak = 1

        # Is this a *change* from the command that's currently active
        # (last spoken)? A change is always worth announcing (subject to
        # dwell + min-gap). A continuation is only re-announced after the
        # cooldown elapses.
        changed = decision.command != self._active_command

        # Layer 1: dwell. STOP bypasses (safety wins). A non-STOP command
        # that hasn't been stable long enough is suppressed.
        if not is_stop and self._candidate_streak < dwell_frames:
            return decision.model_copy(update={"speak": False})

        if changed:
            # New command. Announce it — but respect the min-gap floor for
            # non-STOP so we don't talk over a phrase from half a second ago.
            if (
                not is_stop
                and decision.speak
                and (now - self._last_any_spoken_at) < min_gap
            ):
                return decision.model_copy(update={"speak": False})
            speak = decision.speak
        else:
            # Same command continuing. Re-announce only as an occasional
            # reminder once the cooldown elapses.
            if (now - self._active_spoken_at) < cooldown:
                return decision.model_copy(update={"speak": False})
            speak = decision.speak

        if speak:
            self._active_command = decision.command
            self._active_spoken_at = now
            self._last_any_spoken_at = now
            return decision
        return decision.model_copy(update={"speak": False})

    def approve_alert(
        self,
        alert: ProximityAlert,
        *,
        cooldown_sec: float | None = None,
    ) -> bool:
        """Decide whether this alert should be spoken right now.

        The alert tracker already has its own per-category cooldown, but the
        validator gates a final time so the spoken layer never speaks two
        alerts back-to-back faster than this minimum.
        """
        cd = cooldown_sec if cooldown_sec is not None else 1.5
        now = time.monotonic()
        last = self._last_alert_at.get(alert.category, 0.0)
        if now - last < cd:
            return False
        self._last_alert_at[alert.category] = now
        return True
