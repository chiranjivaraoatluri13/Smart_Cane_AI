"""Text-to-speech for navigation commands."""

from __future__ import annotations

import sys

from navigation.config import Settings
from navigation.models import NavigationCommand, NavigationDecision
from navigation.reasoning.alerts import ProximityAlert

_COMMAND_PHRASES: dict[NavigationCommand, str] = {
    NavigationCommand.MOVE_LEFT: "Move left",
    NavigationCommand.MOVE_RIGHT: "Move right",
    NavigationCommand.GO_FORWARD: "Go forward",
    NavigationCommand.SLOW_DOWN: "Slow down",
    NavigationCommand.STOP: "Stop",
}


class SpeechEngine:
    """Speaks approved navigation decisions (main-thread pyttsx3 on Windows)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._engine = None
        self._ready = False

    def _init_engine(self) -> object:
        if self._engine is not None:
            return self._engine
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", self.settings.tts_rate)
        self._engine = engine
        return engine

    def warmup(self) -> bool:
        """Initialize TTS on the main thread before the camera loop."""
        if not self.settings.tts_enabled:
            print("[TTS] Disabled in .env (TTS_ENABLED=false)", flush=True)
            return False
        try:
            engine = self._init_engine()
            voices = engine.getProperty("voices")
            name = voices[0].name if voices else "default"
            self._ready = True
            print(f"[TTS] Ready — voice: {name}", flush=True)
            return True
        except ImportError:
            print(
                '[TTS] pyttsx3 not installed. Run: pip install -e ".[tts]"',
                file=sys.stderr,
                flush=True,
            )
            return False
        except Exception as e:
            print(f"[TTS] Failed to start: {e}", file=sys.stderr, flush=True)
            return False

    def _say(self, phrase: str) -> None:
        """Low-level: say a phrase synchronously. Caller is responsible for
        deciding whether speech is appropriate."""
        if not self.settings.tts_enabled:
            return
        if not self._ready and not self.warmup():
            print(f"[TTS] (text only) {phrase}", flush=True)
            return
        try:
            engine = self._init_engine()
            engine.say(phrase)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS] Speak failed: {e}", file=sys.stderr, flush=True)

    def speak(self, decision: NavigationDecision) -> str:
        phrase = _COMMAND_PHRASES.get(decision.command, decision.command.value)
        if not self.settings.tts_enabled:
            return phrase
        if not decision.speak:
            return phrase
        print(f"[VOICE] {phrase}", flush=True)
        self._say(phrase)
        return phrase

    def speak_alert(self, alert: ProximityAlert) -> str:
        """Speak a per-class proximity alert ("Person approaching", ...).

        Cooldown decisions belong to the validator/tracker; this function
        always speaks (and prints) the phrase.
        """
        print(f"[ALERT] {alert.phrase}  ({alert.category}, {alert.rationale})", flush=True)
        self._say(alert.phrase)
        return alert.phrase
