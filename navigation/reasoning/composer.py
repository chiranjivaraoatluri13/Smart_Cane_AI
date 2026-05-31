"""PhraseComposer — turns GuidanceFacts into a natural-sounding phrase.

Loads `config/phrases.yaml` once at construction. Selects a paraphrase per
scenario, avoids picking the same paraphrase twice in a row, falls back to
a placeholder-free template when a required placeholder is missing.

This component does the bulk of the "conversational" work — short rule-based
intent in, fluent sentence out. It runs in O(P) where P is the number of
paraphrases for the chosen scenario (≤ 5).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import yaml

from navigation.config import Settings
from navigation.models import NavigationCommand, SIDES
from navigation.reasoning.facts import GuidanceFacts

logger = logging.getLogger(__name__)


_BUILTIN_FALLBACKS: dict[str, str] = {
    "vision_stop_generic": "Stop.",
    "vision_stop_with_cause": "Stop.",
    "directional_warning_simple": "Move.",
    "directional_warning_with_freespace": "Move.",
    "approach_alert_static": "Heads up.",
    "approach_alert_crossing": "Heads up, something's crossing.",
    "map_turn_only": "Turn ahead.",
    "map_turn_with_vision_clear": "Turn ahead, path is clear.",
    "map_turn_with_caution": "Turn ahead, slow down.",
    "map_off_route": "You're off the route. Step to your {turn_direction}.",
    "map_route_loading": "Loading walking directions…",
    "status_update_clear": "Path is clear ahead.",
    "status_update_progress": "On track.",
    "stairs_warning_low_conf": "Step ahead, slow down.",
    "stairs_warning_with_stop": "Stop, step ahead.",
    "lane_blocked_all_sides": "Path blocked, slow down.",
    "destination_reached": "You've arrived.",
}


class PhraseComposer:
    """Renders one spoken phrase per call from GuidanceFacts."""

    def __init__(
        self,
        settings: Settings,
        *,
        phrases_path: Path | str | None = None,
    ):
        self.settings = settings
        self.phrases_path = Path(phrases_path or settings.phrases_path)
        self._templates: dict[str, list[str]] = self._load(self.phrases_path)
        self._last_pick: dict[str, str] = {}
        self._rng = (
            random.Random(settings.composer_seed)
            if settings.composer_seed is not None
            else random.Random()
        )
        self._under_populated_warned: set[str] = set()
        self._warn_under_populated()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compose(self, facts: GuidanceFacts) -> str:
        """Pick a scenario, render a paraphrase, substitute placeholders."""
        tag = self._scenario_for(facts)
        templates = self._templates.get(tag) or [_BUILTIN_FALLBACKS.get(tag, "")]
        templates = [t for t in templates if t]
        if not templates:
            templates = [_BUILTIN_FALLBACKS.get(tag, "Heads up.")]

        # Avoid repeating the same paraphrase twice in a row when there are
        # alternatives (Requirement 10.6).
        last = self._last_pick.get(tag)
        choices = [t for t in templates if t != last] or templates
        template = self._rng.choice(choices)
        self._last_pick[tag] = template

        try:
            phrase = template.format(**self._placeholders(facts))
        except (KeyError, IndexError):
            # Fall back to the first placeholder-free template for this tag.
            safe = next((t for t in templates if "{" not in t), None)
            if safe is None:
                safe = _BUILTIN_FALLBACKS.get(tag, "Heads up.")
            phrase = safe
        return phrase

    def scenario_for(self, facts: GuidanceFacts) -> str:
        """Public-facing scenario classifier (used by VoiceQueue tier mapping)."""
        return self._scenario_for(facts)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _scenario_for(self, facts: GuidanceFacts) -> str:
        # Order matters — vision STOP wins; stairs+stop wins over plain stairs.
        if facts.vision_stop:
            return (
                "vision_stop_with_cause"
                if self._has_named_cause(facts)
                else "vision_stop_generic"
            )
        # Destination cue ("turn = stop") becomes "destination_reached".
        if facts.route_cue is not None and facts.route_cue.turn == "stop":
            return "destination_reached"
        # Stairs override route turns and directional warnings: a step is
        # always worth speaking even if the path otherwise looks fine.
        if facts.stairs.flag and facts.stairs.confidence >= self.settings.stairs_min_confidence:
            return (
                "stairs_warning_with_stop"
                if facts.command == NavigationCommand.STOP
                else "stairs_warning_low_conf"
            )
        if facts.command == NavigationCommand.SLOW_DOWN and self._all_lanes_blocked(facts):
            return "lane_blocked_all_sides"
        if facts.route_cue is not None and facts.route_cue.turn == "loading":
            return "map_route_loading"
        if facts.route_cue is not None and facts.route_cue.rationale.startswith("off_route"):
            return "map_off_route"
        if facts.route_cue is not None and facts.route_cue.turn in ("left", "right"):
            map_cmd = (
                NavigationCommand.MOVE_LEFT
                if facts.route_cue.turn == "left"
                else NavigationCommand.MOVE_RIGHT
            )
            if facts.command == map_cmd:
                return "map_turn_with_vision_clear"
            if facts.command in (NavigationCommand.MOVE_LEFT, NavigationCommand.MOVE_RIGHT):
                return "map_turn_only"
            return "map_turn_with_caution"
        # Route is active and we're heading the right way — give progress update.
        if facts.route_cue is not None and facts.route_cue.turn == "forward":
            return "status_update_progress"
        if facts.command in (NavigationCommand.MOVE_LEFT, NavigationCommand.MOVE_RIGHT):
            opposite = "right" if facts.command == NavigationCommand.MOVE_LEFT else "left"
            opp_walkable = facts.walkable_by_side.get(opposite, 0.0)
            return (
                "directional_warning_with_freespace"
                if opp_walkable >= self.settings.min_lane_walkable_ratio
                else "directional_warning_simple"
            )
        if any(facts.hazards_by_side.get(s, []) for s in SIDES):
            crossing = any(
                h.approach in ("crossing_left_to_right", "crossing_right_to_left")
                for s in SIDES
                for h in facts.hazards_by_side.get(s, [])
            )
            return "approach_alert_crossing" if crossing else "approach_alert_static"
        # Status update — system is going forward and nothing notable in view.
        return "status_update_clear"

    def _placeholders(self, facts: GuidanceFacts) -> dict[str, str]:
        side = self._dominant_hazard_side(facts) or "ahead"
        opposite = {
            "left": "right",
            "right": "left",
            "center": "ahead",
            "ahead": "ahead",
        }[side]
        category_list = self._category_list(facts)
        category = category_list[0] if category_list else "object"
        cat_list_str = ", ".join(category_list) if category_list else "the path"
        turn_direction = facts.route_cue.turn if facts.route_cue else "ahead"
        meters = facts.route_cue.meters_to_turn if facts.route_cue else 0.0
        feet = int(round(meters * 3.281)) if facts.route_cue else 0

        return {
            "distance_phrase": facts.distance_phrase,
            "side": side,
            "opposite_side": opposite,
            "category": category,
            "category_list": cat_list_str,
            "turn_direction": turn_direction,
            "turn_direction_cap": turn_direction.capitalize(),
            "meters_to_turn": f"{meters:.0f}",
            "feet_to_turn": f"{feet}",
        }

    def _has_named_cause(self, facts: GuidanceFacts) -> bool:
        return any(
            facts.hazards_by_side.get(s, []) for s in SIDES
        )

    def _all_lanes_blocked(self, facts: GuidanceFacts) -> bool:
        return all(
            facts.walkable_by_side.get(s, 0.0)
            < self.settings.min_lane_walkable_ratio
            for s in SIDES
        )

    def _dominant_hazard_side(self, facts: GuidanceFacts) -> str | None:
        """Pick the side with the loudest hazard. Ties → center → left → right."""
        best_side = None
        best_weight = -1.0
        for side in ("center", "left", "right"):  # preference order
            for h in facts.hazards_by_side.get(side, []):
                if h.weighted_pixels > best_weight:
                    best_weight = h.weighted_pixels
                    best_side = side
        return best_side

    def _category_list(self, facts: GuidanceFacts) -> list[str]:
        """All categories present, ranked by total weight across sides."""
        totals: dict[str, float] = {}
        for side in SIDES:
            for h in facts.hazards_by_side.get(side, []):
                totals[h.category] = totals.get(h.category, 0.0) + h.weighted_pixels
        return [c for c, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)]

    def _load(self, path: Path) -> dict[str, list[str]]:
        if not path.is_file():
            logger.warning(
                "phrases file %s missing; using built-in fallbacks", path
            )
            return {}
        try:
            with path.open(encoding="utf-8") as f:
                data: Any = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.warning("phrases file %s could not be parsed (%s); using fallbacks", path, e)
            return {}
        if not isinstance(data, dict):
            logger.warning("phrases file %s is not a mapping; using fallbacks", path)
            return {}
        out: dict[str, list[str]] = {}
        for tag, val in data.items():
            if isinstance(val, list):
                out[str(tag)] = [str(x) for x in val if isinstance(x, str) and x]
            elif isinstance(val, str):
                out[str(tag)] = [val]
        return out

    def _warn_under_populated(self) -> None:
        thin = [
            tag
            for tag, paraphrases in self._templates.items()
            if len(paraphrases) < 3
        ]
        if thin:
            logger.warning(
                "phrase tags with fewer than 3 paraphrases: %s",
                ", ".join(sorted(thin)),
            )
            self._under_populated_warned.update(thin)


__all__ = ["PhraseComposer"]
