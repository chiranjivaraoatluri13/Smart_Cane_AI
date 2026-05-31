"""Structured facts the SpatialReasoner emits and the PhraseComposer consumes.

This is the contract between the reasoning and output layers. Keeping it as
a small, well-typed dataclass means the composer can never go off-script
(no LLM, no string templating in the reasoner). It also makes the JSON
response from `/process_frame` deterministic — `summary_dict()` produces the
HUD/debug payload the phone client renders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from navigation.models import NavigationCommand, Side
from navigation.output.distance import DistanceBucket

ApproachDirection = Literal[
    "static",
    "crossing_left_to_right",
    "crossing_right_to_left",
    "closing_in",
    "receding",
]

ApproachDirectionTuple: tuple[ApproachDirection, ...] = (
    "static",
    "crossing_left_to_right",
    "crossing_right_to_left",
    "closing_in",
    "receding",
)


@dataclass(frozen=True)
class HazardEntry:
    """One category present on a Side, with its trend label."""

    category: str
    weighted_pixels: float
    approach: ApproachDirection = "static"


@dataclass(frozen=True)
class StairsResult:
    """Heuristic stairs/curb detector output (Req 14).

    Future trained model is a drop-in replacement: same dataclass, same
    `(frame, segmentation) -> StairsResult` interface.
    """

    flag: bool
    confidence: float
    rationale: str = ""


@dataclass(frozen=True)
class RouteCue:
    """Next turn-by-turn cue from MapGuidance (Req 7).

    `turn = "forward"` means continue straight; `"stop"` means we're at the
    destination. Both cases are encoded so the composer can phrase them
    naturally without inferring intent.
    """

    turn: Literal["left", "right", "forward", "stop", "loading"]
    meters_to_turn: float
    target_bearing_deg: float
    rationale: str = ""


@dataclass(frozen=True)
class GuidanceFacts:
    """Everything the PhraseComposer needs to produce one spoken phrase.

    The reasoner builds this. The composer reads it. Nothing else mutates
    it. `vision_stop=True` is the safety invariant that guarantees STOP
    overrides every other branch (Req 13).
    """

    command: NavigationCommand
    confidence: float
    hazards_by_side: dict[Side, list[HazardEntry]]
    walkable_by_side: dict[Side, float]
    approach_direction_by_category: dict[str, ApproachDirection]
    stairs: StairsResult
    distance_bucket: DistanceBucket
    distance_phrase: str
    route_cue: Optional[RouteCue]
    vision_stop: bool

    def summary_dict(self) -> dict:
        """JSON-safe view used by the HUD and `/process_frame` response."""
        return {
            "command": self.command.value,
            "vision_stop": self.vision_stop,
            "stairs": {
                "flag": self.stairs.flag,
                "confidence": float(self.stairs.confidence),
            },
            "walkable_by_side": dict(self.walkable_by_side),
            "distance_bucket": self.distance_bucket,
            "distance_phrase": self.distance_phrase,
            "route_cue": (
                None
                if self.route_cue is None
                else {
                    "turn": self.route_cue.turn,
                    "meters_to_turn": float(self.route_cue.meters_to_turn),
                }
            ),
            "hazards_by_side": {
                s: [
                    {"category": h.category, "approach": h.approach}
                    for h in self.hazards_by_side.get(s, [])
                ]
                for s in ("left", "center", "right")
            },
            "approach_direction_by_category": dict(
                self.approach_direction_by_category
            ),
        }


__all__ = [
    "ApproachDirection",
    "ApproachDirectionTuple",
    "GuidanceFacts",
    "HazardEntry",
    "RouteCue",
    "StairsResult",
]
