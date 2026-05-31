"""Llama 3.1 decision interpretation layer."""

from __future__ import annotations

import json
import logging

from navigation.config import Settings
from navigation.models import (
    CareResult,
    DepthResult,
    NavigationCommand,
    NavigationDecision,
    PerceptionBundle,
    Position,
    SegmentationResult,
)

logger = logging.getLogger(__name__)


class NavigationInterpreter:
    """Converts structured perception into a NavigationDecision JSON."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._chain = None
        self._map_guidance = None
        self._map_route_attempted = False
        # Static-config map init: only runs if both start AND destination are
        # in .env. The phone path uses ``ensure_map_guidance`` to lazily
        # initialize a route the first time live coordinates arrive.
        if settings.map_ready:
            self._init_map_guidance(
                settings.current_lat,  # type: ignore[arg-type]
                settings.current_lon,  # type: ignore[arg-type]
            )

    def _init_map_guidance(self, start_lat: float, start_lon: float) -> None:
        from navigation.maps.guidance import MapGuidance
        from navigation.maps.router import fetch_route, save_route_debug

        if self.settings.dest_lat is None or self.settings.dest_lon is None:
            logger.warning(
                "Map guidance requested but DEST_LAT/DEST_LON are not set."
            )
            self._map_route_attempted = True
            return
        try:
            route = fetch_route(
                start_lat,
                start_lon,
                self.settings.dest_lat,
                self.settings.dest_lon,
            )
            self._map_guidance = MapGuidance(route, self.settings)
            if self.settings.route_debug_path:
                save_route_debug(self.settings.route_debug_path, route)
                logger.info(
                    "Saved route debug to %s", self.settings.route_debug_path
                )
            logger.info(
                "Map guidance: %.0f m route, %d waypoints",
                route.distance_m,
                len(route.waypoints),
            )
        except Exception as e:
            logger.warning("Map route unavailable (%s); using vision heuristics.", e)
        finally:
            self._map_route_attempted = True

    def ensure_map_guidance(self, position: Position) -> None:
        """Lazily fetch the route once live GPS arrives.

        Called by the phone server on the first request that carries
        coordinates. After this returns, ``self._map_guidance`` is either
        set (route fetched) or remains ``None`` (route fetch failed; vision
        heuristics will continue as fallback).
        """
        if (
            self._map_guidance is not None
            or self._map_route_attempted
            or not self.settings.use_map_guidance
            or not position.has_coords
        ):
            return
        assert position.lat is not None and position.lon is not None
        self._init_map_guidance(position.lat, position.lon)

    def interpret(
        self,
        bundle: PerceptionBundle,
        *,
        position: Position | None = None,
    ) -> NavigationDecision:
        if not self.settings.use_llm:
            return self._heuristic(bundle, position=position)
        return self._llm(bundle, position=position)

    def _obstacle_stop(self, bundle: PerceptionBundle) -> NavigationDecision | None:
        # Only trust CARE's hazard signal. The depth-based shortcut used to
        # also force STOP when ``obstacle_depth_m < 0.9``, but depth is still
        # a mock and that rule produced false STOPs on bright walls. Re-add
        # the depth check once UniDepthV2 is actually wired.
        if bundle.care.hazard_detected:
            return NavigationDecision(
                command=NavigationCommand.STOP,
                confidence=0.85,
                rationale="Obstacle or hazard within critical range",
            )
        return None

    def _resolve_position(self, position: Position | None) -> Position:
        """Merge live position with .env fallbacks."""
        if position is None:
            position = Position()
        return Position(
            lat=position.lat if position.lat is not None else self.settings.current_lat,
            lon=position.lon if position.lon is not None else self.settings.current_lon,
            heading_deg=(
                position.heading_deg
                if position.heading_deg is not None
                else self.settings.current_heading_deg
            ),
            accuracy_m=position.accuracy_m,
        )

    def _heuristic(
        self,
        bundle: PerceptionBundle,
        *,
        position: Position | None = None,
    ) -> NavigationDecision:
        blocked = self._obstacle_stop(bundle)
        if blocked is not None:
            return blocked

        pos = self._resolve_position(position)

        # Lazy route fetch on first GPS sample (phone-as-camera path).
        if (
            self._map_guidance is None
            and self.settings.use_map_guidance
            and pos.has_coords
        ):
            self.ensure_map_guidance(pos)

        if self._map_guidance is not None and pos.has_coords:
            assert pos.lat is not None and pos.lon is not None
            heading = (
                pos.heading_deg
                if pos.heading_deg is not None
                else self.settings.current_heading_deg
            )
            return self._map_guidance.decide(pos.lat, pos.lon, heading)

        care = bundle.care
        if care.safety_score < 0.5:
            return NavigationDecision(
                command=NavigationCommand.SLOW_DOWN,
                confidence=0.75,
                rationale="Reduced safety score",
            )
        deg = care.safe_direction_deg or 0.0
        if deg < -10:
            cmd = NavigationCommand.MOVE_LEFT
        elif deg > 10:
            cmd = NavigationCommand.MOVE_RIGHT
        else:
            cmd = NavigationCommand.GO_FORWARD
        return NavigationDecision(
            command=cmd,
            confidence=care.safety_score,
            rationale=f"CARE direction {deg:.1f}°",
        )

    def _llm(
        self,
        bundle: PerceptionBundle,
        *,
        position: Position | None = None,
    ) -> NavigationDecision:
        # Hazard short-circuit: if CARE already flagged a STOP, don't waste
        # an LLM round-trip on it.
        blocked = self._obstacle_stop(bundle)
        if blocked is not None:
            return blocked

        if (
            not self.settings.openai_api_key
            and "127.0.0.1" not in self.settings.openai_api_base
        ):
            return self._heuristic(bundle, position=position)

        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
        from pydantic import BaseModel, Field

        class LlmNavOutput(BaseModel):
            command: NavigationCommand
            confidence: float = Field(ge=0.0, le=1.0)
            rationale: str = ""

        if self._chain is None:
            llm = ChatOpenAI(
                model=self.settings.openai_model,
                api_key=self.settings.openai_api_key or "ollama",
                base_url=self.settings.openai_api_base,
                temperature=0,
            )
            structured = llm.with_structured_output(LlmNavOutput)
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an assistive navigation interpreter. "
                        "Given perception summaries, output exactly one command from: "
                        "move_left, move_right, go_forward, slow_down, stop. "
                        "Prefer stop when hazard_detected or very close obstacles.",
                    ),
                    ("human", "{context}"),
                ]
            )
            self._chain = prompt | structured

        context = json.dumps(
            {
                "segmentation": bundle.segmentation.model_dump(
                    exclude={"masks", "depth_map"}
                ),
                "depth": bundle.depth.model_dump(exclude={"depth_map"}),
                "care": bundle.care.model_dump(),
            },
            default=str,
        )
        try:
            out: LlmNavOutput = self._chain.invoke({"context": context})
        except Exception as e:
            logger.warning("LLM unavailable (%s); using heuristic commands.", e)
            decision = self._heuristic(bundle, position=position)
            return decision.model_copy(
                update={
                    "rationale": f"{decision.rationale} (LLM fallback: {e})",
                }
            )
        logger.info(
            "LLM: %s (%.0f%%) — %s",
            out.command.value,
            out.confidence * 100,
            out.rationale,
        )
        return NavigationDecision(
            command=out.command,
            confidence=out.confidence,
            rationale=out.rationale,
        )
