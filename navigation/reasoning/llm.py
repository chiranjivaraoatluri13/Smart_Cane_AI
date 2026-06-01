"""Llama 3.1 decision interpretation layer."""

from __future__ import annotations

import json
import logging
import threading
import time

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

    _MAX_ROUTE_FETCH_ATTEMPTS = 3

    def __init__(self, settings: Settings):
        self.settings = settings
        self._chain = None
        self._map_guidance = None
        self._map_route_attempted = False
        self._route_fetch_lock = threading.Lock()
        self._route_fetch_started = False
        self._route_generation = 0
        self._route_fetch_failures = 0
        self._route_permanent_failure = False
        self._last_refetch_at = 0.0
        if settings.map_ready:
            self._init_map_guidance(
                settings.current_lat,  # type: ignore[arg-type]
                settings.current_lon,  # type: ignore[arg-type]
                generation=self._route_generation,
            )

    def reset_map_state(self) -> None:
        """Invalidate route state after a new destination or forced refetch."""
        with self._route_fetch_lock:
            self._route_generation += 1
            self._map_guidance = None
            self._map_route_attempted = False
            self._route_fetch_started = False
            self._route_fetch_failures = 0
            self._route_permanent_failure = False

    def is_route_loading(self) -> bool:
        with self._route_fetch_lock:
            return (
                self._route_fetch_started
                and self._map_guidance is None
                and not self._route_permanent_failure
            )

    def _init_map_guidance(
        self,
        start_lat: float,
        start_lon: float,
        *,
        generation: int,
    ) -> None:
        from navigation.maps.guidance import MapGuidance
        from navigation.maps.router import fetch_route, save_route_debug

        if self.settings.dest_lat is None or self.settings.dest_lon is None:
            logger.warning(
                "Map guidance requested but DEST_LAT/DEST_LON are not set."
            )
            with self._route_fetch_lock:
                if generation == self._route_generation:
                    self._map_route_attempted = True
                    self._route_permanent_failure = True
            return
        try:
            osrm = getattr(self.settings, "osrm_base_url", "") or None
            route = fetch_route(
                start_lat,
                start_lon,
                self.settings.dest_lat,
                self.settings.dest_lon,
                osrm_base=osrm,
                route_provider=getattr(self.settings, "route_provider", "osrm"),
                google_maps_api_key=getattr(self.settings, "google_maps_api_key", ""),
            )
            with self._route_fetch_lock:
                if generation != self._route_generation:
                    logger.info(
                        "Discarding stale route fetch (gen %d != %d).",
                        generation,
                        self._route_generation,
                    )
                    return
                self._map_guidance = MapGuidance(route, self.settings)
                self._map_route_attempted = True
                self._route_fetch_failures = 0
                self._route_permanent_failure = False
            if self.settings.route_debug_path:
                save_route_debug(self.settings.route_debug_path, route)
            logger.info(
                "Map guidance: %.0f m route, %d waypoints",
                route.distance_m,
                len(route.waypoints),
            )
        except Exception as e:
            with self._route_fetch_lock:
                if generation != self._route_generation:
                    return
                self._route_fetch_failures += 1
                if self._route_fetch_failures >= self._MAX_ROUTE_FETCH_ATTEMPTS:
                    self._map_route_attempted = True
                    self._route_permanent_failure = True
                    logger.warning(
                        "Map route unavailable after %d attempts (%s).",
                        self._route_fetch_failures,
                        e,
                    )
                else:
                    self._map_route_attempted = False
                    logger.warning(
                        "Map route fetch failed (attempt %d/%d): %s",
                        self._route_fetch_failures,
                        self._MAX_ROUTE_FETCH_ATTEMPTS,
                        e,
                    )

    def ensure_map_guidance(self, position: Position) -> None:
        """Lazily fetch the route once live GPS arrives (background thread)."""
        if not self.settings.use_map_guidance or not position.has_coords:
            return
        with self._route_fetch_lock:
            if self._map_guidance is not None:
                return
            if self._route_permanent_failure:
                return
            if self._route_fetch_started:
                return
            if self._map_route_attempted:
                return
            self._route_fetch_started = True
            generation = self._route_generation
            lat, lon = position.lat, position.lon

        assert lat is not None and lon is not None

        def _fetch() -> None:
            try:
                self._init_map_guidance(lat, lon, generation=generation)
            finally:
                with self._route_fetch_lock:
                    self._route_fetch_started = False

        threading.Thread(
            target=_fetch, daemon=True, name="route-fetch"
        ).start()

    def prefetch_map_guidance(self, start_lat: float, start_lon: float) -> None:
        """Start route fetch immediately after destination is set (background).

        Does not block /process_frame — route is ready before the user taps Start
        when the phone sends near_lat/near_lon with /set_destination.
        """
        if not self.settings.use_map_guidance:
            return
        if self.settings.dest_lat is None or self.settings.dest_lon is None:
            return
        with self._route_fetch_lock:
            if self._map_guidance is not None:
                return
            if self._route_permanent_failure:
                return
            if self._route_fetch_started:
                return
            if self._map_route_attempted:
                return
            self._route_fetch_started = True
            generation = self._route_generation

        def _fetch() -> None:
            try:
                self._init_map_guidance(start_lat, start_lon, generation=generation)
            finally:
                with self._route_fetch_lock:
                    self._route_fetch_started = False

        threading.Thread(
            target=_fetch, daemon=True, name="route-fetch-prefetch"
        ).start()

    def maybe_refetch_route(self, position: Position, cross_track_m: float) -> None:
        """Re-fetch route from current position when user is far off the polyline."""
        threshold = float(getattr(self.settings, "route_refetch_off_route_m", 45.0))
        if cross_track_m < threshold:
            return
        now = time.monotonic()
        if now - self._last_refetch_at < 60.0:
            return
        self._last_refetch_at = now
        logger.info(
            "Off route by %.0f m — refetching OSRM route from current position.",
            cross_track_m,
        )
        self.reset_map_state()
        self.ensure_map_guidance(position)

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
        if bundle.care.hazard_detected:
            return NavigationDecision(
                command=NavigationCommand.STOP,
                confidence=0.65,
                rationale="Obstacle or hazard within critical range",
            )
        return None

    def _resolve_position(self, position: Position | None) -> Position:
        """Merge live position with .env fallbacks (heading stays None if unknown)."""
        if position is None:
            position = Position()
        heading = position.heading_deg
        if heading is None:
            heading = self.settings.current_heading_deg
        return Position(
            lat=position.lat if position.lat is not None else self.settings.current_lat,
            lon=position.lon if position.lon is not None else self.settings.current_lon,
            heading_deg=heading,
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

        if (
            self._map_guidance is None
            and self.settings.use_map_guidance
            and pos.has_coords
        ):
            self.ensure_map_guidance(pos)

        if self._map_guidance is not None and pos.has_coords:
            assert pos.lat is not None and pos.lon is not None
            heading = pos.heading_deg if pos.heading_deg is not None else 0.0
            return self._map_guidance.decide(pos.lat, pos.lon, heading)

        care = bundle.care
        if care.safety_score < 0.5:
            return NavigationDecision(
                command=NavigationCommand.SLOW_DOWN,
                confidence=0.60,
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
