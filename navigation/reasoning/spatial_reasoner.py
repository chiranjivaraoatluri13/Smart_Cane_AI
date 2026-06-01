"""SpatialReasoner — combines per-side perception, depth, CARE, and route
into a single NavigationDecision plus the GuidanceFacts the composer needs.

Vision-STOP short-circuit (Requirement 13) is structurally enforced: the
`vision_stop` flag is computed first; the composer prioritises it for speech.
`route_cue` remains in facts for HUD turn display even during vision STOP.
"""

from __future__ import annotations

from typing import Optional

from navigation.config import Settings
from navigation.maps.guidance import MapGuidance
from navigation.maps.router import (
    bearing_delta_deg,
    cross_track_distance_m,
    distance_to_destination,
    next_waypoint_ahead,
    route_cue_distance_m,
    side_of_route,
)
from navigation.models import (
    CareResult,
    DepthResult,
    NavigationCommand,
    NavigationDecision,
    SegmentationResult,
    SIDES,
    Side,
)
from navigation.output.distance import DistanceConfig, bucketize, load_distance_config
from navigation.reasoning.hazard_labels import hazard_key_for_class, hazard_priority
from navigation.reasoning.facts import (
    ApproachDirection,
    GuidanceFacts,
    HazardEntry,
    RouteCue,
    StairsResult,
)
from navigation.reasoning.mask_metrics import (
    center_obstacle_ratio,
    walkable_by_side as _walkable_by_side_from_seg,
)


def _next_route_cue(
    map_guidance: MapGuidance | None,
    settings: Settings,
    *,
    current_lat: float | None,
    current_lon: float | None,
    heading_deg: float | None,
) -> Optional[RouteCue]:
    """Pull the next turn cue from MapGuidance without speaking it.

    Returns None when there is no active route or no GPS fix.
    """
    if (
        map_guidance is None
        or current_lat is None
        or current_lon is None
    ):
        return None

    route = map_guidance.route
    dest_dist = distance_to_destination(current_lat, current_lon, route)
    near_dest_m = float(getattr(settings, "route_near_dest_m", 80.0))

    cross = cross_track_distance_m(current_lat, current_lon, route)
    if cross > settings.route_off_route_m:
        side = side_of_route(current_lat, current_lon, route)
        turn = "left" if side >= 0 else "right"
        return RouteCue(
            turn=turn,
            meters_to_turn=cross,
            target_bearing_deg=0.0,
            rationale=f"off_route: {cross:.0f} m — rejoin {turn}",
        )

    if dest_dist <= settings.route_at_dest_m:
        return RouteCue(
            turn="stop",
            meters_to_turn=dest_dist,
            target_bearing_deg=heading_deg or 0.0,
            rationale=f"At destination ({dest_dist:.0f} m)",
        )

    _, dist_to_next, target_bearing = next_waypoint_ahead(
        current_lat, current_lon, route, near_dest_m=near_dest_m
    )
    effective_heading = heading_deg
    if effective_heading is None:
        effective_heading = settings.current_heading_deg
    if effective_heading is None:
        effective_heading = target_bearing

    delta = bearing_delta_deg(target_bearing, effective_heading)
    if abs(delta) <= settings.route_bearing_align_deg:
        turn = "forward"
    elif delta < 0:
        turn = "left"
    else:
        turn = "right"
    cue_dist = route_cue_distance_m(
        turn,
        dest_dist,
        dist_to_next,
        near_dest_m=near_dest_m,
    )
    return RouteCue(
        turn=turn,
        meters_to_turn=cue_dist,
        target_bearing_deg=target_bearing,
        rationale=(
            f"target {target_bearing:.0f}° "
            f"vs heading {effective_heading:.0f}° (delta {delta:+.0f}°)"
        ),
    )


class SpatialReasoner:
    """Combines all signals into one decision + the facts to phrase it."""

    def __init__(self, settings: Settings):
        self.settings = settings
        try:
            yaml_cfg = settings.yaml_config()
        except Exception:
            yaml_cfg = {}
        self._distance_cfg: DistanceConfig = load_distance_config(yaml_cfg)
        self._min_lane_walkable = float(settings.min_lane_walkable_ratio)
        self._min_center_walkable = float(
            getattr(settings, "min_center_walkable_for_forward", 0.18)
        )
        # Hysteresis state to stop oscillating around the hazard threshold.
        # Once vision_stop fires, it stays sticky for ``stop_hold_frames``
        # subsequent frames so we don't flip STOP/SLOW_DOWN/STOP/SLOW_DOWN
        # when the obstacle pixel count is hovering right at the boundary.
        self._stop_hold_remaining: int = 0
        self._stop_hold_frames: int = max(0, int(getattr(settings, "stop_hold_frames", 8)))
        # Hazard ratio gets a hysteresis band: rising edge (start STOP) uses
        # the configured ratio; falling edge (release STOP) needs to drop to
        # ratio * release_factor to be considered "clear".
        self._hazard_release_factor: float = 0.6

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        seg: SegmentationResult,
        depth: DepthResult,
        care: CareResult,
        route_cue: Optional[RouteCue],
        *,
        stairs: StairsResult,
        approach_by_category: dict[str, ApproachDirection] | None = None,
    ) -> tuple[NavigationDecision, GuidanceFacts]:
        approach_by_category = approach_by_category or {}

        # (a) Vision STOP — checked FIRST so nothing below can override.
        vision_stop = self._vision_stop(seg, care)

        # (b) Per-side hazards (Req 3.1, 3.2, 3.5).
        hazards_by_side = self._hazards_by_side(seg, approach_by_category)

        # (c) Per-side walkable ratios — fall back to global if missing.
        walkable_by_side: dict[Side, float] = _walkable_by_side_from_seg(seg)

        # (d) Distance bucket from the depth proxy.
        bucket, distance_phrase = bucketize(depth.obstacle_depth_m, self._distance_cfg)

        # (e) Command selection — strict ordering, vision_stop dominates.
        # NOTE: Confidence threshold is 0.75. Commands with confidence >= 0.75 are safe to execute.
        # Commands with confidence < 0.75 should result in STOP (safety priority).
        if vision_stop:
            command = NavigationCommand.STOP
            confidence = 0.65  # Low confidence = hazard detected, must stop
            rationale = "vision_stop: hazard or center-side obstacle ratio exceeded"
        elif route_cue is not None and route_cue.turn == "stop":
            command = NavigationCommand.STOP
            confidence = 0.95  # At destination — very confident in STOP
            rationale = f"At destination ({route_cue.meters_to_turn:.0f} m)"
        elif route_cue is not None and route_cue.turn == "loading":
            command = NavigationCommand.GO_FORWARD
            confidence = 0.75
            rationale = "route loading"
        elif route_cue is not None and route_cue.turn == "failed":
            command = NavigationCommand.GO_FORWARD
            confidence = 0.75
            rationale = "route fetch failed"
        elif route_cue is not None and route_cue.turn == "forward":
            command = NavigationCommand.GO_FORWARD
            confidence = 0.85
            rationale = f"on route ({route_cue.meters_to_turn:.0f} m remaining)"
        elif route_cue is not None and route_cue.turn in ("left", "right"):
            target_side: Side = "left" if route_cue.turn == "left" else "right"
            target_walkable = walkable_by_side.get(target_side, 0.0)
            center_walkable = walkable_by_side.get("center", 0.0)
            off_route = route_cue.rationale.startswith("off_route")
            map_near = route_cue.meters_to_turn <= 45.0
            min_walk = self._min_lane_walkable * (0.35 if (off_route or map_near) else 1.0)
            if off_route or target_walkable >= max(
                center_walkable * 0.45, min_walk
            ):
                command = (
                    NavigationCommand.MOVE_LEFT
                    if target_side == "left"
                    else NavigationCommand.MOVE_RIGHT
                )
                confidence = 0.85
                rationale = (
                    f"map cue: rejoin route {target_side}"
                    if off_route
                    else f"map cue: turn {target_side}"
                )
            else:
                command = NavigationCommand.SLOW_DOWN
                confidence = 0.70
                rationale = (
                    f"map says turn {target_side} but {target_side} "
                    f"walkable {target_walkable:.2f} < center {center_walkable:.2f}"
                )
        elif self._all_lanes_blocked(walkable_by_side):
            # Req 2.5 — every side below min_lane_walkable_ratio.
            command = NavigationCommand.SLOW_DOWN
            confidence = 0.60  # Low confidence — all lanes blocked
            rationale = "no walkable lane on any side"
        else:
            # CARE-direction fallback gated by per-side walkable.
            deg = float(care.safe_direction_deg or 0.0)
            if deg < -10 and walkable_by_side.get("left", 0.0) >= walkable_by_side.get(
                "center", 0.0
            ):
                command = NavigationCommand.MOVE_LEFT
            elif deg > 10 and walkable_by_side.get("right", 0.0) >= walkable_by_side.get(
                "center", 0.0
            ):
                command = NavigationCommand.MOVE_RIGHT
            else:
                command = NavigationCommand.GO_FORWARD
            care_score = float(care.safety_score)
            confidence = max(care_score, 0.75)
            rationale = f"CARE direction {deg:.1f}° gated by per-side walkable (safety {care_score:.2f})"

        # Map-aligned movement should not be downgraded to STOP by the generic
        # confidence gate — only vision/CARE hazards should halt walking.
        map_active = route_cue is not None and route_cue.turn in (
            "forward", "left", "right", "loading"
        )
        if (
            confidence < 0.75
            and command != NavigationCommand.STOP
            and not map_active
            and (vision_stop or care.hazard_detected)
        ):
            command = NavigationCommand.STOP
            confidence = 1.0 - confidence
            rationale = (
                "Hazard detected with confidence below 0.75 — safety priority"
            )
        
        decision = NavigationDecision(
            command=command, confidence=confidence, rationale=rationale
        )

        facts = GuidanceFacts(
            command=command,
            confidence=confidence,
            hazards_by_side=hazards_by_side,
            walkable_by_side=walkable_by_side,
            approach_direction_by_category=dict(approach_by_category),
            stairs=stairs,
            distance_bucket=bucket,
            distance_phrase=distance_phrase,
            # Keep route_cue for HUD / upcoming-turn display. The composer
            # still prioritises vision_stop for spoken phrases (Req 13).
            route_cue=route_cue,
            vision_stop=vision_stop,
        )
        return decision, facts

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _vision_stop(self, seg: SegmentationResult, care: CareResult) -> bool:
        walkable = _walkable_by_side_from_seg(seg)
        center_walk = walkable.get("center", 0.0)
        path_clear = (
            center_walk >= self._min_center_walkable
            or float(seg.walkable_ratio) >= self._min_center_walkable
        )

        if care.hazard_detected and not path_clear:
            self._stop_hold_remaining = self._stop_hold_frames
            return True

        obstacle_set = self._obstacle_class_set()
        ratio = center_obstacle_ratio(seg, obstacle_set)

        rising_threshold = self.settings.hazard_obstacle_ratio
        falling_threshold = rising_threshold * self._hazard_release_factor

        # Clear center lane — release sticky STOP early and never start one.
        if path_clear:
            if self._stop_hold_remaining > 0:
                self._stop_hold_remaining = 0
            return False

        # Sticky-stop hysteresis around the center obstacle ratio.
        if self._stop_hold_remaining > 0:
            self._stop_hold_remaining -= 1
            if ratio >= falling_threshold:
                self._stop_hold_remaining = max(
                    self._stop_hold_remaining, self._stop_hold_frames // 2
                )
                return True
            if path_clear:
                self._stop_hold_remaining = 0
                return False
            return True

        if ratio >= rising_threshold:
            self._stop_hold_remaining = self._stop_hold_frames
            return True
        return False

    def _hazards_by_side(
        self,
        seg: SegmentationResult,
        approach_by_category: dict[str, ApproachDirection],
    ) -> dict[Side, list[HazardEntry]]:
        out: dict[Side, list[HazardEntry]] = {s: [] for s in SIDES}
        per_side = seg.per_side_class_pixels or {}
        if not isinstance(per_side, dict):
            return out
        obstacle_set = self._obstacle_class_set()
        hazard_set = self._hazard_class_set()
        for side in SIDES:
            entries: dict[str, float] = {}  # spoken label -> total weighted
            side_dict = per_side.get(side, {}) or {}
            for cls_name, weight in side_dict.items():
                category = hazard_key_for_class(
                    cls_name, obstacle_set, hazard_set
                )
                if category is None:
                    continue
                entries[category] = entries.get(category, 0.0) + float(weight)
            ranked = sorted(
                entries.items(),
                key=lambda kv: hazard_priority(kv[0]),
                reverse=True,
            )
            out[side] = [
                HazardEntry(
                    category=cat,
                    weighted_pixels=weighted,
                    approach=approach_by_category.get(cat, "static"),
                )
                for cat, weighted in ranked
                if weighted > 0
            ]
        return out

    def _all_lanes_blocked(self, walkable_by_side: dict[Side, float]) -> bool:
        return all(
            walkable_by_side.get(s, 0.0) < self._min_lane_walkable for s in SIDES
        )

    def _seg_class_cfg(self) -> dict:
        try:
            return self.settings.seg_class_config() or {}
        except Exception:
            return {}

    def _obstacle_class_set(self) -> set[str]:
        return set(self._seg_class_cfg().get("obstacle_classes", []))

    def _hazard_class_set(self) -> set[str]:
        return set(self._seg_class_cfg().get("hazard_classes", []))


__all__ = [
    "SpatialReasoner",
    "_next_route_cue",
]
