"""SpatialReasoner — combines per-side perception, depth, CARE, and route
into a single NavigationDecision plus the GuidanceFacts the composer needs.

Vision-STOP short-circuit (Requirement 13) is structurally enforced: the
`vision_stop` flag is computed first and clears `route_cue` from the
emitted facts, so no downstream code can accidentally pair STOP with a
turn instruction.
"""

from __future__ import annotations

from typing import Optional

from navigation.config import Settings
from navigation.maps.guidance import MapGuidance
from navigation.maps.router import (
    bearing_delta_deg,
    bearing_to_next_waypoint,
    distance_to_destination,
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
from navigation.reasoning.alerts import CATEGORY_PRIORITY, CLASS_TO_CATEGORY
from navigation.reasoning.facts import (
    ApproachDirection,
    GuidanceFacts,
    HazardEntry,
    RouteCue,
    StairsResult,
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
        or heading_deg is None
    ):
        return None

    route = map_guidance.route
    dest_dist = distance_to_destination(current_lat, current_lon, route)
    if dest_dist <= settings.route_at_dest_m:
        return RouteCue(
            turn="stop",
            meters_to_turn=dest_dist,
            target_bearing_deg=heading_deg,
            rationale=f"At destination ({dest_dist:.0f} m)",
        )

    target_bearing = bearing_to_next_waypoint(current_lat, current_lon, route)
    delta = bearing_delta_deg(target_bearing, heading_deg)
    if abs(delta) <= settings.route_bearing_align_deg:
        turn = "forward"
    elif delta < 0:
        turn = "left"
    else:
        turn = "right"
    return RouteCue(
        turn=turn,
        meters_to_turn=dest_dist,
        target_bearing_deg=target_bearing,
        rationale=(
            f"target {target_bearing:.0f}° "
            f"vs heading {heading_deg:.0f}° (delta {delta:+.0f}°)"
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
        walkable_by_side: dict[Side, float] = (
            seg.per_side_walkable_ratio
            or {"left": float(seg.walkable_ratio), "center": float(seg.walkable_ratio), "right": float(seg.walkable_ratio)}
        )

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
        elif route_cue is not None and route_cue.turn in ("left", "right"):
            target_side: Side = "left" if route_cue.turn == "left" else "right"
            target_walkable = walkable_by_side.get(target_side, 0.0)
            center_walkable = walkable_by_side.get("center", 0.0)
            if target_walkable >= max(center_walkable, self._min_lane_walkable):
                command = (
                    NavigationCommand.MOVE_LEFT
                    if target_side == "left"
                    else NavigationCommand.MOVE_RIGHT
                )
                confidence = 0.85  # High confidence in turn direction
                rationale = f"map cue: turn {target_side}, target side walkable"
            else:
                command = NavigationCommand.SLOW_DOWN
                confidence = 0.70  # Moderate confidence — path unclear
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
            # Use CARE safety score directly, but ensure it's >= 0.75 for walking
            care_score = float(care.safety_score)
            if care_score < 0.75:
                # Low safety score — convert to STOP
                command = NavigationCommand.STOP
                confidence = 1.0 - care_score  # Invert: low safety = high confidence in STOP
            else:
                confidence = care_score  # High safety = high confidence in movement
            rationale = f"CARE direction {deg:.1f}° gated by per-side walkable (safety {care_score:.2f})"

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
            # Req 7.3 / 13.3 — vision STOP drops the route cue entirely so
            # the composer cannot pair STOP with a turn instruction.
            route_cue=None if vision_stop else route_cue,
            vision_stop=vision_stop,
        )
        return decision, facts

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _vision_stop(self, seg: SegmentationResult, care: CareResult) -> bool:
        if care.hazard_detected:
            self._stop_hold_remaining = self._stop_hold_frames
            return True
        # Center-side obstacle ratio gate (Req 3.4): if obstacles dominate
        # the center third's walkable area, STOP regardless of CARE.
        per_side = seg.per_side_class_pixels or {}
        center = per_side.get("center", {}) if isinstance(per_side, dict) else {}
        ratio: float = 0.0
        if center:
            obstacle_set = self._obstacle_class_set()
            center_obstacle_weighted = sum(
                float(w) for cls, w in center.items() if cls in obstacle_set
            )
            shape = (seg.metadata or {}).get("shape", [480, 640])
            if not isinstance(shape, (list, tuple)) or len(shape) < 2:
                shape = [480, 640]
            frame_area = max(int(shape[0]) * int(shape[1]), 1)
            center_area = max(frame_area / 3.0, 1.0)
            ratio = center_obstacle_weighted / center_area

        rising_threshold = self.settings.hazard_obstacle_ratio
        falling_threshold = rising_threshold * self._hazard_release_factor

        # Sticky-stop hysteresis: once we go into STOP, stay there until the
        # ratio drops well below the threshold *and* the hold timer expires.
        # This kills STOP/SLOW_DOWN/STOP oscillation around the boundary.
        if self._stop_hold_remaining > 0:
            self._stop_hold_remaining -= 1
            if ratio >= falling_threshold:
                # Re-arm the hold so a still-noisy scene doesn't drop out
                # right after the timer ends.
                self._stop_hold_remaining = max(
                    self._stop_hold_remaining, self._stop_hold_frames // 2
                )
                return True
            # Ratio is well below the falling threshold — still hold for
            # the rest of the timer to give the user a beat to react.
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
        for side in SIDES:
            entries: dict[str, float] = {}  # category -> total weighted
            side_dict = per_side.get(side, {}) or {}
            for cls_name, weight in side_dict.items():
                category = CLASS_TO_CATEGORY.get(cls_name)
                if category is None:
                    continue
                entries[category] = entries.get(category, 0.0) + float(weight)
            ranked = sorted(
                entries.items(),
                key=lambda kv: CATEGORY_PRIORITY.get(kv[0], 0),
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

    def _obstacle_class_set(self) -> set[str]:
        try:
            seg_cfg = self.settings.seg_class_config()
        except Exception:
            seg_cfg = {}
        return set((seg_cfg or {}).get("obstacle_classes", []))


__all__ = [
    "SpatialReasoner",
    "_next_route_cue",
]
