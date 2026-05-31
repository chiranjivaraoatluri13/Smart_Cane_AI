"""Merge map route bearing with navigation commands."""

from __future__ import annotations

from navigation.config import Settings
from navigation.maps.router import (
    RoutePlan,
    bearing_delta_deg,
    bearing_to_next_waypoint,
    cross_track_distance_m,
    distance_to_destination,
    side_of_route,
)
from navigation.models import NavigationCommand, NavigationDecision


class MapGuidance:
    """Turn-by-turn style commands from a walking route (no compass required for off-route)."""

    def __init__(self, route: RoutePlan, settings: Settings):
        self.route = route
        self.settings = settings

    def decide(
        self,
        current_lat: float,
        current_lon: float,
        heading_deg: float,
    ) -> NavigationDecision:
        at_dest_m = self.settings.route_at_dest_m
        off_route_m = self.settings.route_off_route_m
        align_deg = self.settings.route_bearing_align_deg

        dest_dist = distance_to_destination(current_lat, current_lon, self.route)
        if dest_dist <= at_dest_m:
            return NavigationDecision(
                command=NavigationCommand.STOP,
                confidence=0.9,
                rationale=f"At destination ({dest_dist:.0f} m)",
            )

        cross = cross_track_distance_m(current_lat, current_lon, self.route)
        if cross > off_route_m:
            side = side_of_route(current_lat, current_lon, self.route)
            if side >= 0:
                cmd = NavigationCommand.MOVE_LEFT
                hint = "left toward route"
            else:
                cmd = NavigationCommand.MOVE_RIGHT
                hint = "right toward route"
            return NavigationDecision(
                command=cmd,
                confidence=0.8,
                rationale=f"Off route by {cross:.0f} m — move {hint}",
            )

        target_bearing = bearing_to_next_waypoint(
            current_lat, current_lon, self.route
        )
        delta = bearing_delta_deg(target_bearing, heading_deg)
        if abs(delta) <= align_deg:
            return NavigationDecision(
                command=NavigationCommand.GO_FORWARD,
                confidence=0.90,  # High confidence — on route and aligned
                rationale=(
                    f"On route, bearing {target_bearing:.0f}° "
                    f"({dest_dist:.0f} m to destination)"
                ),
            )
        if delta < 0:
            cmd = NavigationCommand.MOVE_LEFT
            turn = "left"
        else:
            cmd = NavigationCommand.MOVE_RIGHT
            turn = "right"
        return NavigationDecision(
            command=cmd,
            confidence=0.80,  # High confidence — clear turn direction
            rationale=(
                f"Turn {turn} toward path "
                f"(target {target_bearing:.0f}°, you face {heading_deg:.0f}°)"
            ),
        )
