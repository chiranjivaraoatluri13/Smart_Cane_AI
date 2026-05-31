"""OSRM walking routes and geocoding helpers."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Public demo server — prefer HTTPS; override with OSRM_BASE_URL in production.
_DEFAULT_OSRM = "https://router.project-osrm.org/route/v1/foot"
OSRM_BASE = os.environ.get("OSRM_BASE_URL", _DEFAULT_OSRM).rstrip("/")
NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "assistive-navigation/0.1 (MVP; local demo)"


@dataclass(frozen=True)
class RoutePlan:
    """Walking route as (lat, lon) waypoints."""

    waypoints: list[tuple[float, float]]
    distance_m: float
    duration_s: float

    @property
    def destination(self) -> tuple[float, float]:
        return self.waypoints[-1]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "waypoints": [{"lat": lat, "lon": lon} for lat, lon in self.waypoints],
            "distance_m": self.distance_m,
            "duration_s": self.duration_s,
        }


def osrm_base_url(override: str | None = None) -> str:
    """Active OSRM foot-routing base URL."""
    if override and override.strip():
        return override.strip().rstrip("/")
    return OSRM_BASE


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def distance_meters(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    return _haversine_m(lat1, lon1, lat2, lon2)


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing from point 1 to 2, degrees 0–360 (0 = north)."""
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(
        lat2r
    ) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def bearing_delta_deg(target_bearing: float, current_heading: float) -> float:
    """Signed shortest turn from current heading to target, in [-180, 180]."""
    return (target_bearing - current_heading + 540.0) % 360.0 - 180.0


def fetch_route(
    start_lat: float,
    start_lon: float,
    dest_lat: float,
    dest_lon: float,
    *,
    client: httpx.Client | None = None,
    osrm_base: str | None = None,
    max_attempts: int = 3,
) -> RoutePlan:
    """Fetch a foot route from OSRM."""
    base = osrm_base_url(osrm_base)
    url = (
        f"{base}/{start_lon},{start_lat};{dest_lon},{dest_lat}"
        "?overview=full&geometries=geojson&steps=false"
    )
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=30.0)

    last_err: Exception | None = None
    try:
        for attempt in range(max(1, max_attempts)):
            try:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
                break
            except (httpx.HTTPError, ValueError) as e:
                last_err = e
                if attempt + 1 < max_attempts:
                    time.sleep(0.5 * (attempt + 1))
        else:
            raise last_err or RuntimeError("OSRM fetch failed")
    finally:
        if own_client:
            client.close()

    if data.get("code") != "Ok" or not data.get("routes"):
        raise ValueError(f"OSRM route failed: {data.get('message', data)}")

    route = data["routes"][0]
    coords = route["geometry"]["coordinates"]
    waypoints = [(float(c[1]), float(c[0])) for c in coords]
    if len(waypoints) < 2:
        waypoints = [(start_lat, start_lon), (dest_lat, dest_lon)]
    return RoutePlan(
        waypoints=waypoints,
        distance_m=float(route.get("distance", 0)),
        duration_s=float(route.get("duration", 0)),
    )


def fetch_route_from_json(data: dict[str, Any]) -> RoutePlan:
    """Build RoutePlan from OSRM-like JSON (for tests)."""
    route = data["routes"][0]
    coords = route["geometry"]["coordinates"]
    waypoints = [(float(c[1]), float(c[0])) for c in coords]
    return RoutePlan(
        waypoints=waypoints,
        distance_m=float(route.get("distance", 0)),
        duration_s=float(route.get("duration", 0)),
    )


@dataclass(frozen=True)
class PlaceSuggestion:
    """One Nominatim search hit for destination autocomplete."""

    display_name: str
    lat: float
    lon: float


def search_places(
    query: str,
    *,
    limit: int = 5,
    client: httpx.Client | None = None,
) -> list[PlaceSuggestion]:
    """Return address suggestions for autocomplete (Nominatim)."""
    q = query.strip()
    if len(q) < 2:
        return []
    params = {"q": q, "format": "json", "limit": max(1, min(limit, 10))}
    headers = {"User-Agent": USER_AGENT}
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=15.0, headers=headers)
    try:
        resp = client.get(NOMINATIM_SEARCH, params=params, headers=headers)
        resp.raise_for_status()
        rows = resp.json()
    finally:
        if own_client:
            client.close()
    out: list[PlaceSuggestion] = []
    for row in rows:
        try:
            out.append(
                PlaceSuggestion(
                    display_name=str(row["display_name"]),
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def geocode_address(
    address: str,
    *,
    client: httpx.Client | None = None,
) -> tuple[float, float]:
    """Resolve an address to (lat, lon) via Nominatim."""
    params = {"q": address, "format": "json", "limit": 1}
    headers = {"User-Agent": USER_AGENT}
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=20.0, headers=headers)
    try:
        resp = client.get(NOMINATIM_SEARCH, params=params, headers=headers)
        resp.raise_for_status()
        rows = resp.json()
    finally:
        if own_client:
            client.close()
    if not rows:
        raise ValueError(f"No geocode result for: {address!r}")
    return float(rows[0]["lat"]), float(rows[0]["lon"])


def _segment_projection_fraction(
    lat: float,
    lon: float,
    a_lat: float,
    a_lon: float,
    b_lat: float,
    b_lon: float,
) -> float:
    """Fraction t in [0,1] along segment A→B closest to point P."""
    ref_lat = (a_lat + b_lat) / 2.0
    cos_lat = math.cos(math.radians(ref_lat))
    ax, ay = a_lon * cos_lat, a_lat
    bx, by = b_lon * cos_lat, b_lat
    px, py = lon * cos_lat, lat
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom < 1e-18:
        return 0.0
    return max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))


def _nearest_segment_index(
    lat: float, lon: float, waypoints: list[tuple[float, float]]
) -> tuple[int, float]:
    """Index of segment start and cross-track distance to polyline (meters)."""
    if len(waypoints) < 2:
        return 0, distance_meters(lat, lon, waypoints[0][0], waypoints[0][1])

    best_dist = float("inf")
    best_i = 0
    for i in range(len(waypoints) - 1):
        a_lat, a_lon = waypoints[i]
        b_lat, b_lon = waypoints[i + 1]
        t = _segment_projection_fraction(lat, lon, a_lat, a_lon, b_lat, b_lon)
        mid_lat = a_lat + t * (b_lat - a_lat)
        mid_lon = a_lon + t * (b_lon - a_lon)
        d = _haversine_m(lat, lon, mid_lat, mid_lon)
        if d < best_dist:
            best_dist = d
            best_i = i
    return best_i, best_dist


def bearing_to_next_waypoint(
    current_lat: float,
    current_lon: float,
    route: RoutePlan,
) -> float:
    """Bearing (degrees) toward the next route vertex ahead of the user."""
    _, _, bearing = next_waypoint_ahead(current_lat, current_lon, route)
    return bearing


def next_waypoint_ahead(
    current_lat: float,
    current_lon: float,
    route: RoutePlan,
) -> tuple[int, float, float]:
    """Next route vertex index, distance to it (m), and bearing toward it."""
    seg_i, _ = _nearest_segment_index(current_lat, current_lon, route.waypoints)
    target_i = min(seg_i + 1, len(route.waypoints) - 1)
    t_lat, t_lon = route.waypoints[target_i]
    dist_to_target = distance_meters(current_lat, current_lon, t_lat, t_lon)
    if dist_to_target < 5.0 and target_i < len(route.waypoints) - 1:
        target_i = min(target_i + 1, len(route.waypoints) - 1)
        t_lat, t_lon = route.waypoints[target_i]
        dist_to_target = distance_meters(current_lat, current_lon, t_lat, t_lon)
    return target_i, dist_to_target, bearing_deg(current_lat, current_lon, t_lat, t_lon)


def cross_track_distance_m(
    current_lat: float,
    current_lon: float,
    route: RoutePlan,
) -> float:
    _, dist = _nearest_segment_index(
        current_lat, current_lon, route.waypoints
    )
    return dist


def distance_to_destination(
    current_lat: float,
    current_lon: float,
    route: RoutePlan,
) -> float:
    d_lat, d_lon = route.destination
    return distance_meters(current_lat, current_lon, d_lat, d_lon)


def save_route_debug(path: str, route: RoutePlan) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(route.to_json_dict(), f, indent=2)


def side_of_route(
    current_lat: float,
    current_lon: float,
    route: RoutePlan,
) -> float:
    """Signed cross-track side in local meters: negative = left, positive = right."""
    seg_i, _ = _nearest_segment_index(current_lat, current_lon, route.waypoints)
    a_lat, a_lon = route.waypoints[seg_i]
    b_lat, b_lon = route.waypoints[min(seg_i + 1, len(route.waypoints) - 1)]
    ref_lat = (a_lat + b_lat + current_lat) / 3.0
    ref_lon = (a_lon + b_lon + current_lon) / 3.0
    cos_lat = math.cos(math.radians(ref_lat))

    def _local(lat: float, lon: float) -> tuple[float, float]:
        return (
            (lon - ref_lon) * cos_lat * 111_320.0,
            (lat - ref_lat) * 110_540.0,
        )

    ax, ay = _local(a_lat, a_lon)
    bx, by = _local(b_lat, b_lon)
    px, py = _local(current_lat, current_lon)
    sx, sy = bx - ax, by - ay
    ux, uy = px - ax, py - ay
    return sx * uy - sy * ux
