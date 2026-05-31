"""OSRM walking routes and geocoding helpers."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Public demo server — prefer HTTPS; override with OSRM_BASE_URL in production.
_DEFAULT_OSRM = "https://router.project-osrm.org/route/v1/foot"
OSRM_BASE = os.environ.get("OSRM_BASE_URL", _DEFAULT_OSRM).rstrip("/")
NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE = "https://nominatim.openstreetmap.org/reverse"
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
    distance_m: float | None = None


def _normalize_search_query(query: str) -> str:
    """Turn 'subway, tempe' into 'subway tempe' for Nominatim."""
    q = re.sub(r"\s+", " ", query.strip())
    q = re.sub(r"\s*,\s*", " ", q)
    return q.strip()


def _viewbox_around(
    lat: float, lon: float, radius_km: float
) -> tuple[float, float, float, float]:
    """Nominatim viewbox: min_lon, max_lat, max_lon, min_lat."""
    cos_lat = max(math.cos(math.radians(lat)), 0.01)
    dlat = radius_km / 110_540.0
    dlon = radius_km / (111_320.0 * cos_lat)
    return (lon - dlon, lat + dlat, lon + dlon, lat - dlat)


def _nominatim_get(
    url: str,
    params: dict[str, str | int | float],
    *,
    client: httpx.Client,
) -> list[dict[str, Any]]:
    headers = {"User-Agent": USER_AGENT}
    resp = client.get(url, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def _reverse_locality(
    lat: float,
    lon: float,
    *,
    client: httpx.Client,
) -> str | None:
    """Best-effort city/town label for biasing vague POI searches."""
    rows = _nominatim_get(
        NOMINATIM_REVERSE,
        {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "zoom": 12,
            "addressdetails": 1,
        },
        client=client,
    )
    if not rows:
        return None
    address = rows[0].get("address") or {}
    if not isinstance(address, dict):
        return None
    parts: list[str] = []
    for key in ("city", "town", "village", "municipality", "county"):
        val = address.get(key)
        if val and str(val) not in parts:
            parts.append(str(val))
    state = address.get("state")
    if state and str(state) not in parts:
        parts.append(str(state))
    return ", ".join(parts) if parts else None


def _rows_to_suggestions(
    rows: list[dict[str, Any]],
    *,
    near_lat: float | None = None,
    near_lon: float | None = None,
) -> list[PlaceSuggestion]:
    out: list[PlaceSuggestion] = []
    for row in rows:
        try:
            lat = float(row["lat"])
            lon = float(row["lon"])
            dist = (
                distance_meters(near_lat, near_lon, lat, lon)
                if near_lat is not None and near_lon is not None
                else None
            )
            out.append(
                PlaceSuggestion(
                    display_name=str(row["display_name"]),
                    lat=lat,
                    lon=lon,
                    distance_m=dist,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    if near_lat is not None and near_lon is not None:
        out.sort(key=lambda p: p.distance_m if p.distance_m is not None else float("inf"))
    return out


def search_places(
    query: str,
    *,
    limit: int = 5,
    near_lat: float | None = None,
    near_lon: float | None = None,
    radius_km: float = 25.0,
    client: httpx.Client | None = None,
) -> list[PlaceSuggestion]:
    """Return address suggestions for autocomplete (Nominatim).

    When ``near_lat``/``near_lon`` are set, results are biased to a local
    viewbox and ranked by distance — needed for vague queries like "subway".
    """
    q = _normalize_search_query(query)
    if len(q) < 2:
        return []

    fetch_limit = max(limit * 2, 10)
    headers = {"User-Agent": USER_AGENT}
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=15.0, headers=headers)

    seen: set[tuple[float, float]] = set()
    merged: list[PlaceSuggestion] = []

    def _collect(rows: list[dict[str, Any]]) -> None:
        for item in _rows_to_suggestions(
            rows, near_lat=near_lat, near_lon=near_lon
        ):
            key = (round(item.lat, 5), round(item.lon, 5))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)

    def _search(params: dict[str, str | int | float]) -> None:
        if len(merged) >= fetch_limit:
            return
        base = {
            "format": "json",
            "limit": fetch_limit,
            "addressdetails": 0,
        }
        base.update(params)
        rows = _nominatim_get(NOMINATIM_SEARCH, base, client=client)
        _collect(rows)

    try:
        if near_lat is not None and near_lon is not None:
            min_lon, max_lat, max_lon, min_lat = _viewbox_around(
                near_lat, near_lon, radius_km
            )
            viewbox = f"{min_lon},{max_lat},{max_lon},{min_lat}"

            # Strict local search first — best for "subway", "starbucks", etc.
            _search({"q": q, "viewbox": viewbox, "bounded": 1})

            # Same viewbox but allow matches outside if nothing local.
            if len(merged) < limit:
                _search({"q": q, "viewbox": viewbox, "bounded": 0})

            # Append city/state from reverse geocode when query has no place name.
            locality = _reverse_locality(near_lat, near_lon, client=client)
            if locality and locality.lower() not in q.lower() and len(merged) < limit:
                _search({"q": f"{q} {locality}", "viewbox": viewbox, "bounded": 1})

        # Global fallback (or primary when no GPS).
        if len(merged) < limit:
            _search({"q": q})
    finally:
        if own_client:
            client.close()

    if near_lat is not None and near_lon is not None:
        merged.sort(
            key=lambda p: p.distance_m if p.distance_m is not None else float("inf")
        )
    return merged[: max(1, min(limit, 10))]


def geocode_address(
    address: str,
    *,
    near_lat: float | None = None,
    near_lon: float | None = None,
    client: httpx.Client | None = None,
) -> tuple[float, float]:
    """Resolve an address to (lat, lon) via Nominatim."""
    q = _normalize_search_query(address)
    headers = {"User-Agent": USER_AGENT}
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=20.0, headers=headers)
    try:
        rows: list[dict[str, Any]] = []
        if near_lat is not None and near_lon is not None:
            min_lon, max_lat, max_lon, min_lat = _viewbox_around(
                near_lat, near_lon, 25.0
            )
            viewbox = f"{min_lon},{max_lat},{max_lon},{min_lat}"
            rows = _nominatim_get(
                NOMINATIM_SEARCH,
                {
                    "q": q,
                    "format": "json",
                    "limit": 5,
                    "viewbox": viewbox,
                    "bounded": 1,
                },
                client=client,
            )
            if not rows:
                locality = _reverse_locality(near_lat, near_lon, client=client)
                if locality and locality.lower() not in q.lower():
                    rows = _nominatim_get(
                        NOMINATIM_SEARCH,
                        {
                            "q": f"{q} {locality}",
                            "format": "json",
                            "limit": 5,
                            "viewbox": viewbox,
                            "bounded": 1,
                        },
                        client=client,
                    )
        if not rows:
            rows = _nominatim_get(
                NOMINATIM_SEARCH,
                {"q": q, "format": "json", "limit": 1},
                client=client,
            )
    finally:
        if own_client:
            client.close()
    if not rows:
        raise ValueError(f"No geocode result for: {address!r}")
    if near_lat is not None and near_lon is not None and len(rows) > 1:
        rows.sort(
            key=lambda r: distance_meters(
                near_lat,
                near_lon,
                float(r["lat"]),
                float(r["lon"]),
            )
        )
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
