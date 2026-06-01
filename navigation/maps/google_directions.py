"""Google Maps Directions API — walking routes only.

Called once when a destination is set (background thread), never on the
per-frame /process_frame hot path.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from navigation.maps.router import RoutePlan

logger = logging.getLogger(__name__)

GOOGLE_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"


def decode_polyline(encoded: str) -> list[tuple[float, float]]:
    """Decode a Google encoded polyline to (lat, lon) pairs."""
    coords: list[tuple[float, float]] = []
    index = 0
    lat = 0
    lng = 0
    length = len(encoded)
    while index < length:
        shift = 0
        result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        shift = 0
        result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coords.append((lat / 1e5, lng / 1e5))
    return coords


def fetch_google_route(
    start_lat: float,
    start_lon: float,
    dest_lat: float,
    dest_lon: float,
    *,
    api_key: str,
    client: httpx.Client | None = None,
    timeout_sec: float = 15.0,
) -> RoutePlan:
    """Fetch a walking route from Google Directions API."""
    if not api_key.strip():
        raise ValueError("GOOGLE_MAPS_API_KEY is required for Google routing")

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=timeout_sec)

    try:
        resp = client.get(
            GOOGLE_DIRECTIONS_URL,
            params={
                "origin": f"{start_lat},{start_lon}",
                "destination": f"{dest_lat},{dest_lon}",
                "mode": "walking",
                "key": api_key.strip(),
            },
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
    finally:
        if own_client:
            client.close()

    status = data.get("status")
    if status != "OK":
        msg = data.get("error_message") or status or "unknown error"
        raise ValueError(f"Google Directions failed: {msg}")

    routes = data.get("routes") or []
    if not routes:
        raise ValueError("Google Directions returned no routes")

    route = routes[0]
    leg = (route.get("legs") or [{}])[0]
    distance_m = float((leg.get("distance") or {}).get("value", 0))
    duration_s = float((leg.get("duration") or {}).get("value", 0))

    poly = (route.get("overview_polyline") or {}).get("points") or ""
    waypoints = decode_polyline(poly) if poly else []
    if len(waypoints) < 2:
        waypoints = [(start_lat, start_lon), (dest_lat, dest_lon)]

    logger.info(
        "Google walking route: %.0f m, %d polyline points",
        distance_m,
        len(waypoints),
    )
    return RoutePlan(
        waypoints=waypoints,
        distance_m=distance_m,
        duration_s=duration_s,
    )


__all__ = ["decode_polyline", "fetch_google_route", "GOOGLE_DIRECTIONS_URL"]
