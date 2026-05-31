"""Tests for map-assisted routing (OSRM mock)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from navigation.config import Settings
from navigation.maps.guidance import MapGuidance
from navigation.maps.router import (
    bearing_deg,
    bearing_to_next_waypoint,
    distance_meters,
    distance_to_destination,
    fetch_route,
    fetch_route_from_json,
    geocode_address,
)
from navigation.models import NavigationCommand, CareResult, DepthResult, PerceptionBundle, SegmentationResult
from navigation.reasoning.llm import NavigationInterpreter

FIXTURE = Path(__file__).parent / "fixtures" / "osrm_route.json"


@pytest.fixture
def sample_route():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return fetch_route_from_json(data)


def test_fetch_route_from_fixture(sample_route):
    assert len(sample_route.waypoints) == 4
    assert sample_route.waypoints[0] == pytest.approx((40.7484, -73.9857))
    assert sample_route.distance_m > 0


def test_bearing_and_distance(sample_route):
    lat, lon = sample_route.waypoints[0]
    dest_lat, dest_lon = sample_route.destination
    assert distance_meters(lat, lon, dest_lat, dest_lon) > 100
    b = bearing_to_next_waypoint(lat, lon, sample_route)
    assert 0 <= b < 360


def test_map_guidance_on_route(sample_route):
    settings = Settings(
        route_at_dest_m=10,
        route_off_route_m=50,
        route_bearing_align_deg=30,
        current_heading_deg=45,
    )
    guidance = MapGuidance(sample_route, settings)
    lat, lon = sample_route.waypoints[1]
    decision = guidance.decide(lat, lon, heading_deg=45)
    assert decision.command in {
        NavigationCommand.GO_FORWARD,
        NavigationCommand.MOVE_LEFT,
        NavigationCommand.MOVE_RIGHT,
    }


def test_map_guidance_at_destination(sample_route):
    settings = Settings(route_at_dest_m=50)
    guidance = MapGuidance(sample_route, settings)
    dest_lat, dest_lon = sample_route.destination
    decision = guidance.decide(dest_lat, dest_lon, heading_deg=0)
    assert decision.command == NavigationCommand.STOP


def test_fetch_route_mocked(sample_route):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def handler(request: httpx.Request) -> httpx.Response:
        assert "router.project-osrm.org" in str(request.url)
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        route = fetch_route(40.7484, -73.9857, 40.7510, -73.9830, client=client)
    assert len(route.waypoints) == len(sample_route.waypoints)


def test_geocode_mocked():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "nominatim.openstreetmap.org" in str(request.url)
        return httpx.Response(
            200,
            json=[{"lat": "40.7580", "lon": "-73.9855", "display_name": "Test"}],
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, headers={"User-Agent": "test"}) as client:
        lat, lon = geocode_address("Times Square", client=client)
    assert lat == pytest.approx(40.7580)
    assert lon == pytest.approx(-73.9855)


def test_obstacle_overrides_map_guidance(sample_route):
    settings = Settings(
        use_map_guidance=True,
        dest_lat=40.7510,
        dest_lon=-73.9830,
        current_lat=40.7484,
        current_lon=-73.9857,
        current_heading_deg=45,
    )
    interp = NavigationInterpreter(settings)
    interp._map_guidance = MapGuidance(sample_route, settings)

    bundle = PerceptionBundle(
        frame_id=0,
        segmentation=SegmentationResult(obstacle_pixels=1000),
        depth=DepthResult(obstacle_depth_m=0.5),
        care=CareResult(hazard_detected=True, safety_score=0.2),
    )
    decision = interp.interpret(bundle)
    assert decision.command == NavigationCommand.STOP
