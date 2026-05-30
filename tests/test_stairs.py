"""Tests for navigation.perception.stairs — heuristic detector."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from navigation.config import Settings
from navigation.models import SegmentationResult
from navigation.perception.stairs import StairsDetector
from navigation.reasoning.facts import StairsResult


def _settings(stairs_min_edge_density: float = 0.08, enabled: bool = True) -> Settings:
    return Settings(
        stairs_detector_enabled=enabled,
        stairs_min_edge_density=stairs_min_edge_density,
        stairs_min_confidence=0.4,
    )


def _seg_with_walkable_floor(h: int, w: int, walkable_id: int = 0, walkable_name: str = "road") -> SegmentationResult:
    class_map = np.full((h, w), walkable_id, dtype=np.int32)
    return SegmentationResult(
        class_map=class_map,
        metadata={"semantic": True, "id_to_name": {walkable_id: walkable_name}},
    )


def _smooth_floor_frame(h: int, w: int, intensity: int = 128) -> np.ndarray:
    return np.full((h, w, 3), intensity, dtype=np.uint8)


def _step_frame(h: int, w: int, step_row: int, light: int = 220, dark: int = 30) -> np.ndarray:
    """Frame with a sharp horizontal edge: top half light, bottom half dark."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:step_row, :, :] = light
    frame[step_row:, :, :] = dark
    return frame


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


# Feature: spatial-aware-natural-language-guidance, Property 11: StairsDetector confidence is bounded
@given(
    contrast=st.integers(min_value=0, max_value=255),
    place_step=st.booleans(),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=100, deadline=None)
def test_stairs_detector_confidence_bounded_and_returns_dataclass(
    contrast, place_step, seed
):
    h, w = 80, 120
    rng = np.random.default_rng(seed)
    seg = _seg_with_walkable_floor(h, w)
    if place_step:
        # Inside the bottom 30% (y >= 56), put a horizontal edge.
        step_row = 70
        frame = np.full((h, w, 3), 50, dtype=np.uint8)
        frame[step_row:, :, :] = np.clip(50 + contrast, 0, 255)
    else:
        # Smooth floor with a tiny random texture
        frame = np.full((h, w, 3), 128, dtype=np.uint8)
        frame += rng.integers(-3, 4, size=frame.shape, dtype=np.int8).astype(
            np.uint8, casting="unsafe"
        )

    det = StairsDetector(_settings())
    result = det.detect(frame, seg)
    assert isinstance(result, StairsResult)
    assert 0.0 <= result.confidence <= 1.0
    if result.flag:
        assert result.confidence >= 0.0


# ---------------------------------------------------------------------------
# Example tests
# ---------------------------------------------------------------------------


def test_flag_on_synthetic_step():
    """Sharp horizontal edge in the bottom 30% should flag stairs."""
    h, w = 80, 120
    seg = _seg_with_walkable_floor(h, w)
    frame = _step_frame(h, w, step_row=70)  # inside bottom 30% (>= 56)
    det = StairsDetector(_settings())
    result = det.detect(frame, seg)
    assert result.flag is True
    assert 0.0 <= result.confidence <= 1.0


def test_no_flag_on_smooth_floor():
    """Uniform floor should not flag."""
    h, w = 80, 120
    seg = _seg_with_walkable_floor(h, w)
    frame = _smooth_floor_frame(h, w, intensity=128)
    det = StairsDetector(_settings())
    result = det.detect(frame, seg)
    assert result.flag is False
    assert result.confidence == 0.0


def test_no_walkable_returns_no_flag_no_exception():
    """When the bottom strip has no walkable pixels, return False, no raise."""
    h, w = 80, 120
    # Class map of all 'building' (1) — not walkable.
    cm = np.full((h, w), 1, dtype=np.int32)
    seg = SegmentationResult(
        class_map=cm,
        metadata={"semantic": True, "id_to_name": {0: "road", 1: "building"}},
    )
    frame = _step_frame(h, w, step_row=70)
    det = StairsDetector(_settings())
    result = det.detect(frame, seg)
    assert result.flag is False
    assert result.confidence == 0.0


def test_returns_stairs_result_dataclass():
    """Contract: detect returns a StairsResult with the three fields."""
    det = StairsDetector(_settings())
    seg = _seg_with_walkable_floor(40, 60)
    frame = _smooth_floor_frame(40, 60)
    result = det.detect(frame, seg)
    assert isinstance(result, StairsResult)
    assert isinstance(result.flag, bool)
    assert isinstance(result.confidence, float)
    assert isinstance(result.rationale, str)


def test_disabled_returns_no_flag_no_phrase():
    """stairs_detector_enabled=False short-circuits to a no-op result."""
    det = StairsDetector(_settings(enabled=False))
    seg = _seg_with_walkable_floor(80, 120)
    frame = _step_frame(80, 120, step_row=70)
    result = det.detect(frame, seg)
    assert result.flag is False
    assert result.confidence == 0.0
    assert result.rationale == "disabled"


def test_no_class_map_returns_no_flag_no_exception():
    det = StairsDetector(_settings())
    seg = SegmentationResult(metadata={"semantic": True})
    frame = _step_frame(40, 60, step_row=30)
    result = det.detect(frame, seg)
    assert result.flag is False


def test_confidence_clamped_to_unit_interval():
    """Even with extremely strong gradients, confidence stays in [0, 1]."""
    h, w = 80, 120
    seg = _seg_with_walkable_floor(h, w)
    # Maximum-contrast step.
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:70, :, :] = 0
    frame[70:, :, :] = 255
    det = StairsDetector(_settings(stairs_min_edge_density=0.0))
    result = det.detect(frame, seg)
    assert 0.0 <= result.confidence <= 1.0
