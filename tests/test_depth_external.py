"""Tests for client-provided (on-device) depth feeding the DepthEstimator.

The phone runs Depth Anything V2 in the browser and posts an approximate
obstacle distance as ``depth_m``. ``DepthEstimator.predict`` consumes it via
``external_depth_m`` and the value flows straight into ``bucketize()``. When no
client depth is provided, the estimator falls back to the segmentation proxy.
"""

from __future__ import annotations

import numpy as np
import pytest

from navigation.config import Settings
from navigation.models import DepthResult, SegmentationResult
from navigation.output.distance import DistanceConfig, bucketize
from navigation.perception.depth import DepthEstimator


@pytest.fixture
def frame() -> np.ndarray:
    return np.full((48, 64, 3), 128, dtype=np.uint8)


@pytest.fixture
def estimator() -> DepthEstimator:
    return DepthEstimator(Settings())


# ---------------------------------------------------------------------------
# External depth wins over proxy/mock
# ---------------------------------------------------------------------------


def test_external_depth_is_used_verbatim(estimator, frame):
    result = estimator.predict(frame, external_depth_m=2.0)
    assert isinstance(result, DepthResult)
    assert result.obstacle_depth_m == pytest.approx(2.0)
    assert result.center_depth_m == pytest.approx(2.0)
    assert result.metadata.get("source") == "client_depth_anything"
    assert result.metadata.get("mock") is False


def test_external_depth_overrides_segmentation(estimator, frame):
    # external_depth_m must win regardless of any other state.
    result = estimator.predict(frame, external_depth_m=0.6)
    assert result.metadata.get("source") == "client_depth_anything"
    assert result.obstacle_depth_m == pytest.approx(0.6)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (0.05, 0.3),    # below floor -> clamped up
        (50.0, 15.0),   # above ceiling -> clamped down
        (float("nan"), 0.3),
        (float("inf"), 15.0),
    ],
)
def test_external_depth_clamped_and_sanitized(estimator, frame, raw, expected):
    result = estimator.predict(frame, external_depth_m=raw)
    assert result.obstacle_depth_m == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Fallback to the segmentation proxy when no client depth
# ---------------------------------------------------------------------------


def _seg_with_class_map(h: int = 48, w: int = 64) -> SegmentationResult:
    cm = np.zeros((h, w), dtype=np.int32)  # all walkable "road"
    return SegmentationResult(
        class_map=cm,
        metadata={"id_to_name": {0: "road"}, "shape": [h, w]},
    )


def test_none_external_depth_falls_back_to_proxy(estimator, frame):
    result = estimator.predict(
        frame, external_depth_m=None, segmentation=_seg_with_class_map()
    )
    # Proxy path never tags itself as the client source.
    assert result.metadata.get("source") != "client_depth_anything"
    assert result.center_depth_m is not None


def test_no_depth_source_raises(estimator, frame):
    # No client depth and no segmentation class map -> depth is undefined.
    with pytest.raises(ValueError):
        estimator.predict(frame)


# ---------------------------------------------------------------------------
# Integration with the distance bucketizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "depth_m, expected_bucket",
    [
        (0.5, "immediate"),
        (2.0, "near"),
        (3.0, "mid"),
        (6.0, "far"),
    ],
)
def test_external_depth_maps_to_expected_bucket(estimator, frame, depth_m, expected_bucket):
    result = estimator.predict(frame, external_depth_m=depth_m)
    bucket, phrase = bucketize(result.obstacle_depth_m, DistanceConfig())
    assert bucket == expected_bucket
    assert isinstance(phrase, str) and phrase
