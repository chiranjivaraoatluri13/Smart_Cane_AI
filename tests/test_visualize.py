"""Smoke tests for segmentation overlay helpers."""

import numpy as np

from navigation.perception.visualize import (
    overlay_from_class_map,
    overlay_mock,
    render_overlay,
)
from navigation.perception.segmentation import YoloSegmenter
from navigation.config import Settings


def test_overlay_mock_shape_and_dtype() -> None:
    frame = np.full((120, 160, 3), 90, dtype=np.uint8)
    out = overlay_mock(frame)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, frame)


def test_render_overlay_dry_run() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    segmenter = YoloSegmenter(Settings())
    segmenter.predict(frame, dry_run=True)
    out = render_overlay(frame, segmenter=segmenter, dry_run=True)
    assert out.shape == frame.shape


def test_overlay_from_class_map() -> None:
    frame = np.full((40, 40, 3), 100, dtype=np.uint8)
    class_map = np.zeros((40, 40), dtype=np.int32)
    class_map[20:, :] = 0
    class_map[:20, :] = 10
    class_map[30:, 10:20] = 11
    names = {0: "road", 10: "sky", 11: "person"}
    out = overlay_from_class_map(frame, class_map, names)
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)
