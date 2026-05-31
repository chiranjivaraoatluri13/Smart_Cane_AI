"""Smoke tests for segmentation overlay helpers."""

import numpy as np

from navigation.config import Settings
from navigation.models import SegmentationResult
from navigation.perception.segmentation_segformer import SegformerSegmenter
from navigation.perception.visualize import (
    overlay_from_class_map,
    render_overlay,
)


def test_render_overlay_from_last_segmentation() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    segmenter = SegformerSegmenter(Settings())
    # Seed last_segmentation directly (no model download): a class map the
    # overlay can tint.
    class_map = np.zeros((64, 64), dtype=np.int32)
    class_map[32:, :] = 0
    class_map[:32, :] = 1
    segmenter._last_segmentation = SegmentationResult(
        class_map=class_map,
        metadata={"id_to_name": {0: "floor", 1: "wall"}},
    )
    out = render_overlay(frame, segmenter=segmenter)
    assert out.shape == frame.shape


def test_render_overlay_without_segmentation_returns_copy() -> None:
    frame = np.full((32, 32, 3), 50, dtype=np.uint8)
    segmenter = SegformerSegmenter(Settings())
    out = render_overlay(frame, segmenter=segmenter)
    assert out.shape == frame.shape
    assert np.array_equal(out, frame)


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
