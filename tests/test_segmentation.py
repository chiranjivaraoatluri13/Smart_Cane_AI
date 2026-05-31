"""Tests for semantic segmentation parsing (SegFormer / ADE20K, no model download).

These drive the real ``_parse_class_map`` path with a synthetic class map and
an injected ``id_to_name``, so they exercise the production parsing logic
without needing the transformers model weights.
"""

from __future__ import annotations

import numpy as np

from navigation.config import Settings
from navigation.perception.segmentation_segformer import SegformerSegmenter


def _segmenter() -> SegformerSegmenter:
    seg = SegformerSegmenter(Settings())
    # ADE20K-style labels: floor walkable, person obstacle.
    seg._id_to_name = {0: "floor", 1: "person"}
    return seg


def test_parse_semantic_class_map_counts() -> None:
    segmenter = _segmenter()

    class_map = np.array(
        [
            [0, 0, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    seg = segmenter._parse_class_map(class_map, 2, 4)

    assert seg.class_map is not None
    assert seg.obstacle_pixels == 6  # six "person" pixels
    assert seg.walkable_ratio > 0
    assert "floor" in seg.class_names
    assert "person" in seg.class_names
    assert seg.metadata.get("semantic") is True
    assert seg.metadata.get("backend") == "segformer"


def test_parse_populates_per_side_fields() -> None:
    segmenter = _segmenter()
    class_map = np.zeros((10, 30), dtype=np.int32)
    class_map[5:, :] = 0          # floor in the bottom
    class_map[:5, 10:20] = 1      # person in top-center
    seg = segmenter._parse_class_map(class_map, 10, 30)

    assert seg.per_side_class_pixels is not None
    assert seg.per_side_walkable_ratio is not None
    # id_to_name is carried in metadata for the overlay / depth proxy.
    assert seg.metadata["id_to_name"] == {0: "floor", 1: "person"}
