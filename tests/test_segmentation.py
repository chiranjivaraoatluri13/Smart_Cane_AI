"""Tests for semantic segmentation parsing (no GPU weights)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from navigation.config import Settings
from navigation.perception.segmentation import YoloSegmenter, is_semantic_model


def test_is_semantic_model() -> None:
    assert is_semantic_model("yolo26n-sem.pt")
    assert is_semantic_model("models/yolo11n-sem.pt")
    assert not is_semantic_model("yolo26n-seg.pt")


def test_parse_semantic_class_map_counts() -> None:
    settings = Settings(yolo_model_path="yolo26n-sem.pt")
    segmenter = YoloSegmenter(settings)

    class_map = np.array(
        [
            [0, 0, 1, 1],
            [11, 11, 1, 1],
        ],
        dtype=np.int32,
    )
    names = {0: "road", 1: "sidewalk", 11: "person"}
    sem = SimpleNamespace(data=MagicMock())
    sem.data.cpu.return_value.numpy.return_value = class_map
    result = SimpleNamespace(names=names, semantic_mask=sem)
    seg = segmenter._parse_semantic([result])

    assert seg.class_map is not None
    assert seg.obstacle_pixels == 2
    assert seg.walkable_ratio > 0
    assert "road" in seg.class_names
    assert seg.metadata.get("semantic") is True


def test_mock_dry_run_semantic_metadata() -> None:
    settings = Settings(yolo_model_path="yolo26n-sem.pt")
    segmenter = YoloSegmenter(settings)
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    seg = segmenter.predict(frame, dry_run=True)
    assert seg.metadata.get("mock") is True
    assert seg.metadata.get("semantic") is True
