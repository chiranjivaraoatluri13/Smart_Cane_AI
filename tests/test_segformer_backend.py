"""Tests for the SegFormer (ADE20K) segmentation backend.

These avoid downloading the real model: they exercise the factory, the
backend class config, label normalization, and the class-map parser (driven
through a tiny synthetic map). The heavy ``transformers`` import stays inside
``_load`` so none of this needs it.
"""

from __future__ import annotations

import numpy as np
import pytest

from navigation.config import Settings
from navigation.perception.segmentation_base import build_segmenter
from navigation.perception.segmentation_segformer import (
    SegformerSegmenter,
    _normalize_label,
)
from navigation.perception.segmentation_segformer_onnx import (
    SegformerOnnxSegmenter,
)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_uses_onnx_when_model_exists(tmp_path):
    """Default backend resolves to the fast ONNX path when the model is present."""
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"")  # factory only checks existence; loading is lazy
    seg = build_segmenter(
        Settings(segmenter_backend="segformer_onnx",
                 segformer_onnx_path=str(onnx_file))
    )
    assert isinstance(seg, SegformerOnnxSegmenter)


def test_factory_falls_back_to_transformers_when_onnx_missing(tmp_path):
    """If the ONNX model isn't exported yet, fall back instead of crashing."""
    missing = tmp_path / "does_not_exist.onnx"
    seg = build_segmenter(
        Settings(segmenter_backend="segformer_onnx",
                 segformer_onnx_path=str(missing))
    )
    assert isinstance(seg, SegformerSegmenter)


def test_factory_builds_transformers_backend_when_requested():
    seg = build_segmenter(Settings(segmenter_backend="segformer"))
    assert isinstance(seg, SegformerSegmenter)


# ---------------------------------------------------------------------------
# Class config (single source of truth) — always ADE20K now
# ---------------------------------------------------------------------------


def test_seg_class_config_is_ade20k():
    cfg = Settings().seg_class_config()
    walkable = cfg.get("walkable_classes", [])
    obstacle = cfg.get("obstacle_classes", [])
    assert "floor" in walkable
    assert "wall" in obstacle  # closed-set fix: wall is a labeled obstacle


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("person;individual;someone", "person"),
        ("wall", "wall"),
        ("stairs;steps", "stairs"),
        ("traffic_light", "traffic light"),
        ("  Sidewalk;Pavement ", "sidewalk"),
    ],
)
def test_normalize_label(raw, expected):
    assert _normalize_label(raw) == expected


# ---------------------------------------------------------------------------
# Interface parity
# ---------------------------------------------------------------------------


def test_segformer_interface_attributes():
    seg = SegformerSegmenter(Settings())
    assert seg.is_semantic is True
    assert seg.last_results is None
    assert seg.last_segmentation is None


def test_onnx_interface_attributes():
    """The ONNX backend exposes the same interface without loading the model.

    Construction must be cheap: no onnxruntime session and no model file are
    touched until ``predict`` is called, so this stays import-light.
    """
    seg = SegformerOnnxSegmenter(Settings())
    assert seg.is_semantic is True
    assert seg.last_results is None
    assert seg.last_segmentation is None


# ---------------------------------------------------------------------------
# Class-map parser (synthetic map, exercises the real parsing path)
# ---------------------------------------------------------------------------


def test_parse_class_map_counts_obstacle_and_walkable():
    seg = SegformerSegmenter(Settings())
    seg._id_to_name = {0: "floor", 1: "person"}
    h, w = 40, 40
    cm = np.zeros((h, w), dtype=np.int32)
    cm[h // 2 :, :] = 0          # bottom half floor
    cm[: h // 2, w // 2 :] = 1   # top-right quadrant person

    result = seg._parse_class_map(cm, h, w)
    assert "floor" in result.class_names
    assert "person" in result.class_names
    assert result.walkable_ratio > 0.0
    assert result.obstacle_pixels > 0
    assert result.metadata["id_to_name"] == {0: "floor", 1: "person"}
    assert result.metadata["backend"] == "segformer"
    # Updating last_segmentation only happens via predict(); parser is pure.
    assert seg.last_segmentation is None


def test_parse_class_map_obstacle_weighting_prefers_bottom_center():
    """A person low-and-center must weigh more than the same area top-edge —
    the region weighting that makes the model usable for a walker."""
    seg = SegformerSegmenter(Settings())
    seg._id_to_name = {0: "floor", 1: "person"}
    h, w = 60, 60

    bottom_center = np.zeros((h, w), dtype=np.int32)
    bottom_center[h - 10 :, w // 2 - 5 : w // 2 + 5] = 1
    top_edge = np.zeros((h, w), dtype=np.int32)
    top_edge[:10, :10] = 1

    wb = seg._parse_class_map(bottom_center, h, w).obstacle_pixels_weighted
    wt = seg._parse_class_map(top_edge, h, w).obstacle_pixels_weighted
    assert wb > wt
