"""Shared segmentation helpers and the segmenter factory.

The project segments scenes with a single backend now: an ADE20K-trained
SegFormer (indoor + outdoor coverage, no closed-set street hallucination).
This module holds the pieces that are backend-agnostic — the region-importance
weight map and the ``build_segmenter`` factory — so perception, reasoning, and
output components import them without depending on a specific model.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from navigation.config import Settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _region_weight_map(shape: tuple[int, int]) -> np.ndarray:
    """Per-pixel importance weights for "is this in the walking path?".

    Bottom-center counts most (where the user is heading), top half is ignored
    (sky / distant ceiling), and edges count half (peripheral hazards still
    matter, just less). Cached because frame shape rarely changes.

    Returns a float32 array with the same HxW as ``shape``, values in [0, 1].
    """
    h, w = shape
    # Vertical falloff: 0 in the top third, ramps to 1 by the bottom.
    yy = np.arange(h, dtype=np.float32) / max(h - 1, 1)
    v = np.clip((yy - 0.33) / 0.67, 0.0, 1.0)

    # Horizontal falloff: 1 in the center 50%, 0.5 at the edges.
    xx = np.arange(w, dtype=np.float32) / max(w - 1, 1)
    dx = np.abs(xx - 0.5) * 2.0  # 0 at center, 1 at edges
    hor = np.where(dx <= 0.5, 1.0, 0.5)

    weights = np.outer(v, hor).astype(np.float32)
    return weights


def build_segmenter(settings: Settings):
    """Construct the segmentation backend.

    - ``segformer_onnx`` (default) — ADE20K SegFormer via onnxruntime INT8.
      Fast CPU inference (~80ms/frame). Requires the exported ONNX model:
      run ``python scripts/export_segformer_onnx.py`` once.
    - ``segformer`` — ADE20K SegFormer via HuggingFace transformers.
      Accurate but slow on CPU (~350-800ms/frame). Use when the ONNX model
      is not yet exported or when running on GPU.
    """
    backend = (settings.segmenter_backend or "segformer_onnx").strip().lower()
    if backend == "segformer_onnx":
        onnx_path = Path(
            getattr(settings, "segformer_onnx_path",
                    "segformer_b0_ade20k_int8.onnx")
        )
        if onnx_path.is_file():
            from navigation.perception.segmentation_segformer_onnx import (
                SegformerOnnxSegmenter,
            )
            return SegformerOnnxSegmenter(settings)
        # The fast path was requested but the model hasn't been exported.
        # Rather than crash on the first frame, fall back to the (slower)
        # transformers backend and tell the user exactly how to get the
        # speedup back.
        logger.warning(
            "ONNX model not found at %s — falling back to the slower "
            "transformers backend. Export it once for fast inference:\n"
            "    python scripts/export_segformer_onnx.py",
            onnx_path,
        )
    # Fallback: full transformers path (accurate, slower)
    from navigation.perception.segmentation_segformer import SegformerSegmenter
    return SegformerSegmenter(settings)


__all__ = ["_region_weight_map", "build_segmenter"]
