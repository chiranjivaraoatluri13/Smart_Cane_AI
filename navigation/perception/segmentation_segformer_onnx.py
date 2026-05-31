"""SegFormer ONNX segmenter — fast CPU inference via onnxruntime.

Runs the exported SegFormer-B0 INT8 ONNX model through onnxruntime instead
of PyTorch. On the same CPU this is typically 3-5× faster than the
transformers eager-mode path:

  SegFormer-B2 PyTorch  ~800ms/frame  (1.2 FPS)
  SegFormer-B0 PyTorch  ~350ms/frame  (2.9 FPS)
  SegFormer-B0 ONNX FP32 ~150ms/frame (6-7 FPS)
  SegFormer-B0 ONNX INT8  ~80ms/frame (12-15 FPS)  ← default target

No torch required at inference time — only onnxruntime and numpy.

Export the model once with:
    python scripts/export_segformer_onnx.py

Then set in .env:
    SEGMENTER_BACKEND=segformer_onnx
    SEGFORMER_ONNX_PATH=segformer_b0_ade20k_int8.onnx

The id2label JSON (same stem, .json extension) is loaded automatically
from the same directory as the ONNX file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from navigation.config import Settings
from navigation.models import SegmentationResult
from navigation.perception.segmentation_segformer import (
    SegformerSegmenter,
    _normalize_label,
)

logger = logging.getLogger(__name__)

# SegFormer processor constants (matches SegformerImageProcessor defaults).
# We replicate the preprocessing here so we don't need transformers at
# inference time — just numpy + cv2.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SIZE = 512  # SegFormer ADE20K models expect 512×512 input


def _preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR frame → normalised NCHW float32 tensor (numpy).

    Replicates SegformerImageProcessor:
      1. BGR → RGB
      2. Resize to 512×512 (bilinear)
      3. Normalise with ImageNet mean/std
      4. HWC → NCHW, add batch dim
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    img = resized.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    # HWC → CHW → NCHW
    return np.transpose(img, (2, 0, 1))[np.newaxis].astype(np.float32)


class SegformerOnnxSegmenter:
    """ADE20K SegFormer segmenter backed by onnxruntime (no torch at runtime).

    Shares ``_parse_class_map`` with ``SegformerSegmenter`` so the downstream
    pipeline sees identical ``SegmentationResult`` objects regardless of which
    backend produced them.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._session: Any = None
        self._id_to_name: dict[int, str] = {}
        self._last_segmentation: SegmentationResult | None = None
        # Reuse the parent's _parse_class_map — it's pure numpy, no torch.
        self._parser = SegformerSegmenter.__new__(SegformerSegmenter)
        self._parser.settings = settings
        self._parser._id_to_name = {}

    # ------------------------------------------------------------------
    # Interface (same as SegformerSegmenter)
    # ------------------------------------------------------------------

    @property
    def last_segmentation(self) -> SegmentationResult | None:
        return self._last_segmentation

    @property
    def last_results(self) -> None:
        return None

    @property
    def is_semantic(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self) -> Any:
        if self._session is not None:
            return self._session

        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime not installed: pip install onnxruntime>=1.18"
            ) from e

        model_path = Path(getattr(self.settings, "segformer_onnx_path",
                                  "segformer_b0_ade20k_int8.onnx"))
        if not model_path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}\n"
                "Export it first: python scripts/export_segformer_onnx.py"
            )

        # Load id2label from the companion JSON (same stem, .json extension).
        label_path = model_path.with_suffix(".json")
        if label_path.is_file():
            raw = json.loads(label_path.read_text())
            self._id_to_name = {
                int(k): _normalize_label(v) for k, v in raw.items()
            }
        else:
            logger.warning(
                "id2label JSON not found at %s — class names will be numeric.",
                label_path,
            )

        # Sync the parser's id_to_name so _parse_class_map resolves names.
        self._parser._id_to_name = self._id_to_name

        opts = ort.SessionOptions()
        # Use all available cores for intra-op parallelism (matrix multiplies).
        # Inter-op parallelism is kept at 1 because the pipeline is single-
        # threaded and spawning extra threads adds overhead.
        opts.intra_op_num_threads = 0   # 0 = use all cores
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        logger.info(
            "SegFormer ONNX loaded: %s (%d classes)",
            model_path.name,
            len(self._id_to_name),
        )
        return self._session

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, frame: np.ndarray) -> SegmentationResult:
        session = self._load()
        display_h, display_w = frame.shape[:2]

        # Preprocess: BGR → normalised NCHW float32
        pixel_values = _preprocess(frame)

        # Run ONNX inference
        logits = session.run(None, {"pixel_values": pixel_values})[0]
        # logits: (1, num_classes, H/4, W/4)  e.g. (1, 150, 128, 128)
        logits = logits[0]  # (num_classes, H/4, W/4)

        # Upsample to display resolution using cv2 (no torch needed).
        # argmax first (cheaper to resize a single-channel map than 150 channels).
        class_map_small = np.argmax(logits, axis=0).astype(np.int32)
        class_map = cv2.resize(
            class_map_small.astype(np.float32),
            (display_w, display_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)

        seg = self._parser._parse_class_map(class_map, display_h, display_w)
        # Tag the backend so the JSON record is accurate.
        meta = dict(seg.metadata)
        meta["backend"] = "segformer_onnx"
        seg = seg.model_copy(update={"metadata": meta})
        self._last_segmentation = seg
        return seg


__all__ = ["SegformerOnnxSegmenter"]
