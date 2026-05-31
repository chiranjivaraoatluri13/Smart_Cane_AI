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
import os
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
_INPUT_SIZE = 512  # SegFormer ADE20K models expect 512×512 input (can be overridden)


def _preprocess(frame_bgr: np.ndarray, input_size: int = _INPUT_SIZE) -> np.ndarray:
    """BGR frame → normalised NCHW float32 tensor (numpy).

    Replicates SegformerImageProcessor:
      1. Resize to input_size×input_size (bilinear)
      2. BGR → RGB
      3. Normalise with ImageNet mean/std
      4. HWC → NCHW, add batch dim

    Resize is done *before* the colour conversion so the BGR→RGB swap runs on
    the small input_size×input_size image (e.g. 65k px) instead of the full
    camera frame (e.g. 400k px) — same result, less work.
    """
    resized_bgr = cv2.resize(frame_bgr, (input_size, input_size),
                             interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    img = rgb.astype(np.float32) / 255.0
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
        # Intra-op threads (the matrix-multiply pool). Do NOT use "all cores"
        # (0): in a container ORT reads the *host's* physical core count, not
        # the cgroup CPU limit, so it oversubscribes a 1-vCPU box and the INT8
        # model runs ~5× slower. Honour an explicit setting; otherwise cap the
        # auto value so we never hit that oversubscription cliff.
        intra = int(getattr(self.settings, "onnx_intra_op_threads", 0) or 0)
        if intra <= 0:
            intra = max(1, min(4, os.cpu_count() or 1))
        opts.intra_op_num_threads = intra
        # Inter-op parallelism is kept at 1 because the pipeline is single-
        # threaded and spawning extra threads adds overhead.
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        # Single-threaded sessions run slightly faster in sequential mode
        # (no thread-pool overhead between ops).
        if intra == 1:
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        logger.info("SegFormer ONNX intra_op_num_threads=%d", intra)

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

        # Get inference size from settings (default 512, can be reduced to 256/384 for speed)
        inference_imgsz = int(getattr(self.settings, "inference_imgsz", 0))
        if inference_imgsz <= 0:
            inference_imgsz = _INPUT_SIZE
        
        # Preprocess: BGR → normalised NCHW float32
        pixel_values = _preprocess(frame, input_size=inference_imgsz)

        # Run ONNX inference
        logits = session.run(None, {"pixel_values": pixel_values})[0]
        # logits: (1, num_classes, H/4, W/4)  e.g. (1, 150, 128, 128) for 512x512 input
        logits = logits[0]  # (num_classes, H/4, W/4)

        # Argmax at the model's native output stride (e.g. 64×64 for a 256px
        # input). This small map is the segmenter's *actual* resolution —
        # upscaling it to the camera frame before analysis adds no information,
        # just work.
        class_map_small = np.argmax(logits, axis=0).astype(np.int32)
        small_h, small_w = class_map_small.shape

        # Run the (scale-invariant) per-side / ratio analysis on the small map.
        # Every reasoning consumer reads ratios or weighted counts paired with
        # ``metadata["shape"]`` — all derived here at the same resolution — so
        # the decisions are identical to analyzing the upscaled map while the
        # postprocessing cost drops by the upscale factor squared.
        seg = self._parser._parse_class_map(class_map_small, small_h, small_w)

        upscale = bool(getattr(self.settings, "seg_upscale_class_map", True))
        if upscale and (small_h, small_w) != (display_h, display_w):
            class_map = cv2.resize(
                class_map_small.astype(np.float32),
                (display_w, display_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
        else:
            class_map = class_map_small

        # Tag the backend so the JSON record is accurate. ``metadata["shape"]``
        # stays at the analysis resolution (set by the parser) so the reasoner's
        # ratio math remains self-consistent with the weighted counts above.
        meta = dict(seg.metadata)
        meta["backend"] = "segformer_onnx"
        meta["inference_imgsz"] = inference_imgsz
        if upscale:
            meta["display_shape"] = [display_h, display_w]
        seg = seg.model_copy(update={"class_map": class_map, "metadata": meta})
        self._last_segmentation = seg
        return seg


__all__ = ["SegformerOnnxSegmenter"]
