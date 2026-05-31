"""SegFormer (ADE20K) semantic segmenter.

Runs an ADE20K-trained SegFormer via HuggingFace ``transformers``. ADE20K's
150 classes cover both indoor and outdoor surfaces (floor, wall, door,
stairs, sidewalk, road, …), which removes the closed-set hallucination a
Cityscapes street model exhibits when pointed at a hallway or an empty path:
those pixels finally have a correct label to land on instead of being forced
into ``road``/``car``/``person``.

Interface (so the runner / phone servers stay backend-agnostic):

    predict(frame) -> SegmentationResult
    last_segmentation -> SegmentationResult | None
    last_results      -> None  (no Ultralytics result objects here)
    is_semantic       -> True

The produced ``SegmentationResult`` carries the same fields the rest of the
pipeline relies on: a dense ``class_map``, region-weighted obstacle counts,
per-side class pixels / walkable ratios, and ``metadata.id_to_name`` so the
overlay and depth proxy can resolve class names.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from navigation.config import Settings
from navigation.models import SegmentationResult, SIDES
from navigation.perception.segmentation_base import _region_weight_map
from navigation.perception.spatial import (
    _per_side_class_pixels,
    _per_side_walkable_ratio,
)

logger = logging.getLogger(__name__)


def _normalize_label(raw: str) -> str:
    """ADE20K labels are like ``"person;individual;someone"`` — take the first
    synonym and normalize whitespace so it matches the names in
    ``config/default.yaml``'s ``ade20k_segmentation`` block."""
    first = str(raw).split(";")[0].strip().lower()
    return first.replace("_", " ").strip()


class SegformerSegmenter:
    """ADE20K SegFormer semantic segmenter (transformers backend)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any = None
        self._processor: Any = None
        self._device: str | None = None
        self._id_to_name: dict[int, str] = {}
        self._last_segmentation: SegmentationResult | None = None

    # ------------------------------------------------------------------
    # Interface (parity with the runner's segmenter expectations)
    # ------------------------------------------------------------------

    @property
    def last_segmentation(self) -> SegmentationResult | None:
        return self._last_segmentation

    @property
    def last_results(self) -> None:
        return None  # No Ultralytics results in the SegFormer path.

    @property
    def is_semantic(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        want = (self.settings.segformer_device or "auto").strip().lower()
        if want not in ("", "auto"):
            return want
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # pragma: no cover - torch optional at import time
            pass
        return "cpu"

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        try:
            import torch  # noqa: F401
            from transformers import (
                SegformerForSemanticSegmentation,
                SegformerImageProcessor,
            )
        except ImportError as e:
            raise ImportError(
                "SegFormer backend needs extras: pip install -e '.[segformer]'"
            ) from e

        model_id = self.settings.segformer_model_id
        logger.info("Loading SegFormer model %s ...", model_id)
        self._processor = SegformerImageProcessor.from_pretrained(model_id)
        self._model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self._device = self._resolve_device()
        self._model.to(self._device)
        self._model.eval()

        # Build id -> normalized name from the model's own label map so we are
        # never out of sync with the checkpoint's class ordering.
        id2label = getattr(self._model.config, "id2label", {}) or {}
        self._id_to_name = {
            int(k): _normalize_label(v) for k, v in id2label.items()
        }
        logger.info(
            "SegFormer ready on %s (%d classes).",
            self._device,
            len(self._id_to_name),
        )
        return self._model, self._processor

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, frame: np.ndarray) -> SegmentationResult:
        seg = self._predict_segformer(frame)
        self._last_segmentation = seg
        return seg

    def _predict_segformer(self, frame: np.ndarray) -> SegmentationResult:
        import torch

        model, processor = self._load()
        display_h, display_w = frame.shape[:2]

        # BGR (OpenCV) -> RGB for the HF processor. Use cvtColor rather than a
        # reversed view (frame[:, :, ::-1]): the latter yields a negative-stride
        # array that torch.from_numpy() in the image processor rejects.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits  # (1, num_classes, h/4, w/4)

        # Upsample logits to the display resolution, then argmax to a dense map.
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=(display_h, display_w),
            mode="bilinear",
            align_corners=False,
        )
        class_map = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
        return self._parse_class_map(class_map, display_h, display_w)

    def _parse_class_map(
        self, class_map: np.ndarray, h: int, w: int
    ) -> SegmentationResult:
        cfg = self.settings.seg_class_config()
        obstacle_set = set(cfg.get("obstacle_classes", []))
        walkable_set = set(cfg.get("walkable_classes", []))

        weight_map = _region_weight_map((h, w))
        obstacle_pixels = 0
        obstacle_weighted = 0.0
        walkable_pixels = 0
        total = int(class_map.size)
        counts: dict[str, int] = {}
        present: list[str] = []

        # Two vectorized passes replace the per-class mask/sum loop: ``bincount``
        # tallies pixels and region-weighted mass for every class id in one C
        # scan each. Iterating ``nonzero`` ids preserves the original ascending
        # order (so name collisions resolve identically) while skipping the
        # O(classes × pixels) boolean masking that dominated postprocessing.
        flat = class_map.ravel()
        length = int(flat.max()) + 1 if flat.size else 1
        pixels_by_id = np.bincount(flat, minlength=length)
        weighted_by_id = np.bincount(
            flat, weights=weight_map.ravel(), minlength=length
        )
        for cls_id in np.nonzero(pixels_by_id)[0]:
            name = self._id_to_name.get(int(cls_id), str(int(cls_id)))
            pixels = int(pixels_by_id[cls_id])
            counts[name] = pixels
            if name in obstacle_set:
                obstacle_pixels += pixels
                obstacle_weighted += float(weighted_by_id[cls_id])
            if name in walkable_set:
                walkable_pixels += pixels
            present.append(name)

        walkable_ratio = walkable_pixels / max(total, 1)
        per_side_pixels = _per_side_class_pixels(
            class_map, self._id_to_name, weight_map
        )
        per_side_walk = _per_side_walkable_ratio(
            class_map, self._id_to_name, walkable_set
        )

        return SegmentationResult(
            class_names=present,
            class_map=class_map,
            obstacle_pixels=obstacle_pixels,
            obstacle_pixels_weighted=obstacle_weighted,
            walkable_ratio=min(1.0, walkable_ratio),
            per_side_class_pixels=per_side_pixels,
            per_side_walkable_ratio=per_side_walk,
            metadata={
                "semantic": True,
                "backend": "segformer",
                "pixel_counts": counts,
                "shape": [h, w],
                "id_to_name": self._id_to_name,
            },
        )

__all__ = ["SegformerSegmenter", "_normalize_label"]
