"""Heuristic stairs/curb detector (Requirement 5, 14).

Looks for a horizontal-edge / luminance discontinuity in the bottom 30%
of the walkable region. This is intentionally a small, swappable module:
a future trained model on ADE20K or Mapillary Vistas drops in by matching
the same `(frame, segmentation) -> StairsResult` contract — no changes
required in the reasoner or composer.

Cost is bounded: input is downscaled to a 256-wide working copy before
Sobel runs (Req 5.7).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from navigation.config import Settings
from navigation.models import SegmentationResult
from navigation.reasoning.facts import StairsResult


class StairsDetector:
    """Heuristic stairs/curb detector. Drop-in for a trained model later."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.min_edge_density = float(settings.stairs_min_edge_density)
        self.min_row_width_ratio = 0.30
        self.enabled = bool(settings.stairs_detector_enabled)

    def detect(
        self, frame: np.ndarray, segmentation: SegmentationResult
    ) -> StairsResult:
        if not self.enabled:
            return StairsResult(False, 0.0, "disabled")

        cm = segmentation.class_map
        if cm is None:
            return StairsResult(False, 0.0, "no_class_map")
        cm = np.asarray(cm)
        if cm.ndim != 2 or cm.size == 0:
            return StairsResult(False, 0.0, "bad_class_map_shape")

        h, w = cm.shape[:2]
        # Bottom 30% — the strip the foot is about to enter.
        y0 = int(h * 0.70)
        cm_bot = cm[y0:, :]
        if cm_bot.size == 0:
            return StairsResult(False, 0.0, "empty_bottom")

        # Walkable mask in the bottom strip (Req 5.1).
        id_to_name_raw = (segmentation.metadata or {}).get("id_to_name", {})
        id_to_name = {
            int(k): str(v) for k, v in id_to_name_raw.items()
        }
        walkable_set = self._walkable_class_set()
        walkable_ids = {
            cid for cid, name in id_to_name.items() if name in walkable_set
        }
        if not walkable_ids:
            return StairsResult(False, 0.0, "no_walkable_classes_known")
        walkable_mask_bot = np.isin(cm_bot, list(walkable_ids))
        if not walkable_mask_bot.any():
            return StairsResult(False, 0.0, "no_walkable_in_bottom")

        # Pull the matching strip from the frame and downscale for bounded cost.
        if frame.shape[:2] != cm.shape[:2]:
            # The pipeline currently aligns frame and class_map shapes via
            # upscale_class_map. If they ever diverge, resize the class map
            # to the frame size to be safe.
            cm_resized = cv2.resize(
                cm.astype(np.int32),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            walkable_mask_bot = np.isin(
                cm_resized[int(frame.shape[0] * 0.70):, :], list(walkable_ids)
            )
            cm_bot = cm_resized[int(frame.shape[0] * 0.70):, :]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        y0_frame = int(frame.shape[0] * 0.70)
        gray_bot = gray[y0_frame:, :]

        if gray_bot.shape[1] > 256:
            scale = 256.0 / gray_bot.shape[1]
            new_h = max(8, int(round(gray_bot.shape[0] * scale)))
            gray_bot = cv2.resize(gray_bot, (256, new_h))
            walkable_mask_small = cv2.resize(
                walkable_mask_bot.astype(np.uint8),
                (gray_bot.shape[1], gray_bot.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            # Snap walkable mask to gray_bot shape (in case of off-by-one
            # from class-map vs frame alignment).
            if walkable_mask_bot.shape != gray_bot.shape:
                walkable_mask_small = cv2.resize(
                    walkable_mask_bot.astype(np.uint8),
                    (gray_bot.shape[1], gray_bot.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                walkable_mask_small = walkable_mask_bot

        if not walkable_mask_small.any():
            return StairsResult(False, 0.0, "no_walkable_after_resize")

        masked = np.where(walkable_mask_small, gray_bot, 0).astype(np.float32)
        # Sobel along Y picks up horizontal edges (rows with sudden luminance
        # change row-to-row) — exactly what stairs and curbs look like.
        sobel_y = cv2.Sobel(masked, cv2.CV_32F, 0, 1, ksize=3)
        row_sums = np.abs(sobel_y).sum(axis=1)
        per_row_walkable_width = walkable_mask_small.sum(axis=1).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            row_density = np.where(
                per_row_walkable_width > 0,
                row_sums / np.maximum(per_row_walkable_width, 1.0),
                0.0,
            )
        # Density values from cv2.Sobel are in raw 8-bit gradient units.
        # Normalize to [0, 1]-ish by dividing by 255 so the configured
        # min_edge_density matches the design's 0.08 default.
        row_density = row_density / 255.0
        max_row = int(np.argmax(row_density))
        max_density = float(row_density[max_row])
        max_row_walkable_width = int(per_row_walkable_width[max_row])
        total_walkable_width = int(walkable_mask_small.shape[1])
        spans_min_width = (
            max_row_walkable_width
            >= self.min_row_width_ratio * total_walkable_width
        )

        if max_density >= self.min_edge_density and spans_min_width:
            confidence = float(
                np.clip((max_density - self.min_edge_density) / 0.12, 0.0, 1.0)
            )
            return StairsResult(
                flag=True,
                confidence=confidence,
                rationale=(
                    f"row={max_row} density={max_density:.3f} "
                    f"span={max_row_walkable_width}/{total_walkable_width}"
                ),
            )
        return StairsResult(
            False,
            0.0,
            f"density={max_density:.3f} below {self.min_edge_density:.3f}",
        )

    def _walkable_class_set(self) -> set[str]:
        try:
            yaml_cfg = self.settings.yaml_config()
        except Exception:
            yaml_cfg = {}
        seg_cfg = (yaml_cfg or {}).get("segmentation", {})
        return set(seg_cfg.get("walkable_classes", []))


__all__ = ["StairsDetector"]
