"""YOLO26 / YOLO11 Cityscapes semantic segmentation adapter."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from navigation.config import Settings
from navigation.models import SegmentationResult, SIDES
from navigation.perception.spatial import (
    _per_side_class_pixels,
    _per_side_walkable_ratio,
    empty_per_side_counts,
    empty_per_side_walkable,
)
from navigation.utils.image_processing import resize_for_inference, upscale_class_map


def is_semantic_model(model_path: str) -> bool:
    """True when weights are Ultralytics semantic (-sem) checkpoints."""
    stem = model_path.lower().rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return "-sem" in stem or stem.endswith("sem.pt")


@lru_cache(maxsize=8)
def _region_weight_map(shape: tuple[int, int]) -> np.ndarray:
    """Per-pixel importance weights for "is this in the walking path?".

    Bottom-center counts most (where the user is heading), top half is ignored
    (sky / distant buildings), and edges count half (peripheral hazards still
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


class YoloSegmenter:
    """Wraps Ultralytics YOLO semantic (-sem) or instance (-seg) models."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any = None
        self._last_results: Any = None
        self._last_segmentation: SegmentationResult | None = None
        self._semantic = is_semantic_model(settings.yolo_model_path)

    @property
    def last_results(self) -> Any:
        """Raw Ultralytics result list from the latest non-mock ``predict``."""
        return self._last_results

    @property
    def last_segmentation(self) -> SegmentationResult | None:
        return self._last_segmentation

    @property
    def is_semantic(self) -> bool:
        return self._semantic

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Install vision extras: pip install -e '.[vision]'"
            ) from e
        self._model = YOLO(self.settings.yolo_model_path)
        self._semantic = is_semantic_model(self.settings.yolo_model_path)
        return self._model

    def predict(self, frame: np.ndarray, *, dry_run: bool = False) -> SegmentationResult:
        self._last_results = None
        if dry_run:
            seg = self._mock(frame)
            self._last_segmentation = seg
            return seg

        display_h, display_w = frame.shape[:2]
        infer_frame = resize_for_inference(frame, self.settings)

        model = self._load_model()
        imgsz = self.settings.yolo_imgsz
        if imgsz <= 0:
            imgsz = max(infer_frame.shape[0], infer_frame.shape[1])
        kwargs: dict[str, Any] = {"verbose": False, "imgsz": imgsz}
        if not self._semantic:
            # Confidence threshold only applies to instance/object detection;
            # semantic models produce a dense class map with no per-detection score.
            kwargs["conf"] = self.settings.yolo_confidence

        results = model.predict(infer_frame, **kwargs)
        self._last_results = results
        seg = self._parse_results(results)
        if seg.class_map is not None:
            seg.class_map = upscale_class_map(seg.class_map, display_h, display_w)
            meta = dict(seg.metadata)
            meta["shape"] = [display_h, display_w]
            seg.metadata = meta
        self._last_segmentation = seg
        return seg

    def _segmentation_yaml(self) -> dict[str, Any]:
        full = self.settings.yaml_config()
        # Use the COCO config block when running an instance-seg model,
        # and the Cityscapes block for semantic models. This ensures the
        # obstacle/walkable class lists match what the model actually outputs.
        if not self._semantic:
            coco = full.get("coco_segmentation", {})
            if coco:
                return coco
        return full.get("segmentation", {})

    def _parse_results(self, results: Any) -> SegmentationResult:
        if self._semantic:
            return self._parse_semantic(results)
        return self._parse_instance(results)

    def _parse_semantic(self, results: Any) -> SegmentationResult:
        yaml_cfg = self._segmentation_yaml()
        obstacle_set = set(yaml_cfg.get("obstacle_classes", []))
        walkable_set = set(yaml_cfg.get("walkable_classes", []))

        class_map: np.ndarray | None = None
        id_to_name: dict[int, str] = {}
        present: list[str] = []

        for r in results:
            id_to_name = {int(k): str(v) for k, v in (r.names or {}).items()}
            sem = getattr(r, "semantic_mask", None)
            if sem is None or sem.data is None:
                continue
            class_map = sem.data.cpu().numpy().astype(np.int32)
            if class_map.ndim == 3:
                class_map = class_map[0]
            break

        if class_map is None:
            return SegmentationResult(
                metadata={"error": "no_semantic_mask", "semantic": True},
            )

        total = int(class_map.size)
        obstacle_pixels = 0
        walkable_pixels = 0
        counts: dict[str, int] = {}

        # Region weight map: bottom-center matters most, top half is ignored.
        # This is what stops a building in the periphery from triggering STOP.
        weight_map = _region_weight_map(class_map.shape)
        obstacle_weighted = 0.0

        for cls_id in np.unique(class_map):
            name = id_to_name.get(int(cls_id), str(int(cls_id)))
            mask = class_map == cls_id
            pixels = int(mask.sum())
            counts[name] = pixels
            if name in obstacle_set:
                obstacle_pixels += pixels
                obstacle_weighted += float(weight_map[mask].sum())
            if name in walkable_set:
                walkable_pixels += pixels
            if pixels > 0 and name not in present:
                present.append(name)

        walkable_ratio = walkable_pixels / max(total, 1)
        # Per-side spatial fields (Requirements 1, 2). Computed via the
        # spatial helpers; sum of per-side counts equals the global weighted
        # count by construction (Property 1). The walkable-class set comes
        # from config/default.yaml so it stays in sync with the heuristic.
        per_side_pixels = _per_side_class_pixels(class_map, id_to_name, weight_map)
        per_side_walk = _per_side_walkable_ratio(
            class_map, id_to_name, walkable_set
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
                "pixel_counts": counts,
                "shape": list(class_map.shape),
                "id_to_name": id_to_name,
            },
        )

    def _parse_instance(self, results: Any) -> SegmentationResult:
        names: list[str] = []
        masks: list[Any] = []
        obstacle_pixels = 0
        obstacle_weighted = 0.0
        walkable_pixels = 0
        total = 1

        yaml_cfg = self._segmentation_yaml()
        obstacle_set = set(yaml_cfg.get("obstacle_classes", []))
        walkable_set = set(yaml_cfg.get("walkable_classes", []))

        for r in results:
            id_to_name = r.names or {}
            if r.masks is None:
                continue
            for i, mask_tensor in enumerate(r.masks.data):
                cls_id = int(r.boxes.cls[i]) if r.boxes is not None else -1
                name = id_to_name.get(cls_id, str(cls_id))
                names.append(name)
                mask = mask_tensor.cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                masks.append(mask)
                pixels = int(mask.sum())
                total += pixels
                if name in obstacle_set:
                    obstacle_pixels += pixels
                    weight_map = _region_weight_map(mask.shape[:2])
                    obstacle_weighted += float(weight_map[mask > 0.5].sum())
                if name in walkable_set:
                    walkable_pixels += pixels

        walkable_ratio = walkable_pixels / max(total, 1)
        return SegmentationResult(
            class_names=names,
            masks=masks,
            obstacle_pixels=obstacle_pixels,
            obstacle_pixels_weighted=obstacle_weighted,
            walkable_ratio=min(1.0, walkable_ratio),
            metadata={"semantic": False},
        )

    def _mock(self, frame: np.ndarray) -> SegmentationResult:
        h, w = frame.shape[:2]
        yaml_cfg = self._segmentation_yaml()
        walkable = yaml_cfg.get("walkable_classes", ["sidewalk"])
        obstacles = yaml_cfg.get("obstacle_classes", ["person"])
        walk_name = walkable[0] if walkable else "sidewalk"
        obs_name = obstacles[0] if obstacles else "person"
        center_obstacle = int(h * w * 0.05)
        walkable_pixels = int(h * w * 0.25)
        # Mock obstacle is in the bottom-center; weight equals raw count there.
        obstacle_weighted = float(center_obstacle)
        # Mock per-side fields: bottom-center mock places the obstacle in
        # `center` and walkable surface across all three (Requirement 1.6 —
        # downstream code never sees None from the mock).
        per_side_pixels: dict[str, dict[str, float]] = {
            s: {} for s in SIDES
        }
        per_side_pixels["center"][obs_name] = float(obstacle_weighted)
        per_side_walk: dict[str, float] = {
            "left": float(walkable_pixels) / max(h * w, 1) * 3,
            "center": float(walkable_pixels) / max(h * w, 1) * 3,
            "right": float(walkable_pixels) / max(h * w, 1) * 3,
        }
        # Clamp into [0, 1] — multiplying by 3 above just compensates for
        # the third-of-frame normalization, but cap to be safe.
        per_side_walk = {k: max(0.0, min(1.0, v)) for k, v in per_side_walk.items()}
        return SegmentationResult(
            class_names=[walk_name, obs_name, "road", "sky", "building", "vegetation"],
            obstacle_pixels=center_obstacle,
            obstacle_pixels_weighted=obstacle_weighted,
            walkable_ratio=walkable_pixels / max(h * w, 1),
            per_side_class_pixels=per_side_pixels,  # type: ignore[arg-type]
            per_side_walkable_ratio=per_side_walk,  # type: ignore[arg-type]
            metadata={"mock": True, "semantic": True, "shape": [h, w]},
        )
