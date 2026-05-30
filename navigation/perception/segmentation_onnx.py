"""ONNX-runtime segmenter — drop-in for YoloSegmenter on CPU-only cloud hosts.

Uses onnxruntime instead of torch/ultralytics so the server stays under
512MB RAM. The model must be exported first:

    python -c "from ultralytics import YOLO; YOLO('yolo26n-sem.pt').export(format='onnx', imgsz=256)"

This produces yolo26n-sem.onnx which is committed to the repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from navigation.config import Settings
from navigation.models import SegmentationResult
from navigation.perception.spatial import (
    _per_side_class_pixels,
    _per_side_walkable_ratio,
)
from navigation.perception.segmentation import _region_weight_map


# Cityscapes 19-class names in order (matches yolo26n-sem.pt training).
_CITYSCAPES_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]


class OnnxSegmenter:
    """Semantic segmenter backed by onnxruntime. No torch required."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._session: Any = None
        self._last_segmentation: SegmentationResult | None = None
        self._id_to_name = {i: n for i, n in enumerate(_CITYSCAPES_NAMES)}

    def _load(self) -> Any:
        if self._session is not None:
            return self._session
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("pip install onnxruntime") from e
        model_path = self.settings.yolo_model_path
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}. "
                "Export it with: python -c \"from ultralytics import YOLO; "
                "YOLO('yolo26n-sem.pt').export(format='onnx', imgsz=256)\""
            )
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        self._session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        return self._session

    @property
    def last_segmentation(self) -> SegmentationResult | None:
        return self._last_segmentation

    @property
    def last_results(self) -> None:
        return None  # No Ultralytics results in ONNX path

    @property
    def is_semantic(self) -> bool:
        return True

    def predict(self, frame: np.ndarray, *, dry_run: bool = False) -> SegmentationResult:
        if dry_run:
            seg = self._mock(frame)
            self._last_segmentation = seg
            return seg
        return self._predict_onnx(frame)

    def _predict_onnx(self, frame: np.ndarray) -> SegmentationResult:
        session = self._load()
        display_h, display_w = frame.shape[:2]

        # Resize to model input size (256×256 by default).
        imgsz = self.settings.yolo_imgsz or 256
        inp = cv2.resize(frame, (imgsz, imgsz))
        inp = inp[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, normalize
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis]    # HWC→NCHW

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: inp})

        # YOLO semantic ONNX output: [1, num_classes, H, W] logits.
        logits = outputs[0]  # shape (1, 19, H, W)
        if logits.ndim == 4:
            logits = logits[0]  # (19, H, W)
        class_map = np.argmax(logits, axis=0).astype(np.int32)  # (H, W)

        # Upscale to display resolution.
        if class_map.shape != (display_h, display_w):
            class_map = cv2.resize(
                class_map.astype(np.float32),
                (display_w, display_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)

        seg = self._parse_class_map(class_map, display_h, display_w)
        self._last_segmentation = seg
        return seg

    def _parse_class_map(
        self, class_map: np.ndarray, h: int, w: int
    ) -> SegmentationResult:
        yaml_cfg = self.settings.yaml_config().get("segmentation", {})
        obstacle_set = set(yaml_cfg.get("obstacle_classes", []))
        walkable_set = set(yaml_cfg.get("walkable_classes", []))

        weight_map = _region_weight_map((h, w))
        obstacle_pixels = 0
        obstacle_weighted = 0.0
        walkable_pixels = 0
        total = int(class_map.size)
        counts: dict[str, int] = {}
        present: list[str] = []

        for cls_id in np.unique(class_map):
            name = self._id_to_name.get(int(cls_id), str(int(cls_id)))
            mask = class_map == cls_id
            pixels = int(mask.sum())
            counts[name] = pixels
            if name in obstacle_set:
                obstacle_pixels += pixels
                obstacle_weighted += float(weight_map[mask].sum())
            if name in walkable_set:
                walkable_pixels += pixels
            if pixels > 0:
                present.append(name)

        walkable_ratio = walkable_pixels / max(total, 1)
        per_side_pixels = _per_side_class_pixels(class_map, self._id_to_name, weight_map)
        per_side_walk = _per_side_walkable_ratio(class_map, self._id_to_name, walkable_set)

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
                "shape": [h, w],
                "id_to_name": self._id_to_name,
            },
        )

    def _mock(self, frame: np.ndarray) -> SegmentationResult:
        h, w = frame.shape[:2]
        yaml_cfg = self.settings.yaml_config().get("segmentation", {})
        walkable = yaml_cfg.get("walkable_classes", ["sidewalk"])
        obstacles = yaml_cfg.get("obstacle_classes", ["person"])
        walk_name = walkable[0] if walkable else "sidewalk"
        obs_name = obstacles[0] if obstacles else "person"
        center_obstacle = int(h * w * 0.05)
        walkable_pixels = int(h * w * 0.25)
        return SegmentationResult(
            class_names=[walk_name, obs_name, "road", "sky", "building"],
            obstacle_pixels=center_obstacle,
            obstacle_pixels_weighted=float(center_obstacle),
            walkable_ratio=walkable_pixels / max(h * w, 1),
            per_side_class_pixels={"left": {}, "center": {obs_name: float(center_obstacle)}, "right": {}},
            per_side_walkable_ratio={"left": 0.25, "center": 0.25, "right": 0.25},
            metadata={"mock": True, "semantic": True, "shape": [h, w],
                      "id_to_name": self._id_to_name},
        )
