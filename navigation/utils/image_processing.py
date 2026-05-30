"""Image processing utilities for frame resizing and upscaling."""

from __future__ import annotations

import cv2
import numpy as np

from navigation.config import Settings


def inference_size(settings: Settings, frame: np.ndarray) -> tuple[int, int] | None:
    """Return (width, height) for YOLO input, or None to use full frame."""
    w = settings.inference_width
    h = settings.inference_height
    if w > 0 and h > 0:
        return (w, h)
    return None


def resize_for_inference(frame: np.ndarray, settings: Settings) -> np.ndarray:
    size = inference_size(settings, frame)
    if size is None:
        return frame
    return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)


def upscale_class_map(class_map: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if class_map.shape[0] == target_h and class_map.shape[1] == target_w:
        return class_map
    return cv2.resize(
        class_map.astype(np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)
