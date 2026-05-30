"""Colored segmentation overlays for laptop preview."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from navigation.perception.segmentation import YoloSegmenter

# Cityscapes-style BGR tints (reference panoptic look)
CITYSCAPES_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "road": (255, 128, 0),        # Blue for walkable roads
    "sidewalk": (255, 180, 0),    # Bright blue for sidewalks
    "building": (70, 70, 70),
    "wall": (102, 102, 156),
    "fence": (190, 153, 153),
    "pole": (153, 153, 153),
    "traffic light": (0, 64, 255),
    "traffic sign": (220, 220, 0),
    "vegetation": (0, 128, 0),
    "terrain": (255, 140, 0),     # Light blue for terrain
    "sky": (180, 165, 255),
    "person": (0, 255, 255),
    "rider": (0, 200, 255),
    "car": (0, 255, 0),
    "truck": (0, 255, 128),
    "bus": (0, 255, 64),
    "train": (128, 255, 0),
    "motorcycle": (0, 220, 255),
    "bicycle": (0, 200, 255),
}

CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    **CITYSCAPES_COLORS_BGR,
    "street": (64, 64, 64),
    "tree": (0, 140, 0),
    "crosswalk": (200, 200, 200),
}

_DEFAULT_COLOR_BGR = (200, 200, 200)
_OVERLAY_ALPHA = 0.40  # Reduced from 0.55 - less aggressive overlay
_WINDOW = "assistive-nav segmentation"


def _blend_region(
    base: np.ndarray, color_bgr: tuple[int, int, int], mask: np.ndarray, alpha: float
) -> None:
    if not mask.any():
        return
    tint = np.array(color_bgr, dtype=np.float32)
    base[mask] = base[mask] * (1.0 - alpha) + tint * alpha


def overlay_mock(frame: np.ndarray) -> np.ndarray:
    """Cityscapes-style demo colors without GPU (for --dry-run)."""
    h, w = frame.shape[:2]
    out = frame.astype(np.float32)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    _blend_region(
        out,
        CITYSCAPES_COLORS_BGR["sky"],
        yy < h // 3,
        _OVERLAY_ALPHA,
    )
    _blend_region(
        out,
        CITYSCAPES_COLORS_BGR["building"],
        (yy >= h // 3)
        & (yy < h * 2 // 3)
        & (xx > w // 4)
        & (xx < 3 * w // 4),
        _OVERLAY_ALPHA * 0.75,
    )
    _blend_region(
        out,
        CITYSCAPES_COLORS_BGR["road"],
        yy >= h * 2 // 3,
        _OVERLAY_ALPHA,
    )
    _blend_region(
        out,
        CITYSCAPES_COLORS_BGR["sidewalk"],
        (yy >= h * 2 // 3) & (xx <= w // 5),
        _OVERLAY_ALPHA * 0.85,
    )
    _blend_region(
        out,
        CITYSCAPES_COLORS_BGR["vegetation"],
        (yy >= h // 3)
        & (yy < h * 2 // 3)
        & ((xx <= w // 4) | (xx >= 3 * w // 4)),
        _OVERLAY_ALPHA * 0.85,
    )
    cx, cy = w // 2, h // 2
    person_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < (min(h, w) * 0.12) ** 2
    _blend_region(out, CITYSCAPES_COLORS_BGR["person"], person_mask, _OVERLAY_ALPHA)
    car_mask = (yy >= h * 2 // 3) & (xx > w // 3) & (xx < 2 * w // 3)
    _blend_region(out, CITYSCAPES_COLORS_BGR["car"], car_mask, _OVERLAY_ALPHA * 0.9)
    return out.astype(np.uint8)


def overlay_from_class_map(
    frame: np.ndarray,
    class_map: np.ndarray,
    id_to_name: dict[int, str],
    *,
    alpha: float = _OVERLAY_ALPHA,
) -> np.ndarray:
    """Tint each semantic class over the camera frame (Cityscapes dense map)."""
    import cv2

    h, w = frame.shape[:2]
    cm = np.asarray(class_map)
    if cm.ndim == 3:
        cm = cm[0]
    if cm.shape[:2] != (h, w):
        cm = cv2.resize(cm.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)

    out = frame.astype(np.float32)
    for cls_id in np.unique(cm):
        name = id_to_name.get(int(cls_id), str(int(cls_id)))
        if name == "sky":
            continue
        mask = cm == cls_id
        color = CLASS_COLORS_BGR.get(name, _DEFAULT_COLOR_BGR)
        _blend_region(out, color, mask, alpha)
    return out.astype(np.uint8)


def overlay_from_ultralytics(results: Any, frame: np.ndarray) -> np.ndarray:
    """Prefer dense semantic tint; fall back to Ultralytics plot() for instance seg."""
    r0 = results[0]
    sem = getattr(r0, "semantic_mask", None)
    if sem is not None and sem.data is not None:
        id_to_name = {int(k): str(v) for k, v in (r0.names or {}).items()}
        class_map = sem.data.cpu().numpy()
        return overlay_from_class_map(frame, class_map, id_to_name)
    return np.asarray(r0.plot())


def overlay_from_masks(
    frame: np.ndarray,
    class_names: list[str],
    masks: list[Any],
) -> np.ndarray:
    """Fallback: tint each instance mask with a class color."""
    import cv2

    h, w = frame.shape[:2]
    out = frame.astype(np.float32)
    for name, mask in zip(class_names, masks):
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[0]
        if m.shape[:2] != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        binary = m > 0.5 if m.dtype != bool else m
        color = CLASS_COLORS_BGR.get(name, _DEFAULT_COLOR_BGR)
        _blend_region(out, color, binary, _OVERLAY_ALPHA)
    return out.astype(np.uint8)


def render_overlay(
    frame: np.ndarray,
    *,
    segmenter: YoloSegmenter,
    dry_run: bool,
) -> np.ndarray:
    if dry_run:
        return overlay_mock(frame)

    seg = segmenter.last_segmentation
    if seg is not None and seg.class_map is not None:
        id_to_name = {}
        meta = seg.metadata.get("id_to_name")
        if isinstance(meta, dict):
            id_to_name = {int(k): str(v) for k, v in meta.items()}
        if not id_to_name:
            results = segmenter.last_results
            if results:
                id_to_name = {
                    int(k): str(v) for k, v in (results[0].names or {}).items()
                }
        return overlay_from_class_map(frame, seg.class_map, id_to_name)

    results = segmenter.last_results
    if results is not None:
        try:
            return overlay_from_ultralytics(results, frame)
        except Exception:
            pass

    if seg is not None and seg.masks:
        return overlay_from_masks(frame, seg.class_names, seg.masks)
    return frame.copy()


def show_frame(
    bgr: np.ndarray,
    *,
    window: str = _WINDOW,
    wait_ms: int = 1,
) -> int:
    """Show frame; returns key code (``ord('q')`` to quit). ``wait_ms=0`` blocks."""
    import cv2

    cv2.imshow(window, bgr)
    return cv2.waitKey(wait_ms) & 0xFF


def close_windows() -> None:
    import cv2

    cv2.destroyAllWindows()


def save_overlay(bgr: np.ndarray, path: Path) -> Path:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)
    return path
