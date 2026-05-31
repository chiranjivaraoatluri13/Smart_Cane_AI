"""Spoken labels for obstacle and hazard classes in guidance phrases.

Maps ADE20K segmentation class names to user-facing words (e.g. ``chair``,
``steps``, ``street light``) and ranks them when several are visible at once.
"""

from __future__ import annotations

from navigation.reasoning.alerts import CATEGORY_PRIORITY, CLASS_TO_CATEGORY

# Elevation / trip hazards → one spoken label.
_HAZARD_LABEL: dict[str, str] = {
    "stairs": "steps",
    "stairway": "steps",
    "step": "steps",
}

# Friendlier wording for awkward ADE20K names.
_OBSTACLE_LABEL: dict[str, str] = {
    "bannister": "handrail",
    "signboard": "sign",
    "streetlight": "street light",
    "minibike": "scooter",
    "rider": "cyclist",
    "motorcycle": "scooter",
}

# Higher number → named first when multiple hazards share a side.
SPOKEN_HAZARD_PRIORITY: dict[str, int] = {
    **CATEGORY_PRIORITY,
    "steps": 95,
    "escalator": 94,
    "van": 75,
    "pole": 50,
    "chair": 45,
    "table": 44,
    "sofa": 43,
    "bench": 42,
    "column": 41,
    "door": 40,
    "fence": 38,
    "railing": 37,
    "bannister": 36,
    "handrail": 36,
    "wall": 35,
    "building": 30,
    "plant": 25,
    "sign": 28,
    "street light": 27,
}


def spoken_label(cls_name: str) -> str:
    """Return the word(s) read aloud for a segmentation class."""
    if cls_name in _HAZARD_LABEL:
        return _HAZARD_LABEL[cls_name]
    if cls_name in _OBSTACLE_LABEL:
        return _OBSTACLE_LABEL[cls_name]
    if cls_name in CLASS_TO_CATEGORY:
        cat = CLASS_TO_CATEGORY[cls_name]
        if cat == "heavy_vehicle" and cls_name in ("truck", "bus", "train"):
            return cls_name
        if cat == "heavy_vehicle":
            return "heavy vehicle"
        return cat
    return cls_name.replace("_", " ")


def hazard_key_for_class(
    cls_name: str,
    obstacle_classes: set[str],
    hazard_classes: set[str],
) -> str | None:
    """Return a spoken hazard key if the class should be named in guidance."""
    if cls_name in hazard_classes or cls_name in obstacle_classes:
        return spoken_label(cls_name)
    if cls_name in CLASS_TO_CATEGORY:
        return spoken_label(cls_name)
    return None


def hazard_priority(label: str) -> int:
    return SPOKEN_HAZARD_PRIORITY.get(label, 10)


__all__ = [
    "SPOKEN_HAZARD_PRIORITY",
    "hazard_key_for_class",
    "hazard_priority",
    "spoken_label",
]
