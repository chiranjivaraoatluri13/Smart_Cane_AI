"""Shared schemas for perception, reasoning, and output."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

Side = Literal["left", "center", "right"]
SIDES: tuple[Side, ...] = ("left", "center", "right")


class NavigationCommand(str, Enum):
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    GO_FORWARD = "go_forward"
    SLOW_DOWN = "slow_down"
    STOP = "stop"


class Position(BaseModel):
    """Live GPS + compass reading from a wearable/phone client.

    All fields are optional so a request can carry partial state (e.g. GPS
    without heading from a phone with no magnetometer); callers should fall
    back to ``Settings`` defaults when a field is None.
    """

    lat: float | None = None
    lon: float | None = None
    heading_deg: float | None = Field(
        default=None,
        description="Compass heading 0-360 (0 = north). None when unknown.",
    )
    accuracy_m: float | None = Field(
        default=None,
        description="GPS accuracy in meters from navigator.geolocation.",
    )

    @property
    def has_coords(self) -> bool:
        return self.lat is not None and self.lon is not None


class SegmentationResult(BaseModel):
    """Pixel-level scene understanding from YOLO26 semantic or instance models."""

    class_names: list[str] = Field(
        default_factory=list,
        description="Classes present in frame (instances or semantic)",
    )
    masks: list[Any] = Field(
        default_factory=list,
        description="Instance masks only (-seg models)",
    )
    class_map: Any | None = Field(
        default=None,
        description="HxW int class-id map (-sem Cityscapes models)",
    )
    obstacle_pixels: int = 0
    obstacle_pixels_weighted: float = Field(
        default=0.0,
        description=(
            "Region-weighted obstacle count: bottom-center matters most, "
            "top half ignored. Falls back to raw obstacle_pixels when "
            "spatial info is unavailable."
        ),
    )
    walkable_ratio: float = Field(ge=0.0, le=1.0, default=0.0)
    # Spatial-aware natural-language guidance — Requirement 1 / 2.
    # Both default to None for back-compat; populated by _parse_semantic.
    per_side_class_pixels: dict[Side, dict[str, float]] | None = Field(
        default=None,
        description=(
            "Region-weighted pixel counts per class, split into left/center/right "
            "thirds of the frame. None when the segmenter path didn't produce "
            "spatial info (instance models, missing class map)."
        ),
    )
    per_side_walkable_ratio: dict[Side, float] | None = Field(
        default=None,
        description=(
            "Walkable-pixel fraction per side, in [0, 1]. None when the "
            "segmenter path couldn't compute it."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class DepthResult(BaseModel):
    """Monocular depth map summary from UniDepthV2."""

    depth_map: Any | None = None
    min_depth_m: float | None = None
    center_depth_m: float | None = None
    obstacle_depth_m: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CareResult(BaseModel):
    """Safety-aware movement signal from CARE."""

    safe_direction_deg: float | None = Field(
        default=None, description="Suggested heading offset in degrees"
    )
    safety_score: float = Field(ge=0.0, le=1.0, default=0.5)
    hazard_detected: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)


class NavigationDecision(BaseModel):
    """Structured command produced by Llama 3.1 interpretation."""

    command: NavigationCommand
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    rationale: str = ""
    speak: bool = True


class PerceptionBundle(BaseModel):
    """Aggregated outputs passed into reasoning layers."""

    frame_id: int
    segmentation: SegmentationResult
    depth: DepthResult
    care: CareResult
