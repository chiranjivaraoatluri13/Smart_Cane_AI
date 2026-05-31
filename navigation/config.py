"""Settings and YAML config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    camera_index: int = 0
    camera_backend: str = "auto"
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 10
    inference_width: int = 0
    inference_height: int = 0
    process_every_n_frames: int = 1

    # Inference image size for the segmenter (0 = use the processor/model
    # default). Kept as a generic knob; the fast profile lowers it for CPU.
    inference_imgsz: int = 0

    # Segmentation backend. Options:
    #   "segformer"      — ADE20K SegFormer via transformers (accurate, slower on CPU)
    #   "segformer_onnx" — ADE20K SegFormer via onnxruntime INT8 (fast, recommended)
    segmenter_backend: str = "segformer_onnx"
    segformer_model_id: str = "nvidia/segformer-b2-finetuned-ade-512-512"
    segformer_device: str = "auto"
    # Path to the exported ONNX model (used when segmenter_backend=segformer_onnx).
    # Export with: python scripts/export_segformer_onnx.py
    segformer_onnx_path: str = "segformer_b0_ade20k_int8.onnx"

    unidepth_model_path: str = ""
    unidepth_device: str = "auto"

    care_endpoint: str = "http://127.0.0.1:8000/predict"
    care_timeout_sec: float = 2.0
    use_care_http: bool = False

    openai_api_key: str = ""
    openai_api_base: str = "http://127.0.0.1:11434/v1"
    openai_model: str = "llama3.1"
    use_llm: bool = False

    command_cooldown_sec: float = 2.0
    repeat_command_suppress: bool = True
    hazard_obstacle_ratio: float = 0.02
    # Anti-jitter knobs (used by CommandValidator):
    # - dwell_frames: a non-STOP command must be the reasoner's pick for at
    #   least N consecutive inference frames before it's spoken. Kills
    #   left/right/left/right wobble on stationary scenes.
    # - min_speech_gap_sec: minimum time between any two non-STOP utterances.
    #   STOP bypasses both because safety always speaks.
    command_dwell_frames: int = 2
    min_speech_gap_sec: float = 1.5
    # Sticky-stop hysteresis (SpatialReasoner). Once vision_stop fires, hold
    # STOP for this many subsequent frames even if the obstacle ratio dips
    # back below the rising threshold. Combined with a 0.6× falling-edge
    # threshold, this prevents STOP/SLOW_DOWN oscillation around the boundary.
    stop_hold_frames: int = 8

    # Proximity alert tuning (AlertTracker). The default model hallucinates
    # street classes (car/bicycle/truck/person) on indoor or empty scenes;
    # these guardrails keep it quiet. Set alerts_enabled=false to disable
    # "X approaching" announcements entirely.
    alerts_enabled: bool = True
    alert_cooldown_sec: float = 5.0
    alert_global_cooldown_sec: float = 4.0
    alert_min_weighted_pixels: float = 1500.0
    alert_growth_factor: float = 1.5
    alert_max_simultaneous_categories: int = 2

    # Spatial-aware natural-language guidance — defaults from
    # config/default.yaml's spatial: / voice: blocks. These are surfaced as
    # Settings fields so downstream components (VoiceQueue,
    # PhraseComposer, SpatialReasoner) don't have to re-parse YAML on the
    # hot path. Real values come from yaml_config() at startup.
    min_lane_walkable_ratio: float = 0.10
    status_update_interval_sec: float = 10.0
    voice_cooldowns: dict[str, float] = Field(
        default_factory=lambda: {
            "vision_stop": 0.0,
            "directional_warning": 2.0,
            "map_turn": 8.0,
            "approach_alert": 3.0,
            "status_update": 10.0,
        }
    )
    phrases_path: str = "config/phrases.yaml"
    composer_seed: int | None = None
    benchmark_mode: bool = False

    use_map_guidance: bool = False
    dest_lat: float | None = None
    dest_lon: float | None = None
    current_lat: float | None = None
    current_lon: float | None = None
    current_heading_deg: float = 0.0
    route_at_dest_m: float = 15.0
    route_off_route_m: float = 30.0
    route_bearing_align_deg: float = 25.0
    route_debug_path: str = "output/route.json"

    tts_enabled: bool = True
    tts_rate: int = 175

    config_path: Path = Path("config/default.yaml")

    def yaml_config(self) -> dict[str, Any]:
        return load_yaml_config(self.config_path)

    def seg_class_config(self) -> dict[str, Any]:
        """Return the walkable/obstacle/hazard class lists for the segmenter.

        The active backend is the ADE20K SegFormer, so this returns the
        ``ade20k_segmentation`` block. Centralizing it means the reasoner,
        stairs detector, and depth proxy all read the label set that matches
        the model's outputs.
        """
        full = self.yaml_config()
        block = full.get("ade20k_segmentation")
        if block:
            return block
        return full.get("segmentation", {})

    @property
    def map_destination_set(self) -> bool:
        return self.dest_lat is not None and self.dest_lon is not None

    @property
    def map_position_set(self) -> bool:
        return self.current_lat is not None and self.current_lon is not None

    @property
    def map_ready(self) -> bool:
        return (
            self.use_map_guidance
            and self.map_destination_set
            and self.map_position_set
        )


def load_settings() -> Settings:
    return Settings()


def apply_fast_profile(settings: Settings) -> Settings:
    """Tune settings for smoother real-time on a laptop CPU."""
    return settings.model_copy(
        update={
            "frame_width": 320,
            "frame_height": 240,
            "inference_width": 256,
            "inference_height": 192,
            "inference_imgsz": 256,
            "process_every_n_frames": 3,
            "target_fps": 24,
            "use_llm": False,
            "use_care_http": False,
            "command_cooldown_sec": 1.0,
        }
    )


def apply_demo_profile(settings: Settings) -> Settings:
    """Fast + frequent voice for live demonstrations."""
    return apply_fast_profile(settings).model_copy(
        update={
            "command_cooldown_sec": 0.8,
            "repeat_command_suppress": True,
            "hazard_obstacle_ratio": 0.03,
        }
    )
