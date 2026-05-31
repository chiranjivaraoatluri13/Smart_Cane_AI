"""Tests for the heuristic interpreter path."""

import numpy as np

from navigation.config import Settings
from navigation.models import CareResult, DepthResult, PerceptionBundle, SegmentationResult
from navigation.reasoning.llm import NavigationInterpreter


def test_heuristic_stop_on_hazard():
    settings = Settings()
    interp = NavigationInterpreter(settings)
    bundle = PerceptionBundle(
        frame_id=0,
        segmentation=SegmentationResult(obstacle_pixels=1000),
        depth=DepthResult(obstacle_depth_m=0.5),
        care=CareResult(hazard_detected=True, safety_score=0.2),
    )
    decision = interp.interpret(bundle)
    assert decision.command.value == "stop"


def test_use_llm_false_skips_api():
    settings = Settings(use_llm=False)
    interp = NavigationInterpreter(settings)
    bundle = PerceptionBundle(
        frame_id=0,
        segmentation=SegmentationResult(),
        depth=DepthResult(),
        care=CareResult(safety_score=0.9),
    )
    decision = interp.interpret(bundle)
    assert decision.command.value == "go_forward"
