"""End-to-end smoke tests for the pipeline runner.

These exercise the perception → reasoning → output orchestration in
``navigation.pipeline.runner``. The segmenter is replaced by a thin test
double that feeds a synthetic class map through the *real* SegFormer parser
(``SegformerSegmenter._parse_class_map``) — so no model download or network
is needed, but the parsing/weighting logic under test is the production code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from navigation.config import Settings
from navigation.models import (
    CareResult,
    DepthResult,
    NavigationCommand,
    PerceptionBundle,
    SegmentationResult,
)
from navigation.output.tts import SpeechEngine
from navigation.output.validator import CommandValidator
from navigation.perception.depth import DepthEstimator
from navigation.perception.segmentation_segformer import SegformerSegmenter
from navigation.pipeline.runner import process_frame
from navigation.reasoning.care import CareNavigator
from navigation.reasoning.llm import NavigationInterpreter


FIXTURE = Path("tests/fixtures/sample.jpg")


def _silent_settings(**overrides) -> Settings:
    base = dict(
        tts_enabled=False,           # never invoke pyttsx3 in tests
        use_llm=False,                # heuristic path only
        use_care_http=False,          # no HTTP
        use_map_guidance=False,
        process_every_n_frames=1,
        command_cooldown_sec=0.0,
        repeat_command_suppress=False,
    )
    base.update(overrides)
    return Settings(**base)


class _StubSegmenter:
    """Test double: produces a fixed SegmentationResult via the real parser.

    No model weights, no transformers inference — just the production
    ``_parse_class_map`` driven by a synthetic class map. This keeps the
    pipeline orchestration tests fast and offline while still exercising the
    real perception parsing/weighting code.
    """

    def __init__(self, settings: Settings, class_map: np.ndarray,
                 id_to_name: dict[int, str]):
        parser = SegformerSegmenter(settings)
        parser._id_to_name = id_to_name
        h, w = class_map.shape[:2]
        self._result = parser._parse_class_map(class_map.astype(np.int32), h, w)
        self.last_segmentation = self._result
        self.last_results = None
        self.is_semantic = True

    def predict(self, frame: np.ndarray) -> SegmentationResult:
        return self._result


def _walkable_with_center_obstacle(h: int = 240, w: int = 320) -> _StubSegmenter:
    """floor everywhere, a person blob low-and-center (triggers a real decision)."""
    cm = np.zeros((h, w), dtype=np.int32)         # 0 = floor (walkable)
    cm[int(h * 0.6):, int(w * 0.35):int(w * 0.65)] = 1   # 1 = person (obstacle)
    return _StubSegmenter(
        _silent_settings(), cm, {0: "floor", 1: "person"}
    )


def test_process_frame_emits_valid_command():
    """One frame through the chain produces a NavigationCommand-shaped record."""
    settings = _silent_settings()
    segmenter = _walkable_with_center_obstacle()
    depth_est = DepthEstimator(settings)
    care = CareNavigator(settings)
    interpreter = NavigationInterpreter(settings)
    validator = CommandValidator(settings)
    tts = SpeechEngine(settings)

    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    record = process_frame(
        frame,
        frame_id=0,
        settings=settings,
        segmenter=segmenter,
        depth_est=depth_est,
        care=care,
        interpreter=interpreter,
        validator=validator,
        tts=tts,
    )

    assert "command" in record
    assert record["command"] in {c.value for c in NavigationCommand}
    assert 0.0 <= record["confidence"] <= 1.0
    assert isinstance(record["phrase"], str) and record["phrase"]
    assert "frame_id" in record


def test_region_weighted_obstacle_ignores_top_periphery():
    """A class only in the top of the frame should not trigger STOP.

    Regression test for the "always says STOP" bug — content filling the
    upper half of the frame must not count the same as a person dead-center.
    """
    from navigation.perception.segmentation_base import _region_weight_map

    h, w = 240, 320
    weights = _region_weight_map((h, w))
    assert weights.shape == (h, w)
    top_quarter_sum = float(weights[: h // 4, :].sum())
    bottom_quarter_sum = float(weights[3 * h // 4 :, :].sum())
    assert top_quarter_sum == 0.0
    assert bottom_quarter_sum > 100 * top_quarter_sum + 1.0
    assert weights[h - 1, w // 2] == pytest.approx(1.0, abs=1e-3)
    assert weights[h - 1, 0] == pytest.approx(0.5, abs=1e-3)


def test_care_weighted_path_does_not_trigger_for_periphery_only():
    """When weighted count is below threshold, hazard must be False."""
    settings = _silent_settings(hazard_obstacle_ratio=0.05)
    care = CareNavigator(settings)
    seg = SegmentationResult(
        obstacle_pixels=5000,
        obstacle_pixels_weighted=10.0,  # well below 240*320*0.05*0.5 = 1920
        walkable_ratio=0.4,
        metadata={"shape": [240, 320], "semantic": True},
    )
    depth = DepthResult(obstacle_depth_m=5.0)
    result = care.predict(np.zeros((240, 320, 3), dtype=np.uint8), seg, depth)
    assert result.hazard_detected is False
    assert result.safety_score >= 0.5


def test_care_weighted_path_triggers_for_in_path_obstacle():
    """When weighted count clears the threshold, hazard must be True."""
    settings = _silent_settings(hazard_obstacle_ratio=0.05)
    care = CareNavigator(settings)
    seg = SegmentationResult(
        obstacle_pixels=5000,
        obstacle_pixels_weighted=4000.0,
        walkable_ratio=0.2,
        metadata={"shape": [240, 320], "semantic": True},
    )
    depth = DepthResult(obstacle_depth_m=5.0)
    result = care.predict(np.zeros((240, 320, 3), dtype=np.uint8), seg, depth)
    assert result.hazard_detected is True


def test_obstacle_stop_no_longer_uses_brightness_depth():
    """The depth-based STOP shortcut was removed (it fired off mock brightness)."""
    settings = _silent_settings()
    interpreter = NavigationInterpreter(settings)
    bundle = PerceptionBundle(
        frame_id=0,
        segmentation=SegmentationResult(
            obstacle_pixels=0,
            obstacle_pixels_weighted=0.0,
            walkable_ratio=0.5,
            metadata={"shape": [240, 320]},
        ),
        depth=DepthResult(obstacle_depth_m=0.5),
        care=CareResult(hazard_detected=False, safety_score=0.9),
    )
    decision = interpreter.interpret(bundle)
    assert decision.command != NavigationCommand.STOP


# ---------------------------------------------------------------------------
# Live position plumbing
# ---------------------------------------------------------------------------


def test_position_model_has_coords():
    from navigation.models import Position

    assert Position(lat=10.0, lon=20.0).has_coords is True
    assert Position().has_coords is False
    assert Position(lat=10.0).has_coords is False


def test_interpreter_uses_live_position_for_map_guidance(monkeypatch):
    """Live GPS sample triggers route fetch and the resulting decision uses it."""
    from navigation.maps.router import RoutePlan
    from navigation.models import Position
    import navigation.maps.router as router_mod

    settings = _silent_settings(
        use_map_guidance=True,
        dest_lat=33.4146,
        dest_lon=-111.9400,
    )
    interp = NavigationInterpreter(settings)
    assert interp._map_guidance is None

    fake_route = RoutePlan(
        waypoints=[(33.4215, -111.9342), (33.4180, -111.9370), (33.4146, -111.9400)],
        distance_m=900.0,
        duration_s=720.0,
    )
    monkeypatch.setattr(router_mod, "fetch_route", lambda *a, **k: fake_route)
    monkeypatch.setattr(router_mod, "save_route_debug", lambda *a, **k: None)

    bundle = PerceptionBundle(
        frame_id=0,
        segmentation=SegmentationResult(walkable_ratio=0.5),
        depth=DepthResult(),
        care=CareResult(safety_score=0.9),
    )
    decision = interp.interpret(
        bundle,
        position=Position(lat=33.4215, lon=-111.9342, heading_deg=270.0),
    )

    import time
    for _ in range(100):
        if interp._map_route_attempted:
            break
        time.sleep(0.02)

    assert interp._map_guidance is not None
    assert decision.command in {
        NavigationCommand.GO_FORWARD,
        NavigationCommand.MOVE_LEFT,
        NavigationCommand.MOVE_RIGHT,
        NavigationCommand.STOP,
    }


# ---------------------------------------------------------------------------
# Segmentation-derived depth proxy
# ---------------------------------------------------------------------------


def test_depth_proxy_far_when_path_is_walkable():
    """Bottom-center walkable → far (~3 m)."""
    settings = _silent_settings()
    est = DepthEstimator(settings)
    h, w = 60, 80
    class_map = np.zeros((h, w), dtype=np.int32)
    seg = SegmentationResult(
        class_map=class_map,
        metadata={"id_to_name": {0: "road"}, "shape": [h, w]},
    )
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    depth = est.predict(frame, segmentation=seg)
    assert depth.center_depth_m == pytest.approx(3.0, abs=0.01)
    assert depth.metadata["source"] == "segmentation_proxy"


def test_depth_proxy_near_when_obstacle_blocks_path():
    """Bottom-center obstacle → near (≤ ~1.5 m)."""
    settings = _silent_settings()
    est = DepthEstimator(settings)
    h, w = 60, 80
    class_map = np.zeros((h, w), dtype=np.int32)
    class_map[int(h * 0.6) :, int(w * 0.3) : int(w * 0.7)] = 11
    seg = SegmentationResult(
        class_map=class_map,
        metadata={"id_to_name": {0: "road", 11: "person"}, "shape": [h, w]},
    )
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    depth = est.predict(frame, segmentation=seg)
    assert depth.center_depth_m is not None
    assert depth.center_depth_m <= 1.5
    assert depth.metadata["source"] == "vertical_position"


def test_depth_raises_without_segmentation_or_client_depth():
    """Brightness mock removed: depth needs a class map or client depth_m."""
    settings = _silent_settings()
    est = DepthEstimator(settings)
    frame = np.full((64, 64, 3), 0, dtype=np.uint8)
    with pytest.raises(ValueError):
        est.predict(frame)


def test_depth_uses_client_external_value():
    """A client-provided external_depth_m wins and needs no segmentation."""
    settings = _silent_settings()
    est = DepthEstimator(settings)
    frame = np.full((64, 64, 3), 0, dtype=np.uint8)
    depth = est.predict(frame, external_depth_m=2.0)
    assert depth.obstacle_depth_m == pytest.approx(2.0)
    assert depth.metadata["source"] == "client_depth_anything"


def test_llm_short_circuits_on_hazard_without_calling_chain():
    """When CARE flags hazard, the LLM path must not invoke the chain."""
    settings = _silent_settings(use_llm=True)
    interp = NavigationInterpreter(settings)
    called = {"n": 0}

    class FakeChain:
        def invoke(self, _):
            called["n"] += 1
            raise AssertionError("should not be called when hazard is true")

    interp._chain = FakeChain()
    bundle = PerceptionBundle(
        frame_id=0,
        segmentation=SegmentationResult(),
        depth=DepthResult(),
        care=CareResult(hazard_detected=True, safety_score=0.2),
    )
    decision = interp._llm(bundle)
    assert decision.command == NavigationCommand.STOP
    assert called["n"] == 0


# ---------------------------------------------------------------------------
# Spatial reasoner integration through process_frame
# ---------------------------------------------------------------------------


def _spatial_components(settings):
    from navigation.reasoning.composer import PhraseComposer
    from navigation.reasoning.spatial_reasoner import SpatialReasoner
    from navigation.reasoning.trend import TrendTracker
    from navigation.perception.stairs import StairsDetector
    from navigation.output.voice_queue import VoiceQueue

    return dict(
        spatial_reasoner=SpatialReasoner(settings),
        composer=PhraseComposer(settings),
        voice_queue=VoiceQueue(settings),
        trend_tracker=TrendTracker(settings),
        stairs_detector=StairsDetector(settings),
    )


def test_benchmark_mode_emits_per_stage_timing():
    """When Settings.benchmark_mode=True, the record carries per-stage ms."""
    settings = _silent_settings(benchmark_mode=True)
    segmenter = _walkable_with_center_obstacle()
    record = process_frame(
        np.full((240, 320, 3), 128, dtype=np.uint8),
        frame_id=0,
        settings=settings,
        segmenter=segmenter,
        depth_est=DepthEstimator(settings),
        care=CareNavigator(settings),
        interpreter=NavigationInterpreter(settings),
        validator=CommandValidator(settings),
        tts=SpeechEngine(settings),
        **_spatial_components(settings),
    )
    assert "timings_ms" in record
    keys = set(record["timings_ms"].keys())
    assert {"seg", "depth", "care", "reasoner", "composer"}.issubset(keys)


def test_pipeline_smoke_through_composer_and_queue():
    """Spatial path produces a record with `facts` and a phrase."""
    settings = _silent_settings()
    segmenter = _walkable_with_center_obstacle()
    record = process_frame(
        np.full((240, 320, 3), 128, dtype=np.uint8),
        frame_id=0,
        settings=settings,
        segmenter=segmenter,
        depth_est=DepthEstimator(settings),
        care=CareNavigator(settings),
        interpreter=NavigationInterpreter(settings),
        validator=CommandValidator(settings),
        tts=SpeechEngine(settings),
        **_spatial_components(settings),
    )
    assert "facts" in record
    assert record["phrase"]
    assert record["command"] in {c.value for c in NavigationCommand}


def test_legacy_reasoner_path_still_works():
    """`use_legacy_reasoner=True` falls back to NavigationInterpreter."""
    settings = _silent_settings()
    segmenter = _walkable_with_center_obstacle()
    record = process_frame(
        np.full((240, 320, 3), 128, dtype=np.uint8),
        frame_id=0,
        settings=settings,
        segmenter=segmenter,
        depth_est=DepthEstimator(settings),
        care=CareNavigator(settings),
        interpreter=NavigationInterpreter(settings),
        validator=CommandValidator(settings),
        tts=SpeechEngine(settings),
        use_legacy_reasoner=True,
    )
    assert "facts" not in record
    assert record["command"] in {c.value for c in NavigationCommand}


# ---------------------------------------------------------------------------
# Vertical-position depth proxy (Layer 1)
# ---------------------------------------------------------------------------


def test_depth_closer_obstacle_returns_smaller_depth():
    """Lower obstacle base → reported closer (monotonic in real distance)."""
    settings = _silent_settings()
    est = DepthEstimator(settings)
    h, w = 240, 320
    meta = {"semantic": True, "id_to_name": {0: "road", 11: "person"}}

    cm_far = np.full((h, w), 0, dtype=np.int32)
    far_row = int(h * 0.55)
    cm_far[far_row : far_row + 4, w // 2 - 6 : w // 2 + 6] = 11

    cm_close = np.full((h, w), 0, dtype=np.int32)
    close_row = int(h * 0.85)
    cm_close[close_row : close_row + 4, w // 2 - 6 : w // 2 + 6] = 11

    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    depth_far = est.predict(
        frame, segmentation=SegmentationResult(class_map=cm_far, metadata=meta)
    ).obstacle_depth_m
    depth_close = est.predict(
        frame, segmentation=SegmentationResult(class_map=cm_close, metadata=meta)
    ).obstacle_depth_m
    assert depth_close < depth_far, (depth_close, depth_far)


def test_depth_falls_back_to_ratio_when_no_obstacles():
    """No obstacles → ratio fallback (mostly walkable → far)."""
    settings = _silent_settings()
    est = DepthEstimator(settings)
    h, w = 240, 320
    cm = np.full((h, w), 0, dtype=np.int32)
    meta = {"semantic": True, "id_to_name": {0: "road"}}
    seg = SegmentationResult(class_map=cm, metadata=meta)
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    result = est.predict(frame, segmentation=seg)
    assert result.metadata["source"] == "segmentation_proxy"
    assert result.obstacle_depth_m == pytest.approx(3.0, abs=0.01)


# ---------------------------------------------------------------------------
# AsyncSegmenter (background-thread inference for smooth preview)
# ---------------------------------------------------------------------------


class _SlowSegmenter:
    """A segmenter that blocks for ``delay`` seconds per ``predict`` call.

    Lets the async wrapper's non-blocking behavior be observed deterministically
    without a real model.
    """

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.calls = 0
        self.is_semantic = True
        self.last_results = None
        self.last_segmentation = None

    def predict(self, frame: np.ndarray) -> SegmentationResult:
        import time

        time.sleep(self.delay)
        self.calls += 1
        self.last_segmentation = SegmentationResult(
            walkable_ratio=0.5, metadata={"call": self.calls}
        )
        return self.last_segmentation


def test_async_segmenter_predict_caches_synchronously():
    from navigation.pipeline.runner import AsyncSegmenter

    inner = _SlowSegmenter(delay=0.0)
    seg = AsyncSegmenter(inner)
    assert seg.is_semantic is True
    assert seg.last_results is None
    assert seg.last_segmentation is None

    result = seg.predict(np.zeros((8, 8, 3), dtype=np.uint8))
    assert result is not None
    assert seg.last_segmentation is result


def test_async_segmenter_predict_async_is_non_blocking():
    """predict_async returns immediately and fills the cache in the background."""
    import time

    from navigation.pipeline.runner import AsyncSegmenter

    inner = _SlowSegmenter(delay=0.1)
    seg = AsyncSegmenter(inner)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    t0 = time.perf_counter()
    first = seg.predict_async(frame)   # kicks off bg work, returns last (None)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.05               # did not block on the 0.1s predict
    assert first is None                # nothing cached yet

    # A second call while inference is in flight must also not block and must
    # not start a second inference.
    seg.predict_async(frame)

    # Wait for the background inference to land.
    deadline = time.perf_counter() + 2.0
    while seg.last_segmentation is None and time.perf_counter() < deadline:
        time.sleep(0.01)

    assert seg.last_segmentation is not None
    assert inner.calls == 1             # only one inference ran, not two
