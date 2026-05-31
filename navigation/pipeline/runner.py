"""Main perception → reasoning → output loop."""

from __future__ import annotations

import json
import logging
import time
import threading
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from navigation.capture.camera import CameraStream, load_image
from navigation.config import Settings, load_settings
from navigation.models import PerceptionBundle, Position, SegmentationResult
from navigation.output.hud import draw_navigation_hud, format_command_banner
from navigation.output.tts import SpeechEngine
from navigation.output.validator import CommandValidator
from navigation.output.voice_queue import VoiceQueue
from navigation.perception.depth import DepthEstimator
from navigation.perception.segmentation_base import build_segmenter
from navigation.perception.visualize import (
    close_windows,
    render_overlay,
    save_overlay,
    show_frame,
)
from navigation.reasoning.alerts import AlertTracker, ProximityAlert
from navigation.reasoning.care import CareNavigator
from navigation.reasoning.composer import PhraseComposer
from navigation.reasoning.facts import GuidanceFacts, RouteCue
from navigation.reasoning.llm import NavigationInterpreter
from navigation.maps.router import cross_track_distance_m
from navigation.reasoning.spatial_reasoner import SpatialReasoner, _next_route_cue
from navigation.reasoning.trend import TrendTracker

logger = logging.getLogger(__name__)


@runtime_checkable
class Segmenter(Protocol):
    """Structural type for any segmentation backend (ONNX or PyTorch)."""

    @property
    def last_segmentation(self) -> SegmentationResult | None: ...
    @property
    def last_results(self) -> None: ...
    @property
    def is_semantic(self) -> bool: ...
    def predict(self, frame: np.ndarray) -> SegmentationResult: ...


class AsyncSegmenter:
    """Wraps any Segmenter to run inference in a background thread.

    The camera loop runs at full display FPS. Inference runs as fast as the
    model allows. The most recent result is reused for frames that arrive
    while inference is in progress — so the display stays smooth even when
    the model is slow.

    This is the difference between:
      - Blocking: display freezes for 200ms every inference frame
      - Async: display updates every frame; decisions update at model speed
    """

    def __init__(self, segmenter: Segmenter):
        self._seg = segmenter
        self._last: SegmentationResult | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def last_segmentation(self) -> SegmentationResult | None:
        with self._lock:
            return self._last

    @property
    def last_results(self) -> None:
        return None

    @property
    def is_semantic(self) -> bool:
        return self._seg.is_semantic

    def predict(self, frame: np.ndarray) -> SegmentationResult:
        """Run inference synchronously and cache the result."""
        result = self._seg.predict(frame)
        with self._lock:
            self._last = result
        return result

    def predict_async(self, frame: np.ndarray) -> SegmentationResult | None:
        """Start inference in the background; return the last cached result.

        Call this every frame. When inference is already running, the cached
        result from the previous frame is returned immediately (no blocking).
        When inference finishes, the cache is updated and the next call
        returns the fresh result.
        """
        if self._running:
            with self._lock:
                return self._last

        frame_copy = frame.copy()  # snapshot so the camera buffer can advance

        def _run() -> None:
            try:
                result = self._seg.predict(frame_copy)
                with self._lock:
                    self._last = result
            finally:
                self._running = False

        self._running = True
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        with self._lock:
            return self._last


def _should_run_inference(frame_id: int, settings: Settings) -> bool:
    n = max(1, settings.process_every_n_frames)
    return frame_id % n == 0


def _warmup_segmenter(segmenter: Segmenter, settings: Settings) -> None:
    """Run one dummy inference so the first real frame isn't a stutter."""
    h = settings.inference_height if settings.inference_height > 0 else 192
    w = settings.inference_width if settings.inference_width > 0 else 256
    dummy = np.zeros((max(64, h), max(64, w), 3), dtype=np.uint8)
    try:
        segmenter.predict(dummy)
        logger.info("Segmenter warmup complete (%dx%d).", w, h)
    except Exception as e:
        # Warmup is best-effort: if the model can't load yet we log and let
        # the first real frame surface the error.
        logger.info("Segmenter warmup skipped (%s).", e)


def _print_command(record: dict) -> None:
    print(format_command_banner(record), flush=True)


def _maybe_show_segmentation(
    frame: np.ndarray,
    *,
    segmenter: Segmenter,
    show_seg: bool,
    seg_save_dir: Path | None,
    frame_id: int,
    block: bool = False,
    hud: dict[str, Any] | None = None,
    hud_stale: bool = False,
) -> bool:
    """Return False if the user pressed ``q`` to quit."""
    if not show_seg and seg_save_dir is None:
        return True
    overlay = render_overlay(frame, segmenter=segmenter)
    if hud is not None:
        overlay = draw_navigation_hud(
            overlay,
            phrase=str(hud.get("phrase") or ""),
            command=str(hud.get("command", "")),
            speak=bool(hud.get("speak", True)),
            confidence=float(hud.get("confidence", 0)),
            rationale=str(hud.get("rationale") or ""),
            stale=hud_stale,
        )
    if seg_save_dir is not None:
        save_overlay(overlay, seg_save_dir / f"seg_{frame_id:06d}.jpg")
    if show_seg:
        wait_ms = 0 if block else 1
        key = show_frame(overlay, wait_ms=wait_ms)
        if key == ord("q"):
            return False
    return True


def process_frame(
    frame: np.ndarray,
    frame_id: int,
    settings: Settings,
    *,
    segmenter: Segmenter,
    depth_est: DepthEstimator,
    care: CareNavigator,
    interpreter: NavigationInterpreter,
    validator: CommandValidator,
    tts: SpeechEngine,
    alert_tracker: AlertTracker | None = None,
    spatial_reasoner: SpatialReasoner | None = None,
    composer: PhraseComposer | None = None,
    voice_queue: VoiceQueue | None = None,
    trend_tracker: TrendTracker | None = None,
    use_legacy_reasoner: bool = False,
    position: Position | None = None,
    client_depth_m: float | None = None,
    show_seg: bool = False,
    seg_save_dir: Path | None = None,
    seg_block: bool = False,
    hud: dict[str, Any] | None = None,
) -> dict:
    timings: dict[str, float] = {}
    bench = bool(settings.benchmark_mode)

    def _t() -> float:
        return time.perf_counter()

    t0 = _t() if bench else 0.0
    seg = segmenter.predict(frame)
    if bench:
        timings["seg"] = (_t() - t0) * 1000

    t0 = _t() if bench else 0.0
    if depth_est is not None:
        depth = depth_est.predict(
            frame,
            segmentation=seg,
            external_depth_m=client_depth_m,
        )
    else:
        # Depth skipped on cloud — use a safe default (mid-range)
        from navigation.models import DepthResult
        depth = DepthResult(center_depth_m=2.0, obstacle_depth_m=2.0, min_depth_m=1.5)
    if bench:
        timings["depth"] = (_t() - t0) * 1000

    t0 = _t() if bench else 0.0
    care_out = care.predict(frame, seg, depth)
    if bench:
        timings["care"] = (_t() - t0) * 1000

    bundle = PerceptionBundle(
        frame_id=frame_id,
        segmentation=seg,
        depth=depth,
        care=care_out,
    )

    facts: GuidanceFacts | None = None

    if use_legacy_reasoner or spatial_reasoner is None:
        # ----- Legacy path (unchanged behavior, preserves existing tests).
        decision = interpreter.interpret(bundle, position=position)
        decision = validator.approve(decision)
        phrase = tts.speak(decision) if decision.speak else None
        spoken = phrase
    else:
        # ----- Spatial path (new).
        # Stairs detection — SegFormer detects stairs/step as hazard classes
        stairs = _no_stairs()

        t0 = _t() if bench else 0.0
        if trend_tracker is not None:
            trend_tracker.update(seg.per_side_class_pixels)
            approach_by_category = trend_tracker.classify_all()
        else:
            approach_by_category = {}
        if bench:
            timings["trend"] = (_t() - t0) * 1000

        # Pull route cue from the legacy interpreter's MapGuidance state.
        route_cue = _resolve_route_cue(interpreter, settings, position)

        t0 = _t() if bench else 0.0
        decision, facts = spatial_reasoner.decide(
            seg, depth, care_out, route_cue,
            stairs=stairs,
            approach_by_category=approach_by_category,
        )
        if bench:
            timings["reasoner"] = (_t() - t0) * 1000

        decision = validator.approve(decision)

        t0 = _t() if bench else 0.0
        phrase = composer.compose(facts) if composer is not None else None
        if bench:
            timings["composer"] = (_t() - t0) * 1000

        # Single speak authority: the CommandValidator. The voice queue used
        # to re-decide here with its own per-tier cooldowns (vision_stop=0),
        # which silently overrode the validator and caused STOP to repeat
        # every frame. Now the validator's verdict (`decision.speak`) is the
        # only gate. The voice queue is kept only as an optional ordering
        # buffer for the laptop CLI; on the phone path we speak directly.
        spoken = phrase if decision.speak else None
        if spoken is not None:
            # The phone path returns the phrase to the client which speaks
            # it via Web Speech API. The laptop path uses pyttsx3 directly.
            tts._say(spoken)

    # Per-class proximity alerts (kept layered on top of either path).
    # The CommandValidator gates them via approve_alert(); approved alerts
    # are both spoken locally (laptop CLI) and included in the JSON record
    # so the phone client speaks them via Web Speech API.
    spoken_alerts: list[ProximityAlert] = []
    if alert_tracker is not None:
        for alert in alert_tracker.update(seg):
            if validator.approve_alert(alert):
                tts.speak_alert(alert)
                spoken_alerts.append(alert)

    record: dict[str, Any] = {
        "frame_id": frame_id,
        "command": decision.command.value,
        "confidence": decision.confidence,
        "rationale": decision.rationale,
        # ``speak`` reflects whether the validator actually approved a
        # phrase this frame (not just that the composer rendered one for
        # the HUD). The phone client uses this flag to decide whether to
        # call the Web Speech API.
        "speak": bool(spoken)
        if (spatial_reasoner is not None and not use_legacy_reasoner)
        else decision.speak,
        "phrase": spoken or phrase or decision.command.value.replace("_", " "),
        "alerts": [
            {
                "category": a.category,
                "phrase": a.phrase,
                "confidence": a.confidence,
            }
            for a in spoken_alerts
        ],
    }
    if facts is not None:
        record["facts"] = facts.summary_dict()
    if bench and timings:
        record["timings_ms"] = timings

    display_hud = hud if hud is not None else record

    if not _maybe_show_segmentation(
        frame,
        segmenter=segmenter,
        show_seg=show_seg,
        seg_save_dir=seg_save_dir,
        frame_id=frame_id,
        block=seg_block,
        hud=display_hud if show_seg or seg_save_dir else None,
    ):
        raise KeyboardInterrupt

    return record


def _no_stairs():
    from navigation.reasoning.facts import StairsResult
    return StairsResult(False, 0.0, "")


def _resolve_route_cue(
    interpreter: NavigationInterpreter,
    settings: Settings,
    position: Position | None,
) -> Optional[RouteCue]:
    pos = (
        interpreter._resolve_position(position)
        if hasattr(interpreter, "_resolve_position")
        else position
    )
    if pos is None or pos.lat is None or pos.lon is None:
        return None
    # Destination/map flags live on interpreter.settings (updated by
    # /set_destination). The module-level Settings passed to process_frame
    # must match, but read the interpreter copy so route fetch cannot be
    # skipped when those objects diverge.
    map_settings = getattr(interpreter, "settings", settings)
    # Lazily fetch the route on the first frame that carries GPS. In the
    # spatial path interpreter.interpret() is never called, so we must
    # trigger the route fetch here — otherwise _map_guidance stays None and
    # the user never gets turn-by-turn directions.
    if map_settings.use_map_guidance and hasattr(interpreter, "ensure_map_guidance"):
        try:
            if getattr(interpreter, "is_route_loading", lambda: False)():
                return RouteCue(
                    turn="loading",
                    meters_to_turn=0.0,
                    target_bearing_deg=0.0,
                    rationale="fetching route",
                )
            if getattr(interpreter, "_map_guidance", None) is None:
                interpreter.ensure_map_guidance(pos)
        except Exception as e:  # pragma: no cover - network failure path
            logger.warning("Route fetch failed (%s); vision-only guidance.", e)

    map_guidance = getattr(interpreter, "_map_guidance", None)
    if map_guidance is None:
        if getattr(interpreter, "is_route_loading", lambda: False)():
            return RouteCue(
                turn="loading",
                meters_to_turn=0.0,
                target_bearing_deg=0.0,
                rationale="fetching route",
            )
        if (
            map_settings.use_map_guidance
            and map_settings.map_destination_set
            and getattr(interpreter, "_route_permanent_failure", False)
        ):
            return RouteCue(
                turn="failed",
                meters_to_turn=0.0,
                target_bearing_deg=0.0,
                rationale="route_failed",
            )
        return None

    if hasattr(interpreter, "maybe_refetch_route"):
        cross = cross_track_distance_m(pos.lat, pos.lon, map_guidance.route)
        interpreter.maybe_refetch_route(pos, cross)

    # Pass heading only when known — do not substitute 0° (north).
    heading = pos.heading_deg
    return _next_route_cue(
        map_guidance,
        map_settings,
        current_lat=pos.lat,
        current_lon=pos.lon,
        heading_deg=heading,
    )


def _build_pipeline_components(
    settings: Settings,
) -> tuple[
    Segmenter,
    DepthEstimator,
    CareNavigator,
    NavigationInterpreter,
    CommandValidator,
    SpeechEngine,
    AlertTracker,
    SpatialReasoner,
    PhraseComposer,
    VoiceQueue,
    TrendTracker,
]:
    """Construct one of every component the pipeline needs.

    Pulled into a helper so `run_live`, `run_image`, and `phone_server.py` all
    instantiate the same set in the same order.
    """
    segmenter = build_segmenter(settings)
    depth_est = DepthEstimator(settings)
    care = CareNavigator(settings)
    interpreter = NavigationInterpreter(settings)
    validator = CommandValidator(settings)
    tts = SpeechEngine(settings)
    alert_tracker = AlertTracker()
    spatial = SpatialReasoner(settings)
    composer = PhraseComposer(settings)
    voice_queue = VoiceQueue(settings)
    trend = TrendTracker(settings)
    alert_tracker = AlertTracker.from_settings(settings)
    return (
        segmenter,
        depth_est,
        care,
        interpreter,
        validator,
        tts,
        alert_tracker,
        spatial,
        composer,
        voice_queue,
        trend,
    )


def run_live(
    settings: Settings,
    *,
    camera_index: int | None = None,
    max_frames: int | None = None,
    show_seg: bool = False,
    seg_save_dir: Path | None = None,
    use_legacy_reasoner: bool = False,
) -> int:
    if camera_index is not None:
        settings = settings.model_copy(update={"camera_index": camera_index})

    (
        segmenter,
        depth_est,
        care,
        interpreter,
        validator,
        tts,
        alert_tracker,
        spatial,
        composer,
        voice_queue,
        trend,
    ) = _build_pipeline_components(settings)
    tts.warmup()
    _warmup_segmenter(segmenter, settings)

    frame_id = 0
    last_hud: dict[str, Any] | None = None
    try:
        with CameraStream(settings) as cam:
            for frame in cam.frames():
                try:
                    if _should_run_inference(frame_id, settings):
                        record = process_frame(
                            frame,
                            frame_id,
                            settings,
                            segmenter=segmenter,
                            depth_est=depth_est,
                            care=care,
                            interpreter=interpreter,
                            validator=validator,
                            tts=tts,
                            alert_tracker=alert_tracker,
                            spatial_reasoner=spatial,
                            composer=composer,
                            voice_queue=voice_queue,
                            trend_tracker=trend,
                            use_legacy_reasoner=use_legacy_reasoner,
                            show_seg=show_seg,
                            seg_save_dir=seg_save_dir,
                        )
                        last_hud = record
                        _print_command(record)
                        print(json.dumps(record), flush=True)
                    elif show_seg or seg_save_dir is not None:
                        if not _maybe_show_segmentation(
                            frame,
                            segmenter=segmenter,
                            show_seg=show_seg,
                            seg_save_dir=seg_save_dir,
                            frame_id=frame_id,
                            hud=last_hud,
                            hud_stale=True,
                        ):
                            break
                except KeyboardInterrupt:
                    break
                frame_id += 1
                if max_frames is not None and frame_id >= max_frames:
                    break
    finally:
        if show_seg or seg_save_dir is not None:
            close_windows()
    return 0


def run_image(
    image_path: Path,
    settings: Settings,
    *,
    show_seg: bool = False,
    seg_save_dir: Path | None = None,
    use_legacy_reasoner: bool = False,
) -> int:
    frame = load_image(str(image_path))
    (
        segmenter,
        depth_est,
        care,
        interpreter,
        validator,
        tts,
        alert_tracker,
        spatial,
        composer,
        voice_queue,
        trend,
    ) = _build_pipeline_components(settings)
    tts.warmup()
    _warmup_segmenter(segmenter, settings)

    record: dict = {}
    try:
        record = process_frame(
            frame,
            0,
            settings,
            segmenter=segmenter,
            depth_est=depth_est,
            care=care,
            interpreter=interpreter,
            validator=validator,
            tts=tts,
            alert_tracker=alert_tracker,
            spatial_reasoner=spatial,
            composer=composer,
            voice_queue=voice_queue,
            trend_tracker=trend,
            use_legacy_reasoner=use_legacy_reasoner,
            show_seg=show_seg,
            seg_save_dir=seg_save_dir,
            seg_block=show_seg,
        )
    except KeyboardInterrupt:
        return 0
    finally:
        if show_seg or seg_save_dir is not None:
            close_windows()
    if record:
        _print_command(record)
        print(json.dumps(record, indent=2))
    return 0


def run_preview(
    *,
    camera_index: int | None = None,
    image_path: Path | None = None,
    max_frames: int | None = None,
    seg_save_dir: Path | None = None,
    settings: Settings | None = None,
) -> int:
    """Segmentation-only loop for laptop overlay demo."""
    settings = settings or load_settings()
    if camera_index is not None:
        settings = settings.model_copy(update={"camera_index": camera_index})

    # Wrap the segmenter so live preview never blocks the display: inference
    # runs in a background thread and the most recent class map is reused for
    # the overlay until a fresh one arrives. The camera loop stays at full FPS.
    segmenter = AsyncSegmenter(build_segmenter(settings))
    frame_id = 0

    def _show(frame: np.ndarray, *, infer: bool) -> bool:
        nonlocal frame_id
        if infer:
            segmenter.predict_async(frame)
        overlay = render_overlay(frame, segmenter=segmenter)
        if seg_save_dir is not None:
            save_overlay(overlay, seg_save_dir / f"seg_{frame_id:06d}.jpg")
        frame_id += 1
        key = show_frame(overlay, wait_ms=0 if image_path else 1)
        return key != ord("q")

    try:
        if image_path is not None:
            frame = load_image(str(image_path))
            # A single still has no "next frame" to fill — run synchronously
            # so the overlay isn't blank.
            segmenter.predict(frame)
            _show(frame, infer=False)
        else:
            with CameraStream(settings) as cam:
                for frame in cam.frames():
                    infer = _should_run_inference(frame_id, settings)
                    if not _show(frame, infer=infer):
                        break
                    if max_frames is not None and frame_id >= max_frames:
                        break
    finally:
        close_windows()
    return 0


def run_pipeline(
    *,
    camera_index: int | None = None,
    image_path: Path | None = None,
    max_frames: int | None = None,
    show_seg: bool = False,
    seg_save_dir: Path | None = None,
    settings: Settings | None = None,
    use_legacy_reasoner: bool = False,
) -> int:
    settings = settings or load_settings()
    if image_path is not None:
        return run_image(
            image_path,
            settings,
            show_seg=show_seg,
            seg_save_dir=seg_save_dir,
            use_legacy_reasoner=use_legacy_reasoner,
        )
    return run_live(
        settings,
        camera_index=camera_index,
        max_frames=max_frames,
        show_seg=show_seg,
        seg_save_dir=seg_save_dir,
        use_legacy_reasoner=use_legacy_reasoner,
    )
