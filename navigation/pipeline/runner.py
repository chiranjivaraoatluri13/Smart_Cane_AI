"""Main perception → reasoning → output loop."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from navigation.capture.camera import CameraStream, load_image
from navigation.config import Settings, load_settings
from navigation.models import PerceptionBundle, Position
from navigation.output.hud import draw_navigation_hud, format_command_banner
from navigation.output.tts import SpeechEngine
from navigation.output.validator import CommandValidator
from navigation.output.voice_queue import VoiceQueue
from navigation.perception.depth import UniDepthEstimator
from navigation.perception.segmentation import YoloSegmenter
from navigation.perception.stairs import StairsDetector
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
from navigation.reasoning.spatial_reasoner import SpatialReasoner, _next_route_cue
from navigation.reasoning.trend import TrendTracker

logger = logging.getLogger(__name__)


def _should_run_inference(frame_id: int, settings: Settings) -> bool:
    n = max(1, settings.process_every_n_frames)
    return frame_id % n == 0


def _warmup_segmenter(segmenter: YoloSegmenter, settings: Settings) -> None:
    """Run one dummy inference so the first real frame isn't a stutter."""
    h = settings.inference_height if settings.inference_height > 0 else 192
    w = settings.inference_width if settings.inference_width > 0 else 256
    dummy = np.zeros((max(64, h), max(64, w), 3), dtype=np.uint8)
    try:
        segmenter.predict(dummy, dry_run=False)
        logger.info("Segmenter warmup complete (%dx%d).", w, h)
    except Exception as e:
        # Warmup is best-effort: if YOLO weights are missing or vision extras
        # aren't installed, the pipeline still runs in dry-run mode.
        logger.info("Segmenter warmup skipped (%s).", e)


def _print_command(record: dict) -> None:
    print(format_command_banner(record), flush=True)


def _maybe_show_segmentation(
    frame: np.ndarray,
    *,
    segmenter: YoloSegmenter,
    dry_run: bool,
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
    overlay = render_overlay(frame, segmenter=segmenter, dry_run=dry_run)
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
    dry_run: bool,
    segmenter: YoloSegmenter,
    depth_est: UniDepthEstimator,
    care: CareNavigator,
    interpreter: NavigationInterpreter,
    validator: CommandValidator,
    tts: SpeechEngine,
    alert_tracker: AlertTracker | None = None,
    spatial_reasoner: SpatialReasoner | None = None,
    composer: PhraseComposer | None = None,
    voice_queue: VoiceQueue | None = None,
    trend_tracker: TrendTracker | None = None,
    stairs_detector: StairsDetector | None = None,
    use_legacy_reasoner: bool = False,
    position: Position | None = None,
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
    seg = segmenter.predict(frame, dry_run=dry_run)
    if bench:
        timings["seg"] = (_t() - t0) * 1000

    t0 = _t() if bench else 0.0
    depth = depth_est.predict(frame, dry_run=dry_run, segmentation=seg)
    if bench:
        timings["depth"] = (_t() - t0) * 1000

    t0 = _t() if bench else 0.0
    care_out = care.predict(frame, seg, depth, dry_run=dry_run)
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
        decision = interpreter.interpret(bundle, dry_run=dry_run, position=position)
        decision = validator.approve(decision)
        phrase = tts.speak(decision) if decision.speak else None
        spoken = phrase
    else:
        # ----- Spatial path (new).
        t0 = _t() if bench else 0.0
        stairs = (
            stairs_detector.detect(frame, seg)
            if stairs_detector is not None
            else _no_stairs()
        )
        if bench:
            timings["stairs"] = (_t() - t0) * 1000

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
        dry_run=dry_run,
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
    # Lazily fetch the route on the first frame that carries GPS. In the
    # spatial path interpreter.interpret() is never called, so we must
    # trigger the route fetch here — otherwise _map_guidance stays None and
    # the user never gets turn-by-turn directions.
    if (
        getattr(interpreter, "_map_guidance", None) is None
        and settings.use_map_guidance
        and hasattr(interpreter, "ensure_map_guidance")
    ):
        try:
            interpreter.ensure_map_guidance(pos)
        except Exception as e:  # pragma: no cover - network failure path
            logger.warning("Route fetch failed (%s); vision-only guidance.", e)

    map_guidance = getattr(interpreter, "_map_guidance", None)
    if map_guidance is None:
        return None
    heading = (
        pos.heading_deg
        if pos.heading_deg is not None
        else settings.current_heading_deg
    )
    return _next_route_cue(
        map_guidance,
        settings,
        current_lat=pos.lat,
        current_lon=pos.lon,
        heading_deg=heading,
    )


def _build_pipeline_components(
    settings: Settings,
) -> tuple[
    YoloSegmenter,
    UniDepthEstimator,
    CareNavigator,
    NavigationInterpreter,
    CommandValidator,
    SpeechEngine,
    AlertTracker,
    SpatialReasoner,
    PhraseComposer,
    VoiceQueue,
    TrendTracker,
    StairsDetector,
]:
    """Construct one of every component the pipeline needs.

    Pulled into a helper so `run_live`, `run_image`, and `phone_server.py` all
    instantiate the same set in the same order.
    """
    segmenter = YoloSegmenter(settings)
    depth_est = UniDepthEstimator(settings)
    care = CareNavigator(settings)
    interpreter = NavigationInterpreter(settings)
    validator = CommandValidator(settings)
    tts = SpeechEngine(settings)
    alert_tracker = AlertTracker()
    spatial = SpatialReasoner(settings)
    composer = PhraseComposer(settings)
    voice_queue = VoiceQueue(settings)
    trend = TrendTracker(settings)
    stairs = StairsDetector(settings)
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
        stairs,
    )


def run_live(
    settings: Settings,
    *,
    dry_run: bool = False,
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
        stairs,
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
                            dry_run=dry_run,
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
                            stairs_detector=stairs,
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
                            dry_run=dry_run,
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
    dry_run: bool = False,
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
        stairs,
    ) = _build_pipeline_components(settings)
    tts.warmup()
    _warmup_segmenter(segmenter, settings)

    record: dict = {}
    try:
        record = process_frame(
            frame,
            0,
            settings,
            dry_run=dry_run,
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
            stairs_detector=stairs,
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
    dry_run: bool = False,
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

    segmenter = YoloSegmenter(settings)
    frame_id = 0

    def _show(frame: np.ndarray, *, infer: bool) -> bool:
        nonlocal frame_id
        if infer:
            segmenter.predict(frame, dry_run=dry_run)
        overlay = render_overlay(frame, segmenter=segmenter, dry_run=dry_run)
        if seg_save_dir is not None:
            save_overlay(overlay, seg_save_dir / f"seg_{frame_id:06d}.jpg")
        frame_id += 1
        key = show_frame(overlay, wait_ms=0 if image_path else 1)
        return key != ord("q")

    try:
        if image_path is not None:
            frame = load_image(str(image_path))
            _show(frame, infer=True)
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
    dry_run: bool = False,
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
            dry_run=dry_run,
            show_seg=show_seg,
            seg_save_dir=seg_save_dir,
            use_legacy_reasoner=use_legacy_reasoner,
        )
    return run_live(
        settings,
        dry_run=dry_run,
        camera_index=camera_index,
        max_frames=max_frames,
        show_seg=show_seg,
        seg_save_dir=seg_save_dir,
        use_legacy_reasoner=use_legacy_reasoner,
    )
