"""Export separate demo videos for each live-pipeline processing factor.

Mirrors ``phone_server_cloud`` → ``process_frame`` (cloud profile: SegFormer
ONNX, segmentation depth proxy, spatial reasoner). Each visual stage is
written to its own MP4 under an output directory (default: Desktop demos).

Usage (PowerShell):
  .\\.venv\\Scripts\\python.exe scripts\\export_pipeline_videos.py \\
      --video "C:/path/to/input.mp4" \\
      --output-dir "C:/Users/you/Desktop/assistive-nav-demos" \\
      --stride 2

Depth Anything V2 (``02_depth_colormap.mp4``) is attempted when
``--depth-model`` is set (default). If the model cannot load, a segmentation-
proxy colormap is written instead and noted in the console.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from navigation.config import apply_cloud_profile, load_settings
from navigation.models import SIDES
from navigation.output.hud import draw_navigation_hud, draw_sparse_navigation_hud
from navigation.output.tts import SpeechEngine
from navigation.output.validator import CommandValidator
from navigation.perception.depth import DepthEstimator, _row_to_depth_m
from navigation.perception.segmentation_base import build_segmenter
from navigation.perception.visualize import overlay_from_class_map, render_overlay
from navigation.pipeline.runner import _warmup_segmenter, process_frame
from navigation.reasoning.alerts import AlertTracker
from navigation.reasoning.care import CareNavigator
from navigation.reasoning.composer import PhraseComposer
from navigation.reasoning.llm import NavigationInterpreter
from navigation.reasoning.mask_metrics import walkable_by_side
from navigation.reasoning.spatial_reasoner import SpatialReasoner
from navigation.reasoning.trend import TrendTracker


DEFAULT_OUTPUT = Path.home() / "OneDrive" / "Desktop" / "assistive-nav-demos"

FOCUS_PHRASE_KEYWORDS = (
    "person",
    "approaching",
    "car",
    "curb",
    "wall",
    "clear path",
    "clear",
    "destination",
    "heads up",
    "crossing",
    "blocking",
)

SPARSE_HUD_MIN_INTERVAL_SEC = 2.0
TTS_GAP_SEC = 0.3
TTS_MIN_INTERVAL_SEC = 2.5

EXPORTS = (
    ("01_segmentation_overlay.mp4", "seg_overlay"),
    ("02_depth_colormap.mp4", "depth"),
    ("03_commands_overlay.mp4", "commands"),
    ("04_walkable_obstacle_masks.mp4", "masks"),
    ("05_original_seg_sidebyside.mp4", "side_by_side"),
    ("06_per_side_lanes.mp4", "lanes"),
    ("07_care_safety.mp4", "care"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, help="Input video path.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Directory for output MP4s (default: {DEFAULT_OUTPUT}).",
    )
    p.add_argument("--stride", type=int, default=2, help="Process every Nth frame.")
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Cap processed frames (0 = all, respecting stride).",
    )
    p.add_argument(
        "--depth-model",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="HuggingFace depth model; pass empty string to skip DA V2.",
    )
    p.add_argument("--device", default="cpu", help="torch device for depth model.")
    p.add_argument(
        "--skip-depth-anything",
        action="store_true",
        help="Force segmentation-proxy depth for 02 (faster).",
    )
    p.add_argument(
        "--sparse-hud",
        action="store_true",
        help="Live demo: minimal caption on original frames → {stem}_hud.mp4.",
    )
    p.add_argument(
        "--tts",
        action="store_true",
        help="With --sparse-hud: mux pyttsx3 (SAPI) audio → {stem}_hud_audio.mp4.",
    )
    return p.parse_args()


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {path}")
    return writer


def _class_sets(settings) -> tuple[set[str], set[str], dict[int, str]]:
    cfg = settings.seg_class_config()
    walkable = set(cfg.get("walkable_classes", []))
    obstacle = set(cfg.get("obstacle_classes", []))
    return walkable, obstacle, {}


def _id_to_name_from_seg(seg) -> dict[int, str]:
    meta = (seg.metadata or {}).get("id_to_name") or {}
    return {int(k): str(v) for k, v in meta.items()}


def render_walkable_obstacle_masks(
    frame: np.ndarray,
    seg,
    *,
    walkable_set: set[str],
    obstacle_set: set[str],
    id_to_name: dict[int, str],
    alpha: float = 0.55,
) -> np.ndarray:
    h, w = frame.shape[:2]
    cm = np.asarray(seg.class_map)
    if cm.ndim == 3:
        cm = cm[0]
    if cm.shape[:2] != (h, w):
        cm = cv2.resize(cm.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)

    out = (frame.astype(np.float32) * 0.35).astype(np.uint8)
    green = np.array([0, 220, 0], dtype=np.float32)
    red = np.array([0, 0, 220], dtype=np.float32)

    for cls_id in np.unique(cm):
        name = id_to_name.get(int(cls_id), str(int(cls_id)))
        mask = cm == cls_id
        if name in walkable_set:
            out[mask] = (
                out[mask].astype(np.float32) * (1 - alpha) + green * alpha
            ).astype(np.uint8)
        elif name in obstacle_set:
            out[mask] = (
                out[mask].astype(np.float32) * (1 - alpha) + red * alpha
            ).astype(np.uint8)
    return out


def render_per_side_lanes(
    frame: np.ndarray,
    seg,
    *,
    walkable_by_side_ratios: dict[str, float] | None = None,
) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    third = w // 3
    rem = w - third * 3
    x_left = third
    x_right = third + (third + rem)
    cv2.line(out, (x_left, 0), (x_left, h), (255, 255, 0), 2)
    cv2.line(out, (x_right, 0), (x_right, h), (255, 255, 0), 2)

    ratios = walkable_by_side_ratios or {}
    if not ratios and seg.per_side_walkable_ratio:
        ratios = dict(seg.per_side_walkable_ratio)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, side in enumerate(SIDES):
        r = float(ratios.get(side, seg.walkable_ratio if side == "center" else 0.0))
        x0 = [0, x_left, x_right][i]
        x1 = [x_left, x_right, w][i]
        bar_w = x1 - x0 - 20
        bar_h = 14
        y_bar = h - 40
        fill = int(bar_w * max(0.0, min(1.0, r)))
        cv2.rectangle(out, (x0 + 10, y_bar), (x0 + 10 + bar_w, y_bar + bar_h), (60, 60, 60), -1)
        cv2.rectangle(
            out,
            (x0 + 10, y_bar),
            (x0 + 10 + fill, y_bar + bar_h),
            (0, 200, 0),
            -1,
        )
        cv2.putText(
            out,
            f"{side}: {r:.0%} walk",
            (x0 + 10, y_bar - 8),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def render_care_overlay(frame: np.ndarray, care_out, depth) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, int(h * 0.72)
    deg = care_out.safe_direction_deg or 0.0
    length = min(w, h) // 5
    rad = np.deg2rad(deg)
    ex = int(cx + length * np.sin(rad))
    ey = int(cy - length * np.cos(rad))
    color = (0, 0, 255) if care_out.hazard_detected else (0, 220, 0)
    cv2.arrowedLine(out, (cx, cy), (ex, ey), color, 3, tipLength=0.25)
    depth_m = depth.obstacle_depth_m
    src = (depth.metadata or {}).get("source", "unknown")
    lines = [
        f"CARE hazard={care_out.hazard_detected}  safety={care_out.safety_score:.2f}",
        f"direction={deg:+.0f} deg",
        f"depth={depth_m:.2f}m ({src})" if depth_m is not None else f"depth=— ({src})",
    ]
    y = 30
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 26
    return out


def proxy_depth_colormap(
    frame: np.ndarray,
    seg,
    *,
    walkable_set: set[str],
    obstacle_set: set[str],
    id_to_name: dict[int, str],
) -> np.ndarray:
    """Per-pixel pseudo-depth from class map (cloud server fallback)."""
    h, w = frame.shape[:2]
    cm = np.asarray(seg.class_map)
    if cm.ndim == 3:
        cm = cm[0]
    if cm.shape[:2] != (h, w):
        cm = cv2.resize(cm.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)

    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    depth_map = np.full((h, w), 5.0, dtype=np.float32)
    obstacle_ids = [cid for cid, name in id_to_name.items() if name in obstacle_set]
    if obstacle_ids:
        obs = np.isin(cm, obstacle_ids)
        row_depth = np.vectorize(_row_to_depth_m)(ys[:, 0])
        depth_map[obs] = row_depth[np.where(obs)[0]]

    walkable_ids = [cid for cid, name in id_to_name.items() if name in walkable_set]
    if walkable_ids:
        depth_map[np.isin(cm, walkable_ids)] = 4.5

    dmin, dmax = float(depth_map.min()), float(depth_map.max())
    if dmax - dmin < 1e-6:
        norm = np.zeros((h, w), dtype=np.uint8)
    else:
        inv = (dmax - depth_map) / (dmax - dmin)
        norm = (inv * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    cv2.putText(
        color,
        "depth: segmentation proxy (cloud fallback)",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return color


def try_load_depth_pipeline(model_id: str, device: str):
    if not model_id:
        return None
    try:
        from transformers import pipeline

        dev = 0 if device.startswith("cuda") else -1
        return pipeline("depth-estimation", model=model_id, device=dev)
    except Exception as exc:  # noqa: BLE001
        print(f"  [depth] Depth Anything unavailable: {exc}")
        return None


def depth_anything_frame(pipe, frame: np.ndarray, colormap_id: int) -> np.ndarray:
    from PIL import Image

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pipe(Image.fromarray(rgb))
    depth = result["predicted_depth"]
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    depth = np.squeeze(np.asarray(depth, dtype=np.float32))
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax - dmin < 1e-6:
        norm = np.zeros((h, w), dtype=np.uint8)
    else:
        norm = ((depth - dmin) / (dmax - dmin) * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, colormap_id)
    cv2.putText(
        color,
        "depth: Depth Anything V2",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return color


def draw_facts_strip(frame: np.ndarray, record: dict) -> np.ndarray:
    facts = record.get("facts") or {}
    if not facts:
        return frame
    out = frame.copy()
    h = out.shape[0]
    bits = []
    if facts.get("vision_stop"):
        bits.append("VISION_STOP")
    bucket = facts.get("distance_bucket")
    if bucket:
        bits.append(f"bucket={bucket}")
    alerts = record.get("alerts") or []
    if alerts:
        bits.append(f"alerts={len(alerts)}")
    if not bits:
        return out
    text = "  |  ".join(bits)
    cv2.putText(
        out,
        text,
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def _stem_hud_paths(video_path: Path, out_dir: Path) -> tuple[Path, Path]:
    stem = video_path.stem
    return out_dir / f"{stem}_hud_live.mp4", out_dir / f"{stem}_hud_live_audio.mp4"


def _phrase_is_focus(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in FOCUS_PHRASE_KEYWORDS)


def _sparse_hud_update(
    phrase: str,
    *,
    speak: bool,
    now: float,
    last_phrase: str,
    last_update: float,
    display_phrase: str,
) -> tuple[bool, str, str, float]:
    """Return (visible, display_phrase, last_phrase, last_update)."""
    if not speak or not phrase.strip():
        return bool(display_phrase), display_phrase, last_phrase, last_update
    p = phrase.strip()
    if p != last_phrase:
        return True, p, p, now
    if now - last_update >= SPARSE_HUD_MIN_INTERVAL_SEC:
        return True, p, last_phrase, now
    return bool(display_phrase), display_phrase, last_phrase, last_update


def _tts_wav_path(cache_dir: Path, phrase: str, rate: int, engine=None) -> Path:
    del engine  # each synthesis runs in an isolated subprocess (avoids SAPI hangs)
    key = hashlib.md5(f"{phrase}|{rate}".encode()).hexdigest()[:12]
    path = cache_dir / f"tts_{key}.wav"
    if path.is_file() and path.stat().st_size >= 44:
        return path

    snippet = f"""
import pyttsx3
engine = pyttsx3.init()
engine.setProperty("rate", {rate})
engine.save_to_file({phrase!r}, {str(path)!r})
engine.runAndWait()
engine.stop()
"""
    if sys.platform == "win32":
        snippet = "import pythoncom\npythoncom.CoInitialize()\n" + snippet

    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"TTS failed for {phrase!r}: {result.stderr or result.stdout or result.returncode}"
        )
    if not path.is_file() or path.stat().st_size < 44:
        raise RuntimeError(f"TTS did not produce audio for: {phrase!r}")
    return path


class _TtsClipCache:
    """Cache synthesized WAV clips for timeline build."""

    def __init__(self, cache_dir: Path, tts_rate: int, target_rate: int):
        self.cache_dir = cache_dir
        self.tts_rate = tts_rate
        self.target_rate = target_rate
        self._clips: dict[str, list[int]] = {}

    def clip(self, phrase: str) -> list[int]:
        key = _normalize_phrase(phrase)
        cached = self._clips.get(key)
        if cached is not None:
            return cached
        print(f"  [tts] synthesizing: {phrase[:60]}{'…' if len(phrase) > 60 else ''}", flush=True)
        wav_path = _tts_wav_path(self.cache_dir, phrase, self.tts_rate)
        clip_rate, raw = _read_wav_mono(wav_path)
        resampled = _resample_mono(raw, clip_rate, self.target_rate)
        self._clips[key] = resampled
        return resampled


def _normalize_phrase(phrase: str) -> str:
    return " ".join(phrase.strip().lower().split())


def _phrases_similar(a: str, b: str) -> bool:
    na, nb = _normalize_phrase(a), _normalize_phrase(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    return False


def _resample_mono(clip: list[int], clip_rate: int, target_rate: int) -> list[int]:
    if clip_rate == target_rate:
        return clip
    ratio = clip_rate / target_rate
    return [
        clip[min(len(clip) - 1, int(i * ratio))]
        for i in range(int(len(clip) / ratio))
    ]


def _phrase_clip(cache: _TtsClipCache, phrase: str) -> list[int]:
    return cache.clip(phrase)


def _clip_duration_sec(clip: list[int], rate: int) -> float:
    return len(clip) / rate if clip else 0.0


def _schedule_tts_events(
    events: list[tuple[float, str]],
    *,
    cache: _TtsClipCache,
    target_rate: int,
    min_interval: float = TTS_MIN_INTERVAL_SEC,
    gap: float = TTS_GAP_SEC,
) -> list[tuple[float, str]]:
    """Place phrases sequentially: no overlap, min gap, skip near-duplicates."""
    scheduled: list[tuple[float, str]] = []
    last_end = 0.0
    last_phrase = ""
    last_start = -min_interval

    for t_sec, phrase in sorted(events, key=lambda item: item[0]):
        p = phrase.strip()
        if not p:
            continue
        if _phrases_similar(p, last_phrase) and t_sec - last_start < min_interval:
            continue

        clip = _phrase_clip(cache, p)
        dur = _clip_duration_sec(clip, target_rate)
        if dur <= 0:
            continue

        start = max(t_sec, last_end + gap)
        if scheduled and start - last_start < min_interval:
            start = last_start + min_interval
        if start + dur > t_sec + 8.0:
            continue

        scheduled.append((start, p))
        last_end = start + dur
        last_start = start
        last_phrase = p

    return scheduled


def _read_wav_mono(path: Path) -> tuple[int, list[int]]:
    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        if sampwidth != 2:
            raise ValueError(f"unsupported sample width: {sampwidth}")
        frames = wf.readframes(wf.getnframes())
    samples = list(
        int.from_bytes(frames[i : i + 2], "little", signed=True)
        for i in range(0, len(frames), 2 * nch)
    )
    if nch > 1:
        samples = samples[::nch]
    return rate, samples


def _build_timeline_wav(
    events: list[tuple[float, str]],
    *,
    duration_sec: float,
    rate: int,
    cache_dir: Path,
    tts_rate: int,
) -> Path:
    clip_cache = _TtsClipCache(cache_dir, tts_rate, rate)
    scheduled = _schedule_tts_events(events, cache=clip_cache, target_rate=rate)
    print(
        f"  [tts] scheduled {len(scheduled)} non-overlapping clips "
        f"(from {len(events)} raw)",
        flush=True,
    )
    if scheduled:
        last_start, last_phrase = scheduled[-1]
        last_clip = _phrase_clip(clip_cache, last_phrase)
        tail = last_start + _clip_duration_sec(last_clip, rate) + TTS_GAP_SEC
        duration_sec = max(duration_sec, tail)
    n_samples = max(1, int(duration_sec * rate) + rate)
    timeline = [0] * n_samples
    for start_sec, phrase in scheduled:
        clip = _phrase_clip(clip_cache, phrase)
        start = int(start_sec * rate)
        for i, sample in enumerate(clip):
            idx = start + i
            if idx >= n_samples:
                break
            timeline[idx] = max(-32768, min(32767, sample))
    out_path = cache_dir / "timeline.wav"
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"".join(int(s).to_bytes(2, "little", signed=True) for s in timeline))
    return out_path


def _mux_audio(video_path: Path, audio_wav: Path, out_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("  [tts] ffmpeg not found — skipping audio mux", flush=True)
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_wav),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  [tts] ffmpeg failed: {exc.stderr or exc}", file=sys.stderr, flush=True)
        return False


def export_live_hud(args: argparse.Namespace) -> int:
    """Single-pass live demo: original frame + sparse HUD (+ optional TTS mux)."""
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hud_path, hud_audio_path = _stem_hud_paths(video_path, out_dir)

    print("Loading cloud-profile pipeline (live HUD export)...")
    pipe = build_pipeline()
    settings = pipe["settings"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {video_path}", file=sys.stderr)
        return 3

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, args.stride)
    out_fps = src_fps / stride

    writer = _open_writer(hud_path, out_fps, (width, height))
    read_idx = 0
    processed = 0
    frame_id = 0
    t0 = time.time()

    display_phrase = ""
    last_phrase = ""
    last_update = 0.0
    tts_events: list[tuple[float, str]] = []
    tts_last_phrase = ""
    tts_last_time = -1e9
    phrases_seen: set[str] = set()
    focus_phrases: set[str] = set()

    print(
        f"Live HUD: {width}x{height} stride={stride} -> {out_fps:.1f}fps | "
        f"sparse={args.sparse_hud} tts={args.tts}"
    )
    print(f"Writing: {hud_path.resolve()}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if read_idx % stride != 0:
            read_idx += 1
            continue
        read_idx += 1

        t_sec = processed / out_fps
        record = process_frame(
            frame,
            frame_id=frame_id,
            settings=settings,
            segmenter=pipe["segmenter"],
            depth_est=pipe["depth_est"],
            care=pipe["care"],
            interpreter=pipe["interpreter"],
            validator=pipe["validator"],
            tts=pipe["tts"],
            alert_tracker=pipe["alert_tracker"],
            spatial_reasoner=pipe["spatial_reasoner"],
            composer=pipe["composer"],
            trend_tracker=pipe["trend_tracker"],
            position=None,
            client_depth_m=None,
        )

        phrase = str(record.get("phrase") or "").strip()
        speak = bool(record.get("speak"))
        if phrase:
            phrases_seen.add(phrase)
            if _phrase_is_focus(phrase):
                focus_phrases.add(phrase)
        for alert in record.get("alerts") or []:
            ap = str(alert.get("phrase") or "").strip()
            if ap:
                phrases_seen.add(ap)
                if _phrase_is_focus(ap):
                    focus_phrases.add(ap)

        prev_display = display_phrase
        visible, display_phrase, last_phrase, last_update = _sparse_hud_update(
            phrase,
            speak=speak,
            now=t_sec,
            last_phrase=last_phrase,
            last_update=last_update,
            display_phrase=display_phrase,
        )

        if args.tts and speak and visible and display_phrase:
            phrase_changed = _normalize_phrase(display_phrase) != _normalize_phrase(prev_display)
            cooldown_ok = t_sec - tts_last_time >= TTS_MIN_INTERVAL_SEC
            if phrase_changed or (
                cooldown_ok and not _phrases_similar(display_phrase, tts_last_phrase)
            ):
                tts_events.append((t_sec, display_phrase))
                tts_last_phrase = display_phrase
                tts_last_time = t_sec

        out_frame = draw_sparse_navigation_hud(
            frame,
            phrase=display_phrase,
            visible=visible,
            command=str(record.get("command") or ""),
            rationale=str(record.get("rationale") or ""),
            facts=record.get("facts"),
            speak=bool(record.get("speak")),
        )
        writer.write(out_frame)
        processed += 1
        frame_id += 1
        if processed % 15 == 0:
            elapsed = time.time() - t0
            print(f"  {processed} frames ({processed / max(elapsed, 0.1):.2f} fps)...", flush=True)
        if args.max_frames and processed >= args.max_frames:
            break

    cap.release()
    writer.release()
    elapsed = time.time() - t0

    if processed == 0:
        print("ERROR: no frames written", file=sys.stderr)
        return 5

    duration_sec = processed / out_fps
    audio_ok = False
    if args.tts and tts_events:
        print(f"  [tts] {len(tts_events)} raw speech events, building track...", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            try:
                wav = _build_timeline_wav(
                    tts_events,
                    duration_sec=duration_sec,
                    rate=22050,
                    cache_dir=cache,
                    tts_rate=settings.tts_rate,
                )
                audio_ok = _mux_audio(hud_path, wav, hud_audio_path)
            except Exception as exc:  # noqa: BLE001
                print(f"  [tts] failed: {exc}", file=sys.stderr, flush=True)

    print(f"\nDone: {processed} frames in {elapsed:.1f}s ({elapsed / processed:.2f}s/frame)")
    print(f"  hud: {hud_path.resolve()}")
    if args.tts:
        if audio_ok:
            print(f"  hud+audio: {hud_audio_path.resolve()}")
        else:
            print("  hud+audio: (not created)")
    print(f"  phrases ({len(phrases_seen)}): {sorted(phrases_seen)[:12]}{'…' if len(phrases_seen) > 12 else ''}")
    if focus_phrases:
        print(f"  focus matches: {sorted(focus_phrases)}")
    return 0


def build_pipeline():
    settings = apply_cloud_profile(load_settings())
    segmenter = build_segmenter(settings)
    _warmup_segmenter(segmenter, settings)
    depth_est = DepthEstimator(settings)
    care = CareNavigator(settings)
    interpreter = NavigationInterpreter(settings)
    validator = CommandValidator(settings)
    tts = SpeechEngine(settings)
    alert_tracker = (
        AlertTracker.from_settings(settings) if settings.alerts_enabled else None
    )
    spatial_reasoner = SpatialReasoner(settings)
    composer = PhraseComposer(settings)
    trend_tracker = TrendTracker(settings)
    return {
        "settings": settings,
        "segmenter": segmenter,
        "depth_est": depth_est,
        "care": care,
        "interpreter": interpreter,
        "validator": validator,
        "tts": tts,
        "alert_tracker": alert_tracker,
        "spatial_reasoner": spatial_reasoner,
        "composer": composer,
        "trend_tracker": trend_tracker,
    }


def main() -> int:
    args = parse_args()
    if args.sparse_hud or args.tts:
        if args.tts and not args.sparse_hud:
            print("NOTE: --tts requires live export; enabling --sparse-hud.", flush=True)
            args.sparse_hud = True
        args.skip_depth_anything = True
        return export_live_hud(args)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: out_dir / name for name, _ in EXPORTS}

    print("Loading cloud-profile pipeline...")
    pipe = build_pipeline()
    settings = pipe["settings"]
    walkable_set, obstacle_set, _ = _class_sets(settings)

    depth_pipe = None
    depth_mode = "proxy"
    if not args.skip_depth_anything:
        print(f"Trying Depth Anything: {args.depth_model} ...")
        depth_pipe = try_load_depth_pipeline(args.depth_model, args.device)
        if depth_pipe is not None:
            depth_mode = "depth_anything_v2"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {video_path}", file=sys.stderr)
        return 3

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, args.stride)
    out_fps = src_fps / stride
    sbs_w = width * 2

    writers: dict[str, cv2.VideoWriter] = {}
    try:
        writers["seg_overlay"] = _open_writer(paths["01_segmentation_overlay.mp4"], out_fps, (width, height))
        writers["depth"] = _open_writer(paths["02_depth_colormap.mp4"], out_fps, (width, height))
        writers["commands"] = _open_writer(paths["03_commands_overlay.mp4"], out_fps, (width, height))
        writers["masks"] = _open_writer(paths["04_walkable_obstacle_masks.mp4"], out_fps, (width, height))
        writers["side_by_side"] = _open_writer(
            paths["05_original_seg_sidebyside.mp4"], out_fps, (sbs_w, height)
        )
        writers["lanes"] = _open_writer(paths["06_per_side_lanes.mp4"], out_fps, (width, height))
        writers["care"] = _open_writer(paths["07_care_safety.mp4"], out_fps, (width, height))
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        cap.release()
        return 4

    colormap_id = cv2.COLORMAP_INFERNO
    read_idx = 0
    processed = 0
    frame_id = 0
    t0 = time.time()

    print(
        f"Input: {width}x{height} @ {src_fps:.1f}fps, ~{total} frames | "
        f"stride={stride} -> {out_fps:.1f}fps out | depth_mode={depth_mode}"
    )
    print(f"Writing to: {out_dir.resolve()}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if read_idx % stride != 0:
            read_idx += 1
            continue
        read_idx += 1

        record = process_frame(
            frame,
            frame_id=frame_id,
            settings=settings,
            segmenter=pipe["segmenter"],
            depth_est=pipe["depth_est"],
            care=pipe["care"],
            interpreter=pipe["interpreter"],
            validator=pipe["validator"],
            tts=pipe["tts"],
            alert_tracker=pipe["alert_tracker"],
            spatial_reasoner=pipe["spatial_reasoner"],
            composer=pipe["composer"],
            trend_tracker=pipe["trend_tracker"],
            position=None,
            client_depth_m=None,
        )

        seg = pipe["segmenter"].last_segmentation
        if seg is None:
            frame_id += 1
            continue

        id_to_name = _id_to_name_from_seg(seg)
        seg_overlay = render_overlay(frame, segmenter=pipe["segmenter"])

        if depth_pipe is not None:
            depth_vis = depth_anything_frame(depth_pipe, frame, colormap_id)
        else:
            depth_vis = proxy_depth_colormap(
                frame,
                seg,
                walkable_set=walkable_set,
                obstacle_set=obstacle_set,
                id_to_name=id_to_name,
            )

        cmd_frame = draw_navigation_hud(
            seg_overlay,
            phrase=record.get("phrase", ""),
            command=record.get("command", ""),
            speak=bool(record.get("speak")),
            confidence=float(record.get("confidence", 0)),
            rationale=record.get("rationale", ""),
        )
        cmd_frame = draw_facts_strip(cmd_frame, record)

        masks = render_walkable_obstacle_masks(
            frame,
            seg,
            walkable_set=walkable_set,
            obstacle_set=obstacle_set,
            id_to_name=id_to_name,
        )
        side_by_side = np.hstack([frame, seg_overlay])
        lanes = render_per_side_lanes(
            frame,
            seg,
            walkable_by_side_ratios=walkable_by_side(seg),
        )

        depth_result = pipe["depth_est"].predict(
            frame, segmentation=seg, external_depth_m=None
        )
        care_out = pipe["care"].predict(frame, seg, depth_result)
        care_vis = render_care_overlay(frame, care_out, depth_result)

        writers["seg_overlay"].write(seg_overlay)
        writers["depth"].write(depth_vis)
        writers["commands"].write(cmd_frame)
        writers["masks"].write(masks)
        writers["side_by_side"].write(side_by_side)
        writers["lanes"].write(lanes)
        writers["care"].write(care_vis)

        processed += 1
        frame_id += 1
        if processed % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {processed} frames ({processed / max(elapsed, 0.1):.2f} fps)...", flush=True)
        if args.max_frames and processed >= args.max_frames:
            break

    cap.release()
    for w in writers.values():
        w.release()

    elapsed = time.time() - t0
    if processed == 0:
        print("ERROR: no frames written", file=sys.stderr)
        return 5

    print(f"\nDone: {processed} frames in {elapsed:.1f}s")
    print(f"Depth track: {depth_mode}")
    for name, _ in EXPORTS:
        p = paths[name]
        print(f"  {p.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
