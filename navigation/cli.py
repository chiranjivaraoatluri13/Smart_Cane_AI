"""CLI entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from navigation.config import Settings, apply_demo_profile, apply_fast_profile, load_settings
from navigation.pipeline.runner import run_pipeline, run_preview


def _configure_logging() -> None:
    """Initialize logging once. Honor LOG_LEVEL env var; default INFO."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_lat_lon(value: str, label: str) -> tuple[float, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"{label} must be 'lat,lon' (got {value!r})"
        )
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"{label} must be numeric lat,lon (got {value!r})"
        ) from e


def _resolve_settings(args: argparse.Namespace) -> Settings:
    settings = load_settings()
    if getattr(args, "demo", False):
        settings = apply_demo_profile(settings)
    elif getattr(args, "fast", False):
        settings = apply_fast_profile(settings)
    if getattr(args, "no_llm", False):
        settings = settings.model_copy(update={"use_llm": False})
    if getattr(args, "benchmark", False):
        settings = settings.model_copy(update={"benchmark_mode": True})

    updates: dict = {}
    if getattr(args, "use_map", False):
        updates["use_map_guidance"] = True
    if getattr(args, "dest", None):
        lat, lon = args.dest
        updates["dest_lat"] = lat
        updates["dest_lon"] = lon
    if getattr(args, "current", None):
        lat, lon = args.current
        updates["current_lat"] = lat
        updates["current_lon"] = lon
    if updates:
        settings = settings.model_copy(update=updates)

    if getattr(args, "dest_address", None):
        from navigation.maps.router import geocode_address

        lat, lon = geocode_address(args.dest_address)
        settings = settings.model_copy(update={"dest_lat": lat, "dest_lon": lon})
        print(f"Geocoded destination: {lat:.6f}, {lon:.6f}", flush=True)

    return settings


def _add_seg_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--show-seg",
        action="store_true",
        help="Open a window with colored segmentation overlay (press q to quit)",
    )
    parser.add_argument(
        "--seg-save-dir",
        type=Path,
        default=None,
        help="Save overlay frames here (e.g. output/)",
    )


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(
        prog="assistive-nav",
        description="Real-time assistive navigation pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run camera or single-image pipeline")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mock perception + heuristic LLM (no weights, no TTS required)",
    )
    run_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use rule-based commands (no Ollama/API); good for demos",
    )
    run_parser.add_argument(
        "--fast",
        action="store_true",
        help="256p YOLO, infer every 3rd frame, no CARE HTTP (smoother CPU)",
    )
    run_parser.add_argument(
        "--demo",
        action="store_true",
        help="Like --fast plus shorter voice cooldown (best for presentations)",
    )
    run_parser.add_argument(
        "--legacy-reasoner",
        dest="legacy_reasoner",
        action="store_true",
        help="Use the original NavigationInterpreter heuristic (skip the spatial reasoner)",
    )
    run_parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Emit per-stage timings in each frame's JSON record (regression diagnostics)",
    )
    run_parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Webcam index (overrides CAMERA_INDEX)",
    )
    run_parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Process a single image instead of live camera",
    )
    run_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N frames (useful for smoke tests)",
    )
    run_parser.add_argument(
        "--use-map",
        "--map",
        dest="use_map",
        action="store_true",
        help="Enable map-assisted routing (OSRM walking route)",
    )
    run_parser.add_argument(
        "--dest",
        type=lambda s: _parse_lat_lon(s, "Destination"),
        default=None,
        metavar="LAT,LON",
        help="Destination coordinates (overrides DEST_LAT/DEST_LON)",
    )
    run_parser.add_argument(
        "--dest-address",
        type=str,
        default=None,
        help="Destination address (geocoded via Nominatim; overrides --dest)",
    )
    run_parser.add_argument(
        "--current",
        type=lambda s: _parse_lat_lon(s, "Current position"),
        default=None,
        metavar="LAT,LON",
        help="Start/current position (overrides CURRENT_LAT/CURRENT_LON)",
    )
    _add_seg_flags(run_parser)

    preview_parser = sub.add_parser(
        "preview",
        help="Show segmentation overlay only (no depth/LLM/TTS)",
    )
    preview_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Colored mock overlay (no YOLO weights)",
    )
    preview_parser.add_argument(
        "--fast",
        action="store_true",
        help="320p, infer every 2nd frame (smoother CPU)",
    )
    preview_parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Webcam index (default: CAMERA_INDEX or 0)",
    )
    preview_parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Show overlay for one image, then wait for a key",
    )
    preview_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N frames",
    )
    preview_parser.add_argument(
        "--seg-save-dir",
        type=Path,
        default=None,
        help="Also save overlay frames here (e.g. output/)",
    )

    args = parser.parse_args()
    if args.command == "run":
        settings = _resolve_settings(args)
        show_seg = args.show_seg or getattr(args, "demo", False)
        if getattr(args, "demo", False) and not args.show_seg:
            print(
                "Demo mode: showing video + on-screen command banner (use --show-seg to hide).",
                flush=True,
            )
        raise SystemExit(
            run_pipeline(
                dry_run=args.dry_run,
                camera_index=args.camera,
                image_path=args.image,
                max_frames=args.max_frames,
                show_seg=show_seg,
                seg_save_dir=args.seg_save_dir,
                settings=settings,
                use_legacy_reasoner=getattr(args, "legacy_reasoner", False),
            )
        )
    if args.command == "preview":
        settings = _resolve_settings(args)
        raise SystemExit(
            run_preview(
                dry_run=args.dry_run,
                camera_index=args.camera,
                image_path=args.image,
                max_frames=args.max_frames,
                seg_save_dir=args.seg_save_dir,
                settings=settings,
            )
        )
