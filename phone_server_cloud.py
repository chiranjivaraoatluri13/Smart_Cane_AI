"""
Cloud-deployable phone server.

Runs the full navigation pipeline with the ADE20K SegFormer segmenter.
Designed for Render / Railway. Prefer the lightweight SegFormer-B0 checkpoint
on constrained free-tier hosts (set SEGFORMER_MODEL_ID).

Required files in repo:
  config/default.yaml
  config/phrases.yaml
  navigation/  (full package)
  phone_client.html

Environment variables to set in Render dashboard:
  SEGMENTER_BACKEND=segformer
  SEGFORMER_MODEL_ID=nvidia/segformer-b0-finetuned-ade-512-512
  TTS_ENABLED=false
  USE_LLM=false
  USE_CARE_HTTP=false
  COMMAND_COOLDOWN_SEC=8.0
  COMMAND_DWELL_FRAMES=4
  MIN_SPEECH_GAP_SEC=3.0
  STOP_HOLD_FRAMES=8
  HAZARD_OBSTACLE_RATIO=0.10
  ALERTS_ENABLED=true
  ALERT_MIN_WEIGHTED_PIXELS=1500.0
  ALERT_MAX_SIMULTANEOUS_CATEGORIES=2
"""

from __future__ import annotations

import os
import sys
import threading
import time
import pathlib

# Ensure the project root is on sys.path so `navigation.*` is importable
# regardless of how the package was installed. This is the most reliable
# approach for cloud deployments where editable installs can be finicky.
_project_root = pathlib.Path(__file__).parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

# Startup diagnostic — printed before any imports so Render logs show it.
print(f"Python {sys.version}", flush=True)
print(f"SEGMENTER_BACKEND={os.environ.get('SEGMENTER_BACKEND', 'segformer')}", flush=True)
print(f"SEGFORMER_MODEL_ID={os.environ.get('SEGFORMER_MODEL_ID', 'default')}", flush=True)
print(f"Working dir: {os.getcwd()}", flush=True)
print(f"sys.path[0]: {sys.path[0]}", flush=True)
print(f"navigation package: {(_project_root / 'navigation' / '__init__.py').is_file()}", flush=True)

from navigation.config import apply_cloud_profile, load_settings
from navigation.maps.router import geocode_address, reverse_geocode, search_places
from navigation.models import Position
from navigation.output.validator import CommandValidator
from navigation.output.voice_queue import VoiceQueue
from navigation.perception.segmentation_base import build_segmenter
from navigation.output.tts import SpeechEngine
from navigation.reasoning.alerts import AlertTracker
from navigation.reasoning.care import CareNavigator
from navigation.reasoning.composer import PhraseComposer
from navigation.reasoning.llm import NavigationInterpreter
from navigation.reasoning.spatial_reasoner import SpatialReasoner
from navigation.reasoning.trend import TrendTracker
from navigation.pipeline.runner import process_frame, _resolve_route_cue, _warmup_segmenter

app = Flask(__name__)

print("Loading navigation models (cloud/ONNX mode)...")
settings = apply_cloud_profile(load_settings())
print(
    f"Cloud profile: INFERENCE_IMGSZ={settings.inference_imgsz} "
    f"ONNX_THREADS={settings.onnx_intra_op_threads} "
    f"UPSCALE_MAP={settings.seg_upscale_class_map}",
    flush=True,
)

# ADE20K SegFormer segmenter (indoor + outdoor, no closed-set hallucination).
segmenter = build_segmenter(settings)
_warmup_segmenter(segmenter, settings)
# Skip depth estimation on cloud - use default "mid" distance
care = CareNavigator(settings)
interpreter = NavigationInterpreter(settings)
validator = CommandValidator(settings)
tts = SpeechEngine(settings)
alert_tracker = (
    AlertTracker.from_settings(settings) if settings.alerts_enabled else None
)
spatial_reasoner = SpatialReasoner(settings)
composer = PhraseComposer(settings)
voice_queue = VoiceQueue(settings)
trend_tracker = TrendTracker(settings)

print("Models loaded! Cloud server ready.")

frame_id = 0
last_process_time = 0
_pipeline_lock = threading.Lock()
_is_processing = False  # non-blocking busy flag


def _optional_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _no_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def index():
    # Serve phone_client.html from the project root
    html_path = _project_root / "phone_client.html"
    if html_path.is_file():
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        response = app.make_response(content)
        response.headers["Content-Type"] = "text/html; charset=utf-8"
        return _no_cache(response)
    return jsonify({"error": "phone_client.html not found"}), 404


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": True, "frame_count": frame_id})


@app.route("/search_places", methods=["GET"])
def search_places_endpoint():
    """Proxy Nominatim — browsers cannot call it directly (CORS / User-Agent)."""
    q = (request.args.get("q") or "").strip()
    if len(q) < 2:
        return jsonify({"ok": True, "results": []}), 200
    near_lat = _optional_float(request.args.get("lat"))
    near_lon = _optional_float(request.args.get("lon"))
    try:
        rows = search_places(q, limit=5, near_lat=near_lat, near_lon=near_lon)
        return jsonify({
            "ok": True,
            "results": [
                {
                    "display_name": p.display_name,
                    "lat": p.lat,
                    "lon": p.lon,
                    "distance_m": p.distance_m,
                }
                for p in rows
            ],
        }), 200
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": "search_unavailable",
            "detail": str(e),
        }), 502


@app.route("/reverse_geocode", methods=["GET"])
def reverse_geocode_endpoint():
    lat = _optional_float(request.args.get("lat"))
    lon = _optional_float(request.args.get("lon"))
    if lat is None or lon is None:
        return jsonify({"ok": False, "error": "missing_coordinates"}), 400
    try:
        label = reverse_geocode(lat, lon)
        return jsonify({
            "ok": True,
            "display_name": label,
            "lat": lat,
            "lon": lon,
        }), 200
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": "reverse_unavailable",
            "detail": str(e),
        }), 502


@app.route("/set_destination", methods=["POST"])
def set_destination():
    address = (request.form.get("address") or "").strip()
    dest_lat = _optional_float(request.form.get("lat"))
    dest_lon = _optional_float(request.form.get("lon"))
    if dest_lat is not None and dest_lon is not None:
        lat, lon = dest_lat, dest_lon
    elif address:
        near_lat = _optional_float(request.form.get("near_lat"))
        near_lon = _optional_float(request.form.get("near_lon"))
        try:
            lat, lon = geocode_address(
                address, near_lat=near_lat, near_lon=near_lon
            )
        except ValueError:
            return jsonify({"ok": False, "error": "address_not_found"}), 422
        except Exception as e:
            return jsonify({"ok": False, "error": "geocoder_unavailable", "detail": str(e)}), 502
    else:
        return jsonify({"ok": False, "error": "missing_address"}), 400

    global settings
    with _pipeline_lock:
        interpreter.settings = interpreter.settings.model_copy(
            update={"dest_lat": lat, "dest_lon": lon, "use_map_guidance": True}
        )
        settings = interpreter.settings
        interpreter.reset_map_state()

    return jsonify({"ok": True, "lat": lat, "lon": lon, "address": address or f"{lat},{lon}"}), 200


@app.route("/process_frame", methods=["POST"])
def process_frame_endpoint():
    global frame_id, last_process_time
    start_time = time.time()

    try:
        if "frame" not in request.files:
            return jsonify({"error": "No frame provided"}), 400

        file = request.files["frame"]
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        position = Position(
            lat=_optional_float(request.form.get("lat")),
            lon=_optional_float(request.form.get("lon")),
            heading_deg=_optional_float(request.form.get("heading")),
            accuracy_m=_optional_float(request.form.get("accuracy")),
        )

        # Real depth measured on-device by the phone (Depth Anything V2).
        # Absent => the pipeline falls back to its geometric proxy.
        client_depth_m = _optional_float(request.form.get("depth_m"))

        with _pipeline_lock:
            record = process_frame(
                frame,
                frame_id=frame_id,
                settings=interpreter.settings,
                segmenter=segmenter,
                depth_est=None,  # Skip depth estimation (saves 28ms)
                care=care,
                interpreter=interpreter,
                validator=validator,
                tts=tts,
                alert_tracker=alert_tracker,
                spatial_reasoner=spatial_reasoner,
                composer=composer,
                voice_queue=voice_queue,
                trend_tracker=trend_tracker,
                position=position,
                client_depth_m=client_depth_m,
            )
            process_time = time.time() - start_time
            fps = 1.0 / process_time if process_time > 0 else 0
            frame_id += 1
            last_process_time = process_time

        response = {
            "command": record["command"],
            "confidence": record["confidence"],
            "phrase": record["phrase"],
            "speak": record["speak"],
            "rationale": record["rationale"],
            "frame_id": frame_id,
            "processing_time_ms": int(process_time * 1000),
            "fps": round(fps, 1),
            "alerts": record.get("alerts", []),
            "route_status": {
                "destination_set": interpreter.settings.map_destination_set,
                "loading": interpreter.is_route_loading(),
                "active": getattr(interpreter, "_map_guidance", None) is not None,
                "failed": getattr(interpreter, "_route_permanent_failure", False),
            },
        }
        if "facts" in record:
            response["facts"] = record["facts"]
        response["depth_source"] = (
            "client" if client_depth_m is not None else "proxy"
        )

        print(
            f"Frame {frame_id}: {record['command'].upper()} "
            f"({'VOICE' if record.get('speak') else 'silent'}, "
            f"{int(process_time * 1000)}ms, "
            f"depth={response['depth_source']})"
        )
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "frames_processed": frame_id,
        "last_process_time_ms": int(last_process_time * 1000),
    })


@app.route("/debug_route", methods=["GET"])
def debug_route():
    """Diagnostic — shows GPS, destination, and route state."""
    s = interpreter.settings
    mg = getattr(interpreter, "_map_guidance", None)
    return jsonify({
        "use_map_guidance": s.use_map_guidance,
        "dest_lat": s.dest_lat,
        "dest_lon": s.dest_lon,
        "map_guidance_active": mg is not None,
        "route_loading": interpreter.is_route_loading(),
        "route_fetch_failures": getattr(interpreter, "_route_fetch_failures", 0),
        "route_permanent_failure": getattr(interpreter, "_route_permanent_failure", False),
        "map_route_attempted": getattr(interpreter, "_map_route_attempted", None),
        "route_waypoints": len(mg.route.waypoints) if mg else 0,
        "route_distance_m": mg.route.distance_m if mg else None,
        "frames_processed": frame_id,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
