"""
Cloud-deployable phone server — uses ONNX runtime instead of PyTorch.

No torch, no ultralytics. Only onnxruntime (~50MB) for inference.
Designed for Render / Railway free tier.

Required files in repo:
  yolo26n-sem.onnx   (export with: python scripts/export_onnx.py)
  config/default.yaml
  config/phrases.yaml
  navigation/  (full package)
  phone_client.html

Environment variables to set in Render dashboard:
  YOLO_MODEL_PATH=yolo26n-sem.onnx
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

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

# Startup diagnostic — printed before any imports so Render logs show it.
print(f"Python {sys.version}", flush=True)
print(f"YOLO_MODEL_PATH={os.environ.get('YOLO_MODEL_PATH', 'NOT SET')}", flush=True)
print(f"Working dir: {os.getcwd()}", flush=True)
import pathlib
print(f"ONNX file exists: {pathlib.Path(os.environ.get('YOLO_MODEL_PATH', 'yolo26n-sem.onnx')).is_file()}", flush=True)

from navigation.config import load_settings
from navigation.maps.router import geocode_address
from navigation.models import Position
from navigation.output.validator import CommandValidator
from navigation.output.voice_queue import VoiceQueue
from navigation.perception.depth import UniDepthEstimator
from navigation.perception.segmentation_onnx import OnnxSegmenter
from navigation.perception.stairs import StairsDetector
from navigation.output.tts import SpeechEngine
from navigation.reasoning.alerts import AlertTracker
from navigation.reasoning.care import CareNavigator
from navigation.reasoning.composer import PhraseComposer
from navigation.reasoning.llm import NavigationInterpreter
from navigation.reasoning.spatial_reasoner import SpatialReasoner
from navigation.reasoning.trend import TrendTracker
from navigation.pipeline.runner import process_frame, _resolve_route_cue

app = Flask(__name__)

print("Loading navigation models (cloud/ONNX mode)...")
settings = load_settings()

# Use the ONNX segmenter — no torch required.
segmenter = OnnxSegmenter(settings)
depth_est = UniDepthEstimator(settings)
care = CareNavigator(settings)
interpreter = NavigationInterpreter(settings)
validator = CommandValidator(settings)
tts = SpeechEngine(settings)
alert_tracker = AlertTracker.from_settings(settings)
spatial_reasoner = SpatialReasoner(settings)
composer = PhraseComposer(settings)
voice_queue = VoiceQueue(settings)
trend_tracker = TrendTracker(settings)
stairs_detector = StairsDetector(settings)

print("Models loaded! Cloud server ready.")

frame_id = 0
last_process_time = 0
_pipeline_lock = threading.Lock()


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
    return _no_cache(send_from_directory(".", "phone_client.html"))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": True, "frame_count": frame_id})


@app.route("/set_destination", methods=["POST"])
def set_destination():
    address = (request.form.get("address") or "").strip()
    if not address:
        return jsonify({"ok": False, "error": "missing_address"}), 400
    try:
        lat, lon = geocode_address(address)
    except ValueError:
        return jsonify({"ok": False, "error": "address_not_found"}), 422
    except Exception as e:
        return jsonify({"ok": False, "error": "geocoder_unavailable", "detail": str(e)}), 502

    with _pipeline_lock:
        interpreter.settings = interpreter.settings.model_copy(
            update={"dest_lat": lat, "dest_lon": lon, "use_map_guidance": True}
        )
        interpreter._map_guidance = None
        interpreter._map_route_attempted = False

    return jsonify({"ok": True, "lat": lat, "lon": lon, "address": address}), 200


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

        with _pipeline_lock:
            record = process_frame(
                frame,
                frame_id=frame_id,
                settings=settings,
                dry_run=False,
                segmenter=segmenter,
                depth_est=depth_est,
                care=care,
                interpreter=interpreter,
                validator=validator,
                tts=tts,
                alert_tracker=alert_tracker,
                spatial_reasoner=spatial_reasoner,
                composer=composer,
                voice_queue=voice_queue,
                trend_tracker=trend_tracker,
                stairs_detector=stairs_detector,
                position=position,
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
        }
        if "facts" in record:
            response["facts"] = record["facts"]

        print(
            f"Frame {frame_id}: {record['command'].upper()} "
            f"({'VOICE' if record.get('speak') else 'silent'}, "
            f"{int(process_time * 1000)}ms)"
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
