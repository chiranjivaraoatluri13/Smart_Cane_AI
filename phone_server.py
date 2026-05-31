"""
Flask server for processing frames from phone.

Run this on your laptop, then connect from phone via WiFi.

Usage:
    python phone_server.py

Access from phone:
    http://<laptop-ip>:5000/
"""

import threading
import time
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np

from navigation.config import load_settings
from navigation.maps.router import geocode_address, search_places
from navigation.models import Position
from navigation.pipeline.runner import (
    _build_pipeline_components,
    process_frame as run_process_frame,
)

app = Flask(__name__)

# Initialize navigation components once (expensive!)
print("Loading navigation models...")
settings = load_settings()
(
    segmenter,
    depth_est,
    care,
    interpreter,
    validator,
    tts,
    alert_tracker,
    spatial_reasoner,
    composer,
    voice_queue,
    trend_tracker,
) = _build_pipeline_components(settings)
print("Models loaded! Server ready.")

frame_id = 0
last_process_time = 0

# Pipeline lock: Flask runs ``threaded=True`` so multiple browsers/phones can
# hit /process_frame simultaneously. The segmenter caches ``last_results``
# and the validator owns mutable cooldown state — both must be serialized.
_pipeline_lock = threading.Lock()


def _optional_float(value: str | None) -> float | None:
    """Parse an optional form field as float; return None on missing or invalid."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _no_cache(response):
    response.headers['Cache-Control'] = (
        'no-store, no-cache, must-revalidate, max-age=0'
    )
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def index():
    """Serve the phone client HTML with no-cache headers."""
    return _no_cache(send_from_directory('.', 'phone_client.html'))


@app.route('/test')
def test_page():
    """Serve the camera test page."""
    return _no_cache(send_from_directory('.', 'test_camera.html'))


@app.route('/simple')
def simple_test():
    """Serve the simple test page."""
    return _no_cache(send_from_directory('.', 'simple_test.html'))


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'models_loaded': True,
        'frame_count': frame_id
    })


@app.route('/debug_route', methods=['GET'])
def debug_route():
    """Diagnostic — shows destination and route fetch state."""
    s = interpreter.settings
    mg = getattr(interpreter, '_map_guidance', None)
    loading = (
        interpreter.is_route_loading()
        if hasattr(interpreter, 'is_route_loading')
        else False
    )
    return jsonify({
        'use_map_guidance': s.use_map_guidance,
        'dest_lat': s.dest_lat,
        'dest_lon': s.dest_lon,
        'map_guidance_active': mg is not None,
        'route_loading': loading,
        'route_fetch_failures': getattr(interpreter, '_route_fetch_failures', 0),
        'route_permanent_failure': getattr(
            interpreter, '_route_permanent_failure', False
        ),
        'map_route_attempted': getattr(interpreter, '_map_route_attempted', None),
        'route_waypoints': len(mg.route.waypoints) if mg else 0,
        'route_distance_m': mg.route.distance_m if mg else None,
        'frames_processed': frame_id,
    })


@app.route('/search_places', methods=['GET'])
def search_places_endpoint():
    """Proxy Nominatim — browsers cannot call it directly (CORS / User-Agent)."""
    q = (request.args.get('q') or '').strip()
    if len(q) < 2:
        return jsonify({'ok': True, 'results': []}), 200
    try:
        rows = search_places(q, limit=5)
        return jsonify({
            'ok': True,
            'results': [
                {'display_name': p.display_name, 'lat': p.lat, 'lon': p.lon}
                for p in rows
            ],
        }), 200
    except Exception as e:  # pragma: no cover - network failure path
        return jsonify({
            'ok': False,
            'error': 'search_unavailable',
            'detail': str(e),
        }), 502


@app.route('/set_destination', methods=['POST'])
def set_destination():
    """Geocode an address (or accept lat/lon) and arm map guidance."""
    address = (request.form.get('address') or '').strip()
    dest_lat = _optional_float(request.form.get('lat'))
    dest_lon = _optional_float(request.form.get('lon'))
    if dest_lat is not None and dest_lon is not None:
        lat, lon = dest_lat, dest_lon
    elif address:
        try:
            lat, lon = geocode_address(address)
        except ValueError:
            return jsonify({'ok': False, 'error': 'address_not_found'}), 422
        except Exception as e:  # pragma: no cover - network failure path
            return jsonify({'ok': False, 'error': 'geocoder_unavailable', 'detail': str(e)}), 502
    else:
        return jsonify({'ok': False, 'error': 'missing_address'}), 400

    with _pipeline_lock:
        interpreter.settings = interpreter.settings.model_copy(
            update={
                'dest_lat': lat,
                'dest_lon': lon,
                'use_map_guidance': True,
            }
        )
        if hasattr(interpreter, 'reset_map_state'):
            interpreter.reset_map_state()
        else:
            interpreter._map_guidance = None
            interpreter._map_route_attempted = False

    label = address or f'{lat},{lon}'
    return jsonify({'ok': True, 'lat': lat, 'lon': lon, 'address': label}), 200


@app.route('/process_frame', methods=['POST'])
def process_frame_endpoint():
    """
    Process a frame from the phone camera.

    Expects multipart/form-data with 'frame' field containing JPEG image.
    Optional GPS form fields: lat, lon, heading, accuracy.

    Returns JSON with navigation command, optional facts payload, and any
    spoken proximity alerts that fired this frame.
    """
    global frame_id, last_process_time

    start_time = time.time()

    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400

        file = request.files['frame']
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Could not decode image'}), 400

        position = Position(
            lat=_optional_float(request.form.get('lat')),
            lon=_optional_float(request.form.get('lon')),
            heading_deg=_optional_float(request.form.get('heading')),
            accuracy_m=_optional_float(request.form.get('accuracy')),
        )

        # Real depth measured on-device by the phone (Depth Anything V2).
        # Absent => the pipeline falls back to its geometric proxy.
        client_depth_m = _optional_float(request.form.get('depth_m'))

        with _pipeline_lock:
            record = run_process_frame(
                frame,
                frame_id=frame_id,
                settings=settings,
                segmenter=segmenter,
                depth_est=None,          # skip depth — saves ~28ms
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
            'command': record['command'],
            'confidence': record['confidence'],
            'phrase': record['phrase'],
            'speak': record['speak'],
            'rationale': record['rationale'],
            'frame_id': frame_id,
            'processing_time_ms': int(process_time * 1000),
            'fps': round(fps, 1),
            'alerts': record.get('alerts', []),
        }
        if 'facts' in record:
            response['facts'] = record['facts']
        if 'timings_ms' in record:
            response['timings_ms'] = record['timings_ms']
        response['depth_source'] = 'client' if client_depth_m is not None else 'proxy'

        print(
            f"Frame {frame_id}: {record['command'].upper()} "
            f"({'VOICE' if record.get('speak') else 'silent'}, "
            f"conf={record['confidence']:.2f}, {int(process_time*1000)}ms, {fps:.1f} FPS, "
            f"depth={response['depth_source']})"
        )
        return jsonify(response)

    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get server statistics."""
    return jsonify({
        'frames_processed': frame_id,
        'last_process_time_ms': int(last_process_time * 1000),
        'uptime_seconds': int(time.time()),
    })


if __name__ == '__main__':
    import os
    import socket

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    use_https = os.environ.get("PHONE_SERVER_HTTP", "").strip() not in ("1", "true", "yes")
    scheme = "https" if use_https else "http"

    print("\n" + "=" * 60)
    print("  ASSISTIVE NAVIGATION - PHONE SERVER")
    print("=" * 60)
    print(f"\n  Server starting on: {scheme}://{local_ip}:5000")
    print(f"\n  From your phone, open browser and go to:")
    print(f"  {scheme}://{local_ip}:5000/")
    if use_https:
        print(
            "\n  NOTE: HTTPS uses a self-signed certificate. The phone will warn"
            "\n        about it once — tap 'Advanced' / 'Visit anyway' to accept."
        )
    print("\n  Make sure phone and laptop are on the same WiFi network!")
    print("\n" + "=" * 60 + "\n")

    ssl_context = "adhoc" if use_https else None

    app.run(
        host='0.0.0.0',
        port=5000,
        threaded=True,
        debug=False,
        ssl_context=ssl_context,
    )
