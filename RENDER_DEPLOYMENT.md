# Render Deployment Guide

## Quick Start

### 1. Connect GitHub Repository to Render

1. Go to [render.com](https://render.com)
2. Sign in with GitHub
3. Click **New +** → **Web Service**
4. Select your GitHub repository: `Smart_Cane-AI`
5. Click **Connect**

### 2. Configure Deployment

**Service Settings:**
- **Name:** `smart-cane-ai` (or your preferred name)
- **Environment:** `Python 3.11`
- **Build Command:** `pip install -r requirements-cloud.txt`
- **Start Command:** `gunicorn phone_server_cloud:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
- **Instance Type:** `Standard` (minimum 2GB RAM recommended)

### 3. Set Environment Variables

In Render dashboard, add these environment variables:

```
PYTHON_VERSION=3.11.9
SEGMENTER_BACKEND=segformer_onnx
SEGFORMER_MODEL_ID=nvidia/segformer-b0-finetuned-ade-512-512
SEGFORMER_DEVICE=cpu
SEGFORMER_ONNX_PATH=segformer_b0_ade20k_int8.onnx
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
USE_MAP_GUIDANCE=true
```

### 4. Deploy

Click **Create Web Service** and wait for deployment to complete (~5-10 minutes).

Once deployed, you'll get a URL like: `https://smart-cane-ai.onrender.com`

---

## Phone Client Setup

### 1. Access the Web Interface

Open your phone browser and navigate to:
```
https://smart-cane-ai.onrender.com
```

### 2. Allow Permissions

When prompted:
- OK Allow **Camera** access
- OK Allow **Location** (GPS) access
- OK Allow **Microphone** (for Web Speech API)

### 3. Set Destination

1. Click **Set Destination**
2. Enter an address (e.g., "Empire State Building, New York")
3. Click **Geocode**
4. The server will fetch the walking route

### 4. Start Navigation

1. Click **Start Camera**
2. Point camera at the path ahead
3. Listen for voice guidance:
   - "Go forward" - Continue on route
   - "Move left/right" - Turn toward path
   - "Stop" - Obstacle detected
   - "Person approaching" - Proximity alert

---

## Architecture

```
Phone (Browser)
    ↓
     Camera → JPEG frames
     GPS → lat, lon, heading
     Depth Anything V2 → depth_m (on-device)
    ↓
Render Server (phone_server_cloud.py)
    ↓
     ADE20K SegFormer (ONNX, INT8)
     Depth proxy (fallback)
     CARE safety reasoning
     Spatial reasoner (per-side analysis)
     Map guidance (OSRM routing)
     Phrase composer
     Alert tracker
    ↓
Response JSON
     command: "go_forward" | "move_left" | "move_right" | "slow_down" | "stop"
     phrase: Natural language guidance
     speak: true/false (whether to speak)
     facts: Detailed reasoning (walkable_by_side, hazards, etc.)
     alerts: Proximity alerts ("car approaching", etc.)
    ↓
Phone (Web Speech API)
    ↓
User hears guidance
```

---

## Performance Tuning

### If Slow (>2s per frame):

1. **Reduce inference resolution:**
   ```
   INFERENCE_IMGSZ=256
   ```

2. **Use B0 model (already set):**
   ```
   SEGFORMER_MODEL_ID=nvidia/segformer-b0-finetuned-ade-512-512
   ```

3. **Upgrade instance:**
   - Render free tier: ~2-3s per frame
   - Standard tier: ~1-1.5s per frame
   - Pro tier: ~0.5-1s per frame

### If Out of Memory:

1. Upgrade to **Standard** instance (2GB → 4GB RAM)
2. Reduce `ALERT_MAX_SIMULTANEOUS_CATEGORIES` to 1
3. Disable `ALERTS_ENABLED=false` if needed

---

## Monitoring

### Health Check

```bash
curl https://smart-cane-ai.onrender.com/health
```

Response:
```json
{
  "status": "ok",
  "models_loaded": true,
  "frame_count": 42
}
```

### Debug Route

```bash
curl https://smart-cane-ai.onrender.com/debug_route
```

Shows GPS, destination, and route state.

### Logs

In Render dashboard:
1. Click your service
2. Go to **Logs** tab
3. Watch real-time processing logs

---

## Troubleshooting

### "Models not loading"

**Symptom:** Deployment fails during build

**Solution:**
1. Check Render logs for specific error
2. Ensure `requirements-cloud.txt` is correct
3. Try Standard instance (free tier may timeout)

### "Slow processing"

**Symptom:** >3 seconds per frame

**Solution:**
1. Check instance type (Standard recommended)
2. Reduce `INFERENCE_IMGSZ` to 256
3. Disable alerts: `ALERTS_ENABLED=false`

### "GPS not working"

**Symptom:** `route_cue` is null in response

**Solution:**
1. Ensure phone has GPS enabled
2. Check browser location permissions
3. Verify `USE_MAP_GUIDANCE=true` in env vars

### "No voice output"

**Symptom:** `speak: false` in all responses

**Solution:**
1. Check phone microphone permissions
2. Ensure Web Speech API is supported (Chrome/Edge)
3. Check browser console for errors

---

## API Reference

### POST `/process_frame`

**Request (multipart/form-data):**
```
frame: JPEG image (required)
lat: float (optional, GPS latitude)
lon: float (optional, GPS longitude)
heading: float (optional, compass heading 0-360)
accuracy: float (optional, GPS accuracy in meters)
depth_m: float (optional, on-device depth estimate)
```

**Response:**
```json
{
  "command": "go_forward",
  "confidence": 0.85,
  "phrase": "Continue forward on the path.",
  "speak": true,
  "rationale": "On route, bearing 45° (500 m to destination)",
  "frame_id": 42,
  "processing_time_ms": 1200,
  "fps": 0.8,
  "alerts": [
    {
      "category": "person",
      "phrase": "Person approaching",
      "confidence": 0.75
    }
  ],
  "facts": {
    "command": "go_forward",
    "vision_stop": false,
    "stairs": {"flag": false, "confidence": 0.0},
    "walkable_by_side": {"left": 0.6, "center": 0.8, "right": 0.5},
    "distance_bucket": "mid",
    "distance_phrase": "about 30 feet away",
    "route_cue": {
      "turn": "forward",
      "meters_to_turn": 500.0
    },
    "hazards_by_side": {"left": [], "center": [], "right": []},
    "approach_direction_by_category": {}
  },
  "depth_source": "client"
}
```

### POST `/set_destination`

**Request (form-data):**
```
address: "Empire State Building, New York"
```

**Response:**
```json
{
  "ok": true,
  "lat": 40.7484,
  "lon": -73.9857,
  "address": "Empire State Building, New York"
}
```

### GET `/health`

**Response:**
```json
{
  "status": "ok",
  "models_loaded": true,
  "frame_count": 42
}
```

### GET `/debug_route`

**Response:**
```json
{
  "use_map_guidance": true,
  "dest_lat": 40.7510,
  "dest_lon": -73.9830,
  "map_guidance_active": true,
  "map_route_attempted": true,
  "route_waypoints": 45,
  "route_distance_m": 1600.0,
  "frames_processed": 42
}
```

---

## Cost Estimation

| Instance | RAM | CPU | Cost/month | Frames/sec |
|----------|-----|-----|-----------|-----------|
| Free | 512MB | Shared | $0 | 0.3-0.5 |
| Standard | 2GB | 0.5 CPU | $7 | 0.8-1.2 |
| Pro | 4GB | 1 CPU | $21 | 1.5-2.0 |

**Recommendation:** Start with **Standard** ($7/month) for reliable performance.

---

## Next Steps

1. OK Push code to GitHub
2. OK Deploy to Render
3. OK Test with phone client
4. OK Adjust environment variables based on performance
5. OK Monitor logs and metrics

---

## Support

For issues:
1. Check Render logs
2. Review `/debug_route` endpoint
3. Test `/health` endpoint
4. Check phone browser console (F12)

---

**Status:** Ready for production phone-only deployment OK
