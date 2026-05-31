# 🏗️ Smart Cane AI - Complete Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHONE CLIENT (Web UI)                       │
│  - Camera capture (real-time video stream)                      │
│  - GPS + Compass (position & heading)                           │
│  - Depth sensor (optional: Depth Anything V2)                   │
│  - Web Speech API (text-to-speech output)                       │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP POST /process_frame
                         │ (JPEG + GPS + heading + depth_m)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              CLOUD SERVER (Render / Railway)                    │
│                  phone_server_cloud.py                          │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              PERCEPTION LAYER                            │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Segmentation (ADE20K SegFormer-B0 ONNX INT8)       │ │  │
│  │  │ - Input: dynamic NxN RGB (INFERENCE_IMGSZ, 256 prod)│ │  │
│  │  │ - Output: 150-class semantic map                   │ │  │
│  │  │ - Latency: ~94ms @256 (~50ms model on cloud CPU)   │ │  │
│  │  │ - Classes: walkable, obstacles, hazards            │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Depth Estimation (Segmentation Proxy)              │ │  │
│  │  │ - Input: class map + frame                         │ │  │
│  │  │ - Output: obstacle distance (0.5-10m)             │ │  │
│  │  │ - Latency: ~28ms                                   │ │  │
│  │  │ - Method: vertical position + ratio fallback       │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Stairs Detection (Edge Density Heuristic)          │ │  │
│  │  │ - Input: class map                                 │ │  │
│  │  │ - Output: stairs/curb detected (bool)              │ │  │
│  │  │ - Latency: ~15ms (disabled on cloud)               │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              REASONING LAYER                             │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ CARE Navigator (Safety Assessment)                 │ │  │
│  │  │ - Input: segmentation + depth                      │ │  │
│  │  │ - Output: hazard_detected, safety_score, direction │ │  │
│  │  │ - Latency: <1ms                                    │ │  │
│  │  │ - Logic: obstacle ratio vs frame area              │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Spatial Reasoner (Decision Making)                 │ │  │
│  │  │ - Input: seg, depth, CARE, route_cue, stairs      │ │  │
│  │  │ - Output: NavigationCommand + confidence           │ │  │
│  │  │ - Latency: ~5ms                                    │ │  │
│  │  │ - Logic:                                           │ │  │
│  │  │   1. Vision STOP (hazard detected)                 │ │  │
│  │  │   2. Route guidance (map turn)                     │ │  │
│  │  │   3. Per-side walkable analysis                    │ │  │
│  │  │   4. CARE direction fallback                       │ │  │
│  │  │   5. Confidence threshold enforcement (< 0.75)     │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Trend Tracker (Approach Detection)                 │ │  │
│  │  │ - Input: per-side class pixels                     │ │  │
│  │  │ - Output: approaching/receding/static              │ │  │
│  │  │ - Latency: <1ms                                    │ │  │
│  │  │ - Logic: frame-to-frame pixel growth               │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Alert Tracker (Proximity Warnings)                 │ │  │
│  │  │ - Input: per-side hazards + trends                 │ │  │
│  │  │ - Output: "Car approaching" alerts                 │ │  │
│  │  │ - Latency: ~9ms (disabled on cloud)                │ │  │
│  │  │ - Logic: category + growth rate + cooldown         │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Map Guidance (Route Navigation)                    │ │  │
│  │  │ - Input: GPS position + heading + destination      │ │  │
│  │  │ - Output: turn direction + distance                │ │  │
│  │  │ - Latency: <1ms (cached)                           │ │  │
│  │  │ - Backend: OSRM (OpenStreetMap Routing)            │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              OUTPUT LAYER                                │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Phrase Composer (Natural Language)                 │ │  │
│  │  │ - Input: GuidanceFacts (command, hazards, etc)     │ │  │
│  │  │ - Output: Human-readable phrase                    │ │  │
│  │  │ - Latency: <1ms                                    │ │  │
│  │  │ - Examples:                                        │ │  │
│  │  │   "Stop, person on your right"                     │ │  │
│  │  │   "Take a left, path is clear"                     │ │  │
│  │  │   "Slow down, car approaching"                     │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Command Validator (Anti-Spam)                      │ │  │
│  │  │ - Input: NavigationDecision                        │ │  │
│  │  │ - Output: speak (bool) + cooldown tracking         │ │  │
│  │  │ - Latency: <1ms                                    │ │  │
│  │  │ - Logic:                                           │ │  │
│  │  │   1. Dwell filter (hold 2 frames before speaking)  │ │  │
│  │  │   2. Cooldown (8s between same command)            │ │  │
│  │  │   3. Min gap (2s between any utterances)           │ │  │
│  │  │   4. STOP bypasses dwell (safety first)            │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Text-to-Speech (TTS)                               │ │  │
│  │  │ - Input: phrase string                             │ │  │
│  │  │ - Output: audio (phone-side via Web Speech API)    │ │  │
│  │  │ - Latency: <1ms (server-side)                      │ │  │
│  │  │ - Backend: pyttsx3 (laptop) or Web Speech (phone)  │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              PIPELINE ORCHESTRATION                       │  │
│  │  process_frame() - Main entry point                      │  │
│  │  - Coordinates all perception → reasoning → output       │  │
│  │  - Handles frame skipping (PROCESS_EVERY_N_FRAMES=10)   │  │
│  │  - Returns JSON with command + phrase + confidence       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                         │ HTTP JSON response
                         │ {command, confidence, phrase, speak}
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PHONE CLIENT (Web UI)                       │
│  - Display command on screen                                    │
│  - Speak phrase via Web Speech API                              │
│  - Log to console for debugging                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow (Per Frame)

```
1. CAPTURE
   Phone camera → JPEG frame (478×850)
   GPS + heading + optional depth_m

2. TRANSMISSION
   HTTP POST to /process_frame
   Payload: frame (binary) + position (JSON)

3. PERCEPTION
   Frame → Segmentation (277ms)
        → Depth estimation (28ms)
        → Stairs detection (15ms, disabled)
   Output: SegmentationResult + DepthResult

4. REASONING
   Seg + Depth → CARE (safety assessment)
             → Spatial Reasoner (decision)
             → Trend Tracker (approach detection)
             → Alert Tracker (warnings)
             → Map Guidance (route cue)
   Output: NavigationDecision + GuidanceFacts

5. OUTPUT
   Decision + Facts → Phrase Composer (natural language)
                   → Command Validator (anti-spam)
                   → TTS (text-to-speech)
   Output: phrase + speak flag

6. RESPONSE
   JSON → Phone client
   Client speaks phrase via Web Speech API

7. FRAME SKIPPING
   Only process 1 out of 10 frames
   Cache result for frames 2-10
   Reduces latency from 334ms to ~33ms per frame
```

---

## Key Components

### 1. **Perception Layer** (`navigation/perception/`)

| Component | Purpose | Input | Output | Latency |
|-----------|---------|-------|--------|---------|
| **SegFormer ONNX** | Semantic segmentation | RGB frame | 150-class map | 230ms |
| **Depth Estimator** | Obstacle distance | Class map | Distance (m) | 28ms |
| **Stairs Detector** | Curb/step detection | Class map | Stairs (bool) | 15ms |
| **Spatial** | Per-side analysis | Class map | Left/center/right ratios | <1ms |

**Key Classes Detected:**
- **Walkable:** floor, road, sidewalk, path, field
- **Obstacles:** person, car, bicycle, pole, wall, building, door
- **Hazards:** stairs, step, escalator

---

### 2. **Reasoning Layer** (`navigation/reasoning/`)

| Component | Purpose | Input | Output | Latency |
|-----------|---------|-------|--------|---------|
| **CARE** | Safety assessment | Seg + Depth | Hazard flag + direction | <1ms |
| **Spatial Reasoner** | Decision making | All perception | Command + confidence | 5ms |
| **Trend Tracker** | Approach detection | Per-side pixels | Approaching/receding | <1ms |
| **Alert Tracker** | Proximity warnings | Hazards + trends | "Car approaching" | 9ms |
| **Map Guidance** | Route navigation | GPS + heading | Turn direction | <1ms |

**Decision Logic (Priority Order):**
1. Vision STOP (hazard detected) → STOP
2. Route guidance (map turn) → MOVE_LEFT/RIGHT
3. Per-side walkable analysis → GO_FORWARD/SLOW_DOWN
4. CARE direction fallback → MOVE_LEFT/RIGHT/GO_FORWARD
5. Confidence threshold (< 0.75) → STOP

---

### 3. **Output Layer** (`navigation/output/`)

| Component | Purpose | Input | Output | Latency |
|-----------|---------|-------|--------|---------|
| **Phrase Composer** | Natural language | GuidanceFacts | Human phrase | <1ms |
| **Command Validator** | Anti-spam | Decision | speak flag | <1ms |
| **TTS** | Text-to-speech | Phrase | Audio (phone-side) | <1ms |
| **Voice Queue** | Utterance ordering | Multiple phrases | Prioritized queue | <1ms |

**Example Phrases:**
- "Stop, person on your right"
- "Take a left, path is clear"
- "Slow down, car approaching"
- "Go forward, all clear"

---

### 4. **Pipeline** (`navigation/pipeline/`)

**Main Entry Point:** `process_frame()`

```python
def process_frame(
    frame,
    frame_id,
    settings,
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
    stairs_detector,
    position,
    client_depth_m,
) -> dict:
    # 1. Perception
    seg = segmenter.predict(frame)
    depth = depth_est.predict(frame, segmentation=seg)
    care_out = care.predict(frame, seg, depth)
    
    # 2. Reasoning
    stairs = stairs_detector.detect(frame, seg)
    trend_tracker.update(seg.per_side_class_pixels)
    approach_by_category = trend_tracker.classify_all()
    route_cue = _resolve_route_cue(interpreter, settings, position)
    decision, facts = spatial_reasoner.decide(
        seg, depth, care_out, route_cue,
        stairs=stairs,
        approach_by_category=approach_by_category,
    )
    
    # 3. Output
    decision = validator.approve(decision)
    phrase = composer.compose(facts)
    
    # 4. Return
    return {
        "command": decision.command.value,
        "confidence": decision.confidence,
        "phrase": phrase,
        "speak": decision.speak,
        "rationale": decision.rationale,
    }
```

---

## Configuration

### Environment Variables (`.env`)

```bash
# Camera
FRAME_WIDTH=320
FRAME_HEIGHT=240
PROCESS_EVERY_N_FRAMES=10  # Process 1 out of 10 frames

# Segmentation
SEGMENTER_BACKEND=segformer_onnx
INFERENCE_IMGSZ=256
SEGFORMER_ONNX_PATH=segformer_b0_ade20k_int8.onnx

# Reasoning
HAZARD_OBSTACLE_RATIO=0.20  # 20% of frame = obstacle
COMMAND_COOLDOWN_SEC=8.0
COMMAND_DWELL_FRAMES=2
MIN_SPEECH_GAP_SEC=2.0
STOP_HOLD_FRAMES=4

# Features
ALERTS_ENABLED=false
STAIRS_DETECTOR_ENABLED=false
USE_MAP_GUIDANCE=true
```

### YAML Config (`config/default.yaml`)

```yaml
ade20k_segmentation:
  walkable_classes: [floor, road, sidewalk, path, ...]
  obstacle_classes: [person, car, bicycle, pole, ...]
  hazard_classes: [stairs, step, escalator]

phrases:
  directional_warning_simple:
    - "Take a {opposite_side}."
    - "Step toward your {opposite_side}."
  status_update_clear:
    - "Path is clear, take a few steps forward."
```

---

## Deployment Targets

### 1. **Phone (Web UI)**
- **Platform:** iOS/Android browser
- **Capture:** Camera + GPS + Compass
- **Processing:** None (all on cloud)
- **Output:** Web Speech API (TTS)
- **Latency:** Network + cloud processing

### 2. **Cloud (Render)**
- **Platform:** Python Flask server
- **Processing:** All perception + reasoning + output
- **Deployment:** `phone_server_cloud.py`
- **Latency:** ~334ms (target: 30-50ms after optimizations)

### 3. **Laptop (CLI)**
- **Platform:** Python CLI
- **Capture:** Webcam or video file
- **Processing:** All perception + reasoning + output
- **Output:** pyttsx3 (local TTS)
- **Latency:** ~334ms

---

## Latency Breakdown (Current)

Cloud profile (`INFERENCE_IMGSZ=256`, depth skipped, alerts disabled), per
*processed* frame. The 512×512 model input was never architecturally required —
the ONNX graph has dynamic H/W axes, so dropping to 256 is the single biggest
lever and is already the production default.

```
Segmentation (model):   ~50ms  ← dominant
Preprocess (resize/norm): ~2ms
Postprocess (analysis):  ~6ms  (was ~22ms before native-resolution analysis)
Reasoning + composer:    ~3ms
```

**Optimization Strategy:**
1. INFERENCE_IMGSZ=256 (dynamic ONNX input) — done, biggest win
2. Analyze segmentation at native 64×64 logit resolution — done
3. Vectorize postprocessing with `np.bincount` — done
4. Frame skipping (`PROCESS_EVERY_N_FRAMES`) for display smoothness
5. Further: INFERENCE_IMGSZ=192 (~60ms) or a smaller model if needed

---

## Models & Weights

| Model | Size | Latency | Backend | Status |
|-------|------|---------|---------|--------|
| SegFormer-B0 ONNX INT8 | 13MB | 230ms | onnxruntime | ✅ Active |
| SegFormer-B0 PyTorch | 50MB | 350ms | transformers | Fallback |
| OSRM Routing | N/A | <1ms | HTTP API | ✅ Active |

---

## Testing & Validation

**Test Coverage:** 307 tests passing

| Category | Tests | Status |
|----------|-------|--------|
| Segmentation | 9 | ✅ |
| Depth | 6 | ✅ |
| CARE | 2 | ✅ |
| Spatial Reasoner | 9 | ✅ |
| Composer | 12 | ✅ |
| Alerts | 13 | ✅ |
| Validator | 11 | ✅ |
| Voice Queue | 9 | ✅ |
| Trend | 14 | ✅ |
| Distance | 11 | ✅ |
| Maps | 7 | ✅ |
| Phone Server | 12 | ✅ |
| Stairs | 8 | ✅ |
| Other | 77 | ✅ |

---

## Future Enhancements

### Short-term (Latency)
- [ ] Reduce inference resolution to 192×192
- [ ] Use smaller model (SegFormer-Tiny)
- [ ] GPU acceleration on phone

### Medium-term (Features)
- [ ] Real depth sensor integration (Depth Anything V2)
- [ ] LLM-based scene understanding
- [ ] Multi-language support
- [ ] User preference learning

### Long-term (Architecture)
- [ ] LocateAnything for semantic understanding
- [ ] On-device inference (phone GPU)
- [ ] Federated learning for personalization
- [ ] Real-time 3D mapping

---

## Summary

**Current State:**
- ✅ Full pipeline implemented (perception → reasoning → output)
- ✅ 307 tests passing
- ✅ Deployed to Render
- ✅ Phone client functional
- ⚠️ Latency: 334ms (needs optimization)

**Next Priority:**
1. Implement frame skipping (10) → 33ms
2. Disable non-critical components → 20ms
3. Test on phone with optimizations
4. Monitor real-world performance
