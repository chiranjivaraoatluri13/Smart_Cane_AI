# Real-Time Assistive Navigation System

Voice-guided navigation using a single forward-facing camera. Chains pretrained vision, depth, safety reasoning, and LLM interpretation into spoken commands such as “Move left”, “Go forward”, or “Stop”.

**Status:** Integration prototype — segmentation runs on a real ADE20K SegFormer model; depth and CARE/LLM use real services with rule-based fallbacks.

## Current capabilities

| Area | Today |
|------|--------|
| CLI | `preview` (segmentation overlay only), `run` (full pipeline JSON per frame) |
| Segmentation | ADE20K **SegFormer** (`nvidia/segformer-b2-finetuned-ade-512-512`) — dense 150-class map covering indoor + outdoor surfaces |
| Depth | On-device Depth Anything V2 (phone, posted as `depth_m`) or a segmentation-derived geometric proxy |
| Reasoning | Spatial reasoner + optional Llama 3.1 (OpenAI-compatible) with rule-based fallback |
| Tests | pytest suite (segmentation parsing, overlays, validator, heuristics, maps, depth) |

Not wired yet: UniDepthV2 metric depth, a bundled CARE server.

## Pipeline stages

| Stage | With a real service | Fallback (service off / unavailable) |
|-------|---------------------|--------------------------------------|
| **Camera / window** | Real webcam or image via OpenCV | — |
| **Segmentation** | ADE20K SegFormer dense class map (`[segformer]` extra) | none — segmentation is required |
| **Depth** | Client `depth_m` from on-device Depth Anything V2 | segmentation-derived geometric proxy |
| **CARE** | HTTP POST to `CARE_ENDPOINT` | rule-based heuristic (obstacle pixels, depth) |
| **LLM** | OpenAI-compatible API (e.g. Ollama) | rule-based commands |
| **Validator** | cooldown / repeat suppression | — |
| **TTS** | pyttsx3 (laptop) / Web Speech API (phone) | speaks only if enabled |

## Architecture

**Product path:** develop and validate on a **workstation** (webcam / CLI) → deploy on **smart glasses** (POV camera + hands-free audio). Phone/cloud prototypes in the repo are experimental only.

Visual diagrams (PNG + Mermaid): **[docs/DIAGRAMS.md](docs/DIAGRAMS.md)** · `python scripts/render_diagrams.py`

| Diagram | Preview |
|---------|---------|
| Dev workstation context | ![dev context](docs/images/01-system-context.png) |
| Pipeline | ![pipeline](docs/images/02-pipeline-architecture.png) |
| Dev → glasses roadmap | ![roadmap](docs/images/03-roadmap-dev-to-glasses.png) |
| Decision priority | ![decision](docs/images/05-decision-priority.png) |

```text
Live Camera Feed
        ↓
ADE20K SegFormer Semantic Segmentation
        ↓
Depth (segmentation proxy in dev; wearable metric depth on glasses later)
        ↓
CARE Navigation / Safety Prediction
        ↓
Spatial Reasoner (+ optional Llama 3.1)
        ↓
Structured Navigation Command (JSON)
        ↓
Safety Validator / Cooldown Logic
        ↓
Text-to-Speech
        ↓
Voice Guidance to User
```

| Stage | Module | Package path |
|-------|--------|--------------|
| Capture | OpenCV webcam | `navigation/capture/camera.py` |
| Segmentation | ADE20K SegFormer | `navigation/perception/segmentation_segformer.py` |
| Depth | client depth / segmentation proxy | `navigation/perception/depth.py` |
| Safety | CARE direction / signal | `navigation/reasoning/care.py` |
| Interpretation | Spatial reasoner + Llama 3.1 | `navigation/reasoning/spatial_reasoner.py`, `navigation/reasoning/llm.py` |
| Guardrails | Cooldown + duplicate suppression | `navigation/output/validator.py` |
| Output | TTS | `navigation/output/tts.py` |
| Orchestration | Frame loop | `navigation/pipeline/runner.py` |

## Setup

Use **one shell consistently** — Command Prompt (`cmd`) or PowerShell. Do not mix syntax (for example `$env:...` only works in PowerShell; `%USERPROFILE%` only in cmd).

### Command Prompt (cmd)

```bat
cd /d %USERPROFILE%\Projects\assistive-navigation
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e ".[dev,segformer]"
copy .env.example .env
```

Edit `.env` for the model id, API keys, and camera index.

### PowerShell

```powershell
cd $env:USERPROFILE\Projects\assistive-navigation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,segformer]"
copy .env.example .env
```

### Optional model stacks

Run these **from the project folder** after activating the venv:

| Extra | Install | Purpose |
|-------|---------|---------|
| `segformer` | `pip install -e ".[segformer]"` | torch + transformers (ADE20K SegFormer) |
| `llm` | `pip install -e ".[llm]"` | OpenAI-compatible LLM client |
| `tts` | `pip install -e ".[tts]"` | pyttsx3 |
| `server` | `pip install -e ".[server]"` | Flask (phone server) |

## Segmentation model

The segmenter is an **ADE20K SegFormer** — every pixel gets one of 150 classes
(floor, road, sidewalk, wall, door, stairs, person, car, …). ADE20K covers both
indoor and outdoor scenes, so the model labels hallways, walls, and empty paths
correctly instead of forcing every pixel into a street class. The walkable /
obstacle / hazard groupings live in `config/default.yaml` (`ade20k_segmentation`).

| Setting | Default | Notes |
|---------|---------|-------|
| `SEGMENTER_BACKEND` | `segformer` | the only backend |
| `SEGFORMER_MODEL_ID` | `nvidia/segformer-b2-finetuned-ade-512-512` | use `...-b0-...` for faster CPU inference |
| `SEGFORMER_DEVICE` | `auto` | `cuda` when available, else `cpu` |
| `INFERENCE_IMGSZ` | `0` | model default; lower for speed |

**CPU vs GPU:** the B2 checkpoint is accurate but slow on CPU (~8–9 s/frame).
For real-time on CPU, switch `SEGFORMER_MODEL_ID` to the **B0** variant.

### Visualization (overlay window)

```bat
cd /d %USERPROFILE%\Projects\assistive-navigation
.venv\Scripts\activate.bat
pip install -e ".[segformer]"
copy .env.example .env

REM Live semantic overlay (downloads the SegFormer checkpoint on first run)
.venv\Scripts\assistive-nav.exe preview --camera 0

REM Overlay for a single image
.venv\Scripts\assistive-nav.exe preview --image tests\fixtures\sample.jpg

REM Save overlay frames
.venv\Scripts\assistive-nav.exe preview --camera 0 --max-frames 10 --seg-save-dir output
```

```powershell
assistive-nav preview --image tests\fixtures\sample.jpg
assistive-nav preview --camera 0
```

Press **q** in the OpenCV window to stop. The overlay tints each ADE20K class in `navigation/perception/visualize.py`.

## Run

### Command Prompt (cmd)

```bat
cd /d %USERPROFILE%\Projects\assistive-navigation
.venv\Scripts\activate.bat

REM Single image through the full pipeline
.venv\Scripts\assistive-nav.exe run --image tests\fixtures\sample.jpg

REM Live webcam
.venv\Scripts\assistive-nav.exe run --camera 0

REM Alternative entrypoint
python -m navigation.cli run --image tests\fixtures\sample.jpg
```

In cmd, lines starting with `REM` are comments. Do not paste lines that begin with `#` — cmd treats them as commands.

### PowerShell

```powershell
cd $env:USERPROFILE\Projects\assistive-navigation
.\.venv\Scripts\Activate.ps1

assistive-nav run --image tests\fixtures\sample.jpg
assistive-nav run --camera 0
```

### Create the sample image (first time)

If `tests\fixtures\sample.jpg` is missing:

```bat
.venv\Scripts\python.exe scripts\create_sample_fixture.py
```

## Map-assisted navigation (MVP)

Turn-by-turn style commands from a **walking route** (OSRM public API, no key). Vision still handles **obstacles** — if segmentation/CARE reports a hazard, **stop** wins over map guidance.

### Limitations

- A laptop has **no real GPS or compass**. Set fixed `CURRENT_LAT` / `CURRENT_LON` in `.env` (or `--current`) for demos; production would stream position from a phone.
- `CURRENT_HEADING_DEG` defaults to **0° (north)**. Adjust it to simulate which way you are facing along the route.
- Route fetch needs network access once at startup.

### Set destination and start

**.env** (copy from `.env.example`):

```env
USE_MAP_GUIDANCE=true
CURRENT_LAT=40.7484
CURRENT_LON=-73.9857
CURRENT_HEADING_DEG=45
DEST_LAT=40.7510
DEST_LON=-73.9830
```

**CLI** (overrides `.env`):

```powershell
assistive-nav run --no-llm --use-map `
  --current "40.7484,-73.9857" `
  --dest "40.7510,-73.9830" `
  --image tests\fixtures\sample.jpg
```

Geocode an address (Nominatim, free):

```powershell
assistive-nav run --no-llm --use-map `
  --current "40.7484,-73.9857" `
  --dest-address "Empire State Building, New York"
```

On success, the route polyline is saved to `output/route.json`.

### Commands on a path

| Command | Meaning (map mode) |
|---------|-------------------|
| **go_forward** | On the route and aligned with the next waypoint |
| **move_left** / **move_right** | Turn toward the path, or step back toward the route when far off it |
| **stop** | Within ~15 m of destination, **or** obstacle/hazard detected (vision) |

Without `--use-map` (or `USE_MAP_GUIDANCE=false`), the pipeline uses the CARE/vision heuristics only.

## On-device depth (Depth Anything V2 on the phone)

The phone web client (`phone_client.html`) runs a real monocular depth network —
**Depth Anything V2-Small** via [`@huggingface/transformers`](https://huggingface.co/docs/transformers.js) —
directly in the browser on **WebGPU** (WASM fallback). This gives a genuine depth
estimate without a hosted GPU: the phone's own hardware does the inference.

Per frame (on a `DEPTH_EVERY_N` cadence) the phone estimates depth, reads the
nearest object in the center-bottom walking band, converts it to an approximate
distance in meters, and posts it as the `depth_m` field to `/process_frame`. The
server passes it to `DepthEstimator.predict(external_depth_m=...)`, which feeds
`bucketize()` unchanged. When `depth_m` is absent (model loading, no WebGPU,
older phone), the server falls back to the segmentation proxy automatically.

| Item | Where | Notes |
|------|-------|-------|
| Model | `phone_client.html` `<script type="module">` | `onnx-community/depth-anything-v2-small` |
| Calibration | `DEPTH_CALIBRATION`, `DEPTH_MIN_M`, `DEPTH_MAX_M` | relative depth → approx meters; tune on device |
| Cadence | `DEPTH_EVERY_N` | run depth on 1 of every N frames |
| Active path | `/process_frame` response `depth_source` | `"client"` (on-device) or `"proxy"` (fallback) |

Depth Anything outputs *relative* depth (normalized per frame), so meters are
approximate — consistent with the project's "monotonic, not metric" stance. See
[PHONE_DEPLOYMENT_GUIDE.md](PHONE_DEPLOYMENT_GUIDE.md).

## Configuration

- **Environment:** `.env` — model id, API keys, camera index, cooldown seconds.
- **YAML:** `config/default.yaml` — class groupings, command vocabulary, distance/voice tuning.

## Project layout

```
assistive-navigation/
  config/default.yaml
  navigation/
    capture/       # camera input
    perception/    # SegFormer segmenter + depth adapters
    reasoning/     # CARE + spatial reasoner + Llama 3.1
    maps/          # OSRM routing + map guidance (MVP)
    output/        # validator + TTS
    pipeline/      # main loop
  tests/
```

## Ethics & limitations

This is assistive **guidance**, not a certified mobility or medical device. Always validate in controlled environments before real-world use. Latency, lighting, and model errors can produce unsafe suggestions — the safety validator reduces spam but does not guarantee correctness.
