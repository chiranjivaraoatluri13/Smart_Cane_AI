# Real-Time Assistive Navigation System

Voice-guided navigation using a single forward-facing camera. Chains pretrained vision, depth, safety reasoning, and LLM interpretation into spoken commands such as “Move left”, “Go forward”, or “Stop”.

**Status:** Scaffold / integration prototype — model weights and GPU runtimes are configured per module, not bundled.

## Current capabilities

| Area | Today |
|------|--------|
| CLI | `preview` (segmentation overlay only), `run` (full pipeline JSON per frame) |
| Default demo | `run` / `preview` with `--dry-run` — mock Cityscapes-style seg, synthetic depth, heuristic CARE/LLM, no weights |
| Live vision | `preview` / `run` with `[vision]` + `yolo26n-sem.pt` — dense semantic overlay and walkable/obstacle ratios |
| Tests | 16+ pytest cases (segmentation, overlays, validator, heuristics, maps) |
| Training | `scripts/train_semantic.py` exists; pretrained `yolo26n-sem.pt` is enough for typical streets |

Not wired yet: UniDepthV2 live inference, a bundled CARE server, and end-to-end voice on live camera without mocks/heuristics. See [Dry-run vs live](#dry-run-vs-live) below.

## Dry-run vs live

`--dry-run` is **not** a separate package — it is a CLI flag (`preview` / `run`) passed into each pipeline stage.

| Stage | With `--dry-run` | Without `--dry-run` (live) |
|-------|------------------|----------------------------|
| **Camera / window** | Real webcam or image via OpenCV | Same |
| **Segmentation** | Fake stats (`_mock`); overlay uses fixed colored bands (sky, road, person, car) — **no YOLO** | Real `yolo26n-sem.pt` dense class map (needs `pip install -e ".[vision]"`) |
| **Depth** | Synthetic depth from image brightness | Still **mock** until `UNIDEPTH_MODEL_PATH` is set and `depth.py` is wired |
| **CARE** | Heuristic rules (obstacle pixels, depth) | HTTP POST to `CARE_ENDPOINT`; on failure → same heuristics |
| **LLM** | Heuristic commands (`stop`, `go_forward`, …) — **no API** | OpenAI-compatible API (e.g. Ollama); on failure → heuristics |
| **Validator** | Real cooldown / repeat suppression | Same |
| **TTS** | Phrase in JSON; speaks only if `[tts]` installed and `TTS_ENABLED=true` | Same |

**What dry-run is for:** test the loop, camera, overlay window, JSON commands, and validator **without** downloading PyTorch or YOLO weights.

**What tests real potential:** install extras stage by stage (see table in chat docs or run without `--dry-run`).

### Staged testing (cmd)

| Stage | Goal | Command |
|-------|------|---------|
| **0** | Pipeline + mock overlay, no vision | `pip install -e ".[dev]"` then `assistive-nav preview --dry-run --image tests\fixtures\sample.jpg` |
| **1** | Real Cityscapes semantic overlay | `pip install -e ".[vision]"` then `assistive-nav preview --camera 0` |
| **2** | Full chain + overlay (depth still mock) | `assistive-nav run --show-seg --camera 0` |
| **3** | Smarter text + voice | `pip install -e ".[llm]"` + Ollama; `pip install -e ".[tts]"`; `assistive-nav run --camera 0` |
| **4** | Not available yet | UniDepthV2 live, bundled CARE server |

## Architecture

```text
Live Camera Feed
        ↓
YOLO26 Semantic Segmentation
        ↓
UniDepthV2 Depth Estimation
        ↓
CARE Navigation / Safety Prediction
        ↓
Llama 3.1 Decision Interpretation
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
| Segmentation | YOLO26 semantic masks | `navigation/perception/segmentation.py` |
| Depth | UniDepthV2 monocular depth | `navigation/perception/depth.py` |
| Safety | CARE direction / signal | `navigation/reasoning/care.py` |
| Interpretation | Llama 3.1 → JSON command | `navigation/reasoning/llm.py` |
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
pip install -e ".[dev]"
copy .env.example .env
```

Edit `.env` for model paths, API keys, and camera index.

### PowerShell

```powershell
cd $env:USERPROFILE\Projects\assistive-navigation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
```

Edit `.env` for model paths, API keys, and camera index.

### Optional model stacks

Run these **from the project folder** after activating the venv:

| Extra | Install (cmd or PowerShell) | Purpose |
|-------|-----------------------------|---------|
| `vision` | `pip install -e ".[vision]"` | ultralytics (YOLO), torch |
| `depth` | `pip install -e ".[depth]"` | UniDepthV2 (when wired) |
| `llm` | `pip install -e ".[llm]"` | OpenAI-compatible LLM client |
| `tts` | `pip install -e ".[tts]"` | pyttsx3 |

## Segmentation overlay on a laptop

The default model is **Cityscapes semantic** (`yolo26n-sem.pt`): every pixel gets one of 19 classes (road, sidewalk, building, sky, person, car, …). That matches the reference-style colored scene map and drives walkable/obstacle ratios in `config/default.yaml`.

| Goal | Approach |
|------|----------|
| Quick demo, no GPU | `preview --dry-run` → mock Cityscapes-style colors |
| Full road scene (recommended) | `pip install -e ".[vision]"` + `YOLO_MODEL_PATH=yolo26n-sem.pt` (default in `.env.example`) |
| Object-only (COCO) | `YOLO_MODEL_PATH=yolo11n-seg.pt` — instances, not dense road/sky |

**Training:** Not required for typical streets — `yolo26n-sem.pt` is already trained on Cityscapes. Fine-tune only for a custom campus/indoor dataset or new classes; see [Fine-tuning semantic weights](#fine-tuning-semantic-weights).

**CPU vs GPU:** PyTorch uses CUDA when available. On CPU, keep `yolo26n-sem.pt` but lower `FRAME_WIDTH` / `FRAME_HEIGHT` in `.env` (semantic models are heavier than nano instance seg).

### Command Prompt (cmd) — visualization

```bat
cd /d %USERPROFILE%\Projects\assistive-navigation
.venv\Scripts\activate.bat
pip install -e ".[dev]"

REM Mock colored overlay (no weights)
.venv\Scripts\assistive-nav.exe preview --dry-run --image tests\fixtures\sample.jpg

REM Live webcam mock overlay — press q in the window to quit
.venv\Scripts\assistive-nav.exe preview --dry-run --camera 0 --max-frames 30

REM Full pipeline + overlay window
.venv\Scripts\assistive-nav.exe run --dry-run --show-seg --image tests\fixtures\sample.jpg

REM Live Cityscapes-style semantic overlay (downloads yolo26n-sem.pt on first run)
pip install -e ".[vision]"
copy .env.example .env
.venv\Scripts\assistive-nav.exe preview --camera 0

REM Save frames instead of / as well as a window
.venv\Scripts\assistive-nav.exe preview --dry-run --camera 0 --max-frames 10 --seg-save-dir output
```

### PowerShell — visualization

```powershell
assistive-nav preview --dry-run --image tests\fixtures\sample.jpg
assistive-nav preview --dry-run --camera 0 --max-frames 30
assistive-nav run --dry-run --show-seg --image tests\fixtures\sample.jpg
assistive-nav preview --camera 0
```

Press **q** in the OpenCV window to stop. Semantic overlays tint each Cityscapes class in `navigation/perception/visualize.py`; instance `-seg` models fall back to Ultralytics `plot()`.

## Run

### Command Prompt (cmd)

```bat
cd /d %USERPROFILE%\Projects\assistive-navigation
.venv\Scripts\activate.bat

REM Dry run on a still image (no GPU weights required)
.venv\Scripts\assistive-nav.exe run --dry-run --image tests\fixtures\sample.jpg

REM Live webcam (optional [vision] extra + weights)
.venv\Scripts\assistive-nav.exe run --camera 0

REM Alternative entrypoint:
python -m navigation.cli run --dry-run --image tests\fixtures\sample.jpg
```

In cmd, lines starting with `REM` are comments. Do not paste lines that begin with `#` — cmd treats them as commands.

### PowerShell

```powershell
cd $env:USERPROFILE\Projects\assistive-navigation
.\.venv\Scripts\Activate.ps1

# Dry run on a still image
assistive-nav run --dry-run --image tests\fixtures\sample.jpg

# Live webcam
assistive-nav run --camera 0
```

### Create the sample image (first time)

If `tests\fixtures\sample.jpg` is missing:

```bat
cd /d %USERPROFILE%\Projects\assistive-navigation
.venv\Scripts\activate.bat
.venv\Scripts\python.exe scripts\create_sample_fixture.py
```

Or run the full cmd smoke test: `scripts\smoke_cmd.bat`

## Models (vision)

Ultralytics **YOLO26** segmentation checkpoints:

| Weights | Task | Pretraining | Use here |
|---------|------|-------------|----------|
| `yolo26n-sem.pt` (**default**) | Semantic segmentation | Cityscapes (19 classes) | Dense class map — road, sidewalk, sky, vehicles, etc. |
| `yolo26n-seg.pt` | Instance segmentation | COCO (80 classes) | Per-object masks only; no full-scene road labels |

Requires **ultralytics ≥ 8.4.52** for `-sem` weights (`pip install -e ".[vision]"`).

`config/default.yaml` lists the exact Cityscapes class names and navigation groupings (`road_classes`, `walkable_classes`, `obstacle_classes`).

### Fine-tuning semantic weights

Skip training if pretrained Cityscapes weights match your environment. Fine-tune when:

- Lighting/terrain differs strongly from Cityscapes (e.g. indoor corridors, gravel campus paths)
- You need extra classes beyond the 19 Cityscapes labels

```bat
pip install -e ".[vision]"
REM Prepare Cityscapes per Ultralytics docs, then:
python scripts\train_semantic.py --epochs 50 --device 0
REM Point .env at best.pt under runs/semantic/fine-tune/weights/
```

Or: `yolo train model=yolo26n-sem.pt data=cityscapes.yaml task=semantic epochs=50 imgsz=1024 device=0`

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
assistive-nav run --dry-run --no-llm --use-map `
  --current "40.7484,-73.9857" `
  --dest "40.7510,-73.9830" `
  --image tests\fixtures\sample.jpg
```

Geocode an address (Nominatim, free):

```powershell
assistive-nav run --dry-run --no-llm --use-map `
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

Without `--use-map` (or `USE_MAP_GUIDANCE=false`), the pipeline uses the existing CARE/vision heuristics only.

## Configuration

- **Environment:** `.env` — API keys, model paths, camera index, cooldown seconds.
- **YAML:** `config/default.yaml` — class names, command vocabulary, CARE/LLM endpoints.

## Project layout

```
assistive-navigation/
  config/default.yaml
  navigation/
    capture/       # camera input
    perception/    # YOLO26 + UniDepthV2 adapters
    reasoning/     # CARE + Llama 3.1
    maps/          # OSRM routing + map guidance (MVP)
    output/        # validator + TTS
    pipeline/      # main loop
  tests/
```

## Ethics & limitations

This is assistive **guidance**, not a certified mobility or medical device. Always validate in controlled environments before real-world use. Latency, lighting, and model errors can produce unsafe suggestions — the safety validator reduces spam but does not guarantee correctness.
