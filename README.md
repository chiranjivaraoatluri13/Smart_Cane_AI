# Smart Cane AI

Real-time walking guidance for smart glasses. A forward camera sends each frame, position, and heading to the navigation service. The service segments the scene, checks obstacles, and follows a walking route, then returns a short spoken command such as “Stop”, “Move left”, or “Go forward”.

The wearer hears the phrase. The camera that captures the frame does not change the decision. While the glasses build is in progress, any forward camera can feed the same pipeline.

## What it does

| Piece | Role |
|-------|------|
| Glasses | Forward camera, position, heading, and audio in the ear |
| Navigation service | Segmentation, depth, route, and one spoken command |
| Segmentation | ADE20K SegFormer, 150 classes. Deployed model is INT8 ONNX (`segformer_b0_ade20k_int8.onnx`) |
| Depth | Metric depth from the glasses when present, otherwise a segmentation proxy |
| Safety | CARE heuristic (optional HTTP). Stop wins over route guidance |
| Route | Google Directions walking routes, with OSRM as the no-key fallback |
| Speech | Short commands spoken to the wearer |

Segmentation is required. Depth, CARE, and the LLM all fall back to rules when their service is off.

## Architecture

The glasses see the path. The service decides. The wearer hears one command.

Diagrams: **[docs/DIAGRAMS.md](docs/DIAGRAMS.md)**. Regenerate the PNGs with `python scripts/render_diagrams.py`.

| Diagram | Preview |
|---------|---------|
| Glasses and spoken guidance | ![glasses and guidance](docs/images/01-glasses-guidance.png) |
| Frame pipeline | ![pipeline](docs/images/02-frame-pipeline.png) |
| Decision priority | ![decision](docs/images/03-decision-priority.png) |

## Pipeline

```text
Glasses camera + position + heading
        |
Frame in
        |
ADE20K SegFormer
        |
Depth (glasses meters, else segmentation proxy)
        |
CARE safety check
        |
Spatial reasoner (+ optional Llama 3.1)
        |
Phrase composer + cooldown validator
        |
Spoken command
        |
Voice in the ear
```

| Stage | Module |
|-------|--------|
| Capture | Forward camera frame into the pipeline |
| Segmentation | `navigation/perception/segmentation_segformer_onnx.py`, `segmentation_segformer.py` |
| Depth | `navigation/perception/depth.py` |
| Safety | `navigation/reasoning/care.py` |
| Decision | `navigation/reasoning/spatial_reasoner.py` |
| Route | `navigation/maps/google_directions.py`, `navigation/maps/router.py` |
| Phrases | `navigation/reasoning/composer.py` |
| Guardrails | `navigation/output/validator.py` |
| Speech | `navigation/output/tts.py` |
| Frame loop | `navigation/pipeline/runner.py` |

Class groups, distance buckets, and phrase templates live in `config/default.yaml` and `config/phrases.yaml`.

## Setup

Python 3.10 or newer. Run these from the cloned repo root.

### PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,segformer,server]"
copy .env.example .env
```

### Command Prompt

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e ".[dev,segformer,server]"
copy .env.example .env
```

| Extra | Install | Purpose |
|-------|---------|---------|
| `segformer` | `pip install -e ".[segformer]"` | torch, transformers, onnxruntime |
| `server` | `pip install -e ".[server]"` | Flask frame server |
| `llm` | `pip install -e ".[llm]"` | OpenAI-compatible Llama client |
| `tts` | `pip install -e ".[tts]"` | Spoken output while developing |
| `dev` | `pip install -e ".[dev]"` | pytest |

Cloud deploys do not install torch. They use `requirements-cloud.txt` and the ONNX file already in the repo.

### Segmentation settings

| Setting | Local default (`.env.example`) | Cloud (`render.yaml`) |
|---------|----------------------------------|------------------------|
| `SEGMENTER_BACKEND` | `segformer` | `segformer_onnx` |
| `SEGFORMER_MODEL_ID` | `nvidia/segformer-b2-finetuned-ade-512-512` | `nvidia/segformer-b0-finetuned-ade-512-512` |
| `SEGFORMER_ONNX_PATH` | — | `segformer_b0_ade20k_int8.onnx` |
| `SEGFORMER_DEVICE` | `auto` | `cpu` |

The B2 checkpoint is slow on CPU. For a live demo, set `SEGMENTER_BACKEND=segformer_onnx` or pass `--fast`.

Export a new ONNX file with:

```powershell
python scripts/export_segformer_onnx.py
```

## Run the pipeline

```powershell
assistive-nav preview --camera 0
assistive-nav run --camera 0 --fast
assistive-nav run --image tests\fixtures\sample.jpg --no-llm
```

Press **q** in the overlay window to stop a preview. Press **Ctrl+C** to stop a live run.

If `tests\fixtures\sample.jpg` is missing:

```powershell
python scripts\create_sample_fixture.py
```

Map demo (vision stop still overrides the route):

```powershell
assistive-nav run --no-llm --use-map `
  --current "40.7484,-73.9857" `
  --dest "40.7510,-73.9830" `
  --image tests\fixtures\sample.jpg
```

`--dest-address "Empire State Building, New York"` geocodes through Nominatim. A successful route is written to `output/route.json`.

| Command | Meaning |
|---------|---------|
| `go_forward` | On the route and aligned with the next waypoint |
| `move_left` / `move_right` | Turn toward the path, or step back when far off it |
| `stop` | Near the destination, or an obstacle in the walking band |

## Development camera

The glasses are the product. Until that camera is wired in, any forward camera can post frames to the same service:

```powershell
python phone_server.py
```

Or double-click `START_PHONE_SERVER.bat`, open the page it prints, allow camera and location, set a destination, then start the camera.

Cloud: connect this repo on [Render](https://render.com) using `render.yaml`. Build command `pip install -r requirements-cloud.txt`. Start command:

```text
gunicorn phone_server_cloud:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
```

Set `GOOGLE_MAPS_API_KEY` in the host dashboard. Without it, routes fall back to OSRM. Details: [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md).

A development camera can estimate depth itself (Depth Anything V2 Small) and post it as `depth_m`. The `/process_frame` response field `depth_source` is `client` or `proxy`. Meters are approximate. Tuning notes: [PHONE_DEPLOYMENT_GUIDE.md](PHONE_DEPLOYMENT_GUIDE.md).

## Tests

```powershell
pytest
```

## More docs

| Guide | Contents |
|-------|----------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Per-frame data flow |
| [RUN_GUIDE.md](RUN_GUIDE.md) | CLI modes: preview, live camera, `--fast`, `--demo` |
| [WALKING_GUIDE.md](WALKING_GUIDE.md) | Outdoor walking with `walking_mode.bat` |
| [PHONE_DEPLOYMENT_GUIDE.md](PHONE_DEPLOYMENT_GUIDE.md) | Development camera and on-device depth |
| [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md) | Render web service |
| [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) | Pre-deploy checks |
| [VIDEO_PROCESSING_GUIDE.md](VIDEO_PROCESSING_GUIDE.md) | Replay a recorded walk |

## Layout

```text
config/                  default.yaml, phrases.yaml
navigation/              capture, perception, reasoning, maps, output, pipeline
phone_client.html        development camera page
phone_server.py          local frame server
phone_server_cloud.py    cloud entrypoint
segformer_b0_ade20k_int8.onnx
scripts/                 fixture, ONNX export, smoke checks
tests/
```

## Limits

This is guidance software, not a certified mobility or medical device. Lighting, latency, and model mistakes can produce a bad suggestion. The validator cuts repeated speech. It does not make a command safe. Try it in a controlled space before relying on it while walking.
