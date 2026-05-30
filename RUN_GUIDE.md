# How to Run the Assistive Navigation Model

## Prerequisites

Make sure you're in the project directory and the virtual environment is activated:

```bash
cd /mnt/c/Users/chira/Projects/assistive-navigation
.venv/Scripts/activate  # Windows
# OR: source .venv/bin/activate  # Linux/Mac
```

---

## Quick Examples

### 1. Test with Static Image (Fastest)

```bash
assistive-nav run --image tests/fixtures/sample.jpg
```

**Output:**
- JSON with navigation command
- Voice output via TTS
- Processing time ~4-5 seconds

---

### 2. Live Camera (Basic)

```bash
assistive-nav run --camera 0
```

**What it does:**
- Opens your webcam (index 0)
- Runs YOLO segmentation in real-time
- Outputs navigation commands
- Speaks commands via TTS
- Prints JSON to terminal

**Stop:** Press `Ctrl+C`

---

### 3. Live Camera with Visualization

```bash
assistive-nav run --camera 0 --show-seg
```

**What it does:**
- Everything from example 2
- PLUS: Opens window showing segmentation overlay
- Colored regions for road, sidewalk, obstacles, etc.

**Stop:** Press `q` in window OR `Ctrl+C`

---

### 4. Fast Mode (Recommended for CPU)

```bash
assistive-nav run --fast --camera 0
```

**Optimizations:**
- Lower resolution (256x192)
- Process every 3rd frame
- ~12 FPS performance
- Still accurate for navigation

---

### 5. Demo Mode (Presentations)

```bash
assistive-nav run --demo --camera 0
```

**Features:**
- Fast mode settings
- Frequent voice updates
- Shows video + command overlay
- Great for demos

---

### 6. Map-Assisted Navigation

```bash
assistive-nav run --camera 0 --use-map \
  --current "40.7484,-73.9857" \
  --dest "40.7510,-73.9830"
```

**Features:**
- Fetches walking route from OSRM API
- Turn-by-turn guidance
- Distance: ~1.6km with 45 waypoints
- Obstacle detection overrides route

**With Address:**
```bash
assistive-nav run --use-map \
  --current "40.7484,-73.9857" \
  --dest-address "Empire State Building, New York" \
  --camera 0
```

---

### 7. Save Output Frames

```bash
assistive-nav run --camera 0 --show-seg \
  --seg-save-dir output/my_session \
  --max-frames 100
```

**Output:**
- Saves first 100 frames
- Location: `output/my_session/seg_000000.jpg`, etc.
- Useful for analysis/debugging

---

### 8. Preview Mode (Segmentation Only)

```bash
assistive-nav preview --camera 0
```

**What it does:**
- Shows ONLY segmentation overlay
- No decision making, TTS, or validation
- Faster than full pipeline
- Good for testing vision model

---

## Command Options

### Input Source
- `--image PATH` - Process single image
- `--camera INDEX` - Use webcam (usually 0 or 1)

### Display
- `--show-seg` - Show segmentation overlay window
- `--seg-save-dir DIR` - Save frames to directory

### Performance
- `--fast` - CPU-optimized (256p, every 3rd frame)
- `--demo` - Fast + frequent voice (good for presentations)
- `--max-frames N` - Stop after N frames

### Navigation
- `--use-map` - Enable map-assisted routing
- `--current "LAT,LON"` - Starting position
- `--dest "LAT,LON"` - Destination coordinates
- `--dest-address "ADDRESS"` - Destination address (geocoded)

### Processing
- `--dry-run` - Mock all models (no GPU/weights needed)
- `--no-llm` - Use heuristic decisions (no Ollama needed)

---

## Configuration Files

### .env File

Edit `.env` to configure:

```bash
# Camera settings
CAMERA_INDEX=0
FRAME_WIDTH=320
FRAME_HEIGHT=240

# Model
YOLO_MODEL_PATH=yolo26n-sem.pt

# Performance
PROCESS_EVERY_N_FRAMES=3
INFERENCE_WIDTH=256
INFERENCE_HEIGHT=192

# Voice
TTS_ENABLED=true
TTS_RATE=175

# LLM (optional - requires Ollama)
USE_LLM=false
OPENAI_API_BASE=http://127.0.0.1:11434/v1
OPENAI_MODEL=llama3.1
```

---

## Programmatic Usage

If you want to use the pipeline in your own Python code:

```python
from navigation.config import load_settings
from navigation.capture.camera import CameraStream, load_image
from navigation.perception.segmentation import YoloSegmenter
from navigation.perception.depth import UniDepthEstimator
from navigation.reasoning.care import CareNavigator
from navigation.reasoning.llm import NavigationInterpreter
from navigation.output.validator import CommandValidator
from navigation.output.tts import SpeechEngine
from navigation.models import PerceptionBundle

# Load configuration
settings = load_settings()

# Initialize components
segmenter = YoloSegmenter(settings)
depth_est = UniDepthEstimator(settings)
care = CareNavigator(settings)
interpreter = NavigationInterpreter(settings)
validator = CommandValidator(settings)
tts = SpeechEngine(settings)
tts.warmup()

# Process a single image
frame = load_image("tests/fixtures/sample.jpg")

# Run pipeline
seg = segmenter.predict(frame, dry_run=False)
depth = depth_est.predict(frame, dry_run=True)
care_out = care.predict(frame, seg, depth, dry_run=False)

bundle = PerceptionBundle(
    frame_id=0,
    segmentation=seg,
    depth=depth,
    care=care_out,
)

decision = interpreter.interpret(bundle, dry_run=False)
validated = validator.approve(decision)

if validated.speak:
    phrase = tts.speak(validated)
    print(f"Command: {validated.command.value}")
    print(f"Spoken: {phrase}")
```

---

## Troubleshooting

### Camera not opening?

```bash
# List available cameras
python scripts/list_cameras.py

# Try different camera index
assistive-nav run --camera 1
```

### Slow performance?

```bash
# Use fast mode
assistive-nav run --fast --camera 0

# Or edit .env:
PROCESS_EVERY_N_FRAMES=5  # Skip more frames
INFERENCE_WIDTH=192       # Lower resolution
INFERENCE_HEIGHT=144
```

### Model not found?

The YOLO model downloads automatically on first run. If it fails:

```bash
# Download manually
python -c "from ultralytics import YOLO; YOLO('yolo26n-sem.pt')"
```

### No voice output?

Check TTS settings in `.env`:

```bash
TTS_ENABLED=true

# Test TTS separately
python -c "
from navigation.output.tts import SpeechEngine
from navigation.config import load_settings
tts = SpeechEngine(load_settings())
tts.warmup()
print('TTS working!')
"
```

---

## Performance Tips

### For Real-Time on CPU:
1. Use `--fast` mode
2. Process every 3-5 frames
3. Lower inference resolution (256x192 or 192x144)
4. Close other applications

### For Best Quality:
1. Use full resolution
2. Process every frame
3. Enable GPU (if available)
4. Use `--camera 0` without `--fast`

### For Demos:
1. Use `--demo` mode
2. Pre-test camera and lighting
3. Have backup static image ready
4. Use `--max-frames 300` to limit duration

---

## Output Formats

### Terminal Output (Default)

```
[TTS] Ready — voice: Microsoft David Desktop
[VOICE] Stop
>>> STOP  (stop, 85%)
{
  "frame_id": 0,
  "command": "stop",
  "confidence": 0.85,
  ...
}
```

### Saved Files

When using `--seg-save-dir output/`:

```
output/
├── seg_000000.jpg    # Frame 0
├── seg_000001.jpg    # Frame 1
├── seg_000002.jpg    # Frame 2
└── ...
```

When using `--use-map`:

```
output/
└── route.json        # OSRM route data
```

---

## Example Sessions

### Session 1: Quick Test
```bash
# Test static image
assistive-nav run --image tests/fixtures/sample.jpg

# Expected output: JSON + voice "Stop"
# Time: ~4-5 seconds
```

### Session 2: Live Demo
```bash
# Real-time with visualization
assistive-nav run --demo --camera 0

# Shows: Live video + command overlay
# Performance: ~10-12 FPS
# Stop: Ctrl+C
```

### Session 3: Navigation Test
```bash
# With map routing
assistive-nav run --fast --use-map \
  --current "40.7484,-73.9857" \
  --dest "40.7510,-73.9830" \
  --camera 0

# Fetches route, provides turn guidance
# Obstacle detection active
```

### Session 4: Data Collection
```bash
# Capture 300 frames for analysis
assistive-nav run --camera 0 \
  --seg-save-dir output/session_$(date +%Y%m%d_%H%M%S) \
  --max-frames 300

# Saves all frames to timestamped directory
```

---

## Advanced Usage

### With Ollama LLM

1. Start Ollama:
```bash
ollama serve
```

2. Pull model:
```bash
ollama pull llama3.1
```

3. Enable in `.env`:
```bash
USE_LLM=true
OPENAI_API_BASE=http://127.0.0.1:11434/v1
OPENAI_MODEL=llama3.1
```

4. Run:
```bash
assistive-nav run --camera 0
```

### With CARE Endpoint

1. Deploy CARE server (not included, needs separate setup)

2. Configure `.env`:
```bash
USE_CARE_HTTP=true
CARE_ENDPOINT=http://127.0.0.1:8000/predict
```

3. Run:
```bash
assistive-nav run --camera 0
```

---

## Summary: Most Common Commands

```bash
# Quick test
assistive-nav run --image tests/fixtures/sample.jpg

# Live camera (basic)
assistive-nav run --fast --camera 0

# Live with visualization
assistive-nav run --fast --camera 0 --show-seg

# Demo mode
assistive-nav run --demo --camera 0

# With map navigation
assistive-nav run --fast --use-map \
  --current "40.7484,-73.9857" \
  --dest "40.7510,-73.9830" \
  --camera 0
```

---

## Getting Help

```bash
# Main help
assistive-nav --help

# Run command help
assistive-nav run --help

# Preview command help
assistive-nav preview --help
```

---

**Project Location:** `C:\Users\chira\Projects\assistive-navigation`  
**Documentation:** See README.md, COMPLETE_FLOW_REPORT.md  
**Support:** Check project issues or documentation
