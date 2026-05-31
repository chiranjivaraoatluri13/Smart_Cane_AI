# 🚀 AGGRESSIVE LATENCY OPTIMIZATION PLAN

## Current Bottleneck Analysis
- **Total Latency:** 334ms (3 FPS)
- **Segmentation:** 277ms (83% of total) ← **MAIN BOTTLENECK**
- **Depth:** 28ms (8%)
- **Stairs:** 15ms (4%)
- **Alerts:** 9ms (3%)
- **Other:** 5ms (2%)

---

## Root Cause: SegFormer Model Size
The SegFormer-B0 ONNX INT8 model is still **too large** for real-time inference:
- Model: 512×512 input resolution (required by architecture)
- Inference time: ~230-240ms per frame on CPU
- Even with INT8 quantization, it's still slow

---

## Optimization Strategy (Priority Order)

### ✅ OPTION 1: Skip Segmentation on Most Frames (FASTEST)
**Impact:** 3x speedup (334ms → 100ms)

**Implementation:**
```python
# Only run segmentation every N frames, cache result for others
PROCESS_EVERY_N_FRAMES = 10  # Process 1 out of 10 frames
# Use cached segmentation for frames 2-10
```

**Pros:**
- Simplest to implement
- No accuracy loss (segmentation doesn't change much frame-to-frame)
- 3x speedup immediately

**Cons:**
- Slightly delayed response to sudden obstacles
- But with 10 FPS output, delay is only 100ms (acceptable)

**Recommendation:** ✅ **DO THIS FIRST**

---

### ✅ OPTION 2: Reduce Model Input Size (MODERATE)
**Impact:** 2x speedup (277ms → 140ms)

**Implementation:**
```python
# Use 256×256 input instead of 512×512
INFERENCE_IMGSZ = 256
```

**Pros:**
- 4x fewer pixels = ~2x faster (due to model architecture)
- Minimal accuracy loss for navigation

**Cons:**
- Slightly coarser segmentation boundaries

**Status:** ✅ **DONE.** The exported ONNX model has *dynamic* height/width axes
(`scripts/export_segformer_onnx.py` sets `dynamic_axes` for dims 2 and 3), so it
runs at any input size with no re-export. Setting `INFERENCE_IMGSZ=256` is honored
by `SegformerOnnxSegmenter.predict()`. Measured on CPU (478×850 frame):

| INFERENCE_IMGSZ | inference time | FPS |
|-----------------|---------------|-----|
| 512 (old default) | ~219 ms | 4.6 |
| 384 | ~123 ms | 8.1 |
| 256 (**current prod**, `render.yaml`) | ~94 ms | 10.6 |
| 192 | ~60 ms | 16.8 |

The earlier "fixed at 512×512" note was incorrect.

---

### ✅ OPTION 3: Disable Expensive Components (QUICK WINS)
**Impact:** 50ms saved (334ms → 284ms)

**Components to disable:**
1. **Stairs Detection** (15ms) → Disable if not critical
2. **Alerts** (9ms) → Already disabled on cloud
3. **Depth Estimation** (28ms) → Use segmentation proxy only

**Implementation:**
```python
# In render.yaml
STAIRS_ENABLED=false
ALERTS_ENABLED=false
USE_DEPTH_ESTIMATION=false  # Use segmentation proxy only
```

**Pros:**
- Quick to implement
- 50ms saved

**Cons:**
- Lose stairs/curb detection
- Lose proximity alerts

**Recommendation:** ✅ **DO THIS SECOND**

---

### ✅ OPTION 4: Use Smaller Model (BEST LONG-TERM)
**Impact:** 5x speedup (277ms → 55ms)

**Options:**
1. **SegFormer-Tiny** (~50ms/frame)
2. **MobileNet-based segmenter** (~30ms/frame)
3. **Lightweight custom model** (~20ms/frame)

**Pros:**
- Permanent solution
- Scales well

**Cons:**
- Requires model retraining/export
- Potential accuracy loss

**Recommendation:** ⏳ **DO THIS LATER (if needed)**

---

### ✅ OPTION 5: Analyze at the Model's Native Resolution (DONE)
**Impact:** Postprocessing 22ms → 6.5ms (~3.4x), no decision change

The segmenter only produces logits at the model stride (e.g. 64×64 for a 256px
input). The old path *upscaled* that to the full camera frame (478×850 ≈ 406k px)
and then ran the per-side / obstacle / walkable analysis over all 406k pixels —
even though every reasoning consumer reads **ratios** (scale-invariant) or weighted
counts paired with `metadata["shape"]`.

`SegformerOnnxSegmenter.predict()` now runs `_parse_class_map` on the native 64×64
map and only upscales the dense `class_map` for the consumers that genuinely need
frame-resolution masks (laptop overlay, depth proxy, proximity alerts). Decisions
are identical; the O(display_pixels) scans disappear.

Additionally, `_parse_class_map` and the per-side helpers (`navigation/perception/
spatial.py`) and the alert tally (`navigation/reasoning/alerts.py`) were rewritten
from per-class boolean-mask loops to single `np.bincount` passes — same outputs,
one C-level scan instead of one scan per distinct class id. This speeds up the
laptop (transformers) path too.

---

### ✅ OPTION 6: Fix onnxruntime thread oversubscription (DONE — big cloud win)
**Impact:** Up to ~5× on a multi-core / mis-sized container; likely the main
cause of slow Render inference.

The session was created with `intra_op_num_threads = 0` ("use all cores"). In a
container, ORT reads the **host's** physical core count, not the cgroup CPU
limit, so on a 1-vCPU box it spawns 8+ threads and thrashes. The INT8 model is
especially sensitive (dynamic-quant ops scale poorly across threads). Measured
on an 8-core CPU, 256px input:

| threads | INT8 256 | FP32 256 |
|---------|----------|----------|
| 1 | 94 ms | 129 ms |
| 2 | 63 ms | 78 ms |
| 4 | 45 ms | 37 ms |
| all (8) | **220 ms** ⚠️ | 31 ms |

Fix: `onnx_intra_op_threads` setting (`ONNX_INTRA_OP_THREADS` env), defaulting to
`min(4, cpu_count)` instead of "all", and set to `"1"` in `render.yaml` to match
the allocated vCPU. **Set the same env on Railway** (`railway.toml` reads env
from the dashboard). Tuning notes:
- 1 vCPU plan → `ONNX_INTRA_OP_THREADS=1`
- 2 vCPU plan → `=2`
- ≥4 real cores → INT8 plateaus/regresses; the **FP32** model
  (`SEGFORMER_ONNX_PATH=segformer_b0_ade20k.onnx`) with more threads is fastest
  (~31 ms at 8 threads). INT8 only wins on 1–2 cores, which is the typical
  free/standard cloud box — so INT8 stays the cloud default.

Also: preprocessing now resizes *before* the BGR→RGB convert, so the colour swap
runs on the 256×256 image instead of the full camera frame.

---

## Recommended Implementation (IMMEDIATE)

### Step 1: Aggressive Frame Skipping
```python
# .env
PROCESS_EVERY_N_FRAMES=10  # Process 1 out of 10 frames
```
**Result:** 334ms → 100ms (3.3x speedup)

### Step 2: Disable Non-Critical Components
```yaml
# render.yaml
STAIRS_ENABLED: false
ALERTS_ENABLED: false
```
**Result:** 100ms → 50ms (additional 2x speedup)

### Step 3: Skip Depth Estimation
```python
# In phone_server_cloud.py
# Use segmentation proxy only, skip UniDepth
```
**Result:** 50ms → 30ms (additional 1.7x speedup)

---

## Expected Final Performance

| Strategy | Latency | FPS | Speedup |
|----------|---------|-----|---------|
| Current | 334ms | 3.0 | 1x |
| + Frame Skipping (10) | 100ms | 10.0 | 3.3x |
| + Disable Stairs/Alerts | 50ms | 20.0 | 6.7x |
| + Skip Depth | 30ms | 33.0 | 11x |

**Final Target:** 30-50ms latency (20-33 FPS) ✅

---

## Implementation Priority

1. **IMMEDIATE (5 min):**
   - Set `PROCESS_EVERY_N_FRAMES=10`
   - Disable stairs detection
   - Disable alerts

2. **QUICK (10 min):**
   - Skip depth estimation
   - Use segmentation proxy only

3. **LATER (if needed):**
   - Export smaller ONNX model
   - Use MobileNet-based segmenter

---

## Testing Plan

1. Deploy with frame skipping (10)
2. Test on phone - measure response time
3. If still slow, disable depth estimation
4. If still slow, use smaller model

---

## Notes

- Frame skipping is safe because:
  - Segmentation doesn't change much frame-to-frame
  - User movement is slow (walking speed)
  - 100ms delay is imperceptible
  
- Depth estimation is redundant because:
  - Segmentation proxy works well
  - No real depth sensor on phone
  - Saves 28ms per frame

- Stairs detection is optional because:
  - Not critical for basic navigation
  - Can be re-enabled later if needed
  - Saves 15ms per frame
