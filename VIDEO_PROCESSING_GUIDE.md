# Video Processing Guide

## 🎥 Process Pre-Recorded Videos

The assistive navigation system can now process **pre-recorded videos** to analyze what went wrong during your test.

---

## 🚀 Quick Start

### **Method 1: Use the Batch Script** (Easiest)

```batch
cd C:\Users\chira\Projects\assistive-navigation
analyze_recording.bat
```

This will:
- Process your screen recording from today
- Show visualization with colored overlays
- Display command history and statistics
- Tell you what commands were issued and when

---

### **Method 2: Manual Command**

```batch
cd C:\Users\chira\Projects\assistive-navigation
.venv\Scripts\activate.bat
python process_video.py "C:\Users\chira\Videos\Screen Recordings\Screen Recording 2026-05-25 193552.mp4" --show
```

---

## 📊 What You'll See

### **Console Output:**
```
============================================================
VIDEO PROCESSING - ASSISTIVE NAVIGATION
============================================================
Video: Screen Recording 2026-05-25 193552.mp4
Resolution: 1920x1080
FPS: 30.0
Frames: 10752
Duration: 358.4s
============================================================

Processing frames...
============================================================
Frame      0 (   0.0s): GO_FORWARD   [VOICE]   conf=0.90
Frame      3 (   0.1s): GO_FORWARD   [silent]  conf=0.88
Frame      6 (   0.2s): STOP         [VOICE]   conf=0.85
Frame      9 (   0.3s): STOP         [silent]  conf=0.85
Frame     12 (   0.4s): STOP         [silent]  conf=0.85
...
```

### **Visualization Window:**
- **Green overlay** = Walkable areas (sidewalk, road)
- **Red/Orange** = Obstacles (people, cars)
- **HUD overlay** shows:
  - Current command
  - Confidence level
  - Rationale
  - Voice status

### **Final Summary:**
```
============================================================
PROCESSING COMPLETE
============================================================
Frames processed: 3584

Command Distribution:
  stop           : 3200 (89.3%)
  go_forward     :  300 ( 8.4%)
  move_left      :   50 ( 1.4%)
  move_right     :   34 ( 0.9%)

Voice commands spoken: 450
Silent commands: 3134

Avg obstacle pixels: 185000
Avg walkable ratio: 15.3%
============================================================
```

---

## 🎯 Advanced Usage

### **Process Any Video:**
```batch
python process_video.py "path\to\your\video.mp4" --show
```

### **Save Annotated Frames:**
```batch
python process_video.py "video.mp4" --show --save-dir output\analysis
```
Creates annotated frames in `output/analysis/` with:
- `frame_000000.jpg`, `frame_000003.jpg`, etc.
- `analysis_summary.json` with detailed statistics

### **Process First 500 Frames Only:**
```batch
python process_video.py "video.mp4" --show --max-frames 500
```

### **With Map Navigation:**
```batch
python process_video.py "video.mp4" --show --use-map ^
  --current "33.4215,-111.9342" ^
  --dest "33.4146,-111.9400"
```

---

## 🔍 Analyzing Your Recording

The screen recording from today (193552.mp4) will show us:

1. **What the camera saw** during your test
2. **What the YOLO model detected** (buildings, sidewalks, obstacles)
3. **What commands were issued** (stop, go_forward, etc.)
4. **Why it kept saying STOP** (too many obstacle pixels?)

### **Expected Issues to Find:**

**Before the fix:**
- High obstacle pixel count (buildings detected as obstacles)
- 80-90% STOP commands
- Low walkable ratio

**After the fix:**
- Lower obstacle pixel count (buildings ignored)
- More go_forward commands
- Higher walkable ratio

---

## 📁 Output Files

When using `--save-dir`, you get:

### **1. Annotated Frames**
- `frame_000000.jpg` - Every processed frame with overlays
- Useful for debugging what was detected

### **2. analysis_summary.json**
```json
{
  "video_path": "...",
  "frames_processed": 3584,
  "command_distribution": {
    "stop": 3200,
    "go_forward": 300,
    "move_left": 50
  },
  "voice_commands": 450,
  "avg_obstacle_pixels": 185000,
  "avg_walkable_ratio": 0.153,
  "command_history": [
    {
      "frame_id": 0,
      "time_sec": 0.0,
      "command": "go_forward",
      "confidence": 0.90,
      "speak": true,
      "obstacle_pixels": 12000,
      "walkable_ratio": 0.45
    },
    ...
  ]
}
```

---

## 🛠️ Troubleshooting

### **"Could not open video"**
- Check the path is correct
- Make sure the video file exists
- Try copying video to project directory first

### **"No module named cv2"**
- OpenCV not installed
- Run: `.venv\Scripts\activate.bat` then `pip install opencv-python`

### **Processing very slow**
- Normal! YOLO inference takes time
- Use `--max-frames 300` to test first 10 seconds
- Processing speed: ~2-5 FPS on CPU

### **Video window not appearing**
- Remove `--show` flag to run headless
- Use `--save-dir` to save frames instead

---

## 🎯 Comparing Before/After Fix

### **Test 1: Your Screen Recording (Before Fix)**
```batch
python process_video.py "C:\Users\chira\Videos\Screen Recordings\Screen Recording 2026-05-25 193552.mp4" --show
```

**Expected results:**
- Lots of STOP commands (80-90%)
- High obstacle_pixels
- Low walkable_ratio

### **Test 2: New Recording (After Fix)**
Record a new video, then:
```batch
python process_video.py "new_recording.mp4" --show
```

**Expected results:**
- More go_forward commands (60-80%)
- Lower obstacle_pixels
- Higher walkable_ratio

---

## 📊 Interpreting Results

### **Good Signs:**
- ✅ go_forward > 60% of frames
- ✅ walkable_ratio > 0.3 (30%)
- ✅ obstacle_pixels < 50,000 on clear paths
- ✅ STOP only when actually needed

### **Bad Signs (Broken):**
- ❌ STOP > 80% of frames
- ❌ walkable_ratio < 0.2 (20%)
- ❌ obstacle_pixels > 150,000 constantly
- ❌ STOP even on empty sidewalks

---

## 🔬 Detailed Analysis

The JSON output includes frame-by-frame data:

```python
# Example: Find all frames that said STOP
import json
with open("output/analysis/analysis_summary.json") as f:
    data = json.load(f)

stop_frames = [r for r in data["command_history"] if r["command"] == "stop"]
print(f"STOP issued {len(stop_frames)} times")
print(f"Avg obstacle pixels on STOP: {sum(r['obstacle_pixels'] for r in stop_frames) / len(stop_frames):.0f}")
```

---

## 💡 Tips

1. **Start with first 300 frames** (10 seconds) to test quickly
2. **Use `--show`** to see visual overlay in real-time
3. **Check the JSON** for detailed per-frame analysis
4. **Compare before/after** the config fix

---

## 🚀 Quick Commands Reference

```batch
# Your screen recording (analyze today's test)
analyze_recording.bat

# Manual with visualization
python process_video.py "video.mp4" --show

# Save annotated frames
python process_video.py "video.mp4" --save-dir output/analysis

# Quick test (first 300 frames)
python process_video.py "video.mp4" --max-frames 300

# With map navigation
python process_video.py "video.mp4" --use-map --current "33.42,-111.93" --dest "33.41,-111.94"
```

---

**Run `analyze_recording.bat` to analyze your screen recording now!** 🎬
