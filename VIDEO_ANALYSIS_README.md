# 🎥 YES! Process Your Screen Recording

## ✅ One Command to Analyze Your Video

```batch
C:\Users\chira\Projects\assistive-navigation\ANALYZE_NOW.bat
```

**Double-click `ANALYZE_NOW.bat` in File Explorer!**

---

## 📹 What This Does

Your screen recording from today:
- **File:** `C:\Users\chira\Videos\Screen Recordings\Screen Recording 2026-05-25 193552.mp4`
- **Size:** 359 MB
- **Recorded:** Today at 7:35 PM

The script will:
1. ✅ Process every frame through the navigation system
2. ✅ Show what YOLO detected (buildings, sidewalks, obstacles)
3. ✅ Display what commands were issued (STOP, go_forward, etc.)
4. ✅ Save annotated frames to `output/video_analysis/`
5. ✅ Generate detailed statistics in `analysis_summary.json`

---

## 📊 What You'll See

### **While Processing:**
```
Frame      0 (   0.0s): GO_FORWARD   [VOICE]   conf=0.90
Frame      3 (   0.1s): GO_FORWARD   [silent]  conf=0.88
Frame      6 (   0.2s): STOP         [VOICE]   conf=0.85
Frame      9 (   0.3s): STOP         [silent]  conf=0.85
Frame     12 (   0.4s): STOP         [silent]  conf=0.85
...
```

### **Visualization Window:**
- **Green areas** = Walkable (sidewalk, road)
- **Red/Orange** = Obstacles (people, cars, buildings)
- **HUD overlay** shows current command
- **Press 'q' to stop early**

### **Final Summary:**
```
Command Distribution:
  stop           : 3200 (89.3%)  ← If this is high, bug confirmed!
  go_forward     :  300 ( 8.4%)
  move_left      :   50 ( 1.4%)

Voice commands spoken: 450
Avg obstacle pixels: 185000        ← High = buildings detected
Avg walkable ratio: 15.3%          ← Low = not enough walkable area
```

---

## 🔍 What to Look For

### **Before Fix (Your Recording):**
- ❌ **STOP dominates** (80-90% of commands)
- ❌ **High obstacle pixels** (>150,000)
- ❌ **Low walkable ratio** (<20%)
- ❌ **Buildings shown as red/orange** (detected as obstacles)

### **After Fix (Expected):**
- ✅ **go_forward dominates** (60-80% of commands)
- ✅ **Lower obstacle pixels** (<50,000)
- ✅ **Higher walkable ratio** (>30%)
- ✅ **Buildings shown as gray** (ignored)

---

## 📁 Output Files

After processing, check:

### **1. Video Analysis Folder**
`C:\Users\chira\Projects\assistive-navigation\output\video_analysis\`

### **2. Annotated Frames**
- `frame_000000.jpg` - First frame with overlays
- `frame_000003.jpg` - Every 3rd frame
- Shows exactly what YOLO detected

### **3. Analysis Summary JSON**
`output\video_analysis\analysis_summary.json`

Contains:
- Frame-by-frame command history
- Per-frame obstacle pixel counts
- Walkable ratio statistics
- Command distribution

Open with:
```batch
notepad output\video_analysis\analysis_summary.json
```

---

## 🚀 Alternative Commands

### **Quick Test (First 10 Seconds Only):**
```batch
cd C:\Users\chira\Projects\assistive-navigation
.venv\Scripts\activate.bat
python process_video.py "C:\Users\chira\Videos\Screen Recordings\Screen Recording 2026-05-25 193552.mp4" --show --max-frames 300
```

### **No Visualization (Faster):**
```batch
python process_video.py "C:\Users\chira\Videos\Screen Recordings\Screen Recording 2026-05-25 193552.mp4" --save-dir output\video_analysis
```

### **Process Any Video:**
```batch
python process_video.py "path\to\your\video.mp4" --show
```

---

## 🎯 Diagnosing the Voice Issue

The video analysis will reveal:

### **Why No Voice Instructions?**

**Possible causes:**
1. **TTS not enabled** - Check `.env`: `TTS_ENABLED=true`
2. **Volume muted** - Check Windows volume mixer
3. **Always on cooldown** - Commands too frequent (fixed with COMMAND_COOLDOWN_SEC=1.5)
4. **Same command repeated** - Suppression working as designed (only speaks on change)

**The JSON will show:**
```json
{
  "frame_id": 0,
  "speak": true,     ← Should be true for first command
  "phrase": "Stop"   ← What it tried to say
}
```

If `speak: false` on most frames → cooldown/suppression working  
If `speak: true` but no audio → TTS issue

---

## 🛠️ Troubleshooting

### **"Could not open video"**
- Check file path in `ANALYZE_NOW.bat`
- Verify file exists: `dir "C:\Users\chira\Videos\Screen Recordings\"`

### **Script runs but nothing happens**
- Wait! Processing is slow (~2-5 FPS)
- Check terminal for progress
- Press 'q' in window to stop early

### **Out of memory**
- Process fewer frames: `--max-frames 300`
- Close other applications
- Video is 6 minutes long, that's ~10,000 frames!

### **Want faster processing**
- Use `--max-frames 300` (first 10 seconds)
- Remove `--show` (no visualization window)
- Process every 10th frame only (edit code)

---

## 📊 Expected Processing Time

| Frames | Duration | Time to Process |
|--------|----------|-----------------|
| 300    | 10 sec   | ~2 minutes      |
| 900    | 30 sec   | ~6 minutes      |
| 3000   | 100 sec  | ~20 minutes     |
| 10752  | 358 sec  | ~60 minutes     |

**Recommendation:** Start with `--max-frames 300` to test quickly!

---

## 🎬 Step-by-Step Guide

### **Step 1: Run Analysis**
```batch
ANALYZE_NOW.bat
```

### **Step 2: Watch the Window**
- See what colors are detected
- Check if buildings are red (bad) or gray (good)
- Watch command distribution

### **Step 3: Check JSON Output**
```batch
notepad output\video_analysis\analysis_summary.json
```

Look for:
- `"command": "stop"` count vs `"command": "go_forward"`
- `"obstacle_pixels"` values
- `"walkable_ratio"` values
- `"speak": true` count (voice commands)

### **Step 4: Compare Results**

**If STOP > 80%:**
- Bug confirmed - buildings detected as obstacles
- Our fix should resolve this

**If go_forward > 60%:**
- System working correctly
- Voice issue is separate (TTS problem)

---

## 💡 Pro Tips

1. **Test with first 300 frames** to save time
2. **Keep the visualization window open** to see detections
3. **Check the JSON** for detailed stats
4. **Save annotated frames** to review later
5. **Compare before/after** fix by recording again

---

## 🚀 Quick Commands Summary

```batch
# ONE-CLICK ANALYSIS
ANALYZE_NOW.bat

# QUICK TEST (10 seconds)
python process_video.py "C:\Users\chira\Videos\Screen Recordings\Screen Recording 2026-05-25 193552.mp4" --show --max-frames 300

# FULL ANALYSIS
python process_video.py "C:\Users\chira\Videos\Screen Recordings\Screen Recording 2026-05-25 193552.mp4" --show --save-dir output\video_analysis

# HEADLESS (NO WINDOW)
python process_video.py "C:\Users\chira\Videos\Screen Recordings\Screen Recording 2026-05-25 193552.mp4" --save-dir output\video_analysis
```

---

## 📝 Next Steps

1. **Run ANALYZE_NOW.bat**
2. **Wait for processing** (2-60 minutes depending on frames)
3. **Check output/video_analysis/analysis_summary.json**
4. **Share the results** with me (command distribution, avg obstacle pixels)

Then we can:
- ✅ Confirm the bug (if STOP > 80%)
- ✅ Verify the fix worked
- ✅ Debug voice issue (if separate from obstacle detection)

---

**Double-click `ANALYZE_NOW.bat` to start!** 🎬
