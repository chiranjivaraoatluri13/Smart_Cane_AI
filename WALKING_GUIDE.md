# Real-World Walking Navigation Guide

**Project:** Assistive Navigation System  
**Purpose:** Guide for using the system while walking outdoors or indoors

---

## Quick Start for Walking

### **Easiest Way: Use the Launcher**

1. Open Windows Explorer
2. Navigate to: `C:\Users\chira\Projects\assistive-navigation`
3. Double-click: `walking_mode.bat`
4. Choose mode from menu

---

## Equipment Setup

### **Minimum Setup:**
- Laptop with webcam
- Headphones/earbuds
- Charged battery

### **Recommended Setup:**
- Laptop in backpack with camera facing forward
- Wireless earbuds for mobility
- Portable power bank
- Stable mount/rig for laptop

### **Optimal Setup:**
- Phone as camera (connected to laptop via USB)
- Laptop in backpack processing
- Bluetooth earbuds
- Phone mounted at chest level

---

## Walking Modes

### **Mode 1: Basic Walking** (Start Here)

```bash
cd C:\Users\chira\Projects\assistive-navigation
.venv\Scripts\activate
assistive-nav run --fast --camera 0
```

**Features:**
- Real-time obstacle detection
- Voice commands every 1 second (configurable)
- Commands: "Stop", "Go forward", "Move left", "Move right", "Slow down"
- 12+ FPS on CPU

**Voice Commands You'll Hear:**
- **"Stop"** - Obstacle detected (person, car, wall)
- **"Go forward"** - Path is clear
- **"Move left"** - Turn left to avoid obstacle
- **"Move right"** - Turn right to avoid obstacle
- **"Slow down"** - Caution, reduced safety

**Best For:**
- Indoor corridors
- Campus walkways
- Controlled environments
- Testing the system

---

### **Mode 2: Demo Mode** (Optimized for Real Use)

```bash
assistive-nav run --demo --camera 0
```

**Features:**
- Faster response (0.8s cooldown)
- More frequent updates
- Optimized for outdoor use
- Better for walking speed

**Differences from Basic:**
- Commands update faster
- More sensitive to changes
- Better for dynamic environments

**Best For:**
- Outdoor walking
- Busy areas
- Real-world navigation
- Live demonstrations

---

### **Mode 3: Map Navigation** (Turn-by-Turn)

```bash
assistive-nav run --fast --use-map --current "40.7484,-73.9857" --dest "40.7510,-73.9830" --camera 0
```

**Features:**
- Walking route from OSRM
- Turn-by-turn directions
- Obstacle detection overrides route
- Distance and waypoint tracking

**Voice Commands You'll Hear:**
- **"Go forward"** - On route, continue
- **"Move left"** - Turn left (route or obstacle)
- **"Move right"** - Turn right (route or obstacle)
- **"Stop"** - Obstacle ahead OR arrived at destination

**How to Get Coordinates:**

1. **Current Position:**
   - Open Google Maps on phone
   - Tap current location (blue dot)
   - Coordinates appear at top
   - Format: `40.7484,-73.9857`

2. **Destination:**
   - Search destination in Google Maps
   - Long-press the location
   - Copy coordinates
   - Format: `40.7510,-73.9830`

3. **OR Use Address:**
   ```bash
   assistive-nav run --fast --use-map \
     --current "40.7484,-73.9857" \
     --dest-address "Central Park, New York" \
     --camera 0
   ```

**Best For:**
- Navigating to specific location
- Longer walks
- Unfamiliar areas
- Structured routes

---

### **Mode 4: Preview Only** (Testing Camera)

```bash
assistive-nav preview --fast --camera 0
```

**Features:**
- Shows segmentation overlay only
- No voice commands
- Faster (less processing)
- Good for testing setup

**Best For:**
- Testing camera angle
- Checking lighting
- Verifying model works
- Setup validation

---

## How to Use While Walking

### **Setup Process:**

1. **Before You Start:**
   ```bash
   cd C:\Users\chira\Projects\assistive-navigation
   .venv\Scripts\activate
   ```

2. **Test Camera First:**
   ```bash
   assistive-nav preview --fast --camera 0
   ```
   - Make sure camera sees the path ahead
   - Adjust angle if needed
   - Press `q` to close

3. **Start Walking Mode:**
   ```bash
   assistive-nav run --demo --camera 0
   ```

4. **Put on Headphones**

5. **Start Walking Slowly**

---

### **Walking Technique:**

1. **Camera Position:**
   - Point camera straight ahead
   - Chest-level height works best
   - Keep camera stable
   - Avoid shaking too much

2. **Walking Speed:**
   - Start SLOW (1-2 mph)
   - System processes ~12 frames/second
   - Faster walking = less reaction time
   - Increase speed as comfortable

3. **Listen to Commands:**
   - **"Stop"** → STOP immediately
   - **"Go forward"** → Continue walking
   - **"Move left/right"** → Adjust direction
   - **"Slow down"** → Reduce speed

4. **Safety First:**
   - System is ASSISTIVE, not autonomous
   - You are still responsible
   - Use your own judgment
   - Don't rely 100% on system

---

## Configuration for Walking

### **Edit .env for Walking:**

```bash
# Open in notepad
notepad .env
```

**Recommended Settings for Walking:**

```env
# Camera
CAMERA_INDEX=0
FRAME_WIDTH=320
FRAME_HEIGHT=240
TARGET_FPS=24

# Performance (CPU)
PROCESS_EVERY_N_FRAMES=2
INFERENCE_WIDTH=256
INFERENCE_HEIGHT=192
YOLO_IMGSZ=256

# Commands (adjust for walking speed)
COMMAND_COOLDOWN_SEC=0.8
REPEAT_COMMAND_SUPPRESS=true
HAZARD_OBSTACLE_RATIO=0.03

# Voice
TTS_ENABLED=true
TTS_RATE=175

# Map (optional)
USE_MAP_GUIDANCE=false
CURRENT_LAT=
CURRENT_LON=
DEST_LAT=
DEST_LON=
```

**Adjustments:**

- **Slower walking:** `COMMAND_COOLDOWN_SEC=1.5`
- **Faster updates:** `COMMAND_COOLDOWN_SEC=0.5`
- **More sensitive:** `HAZARD_OBSTACLE_RATIO=0.02`
- **Less sensitive:** `HAZARD_OBSTACLE_RATIO=0.05`

---

## Map Example Walking Routes

### **Example 1: Campus Walk**

```bash
# NYU Washington Square to Library
assistive-nav run --demo --use-map \
  --current "40.7308,-73.9973" \
  --dest "40.7295,-73.9965" \
  --camera 0
```

### **Example 2: City Block**

```bash
# Times Square to Bryant Park
assistive-nav run --demo --use-map \
  --current "40.7580,-73.9855" \
  --dest "40.7536,-73.9832" \
  --camera 0
```

### **Example 3: Indoor Navigation**

```bash
# Building corridor (no map needed)
assistive-nav run --demo --camera 0
```

---

## What the System Detects

### **Cityscapes Classes (19 total):**

**Walkable Surfaces:**
- Road
- Sidewalk
- Terrain

**Obstacles:**
- Person
- Rider (cyclist)
- Car, Truck, Bus, Train
- Motorcycle, Bicycle
- Building, Wall, Fence
- Pole
- Traffic light, Traffic sign
- Vegetation

**Neutral:**
- Sky

---

## Voice Command Meanings

| Command | Meaning | Action |
|---------|---------|--------|
| **Stop** | Obstacle ahead OR hazard detected | Stop immediately |
| **Go forward** | Path is clear | Continue walking |
| **Move left** | Obstacle on right OR turn left (map) | Shift left |
| **Move right** | Obstacle on left OR turn right (map) | Shift right |
| **Slow down** | Reduced safety score | Walk slower |

---

## Troubleshooting While Walking

### **Camera Issues:**

**Problem:** Can't see overlay/camera not opening
```bash
# List cameras
python scripts/list_cameras.py

# Try different index
assistive-nav run --demo --camera 1
```

**Problem:** Camera shaking too much
- Use chest mount or stable rig
- Walk slower
- Increase `PROCESS_EVERY_N_FRAMES=5`

### **Performance Issues:**

**Problem:** Too slow / laggy
```bash
# Lower resolution
# Edit .env:
INFERENCE_WIDTH=192
INFERENCE_HEIGHT=144
PROCESS_EVERY_N_FRAMES=5
```

**Problem:** Commands too frequent
```bash
# Increase cooldown
# Edit .env:
COMMAND_COOLDOWN_SEC=2.0
```

**Problem:** Not enough warnings
```bash
# More sensitive
# Edit .env:
HAZARD_OBSTACLE_RATIO=0.02
```

### **Voice Issues:**

**Problem:** No voice output
```bash
# Check TTS in .env:
TTS_ENABLED=true

# Test TTS
python -c "
from navigation.output.tts import SpeechEngine
from navigation.config import load_settings
tts = SpeechEngine(load_settings())
tts.warmup()
print('TTS working')
"
```

**Problem:** Voice too fast/slow
```bash
# Edit .env:
TTS_RATE=150  # Slower
TTS_RATE=200  # Faster
```

---

## GPS Integration (Future)

Currently, the system uses **fixed coordinates** because laptops don't have GPS.

### **To Use Real GPS:**

1. **Option A: Phone GPS**
   - Use phone as camera via USB
   - Enable location services
   - Stream GPS to laptop (requires additional setup)

2. **Option B: External GPS**
   - USB GPS receiver
   - Connect to laptop
   - Update code to read GPS stream

3. **Option C: Phone App Integration**
   - Build companion phone app
   - Send GPS coordinates to laptop
   - System uses real-time position

---

## Safety Guidelines

### **IMPORTANT: Read Before Walking**

1. **This is ASSISTIVE technology, NOT autonomous**
   - You are responsible for your safety
   - System helps, doesn't replace judgment
   - Use common sense

2. **Start in Safe Environments:**
   - Indoor corridors first
   - Empty parking lots
   - Quiet parks
   - Avoid busy streets initially

3. **Walking Safety:**
   - Start VERY slow
   - Test in daylight first
   - Have a sighted companion initially
   - Don't wear noise-canceling headphones (hear traffic)

4. **System Limitations:**
   - CPU processing: ~12 FPS (80ms delay)
   - No depth perception (mock depth only)
   - Lighting affects detection
   - Fast-moving objects may not be detected in time

5. **Do NOT Use For:**
   - Crossing busy streets alone
   - Night walking (poor lighting)
   - High-speed situations
   - Critical navigation without backup

---

## Session Logging

### **Save Your Walking Session:**

```bash
# Create timestamped session folder
assistive-nav run --demo --camera 0 \
  --seg-save-dir "output/walk_$(date +%Y%m%d_%H%M%S)" \
  --max-frames 1000
```

This saves:
- All segmentation frames
- Command history
- Performance metrics

---

## Quick Launch (Copy & Paste)

### **Command Prompt Quick Launch:**

```batch
cd C:\Users\chira\Projects\assistive-navigation && .venv\Scripts\activate && assistive-nav run --demo --camera 0
```

### **OR Use the Launcher:**

```batch
cd C:\Users\chira\Projects\assistive-navigation
walking_mode.bat
```

Then select mode from menu.

---

## Expected Performance

### **CPU Mode (Current):**
- **FPS:** 12-15 frames/second
- **Latency:** ~80ms per frame
- **Range:** ~5-10 meters detection
- **Walking Speed:** Slow (1-2 mph recommended)

### **GPU Mode (If You Add GPU):**
- **FPS:** 30+ frames/second
- **Latency:** ~30ms per frame
- **Range:** Same
- **Walking Speed:** Normal (3-4 mph possible)

---

## Example Session

```bash
# 1. Navigate to project
cd C:\Users\chira\Projects\assistive-navigation

# 2. Activate environment
.venv\Scripts\activate

# 3. Start demo mode
assistive-nav run --demo --camera 0

# Expected output:
[TTS] Ready — voice: Microsoft David Desktop
[VOICE] Go forward
>>> GO_FORWARD  (go_forward, 89%)
...

# 4. Put on headphones

# 5. Start walking slowly

# 6. Listen to commands and follow them

# 7. Press Ctrl+C when done
```

---

## Support

If you encounter issues:

1. Check `COMPLETE_FLOW_REPORT.md` for troubleshooting
2. Check `RUN_GUIDE.md` for detailed commands
3. Test with static image first: `assistive-nav run --image tests\fixtures\sample.jpg`
4. Verify camera works: `assistive-nav preview --camera 0`

---

**Stay Safe and Enjoy Your Walk!** 

---

**Project:** `C:\Users\chira\Projects\assistive-navigation`  
**Launcher:** Double-click `walking_mode.bat`  
**Documentation:** `RUN_GUIDE.md`, `COMPLETE_FLOW_REPORT.md`
