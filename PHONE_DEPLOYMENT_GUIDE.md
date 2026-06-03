# Running Assistive Navigation on Your Phone

## OK YES! You Can Run It on Your Phone

There are **3 main approaches** to run the assistive navigation system on your phone:

---

## **Recommended Approaches**

### **Option 1: Termux + Python** * (Best for Android)
Run the full Python system natively on Android using Termux.

### **Option 2: Client-Server Architecture**  (Easiest)
Phone captures video → sends to laptop → laptop processes → sends commands back

### **Option 3: Android App (Java/Kotlin)**  (Most Native)
Build a native Android app using TensorFlow Lite or ONNX

---

## On-device depth (Depth Anything V2 in the browser)

The client-server build (Option 2) now runs a **real monocular depth network on
the phone itself**, in the browser, instead of the server's geometric depth
proxy. This gives true "how far is the obstacle?" distances without needing a
hosted GPU.

**How it works**
- `phone_client.html` loads [`@huggingface/transformers`](https://huggingface.co/docs/transformers.js)
  and the `onnx-community/depth-anything-v2-small` model, running on **WebGPU**
  (falls back to WASM on browsers without WebGPU).
- On every `DEPTH_EVERY_N`th frame, the phone estimates depth, reads the nearest
  object in the center-bottom walking band, converts it to an approximate
  distance in meters, and posts it as the `depth_m` form field.
- The server feeds `depth_m` straight into `bucketize()` (the existing
  immediate/near/mid/far distance phrasing) via
  `UniDepthEstimator.predict(external_depth_m=...)`.
- If `depth_m` is missing (model still loading, WebGPU unsupported, older phone),
  the server **automatically falls back to the geometric proxy** — nothing breaks.

**Calibration** — Depth Anything outputs *relative* depth normalized per frame,
so meters are approximate. Tune these constants in the `<script type="module">`
block of `phone_client.html`:
- `DEPTH_CALIBRATION` — multiplicative scale; stand a known distance from a wall,
  open the on-screen DBG panel, read the logged `depth Xm`, and adjust so it
  matches reality.
- `DEPTH_MIN_M` / `DEPTH_MAX_M` — clamp range for the spoken distance buckets.
- `DEPTH_EVERY_N` — run depth on 1 of every N frames to bound latency.

**Requirements / caveats**
- Must be served over **HTTPS** (already required for camera access).
- **WebGPU**: Chrome/Edge on Android works well; **iOS Safari** support is newer
  (Safari 18+) and may be limited — the WASM fallback or the proxy covers it.
- First load downloads ~25-50 MB of model weights (cached afterward).
- The `/process_frame` JSON response includes `depth_source: "client" | "proxy"`
  so you can confirm which path is active.

---

# **OPTION 1: Termux + Python (Android)**

## OK **Advantages:**
- Full Python environment on Android
- No laptop needed (completely standalone)
- Use phone camera directly
- Real-time TTS output

## Warning: **Challenges:**
- PyTorch on Android (large, slow)
- Limited by phone CPU/GPU
- Complex setup

---

## **Setup Guide - Termux**

### **Step 1: Install Termux**
Download from F-Droid (NOT Google Play - outdated):
- https://f-droid.org/en/packages/com.termux/

### **Step 2: Setup Python Environment**
```bash
# Update packages
pkg update && pkg upgrade

# Install Python and dependencies
pkg install python python-pip git clang cmake

# Install OpenCV dependencies
pkg install opencv-python

# Create project directory
mkdir -p ~/assistive-navigation
cd ~/assistive-navigation
```

### **Step 3: Clone/Copy Your Project**
```bash
# Option A: Git clone (if pushed to GitHub)
git clone https://github.com/yourusername/assistive-navigation.git
cd assistive-navigation

# Option B: Copy from laptop via ADB
# On laptop: adb push C:\Users\chira\Projects\assistive-navigation /sdcard/
# On phone: cp -r /sdcard/assistive-navigation ~/
```

### **Step 4: Install Dependencies**
```bash
# Install core dependencies
pip install numpy opencv-python pydantic pydantic-settings python-dotenv pyyaml httpx

# Install PyTorch for Android (lightweight version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Ultralytics YOLO
pip install ultralytics

# Install TTS (may not work on Termux - use Android TTS API)
# pip install pyttsx3
```

### **Step 5: Configure for Phone**
Edit `.env` file:
```bash
nano .env
```

Update:
```env
CAMERA_INDEX=0  # Usually 0 or 1 for phone camera
FRAME_WIDTH=640
FRAME_HEIGHT=480
YOLO_MODEL_PATH=yolo26n-sem.pt
TTS_ENABLED=false  # Use Android TTS instead
```

### **Step 6: Run**
```bash
# Preview mode (see colored overlay)
python -m navigation.cli preview --camera 0

# Live navigation
python -m navigation.cli run --camera 0
```

---

## Warning: **Termux Challenges:**

1. **PyTorch is HUGE** (~500MB) and slow on phone CPU
2. **Camera access** may need Termux:API addon
3. **TTS** may not work - need Android TTS integration
4. **Performance** will be slower than laptop (3-5 FPS max)

---

# **OPTION 2: Client-Server (Recommended!)**

This is the **EASIEST** and **BEST PERFORMANCE** approach!

## OK **How It Works:**
1. Phone app captures video frames
2. Sends frames to laptop server via WiFi
3. Laptop processes with YOLO (fast!)
4. Sends back commands ("GO FORWARD", "STOP")
5. Phone speaks commands via TTS

## **Setup Guide**

### **Part A: Laptop Server**

I'll create a Flask/FastAPI server that accepts video frames and returns commands.

```python
# server.py - Run on laptop
from flask import Flask, request, jsonify
import cv2
import numpy as np
from navigation.perception.segmentation import YoloSegmenter
from navigation.perception.depth import UniDepthEstimator
from navigation.reasoning.care import CareNavigator
from navigation.reasoning.llm import NavigationInterpreter
from navigation.config import load_settings
from navigation.models import PerceptionBundle

app = Flask(__name__)

# Initialize navigation components once
settings = load_settings()
segmenter = YoloSegmenter(settings)
depth_est = UniDepthEstimator(settings)
care = CareNavigator(settings)
interpreter = NavigationInterpreter(settings)

frame_id = 0

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global frame_id
    
    # Receive image from phone
    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Process frame
    seg = segmenter.predict(frame, dry_run=False)
    depth = depth_est.predict(frame, dry_run=True)
    care_out = care.predict(frame, seg, depth, dry_run=False)
    
    bundle = PerceptionBundle(
        frame_id=frame_id,
        segmentation=seg,
        depth=depth,
        care=care_out,
    )
    
    decision = interpreter.interpret(bundle, dry_run=False)
    
    frame_id += 1
    
    # Return command
    return jsonify({
        'command': decision.command.value,
        'confidence': decision.confidence,
        'phrase': decision.command.value.replace('_', ' '),
        'speak': decision.speak
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
```

**Run server on laptop:**
```batch
cd C:\Users\chira\Projects\assistive-navigation
.venv\Scripts\activate.bat
pip install flask
python server.py
```

**Find laptop IP address:**
```batch
ipconfig
# Look for IPv4 Address (e.g., 192.168.1.100)
```

---

### **Part B: Phone App (Android)**

**Option B1: Simple HTML5 Web App** (No installation needed!)

Create `phone_client.html`:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Assistive Navigation</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial; text-align: center; padding: 20px; }
        #video { width: 100%; max-width: 640px; }
        #status { font-size: 24px; margin: 20px; padding: 10px; }
        .go { background: blue; color: white; }
        .stop { background: red; color: white; }
        button { font-size: 20px; padding: 15px 30px; margin: 10px; }
    </style>
</head>
<body>
    <h1>Assistive Navigation</h1>
    <video id="video" autoplay playsinline></video>
    <div id="status">Ready</div>
    <button onclick="startNavigation()">Start Navigation</button>
    <button onclick="stopNavigation()">Stop</button>
    
    <script>
        const SERVER_URL = 'http://192.168.1.100:5000';  // UPDATE WITH YOUR LAPTOP IP!
        let video = document.getElementById('video');
        let statusDiv = document.getElementById('status');
        let isRunning = false;
        let synth = window.speechSynthesis;
        
        // Access camera
        navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment', width: 640, height: 480 } 
        }).then(stream => {
            video.srcObject = stream;
        });
        
        function startNavigation() {
            isRunning = true;
            processFrame();
        }
        
        function stopNavigation() {
            isRunning = false;
            statusDiv.textContent = 'Stopped';
        }
        
        async function processFrame() {
            if (!isRunning) return;
            
            // Capture frame from video
            let canvas = document.createElement('canvas');
            canvas.width = 640;
            canvas.height = 480;
            canvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
            
            // Convert to blob
            canvas.toBlob(async (blob) => {
                let formData = new FormData();
                formData.append('frame', blob, 'frame.jpg');
                
                try {
                    let response = await fetch(SERVER_URL + '/process_frame', {
                        method: 'POST',
                        body: formData
                    });
                    
                    let result = await response.json();
                    
                    // Update UI
                    statusDiv.textContent = result.phrase.toUpperCase();
                    statusDiv.className = result.command === 'stop' ? 'stop' : 'go';
                    
                    // Speak command
                    if (result.speak) {
                        let utterance = new SpeechSynthesisUtterance(result.phrase);
                        synth.speak(utterance);
                    }
                    
                } catch (error) {
                    statusDiv.textContent = 'Error: ' + error.message;
                }
                
                // Process next frame after 300ms
                setTimeout(processFrame, 300);
            }, 'image/jpeg', 0.8);
        }
    </script>
</body>
</html>
```

**To use:**
1. Update `SERVER_URL` with your laptop's IP address
2. Save file on laptop: `C:\Users\chira\Projects\assistive-navigation\phone_client.html`
3. Make sure laptop server is running
4. On phone, open browser and go to: `file:///sdcard/phone_client.html` or serve it via laptop

---

**Option B2: Native Android App (Flutter/React Native)**

Would you like me to create a full Android app? This requires more setup but provides better performance.

---

# **OPTION 3: TensorFlow Lite / ONNX (Native Android)**

Convert YOLO model to TensorFlow Lite or ONNX format for mobile deployment.

## **Steps:**

1. **Export YOLO to TFLite:**
```python
from ultralytics import YOLO

model = YOLO('yolo26n-sem.pt')
model.export(format='tflite')  # Creates yolo26n-sem.tflite
```

2. **Build Android App with TFLite**
3. **Use Android Camera2 API**
4. **Run inference on-device**

This is most complex but best performance.

---

# **Recommendation: Client-Server (Option 2)**

**Best approach for you:**

OK **Keep laptop processing** (fast YOLO inference)  
OK **Phone as camera + display** (simple)  
OK **WiFi connection** (no cables)  
OK **Easy to implement** (HTML5 web app)  
OK **Good performance** (~3-5 FPS)

---

# **Next Steps - Which Option?**

1. **Quick test:** Option 2 (Client-Server) - I'll create the server now
2. **Full Android:** Option 1 (Termux) - More setup, standalone
3. **Native app:** Option 3 (TFLite) - Most work, best performance

**Which approach would you like me to implement?**

I can create:
- OK Flask server for laptop (5 minutes)
- OK HTML5 phone client (5 minutes)
- OK Native Android app (1-2 hours)
- OK Termux setup script (30 minutes)

**Let me know and I'll build it!** 
