# 📋 Render Deployment Checklist

## Pre-Deployment ✅

- [x] Code migrated from YOLO to ADE20K SegFormer
- [x] All 307 tests passing
- [x] Phone server configured for cloud (`phone_server_cloud.py`)
- [x] Requirements file updated (`requirements-cloud.txt`)
- [x] Render manifest created (`render.yaml`)
- [x] All code pushed to GitHub
- [x] Deployment documentation complete

---

## Render Setup Steps

### Step 1: Create Render Account
- [ ] Go to https://render.com
- [ ] Sign up with GitHub
- [ ] Authorize GitHub access

### Step 2: Create Web Service
- [ ] Click **New +** → **Web Service**
- [ ] Select repository: `Smart_Cane-AI`
- [ ] Click **Connect**

### Step 3: Configure Service
- [ ] **Name:** `smart-cane-ai`
- [ ] **Environment:** Python 3.11
- [ ] **Build Command:** `pip install -r requirements-cloud.txt`
- [ ] **Start Command:** `gunicorn phone_server_cloud:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
- [ ] **Instance Type:** Standard (2GB RAM)

### Step 4: Set Environment Variables
Copy these into Render dashboard:

```
PYTHON_VERSION=3.11.9
SEGMENTER_BACKEND=segformer_onnx
SEGFORMER_MODEL_ID=nvidia/segformer-b0-finetuned-ade-512-512
SEGFORMER_DEVICE=cpu
SEGFORMER_ONNX_PATH=segformer_b0_ade20k_int8.onnx
TTS_ENABLED=false
USE_LLM=false
USE_CARE_HTTP=false
COMMAND_COOLDOWN_SEC=8.0
COMMAND_DWELL_FRAMES=4
MIN_SPEECH_GAP_SEC=3.0
STOP_HOLD_FRAMES=8
HAZARD_OBSTACLE_RATIO=0.10
ALERTS_ENABLED=true
ALERT_MIN_WEIGHTED_PIXELS=1500.0
ALERT_MAX_SIMULTANEOUS_CATEGORIES=2
USE_MAP_GUIDANCE=true
```

- [ ] All variables entered
- [ ] No typos in values

### Step 5: Deploy
- [ ] Click **Create Web Service**
- [ ] Wait for build to complete (5-10 minutes)
- [ ] Check logs for errors
- [ ] Verify service is running

---

## Post-Deployment Verification

### Health Check
```bash
curl https://smart-cane-ai.onrender.com/health
```
- [ ] Returns `{"status": "ok", "models_loaded": true}`

### Debug Route
```bash
curl https://smart-cane-ai.onrender.com/debug_route
```
- [ ] Returns route configuration

### Phone Client
- [ ] Open https://smart-cane-ai.onrender.com on phone
- [ ] Page loads without errors
- [ ] Camera permission prompt appears
- [ ] Location permission prompt appears

---

## Phone Testing

### Initial Setup
- [ ] Allow camera access
- [ ] Allow location (GPS) access
- [ ] Allow microphone access

### Set Destination
- [ ] Click "Set Destination"
- [ ] Enter test address (e.g., "Times Square, New York")
- [ ] Click "Geocode"
- [ ] Verify address is geocoded

### Start Navigation
- [ ] Click "Start Camera"
- [ ] Point camera at path
- [ ] Wait for first frame processing
- [ ] Listen for voice guidance

### Expected Behavior
- [ ] First frame processes in 1-2 seconds
- [ ] Voice says "Go forward" or "Stop"
- [ ] JSON response shows command and confidence
- [ ] Subsequent frames process faster

---

## Performance Monitoring

### Check Processing Speed
- [ ] Open browser console (F12)
- [ ] Look for `processing_time_ms` in responses
- [ ] Should be 800-1200ms for Standard instance

### Monitor Logs
- [ ] Go to Render dashboard
- [ ] Click service name
- [ ] Go to **Logs** tab
- [ ] Watch real-time processing logs

### Check Resource Usage
- [ ] Render dashboard → **Metrics** tab
- [ ] Monitor CPU usage (should be <80%)
- [ ] Monitor memory usage (should be <1.5GB)

---

## Troubleshooting

### Build Fails
- [ ] Check Render logs for specific error
- [ ] Verify `requirements-cloud.txt` is correct
- [ ] Try Standard instance (free tier may timeout)
- [ ] Check Python version (should be 3.11)

### Service Won't Start
- [ ] Check start command in Render dashboard
- [ ] Verify all environment variables are set
- [ ] Check logs for import errors
- [ ] Ensure `phone_server_cloud.py` exists in repo

### Slow Processing (>2s per frame)
- [ ] Check instance type (Standard minimum)
- [ ] Reduce `INFERENCE_IMGSZ` to 256
- [ ] Disable alerts: `ALERTS_ENABLED=false`
- [ ] Upgrade to Pro instance

### GPS Not Working
- [ ] Verify phone has GPS enabled
- [ ] Check browser location permissions
- [ ] Confirm `USE_MAP_GUIDANCE=true`
- [ ] Test `/debug_route` endpoint

### No Voice Output
- [ ] Check phone microphone permissions
- [ ] Verify browser supports Web Speech API (Chrome/Edge)
- [ ] Check browser console for errors
- [ ] Ensure `speak: true` in JSON response

---

## Optimization Tips

### For Better Performance
1. Use Standard instance ($7/month)
2. Keep `INFERENCE_IMGSZ=0` (model default)
3. Enable alerts for safety
4. Monitor logs regularly

### For Lower Cost
1. Start with free tier for testing
2. Upgrade to Standard for production
3. Disable alerts if not needed
4. Reduce `COMMAND_COOLDOWN_SEC` to 5.0

### For Better Accuracy
1. Ensure good lighting
2. Point camera at path ahead
3. Keep phone steady
4. Allow GPS to stabilize (first 30 seconds)

---

## Maintenance

### Weekly
- [ ] Check Render logs for errors
- [ ] Monitor processing time trends
- [ ] Verify health endpoint

### Monthly
- [ ] Review performance metrics
- [ ] Check for any error patterns
- [ ] Update environment variables if needed

### As Needed
- [ ] Restart service if issues occur
- [ ] Upgrade instance if performance degrades
- [ ] Pull latest code from GitHub

---

## Rollback Plan

If deployment has issues:

1. **Immediate:** Disable service in Render dashboard
2. **Check:** Review logs for root cause
3. **Fix:** Update code or environment variables
4. **Redeploy:** Push changes to GitHub, Render auto-deploys
5. **Verify:** Test health endpoint and phone client

---

## Success Criteria

✅ **Deployment Successful When:**
- [ ] Service is running on Render
- [ ] Health endpoint returns 200 OK
- [ ] Phone client loads without errors
- [ ] Camera and GPS permissions work
- [ ] First frame processes in <2 seconds
- [ ] Voice guidance is heard
- [ ] Navigation commands are accurate

---

## Support Resources

- **Render Docs:** https://render.com/docs
- **Project Repo:** https://github.com/chiranjivaraoatluri13/Smart_Cane-AI
- **Deployment Guide:** `RENDER_DEPLOYMENT.md`
- **Phone Guide:** `PHONE_DEPLOYMENT_GUIDE.md`

---

## Final Notes

- **Estimated Deployment Time:** 15-20 minutes
- **Estimated Monthly Cost:** $7 (Standard instance)
- **Expected Performance:** 0.8-1.2 fps on Standard
- **Support:** Check logs and debug endpoints first

---

**Status:** Ready for deployment ✅  
**Last Updated:** May 31, 2026  
**Target:** Render (phone-only)
