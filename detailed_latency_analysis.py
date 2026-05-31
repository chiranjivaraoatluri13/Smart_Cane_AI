"""
Detailed latency analysis - measure each component's execution time.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import time

_project_root = Path(__file__).parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from navigation.config import load_settings
from navigation.perception.segmentation_base import build_segmenter
from navigation.perception.depth import UniDepthEstimator
from navigation.reasoning.care import CareNavigator
from navigation.reasoning.spatial_reasoner import SpatialReasoner
from navigation.reasoning.composer import PhraseComposer
from navigation.reasoning.alerts import AlertTracker
from navigation.output.validator import CommandValidator
from navigation.output.voice_queue import VoiceQueue
from navigation.reasoning.trend import TrendTracker
from navigation.reasoning.llm import NavigationInterpreter
from navigation.output.tts import SpeechEngine
from navigation.models import PerceptionBundle, Position


def detailed_latency_analysis(video_path, max_frames=50):
    """Analyze latency of each component."""
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return None
    
    print(f"\n{'='*80}")
    print(f"⏱️  DETAILED LATENCY ANALYSIS: {video_path.name}")
    print(f"{'='*80}")
    
    # Load models
    print("Loading models...")
    settings = load_settings()
    
    # Enable benchmark mode
    settings.benchmark_mode = True
    
    segmenter = build_segmenter(settings)
    depth_est = UniDepthEstimator(settings)
    care = CareNavigator(settings)
    spatial_reasoner = SpatialReasoner(settings)
    composer = PhraseComposer(settings)
    alert_tracker = AlertTracker.from_settings(settings)
    validator = CommandValidator(settings)
    tts = SpeechEngine(settings)
    voice_queue = VoiceQueue(settings)
    trend_tracker = TrendTracker(settings)
    interpreter = NavigationInterpreter(settings)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ Video: {width}x{height} @ {fps:.1f}fps, {total_frames} total frames")
    print(f"✓ Analyzing first {min(max_frames, total_frames)} frames with timing...\n")
    
    # Timing data
    timings = defaultdict(list)
    frame_idx = 0
    
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        try:
            # Segmentation
            t0 = time.perf_counter()
            seg = segmenter.predict(frame)
            seg_time = (time.perf_counter() - t0) * 1000
            timings["segmentation"].append(seg_time)
            
            # Depth
            t0 = time.perf_counter()
            depth = depth_est.predict(frame, segmentation=seg, external_depth_m=None)
            depth_time = (time.perf_counter() - t0) * 1000
            timings["depth"].append(depth_time)
            
            # CARE
            t0 = time.perf_counter()
            care_out = care.predict(frame, seg, depth)
            care_time = (time.perf_counter() - t0) * 1000
            timings["care"].append(care_time)
            
            # Trend tracking
            t0 = time.perf_counter()
            if trend_tracker is not None:
                trend_tracker.update(seg.per_side_class_pixels)
                approach_by_category = trend_tracker.classify_all()
            else:
                approach_by_category = {}
            trend_time = (time.perf_counter() - t0) * 1000
            timings["trend"].append(trend_time)
            
            # Spatial reasoner (no stairs — SegFormer handles that)
            from navigation.reasoning.facts import StairsResult
            stairs = StairsResult(False, 0.0, "")
            
            t0 = time.perf_counter()
            decision, facts = spatial_reasoner.decide(
                seg, depth, care_out, None,
                stairs=stairs,
                approach_by_category=approach_by_category,
            )
            reasoner_time = (time.perf_counter() - t0) * 1000
            timings["reasoner"].append(reasoner_time)
            
            # Validator
            t0 = time.perf_counter()
            decision = validator.approve(decision)
            validator_time = (time.perf_counter() - t0) * 1000
            timings["validator"].append(validator_time)
            
            # Composer
            t0 = time.perf_counter()
            phrase = composer.compose(facts) if composer is not None else None
            composer_time = (time.perf_counter() - t0) * 1000
            timings["composer"].append(composer_time)
            
            # Alerts
            t0 = time.perf_counter()
            spoken_alerts = []
            if alert_tracker is not None:
                for alert in alert_tracker.update(seg):
                    if validator.approve_alert(alert):
                        spoken_alerts.append(alert)
            alerts_time = (time.perf_counter() - t0) * 1000
            timings["alerts"].append(alerts_time)
            
            # Total
            total_time = seg_time + depth_time + care_time + trend_time + reasoner_time + validator_time + composer_time + alerts_time
            timings["total"].append(total_time)
            
            if frame_idx % 10 == 0:
                print(f"Frame {frame_idx}: {total_time:.1f}ms total")
                print(f"  Seg: {seg_time:.1f}ms | Depth: {depth_time:.1f}ms | CARE: {care_time:.1f}ms")
                print(f"  Trend: {trend_time:.1f}ms | Reasoner: {reasoner_time:.1f}ms | Validator: {validator_time:.1f}ms")
                print(f"  Composer: {composer_time:.1f}ms | Alerts: {alerts_time:.1f}ms")
            
        except Exception as e:
            print(f"❌ Error frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    cap.release()
    
    # Generate report
    print(f"\n{'='*80}")
    print("📊 LATENCY BREAKDOWN")
    print(f"{'='*80}\n")
    
    components = [
        "segmentation",
        "depth",
        "care",
        "trend",
        "reasoner",
        "validator",
        "composer",
        "alerts",
        "total",
    ]
    
    for component in components:
        times = timings[component]
        if times:
            mean = np.mean(times)
            median = np.median(times)
            min_t = np.min(times)
            max_t = np.max(times)
            pct_of_total = (mean / np.mean(timings["total"])) * 100 if component != "total" else 100
            
            print(f"{component:15s}: {mean:6.1f}ms (median: {median:6.1f}ms, range: {min_t:6.1f}-{max_t:6.1f}ms) [{pct_of_total:5.1f}%]")
    
    # Identify bottlenecks
    print(f"\n{'='*80}")
    print("🔴 BOTTLENECK ANALYSIS")
    print(f"{'='*80}\n")
    
    component_times = {}
    for component in components[:-1]:  # Exclude total
        component_times[component] = np.mean(timings[component])
    
    sorted_components = sorted(component_times.items(), key=lambda x: x[1], reverse=True)
    
    total_avg = np.mean(timings["total"])
    
    print("Components by execution time (slowest first):\n")
    for i, (component, avg_time) in enumerate(sorted_components, 1):
        pct = (avg_time / total_avg) * 100
        bar = "█" * int(pct / 5)
        print(f"{i}. {component:15s}: {avg_time:6.1f}ms ({pct:5.1f}%) {bar}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("💡 OPTIMIZATION RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    recommendations = []
    
    if component_times.get("segmentation", 0) > 100:
        recommendations.append("❌ SEGMENTATION is the main bottleneck (>100ms)")
        recommendations.append("   → Reduce INFERENCE_IMGSZ to 192 or 128")
        recommendations.append("   → Or use a smaller model (SegFormer-B0 vs B1/B2)")
    
    if component_times.get("depth", 0) > 50:
        recommendations.append("❌ DEPTH estimation is slow (>50ms)")
        recommendations.append("   → Pass depth_est=None to skip it")
    
    if component_times.get("trend", 0) > 20:
        recommendations.append("❌ TREND tracking is slow (>20ms)")
        recommendations.append("   → Disable trend tracking")
    
    if component_times.get("alerts", 0) > 20:
        recommendations.append("❌ ALERTS processing is slow (>20ms)")
        recommendations.append("   → Already disabled on cloud (ALERTS_ENABLED=false)")
    
    if component_times.get("composer", 0) > 20:
        recommendations.append("❌ COMPOSER is slow (>20ms)")
        recommendations.append("   → Simplify phrase composition")
    
    if not recommendations:
        recommendations.append("✅ All components are reasonably fast (<20ms each)")
    
    for rec in recommendations:
        print(rec)
    
    print(f"\n{'='*80}")
    print(f"⏱️  AVERAGE LATENCY: {total_avg:.1f}ms ({1000/total_avg:.1f} FPS)")
    print(f"{'='*80}\n")
    
    return {
        "video": str(video_path.name),
        "frames_analyzed": frame_idx,
        "component_times": component_times,
        "total_avg_ms": total_avg,
        "fps": 1000 / total_avg,
    }


def main():
    videos = [
        r"C:\Users\chira\Downloads\1000118169.mp4",
    ]
    
    for video_path in videos:
        report = detailed_latency_analysis(video_path, max_frames=50)


if __name__ == "__main__":
    main()
