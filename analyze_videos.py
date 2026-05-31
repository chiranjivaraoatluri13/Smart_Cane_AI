"""
Comprehensive video analysis script for model performance evaluation.
Processes videos and generates detailed reports on segmentation, commands, and issues.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import time

# Add project root to path
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
from navigation.perception.stairs import StairsDetector
from navigation.reasoning.llm import NavigationInterpreter
from navigation.output.tts import SpeechEngine
from navigation.pipeline.runner import process_frame


def analyze_video(video_path, output_dir="output/video_analysis"):
    """Process video and generate detailed analysis report."""
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return None
    
    output_dir = Path(output_dir) / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"📹 Analyzing: {video_path.name}")
    print(f"{'='*80}")
    
    # Load models
    print("Loading models...")
    settings = load_settings()
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
    stairs_detector = StairsDetector(settings)
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
    
    print(f"✓ Video info: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    
    # Analysis data
    frame_data = []
    command_counts = defaultdict(int)
    confidence_values = []
    walkable_ratios = []
    obstacle_pixels_list = []
    vision_stop_count = 0
    hazard_detected_count = 0
    false_stops = 0
    
    frame_idx = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"  Processing frame {frame_idx}/{total_frames}...", end='\r')
        
        try:
            # Process frame
            record = process_frame(
                frame,
                frame_id=frame_idx,
                settings=settings,
                segmenter=segmenter,
                depth_est=depth_est,
                care=care,
                interpreter=interpreter,
                validator=validator,
                tts=tts,
                alert_tracker=alert_tracker,
                spatial_reasoner=spatial_reasoner,
                composer=composer,
                voice_queue=voice_queue,
                trend_tracker=trend_tracker,
                stairs_detector=stairs_detector,
                position=None,
                client_depth_m=None,
            )
            
            # Extract data
            command = record.get("command", "unknown")
            confidence = record.get("confidence", 0.0)
            phrase = record.get("phrase", "")
            speak = record.get("speak", False)
            rationale = record.get("rationale", "")
            
            # Get segmentation data
            seg = segmenter.last_segmentation
            if seg:
                walkable_ratio = seg.walkable_ratio
                obstacle_pixels = seg.obstacle_pixels
                walkable_ratios.append(walkable_ratio)
                obstacle_pixels_list.append(obstacle_pixels)
            else:
                walkable_ratio = 0.0
                obstacle_pixels = 0
            
            # Track metrics
            command_counts[command] += 1
            confidence_values.append(confidence)
            
            # Check for vision_stop
            if "vision_stop" in rationale.lower():
                vision_stop_count += 1
            
            # Check for hazard
            if "hazard" in rationale.lower():
                hazard_detected_count += 1
            
            # Detect false stops (STOP on plain path with high walkable ratio)
            if command == "stop" and walkable_ratio > 0.6:
                false_stops += 1
            
            frame_data.append({
                "frame": frame_idx,
                "command": command,
                "confidence": float(confidence),
                "phrase": phrase,
                "speak": speak,
                "rationale": rationale,
                "walkable_ratio": float(walkable_ratio),
                "obstacle_pixels": int(obstacle_pixels),
            })
            
        except Exception as e:
            print(f"❌ Error processing frame {frame_idx}: {e}")
            continue
    
    cap.release()
    elapsed = time.time() - start_time
    
    print(f"\n✓ Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
    
    # Generate report
    report = {
        "video": str(video_path),
        "total_frames": frame_idx,
        "fps": fps,
        "resolution": f"{width}x{height}",
        "processing_time_sec": elapsed,
        "processing_fps": frame_idx / elapsed,
        
        "command_distribution": dict(command_counts),
        "command_percentages": {
            cmd: f"{(count/frame_idx)*100:.1f}%" 
            for cmd, count in command_counts.items()
        },
        
        "confidence_stats": {
            "mean": float(np.mean(confidence_values)) if confidence_values else 0.0,
            "median": float(np.median(confidence_values)) if confidence_values else 0.0,
            "min": float(np.min(confidence_values)) if confidence_values else 0.0,
            "max": float(np.max(confidence_values)) if confidence_values else 0.0,
            "std": float(np.std(confidence_values)) if confidence_values else 0.0,
        },
        
        "walkable_ratio_stats": {
            "mean": float(np.mean(walkable_ratios)) if walkable_ratios else 0.0,
            "median": float(np.median(walkable_ratios)) if walkable_ratios else 0.0,
            "min": float(np.min(walkable_ratios)) if walkable_ratios else 0.0,
            "max": float(np.max(walkable_ratios)) if walkable_ratios else 0.0,
        },
        
        "obstacle_pixels_stats": {
            "mean": float(np.mean(obstacle_pixels_list)) if obstacle_pixels_list else 0.0,
            "median": float(np.median(obstacle_pixels_list)) if obstacle_pixels_list else 0.0,
            "max": float(np.max(obstacle_pixels_list)) if obstacle_pixels_list else 0.0,
        },
        
        "issues": {
            "vision_stop_count": vision_stop_count,
            "hazard_detected_count": hazard_detected_count,
            "false_stops": false_stops,
            "false_stop_percentage": f"{(false_stops/frame_idx)*100:.1f}%" if frame_idx > 0 else "0%",
        },
    }
    
    # Save detailed frame data
    frame_log_path = output_dir / "frame_log.json"
    with open(frame_log_path, "w") as f:
        json.dump(frame_data, f, indent=2)
    
    # Save report
    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*80)
    print("📊 ANALYSIS REPORT")
    print("="*80)
    print(f"\n📹 Video: {video_path.name}")
    print(f"   Frames: {frame_idx} @ {fps:.1f}fps")
    print(f"   Processing: {frame_idx/elapsed:.1f} fps ({elapsed:.1f}s total)")
    
    print(f"\n🎯 Command Distribution:")
    for cmd, count in sorted(command_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count/frame_idx)*100
        print(f"   {cmd:15s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\n📈 Confidence Stats:")
    print(f"   Mean:   {report['confidence_stats']['mean']:.3f}")
    print(f"   Median: {report['confidence_stats']['median']:.3f}")
    print(f"   Range:  {report['confidence_stats']['min']:.3f} - {report['confidence_stats']['max']:.3f}")
    print(f"   Std:    {report['confidence_stats']['std']:.3f}")
    
    print(f"\n🚶 Walkable Ratio Stats:")
    print(f"   Mean:   {report['walkable_ratio_stats']['mean']:.3f}")
    print(f"   Median: {report['walkable_ratio_stats']['median']:.3f}")
    print(f"   Range:  {report['walkable_ratio_stats']['min']:.3f} - {report['walkable_ratio_stats']['max']:.3f}")
    
    print(f"\n⚠️  Issues Detected:")
    print(f"   Vision STOP triggers: {vision_stop_count}")
    print(f"   Hazard detections:    {hazard_detected_count}")
    print(f"   False STOP triggers:  {false_stops} ({report['issues']['false_stop_percentage']})")
    
    print(f"\n💾 Output saved to: {output_dir}")
    print(f"   - frame_log.json (detailed frame-by-frame data)")
    print(f"   - report.json (summary statistics)")
    
    return report


def main():
    videos = [
        r"C:\Users\chira\Downloads\1000118169.mp4",
        r"C:\Users\chira\Downloads\1000118165.mp4",
    ]
    
    all_reports = {}
    
    for video_path in videos:
        report = analyze_video(video_path)
        if report:
            all_reports[Path(video_path).name] = report
    
    # Comparative analysis
    if len(all_reports) > 1:
        print("\n" + "="*80)
        print("📊 COMPARATIVE ANALYSIS")
        print("="*80)
        
        for video_name, report in all_reports.items():
            print(f"\n{video_name}:")
            print(f"  False STOPs: {report['issues']['false_stops']} ({report['issues']['false_stop_percentage']})")
            print(f"  Avg Confidence: {report['confidence_stats']['mean']:.3f}")
            print(f"  Avg Walkable: {report['walkable_ratio_stats']['mean']:.3f}")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
