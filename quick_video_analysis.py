"""Quick video analysis - process first 100 frames to identify issues."""

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
from navigation.perception.stairs import StairsDetector
from navigation.reasoning.llm import NavigationInterpreter
from navigation.output.tts import SpeechEngine
from navigation.pipeline.runner import process_frame


def quick_analyze(video_path, max_frames=100):
    """Quick analysis of first N frames."""
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return None
    
    print(f"\n{'='*80}")
    print(f"📹 Quick Analysis: {video_path.name}")
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
    
    print(f"✓ Video: {width}x{height} @ {fps:.1f}fps, {total_frames} total frames")
    print(f"✓ Analyzing first {min(max_frames, total_frames)} frames...\n")
    
    # Analysis data
    frame_data = []
    command_counts = defaultdict(int)
    confidence_values = []
    walkable_ratios = []
    false_stops = 0
    
    frame_idx = 0
    start_time = time.time()
    
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
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
            rationale = record.get("rationale", "")
            
            # Get segmentation data
            seg = segmenter.last_segmentation
            if seg:
                walkable_ratio = seg.walkable_ratio
                obstacle_pixels = seg.obstacle_pixels
                walkable_ratios.append(walkable_ratio)
            else:
                walkable_ratio = 0.0
                obstacle_pixels = 0
            
            # Track metrics
            command_counts[command] += 1
            confidence_values.append(confidence)
            
            # Detect false stops
            if command == "stop" and walkable_ratio > 0.6:
                false_stops += 1
                print(f"  ⚠️  Frame {frame_idx}: FALSE STOP (walkable={walkable_ratio:.1%}, conf={confidence:.2f})")
            
            frame_data.append({
                "frame": frame_idx,
                "command": command,
                "confidence": float(confidence),
                "phrase": phrase,
                "rationale": rationale,
                "walkable_ratio": float(walkable_ratio),
                "obstacle_pixels": int(obstacle_pixels),
            })
            
        except Exception as e:
            print(f"❌ Error frame {frame_idx}: {e}")
            continue
    
    cap.release()
    elapsed = time.time() - start_time
    
    print(f"\n✓ Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
    
    # Generate report
    print(f"\n{'='*80}")
    print("📊 ANALYSIS REPORT")
    print(f"{'='*80}")
    
    print(f"\n🎯 Command Distribution:")
    for cmd, count in sorted(command_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count/frame_idx)*100
        print(f"   {cmd:15s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\n📈 Confidence Stats:")
    if confidence_values:
        print(f"   Mean:   {np.mean(confidence_values):.3f}")
        print(f"   Median: {np.median(confidence_values):.3f}")
        print(f"   Min:    {np.min(confidence_values):.3f}")
        print(f"   Max:    {np.max(confidence_values):.3f}")
        print(f"   Std:    {np.std(confidence_values):.3f}")
    
    print(f"\n🚶 Walkable Ratio Stats:")
    if walkable_ratios:
        print(f"   Mean:   {np.mean(walkable_ratios):.3f}")
        print(f"   Median: {np.median(walkable_ratios):.3f}")
        print(f"   Min:    {np.min(walkable_ratios):.3f}")
        print(f"   Max:    {np.max(walkable_ratios):.3f}")
    
    print(f"\n⚠️  ISSUES DETECTED:")
    print(f"   False STOP triggers: {false_stops} ({(false_stops/frame_idx)*100:.1f}%)")
    
    # Identify main issues
    issues = []
    
    if false_stops > frame_idx * 0.1:
        issues.append("❌ HIGH FALSE STOP RATE - System is too sensitive")
    
    if command_counts.get("stop", 0) > frame_idx * 0.5:
        issues.append("❌ EXCESSIVE STOP COMMANDS - Model is over-cautious")
    
    if np.mean(confidence_values) < 0.6:
        issues.append("❌ LOW AVERAGE CONFIDENCE - Model is uncertain")
    
    if np.mean(walkable_ratios) < 0.3:
        issues.append("❌ LOW WALKABLE RATIO - Model sees too many obstacles")
    
    if not issues:
        issues.append("✅ No major issues detected")
    
    print(f"\n🔍 MAIN ISSUES:")
    for issue in issues:
        print(f"   {issue}")
    
    return {
        "video": str(video_path.name),
        "frames_analyzed": frame_idx,
        "command_distribution": dict(command_counts),
        "confidence_mean": float(np.mean(confidence_values)) if confidence_values else 0.0,
        "walkable_mean": float(np.mean(walkable_ratios)) if walkable_ratios else 0.0,
        "false_stops": false_stops,
        "false_stop_pct": (false_stops/frame_idx)*100 if frame_idx > 0 else 0.0,
        "issues": issues,
    }


def main():
    videos = [
        r"C:\Users\chira\Downloads\1000118169.mp4",
        r"C:\Users\chira\Downloads\1000118165.mp4",
    ]
    
    all_reports = {}
    
    for video_path in videos:
        report = quick_analyze(video_path, max_frames=100)
        if report:
            all_reports[report["video"]] = report
    
    # Summary
    print(f"\n{'='*80}")
    print("📋 SUMMARY")
    print(f"{'='*80}\n")
    
    for video_name, report in all_reports.items():
        print(f"{video_name}:")
        print(f"  False STOPs: {report['false_stops']} ({report['false_stop_pct']:.1f}%)")
        print(f"  Avg Confidence: {report['confidence_mean']:.3f}")
        print(f"  Avg Walkable: {report['walkable_mean']:.3f}")
        print()


if __name__ == "__main__":
    main()
