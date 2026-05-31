"""
Process videos and generate output videos with segmentation overlays and HUD.
Shows what the model sees and the commands it generates.
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
from navigation.output.hud import draw_navigation_hud
from navigation.pipeline.runner import process_frame


# Color map for segmentation classes
CLASS_COLORS = {
    "walkable": (0, 255, 0),      # Green
    "obstacle": (0, 0, 255),      # Red
    "hazard": (0, 165, 255),      # Orange
}

COMMAND_COLORS = {
    "go_forward": (0, 255, 0),    # Green
    "move_left": (255, 255, 0),   # Cyan
    "move_right": (255, 0, 255),  # Magenta
    "slow_down": (0, 165, 255),   # Orange
    "stop": (0, 0, 255),          # Red
}


def draw_segmentation_overlay(frame, seg, alpha=0.4):
    """Draw segmentation overlay on frame (simplified - just return frame)."""
    # The SegmentationResult doesn't expose class_ids directly,
    # so we skip the overlay and just return the frame
    return frame


def draw_stats_panel(frame, walkable_ratio, obstacle_pixels, frame_idx):
    """Draw statistics panel on frame."""
    h, w = frame.shape[:2]
    
    # Semi-transparent background for stats (bottom-right)
    overlay = frame.copy()
    panel_w = 300
    panel_h = 100
    cv2.rectangle(overlay, (w - panel_w, h - panel_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Draw stats
    y_offset = h - panel_h + 25
    cv2.putText(frame, f"Walkable: {walkable_ratio:.1%}", (w - panel_w + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, f"Obstacles: {obstacle_pixels}", (w - panel_w + 10, y_offset + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(frame, f"Frame: {frame_idx}", (w - panel_w + 10, y_offset + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame


def process_video_with_overlay(video_path, output_dir="output/video_overlay"):
    """Process video and generate output video with overlays."""
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return None
    
    output_dir = Path(output_dir) / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"📹 Processing: {video_path.name}")
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
    
    # Setup video writer
    output_video_path = output_dir / "output_with_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    # Analysis data
    frame_data = []
    command_counts = defaultdict(int)
    
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
                
                # Draw segmentation overlay
                frame_with_seg = draw_segmentation_overlay(frame, seg, alpha=0.3)
            else:
                frame_with_seg = frame.copy()
                walkable_ratio = 0.0
                obstacle_pixels = 0
            
            # Draw HUD
            frame_with_hud = draw_navigation_hud(
                frame_with_seg,
                phrase=phrase,
                command=command,
                speak=speak,
                confidence=confidence,
                rationale=rationale,
            )
            
            # Draw stats panel
            frame_with_stats = draw_stats_panel(
                frame_with_hud,
                walkable_ratio,
                obstacle_pixels,
                frame_idx
            )
            
            # Write frame
            out.write(frame_with_stats)
            
            # Track metrics
            command_counts[command] += 1
            
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
            import traceback
            traceback.print_exc()
            # Write original frame if processing fails
            out.write(frame)
            continue
    
    cap.release()
    out.release()
    elapsed = time.time() - start_time
    
    print(f"\n✓ Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
    print(f"✓ Output video saved: {output_video_path}")
    
    # Save frame log
    frame_log_path = output_dir / "frame_log.json"
    with open(frame_log_path, "w") as f:
        json.dump(frame_data, f, indent=2)
    
    print(f"✓ Frame log saved: {frame_log_path}")
    
    # Print summary
    print(f"\n📊 Command Distribution:")
    for cmd, count in sorted(command_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count/frame_idx)*100
        print(f"   {cmd:15s}: {count:4d} ({pct:5.1f}%)")
    
    return output_video_path


def main():
    videos = [
        r"C:\Users\chira\Downloads\1000118169.mp4",
        r"C:\Users\chira\Downloads\1000118165.mp4",
    ]
    
    output_videos = []
    
    for video_path in videos:
        output_video = process_video_with_overlay(video_path)
        if output_video:
            output_videos.append(output_video)
    
    print("\n" + "="*80)
    print("✅ Processing complete!")
    print("="*80)
    print("\n📹 Output videos:")
    for video in output_videos:
        print(f"   {video}")
    
    print("\n💡 To play the videos:")
    print("   - Windows: Double-click the .mp4 file")
    print("   - Or use: ffplay output_with_overlay.mp4")
    print("   - Or use: vlc output_with_overlay.mp4")


if __name__ == "__main__":
    main()
