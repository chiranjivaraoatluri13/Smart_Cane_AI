"""Process a video file and save output with segmentation overlay and JSON logs."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from navigation.config import load_settings
from navigation.models import Position
from navigation.output.hud import draw_navigation_hud
from navigation.perception.visualize import render_overlay, save_overlay
from navigation.pipeline.runner import (
    _build_pipeline_components,
    process_frame,
)

def process_video(video_path: str, output_dir: str = "output/video_debug"):
    """Process video and save output frames with overlay and JSON logs."""
    
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    print(f"📹 Processing video: {video_path}")
    print(f"💾 Output directory: {output_dir}")
    
    # Load settings and components
    settings = load_settings()
    (
        segmenter,
        depth_est,
        care,
        interpreter,
        validator,
        tts,
        alert_tracker,
        spatial_reasoner,
        composer,
        voice_queue,
        trend_tracker,
        stairs_detector,
    ) = _build_pipeline_components(settings)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video info: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")
    
    # Prepare output video writer
    output_video_path = output_dir / "output_with_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    # Process frames
    frame_id = 0
    json_logs = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            print(f"Processing frame {frame_id + 1}/{total_frames}...", end="\r")
            
            try:
                # Process frame
                record = process_frame(
                    frame,
                    frame_id=frame_id,
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
                    show_seg=False,
                    seg_save_dir=None,
                )
                
                # Save JSON log
                json_logs.append(record)
                
                # Create overlay with HUD
                overlay = render_overlay(frame, segmenter=segmenter)
                overlay = draw_navigation_hud(
                    overlay,
                    phrase=str(record.get("phrase", "")),
                    command=str(record.get("command", "")),
                    speak=bool(record.get("speak", False)),
                    confidence=float(record.get("confidence", 0)),
                    rationale=str(record.get("rationale", "")),
                    stale=False,
                )
                
                # Write to output video
                out.write(overlay)
                
                # Save individual frame
                frame_path = output_dir / f"frame_{frame_id:06d}.jpg"
                cv2.imwrite(str(frame_path), overlay)
                
            except Exception as e:
                print(f"❌ Error processing frame {frame_id}: {e}")
                json_logs.append({
                    "frame_id": frame_id,
                    "error": str(e),
                })
            
            frame_id += 1
    
    finally:
        cap.release()
        out.release()
    
    # Save JSON logs
    json_path = output_dir / "logs.json"
    with open(json_path, "w") as f:
        json.dump(json_logs, f, indent=2, default=str)
    
    print(f"\n✅ Processing complete!")
    print(f"📹 Output video: {output_video_path}")
    print(f"📊 JSON logs: {json_path}")
    print(f"🖼️  Frames saved to: {output_dir}")
    print(f"\n📈 Summary:")
    print(f"   Total frames: {frame_id}")
    print(f"   Output video: {output_video_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Print first few frames for analysis
    print(f"\n🔍 First 5 frames analysis:")
    for i, log in enumerate(json_logs[:5]):
        if "error" not in log:
            print(f"   Frame {i}: {log.get('command')} (confidence: {log.get('confidence', 0):.2f})")
            print(f"      Phrase: {log.get('phrase', 'N/A')}")
            print(f"      Rationale: {log.get('rationale', 'N/A')}")

if __name__ == "__main__":
    video_path = r"C:\Users\chira\Downloads\WhatsApp Video 2026-05-31 at 12.59.16 AM.mp4"
    process_video(video_path)
