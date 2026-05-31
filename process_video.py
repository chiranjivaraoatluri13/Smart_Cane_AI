"""
Process pre-recorded video through assistive navigation pipeline.

Usage:
    python process_video.py <video_path> [--show] [--save-dir OUTPUT_DIR]

Examples:
    python process_video.py "C:/Users/chira/Videos/Screen Recordings/Screen Recording 2026-05-25 193552.mp4" --show
    python process_video.py walking_test.mp4 --save-dir output/video_analysis
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from navigation.config import load_settings
from navigation.output.hud import draw_navigation_hud
from navigation.output.tts import SpeechEngine
from navigation.output.validator import CommandValidator
from navigation.perception.depth import UniDepthEstimator
from navigation.perception.segmentation_base import build_segmenter
from navigation.perception.visualize import render_overlay
from navigation.reasoning.care import CareNavigator
from navigation.reasoning.llm import NavigationInterpreter
from navigation.models import PerceptionBundle


def process_video(
    video_path: str,
    *,
    show: bool = False,
    save_dir: Path | None = None,
    output_video: Path | None = None,
    max_frames: int | None = None,
    use_map: bool = False,
    current_coords: tuple[float, float] | None = None,
    dest_coords: tuple[float, float] | None = None,
):
    """Process video file through navigation pipeline."""
    
    # Convert WSL path to Windows path if needed
    if video_path.startswith('/mnt/'):
        video_path = video_path.replace('/mnt/c/', 'C:\\').replace('/mnt/d/', 'D:\\').replace('/', '\\')
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        print("Trying alternative path...")
        # Try raw path
        import os
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open video. Please check the file path.")
            return 1
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("=" * 60)
    print("VIDEO PROCESSING - ASSISTIVE NAVIGATION")
    print("=" * 60)
    print(f"Video: {Path(video_path).name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.1f}")
    print(f"Frames: {frame_count}")
    print(f"Duration: {frame_count/fps:.1f}s")
    print("=" * 60)
    
    # Setup save directory
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving frames to: {save_dir}")
    
    # Setup output video writer
    video_writer = None
    if output_video:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_video),
            fourcc,
            fps,
            (width, height)
        )
        print(f"Saving output video to: {output_video}")
    
    # Initialize navigation components
    settings = load_settings()
    
    # Override with map settings if provided
    if use_map and current_coords and dest_coords:
        settings = settings.model_copy(update={
            "current_lat": current_coords[0],
            "current_lon": current_coords[1],
            "dest_lat": dest_coords[0],
            "dest_lon": dest_coords[1],
        })
    
    segmenter = build_segmenter(settings)
    depth_est = UniDepthEstimator(settings)
    care = CareNavigator(settings)
    interpreter = NavigationInterpreter(settings)
    validator = CommandValidator(settings)
    tts = SpeechEngine(settings)
    tts.warmup()
    
    print("\nProcessing frames...")
    print("=" * 60)
    
    frame_id = 0
    command_history = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame (configurable)
            if frame_id % settings.process_every_n_frames != 0:
                frame_id += 1
                continue
            
            # Run navigation pipeline
            seg = segmenter.predict(frame)
            depth = depth_est.predict(frame, segmentation=seg)
            care_out = care.predict(frame, seg, depth)
            
            bundle = PerceptionBundle(
                frame_id=frame_id,
                segmentation=seg,
                depth=depth,
                care=care_out,
            )
            
            decision = interpreter.interpret(bundle)
            decision = validator.approve(decision)
            
            # Speak if needed
            phrase = tts.speak(decision) if decision.speak else None
            
            # Record command
            record = {
                "frame_id": frame_id,
                "time_sec": frame_id / fps,
                "command": decision.command.value,
                "confidence": decision.confidence,
                "rationale": decision.rationale,
                "speak": decision.speak,
                "phrase": phrase or decision.command.value.replace("_", " "),
                "obstacle_pixels": seg.obstacle_pixels,
                "walkable_ratio": seg.walkable_ratio,
            }
            command_history.append(record)
            
            # Print progress
            status = "[VOICE]" if decision.speak else "[silent]"
            print(f"Frame {frame_id:6d} ({record['time_sec']:6.1f}s): "
                  f"{decision.command.value.upper():12s} {status:8s} "
                  f"conf={decision.confidence:.2f}")
            
            # Render overlay if showing or saving
            if show or save_dir or video_writer:
                overlay = render_overlay(frame, segmenter=segmenter)
                overlay = draw_navigation_hud(
                    overlay,
                    phrase=record["phrase"],
                    command=decision.command.value,
                    speak=decision.speak,
                    confidence=decision.confidence,
                    rationale=decision.rationale,
                )
                
                if video_writer:
                    video_writer.write(overlay)
                
                if show:
                    cv2.imshow("Assistive Navigation - Video Processing", overlay)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        print("\nUser pressed 'q' - stopping...")
                        break
                
                if save_dir:
                    output_path = save_dir / f"frame_{frame_id:06d}.jpg"
                    cv2.imwrite(str(output_path), overlay)
            
            frame_id += 1
            
            if max_frames and frame_id >= max_frames:
                print(f"\nReached max frames ({max_frames})")
                break
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
            print(f"\nOutput video saved to: {output_video}")
        if show:
            cv2.destroyAllWindows()
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Frames processed: {len(command_history)}")
    
    if command_history:
        # Count commands
        from collections import Counter
        command_counts = Counter(r["command"] for r in command_history)
        voice_count = sum(1 for r in command_history if r["speak"])
        
        print("\nCommand Distribution:")
        for cmd, count in command_counts.most_common():
            pct = 100 * count / len(command_history)
            print(f"  {cmd:15s}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\nVoice commands spoken: {voice_count}")
        print(f"Silent commands: {len(command_history) - voice_count}")
        
        # Avg metrics
        avg_obstacles = sum(r["obstacle_pixels"] for r in command_history) / len(command_history)
        avg_walkable = sum(r["walkable_ratio"] for r in command_history) / len(command_history)
        print(f"\nAvg obstacle pixels: {avg_obstacles:.0f}")
        print(f"Avg walkable ratio: {avg_walkable:.2%}")
        
        # Save summary
        if save_dir:
            import json
            summary_path = save_dir / "analysis_summary.json"
            with open(summary_path, "w") as f:
                json.dump({
                    "video_path": video_path,
                    "frames_processed": len(command_history),
                    "command_distribution": dict(command_counts),
                    "voice_commands": voice_count,
                    "avg_obstacle_pixels": avg_obstacles,
                    "avg_walkable_ratio": avg_walkable,
                    "command_history": command_history,
                }, f, indent=2)
            print(f"\nSummary saved to: {summary_path}")
    
    print("=" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Process video through assistive navigation")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--show", action="store_true", help="Show live visualization")
    parser.add_argument("--save-dir", type=Path, help="Save annotated frames to directory")
    parser.add_argument("--output-video", type=Path, help="Save annotated video to file")
    parser.add_argument("--max-frames", type=int, help="Max frames to process")
    parser.add_argument("--use-map", action="store_true", help="Enable map navigation")
    parser.add_argument("--current", help="Current coordinates (lat,lon)")
    parser.add_argument("--dest", help="Destination coordinates (lat,lon)")
    
    args = parser.parse_args()
    
    # Parse coordinates
    current_coords = None
    dest_coords = None
    if args.current:
        try:
            lat, lon = map(float, args.current.split(","))
            current_coords = (lat, lon)
        except ValueError:
            print("ERROR: --current must be 'lat,lon' format")
            return 1
    
    if args.dest:
        try:
            lat, lon = map(float, args.dest.split(","))
            dest_coords = (lat, lon)
        except ValueError:
            print("ERROR: --dest must be 'lat,lon' format")
            return 1
    
    return process_video(
        args.video,
        show=args.show,
        save_dir=args.save_dir,
        output_video=args.output_video,
        max_frames=args.max_frames,
        use_map=args.use_map,
        current_coords=current_coords,
        dest_coords=dest_coords,
    )


if __name__ == "__main__":
    sys.exit(main())
