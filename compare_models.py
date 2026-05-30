"""Compare YOLO segmentation models for accuracy."""

import sys
from pathlib import Path
from ultralytics import YOLO

def download_and_test_model(model_name: str):
    """Download a YOLO model and show its info."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print('='*60)
    
    try:
        model = YOLO(model_name)
        print(f"✓ Model downloaded: {model_name}")
        print(f"  Task: {model.task}")
        print(f"  Classes: {len(model.names) if hasattr(model, 'names') else 'N/A'}")
        
        # Show model info
        model.info()
        
        return True
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("YOLO SEMANTIC SEGMENTATION MODEL COMPARISON")
    print("="*60)
    print("\nDownloading models for testing...")
    print("(First download may take a few minutes)")
    
    models = [
        ("yolo11n-sem.pt", "Nano - Fast, Low Accuracy"),
        ("yolo11s-sem.pt", "Small - Balanced (RECOMMENDED)"),
        ("yolo11m-sem.pt", "Medium - Slower, High Accuracy"),
    ]
    
    results = {}
    for model_name, description in models:
        print(f"\n{description}")
        success = download_and_test_model(model_name)
        results[model_name] = success
    
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for model_name, success in results.items():
        status = "✓ Ready" if success else "✗ Failed"
        print(f"{status}: {model_name}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Edit .env file and change YOLO_MODEL to one of:")
    for model_name in results.keys():
        print(f"   YOLO_MODEL={model_name}")
    
    print("\n2. Reprocess your video:")
    print("   python process_video.py VIDEO.mp4 --output-video OUTPUT.mp4")
    
    print("\n3. Compare outputs to see accuracy improvement")
    print("="*60)

if __name__ == "__main__":
    main()
