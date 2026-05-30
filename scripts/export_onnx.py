"""Export yolo26n-sem.pt to ONNX for cloud deployment.

Run once from the project root (with [vision] extras installed):
    python scripts/export_onnx.py

Produces yolo26n-sem.onnx (~6MB) which is committed to the repo and
used by phone_server_cloud.py on Render/Railway.
"""

from pathlib import Path


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install vision extras first: pip install -e '.[vision]'")
        return

    model_path = "yolo26n-sem.pt"
    if not Path(model_path).is_file():
        print(f"Model not found: {model_path}")
        return

    print(f"Exporting {model_path} to ONNX...")
    m = YOLO(model_path)
    out = m.export(format="onnx", imgsz=256, simplify=True)
    import os
    size = os.path.getsize(out) / 1024 / 1024
    print(f"Exported: {out} ({size:.1f} MB)")
    print("Commit yolo26n-sem.onnx to your repo before deploying to Render.")


if __name__ == "__main__":
    main()
