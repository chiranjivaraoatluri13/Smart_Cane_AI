#!/usr/bin/env python3
"""Fine-tune YOLO semantic segmentation on Cityscapes (or a custom YAML dataset).

Training is intentionally not run in CI — it needs a GPU and the Cityscapes dataset.
Pretrained ``yolo26n-sem.pt`` already labels all 19 Cityscapes classes for typical streets.

Usage (from project root, venv with [vision] installed):

    pip install -e ".[vision]"
    python scripts/train_semantic.py --epochs 50 --device 0

Equivalent Ultralytics CLI:

    yolo train model=yolo26n-sem.pt data=cityscapes.yaml task=semantic epochs=50 imgsz=1024 batch=8 device=0

For a custom environment (indoor, campus), duplicate ``config/cityscapes_train.yaml``,
point ``path`` at your images/labels, and keep class names aligned with
``config/default.yaml`` or extend both consistently.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26 semantic on Cityscapes")
    parser.add_argument(
        "--model",
        default="yolo26n-sem.pt",
        help="Base weights (downloads on first run)",
    )
    parser.add_argument(
        "--data",
        default="cityscapes.yaml",
        help="Ultralytics dataset YAML (built-in or config/cityscapes_train.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'")
    parser.add_argument("--project", default="runs/semantic")
    parser.add_argument("--name", default="fine-tune")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install vision extras: pip install -e '.[vision]'")
        return 1

    model = YOLO(args.model)
    model.train(
        data=args.data,
        task="semantic",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
