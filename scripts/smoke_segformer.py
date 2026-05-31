"""Real SegFormer ADE20K inference smoke test (no mocks).

Builds the segmenter via the production factory so it exercises whatever
backend is configured — by default the fast ONNX INT8 path once the model is
exported (``python scripts/export_segformer_onnx.py``). Runs the actual model
on the sample fixture, reports steady-state latency, and prints the class
breakdown so we can confirm the backend works end-to-end (including class-name
resolution) and judge the obstacle/walkable class mapping.

Usage:
    python scripts/smoke_segformer.py [image_path] [backend]

``backend`` overrides settings.segmenter_backend (e.g. ``segformer`` to force
the transformers path for comparison).
"""

from __future__ import annotations

import statistics
import sys
import time

import cv2

from navigation.config import Settings
from navigation.perception.segmentation_base import build_segmenter

_RUNS = 5  # timed runs after warmup, for a steady-state latency estimate


def main() -> int:
    img_path = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/sample.jpg"
    backend = sys.argv[2] if len(sys.argv) > 2 else None

    frame = cv2.imread(img_path)
    if frame is None:
        print(f"ERROR: could not read {img_path}", flush=True)
        return 1
    print(f"frame shape: {frame.shape}", flush=True)

    overrides = {"segmenter_backend": backend} if backend else {}
    settings = Settings(**overrides)
    seg = build_segmenter(settings)
    print(f"segmenter class: {type(seg).__name__}", flush=True)

    # Warmup: first call loads the model (and downloads it for the transformers
    # backend). Excluded from the latency numbers.
    t0 = time.perf_counter()
    print("warmup (loads model; transformers backend downloads ~110MB)...", flush=True)
    result = seg.predict(frame)
    print(f"warmup wall time (incl. any load/download): {time.perf_counter() - t0:.2f}s",
          flush=True)

    # Steady-state latency.
    times_ms: list[float] = []
    for _ in range(_RUNS):
        t = time.perf_counter()
        result = seg.predict(frame)
        times_ms.append((time.perf_counter() - t) * 1000)
    mean_ms = statistics.mean(times_ms)
    fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
    print(
        f"steady-state latency over {_RUNS} runs: "
        f"{mean_ms:.1f} ms/frame  (~{fps:.1f} FPS)  "
        f"[min {min(times_ms):.1f}, max {max(times_ms):.1f}]",
        flush=True,
    )

    print(f"backend: {result.metadata.get('backend')}", flush=True)
    print(f"num classes in id_to_name: {len(result.metadata.get('id_to_name', {}))}", flush=True)
    print(f"classes present: {sorted(result.class_names)}", flush=True)
    print(f"walkable_ratio: {result.walkable_ratio:.3f}", flush=True)
    print(f"obstacle_pixels: {result.obstacle_pixels}", flush=True)
    print(f"obstacle_pixels_weighted: {result.obstacle_pixels_weighted:.0f}", flush=True)

    # Top classes by pixel count, to judge the mapping.
    counts = result.metadata.get("pixel_counts", {})
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("top classes by pixels:", flush=True)
    for name, px in top:
        print(f"  {name:20s} {px}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
