"""Benchmark B0 vs B2 SegFormer inference time on this machine."""
import time
import numpy as np
from navigation.config import Settings
from navigation.perception.segmentation_segformer import SegformerSegmenter

frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
frame_small = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

configs = [
    ("B0 480x640", "nvidia/segformer-b0-finetuned-ade-512-512", frame),
    ("B0 240x320", "nvidia/segformer-b0-finetuned-ade-512-512", frame_small),
    ("B2 480x640", "nvidia/segformer-b2-finetuned-ade-512-512", frame),
    ("B2 240x320", "nvidia/segformer-b2-finetuned-ade-512-512", frame_small),
]

for label, model_id, f in configs:
    s = Settings(segmenter_backend="segformer", segformer_model_id=model_id)
    seg = SegformerSegmenter(s)
    seg.predict(f)  # warmup / model load
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        seg.predict(f)
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    mn = min(times)
    print(f"{label:15s}  avg={avg:6.0f}ms  min={mn:6.0f}ms  => {1000/avg:.1f} FPS", flush=True)
    del seg
