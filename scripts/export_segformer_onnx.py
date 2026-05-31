"""Export SegFormer-B0 (ADE20K) to ONNX and apply INT8 quantization.

Run once from the project root (with [segformer] extras installed):

    python scripts/export_segformer_onnx.py

Produces two files:
  segformer_b0_ade20k.onnx          — FP32 ONNX (baseline)
  segformer_b0_ade20k_int8.onnx     — INT8 quantized (fastest on CPU)

The INT8 model is what the pipeline uses by default when
SEGMENTER_BACKEND=segformer_onnx is set in .env.

Why B0?
  B0 has 3.7M parameters vs B2's 25M. On CPU, B0 ONNX INT8 targets
  ~50-100ms/frame (10-20 FPS) vs B2 PyTorch's 800ms (1.2 FPS).

Why INT8?
  Dynamic INT8 quantization converts weights from float32 to int8,
  cutting memory bandwidth and compute roughly in half. Accuracy drop
  on ADE20K is small (~1-2 mIoU). No calibration dataset needed for
  dynamic quantization — it quantizes weights statically and activations
  dynamically at runtime.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def export(model_id: str, output_fp32: Path, output_int8: Path) -> None:
    try:
        import torch
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
    except ImportError:
        print("Install segformer extras first: pip install -e '.[segformer]'")
        sys.exit(1)

    print(f"Loading {model_id} ...", flush=True)
    processor = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id)
    model.eval()

    # Save the id2label map alongside the ONNX so the runtime can resolve
    # class names without needing transformers at inference time. The runtime
    # loads the label file matching the *active* model's stem, so write one
    # next to BOTH the FP32 and INT8 models — otherwise the default INT8 path
    # finds no labels and class names degrade to numeric IDs (which breaks the
    # name-based walkable/obstacle reasoning).
    id2label = {str(k): str(v) for k, v in model.config.id2label.items()}
    payload = json.dumps(id2label, indent=2)
    for stem_path in (output_fp32, output_int8):
        label_path = stem_path.with_suffix(".json")
        label_path.write_text(payload)
        print(f"Saved id2label -> {label_path}", flush=True)

    # Dummy input: the processor resizes to 512x512 by default.
    dummy_rgb = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    inputs = processor(images=dummy_rgb, return_tensors="pt")
    pixel_values = inputs["pixel_values"]  # (1, 3, 512, 512)

    print(f"Exporting to ONNX -> {output_fp32} ...", flush=True)
    # PyTorch 2.x defaults to the new dynamo exporter, which needs the extra
    # ``onnxscript`` package and a different (dynamic_shapes) API. Our args are
    # the classic TorchScript-exporter style, so force ``dynamo=False`` to use
    # the legacy path. This keeps the export dependency-light (no onnxscript)
    # and matches input_names/dynamic_axes/opset_version below.
    export_kwargs = dict(
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "out_height", 3: "out_width"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    try:
        # ``dynamo`` kwarg exists on torch>=2.0; pin it off where supported.
        torch.onnx.export(
            model, (pixel_values,), str(output_fp32),
            dynamo=False, **export_kwargs,
        )
    except TypeError:
        # Older torch without the ``dynamo`` kwarg — legacy exporter is default.
        torch.onnx.export(
            model, (pixel_values,), str(output_fp32), **export_kwargs,
        )
    size_mb = output_fp32.stat().st_size / 1024 / 1024
    print(f"FP32 ONNX: {output_fp32} ({size_mb:.1f} MB)", flush=True)

    # INT8 dynamic quantization — weights quantized statically,
    # activations quantized dynamically at runtime. No calibration data needed.
    print(f"Quantizing to INT8 -> {output_int8} ...", flush=True)
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        # Note: ``optimize_model`` was removed in newer onnxruntime
        # (>=1.16). Graph optimization now happens at session-creation time
        # via SessionOptions.graph_optimization_level (set in the runtime
        # backend), so it's not needed here.
        quantize_dynamic(
            str(output_fp32),
            str(output_int8),
            weight_type=QuantType.QInt8,
        )
        size_mb_int8 = output_int8.stat().st_size / 1024 / 1024
        print(f"INT8 ONNX: {output_int8} ({size_mb_int8:.1f} MB)", flush=True)
    except ImportError:
        print(
            "onnxruntime.quantization not available — "
            "install onnxruntime: pip install onnxruntime>=1.18",
            flush=True,
        )
        return

    # Quick sanity check: run one inference through the INT8 model.
    print("Sanity check: running one inference through INT8 model ...", flush=True)
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(output_int8),
        providers=["CPUExecutionProvider"],
    )
    out = sess.run(None, {"pixel_values": pixel_values.numpy()})
    logits = out[0]  # (1, 150, H/4, W/4)
    print(f"Output shape: {logits.shape}  dtype: {logits.dtype}", flush=True)
    print("Export complete.", flush=True)


def main() -> None:
    model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
    fp32 = Path("segformer_b0_ade20k.onnx")
    int8 = Path("segformer_b0_ade20k_int8.onnx")
    export(model_id, fp32, int8)


if __name__ == "__main__":
    main()
