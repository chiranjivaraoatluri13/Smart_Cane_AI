"""Render the project architecture Mermaid diagram to a PNG.

Uses the public mermaid.ink renderer (base64-encoded graph -> PNG image),
so no local Chromium/Node install is required. Saves to docs/architecture.png.
"""

from __future__ import annotations

import base64
import urllib.request
from pathlib import Path

MERMAID = r"""
flowchart TD
    cam["Camera / image / phone JPEG<br/>(+ GPS, heading, on-device depth_m)"]

    cam --> factory{{"build_segmenter()<br/>settings.segmenter_backend"}}
    factory -->|segformer_onnx default| onnx["SegformerOnnxSegmenter<br/>ADE20K B0 INT8 via onnxruntime<br/>~80ms/frame CPU"]
    factory -->|segformer fallback| segf["SegformerSegmenter<br/>ADE20K B2 via transformers"]
    onnx --> seg["SegmentationResult<br/>class_map, weighted obstacles,<br/>per-side walkable, id_to_name"]
    segf --> seg

    cfg["Settings.seg_class_config()<br/>backend-aware class lists"] -.-> seg

    seg --> depth["UniDepthEstimator<br/>1. external depth_m (phone)<br/>2. segmentation proxy<br/>3. brightness fallback"]
    depth --> care["CareNavigator<br/>HTTP endpoint or heuristic"]
    care --> bundle["PerceptionBundle"]

    bundle --> reasoner["SpatialReasoner (default)<br/>or legacy NavigationInterpreter"]
    route["MapGuidance / OSRM RouteCue"] --> reasoner
    stairs["StairsDetector"] --> reasoner
    trend["TrendTracker<br/>crossing / closing-in / receding"] --> reasoner

    reasoner --> facts["GuidanceFacts + NavigationDecision<br/>(vision_stop strips route cue)"]
    facts --> validator["CommandValidator<br/>dwell - speak-on-change - min-gap"]
    validator --> composer["PhraseComposer<br/>phrases.yaml to spoken phrase"]
    composer --> tts["SpeechEngine (pyttsx3)<br/>/ Web Speech API on phone"]

    seg --> alerts["AlertTracker<br/>'Car approaching' + guardrails"]
    alerts --> validator

    validator --> record["JSON record<br/>command, phrase, speak, facts,<br/>alerts, timings_ms"]
    record --> phone["Phone client UI / HUD"]

    classDef input fill:#1f6feb,stroke:#0b3d91,color:#fff;
    classDef percep fill:#2ea043,stroke:#176f2c,color:#fff;
    classDef reason fill:#bf8700,stroke:#7d5700,color:#fff;
    classDef output fill:#8957e5,stroke:#553098,color:#fff;
    class cam,phone input;
    class factory,onnx,segf,seg,cfg,depth,care,bundle,stairs,trend,alerts percep;
    class reasoner,facts,route reason;
    class validator,composer,tts,record output;
"""


def main() -> int:
    out = Path("docs/architecture.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    graph_b64 = base64.urlsafe_b64encode(MERMAID.strip().encode("utf-8")).decode("ascii")
    # type=png, white background, higher scale for a crisp portfolio image.
    url = f"https://mermaid.ink/img/{graph_b64}?type=png&bgColor=ffffff&scale=3"

    req = urllib.request.Request(url, headers={"User-Agent": "assistive-nav-doc/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()

    out.write_bytes(data)
    print(f"Wrote {out} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
