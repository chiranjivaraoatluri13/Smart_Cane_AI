# Architecture diagrams

PNG previews used on the repository README. Regenerate them with:

```powershell
python scripts/render_diagrams.py
```

| Diagram | File |
|---------|------|
| Glasses and spoken guidance | [images/01-system-context.png](images/01-system-context.png) |
| Frame pipeline | [images/02-pipeline-architecture.png](images/02-pipeline-architecture.png) |
| Decision priority | [images/05-decision-priority.png](images/05-decision-priority.png) |

## Glasses and spoken guidance

```mermaid
flowchart TD
    glasses["Smart glasses<br/>forward camera, position, heading"]
    glasses --> service["Navigation service<br/>segmentation, depth, route"]
    service --> voice["Voice in the ear<br/>stop, turn, or go forward"]
```

## Frame pipeline

```mermaid
flowchart TD
    capture["Capture"] --> seg["ADE20K SegFormer"]
    seg --> depth["Depth"]
    depth --> care["CARE hazard check"]
    care --> reason["Spatial reasoner"]
    reason --> phrase["Phrase and validator"]
    phrase --> tts["Speech"]
```

## Decision priority

The spatial reasoner uses the first match:

1. Vision stop when a hazard is in the walking band.
2. Stop near the destination.
3. Route cue (forward, left, or right) when that side is walkable.
4. Slow down when every lane is blocked.
5. CARE direction otherwise.
