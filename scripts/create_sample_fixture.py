"""Create tests/fixtures/sample.jpg for dry-run smoke tests."""

from pathlib import Path

import cv2
import numpy as np

out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample.jpg"
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), np.full((240, 320, 3), 128, dtype=np.uint8))
print(f"Wrote {out}")
