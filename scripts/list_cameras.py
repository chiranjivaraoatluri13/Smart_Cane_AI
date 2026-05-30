"""Probe which camera indices work on this machine."""

from __future__ import annotations

import sys

from navigation.capture.camera import probe_camera_indices


def main() -> int:
    print("Probing camera indices 0–3 (may take a few seconds)...")
    try:
        indices = probe_camera_indices(4)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if not indices:
        print("No working camera found.")
        print("Check: cable/built-in cam, Privacy → Camera, close Teams/Zoom.")
        print("Try in .env: CAMERA_BACKEND=DSHOW  or  CAMERA_INDEX=1")
        return 1
    print(f"Working indices: {indices}")
    print(f"Set in .env: CAMERA_INDEX={indices[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
