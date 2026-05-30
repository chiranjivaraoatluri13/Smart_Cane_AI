"""Live camera capture via OpenCV."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from navigation.config import Settings


def _opencv_backends() -> list[int | None]:
    """Preferred capture APIs per platform (``None`` = OpenCV default)."""
    import cv2

    if sys.platform == "win32":
        # MSMF often fails (-1072875772); DirectShow is more reliable on Windows laptops.
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
    return [None]


def open_video_capture(
    index: int,
    *,
    width: int | None = None,
    height: int | None = None,
    backend: int | None = None,
) -> object:
    """Open a webcam; tries alternate backends on Windows when needed."""
    import cv2

    backends: list[int | None]
    if backend is not None:
        backends = [backend]
    else:
        backends = _opencv_backends()

    last_error: str | None = None
    for api in backends:
        cap = (
            cv2.VideoCapture(index, api)
            if api is not None
            else cv2.VideoCapture(index)
        )
        if not cap.isOpened():
            cap.release()
            last_error = f"index {index}" + (f" api={api}" if api is not None else "")
            continue

        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Warm-up: some drivers fail the first grabs.
        for _ in range(5):
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
        last_error = f"index {index} opened but could not read frames (api={api})"

    raise RuntimeError(
        f"Could not open camera {last_error}. "
        "Try: close other apps using the camera, Settings → Privacy → Camera, "
        "or run: python scripts\\list_cameras.py"
    )


def probe_camera_indices(max_index: int = 4) -> list[int]:
    """Return indices that open and return at least one frame."""
    found: list[int] = []
    for i in range(max_index):
        try:
            cap = open_video_capture(i)
        except RuntimeError:
            continue
        import cv2

        ok, _ = cap.read()
        cap.release()
        if ok:
            found.append(i)
    return found


@dataclass
class CameraStream:
    """Context-managed webcam reader."""

    settings: Settings
    _cap: object | None = None

    def open(self) -> None:
        backend: int | None = None
        if self.settings.camera_backend.strip().lower() not in ("", "auto"):
            import cv2

            name = self.settings.camera_backend.strip().upper()
            backend = getattr(cv2, f"CAP_{name}", None)
            if backend is None:
                raise ValueError(
                    f"Unknown CAMERA_BACKEND={self.settings.camera_backend!r}. "
                    "Use auto, DSHOW, or MSMF."
                )

        self._cap = open_video_capture(
            self.settings.camera_index,
            width=self.settings.frame_width,
            height=self.settings.frame_height,
            backend=backend,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> CameraStream:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def frames(self) -> Iterator[np.ndarray]:
        import cv2

        if self._cap is None:
            raise RuntimeError("Camera not opened")
        consecutive_failures = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    raise RuntimeError(
                        "Camera stopped delivering frames. "
                        "Another app may have taken the device, or try "
                        "CAMERA_INDEX=1 or CAMERA_BACKEND=DSHOW in .env"
                    )
                cv2.waitKey(50)
                continue
            consecutive_failures = 0
            yield frame
            delay_ms = max(1, int(1000 / max(1, self.settings.target_fps)))
            cv2.waitKey(delay_ms)


def load_image(path: str) -> np.ndarray:
    import cv2

    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return frame
