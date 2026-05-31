"""Performance helpers."""

import numpy as np

from navigation.config import Settings, apply_fast_profile
from navigation.utils.image_processing import resize_for_inference, upscale_class_map


def test_apply_fast_profile():
    s = apply_fast_profile(Settings())
    assert s.frame_width == 320
    assert s.process_every_n_frames == 3
    assert s.inference_imgsz == 256


def test_upscale_class_map():
    small = np.zeros((120, 160), dtype=np.int32)
    big = upscale_class_map(small, 240, 320)
    assert big.shape == (240, 320)


def test_resize_for_inference():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    s = Settings(inference_width=320, inference_height=240)
    out = resize_for_inference(frame, s)
    assert out.shape[:2] == (240, 320)
