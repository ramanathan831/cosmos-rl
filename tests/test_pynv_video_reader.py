import json
import sys
from types import SimpleNamespace

import numpy as np
import torch

from cosmos_rl.utils.pynv_video_reader import register_pynv_video_reader


def test_scanned_metadata_is_limited_to_override_targets(tmp_path, monkeypatch):
    target = tmp_path / "override.mp4"
    ordinary = tmp_path / "ordinary.mp4"
    target.write_bytes(b"video")
    ordinary.write_bytes(b"video")
    override_map = tmp_path / "overrides.json"
    override_map.write_text(json.dumps({"logical.mp4": str(target)}))
    options = []

    class Decoder:
        def __init__(self, _path, **kwargs):
            options.append(kwargs)

        def get_stream_metadata(self):
            return SimpleNamespace(average_fps=30.0)

        def __len__(self):
            return 8

        def get_batch_frames_by_index(self, indices):
            class Frame:
                shape = (2, 2, 3)
                strides = (6, 3, 1)

                def __init__(self):
                    self.value = np.zeros((2, 2, 3), dtype=np.uint8).ctypes

                def framesize(self):
                    return 12

                def GetPtrToPlane(self, _index):
                    return self.value.data

            return [Frame() for _ in indices]

    vision = SimpleNamespace(
        VIDEO_READER_BACKENDS={}, FORCE_QWENVL_VIDEO_READER=None,
        calculate_video_frame_range=lambda _element, total, _fps: (0, total - 1, total),
        smart_nframes=lambda _element, **_kwargs: 8,
        get_video_reader_backend=SimpleNamespace(cache_clear=lambda: None),
    )
    monkeypatch.setattr("ctypes.CDLL", lambda *_args: object())
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", SimpleNamespace(
        SimpleDecoder=Decoder, OutputColorType=SimpleNamespace(RGB="rgb"), __version__="2.2.0",
    ))
    monkeypatch.setitem(sys.modules, "qwen_vl_utils.vision_process", vision)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", SimpleNamespace(vision_process=vision))
    register_pynv_video_reader(video_override_map=str(override_map))
    reader = vision.VIDEO_READER_BACKENDS["pynvvideocodec"]
    reader({"video": "logical.mp4", "nframes": 8})
    reader({"video": str(ordinary), "nframes": 8})
    assert options[0]["need_scanned_stream_metadata"] is True
    assert options[1]["need_scanned_stream_metadata"] is False
