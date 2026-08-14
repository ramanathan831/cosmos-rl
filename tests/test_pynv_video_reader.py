import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "cosmos_rl" / "utils" / "pynv_video_reader.py"
SPEC = importlib.util.spec_from_file_location("pynv_video_reader", SCRIPT)
assert SPEC and SPEC.loader
pynv_video_reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pynv_video_reader)
register_pynv_video_reader = pynv_video_reader.register_pynv_video_reader


def _install_fake_runtime(monkeypatch, decoder_type):
    vision = SimpleNamespace(
        VIDEO_READER_BACKENDS={},
        FORCE_QWENVL_VIDEO_READER=None,
        calculate_video_frame_range=lambda _element, total, _fps: (
            0,
            total - 1,
            total,
        ),
        smart_nframes=lambda _element, **_kwargs: 8,
        get_video_reader_backend=SimpleNamespace(cache_clear=lambda: None),
    )
    vision.fetch_video = lambda element, **_kwargs: vision.VIDEO_READER_BACKENDS[
        vision.FORCE_QWENVL_VIDEO_READER
    ](element)
    driver = SimpleNamespace(
        cuInit=lambda _flags: (0,),
        cuDeviceGet=lambda ordinal: (0, ordinal + 10),
        cuDevicePrimaryCtxRetain=lambda device: (0, device + 10),
        cuCtxSetCurrent=lambda _context: (0,),
        cuStreamCreate=lambda _flags: (0, 33),
        cuStreamDestroy=lambda _stream: (0,),
        cuDevicePrimaryCtxRelease=lambda _device: (0,),
    )
    cuda_module = ModuleType("cuda")
    bindings_module = ModuleType("cuda.bindings")
    bindings_module.driver = driver
    cuda_module.bindings = bindings_module

    monkeypatch.setattr("ctypes.CDLL", lambda *_args: object())
    monkeypatch.setattr(
        pynv_video_reader.atexit,
        "register",
        lambda _callback: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "PyNvVideoCodec",
        SimpleNamespace(
            SimpleDecoder=decoder_type,
            OutputColorType=SimpleNamespace(RGB="rgb"),
            __version__="2.2.0",
        ),
    )
    monkeypatch.setitem(sys.modules, "qwen_vl_utils.vision_process", vision)
    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        SimpleNamespace(vision_process=vision),
    )
    monkeypatch.setitem(sys.modules, "cuda", cuda_module)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings_module)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return vision


def test_gpu_reader_reuses_context_stream_and_decoder(tmp_path, monkeypatch, capsys):
    target = tmp_path / "override.mp4"
    ordinary = tmp_path / "ordinary.mp4"
    target.write_bytes(b"video")
    ordinary.write_bytes(b"video")
    override_map = tmp_path / "overrides.json"
    override_map.write_text(
        json.dumps({"logical.mp4": str(target)}),
        encoding="utf-8",
    )
    decoder_paths = []
    decoder_options = []

    class Decoder:
        def __init__(self, path, **kwargs):
            decoder_paths.append(str(path))
            decoder_options.append(kwargs)

        def reconfigure_decoder(self, path):
            decoder_paths.append(str(path))

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

        def stop(self):
            return None

    vision = _install_fake_runtime(monkeypatch, Decoder)
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")

    profile = register_pynv_video_reader(
        cache_size=0,
        decoder_cache_size=37,
        video_override_map=str(override_map),
        strict=True,
    )
    reader = vision.VIDEO_READER_BACKENDS["pynvvideocodec"]
    first, metadata, _sample_fps = reader({"video": "logical.mp4", "nframes": 8})
    reader({"video": str(ordinary), "nframes": 8})

    assert decoder_paths == [str(target), str(ordinary)]
    assert len(decoder_options) == 1
    assert decoder_options[0]["gpu_id"] == 0
    assert decoder_options[0]["cuda_context"] == 20
    assert decoder_options[0]["cuda_stream"] == 33
    assert decoder_options[0]["decoder_cache_size"] == 37
    assert decoder_options[0]["need_scanned_stream_metadata"] is False
    assert tuple(first.shape) == (8, 3, 2, 2)
    assert metadata["video_backend"] == "pynvvideocodec"
    assert profile == {
        "backend": "pynvvideocodec",
        "version": "2.2.0",
        "cache_size": 0,
        "decoder_cache_size": 37,
        "cache_boundary": "processed_fetch_video",
        "video_overrides": 1,
        "strict": True,
    }
    assert capsys.readouterr().out.count("TAO_GPU_VIDEO_DECODE_ATTESTATION") == 1


def test_gpu_reader_exports_spawn_worker_contract(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")

    class Decoder:
        def __init__(self, _path, **_kwargs):
            pass

        def stop(self):
            pass

    vision = _install_fake_runtime(monkeypatch, Decoder)
    register_pynv_video_reader(
        cache_size=4,
        decoder_cache_size=19,
        strict=True,
    )

    assert vision.FORCE_QWENVL_VIDEO_READER == "pynvvideocodec"
    assert pynv_video_reader.os.environ["FORCE_QWENVL_VIDEO_READER"] == "pynvvideocodec"
    assert pynv_video_reader.os.environ["TAO_PYNV_VIDEO_STRICT"] == "1"
    assert pynv_video_reader.os.environ["TAO_PYNV_VIDEO_CACHE_SIZE"] == "4"
    assert pynv_video_reader.os.environ["TAO_PYNV_DECODER_CACHE_SIZE"] == "19"
    with pytest.raises(RuntimeError, match="CPU video decoding fallback"):
        vision.VIDEO_READER_BACKENDS["torchvision"]({"video": str(video)})


def test_gpu_reader_caches_processed_fetch_result(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    decode_calls = 0

    class Decoder:
        def __init__(self, _path, **_kwargs):
            pass

        def get_stream_metadata(self):
            return SimpleNamespace(average_fps=30.0)

        def __len__(self):
            return 8

        def get_batch_frames_by_index(self, indices):
            nonlocal decode_calls
            decode_calls += 1

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

        def stop(self):
            pass

    vision = _install_fake_runtime(monkeypatch, Decoder)
    profile = register_pynv_video_reader(cache_size=2, strict=True)
    element = {"video": str(video), "nframes": 8, "max_pixels": 81920}
    first = vision.fetch_video(element, return_video_metadata=True)
    second = vision.fetch_video(element, return_video_metadata=True)

    assert first is second
    assert decode_calls == 1
    assert profile["cache_boundary"] == "processed_fetch_video"
