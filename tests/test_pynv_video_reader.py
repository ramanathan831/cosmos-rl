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
    monkeypatch.setenv("TAO_PYNV_FRAME_TRANSFER", "host_rgb")
    vision = SimpleNamespace(
        VIDEO_READER_BACKENDS={},
        FORCE_QWENVL_VIDEO_READER=None,
        calculate_video_frame_range=lambda _element, total, _fps: (
            0,
            total - 1,
            total,
        ),
        smart_nframes=lambda _element, **_kwargs: 8,
        fetch_video=lambda *_args, **_kwargs: None,
        get_video_reader_backend=SimpleNamespace(cache_clear=lambda: None),
    )
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
            OutputColorType=SimpleNamespace(RGB="rgb", RGBP="rgbp"),
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
            raise AssertionError(f"decoder reuse across sources is forbidden: {path}")

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
        video_override_map=str(override_map),
        strict=True,
    )
    reader = vision.VIDEO_READER_BACKENDS["pynvvideocodec"]
    first, metadata, _sample_fps = reader({"video": "logical.mp4", "nframes": 8})
    reader({"video": str(ordinary), "nframes": 8})

    assert decoder_paths == [str(target), str(ordinary)]
    assert len(decoder_options) == 2
    assert decoder_options[0]["gpu_id"] == 0
    assert decoder_options[0]["cuda_context"] == 20
    assert decoder_options[0]["cuda_stream"] == 33
    assert all(option["decoder_cache_size"] == 1 for option in decoder_options)
    assert all(
        option["need_scanned_stream_metadata"] is False
        for option in decoder_options
    )
    assert tuple(first.shape) == (8, 3, 2, 2)
    assert metadata["video_backend"] == "pynvvideocodec"
    assert profile == {
        "backend": "pynvvideocodec",
        "version": "2.2.0",
        "cache_size": 0,
        "decoder_cache_size": 1,
        "frame_transfer": "host_rgb",
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
    register_pynv_video_reader(cache_size=4, strict=True)

    assert vision.FORCE_QWENVL_VIDEO_READER == "pynvvideocodec"
    assert pynv_video_reader.os.environ["FORCE_QWENVL_VIDEO_READER"] == "pynvvideocodec"
    assert pynv_video_reader.os.environ["TAO_PYNV_VIDEO_STRICT"] == "1"
    assert pynv_video_reader.os.environ["TAO_PYNV_VIDEO_CACHE_SIZE"] == "4"
    assert pynv_video_reader.os.environ["TAO_PYNV_DECODER_CACHE_SIZE"] == "1"
    assert pynv_video_reader.os.environ["TAO_PYNV_FRAME_TRANSFER"] == "host_rgb"
    with pytest.raises(RuntimeError, match="CPU video decoding fallback"):
        vision.VIDEO_READER_BACKENDS["torchvision"]({"video": str(video)})


def test_gpu_reader_rescans_overstated_container_frame_count(
    tmp_path, monkeypatch, capsys
):
    video = tmp_path / "overstated.mp4"
    video.write_bytes(b"video")
    decoder_options = []

    class Frame:
        shape = (2, 2, 3)
        strides = (6, 3, 1)

        def __init__(self):
            self.value = np.zeros((2, 2, 3), dtype=np.uint8).ctypes

        def framesize(self):
            return 12

        def GetPtrToPlane(self, _index):
            return self.value.data

    class Decoder:
        def __init__(self, _path, **kwargs):
            self.scanned = kwargs["need_scanned_stream_metadata"]
            decoder_options.append(kwargs)

        def get_stream_metadata(self):
            return SimpleNamespace(average_fps=30.0)

        def __len__(self):
            return 275 if self.scanned else 301

        def get_batch_frames_by_index(self, indices):
            if not self.scanned:
                raise IndexError("container metadata included undecodable tail frames")
            assert indices == [0, 39, 78, 117, 157, 196, 235, 274]
            return [Frame() for _ in indices]

        def stop(self):
            return None

    vision = _install_fake_runtime(monkeypatch, Decoder)
    register_pynv_video_reader(cache_size=0, strict=True)
    reader = vision.VIDEO_READER_BACKENDS["pynvvideocodec"]

    frames, metadata, _sample_fps = reader({"video": str(video), "nframes": 8})

    assert [
        option["need_scanned_stream_metadata"] for option in decoder_options
    ] == [False, True]
    assert tuple(frames.shape) == (8, 3, 2, 2)
    assert metadata["frames_indices"] == [0, 39, 78, 117, 157, 196, 235, 274]
    assert metadata["total_num_frames"] == 275
    output = capsys.readouterr().out
    assert "TAO_GPU_VIDEO_BATCH_RETRY" in output
    assert "unscanned_total_frames=301" in output
    assert "TAO_GPU_VIDEO_BATCH_RECOVERED" in output
    assert "scanned_total_frames=275" in output


def test_gpu_reader_device_rgbp_uses_dlpack_and_native_session_cache(
    tmp_path, monkeypatch
):
    first_video = tmp_path / "first.mp4"
    second_video = tmp_path / "second.mp4"
    first_video.write_bytes(b"video")
    second_video.write_bytes(b"video")
    decoder_options = []
    reconfigured = []

    class Frame:
        pass

    class Decoder:
        def __init__(self, _path, **kwargs):
            decoder_options.append(kwargs)

        def reconfigure_decoder(self, path):
            reconfigured.append(str(path))

        def get_stream_metadata(self):
            return SimpleNamespace(average_fps=30.0)

        def __len__(self):
            return 8

        def get_batch_frames_by_index(self, indices):
            return [Frame() for _ in indices]

        def stop(self):
            return None

    class DeviceTensor:
        device = SimpleNamespace(type="cuda")
        ndim = 3
        shape = (3, 2, 2)

    class DeviceBatch:
        def cpu(self):
            return torch.zeros((8, 3, 2, 2), dtype=torch.uint8)

    vision = _install_fake_runtime(monkeypatch, Decoder)
    monkeypatch.setenv("TAO_PYNV_FRAME_TRANSFER", "device_rgbp")
    monkeypatch.setattr(torch, "from_dlpack", lambda _frame: DeviceTensor())
    monkeypatch.setattr(torch, "stack", lambda _tensors: DeviceBatch())
    synchronized = []
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized.append)

    profile = register_pynv_video_reader(
        cache_size=17,
        decoder_cache_size=37,
        strict=True,
    )
    reader = vision.VIDEO_READER_BACKENDS["pynvvideocodec"]
    frames, _metadata, _sample_fps = reader(
        {"video": str(first_video), "nframes": 8}
    )
    reader({"video": str(second_video), "nframes": 8})

    assert tuple(frames.shape) == (8, 3, 2, 2)
    assert len(decoder_options) == 1
    assert decoder_options[0]["use_device_memory"] is True
    assert decoder_options[0]["output_color_type"] == "rgbp"
    assert decoder_options[0]["decoder_cache_size"] == 37
    assert reconfigured == [str(second_video)]
    assert synchronized == [0, 0]
    assert profile["frame_transfer"] == "device_rgbp"
    assert profile["decoder_cache_size"] == 37


def test_gpu_reader_rejects_unknown_frame_transfer(monkeypatch):
    monkeypatch.setenv("TAO_PYNV_FRAME_TRANSFER", "unknown")
    with pytest.raises(ValueError, match="host_rgb or device_rgbp"):
        register_pynv_video_reader(cache_size=0, strict=True)
