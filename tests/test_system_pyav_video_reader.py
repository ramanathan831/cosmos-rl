from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading

import torch

from cosmos_rl.utils import system_pyav_video_reader as reader


def test_registration_replaces_qwen_torchvision_backend(monkeypatch):
    import qwen_vl_utils.vision_process as vision_process

    monkeypatch.setenv("FORCE_QWENVL_VIDEO_READER", "torchvision")
    reader.register_system_pyav_video_reader()
    assert (
        vision_process.VIDEO_READER_BACKENDS["torchvision"]
        is reader.read_video_system_pyav
    )


def test_repeated_parallel_reads_are_single_flight(monkeypatch):
    reader.clear_video_cache()
    monkeypatch.setattr(reader, "_CACHE_MAX_ITEMS", 8)
    calls = 0
    calls_lock = threading.Lock()

    def fake_decode(element):
        nonlocal calls
        with calls_lock:
            calls += 1
        return torch.zeros(8, 3, 2, 2), {"frames_indices": list(range(8))}, 1.0

    monkeypatch.setattr(reader, "_decode_sparse", fake_decode)
    element = {"video": "/tmp/repeated.mp4", "nframes": 8}
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda _: reader.read_video_system_pyav(element), range(16))
        )

    assert calls == 1
    assert all(result[0] is results[0][0] for result in results)


def test_cache_key_separates_sampling_configuration():
    assert reader._cache_key({"video": "a.mp4", "nframes": 8}) != reader._cache_key(
        {"video": "a.mp4", "nframes": 16}
    )


def test_distinct_cold_decodes_are_serialized(monkeypatch):
    reader.clear_video_cache()
    monkeypatch.setattr(reader, "_CACHE_MAX_ITEMS", 0)
    active = 0
    peak_active = 0
    calls_lock = threading.Lock()

    def fake_decode(element):
        nonlocal active, peak_active
        with calls_lock:
            active += 1
            peak_active = max(peak_active, active)
        threading.Event().wait(0.02)
        with calls_lock:
            active -= 1
        return torch.zeros(8, 3, 2, 2), {"frames_indices": list(range(8))}, 1.0

    monkeypatch.setattr(reader, "_decode_sparse", fake_decode)
    elements = [
        {"video": f"/tmp/video-{index}.mp4", "nframes": 8} for index in range(4)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(reader.read_video_system_pyav, elements))

    assert peak_active == 1
