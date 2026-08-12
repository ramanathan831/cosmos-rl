# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Qwen video reader backed by NVDEC/PyNvVideoCodec."""

from __future__ import annotations

import ctypes
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any


def register_pynv_video_reader(
    *,
    cache_size: int = 2,
    video_override_map: str | None = None,
) -> dict[str, Any]:
    """Register a repository-supported ``pynvvideocodec`` Qwen backend.

    Raises during preflight/initialization if NVDEC is unavailable; it never
    silently falls back to CPU decoding. Relative media resolution remains the
    dataset adapter's responsibility.
    """
    if cache_size < 0:
        raise ValueError("cache_size must be non-negative")
    try:
        ctypes.CDLL("libnvcuvid.so.1")
    except OSError as exc:
        raise RuntimeError("GPU video decoding requires readable libnvcuvid.so.1") from exc
    try:
        import numpy as np
        import PyNvVideoCodec as nvc
        import qwen_vl_utils.vision_process as vision_process
        import torch
    except ImportError as exc:
        raise RuntimeError("GPU video decoding requires PyNvVideoCodec and qwen-vl-utils") from exc

    overrides: dict[str, str] = {}
    if video_override_map:
        override_path = Path(video_override_map).expanduser().resolve(strict=True)
        value = json.loads(override_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
            raise ValueError("video_override_map must be a JSON object of string paths")
        overrides = value
    video_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
    decoder_lock = threading.RLock()

    def read_video_pynv(element):
        with decoder_lock:
            video_path = element["video"]
            if video_path.startswith("file://"):
                video_path = video_path[7:]
            if video_path.startswith(("http://", "https://")):
                raise ValueError("PyNvVideoCodec training requires compute-node-accessible local media")
            video_path = overrides.get(video_path, video_path)
            video_path = str(Path(video_path).expanduser().resolve(strict=True))
            key = (
                video_path,
                element.get("video_start"),
                element.get("video_end"),
                element.get("nframes"),
                element.get("fps"),
                element.get("min_frames"),
                element.get("max_frames"),
            )
            cached = video_cache.pop(key, None)
            if cached is not None:
                video_cache[key] = cached
                return cached

            import os

            gpu_id = int(os.environ.get("LOCAL_RANK", "0"))
            decoder = nvc.SimpleDecoder(
                video_path,
                gpu_id=gpu_id,
                use_device_memory=False,
                # Prepared override streams are deterministic H.264/MP4
                # artifacts whose header frame counts are validated against a
                # full metadata scan before use. Re-scanning those streams in
                # every data-loader process can deadlock PyNvVideoCodec under
                # high-rank cache prewarming, so runtime decoding trusts the
                # validated container metadata for both source and override
                # paths.
                need_scanned_stream_metadata=False,
                output_color_type=nvc.OutputColorType.RGB,
            )
            try:
                metadata = decoder.get_stream_metadata()
                source_total_frames = len(decoder)
                video_fps = float(metadata.average_fps)
                start_frame, end_frame, selected_total_frames = vision_process.calculate_video_frame_range(
                    element, source_total_frames, video_fps
                )
                nframes = vision_process.smart_nframes(
                    element, total_frames=selected_total_frames, video_fps=video_fps
                )
                indices = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
                arrays = []
                for frame in decoder.get_batch_frames_by_index(indices):
                    shape = tuple(int(value) for value in frame.shape)
                    strides = tuple(int(value) for value in frame.strides)
                    if len(shape) != 3 or shape[2] != 3:
                        raise RuntimeError(f"unexpected RGB frame layout {shape} for {video_path}")
                    raw = np.ctypeslib.as_array(
                        ctypes.cast(frame.GetPtrToPlane(0), ctypes.POINTER(ctypes.c_uint8)),
                        shape=(int(frame.framesize()),),
                    )
                    arrays.append(np.ndarray(shape=shape, dtype=np.uint8, buffer=raw, strides=strides).copy())
                video = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
            finally:
                del decoder
            sample_fps = nframes / max(selected_total_frames, 1.0) * video_fps
            result = (
                video,
                {
                    "fps": video_fps,
                    "frames_indices": indices,
                    "total_num_frames": selected_total_frames,
                    "video_backend": "pynvvideocodec",
                },
                sample_fps,
            )
            if cache_size:
                video_cache[key] = result
                while len(video_cache) > cache_size:
                    video_cache.popitem(last=False)
            return result

    vision_process.VIDEO_READER_BACKENDS["pynvvideocodec"] = read_video_pynv
    vision_process.FORCE_QWENVL_VIDEO_READER = "pynvvideocodec"
    vision_process.get_video_reader_backend.cache_clear()
    return {
        "backend": "pynvvideocodec",
        "version": getattr(nvc, "__version__", "unknown"),
        "cache_size": cache_size,
        "video_overrides": len(overrides),
    }
