# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Qwen video reader backed by NVDEC/PyNvVideoCodec."""

from __future__ import annotations

import atexit
import ctypes
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any


def register_pynv_video_reader(
    *,
    cache_size: int = 0,
    video_override_map: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Register a Qwen NVDEC reader and optionally forbid CPU fallback."""
    if cache_size < 0:
        raise ValueError("cache_size must be non-negative")
    # DataLoader workers use the spawn start method, so parent-process monkey
    # patches are not inherited.  Export the complete registration contract so
    # the baked qwen-vl-utils worker hook can recreate this reader in every
    # spawned worker before its first decode.
    os.environ["FORCE_QWENVL_VIDEO_READER"] = "pynvvideocodec"
    os.environ["TAO_PYNV_VIDEO_STRICT"] = "1" if strict else "0"
    os.environ["TAO_PYNV_VIDEO_CACHE_SIZE"] = str(cache_size)
    try:
        ctypes.CDLL("libnvcuvid.so.1")
    except OSError as exc:
        raise RuntimeError(
            "GPU video decoding requires readable libnvcuvid.so.1"
        ) from exc
    try:
        import numpy as np
        import PyNvVideoCodec as nvc
        import qwen_vl_utils.vision_process as vision_process
        import torch
        from cuda.bindings import driver as cuda_driver
    except ImportError as exc:
        raise RuntimeError(
            "GPU video decoding requires PyNvVideoCodec, cuda.bindings, and qwen-vl-utils"
        ) from exc

    overrides: dict[str, str] = {}
    if video_override_map:
        override_path = Path(video_override_map).expanduser().resolve(strict=True)
        value = json.loads(override_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in value.items()
        ):
            raise ValueError("video_override_map must be a JSON object of string paths")
        overrides = value
        os.environ["TAO_PYNV_VIDEO_OVERRIDE_MAP"] = str(override_path)
    else:
        os.environ.pop("TAO_PYNV_VIDEO_OVERRIDE_MAP", None)
    processed_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
    processed_inflight: dict[tuple[Any, ...], threading.Event] = {}
    cache_lock = threading.RLock()
    decoder_lock = threading.RLock()
    cuda_state: dict[str, Any] = {}
    decode_attested = False

    def release_decoder(decoder) -> None:
        """Best-effort release across PyNvVideoCodec wrapper variants."""
        try:
            decoder.stop()
        except AttributeError:
            # PyNvVideoCodec 2.2.0's Python wrapper exposes ``stop`` even when
            # the selected native SimpleDecoder implementation does not.
            pass

    def cuda_check(result, operation):
        error, *values = result
        if int(error) != 0:
            raise RuntimeError(f"{operation} failed: {error}")
        return values[0] if len(values) == 1 else tuple(values)

    def decoder_gpu_ordinal() -> int:
        """Return the process-visible GPU ordinal for this policy rank."""
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        visible = [
            token.strip()
            for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if token.strip()
        ]
        # A launcher may expose one physical GPU per policy process. CUDA then
        # remaps that device to process-local ordinal zero even when LOCAL_RANK
        # retains the node-global rank.
        return 0 if len(visible) == 1 else local_rank

    def cleanup_cuda_state() -> None:
        decoder = cuda_state.pop("decoder", None)
        cuda_state.pop("decoder_path", None)
        stream = cuda_state.pop("stream", None)
        device = cuda_state.pop("device", None)
        cuda_state.pop("context", None)
        if decoder is not None:
            release_decoder(decoder)
            del decoder
        if stream is not None:
            cuda_driver.cuStreamDestroy(stream)
        if device is not None:
            cuda_driver.cuCtxSetCurrent(None)
            cuda_driver.cuDevicePrimaryCtxRelease(device)

    def get_cuda_state() -> tuple[int, int, int]:
        """Retain one primary context and stream per spawned data worker."""
        if cuda_state:
            return (
                int(cuda_state["context"]),
                int(cuda_state["stream"]),
                int(cuda_state["gpu_id"]),
            )
        cuda_check(cuda_driver.cuInit(0), "cuInit")
        gpu_id = decoder_gpu_ordinal()
        device = cuda_check(cuda_driver.cuDeviceGet(gpu_id), "cuDeviceGet")
        context = cuda_check(
            cuda_driver.cuDevicePrimaryCtxRetain(device),
            "cuDevicePrimaryCtxRetain",
        )
        cuda_check(cuda_driver.cuCtxSetCurrent(context), "cuCtxSetCurrent")
        stream = cuda_check(cuda_driver.cuStreamCreate(0), "cuStreamCreate")
        cuda_state.update(
            device=device,
            context=context,
            stream=stream,
            gpu_id=gpu_id,
        )
        atexit.register(cleanup_cuda_state)
        return int(context), int(stream), gpu_id

    def make_decoder(
        video_path: str,
        cuda_context: int,
        cuda_stream: int,
        gpu_id: int,
    ):
        return nvc.SimpleDecoder(
            video_path,
            gpu_id=gpu_id,
            cuda_context=cuda_context,
            cuda_stream=cuda_stream,
            use_device_memory=False,
            # Runtime decoding trusts container metadata; source and
            # prepared override frame counts are validated separately.
            need_scanned_stream_metadata=False,
            decoder_cache_size=1,
            output_color_type=nvc.OutputColorType.RGB,
        )

    def copy_rgb_frame(frame, video_path: str):
        shape = tuple(int(value) for value in frame.shape)
        strides = tuple(int(value) for value in frame.strides)
        if len(shape) != 3 or shape[2] != 3:
            raise RuntimeError(f"unexpected RGB frame layout {shape} for {video_path}")
        raw = np.ctypeslib.as_array(
            ctypes.cast(frame.GetPtrToPlane(0), ctypes.POINTER(ctypes.c_uint8)),
            shape=(int(frame.framesize()),),
        )
        return np.ndarray(
            shape=shape, dtype=np.uint8, buffer=raw, strides=strides
        ).copy()

    def read_video_pynv(element):
        nonlocal decode_attested
        with decoder_lock:
            video_path = str(element["video"])
            if video_path.startswith("file://"):
                video_path = video_path[7:]
            if video_path.startswith(("http://", "https://")):
                raise ValueError(
                    "PyNvVideoCodec evaluation requires compute-node-local media"
                )
            video_path = overrides.get(video_path, video_path)
            video_path = str(Path(video_path).expanduser().resolve(strict=True))
            cuda_context, cuda_stream, gpu_id = get_cuda_state()
            cuda_check(
                cuda_driver.cuCtxSetCurrent(cuda_state["context"]), "cuCtxSetCurrent"
            )
            decoder = cuda_state.get("decoder")
            if decoder is not None and cuda_state["decoder_path"] != video_path:
                # PyNvVideoCodec 2.2.0 can return fewer frames than requested
                # after SimpleDecoder.reconfigure_decoder(), while its Python
                # wrapper assumes a one-to-one result and raises IndexError.
                # Keep the retained CUDA context/stream, but create a fresh
                # decoder session for each source path. This remains NVDEC-only
                # and preserves the exact requested uniform frame indices.
                cuda_state.pop("decoder", None)
                cuda_state.pop("decoder_path", None)
                release_decoder(decoder)
                del decoder
                decoder = None
            if decoder is None:
                decoder = make_decoder(
                    video_path, cuda_context, cuda_stream, gpu_id
                )
                cuda_state["decoder"] = decoder
                cuda_state["decoder_path"] = video_path

            metadata = decoder.get_stream_metadata()
            source_total_frames = len(decoder)
            video_fps = float(metadata.average_fps)
            start_frame, end_frame, selected_total_frames = (
                vision_process.calculate_video_frame_range(
                    element, source_total_frames, video_fps
                )
            )
            nframes = vision_process.smart_nframes(
                element, total_frames=selected_total_frames, video_fps=video_fps
            )
            indices = (
                torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
            )
            try:
                batch_frames = decoder.get_batch_frames_by_index(indices)
                if len(batch_frames) != len(indices):
                    raise RuntimeError(
                        f"NVDEC batch returned {len(batch_frames)} of {len(indices)} frames"
                    )
                arrays = [copy_rgb_frame(frame, video_path) for frame in batch_frames]
            except Exception as batch_error:
                # PyNvVideoCodec 2.2.0's native batch seek can transiently
                # return fewer frames than requested even when every exact
                # index decodes successfully. Preserve the same indices and
                # remain strictly GPU-only by reading every exact index from
                # one fresh NVDEC decoder session. Retrying the native batch
                # call can repeat the omission, while recreating a decoder for
                # every individual frame can crash PyNvVideoCodec 2.2.0 during
                # decoder teardown. Never route through Qwen's CPU fallback.
                cuda_state.pop("decoder", None)
                cuda_state.pop("decoder_path", None)
                release_decoder(decoder)
                del decoder
                print(
                    "TAO_GPU_VIDEO_BATCH_RETRY "
                    f"backend=pynvvideocodec video={video_path} "
                    f"indices={indices} error={type(batch_error).__name__}",
                    flush=True,
                )
                arrays = None
                last_error = batch_error
                for _attempt in range(3):
                    retry_decoder = make_decoder(
                        video_path, cuda_context, cuda_stream, gpu_id
                    )
                    try:
                        arrays = [
                            copy_rgb_frame(retry_decoder[index], video_path)
                            for index in indices
                        ]
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                    finally:
                        release_decoder(retry_decoder)
                        del retry_decoder
                if last_error is not None or arrays is None:
                    raise RuntimeError(
                        "GPU-only PyNvVideoCodec indexed retry failed for "
                        f"{video_path} indices {indices}"
                    ) from last_error
            video = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
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
            if not decode_attested:
                worker_info = torch.utils.data.get_worker_info()
                worker_id = worker_info.id if worker_info is not None else -1
                print(
                    "TAO_GPU_VIDEO_DECODE_ATTESTATION "
                    f"backend=pynvvideocodec pid={os.getpid()} "
                    f"local_rank={os.environ.get('LOCAL_RANK', '0')} "
                    f"worker_id={worker_id} gpu_ordinal={gpu_id} "
                    f"decoded_frames={len(indices)}",
                    flush=True,
                )
                decode_attested = True
            return result

    vision_process.VIDEO_READER_BACKENDS["pynvvideocodec"] = read_video_pynv

    original_fetch_video = getattr(
        vision_process,
        "_tao_pynv_original_fetch_video",
        vision_process.fetch_video,
    )
    vision_process._tao_pynv_original_fetch_video = original_fetch_video

    def fetch_video_pynv_cached(
        element,
        image_patch_size=14,
        return_video_sample_fps=False,
        return_video_metadata=False,
    ):
        """Cache the fully resized Qwen result, not full-resolution RGB frames."""
        key = tuple(
            element.get(name)
            for name in (
                "video",
                "video_start",
                "video_end",
                "nframes",
                "fps",
                "min_frames",
                "max_frames",
                "min_pixels",
                "max_pixels",
                "total_pixels",
                "resized_height",
                "resized_width",
            )
        ) + (
            image_patch_size,
            return_video_sample_fps,
            return_video_metadata,
        )
        while True:
            with cache_lock:
                cached = processed_cache.get(key)
                if cached is not None:
                    processed_cache.move_to_end(key)
                    return cached
                event = processed_inflight.get(key)
                if event is None:
                    event = threading.Event()
                    processed_inflight[key] = event
                    owner = True
                else:
                    owner = False
            if owner:
                break
            event.wait()

        try:
            result = original_fetch_video(
                element,
                image_patch_size=image_patch_size,
                return_video_sample_fps=return_video_sample_fps,
                return_video_metadata=return_video_metadata,
            )
            with cache_lock:
                if cache_size:
                    processed_cache[key] = result
                    processed_cache.move_to_end(key)
                    while len(processed_cache) > cache_size:
                        processed_cache.popitem(last=False)
            return result
        finally:
            with cache_lock:
                completed = processed_inflight.pop(key, None)
                if completed is not None:
                    completed.set()

    vision_process.fetch_video = fetch_video_pynv_cached
    if strict:

        def reject_cpu_fallback(_element):
            raise RuntimeError(
                "CPU video decoding fallback is disabled for this regression"
            )

        vision_process.VIDEO_READER_BACKENDS["torchvision"] = reject_cpu_fallback
    vision_process.FORCE_QWENVL_VIDEO_READER = "pynvvideocodec"
    vision_process.get_video_reader_backend.cache_clear()
    return {
        "backend": "pynvvideocodec",
        "version": getattr(nvc, "__version__", "unknown"),
        "cache_size": cache_size,
        "cache_boundary": "processed_fetch_video",
        "video_overrides": len(overrides),
        "strict": strict,
    }
