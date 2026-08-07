# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from cosmos_rl.tools.prewarm_sft_cache import (
    cache_path,
    combined_cache_fingerprint,
    entry_path,
    finalize_marker,
    finish_distributed_prewarm,
    wait_for_finalize_marker,
    write_finalize_marker,
)


def test_cache_key_is_deterministic_and_runtime_root_is_preserved(tmp_path) -> None:
    fingerprint = combined_cache_fingerprint("dataset-sha", "model-sha", "processor-sha")
    assert fingerprint == combined_cache_fingerprint("dataset-sha", "model-sha", "processor-sha")
    root = cache_path(tmp_path / "user-cache", "train", fingerprint)
    assert root == (tmp_path / "user-cache" / f"train-{fingerprint}").resolve()
    assert entry_path(root, 10001) == root / "1" / "10001.pt"


def test_different_dataset_or_model_never_reuses_cache() -> None:
    original = combined_cache_fingerprint("dataset-a", "model-a", "processor")
    assert combined_cache_fingerprint("dataset-b", "model-a", "processor") != original
    assert combined_cache_fingerprint("dataset-a", "model-b", "processor") != original


def test_distributed_prewarm_closes_group_before_rank_zero_hashing(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(
        "cosmos_rl.tools.prewarm_sft_cache.torch.distributed.barrier",
        lambda: events.append("barrier"),
    )
    monkeypatch.setattr(
        "cosmos_rl.tools.prewarm_sft_cache.torch.distributed.destroy_process_group",
        lambda: events.append("destroy"),
    )

    assert finish_distributed_prewarm(rank=0, world_size=8, initialized_here=True)
    assert events == ["barrier", "destroy"]


def test_nonzero_rank_does_not_write_manifest(monkeypatch) -> None:
    monkeypatch.setattr(
        "cosmos_rl.tools.prewarm_sft_cache.torch.distributed.barrier", lambda: None
    )
    monkeypatch.setattr(
        "cosmos_rl.tools.prewarm_sft_cache.torch.distributed.destroy_process_group",
        lambda: None,
    )

    assert not finish_distributed_prewarm(rank=3, world_size=8, initialized_here=True)


def test_finalize_marker_is_job_and_split_specific(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TAO_JOB_ID", "job-one")
    train = finalize_marker(tmp_path, "train")
    validation = finalize_marker(tmp_path, "val")
    monkeypatch.setenv("TAO_JOB_ID", "job-two")
    other_job = finalize_marker(tmp_path, "train")

    assert train != validation
    assert train != other_job


def test_nonzero_rank_waits_for_success_marker(tmp_path) -> None:
    marker = tmp_path / "finalize.json"
    write_finalize_marker(marker, status="success")
    wait_for_finalize_marker(marker, timeout_seconds=0.01)
    assert json.loads(marker.read_text())["status"] == "success"


def test_nonzero_rank_propagates_rank_zero_failure(tmp_path) -> None:
    marker = tmp_path / "finalize.json"
    write_finalize_marker(marker, status="failure", error="hash failed")
    with pytest.raises(RuntimeError, match="hash failed"):
        wait_for_finalize_marker(marker, timeout_seconds=0.01)


def test_nonzero_rank_finalize_wait_times_out(tmp_path, monkeypatch) -> None:
    marker = tmp_path / "missing.json"
    monkeypatch.setattr("cosmos_rl.tools.prewarm_sft_cache.time.sleep", lambda _: None)
    with pytest.raises(TimeoutError, match="Timed out"):
        wait_for_finalize_marker(marker, timeout_seconds=0.001)
