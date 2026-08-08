# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1] / "cosmos_rl" / "utils" / "runtime_dependency_contract.py"
)
SPEC = importlib.util.spec_from_file_location("runtime_dependency_contract", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
missing_deepep_symbols = MODULE.missing_deepep_symbols
repair_vllm_conv3d_source = MODULE.repair_vllm_conv3d_source


OLD_SOURCE = """
from vllm.utils.torch_utils import is_torch_equal
if self.enable_linear and (is_torch_equal("2.9.0") or is_torch_equal("2.9.1")):
    return self._forward_mulmat(x)
"""


def test_repair_vllm_conv3d_source_is_idempotent() -> None:
    repaired, changed = repair_vllm_conv3d_source(OLD_SOURCE)
    assert changed
    assert "is_torch_equal_or_newer" in repaired
    assert 'is_torch_equal_or_newer("2.9.0")' in repaired
    assert repair_vllm_conv3d_source(repaired) == (repaired, False)


def test_repair_vllm_conv3d_rejects_unknown_source() -> None:
    with pytest.raises(RuntimeError, match="Unrecognized"):
        repair_vllm_conv3d_source("def forward_cuda(self, x): pass")


def test_deepep_symbol_contract_reports_only_missing_symbols() -> None:
    symbols = "\n".join(
        (
            "deep_ep::internode_ll::clean_mask_buffer(int*)",
            "deep_ep::internode_ll::update_mask_buffer(int*)",
        )
    )
    assert missing_deepep_symbols(symbols) == ["internode_ll::query_mask_buffer"]
