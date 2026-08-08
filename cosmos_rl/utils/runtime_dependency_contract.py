# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time compatibility checks for compiled Cosmos-RL dependencies."""

from __future__ import annotations

import argparse
import importlib
import inspect
from pathlib import Path
import subprocess

from packaging.version import Version


_OLD_CONV_IMPORT = "from vllm.utils.torch_utils import is_torch_equal"
_NEW_CONV_IMPORT = "from vllm.utils.torch_utils import is_torch_equal_or_newer"
_OLD_CONV_GUARD = (
    'if self.enable_linear and (is_torch_equal("2.9.0") or is_torch_equal("2.9.1")):'
)
_NEW_CONV_GUARD = 'if self.enable_linear and is_torch_equal_or_newer("2.9.0"):'
_DEEPEP_REQUIRED_SYMBOLS = (
    "internode_ll::clean_mask_buffer",
    "internode_ll::query_mask_buffer",
    "internode_ll::update_mask_buffer",
)


def repair_vllm_conv3d_source(source: str) -> tuple[str, bool]:
    """Return source with vLLM's PyTorch >=2.9 linear Conv3D guard."""
    if _NEW_CONV_GUARD in source and _NEW_CONV_IMPORT in source:
        return source, False
    if source.count(_OLD_CONV_IMPORT) != 1 or source.count(_OLD_CONV_GUARD) != 1:
        raise RuntimeError(
            "Unrecognized vLLM Conv3D implementation; refusing blind patch"
        )
    return (
        source.replace(_OLD_CONV_IMPORT, _NEW_CONV_IMPORT).replace(
            _OLD_CONV_GUARD, _NEW_CONV_GUARD
        ),
        True,
    )


def missing_deepep_symbols(symbols: str) -> list[str]:
    """Return required DeepEP internode symbols absent from ``nm -D -C`` output."""
    return [symbol for symbol in _DEEPEP_REQUIRED_SYMBOLS if symbol not in symbols]


def repair_vllm_conv3d() -> None:
    """Patch an installed vLLM tree and verify the live method uses the new guard."""
    import torch

    torch_version = Version(torch.__version__.split("+", 1)[0])
    if torch_version < Version("2.9"):
        print(
            f"vLLM Conv3D compatibility guard not required for PyTorch {torch_version}"
        )
        return
    conv = importlib.import_module("vllm.model_executor.layers.conv")
    path = Path(conv.__file__)
    source, changed = repair_vllm_conv3d_source(path.read_text(encoding="utf-8"))
    if changed:
        path.write_text(source, encoding="utf-8")
        importlib.invalidate_caches()
        conv = importlib.reload(conv)
    verify_vllm_conv3d()


def verify_vllm_conv3d() -> None:
    """Verify installed vLLM cannot regress PyTorch >=2.9 to cuDNN Conv3D."""
    import torch

    torch_version = Version(torch.__version__.split("+", 1)[0])
    if torch_version < Version("2.9"):
        print(
            f"vLLM Conv3D compatibility guard not required for PyTorch {torch_version}"
        )
        return
    conv = importlib.import_module("vllm.model_executor.layers.conv")
    live_source = inspect.getsource(conv.Conv3dLayer.forward_cuda)
    if _NEW_CONV_GUARD not in live_source:
        raise RuntimeError("vLLM Conv3D equal-or-newer guard verification failed")
    print(f"vLLM Conv3D compatibility guard ready: {conv.__file__}")


def verify_deepep() -> None:
    """Fail when DeepEP's Python bindings and compiled extension disagree."""
    importlib.import_module("deep_ep")
    extension = importlib.import_module("deep_ep_cpp")
    extension_path = Path(extension.__file__)
    result = subprocess.run(
        ["nm", "-D", "-C", str(extension_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    missing = missing_deepep_symbols(result.stdout)
    if missing:
        raise RuntimeError(
            "DeepEP extension is missing required internode symbols: "
            + ", ".join(missing)
        )
    print(f"DeepEP Python/extension contract ready: {extension_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-vllm-conv3d", action="store_true")
    parser.add_argument("--verify-vllm-conv3d", action="store_true")
    parser.add_argument("--verify-deepep", action="store_true")
    args = parser.parse_args()
    if not any((args.repair_vllm_conv3d, args.verify_vllm_conv3d, args.verify_deepep)):
        parser.error("select at least one runtime dependency check")
    if args.repair_vllm_conv3d:
        repair_vllm_conv3d()
    elif args.verify_vllm_conv3d:
        verify_vllm_conv3d()
    if args.verify_deepep:
        verify_deepep()


if __name__ == "__main__":
    main()
