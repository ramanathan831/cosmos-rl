import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "cosmos_rl"
    / "tools"
    / "custom_hooks"
    / "lifecycle_status.py"
)
SPEC = importlib.util.spec_from_file_location("tao_lifecycle_status_test", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
append_terminal_status = MODULE.append_terminal_status


@pytest.mark.parametrize("status", ["SUCCESS", "FAILURE"])
def test_append_terminal_status_is_final_and_durable(tmp_path, status):
    status_file = tmp_path / "status.json"
    status_file.write_text('{"status":"RUNNING"}\n', encoding="utf-8")

    append_terminal_status(str(status_file), status, "terminal message")

    records = [json.loads(line) for line in status_file.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["status"] == status
    assert records[-1]["message"] == "terminal message"


def test_append_terminal_status_rejects_nonterminal_state(tmp_path):
    with pytest.raises(ValueError, match="terminal TAO status"):
        append_terminal_status(str(tmp_path / "status.json"), "RUNNING", "not terminal")
