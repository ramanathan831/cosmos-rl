import ast
from pathlib import Path


PACKER = (
    Path(__file__).parents[1]
    / "cosmos_rl"
    / "dispatcher"
    / "data"
    / "packer"
    / "hf_vlm_data_packer.py"
)


def test_vlm_packer_resolves_fetch_video_dynamically():
    """Keep runtime cache wrappers on the training packer's live call path."""
    tree = ast.parse(PACKER.read_text(encoding="utf-8"))
    imported_by_value = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "qwen_vl_utils"
        and any(alias.name == "fetch_video" for alias in node.names)
        for node in ast.walk(tree)
    )
    dynamic_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fetch_video"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "vision_process"
    ]

    assert not imported_by_value
    assert dynamic_calls
