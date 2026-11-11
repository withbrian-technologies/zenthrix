"""Tests for the .zx binary format parser, serializer, and CLI inspect."""

import json
from pathlib import Path

import pytest
from zenthrix.cli import main
from zenthrix.native import NativeGraph, is_native_available
from zenthrix.zx_format import is_zx_file, read_zx_info, write_zx_binary

native_required = pytest.mark.skipif(
    not is_native_available(),
    reason="Native libzenthrix library not built (build via CMake first)",
)


def test_is_zx_file(tmp_path: Path) -> None:
    non_existent = tmp_path / "does_not_exist.zx"
    assert is_zx_file(non_existent) is False

    empty_file = tmp_path / "empty.zx"
    empty_file.write_bytes(b"")
    assert is_zx_file(empty_file) is False

    placeholder = tmp_path / "placeholder.zx"
    placeholder.write_bytes(b"placeholder content")
    assert is_zx_file(placeholder) is False

    valid_file = tmp_path / "valid.zx"
    write_zx_binary(
        valid_file,
        target="cpu",
        tensors=[
            {"name": "x", "dtype": "f32", "shape": [1, 16], "offset": 0, "size": 64}
        ],
        nodes=[{"name": "op1", "operation": "Identity", "inputs": [], "outputs": [0]}],
        arena_size=64,
    )
    assert is_zx_file(valid_file) is True


def test_write_and_read_zx_metadata(tmp_path: Path) -> None:
    zx_path = tmp_path / "model.zx"
    tensors = [
        {
            "name": "weight",
            "dtype": "f32",
            "shape": [32, 64],
            "offset": 0,
            "size": 8192,
            "is_output": False,
        },
        {
            "name": "input",
            "dtype": "f32",
            "shape": [1, 32],
            "offset": 8192,
            "size": 128,
            "is_output": False,
        },
        {
            "name": "output",
            "dtype": "f32",
            "shape": [1, 64],
            "offset": 8320,
            "size": 256,
            "is_output": True,
        },
    ]
    nodes = [
        {
            "name": "gemm",
            "operation": "MatMul",
            "inputs": [1, 0],
            "outputs": [2],
        }
    ]

    write_zx_binary(
        zx_path,
        target="apple",
        tensors=tensors,
        nodes=nodes,
        arena_size=8576,
    )

    info = read_zx_info(zx_path)
    assert info.target_arch == "Apple Silicon (Metal/AMX)"
    assert info.tensor_count == 3
    assert info.node_count == 1
    assert info.arena_size_bytes == 8576
    assert info.parameter_count == (32 * 64 + 1 * 32 + 1 * 64)
    assert len(info.tensors) == 3
    assert info.tensors[0].name == "weight"
    assert info.tensors[0].shape == (32, 64)
    assert info.tensors[2].is_output is True
    assert info.nodes[0].name == "gemm"
    assert info.nodes[0].operation == "MatMul"


@native_required
def test_native_graph_save_load_roundtrip(tmp_path: Path) -> None:
    zx_path = tmp_path / "native_roundtrip.zx"

    g = NativeGraph()
    t0 = g.add_tensor("w", "f32", [4, 8])
    t1 = g.add_tensor("x", "f32", [1, 4])
    t2 = g.add_tensor("y", "f32", [1, 8])
    g.add_node("matmul", "MatMul", [t1, t0], [t2])
    g.mark_output(t2)
    assert g.validate()

    plan = g.plan_memory(64)
    g.save_zx(zx_path, plan, target="cpu")

    assert is_zx_file(zx_path)

    loaded_g, loaded_plan = NativeGraph.load_zx(zx_path)
    assert loaded_g.tensor_count == 3
    assert loaded_g.node_count == 1
    assert loaded_g.validate()
    assert loaded_plan.arena_size == plan.arena_size
    assert len(loaded_plan.allocations) == 3


def test_cli_inspect_zx_binary(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    zx_path = tmp_path / "cli_model.zx"
    write_zx_binary(
        zx_path,
        target="cpu",
        tensors=[
            {
                "name": "tensor_a",
                "dtype": "f32",
                "shape": [1, 4],
                "offset": 0,
                "size": 16,
            }
        ],
        nodes=[
            {
                "name": "node_a",
                "operation": "Identity",
                "inputs": [],
                "outputs": [0],
            }
        ],
        arena_size=64,
    )

    exit_code = main(["inspect", str(zx_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Zenthrix Executable (.zx v1.0)" in out
    assert "CPU Reference" in out
    assert "Tensors:        1" in out
    assert "Compute Nodes:  1" in out


def test_cli_inspect_zx_memory_profile(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    zx_path = tmp_path / "cli_mem.zx"
    write_zx_binary(
        zx_path,
        target="snapdragon",
        tensors=[
            {
                "name": "tensor_x",
                "dtype": "f32",
                "shape": [1, 128],
                "offset": 0,
                "size": 512,
            }
        ],
        nodes=[],
        arena_size=512,
    )

    exit_code = main(["inspect", str(zx_path), "--memory-profile"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Memory Arena Profile:" in out
    assert "tensor_x" in out
    assert "512" in out


def test_cli_inspect_zx_json(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    zx_path = tmp_path / "cli_json.zx"
    write_zx_binary(
        zx_path,
        target="apple",
        tensors=[
            {"name": "t0", "dtype": "f16", "shape": [2, 2], "offset": 0, "size": 8}
        ],
        nodes=[],
        arena_size=64,
    )

    exit_code = main(["inspect", str(zx_path), "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "zx"
    assert payload["tensor_count"] == 1
    assert payload["target"] == "Apple Silicon (Metal/AMX)"
    assert payload["tensors"][0]["name"] == "t0"
