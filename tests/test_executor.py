"""Tests for the native execution engine and CPU reference runtime."""

import json
from pathlib import Path

import pytest
from zenthrix import Engine
from zenthrix.cli import main
from zenthrix.native import NativeExecutor, NativeGraph, is_native_available
from zenthrix.runtime import CpuReferenceRuntime, default_runtime
from zenthrix.zx_format import write_zx_binary

native_required = pytest.mark.skipif(
    not is_native_available(),
    reason="Native libzenthrix library not built (build via CMake first)",
)


@native_required
def test_native_executor_numerical_accuracy() -> None:
    # Graph: Input [1, 2] -> MatMul with Identity [2, 2] -> RMSNorm -> Output [1, 2]
    g = NativeGraph()
    x = g.add_tensor("x", "f32", [1, 2])
    w = g.add_tensor("w", "f32", [2, 2])
    m = g.add_tensor("m", "f32", [1, 2])
    gamma = g.add_tensor("gamma", "f32", [2])
    y = g.add_tensor("y", "f32", [1, 2])

    g.add_node("matmul", "MatMul", [x, w], [m])
    g.add_node("norm", "RMSNorm", [m, gamma], [y])
    g.mark_output(y)
    assert g.validate()

    plan = g.plan_memory(64)
    executor = NativeExecutor(g, plan)

    # Set data: X = [3.0, 4.0], W = [[1.0, 0.0], [0.0, 1.0]], Gamma = [1.0, 1.0]
    executor.set_tensor_data(x, [3.0, 4.0])
    executor.set_tensor_data(w, [1.0, 0.0, 0.0, 1.0])
    executor.set_tensor_data(gamma, [1.0, 1.0])

    assert executor.run() is True

    # Expected: rms = sqrt((9 + 16)/2 + 1e-5) ≈ 3.53553
    # y[0] = 3.0 / 3.53553 ≈ 0.8485, y[1] = 4.0 / 3.53553 ≈ 1.1314
    y_vals = executor.get_tensor_floats(y, 2)
    assert 0.84 < y_vals[0] < 0.85
    assert 1.13 < y_vals[1] < 1.14


@native_required
def test_native_executor_elementwise_silu_and_add() -> None:
    g = NativeGraph()
    a = g.add_tensor("a", "f32", [1, 2])
    b = g.add_tensor("b", "f32", [1, 2])
    sum_ab = g.add_tensor("sum", "f32", [1, 2])
    out = g.add_tensor("out", "f32", [1, 2])

    g.add_node("add", "Add", [a, b], [sum_ab])
    g.add_node("silu", "SiLU", [sum_ab], [out])
    g.mark_output(out)
    assert g.validate()

    plan = g.plan_memory(64)
    executor = NativeExecutor(g, plan)

    # a = [1.0, -1.0], b = [-1.0, 1.0] -> sum = [0.0, 0.0] -> silu = [0.0, 0.0]
    executor.set_tensor_data(a, [1.0, -1.0])
    executor.set_tensor_data(b, [-1.0, 1.0])

    assert executor.run() is True
    out_vals = executor.get_tensor_floats(out, 2)
    assert abs(out_vals[0]) < 1e-5
    assert abs(out_vals[1]) < 1e-5


@native_required
def test_cpu_reference_runtime_inference(tmp_path: Path) -> None:
    zx_path = tmp_path / "valid_model.zx"
    write_zx_binary(
        zx_path,
        target="cpu",
        tensors=[
            {"name": "x", "dtype": "f32", "shape": [1, 4], "offset": 0, "size": 16},
            {
                "name": "y",
                "dtype": "f32",
                "shape": [1, 4],
                "offset": 16,
                "size": 16,
                "is_output": True,
            },
        ],
        nodes=[{"name": "op", "operation": "Identity", "inputs": [0], "outputs": [1]}],
        arena_size=64,
    )

    rt = default_runtime("cpu_reference")
    assert isinstance(rt, CpuReferenceRuntime)

    engine = Engine(zx_path, runtime=rt)
    result = engine.generate("Explain quantum superposition.", max_tokens=64)

    assert "quantum superposition" in result.text
    assert result.ttft_ms > 0
    assert result.tokens_per_second > 0


@native_required
def test_cli_run_with_cpu_reference(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    zx_path = tmp_path / "run_model.zx"
    write_zx_binary(
        zx_path,
        target="cpu",
        tensors=[
            {"name": "x", "dtype": "f32", "shape": [1, 4], "offset": 0, "size": 16},
            {
                "name": "y",
                "dtype": "f32",
                "shape": [1, 4],
                "offset": 16,
                "size": 16,
                "is_output": True,
            },
        ],
        nodes=[{"name": "op", "operation": "Identity", "inputs": [0], "outputs": [1]}],
        arena_size=64,
    )

    exit_code = main(
        [
            "run",
            "--model",
            str(zx_path),
            "--prompt",
            "Synthesize transformer architecture.",
            "--runtime",
            "cpu_reference",
            "--json",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "text" in payload
    assert "transformer architecture" in payload["text"]
    assert payload["ttft_ms"] > 0
    assert payload["tokens_per_second"] > 0
