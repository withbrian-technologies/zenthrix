"""Tests for the native C API bridge."""

import pytest
from zenthrix.exceptions import InputValidationError
from zenthrix.native import (
    NativeGraph,
    NativePassManager,
    get_native_version,
    is_backend_supported,
    is_native_available,
)

native_required = pytest.mark.skipif(
    not is_native_available(),
    reason="Native libzenthrix library not built (build via CMake first)",
)


def test_native_availability_flag() -> None:
    # Always verifies discovery contract regardless of whether native library is built
    assert isinstance(is_native_available(), bool)


@native_required
def test_native_version() -> None:
    major, minor, patch = get_native_version()
    assert (major, minor, patch) == (0, 2, 3)


@native_required
def test_backend_support() -> None:
    assert is_backend_supported("cpu") is True
    assert is_backend_supported("cpu_reference") is True
    assert is_backend_supported("apple_metal") is False
    assert is_backend_supported("snapdragon") is False
    assert is_backend_supported("hexagon") is False

    with pytest.raises(InputValidationError, match="Unknown backend"):
        is_backend_supported("non_existent_accelerator")


@native_required
def test_native_graph_construction() -> None:
    g = NativeGraph()
    assert g.tensor_count == 0
    assert g.node_count == 0

    t0 = g.add_tensor("input", "f32", [1, 32])
    assert t0 == 0
    assert g.tensor_count == 1

    t1 = g.add_tensor("output", "f32", [1, 32])
    assert t1 == 1
    assert g.tensor_count == 2

    n0 = g.add_node("identity", "Identity", [t0], [t1])
    assert n0 == 0
    assert g.node_count == 1

    g.mark_output(t1)
    assert g.validate()


@native_required
def test_native_memory_planning() -> None:
    g = NativeGraph()
    t0 = g.add_tensor("x", "f32", [1, 64])  # 256 bytes
    t1 = g.add_tensor("y", "f32", [1, 64])  # 256 bytes
    g.add_node("op", "Identity", [t0], [t1])
    g.mark_output(t1)
    assert g.validate()

    plan = g.plan_memory(alignment=64)
    assert plan.arena_size >= 256
    assert len(plan.allocations) == 2

    alloc0, alloc1 = plan.allocations
    assert alloc0.size == 256
    assert alloc1.size == 256
    assert alloc0.offset % 64 == 0
    assert alloc1.offset % 64 == 0


@native_required
def test_native_memory_plan_invalid_alignment() -> None:
    g = NativeGraph()
    t0 = g.add_tensor("x", "f32", [1, 8])
    t1 = g.add_tensor("y", "f32", [1, 8])
    g.add_node("op", "Identity", [t0], [t1])
    g.mark_output(t1)

    # 0 is invalid alignment (must be power of two)
    with pytest.raises(InputValidationError):
        g.plan_memory(alignment=0)

    # 3 is not a power of two
    with pytest.raises(InputValidationError):
        g.plan_memory(alignment=3)


@native_required
def test_pass_manager_dead_code_elimination() -> None:
    g = NativeGraph()
    t_in = g.add_tensor("input", "f32", [1, 16])
    t_out = g.add_tensor("live_out", "f32", [1, 16])
    t_dead = g.add_tensor("dead_out", "f32", [1, 16])

    # Node 1: live
    g.add_node("live_node", "Identity", [t_in], [t_out])
    # Node 2: dead (produces tensor never used as output)
    g.add_node("dead_node", "Identity", [t_in], [t_dead])

    g.mark_output(t_out)
    assert g.node_count == 2

    pm = NativePassManager()
    pm.add_validate()
    pm.add_dead_code_elimination()

    success = pm.run(g)
    assert success is True
    # The dead node must have been removed!
    assert g.node_count == 1


@native_required
def test_invalid_tensor_arguments() -> None:
    g = NativeGraph()
    with pytest.raises(InputValidationError, match="Unsupported dtype"):
        g.add_tensor("bad_dtype", "unknown_dtype_xyz", [1, 2])

    with pytest.raises(InputValidationError, match="Tensor shape must not be empty"):
        g.add_tensor("bad_shape", "f32", [])


@native_required
def test_invalid_node_arguments() -> None:
    g = NativeGraph()
    t0 = g.add_tensor("x", "f32", [1, 4])
    with pytest.raises(InputValidationError, match="at least one output"):
        g.add_node("bad_node", "NoOp", [t0], [])


@native_required
def test_double_mark_output() -> None:
    g = NativeGraph()
    t0 = g.add_tensor("x", "f32", [1, 4])
    t1 = g.add_tensor("y", "f32", [1, 4])
    g.add_node("op", "Identity", [t0], [t1])
    g.mark_output(t1)

    with pytest.raises(InputValidationError, match="already marked"):
        g.mark_output(t1)
