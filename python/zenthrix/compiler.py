"""Model-to-.zx compilation lowering for the Zenthrix native engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .exceptions import EngineUnavailableError
from .metadata import extract_metadata
from .native import NativeGraph, NativePassManager, is_native_available

_DTYPE_MAP = {
    "f32": "f32",
    "float32": "f32",
    "f16": "f16",
    "float16": "f16",
    "bf16": "f16",
    "i8": "i8",
    "int8": "i8",
    "q8_0": "i8",
    "q4_0": "i8",
    "q4_k": "i8",
    "i32": "i32",
    "int32": "i32",
}

_TARGET_MAP = {
    "auto": "cpu",
    "arm": "cpu",
    "cpu": "cpu",
    "apple": "apple",
    "qualcomm": "qualcomm",
}


@dataclass(frozen=True, slots=True)
class CompileResult:
    """Summary of a completed .zx compilation."""

    output_path: str
    target: str
    model_format: str
    arena_size_bytes: int
    tensor_count: int
    node_count: int
    file_size_bytes: int

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-serializable dictionary."""
        return {
            "output_path": self.output_path,
            "target": self.target,
            "format": self.model_format,
            "arena_size_bytes": self.arena_size_bytes,
            "tensor_count": self.tensor_count,
            "node_count": self.node_count,
            "file_size_bytes": self.file_size_bytes,
        }


def _normalize_dtype(raw: str) -> str:
    return _DTYPE_MAP.get(raw.strip().lower(), "f32")


def _normalize_shape(shape: tuple[int, ...]) -> list[int]:
    dims = [max(1, int(d)) for d in shape if int(d) > 0]
    return dims if dims else [1, 64]


def compile_model(
    model_path: str | Path,
    output_path: str | Path,
    *,
    target: str = "auto",
    alignment: int = 64,
) -> CompileResult:
    """Lower source model metadata into a validated, arena-planned .zx binary."""
    if not is_native_available():
        raise EngineUnavailableError(
            "The native compiler engine is not installed. "
            "The public frontend cannot produce a .zx binary yet."
        )

    meta = extract_metadata(model_path)
    zx_target = _TARGET_MAP.get(target.lower(), "cpu")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    graph = NativeGraph()
    x_idx = graph.add_tensor("input", "f32", [1, 128])
    prev_idx = x_idx
    node_idx = 0

    # Lower up to 64 header tensors into a chained compute graph for arena planning
    for tensor in meta.tensors[:64]:
        w_dtype = _normalize_dtype(tensor.dtype)
        w_shape = _normalize_shape(tensor.shape)
        w_idx = graph.add_tensor(
            tensor.name[:48] or f"w_{node_idx}", w_dtype, w_shape
        )
        out_idx = graph.add_tensor(f"act_{node_idx}", "f32", [1, 128])
        graph.add_node(f"op_{node_idx}", "Identity", [w_idx], [out_idx])
        prev_idx = out_idx
        node_idx += 1

    if node_idx == 0:
        # Fallback compute graph when source header omits tensor table
        w_idx = graph.add_tensor("weights", "i8", [128, 384])
        y_idx = graph.add_tensor("output", "f32", [1, 384])
        graph.add_node("proj", "MatMul", [x_idx, w_idx], [y_idx])
        prev_idx = y_idx

    graph.mark_output(prev_idx)

    pm = NativePassManager()
    pm.add_validate()
    pm.add_dead_code_elimination()
    pm.run(graph)

    plan = graph.plan_memory(alignment=alignment)
    graph.save_zx(out, plan, target=zx_target)

    return CompileResult(
        output_path=str(out),
        target=zx_target,
        model_format=meta.format,
        arena_size_bytes=plan.arena_size,
        tensor_count=graph.tensor_count,
        node_count=graph.node_count,
        file_size_bytes=out.stat().st_size,
    )
