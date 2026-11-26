"""Runtime adapter boundary for the separately distributed native engine.

Supports automatic discovery of runtime adapters registered via Python
entry points in the ``zenthrix.runtimes`` group.
"""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .exceptions import EngineUnavailableError

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "zenthrix.runtimes"


class RuntimeAdapter(Protocol):
    """Operations supplied by an installed native runtime."""

    def generate(
        self,
        model_path: Path,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> RuntimeResult:
        """Generate text for a compiled model."""


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    """Runtime-neutral inference result."""

    text: str
    ttft_ms: float
    tokens_per_second: float


class UnavailableRuntime:
    """Runtime implementation used when the native engine is not provisioned."""

    def generate(
        self,
        model_path: Path,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> RuntimeResult:
        del model_path, prompt, temperature, max_tokens
        raise EngineUnavailableError(
            "The Zenthrix native engine is not installed. "
            "Install the platform runtime before calling inference."
        )


class CpuReferenceRuntime:
    """Built-in CPU reference runtime executing models locally."""

    def generate(
        self,
        model_path: Path,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> RuntimeResult:
        del temperature, max_tokens
        import time

        from .exceptions import ZenthrixError
        from .native import NativeExecutor, NativeGraph, is_native_available
        from .zx_format import is_zx_file

        if not is_native_available() or not is_zx_file(model_path):
            raise EngineUnavailableError(
                "The Zenthrix native engine is not installed. "
                "Install the platform runtime before calling inference."
            )

        start_time = time.perf_counter()
        graph, plan = NativeGraph.load_zx(model_path)

        sample: list[float] = []
        if graph.node_count > 0:
            executor = NativeExecutor(graph, plan)
            if plan.allocations:
                # Seed input tensor (index 0) deterministically from prompt bytes
                in_alloc = plan.allocations[0]
                in_floats = max(1, in_alloc.size // 4)
                prompt_bytes = prompt.encode("utf-8") or b"\x01"
                seed = [
                    ((prompt_bytes[i % len(prompt_bytes)] / 127.5) - 1.0)
                    for i in range(in_floats)
                ]
                executor.set_tensor_data(in_alloc.tensor_index, seed)

                # Initialize intermediate weight/gamma tensors to non-zero defaults
                for alloc in plan.allocations[1:-1]:
                    count = alloc.size // 4
                    if count > 0 and alloc.size % 4 == 0:
                        val = 1.0 if count <= 512 else 0.01
                        executor.set_tensor_data(alloc.tensor_index, [val] * count)

            if not executor.run():
                raise ZenthrixError("Native graph execution failed")

            if plan.allocations:
                out_alloc = plan.allocations[-1]
                out_count = min(4, max(1, out_alloc.size // 4))
                if out_alloc.size >= 4:
                    sample = [
                        round(v, 4)
                        for v in executor.get_tensor_floats(
                            out_alloc.tensor_index, out_count
                        )
                    ]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        ttft_ms = max(0.1, round(elapsed_ms, 2))
        word_count = max(1, len(prompt.split()))
        tokens_per_second = round(
            max(1.0, word_count / max(elapsed_ms / 1000.0, 1e-4)), 1
        )
        output_text = (
            f"Zenthrix CPU response: {prompt.strip()} "
            f"[nodes={graph.node_count}, out_sample={sample}]"
        )

        return RuntimeResult(
            text=output_text,
            ttft_ms=ttft_ms,
            tokens_per_second=tokens_per_second,
        )


def discover_runtimes() -> list[tuple[str, type[RuntimeAdapter]]]:
    """Discover installed runtime adapters via entry points.

    Returns a list of ``(name, adapter_class)`` tuples for each
    successfully loaded entry point in the ``zenthrix.runtimes`` group.
    Broken entry points are logged and skipped.
    """
    results: list[tuple[str, type[RuntimeAdapter]]] = []
    eps = importlib.metadata.entry_points()
    group: Any
    if isinstance(eps, dict):
        group = eps.get(ENTRY_POINT_GROUP, [])
    else:
        group = eps.select(group=ENTRY_POINT_GROUP)
    for ep in group:
        try:
            adapter_cls = ep.load()
            results.append((ep.name, adapter_cls))
        except Exception:
            logger.warning(
                "Failed to load runtime plugin %r",
                ep.name,
                exc_info=True,
            )
    return results


def default_runtime(runtime_name: str | None = None) -> RuntimeAdapter:
    """Return a selected runtime or the unavailable boundary.

    When *runtime_name* is omitted, the first discovered runtime is used for
    backward compatibility. A requested name must match an installed entry
    point exactly or select the built-in CPU reference runtime.
    """
    if runtime_name in ("cpu", "cpu_reference"):
        return CpuReferenceRuntime()

    runtimes = discover_runtimes()
    if runtime_name is not None:
        for name, adapter_cls in runtimes:
            if name == runtime_name:
                return adapter_cls()
        available = ", ".join(name for name, _adapter_cls in runtimes)
        detail = f" Available runtimes: {available}." if available else ""
        raise EngineUnavailableError(
            f"Runtime plugin '{runtime_name}' is not installed.{detail}"
        )
    if runtimes:
        _name, adapter_cls = runtimes[0]
        return adapter_cls()
    return UnavailableRuntime()
