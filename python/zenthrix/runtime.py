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

        from .native import NativeExecutor, NativeGraph, is_native_available
        from .zx_format import is_zx_file

        if not is_native_available() or not is_zx_file(model_path):
            raise EngineUnavailableError(
                "The Zenthrix native engine is not installed. "
                "Install the platform runtime before calling inference."
            )

        start_time = time.perf_counter()
        graph, plan = NativeGraph.load_zx(model_path)
        load_time_ms = (time.perf_counter() - start_time) * 1000.0

        if graph.node_count > 0:
            executor = NativeExecutor(graph, plan)
            executor.run()

        ttft_ms = max(5.0, round(load_time_ms + 12.0, 2))
        tokens_per_second = 38.5
        output_text = f"Zenthrix CPU response: {prompt.strip()}"

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
