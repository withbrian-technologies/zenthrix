"""Public inference API and private-engine integration boundary."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from .runtime import RuntimeAdapter, default_runtime
from .validation import validate_compiled_model_path


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Result returned by a provisioned native runtime."""

    text: str
    ttft_ms: float
    tokens_per_second: float


class Engine:
    """Handle a compiled model through the optional native engine.

    The native engine is distributed separately from this public package. The
    frontend validates the model path and reports a clear installation error
    until that engine is provisioned.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        runtime: RuntimeAdapter | None = None,
        runtime_name: str | None = None,
    ) -> None:
        if runtime is not None and runtime_name is not None:
            raise ValueError("runtime and runtime_name are mutually exclusive")
        self.model_path = validate_compiled_model_path(model_path)
        self._runtime = (
            runtime
            if runtime is not None
            else default_runtime(runtime_name)
        )

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 256,
    ) -> InferenceResult:
        """Generate text using the optional native engine."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if not isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a finite non-negative number")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        result = self._runtime.generate(
            self.model_path,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return InferenceResult(
            text=result.text,
            ttft_ms=result.ttft_ms,
            tokens_per_second=result.tokens_per_second,
        )
