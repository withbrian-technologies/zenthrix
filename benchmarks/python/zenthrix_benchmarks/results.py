"""Benchmark result loading, validation, and comparison."""

import json
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any

from .exceptions import BenchmarkValidationError

METRICS = ("ttft_ms", "tokens_per_second", "peak_memory_mb", "load_time_ms")
METRIC_LABELS = {
    "ttft_ms": "TTFT (ms)",
    "tokens_per_second": "Tokens/s",
    "peak_memory_mb": "Peak memory (MB)",
    "load_time_ms": "Load time (ms)",
}
HIGHER_IS_BETTER = frozenset({"tokens_per_second"})
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Metadata identifying one reproducible benchmark run."""

    timestamp: str
    commit: str
    environment: str
    configuration: str

    @classmethod
    def from_mapping(cls, value: object) -> "RunMetadata":
        """Validate and construct run metadata from decoded JSON."""
        if not isinstance(value, dict):
            raise BenchmarkValidationError("metadata must be a JSON object")
        metadata = cls(
            _required_text(value, "timestamp"),
            _required_text(value, "commit"),
            _required_text(value, "environment"),
            _required_text(value, "configuration"),
        )
        try:
            datetime.fromisoformat(metadata.timestamp.replace("Z", "+00:00"))
        except ValueError as error:
            raise BenchmarkValidationError(
                "metadata timestamp must be ISO 8601"
            ) from error
        return metadata


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One measured implementation on one hardware/model configuration."""

    implementation: str
    hardware: str
    model: str
    metrics: dict[str, float]

    @classmethod
    def from_mapping(cls, value: object) -> "BenchmarkResult":
        """Validate and construct a result from decoded JSON."""
        if not isinstance(value, dict):
            raise BenchmarkValidationError("Each result must be a JSON object")
        implementation = _required_text(value, "implementation")
        hardware = _required_text(value, "hardware")
        model = _required_text(value, "model")
        raw_metrics = value.get("metrics")
        if not isinstance(raw_metrics, dict) or not raw_metrics:
            raise BenchmarkValidationError("metrics must be a non-empty object")
        metrics: dict[str, float] = {}
        for name, raw_value in raw_metrics.items():
            if name not in METRICS:
                raise BenchmarkValidationError(f"Unsupported metric: {name}")
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise BenchmarkValidationError(f"Metric {name} must be numeric")
            if not isfinite(float(raw_value)) or raw_value <= 0:
                raise BenchmarkValidationError(f"Metric {name} must be positive")
            metrics[name] = float(raw_value)
        return cls(implementation, hardware, model, metrics)


def load_results(path: str | Path) -> list[BenchmarkResult]:
    """Load and validate benchmark results from an array or run envelope."""
    return load_run(path)[1]


def load_run(path: str | Path) -> tuple[RunMetadata | None, list[BenchmarkResult]]:
    """Load results and optional reproducibility metadata."""
    result_path = Path(path)
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkValidationError(
            f"Unable to read benchmark results: {result_path}"
        ) from error
    metadata: RunMetadata | None = None
    if isinstance(payload, dict):
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise BenchmarkValidationError(
                f"schema_version must be {SCHEMA_VERSION}"
            )
        metadata = RunMetadata.from_mapping(payload.get("metadata"))
        payload = payload.get("results")
    if not isinstance(payload, list) or not payload:
        raise BenchmarkValidationError("Results must contain a non-empty array")
    results = [BenchmarkResult.from_mapping(item) for item in payload]
    _validate_consistency(results)
    return metadata, results


def render_summary(results: list[BenchmarkResult]) -> str:
    """Render a stable Markdown table from validated results."""
    ordered_results = sorted(
        results,
        key=lambda result: (
            result.implementation,
            result.hardware,
            result.model,
        ),
    )
    metric_names = sorted(
        {name for result in ordered_results for name in result.metrics},
        key=lambda name: METRICS.index(name),
    )
    best_values = {
        name: (
            max(
                result.metrics[name]
                for result in ordered_results
                if name in result.metrics
            )
            if name in HIGHER_IS_BETTER
            else min(
                result.metrics[name]
                for result in ordered_results
                if name in result.metrics
            )
        )
        for name in metric_names
    }
    headers = ["Implementation", "Hardware", "Model"] + [
        METRIC_LABELS[name] for name in metric_names
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for result in ordered_results:
        values = []
        for name in metric_names:
            value = result.metrics.get(name)
            if value is None:
                values.append("—")
            elif value == best_values[name]:
                values.append(f"**{value:g}**")
            else:
                values.append(f"{value:g}")
        lines.append(
            "| "
            + " | ".join(
                [result.implementation, result.hardware, result.model, *values]
            )
            + " |"
        )
    return "\n".join(lines)


def _required_text(value: dict[str, Any], key: str) -> str:
    raw_value = value.get(key)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise BenchmarkValidationError(f"{key} must be a non-empty string")
    return raw_value.strip()


def _validate_consistency(results: list[BenchmarkResult]) -> None:
    contexts = {(result.hardware, result.model) for result in results}
    if len(contexts) > 1:
        raise BenchmarkValidationError(
            "Results must use one hardware and model context per run"
        )
