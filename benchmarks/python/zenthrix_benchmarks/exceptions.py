"""Errors raised by the benchmark tooling."""


class BenchmarkError(Exception):
    """Base class for expected benchmark errors."""


class BenchmarkValidationError(BenchmarkError, ValueError):
    """Raised when a benchmark result file is invalid."""

