# Contributing

Use Python 3.10 or newer. With `uv`, install the development environment and
run the checks:

```bash
uv sync
uv run ruff check .
uv run mypy python
uv run pytest
```

Benchmark result files must be reproducible, identify the implementation,
hardware, and model, and contain only positive values for the supported
metrics. Do not commit private model weights, credentials, or unverified
performance claims.
