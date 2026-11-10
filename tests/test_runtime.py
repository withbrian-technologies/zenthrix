"""Tests for runtime plugin discovery."""

from pathlib import Path
from unittest.mock import patch

import pytest
from zenthrix.runtime import (
    RuntimeResult,
    UnavailableRuntime,
    default_runtime,
    discover_runtimes,
)


class FakeRuntime:
    """A minimal RuntimeAdapter implementation for testing."""

    def generate(
        self,
        model_path: Path,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> RuntimeResult:
        return RuntimeResult(text="fake", ttft_ms=1.0, tokens_per_second=50.0)


class FakeEntryPoint:
    """Mock importlib.metadata entry point."""

    def __init__(self, name: str, cls: type | None = None, *, broken: bool = False):
        self.name = name
        self._cls = cls
        self._broken = broken

    def load(self) -> type:
        if self._broken:
            raise ImportError("broken plugin")
        assert self._cls is not None
        return self._cls


class TestDiscoverRuntimes:
    def test_returns_empty_by_default(self):
        runtimes = discover_runtimes()
        assert runtimes == []

    def test_finds_registered_adapter(self):
        ep = FakeEntryPoint("fake", FakeRuntime)
        with patch(
            "zenthrix.runtime.importlib.metadata.entry_points",
            return_value={"zenthrix.runtimes": [ep]},
        ):
            runtimes = discover_runtimes()
        assert len(runtimes) == 1
        name, cls = runtimes[0]
        assert name == "fake"
        assert cls is FakeRuntime

    def test_skips_broken_entry_point(self):
        good = FakeEntryPoint("good", FakeRuntime)
        bad = FakeEntryPoint("bad", broken=True)
        with patch(
            "zenthrix.runtime.importlib.metadata.entry_points",
            return_value={"zenthrix.runtimes": [bad, good]},
        ):
            runtimes = discover_runtimes()
        assert len(runtimes) == 1
        assert runtimes[0][0] == "good"


class TestDefaultRuntime:
    def test_uses_discovered_adapter(self):
        ep = FakeEntryPoint("fake", FakeRuntime)
        with patch(
            "zenthrix.runtime.importlib.metadata.entry_points",
            return_value={"zenthrix.runtimes": [ep]},
        ):
            rt = default_runtime()
        assert isinstance(rt, FakeRuntime)

    def test_falls_back_to_unavailable(self):
        rt = default_runtime()
        assert isinstance(rt, UnavailableRuntime)

    def test_selects_named_runtime(self):
        first = FakeEntryPoint("first", FakeRuntime)
        second = FakeEntryPoint("second", FakeRuntime)
        with patch(
            "zenthrix.runtime.importlib.metadata.entry_points",
            return_value={"zenthrix.runtimes": [first, second]},
        ):
            rt = default_runtime("second")
        assert isinstance(rt, FakeRuntime)

    def test_rejects_unknown_runtime(self):
        with patch(
            "zenthrix.runtime.importlib.metadata.entry_points",
            return_value={"zenthrix.runtimes": []},
        ):
            with pytest.raises(RuntimeError, match="not installed"):
                default_runtime("missing")

    def test_unavailable_runtime_raises(self):
        rt = UnavailableRuntime()
        with pytest.raises(RuntimeError, match="native engine"):
            rt.generate(
                Path("model.zx"), "hello", temperature=0.2, max_tokens=128
            )
