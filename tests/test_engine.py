from pathlib import Path

import pytest
from zenthrix import Engine, EngineUnavailableError
from zenthrix.runtime import RuntimeResult


class FakeRuntime:
    def generate(self, model_path, prompt, *, temperature, max_tokens):
        assert model_path.name == "model.zx"
        assert prompt == "hello"
        assert temperature == 0.2
        assert max_tokens == 4
        return RuntimeResult("generated", 10.0, 20.0)


def test_engine_validates_model_path(tmp_path: Path) -> None:
    model = tmp_path / "model.zx"
    model.write_bytes(b"placeholder")
    assert Engine(model).model_path == model


def test_engine_reports_missing_runtime(tmp_path: Path) -> None:
    model = tmp_path / "model.zx"
    model.write_bytes(b"placeholder")

    with pytest.raises(EngineUnavailableError, match="native engine"):
        Engine(model).generate("hello")


def test_engine_delegates_to_injected_runtime(tmp_path: Path) -> None:
    model = tmp_path / "model.zx"
    model.write_bytes(b"placeholder")

    result = Engine(model, runtime=FakeRuntime()).generate(
        "hello", max_tokens=4
    )

    assert result.text == "generated"
    assert result.ttft_ms == 10.0
    assert result.tokens_per_second == 20.0


def test_engine_rejects_runtime_selection_with_injected_runtime(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.zx"
    model.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="mutually exclusive"):
        Engine(model, runtime=FakeRuntime(), runtime_name="fake")


def test_engine_rejects_non_finite_temperature(tmp_path: Path) -> None:
    model = tmp_path / "model.zx"
    model.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="finite"):
        Engine(model).generate("hello", temperature=float("nan"))


def test_engine_rejects_non_compiled_model(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match=".zx compiled model"):
        Engine(model)
