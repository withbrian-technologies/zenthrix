from pathlib import Path

import pytest
from zenthrix.exceptions import InputValidationError
from zenthrix.validation import (
    validate_compile_options,
    validate_model_format,
    validate_model_input,
    validate_model_path,
)


def test_validate_model_format_normalizes_case() -> None:
    assert validate_model_format("ONNX") == "onnx"


def test_validate_model_format_rejects_unknown_format() -> None:
    with pytest.raises(InputValidationError, match="Unsupported model format"):
        validate_model_format("tflite")


def test_validate_model_path_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="not a file"):
        validate_model_path(tmp_path)


def test_validate_model_input_uses_declared_adapter(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"placeholder")
    assert validate_model_input(model, "GGUF") == model


def test_validate_model_input_rejects_mismatched_extension(tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"placeholder")
    with pytest.raises(InputValidationError, match="Expected a .gguf file"):
        validate_model_input(model, "gguf")


def test_validate_compile_options_rejects_unknown_target(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="Unsupported target"):
        validate_compile_options("cuda", None, tmp_path / "model.zx")


def test_validate_compile_options_accepts_supported_values(tmp_path: Path) -> None:
    output = tmp_path / "model.zx"
    assert validate_compile_options("QUALCOMM", "INT4", output) == output


# -- SafeTensors format support -----------------------------------------------


def test_validate_model_format_accepts_safetensors() -> None:
    assert validate_model_format("SAFETENSORS") == "safetensors"


def test_validate_model_input_validates_safetensors_extension(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"placeholder")
    assert validate_model_input(model, "safetensors") == model


def test_validate_model_input_rejects_mismatched_safetensors(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"placeholder")
    with pytest.raises(InputValidationError, match="Expected a .safetensors file"):
        validate_model_input(model, "safetensors")
