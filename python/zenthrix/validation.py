"""Validation helpers for model inputs and compiler options."""

from pathlib import Path

from .config import SUPPORTED_QUANTIZATIONS, SUPPORTED_TARGETS
from .exceptions import InputValidationError

SUPPORTED_FORMATS = frozenset({"onnx", "pytorch", "gguf", "safetensors"})


def validate_model_path(path: str | Path) -> Path:
    """Return an existing regular model path or raise a useful error."""
    model_path = Path(path)
    if not model_path.exists():
        raise InputValidationError(f"Model path does not exist: {model_path}")
    if not model_path.is_file():
        raise InputValidationError(f"Model path is not a file: {model_path}")
    return model_path


def validate_compiled_model_path(path: str | Path) -> Path:
    """Validate an existing Zenthrix compiled model artifact."""
    model_path = validate_model_path(path)
    if model_path.suffix.lower() != ".zx":
        raise InputValidationError(
            f"Expected a .zx compiled model, got: {model_path}"
        )
    return model_path


def validate_model_format(model_format: str) -> str:
    """Normalize and validate a supported model format."""
    normalized = model_format.lower()
    if normalized not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise InputValidationError(
            f"Unsupported model format '{model_format}'. Expected one of: {supported}"
        )
    return normalized


def validate_model_input(path: str | Path, model_format: str) -> Path:
    """Validate a model path using the adapter for its declared format."""
    from .adapters.gguf_loader import validate_gguf
    from .adapters.onnx_loader import validate_onnx
    from .adapters.pytorch_loader import validate_pytorch_export
    from .adapters.safetensors_loader import validate_safetensors

    normalized = validate_model_format(model_format)
    validators = {
        "gguf": validate_gguf,
        "onnx": validate_onnx,
        "pytorch": validate_pytorch_export,
        "safetensors": validate_safetensors,
    }
    return validators[normalized](path)


def validate_compile_options(
    target: str, quantization: str | None, output: str | Path
) -> Path:
    """Validate compiler options and return the requested output path."""
    normalized_target = target.lower()
    if normalized_target not in SUPPORTED_TARGETS:
        supported = ", ".join(sorted(SUPPORTED_TARGETS))
        raise InputValidationError(
            f"Unsupported target '{target}'. Expected one of: {supported}"
        )
    if quantization is not None and quantization.lower() not in SUPPORTED_QUANTIZATIONS:
        supported = ", ".join(sorted(SUPPORTED_QUANTIZATIONS))
        raise InputValidationError(
            f"Unsupported quantization '{quantization}'. Expected one of: {supported}"
        )
    output_path = Path(output)
    if output_path.suffix.lower() != ".zx":
        raise InputValidationError(
            f"Output path must use the .zx extension: {output_path}"
        )
    if not output_path.parent.exists():
        raise InputValidationError(
            f"Output directory does not exist: {output_path.parent}"
        )
    return output_path
