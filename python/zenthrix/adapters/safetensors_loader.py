"""SafeTensors input validation."""

from pathlib import Path

from ..exceptions import InputValidationError
from ..validation import validate_model_path


def validate_safetensors(path: str | Path) -> Path:
    """Validate a SafeTensors model path."""
    model_path = validate_model_path(path)
    if model_path.suffix.lower() != ".safetensors":
        raise InputValidationError(
            f"Expected a .safetensors file, got: {model_path}"
        )
    return model_path
