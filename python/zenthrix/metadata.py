"""Model metadata extraction from source model headers.

Reads structural metadata (tensor names, shapes, dtypes, parameter counts)
from GGUF, ONNX, and SafeTensors files without loading weights into memory.
No third-party dependencies are required.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import InputValidationError

# -- Public data types --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TensorInfo:
    """Metadata about a single tensor/parameter."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Metadata extracted from a source model file."""

    format: str
    path: str
    file_size_bytes: int
    tensors: tuple[TensorInfo, ...] = ()
    parameter_count: int = 0
    estimated_memory_mb: float = 0.0
    architecture: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "format": self.format,
            "path": self.path,
            "file_size_bytes": self.file_size_bytes,
            "parameter_count": self.parameter_count,
            "estimated_memory_mb": round(self.estimated_memory_mb, 2),
            "architecture": self.architecture,
            "tensor_count": len(self.tensors),
            "extra": dict(self.extra),
        }


# -- Dtype size lookup --------------------------------------------------------

_DTYPE_SIZES: dict[str, int] = {
    # SafeTensors dtype strings
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    # GGUF type IDs (as strings)
    "f32": 4,
    "f16": 2,
    "q4_0": 1,  # approximate for quantized types
    "q4_1": 1,
    "q8_0": 1,
    "q8_1": 1,
}


def _element_size(dtype: str) -> int:
    """Return the byte size per element for a dtype string."""
    return _DTYPE_SIZES.get(dtype, _DTYPE_SIZES.get(dtype.upper(), 4))


def _count_parameters(shape: tuple[int, ...]) -> int:
    """Return the number of elements described by *shape*."""
    result = 1
    for dim in shape:
        result *= dim
    return result


# -- SafeTensors parser -------------------------------------------------------


def _parse_safetensors_header(path: Path) -> ModelMetadata:
    """Parse the SafeTensors JSON header to extract tensor metadata."""
    file_size = path.stat().st_size
    try:
        with open(path, "rb") as fh:
            header_len_bytes = fh.read(8)
            if len(header_len_bytes) < 8:
                raise InputValidationError(
                    f"SafeTensors file is too small to contain a header: {path}"
                )
            (header_len,) = struct.unpack("<Q", header_len_bytes)
            if header_len > file_size - 8 or header_len > 100_000_000:
                raise InputValidationError(
                    f"SafeTensors header length is invalid ({header_len}): {path}"
                )
            header_bytes = fh.read(header_len)
            if len(header_bytes) < header_len:
                raise InputValidationError(
                    f"SafeTensors header is truncated: {path}"
                )
    except OSError as exc:
        raise InputValidationError(
            f"Cannot read SafeTensors file: {path}: {exc}"
        ) from exc

    try:
        header: dict[str, Any] = json.loads(header_bytes)
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            f"SafeTensors header is not valid JSON: {path}: {exc}"
        ) from exc

    tensors: list[TensorInfo] = []
    total_params = 0
    total_bytes = 0

    for name, info in sorted(header.items()):
        if name == "__metadata__":
            continue
        dtype = info.get("dtype", "F32")
        shape = tuple(info.get("shape", ()))
        n_params = _count_parameters(shape)
        elem_size = _element_size(dtype)
        size_bytes = n_params * elem_size
        tensors.append(
            TensorInfo(name=name, shape=shape, dtype=dtype, size_bytes=size_bytes)
        )
        total_params += n_params
        total_bytes += size_bytes

    return ModelMetadata(
        format="safetensors",
        path=str(path),
        file_size_bytes=file_size,
        tensors=tuple(tensors),
        parameter_count=total_params,
        estimated_memory_mb=total_bytes / (1024 * 1024),
        extra=dict(header.get("__metadata__", {})),
    )


# -- GGUF parser --------------------------------------------------------------

_GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian

# GGUF metadata value type tags
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_UINT32 = 4


def _read_gguf_string(fh: Any) -> str:
    """Read a GGUF length-prefixed string."""
    (length,) = struct.unpack("<Q", fh.read(8))
    raw: bytes = fh.read(length)
    return raw.decode("utf-8", errors="replace")


def _skip_gguf_value(fh: Any, value_type: int) -> Any:
    """Read or skip a GGUF metadata value and return it if simple."""
    # Size per type tag (types 0-12)
    sizes = {
        0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4,
        6: 4, 7: 4, 8: None, 9: None, 10: 8, 11: 8, 12: 1,
    }
    if value_type == _GGUF_TYPE_STRING:
        return _read_gguf_string(fh)
    if value_type == 9:  # array
        (arr_type,) = struct.unpack("<I", fh.read(4))
        (arr_len,) = struct.unpack("<Q", fh.read(8))
        for _ in range(arr_len):
            _skip_gguf_value(fh, arr_type)
        return None
    size = sizes.get(value_type)
    if size is None:
        return None
    data = fh.read(size)
    if value_type == _GGUF_TYPE_UINT32 and len(data) == 4:
        return struct.unpack("<I", data)[0]
    return None


def _parse_gguf_header(path: Path) -> ModelMetadata:
    """Parse the GGUF binary header for model metadata."""
    file_size = path.stat().st_size
    try:
        with open(path, "rb") as fh:
            # Magic + version + tensor_count + metadata_kv_count
            header_bytes = fh.read(24)
            if len(header_bytes) < 24:
                raise InputValidationError(
                    f"GGUF file is too small to contain a valid header: {path}"
                )
            magic, version, tensor_count, kv_count = struct.unpack(
                "<IIQQ", header_bytes
            )
            if magic != _GGUF_MAGIC:
                raise InputValidationError(
                    f"Not a valid GGUF file (bad magic bytes): {path}"
                )

            architecture = ""
            extra: dict[str, str] = {"version": str(version)}

            # Read metadata key-value pairs
            for _ in range(min(kv_count, 512)):
                try:
                    key = _read_gguf_string(fh)
                    (value_type,) = struct.unpack("<I", fh.read(4))
                    value = _skip_gguf_value(fh, value_type)
                    if key == "general.architecture" and isinstance(value, str):
                        architecture = value
                    elif key == "general.name" and isinstance(value, str):
                        extra["name"] = value
                except Exception:
                    break

    except OSError as exc:
        raise InputValidationError(
            f"Cannot read GGUF file: {path}: {exc}"
        ) from exc

    return ModelMetadata(
        format="gguf",
        path=str(path),
        file_size_bytes=file_size,
        parameter_count=0,  # tensor details require deep parsing
        estimated_memory_mb=file_size / (1024 * 1024),
        architecture=architecture,
        extra=extra,
        tensors=(),
    )


# -- ONNX parser (lightweight, no onnx dependency) ----------------------------


def _parse_onnx_metadata(path: Path) -> ModelMetadata:
    """Extract basic ONNX metadata from the file header.

    This reads only the outer protobuf envelope to extract *ir_version*
    and *producer_name*.  Full graph introspection requires the ``onnx``
    package, which is not a dependency of zenthrix.
    """
    file_size = path.stat().st_size
    extra: dict[str, str] = {}

    try:
        with open(path, "rb") as fh:
            # ONNX files are protobuf ModelProto.  We read a few fields
            # from the outer message without a full protobuf parser.
            preview = fh.read(min(4096, file_size))

            # Field 1 (ir_version) is a varint with tag 0x08
            if len(preview) >= 3 and preview[0] == 0x08:
                ir_version = preview[1] & 0x7F
                extra["ir_version"] = str(ir_version)

            # Field 2 (opset_import) and field 3 (producer_name) are
            # length-delimited.  A minimal scan for printable ASCII
            # after field tag 0x1A (field 3, wire type 2).
            idx = preview.find(b"\x1a")
            if idx != -1 and idx + 2 < len(preview):
                str_len = preview[idx + 1]
                if 0 < str_len < 128 and idx + 2 + str_len <= len(preview):
                    candidate = preview[idx + 2 : idx + 2 + str_len]
                    try:
                        text = candidate.decode("ascii")
                        if text.isprintable():
                            extra["producer_name"] = text
                    except (UnicodeDecodeError, ValueError):
                        pass
    except OSError as exc:
        raise InputValidationError(
            f"Cannot read ONNX file: {path}: {exc}"
        ) from exc

    return ModelMetadata(
        format="onnx",
        path=str(path),
        file_size_bytes=file_size,
        estimated_memory_mb=file_size / (1024 * 1024),
        extra=extra,
    )


# -- Public API ---------------------------------------------------------------

_FORMAT_PARSERS: dict[str, Any] = {
    ".safetensors": _parse_safetensors_header,
    ".gguf": _parse_gguf_header,
    ".onnx": _parse_onnx_metadata,
}


def extract_metadata(path: str | Path) -> ModelMetadata:
    """Extract metadata from a source model file.

    Detects format by file extension and delegates to the appropriate
    parser.  Works without any third-party dependencies.

    Raises :class:`InputValidationError` if the file does not exist, is
    not a supported format, or has a corrupted header.
    """
    model_path = Path(path)
    if not model_path.exists():
        raise InputValidationError(
            f"Model path does not exist: {model_path}"
        )
    if not model_path.is_file():
        raise InputValidationError(
            f"Model path is not a file: {model_path}"
        )

    suffix = model_path.suffix.lower()
    parser: Callable[[Path], ModelMetadata] | None = _FORMAT_PARSERS.get(
        suffix
    )
    if parser is None:
        supported = ", ".join(sorted(_FORMAT_PARSERS.keys()))
        raise InputValidationError(
            f"Unsupported file extension '{suffix}' for metadata extraction. "
            f"Expected one of: {supported}"
        )
    return parser(model_path)
