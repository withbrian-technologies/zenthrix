"""Zenthrix binary executable (.zx) file container format parser and serializer."""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import InputValidationError

ZX_MAGIC = b"ZX01"
ZX_HEADER_SIZE = 64

TARGET_NAMES = {
    0: "CPU Reference",
    1: "Apple Silicon (Metal/AMX)",
    2: "Qualcomm Snapdragon (Hexagon NPU)",
}

TARGET_CODES = {
    "auto": 0,
    "cpu": 0,
    "cpu_reference": 0,
    "apple": 1,
    "apple_metal": 1,
    "snapdragon": 2,
    "hexagon": 2,
    "qualcomm_hexagon": 2,
}

DTYPE_NAMES = {
    0: "f32",
    1: "f16",
    2: "i8",
    3: "i32",
}

DTYPE_CODES = {
    "f32": 0,
    "float32": 0,
    "f16": 1,
    "float16": 1,
    "i8": 2,
    "int8": 2,
    "i32": 3,
    "int32": 3,
}


@dataclass(frozen=True, slots=True)
class ZxTensorRecord:
    """Descriptor of one compiled tensor stored in the .zx binary."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    arena_offset: int
    arena_size: int
    is_output: bool


@dataclass(frozen=True, slots=True)
class ZxNodeRecord:
    """Descriptor of one compute node stored in the .zx binary."""

    name: str
    operation: str
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ZxModelInfo:
    """Structural metadata extracted from a compiled .zx binary."""

    path: str
    file_size_bytes: int
    target_arch: str
    parameter_count: int
    tensor_count: int
    node_count: int
    arena_size_bytes: int
    tensors: tuple[ZxTensorRecord, ...]
    nodes: tuple[ZxNodeRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert into a JSON-compatible dictionary."""
        return {
            "format": "zx",
            "path": self.path,
            "file_size_bytes": self.file_size_bytes,
            "target": self.target_arch,
            "parameter_count": self.parameter_count,
            "tensor_count": self.tensor_count,
            "node_count": self.node_count,
            "arena_size_bytes": self.arena_size_bytes,
            "tensors": [
                {
                    "name": t.name,
                    "dtype": t.dtype,
                    "shape": list(t.shape),
                    "offset": t.arena_offset,
                    "size": t.arena_size,
                    "is_output": t.is_output,
                }
                for t in self.tensors
            ],
            "nodes": [
                {
                    "name": n.name,
                    "operation": n.operation,
                    "inputs": list(n.inputs),
                    "outputs": list(n.outputs),
                }
                for n in self.nodes
            ],
        }


def is_zx_file(path: str | Path) -> bool:
    """Return True if the file exists and begins with the .zx magic bytes ZX01."""
    try:
        p = Path(path)
        if not p.is_file() or p.stat().st_size < 4:
            return False
        with open(p, "rb") as f:
            return f.read(4) == ZX_MAGIC
    except OSError:
        return False


def read_zx_info(path: str | Path) -> ZxModelInfo:
    """Parse and return structural metadata from a compiled .zx binary."""
    p = Path(path)
    file_size = p.stat().st_size
    if file_size < ZX_HEADER_SIZE:
        raise InputValidationError(f"File too small to be a valid .zx binary: {p}")

    with open(p, "rb") as f:
        header_data = f.read(ZX_HEADER_SIZE)
        (
            magic,
            v_maj,
            v_min,
            target_code,
            flags,
            param_count,
            tensor_count,
            node_count,
            arena_size,
            tensor_offset,
            node_offset,
        ) = struct.unpack("<4sHHIIQIIQQQ", header_data[:56])

        if magic != ZX_MAGIC:
            raise InputValidationError(f"Invalid .zx magic bytes in {p}: {magic!r}")

        # Read tensor table
        f.seek(tensor_offset)
        tensors: list[ZxTensorRecord] = []
        for _ in range(tensor_count):
            (name_len,) = struct.unpack("<H", f.read(2))
            name = f.read(name_len).decode("utf-8") if name_len else ""
            type_val, rank = struct.unpack("<BB", f.read(2))
            shape_dims: list[int] = []
            for _ in range(rank):
                (dim,) = struct.unpack("<Q", f.read(8))
                shape_dims.append(dim)
            offset, size, is_out = struct.unpack("<QQB", f.read(17))
            tensors.append(
                ZxTensorRecord(
                    name=name,
                    dtype=DTYPE_NAMES.get(type_val, f"dtype_{type_val}"),
                    shape=tuple(shape_dims),
                    arena_offset=offset,
                    arena_size=size,
                    is_output=bool(is_out != 0),
                )
            )

        # Read node table
        f.seek(node_offset)
        nodes: list[ZxNodeRecord] = []
        for _ in range(node_count):
            (name_len,) = struct.unpack("<H", f.read(2))
            name = f.read(name_len).decode("utf-8") if name_len else ""
            (op_len,) = struct.unpack("<H", f.read(2))
            op = f.read(op_len).decode("utf-8") if op_len else ""
            (in_count,) = struct.unpack("<I", f.read(4))
            inputs: list[int] = []
            for _ in range(in_count):
                (idx,) = struct.unpack("<I", f.read(4))
                inputs.append(idx)
            (out_count,) = struct.unpack("<I", f.read(4))
            outputs: list[int] = []
            for _ in range(out_count):
                (idx,) = struct.unpack("<I", f.read(4))
                outputs.append(idx)

            nodes.append(
                ZxNodeRecord(
                    name=name,
                    operation=op,
                    inputs=tuple(inputs),
                    outputs=tuple(outputs),
                )
            )

    return ZxModelInfo(
        path=str(p),
        file_size_bytes=file_size,
        target_arch=TARGET_NAMES.get(target_code, f"Target_{target_code}"),
        parameter_count=param_count,
        tensor_count=tensor_count,
        node_count=node_count,
        arena_size_bytes=arena_size,
        tensors=tuple(tensors),
        nodes=tuple(nodes),
    )


def write_zx_binary(
    path: str | Path,
    *,
    target: str = "cpu",
    tensors: Sequence[dict[str, Any]],
    nodes: Sequence[dict[str, Any]],
    arena_size: int,
) -> None:
    """Write a standalone .zx binary file."""
    p = Path(path)
    target_code = TARGET_CODES.get(target.lower(), 0)

    total_params = 0
    for t in tensors:
        elements = 1
        for dim in t["shape"]:
            elements *= dim
        total_params += elements

    with open(p, "wb") as f:
        # Reserve 64 bytes for header
        f.write(b"\x00" * ZX_HEADER_SIZE)

        tensor_offset = f.tell()
        for t in tensors:
            name_bytes = t["name"].encode("utf-8")
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)
            dtype_code = DTYPE_CODES.get(t["dtype"].lower(), 0)
            shape = t["shape"]
            f.write(struct.pack("<BB", dtype_code, len(shape)))
            for dim in shape:
                f.write(struct.pack("<Q", dim))
            f.write(
                struct.pack(
                    "<QQB",
                    t.get("offset", 0),
                    t.get("size", 0),
                    1 if t.get("is_output", False) else 0,
                )
            )

        node_offset = f.tell()
        for n in nodes:
            name_bytes = n["name"].encode("utf-8")
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)
            op_bytes = n["operation"].encode("utf-8")
            f.write(struct.pack("<H", len(op_bytes)))
            f.write(op_bytes)
            inputs = n.get("inputs", [])
            f.write(struct.pack("<I", len(inputs)))
            for idx in inputs:
                f.write(struct.pack("<I", idx))
            outputs = n.get("outputs", [])
            f.write(struct.pack("<I", len(outputs)))
            for idx in outputs:
                f.write(struct.pack("<I", idx))

        # Seek back and write header
        f.seek(0)
        header = struct.pack(
            "<4sHHIIQIIQQQ",
            ZX_MAGIC,
            1,
            0,
            target_code,
            0,
            total_params,
            len(tensors),
            len(nodes),
            arena_size,
            tensor_offset,
            node_offset,
        )
        f.write(header.ljust(ZX_HEADER_SIZE, b"\x00"))
