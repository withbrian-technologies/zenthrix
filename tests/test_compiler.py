"""Tests for model-to-.zx compilation lowering."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from zenthrix.cli import main
from zenthrix.compiler import compile_model
from zenthrix.native import is_native_available
from zenthrix.zx_format import is_zx_file, read_zx_info


def _write_min_safetensors(path: Path) -> None:
    header = {
        "layer0.weight": {
            "dtype": "F32",
            "shape": [64, 128],
            "data_offsets": [0, 32768],
        },
        "layer1.weight": {
            "dtype": "F16",
            "shape": [128, 64],
            "data_offsets": [32768, 49152],
        },
    }
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * 64)


def test_compile_safetensors_to_zx(tmp_path: Path) -> None:
    if not is_native_available():
        return
    src = tmp_path / "model.safetensors"
    out = tmp_path / "out.zx"
    _write_min_safetensors(src)

    res = compile_model(src, out, target="auto")
    assert out.is_file()
    assert is_zx_file(out)
    assert res.arena_size_bytes > 0
    assert res.node_count >= 1

    info = read_zx_info(out)
    assert info.arena_size_bytes == res.arena_size_bytes


def test_cli_compile_json_roundtrip(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    if not is_native_available():
        return
    src = tmp_path / "model.safetensors"
    out = tmp_path / "cli.zx"
    _write_min_safetensors(src)

    rc = main(
        [
            "compile",
            "--model",
            str(src),
            "--format",
            "safetensors",
            "--target",
            "auto",
            "--output",
            str(out),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "safetensors"
    assert payload["arena_size_bytes"] > 0
    assert is_zx_file(out)
