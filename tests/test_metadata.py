"""Tests for model metadata extraction."""

import json
import struct

import pytest
from zenthrix.exceptions import InputValidationError
from zenthrix.metadata import extract_metadata

# -- SafeTensors fixtures & tests --------------------------------------------


def _write_safetensors(path, tensors=None, metadata=None):
    """Write a minimal valid SafeTensors file for testing."""
    if tensors is None:
        tensors = {
            "weight": {"dtype": "F32", "shape": [4, 3], "data_offsets": [0, 48]},
            "bias": {"dtype": "F32", "shape": [4], "data_offsets": [48, 64]},
        }
    header = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    header.update(tensors)
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<Q", len(header_bytes))
    # Data section: just enough zero bytes to satisfy the offsets
    data_size = 64
    path.write_bytes(header_len + header_bytes + b"\x00" * data_size)
    return path


class TestExtractSafetensorsMetadata:
    def test_basic_metadata(self, tmp_path):
        model = _write_safetensors(tmp_path / "model.safetensors")
        meta = extract_metadata(model)

        assert meta.format == "safetensors"
        assert meta.parameter_count == 16  # 4*3 + 4
        assert len(meta.tensors) == 2
        assert meta.file_size_bytes > 0

    def test_tensor_details(self, tmp_path):
        model = _write_safetensors(tmp_path / "model.safetensors")
        meta = extract_metadata(model)

        names = {t.name for t in meta.tensors}
        assert names == {"weight", "bias"}

        weight = next(t for t in meta.tensors if t.name == "weight")
        assert weight.shape == (4, 3)
        assert weight.dtype == "F32"

    def test_estimated_memory(self, tmp_path):
        model = _write_safetensors(tmp_path / "model.safetensors")
        meta = extract_metadata(model)

        # 16 params * 4 bytes (F32) = 64 bytes
        assert meta.estimated_memory_mb == pytest.approx(64 / (1024 * 1024))

    def test_metadata_section_captured(self, tmp_path):
        model = _write_safetensors(
            tmp_path / "model.safetensors",
            metadata={"model_type": "test"},
        )
        meta = extract_metadata(model)
        assert meta.extra.get("model_type") == "test"

    def test_to_dict_serializable(self, tmp_path):
        model = _write_safetensors(tmp_path / "model.safetensors")
        meta = extract_metadata(model)
        d = meta.to_dict()
        # Must be JSON-serializable
        assert json.loads(json.dumps(d))
        assert d["format"] == "safetensors"
        assert d["tensor_count"] == 2

    def test_corrupted_header_rejected(self, tmp_path):
        model = tmp_path / "bad.safetensors"
        model.write_bytes(b"\x00" * 4)  # too short
        with pytest.raises(InputValidationError, match="too small"):
            extract_metadata(model)

    def test_invalid_json_header_rejected(self, tmp_path):
        model = tmp_path / "bad.safetensors"
        header_len = struct.pack("<Q", 5)
        model.write_bytes(header_len + b"hello")
        with pytest.raises(InputValidationError, match="not valid JSON"):
            extract_metadata(model)


# -- GGUF fixtures & tests ---------------------------------------------------

_GGUF_MAGIC = 0x46475547


def _write_gguf(path, version=3, tensor_count=5, architecture="llama"):
    """Write a minimal valid GGUF file header for testing."""
    buf = bytearray()
    # magic + version + tensor_count + kv_count
    kv_count = 1 if architecture else 0
    buf += struct.pack("<IIQQ", _GGUF_MAGIC, version, tensor_count, kv_count)
    if architecture:
        # Write one KV pair: general.architecture = <string>
        key = "general.architecture"
        key_bytes = key.encode("utf-8")
        buf += struct.pack("<Q", len(key_bytes))
        buf += key_bytes
        buf += struct.pack("<I", 8)  # type STRING
        arch_bytes = architecture.encode("utf-8")
        buf += struct.pack("<Q", len(arch_bytes))
        buf += arch_bytes
    path.write_bytes(bytes(buf))
    return path


class TestExtractGgufMetadata:
    def test_basic_metadata(self, tmp_path):
        model = _write_gguf(tmp_path / "model.gguf")
        meta = extract_metadata(model)

        assert meta.format == "gguf"
        assert meta.architecture == "llama"
        assert meta.file_size_bytes > 0

    def test_version_in_extra(self, tmp_path):
        model = _write_gguf(tmp_path / "model.gguf", version=3)
        meta = extract_metadata(model)
        assert meta.extra.get("version") == "3"

    def test_bad_magic_rejected(self, tmp_path):
        model = tmp_path / "bad.gguf"
        buf = struct.pack("<IIQQ", 0xDEADBEEF, 3, 0, 0)
        model.write_bytes(buf)
        with pytest.raises(InputValidationError, match="bad magic"):
            extract_metadata(model)

    def test_truncated_header_rejected(self, tmp_path):
        model = tmp_path / "tiny.gguf"
        model.write_bytes(b"\x00" * 8)  # too short for GGUF header
        with pytest.raises(InputValidationError, match="too small"):
            extract_metadata(model)


# -- ONNX fixture & test -----------------------------------------------------


def _write_onnx(path):
    """Write a minimal ONNX-like protobuf envelope for testing."""
    # Field 1 (ir_version) = 9, tag 0x08, varint 9
    # Field 3 (producer_name) = "zenthrix-test", tag 0x1a, len-delimited
    producer = b"zenthrix-test"
    buf = bytes([0x08, 0x09, 0x1A, len(producer)]) + producer
    path.write_bytes(buf)
    return path


class TestExtractOnnxMetadata:
    def test_basic_metadata(self, tmp_path):
        model = _write_onnx(tmp_path / "model.onnx")
        meta = extract_metadata(model)

        assert meta.format == "onnx"
        assert meta.extra.get("ir_version") == "9"
        assert meta.extra.get("producer_name") == "zenthrix-test"


# -- Error cases across formats -----------------------------------------------


class TestExtractMetadataErrors:
    def test_unknown_extension_rejected(self, tmp_path):
        model = tmp_path / "model.bin"
        model.write_bytes(b"\x00")
        with pytest.raises(InputValidationError, match="Unsupported file extension"):
            extract_metadata(model)

    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(InputValidationError, match="does not exist"):
            extract_metadata(tmp_path / "nonexistent.safetensors")

    def test_directory_rejected(self, tmp_path):
        d = tmp_path / "notafile.safetensors"
        d.mkdir()
        with pytest.raises(InputValidationError, match="not a file"):
            extract_metadata(d)
