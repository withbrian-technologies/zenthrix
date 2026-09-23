"""Tests for zenthrix.toml configuration support."""

import pytest
from zenthrix.exceptions import InputValidationError
from zenthrix.settings import (
    ZenthrixSettings,
    find_config,
    load_config,
    resolve_settings,
)


class TestFindConfig:
    def test_locates_file_in_directory(self, tmp_path):
        config = tmp_path / "zenthrix.toml"
        config.write_text("[compile]\ntarget = 'apple'\n")
        assert find_config(tmp_path) == config

    def test_walks_parents(self, tmp_path):
        config = tmp_path / "zenthrix.toml"
        config.write_text("[compile]\ntarget = 'arm'\n")
        child = tmp_path / "subdir" / "deep"
        child.mkdir(parents=True)
        assert find_config(child) == config

    def test_returns_none_when_absent(self, tmp_path):
        assert find_config(tmp_path) is None


class TestLoadConfig:
    def test_parses_compile_section(self, tmp_path):
        config = tmp_path / "zenthrix.toml"
        config.write_text(
            "[compile]\n"
            "target = 'qualcomm'\n"
            "quantization = 'int4'\n"
            "output = './out/model.zx'\n"
        )
        data = load_config(config)
        assert data["compile"]["target"] == "qualcomm"
        assert data["compile"]["quantization"] == "int4"
        assert data["compile"]["output"] == "./out/model.zx"

    def test_parses_run_section(self, tmp_path):
        config = tmp_path / "zenthrix.toml"
        config.write_text(
            "[run]\n"
            "max_tokens = 512\n"
            "temperature = 0.1\n"
        )
        data = load_config(config)
        assert data["run"]["max_tokens"] == 512
        assert data["run"]["temperature"] == 0.1

    def test_ignores_unknown_sections(self, tmp_path):
        config = tmp_path / "zenthrix.toml"
        config.write_text(
            "[custom]\nfoo = 'bar'\n"
            "[compile]\ntarget = 'auto'\n"
        )
        data = load_config(config)
        assert "custom" in data  # parsed but harmless
        assert data["compile"]["target"] == "auto"

    def test_rejects_malformed_toml(self, tmp_path):
        config = tmp_path / "zenthrix.toml"
        config.write_text("[[[[invalid toml")
        with pytest.raises(InputValidationError, match="Failed to parse"):
            load_config(config)


class TestResolveSettings:
    def test_cli_overrides_config(self):
        cli = {"target": "qualcomm", "quantization": None, "output": None,
               "max_tokens": None, "temperature": None, "model_format": None}
        config = {"compile": {"target": "apple", "quantization": "int8"}}
        settings = resolve_settings(cli, config)
        assert settings.target == "qualcomm"  # CLI wins
        assert settings.quantization == "int8"  # falls through to config

    def test_uses_config_defaults(self):
        cli = {"target": None, "quantization": None, "output": None,
               "max_tokens": None, "temperature": None, "model_format": None}
        config = {"compile": {"target": "arm"}, "run": {"max_tokens": 256}}
        settings = resolve_settings(cli, config)
        assert settings.target == "arm"
        assert settings.max_tokens == 256

    def test_uses_hardcoded_defaults(self):
        cli = {"target": None, "quantization": None, "output": None,
               "max_tokens": None, "temperature": None, "model_format": None}
        settings = resolve_settings(cli)
        assert settings == ZenthrixSettings()

    def test_empty_config_safe(self):
        cli = {"target": None, "quantization": None, "output": None,
               "max_tokens": None, "temperature": None, "model_format": None}
        settings = resolve_settings(cli, {})
        assert settings.target == "auto"
