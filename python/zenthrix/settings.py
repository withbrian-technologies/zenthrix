"""Project and user configuration via zenthrix.toml.

Walks the directory tree upward from the working directory looking for a
``zenthrix.toml`` file.  CLI arguments always take precedence over values
read from the configuration file.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import InputValidationError

# tomllib is stdlib in 3.11+; tomli is the backport for 3.10
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ModuleNotFoundError as exc:
        raise ImportError(
            "zenthrix requires the 'tomli' package on Python < 3.11. "
            "Install it with: pip install tomli"
        ) from exc


CONFIG_FILENAME = "zenthrix.toml"


@dataclass(frozen=True, slots=True)
class ZenthrixSettings:
    """Resolved configuration from zenthrix.toml."""

    target: str = "auto"
    quantization: str | None = None
    output: str | None = None
    model_format: str | None = None
    max_tokens: int = 128
    temperature: float = 0.2


def find_config(start: Path | None = None) -> Path | None:
    """Walk up from *start* (default: cwd) looking for ``zenthrix.toml``.

    Returns the path to the first file found, or ``None``.
    """
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_config(path: Path) -> dict[str, Any]:
    """Parse a ``zenthrix.toml`` file and return its contents.

    Raises :class:`InputValidationError` on malformed TOML.
    """
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except Exception as exc:
        raise InputValidationError(
            f"Failed to parse {path}: {exc}"
        ) from exc


def resolve_settings(
    cli_args: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> ZenthrixSettings:
    """Merge CLI arguments over config-file defaults.

    *cli_args* values that are ``None`` fall through to config-file
    defaults.  Config sections are ``[compile]`` and ``[run]``.
    """
    compile_cfg: dict[str, Any] = {}
    run_cfg: dict[str, Any] = {}
    if config:
        compile_cfg = config.get("compile", {})
        run_cfg = config.get("run", {})

    def _pick(key: str, *fallback_dicts: dict[str, Any], default: Any = None) -> Any:
        value = cli_args.get(key)
        if value is not None:
            return value
        for d in fallback_dicts:
            if key in d:
                return d[key]
        return default

    return ZenthrixSettings(
        target=_pick("target", compile_cfg, default="auto"),
        quantization=_pick("quantization", compile_cfg),
        output=_pick("output", compile_cfg),
        model_format=_pick("model_format", compile_cfg),
        max_tokens=_pick("max_tokens", run_cfg, default=128),
        temperature=_pick("temperature", run_cfg, default=0.2),
    )
