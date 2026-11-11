"""Command-line interface for the Zenthrix frontend."""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .metadata import ModelMetadata
    from .zx_format import ZxModelInfo

from . import __version__
from .engine import Engine
from .exceptions import EngineUnavailableError, InputValidationError, ZenthrixError
from .validation import (
    validate_compile_options,
    validate_compiled_model_path,
    validate_model_input,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(prog="zenthrix")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    compile_parser = commands.add_parser("compile", help="Compile a model")
    compile_parser.add_argument("--model", required=True)
    compile_parser.add_argument("--format", default=None, dest="model_format")
    compile_parser.add_argument("--target", default=None)
    compile_parser.add_argument("--quantization", default=None)
    compile_parser.add_argument("--output", default=None)
    compile_parser.add_argument("--json", action="store_true")

    inspect_parser = commands.add_parser(
        "inspect", help="Inspect a compiled or source model"
    )
    inspect_parser.add_argument("model")
    inspect_parser.add_argument("--memory-profile", action="store_true")
    inspect_parser.add_argument("--json", action="store_true")

    run_parser = commands.add_parser("run", help="Run inference")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--prompt", required=True)
    run_parser.add_argument("--max-tokens", type=int, default=None)
    run_parser.add_argument("--temperature", type=float, default=None)
    run_parser.add_argument("--runtime", default=None)
    run_parser.add_argument("--json", action="store_true")

    plugins_parser = commands.add_parser(
        "plugins", help="List installed runtime plugins"
    )
    plugins_parser.add_argument("--json", action="store_true")

    return parser


def _compile(args: argparse.Namespace) -> int:
    if args.model_format is None:
        raise InputValidationError(
            "compile requires --format or model_format in zenthrix.toml"
        )
    if args.output is None:
        raise InputValidationError(
            "compile requires --output or output in zenthrix.toml"
        )
    validate_model_input(args.model, args.model_format)
    validate_compile_options(
        args.target if args.target is not None else "auto",
        args.quantization,
        args.output,
    )
    raise ZenthrixError(
        "The native compiler engine is not installed. "
        "The public frontend cannot produce a .zx binary yet."
    )


def _inspect(args: argparse.Namespace) -> int:
    # Source model inspection — works without native engine
    model_path = args.model
    suffix = model_path.rsplit(".", 1)[-1].lower() if "." in model_path else ""

    if suffix == "zx":
        validate_compiled_model_path(model_path)
        from .native import is_native_available
        from .zx_format import is_zx_file, read_zx_info

        if not is_native_available() or not is_zx_file(model_path):
            raise EngineUnavailableError(
                "The Zenthrix native engine is not installed. "
                "Install the platform runtime before inspecting compiled models."
            )

        zx_info = read_zx_info(model_path)
        if args.json:
            print(json.dumps(zx_info.to_dict()))
        else:
            _print_zx_info(zx_info, show_memory_profile=args.memory_profile)
        return 0

    # Inspect a source model (safetensors, gguf, onnx)
    from .metadata import extract_metadata

    meta = extract_metadata(model_path)

    if args.json:
        print(json.dumps(meta.to_dict()))
    else:
        _print_metadata(meta)
    return 0


def _print_zx_info(info: ZxModelInfo, *, show_memory_profile: bool = False) -> None:
    """Pretty-print compiled .zx model metadata and memory profile."""
    lines = [
        "",
        f"  Compiled Model: {info.path}",
        "  Format:         Zenthrix Executable (.zx v1.0)",
        f"  Target Silicon: {info.target_arch}",
        f"  File size:      {_format_size(info.file_size_bytes)}",
    ]
    if info.parameter_count:
        lines.append(f"  Parameters:     {info.parameter_count:,}")
    if info.arena_size_bytes:
        arena_mb = info.arena_size_bytes / (1024 * 1024)
        lines.append(
            f"  Planned Arena:  {_format_size(info.arena_size_bytes)} "
            f"({arena_mb:.2f} MB)"
        )
    lines.append(f"  Tensors:        {info.tensor_count}")
    lines.append(f"  Compute Nodes:  {info.node_count}")

    if show_memory_profile and info.tensors:
        lines.append("")
        lines.append("  Memory Arena Profile:")
        lines.append(
            f"    {'Offset':<12} {'Size':<10} {'Type':<8} {'Shape':<16} Tensor Name"
        )
        lines.append(f"    {'-'*12} {'-'*10} {'-'*8} {'-'*16} {'-'*20}")
        for t in info.tensors:
            shape_str = f"[{'x'.join(str(d) for d in t.shape)}]"
            lines.append(
                f"    {t.arena_offset:<12} {_format_size(t.arena_size):<10} "
                f"{t.dtype:<8} {shape_str:<16} {t.name}"
            )

    lines.append("")
    print("\n".join(lines))


def _print_metadata(meta: ModelMetadata) -> None:
    """Pretty-print model metadata to stdout."""

    lines = [
        "",
        f"  Model:          {meta.path}",
        f"  Format:         {meta.format}",
        f"  File size:      {_format_size(meta.file_size_bytes)}",
    ]
    if meta.parameter_count:
        lines.append(f"  Parameters:     {meta.parameter_count:,}")
    if meta.estimated_memory_mb:
        lines.append(f"  Est. memory:    {meta.estimated_memory_mb:.1f} MB")
    if meta.architecture:
        lines.append(f"  Architecture:   {meta.architecture}")
    if meta.tensor_count is not None:
        lines.append(f"  Tensors:        {meta.tensor_count}")
    if meta.extra:
        for key, value in sorted(meta.extra.items()):
            lines.append(f"  {key}:  {value}")
    lines.append("")
    print("\n".join(lines))


def _format_size(size_bytes: int) -> str:
    """Format byte count as a human-readable string."""
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def _run(args: argparse.Namespace) -> int:
    result = Engine(args.model, runtime_name=args.runtime).generate(
        args.prompt,
        temperature=args.temperature if args.temperature is not None else 0.2,
        max_tokens=args.max_tokens if args.max_tokens is not None else 128,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "text": result.text,
                    "ttft_ms": result.ttft_ms,
                    "tokens_per_second": result.tokens_per_second,
                }
            )
        )
    else:
        print(result.text)
    return 0


def _plugins(args: argparse.Namespace) -> int:
    from .runtime import discover_runtimes

    runtimes = discover_runtimes()
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "name": name,
                        "adapter": (
                            f"{adapter_cls.__module__}:{adapter_cls.__qualname__}"
                        ),
                    }
                    for name, adapter_cls in runtimes
                ]
            )
        )
        return 0

    if not runtimes:
        print("No runtime plugins installed.")
        print()
        print("Install a compatible runtime package, or see:")
        print("  https://github.com/withbrian-technologies/zenthrix#plugins")
        return 0

    print("Installed runtimes:")
    for name, adapter_cls in runtimes:
        module = f"{adapter_cls.__module__}:{adapter_cls.__qualname__}"
        print(f"  • {name} ({module})")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)

    try:
        # Merge config-file defaults for compile/run when available
        _apply_config_defaults(args)

        if args.command == "compile":
            return _compile(args)
        if args.command == "inspect":
            return _inspect(args)
        if args.command == "run":
            return _run(args)
        if args.command == "plugins":
            return _plugins(args)
    except (ZenthrixError, ValueError) as error:
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "error": type(error).__name__,
                        "message": str(error),
                    }
                )
            )
        else:
            print(f"zenthrix: error: {error}", file=sys.stderr)
        return 2
    return 0


def _apply_config_defaults(args: argparse.Namespace) -> None:
    """Load zenthrix.toml and fill in missing CLI arguments."""
    try:
        from .settings import find_config, load_config
    except ImportError:
        return

    config_path = find_config()
    if config_path is None:
        return

    config = load_config(config_path)

    command = getattr(args, "command", None)
    section = config.get(command, {}) if command else {}
    if not isinstance(section, dict):
        return

    for key, value in section.items():
        # CLI arguments take precedence — only fill in None values
        current = getattr(args, key, _SENTINEL)
        if current is None:
            setattr(args, key, value)


_SENTINEL = object()


if __name__ == "__main__":
    raise SystemExit(main())
