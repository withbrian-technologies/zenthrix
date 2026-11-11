"""Command-line interface for benchmark result tooling."""

import argparse
import sys

from .exceptions import BenchmarkError
from .results import load_run, render_summary


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI parser."""
    parser = argparse.ArgumentParser(prog="zbench")
    parser.add_argument("--version", action="version", version="0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate a result JSON file")
    validate.add_argument("results")
    report = commands.add_parser("report", help="Render validated results as Markdown")
    report.add_argument("results")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        metadata, results = load_run(args.results)
        if args.command == "validate":
            print(f"Validated {len(results)} benchmark result(s).")
            if metadata is not None:
                print(f"Run timestamp: {metadata.timestamp}")
        else:
            print(render_summary(results))
        return 0
    except BenchmarkError as error:
        print(f"zbench: error: {error}", file=sys.stderr)
        return 2
