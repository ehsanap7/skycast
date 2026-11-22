"""Command-line entry point for SkyCast."""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from skycast import __version__
from skycast.config import DEFAULT_CONFIG_DIR, load_settings
from skycast.data.pipeline import PipelineSummary, run_data_pipeline
from skycast.errors import SkyCastError

logger = logging.getLogger(__name__)


def _greet(_args: argparse.Namespace) -> int:
    """Print a greeting with the installed version."""
    print(f"Hello from SkyCast v{__version__}!")
    return 0


def _print_summary(summary: PipelineSummary) -> None:
    """Print what a data build produced."""
    print(f"Bronze: {summary.bronze_rows:,} rows")
    print(f"Silver: {summary.silver_rows:,} rows")
    print(f"Gold:   {summary.gold_rows:,} rows")
    print(f"Saved:  {summary.gold_key}")


def _build_data(args: argparse.Namespace) -> int:
    """Run the data pipeline and report the outcome."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        summary = run_data_pipeline(load_settings(args.config_dir))
    except SkyCastError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    except Exception:
        logger.exception("Data build failed unexpectedly")
        return 1
    _print_summary(summary)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Define the available commands."""
    parser = argparse.ArgumentParser(prog="skycast")
    parser.set_defaults(handler=_greet)
    groups = parser.add_subparsers(dest="group")
    data = groups.add_parser("data", help="work with datasets")
    actions = data.add_subparsers(dest="action", required=True)
    build = actions.add_parser("build", help="build silver and gold from bronze")
    build.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    build.set_defaults(handler=_build_data)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested command and return its exit code."""
    args = _build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
