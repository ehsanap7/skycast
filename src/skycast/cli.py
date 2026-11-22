"""Command-line entry point for SkyCast."""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from skycast import __version__
from skycast.config import DEFAULT_CONFIG_DIR, load_settings
from skycast.data.contracts import SourceKind
from skycast.data.pipeline import PipelineSummary, run_data_pipeline
from skycast.errors import ConfigError, SkyCastError
from skycast.ingestion.runner import UpdateSummary, update_source

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


def _print_update(summary: UpdateSummary) -> None:
    """Print what a source update produced."""
    print(f"Source:    {summary.source}")
    print(f"New hours: {summary.new_hours:,}")
    print(f"Last hour: {summary.last_hour}")
    if summary.gold_rows is not None:
        print(f"Gold:      {summary.gold_rows:,} rows")


def _guarded(action: Callable[[], None], description: str) -> int:
    """Run a command and turn failures into an exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        action()
    except SkyCastError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    except Exception:
        logger.exception("%s failed unexpectedly", description)
        return 1
    return 0


def _build_data(args: argparse.Namespace) -> int:
    """Run the data pipeline and report the outcome."""
    return _guarded(
        lambda: _print_summary(run_data_pipeline(load_settings(args.config_dir))),
        "Data build",
    )


def _update_data(args: argparse.Namespace) -> int:
    """Update one source now and report the outcome."""
    kind = SourceKind(args.source)
    return _guarded(
        lambda: _print_update(update_source(load_settings(args.config_dir), kind)),
        "Data update",
    )


def _start_schedules(config_dir: Path) -> None:
    """Load Prefect only when schedules are requested."""
    try:
        from skycast.ingestion.flows import serve_schedules
    except ModuleNotFoundError as err:
        if err.name and err.name.split(".")[0] == "prefect":
            raise ConfigError(
                "Prefect is not installed. Run: uv sync --extra orchestration"
            ) from err
        raise
    serve_schedules(load_settings(config_dir), config_dir)


def _schedule_data(args: argparse.Namespace) -> int:
    """Start the scheduled data updates."""
    return _guarded(lambda: _start_schedules(args.config_dir), "Data schedule")


def _run_migration(config_dir: Path) -> None:
    """Copy local data to object storage and report it."""
    from skycast.data.migration import run_migration

    summary = run_migration(load_settings(config_dir))
    print(f"Files verified: {summary.files:,}")
    print(f"Copied:         {summary.copied:,}")
    print(f"Already there:  {summary.already_present:,}")
    print(f"Bytes verified: {summary.bytes_verified:,}")


def _migrate_data(args: argparse.Namespace) -> int:
    """Copy local datasets into object storage."""
    return _guarded(lambda: _run_migration(args.config_dir), "Data migration")


def _add_command(
    actions: argparse._SubParsersAction,
    name: str,
    help_text: str,
    handler: Callable[[argparse.Namespace], int],
) -> argparse.ArgumentParser:
    """Add a data command with the shared config option."""
    command = actions.add_parser(name, help=help_text)
    command.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    command.set_defaults(handler=handler)
    return command


def _build_parser() -> argparse.ArgumentParser:
    """Define the available commands."""
    parser = argparse.ArgumentParser(prog="skycast")
    parser.set_defaults(handler=_greet)
    groups = parser.add_subparsers(dest="group")
    data = groups.add_parser("data", help="work with datasets")
    actions = data.add_subparsers(dest="action", required=True)
    _add_command(actions, "build", "build silver and gold from bronze", _build_data)
    update = _add_command(actions, "update", "fetch one source and rebuild", _update_data)
    update.add_argument("--source", required=True, choices=[kind.value for kind in SourceKind])
    _add_command(actions, "schedule", "run scheduled updates until stopped", _schedule_data)
    _add_command(actions, "migrate", "copy local data to object storage", _migrate_data)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested command and return its exit code."""
    args = _build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
