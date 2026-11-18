"""Tests for the SkyCast command-line entry point."""

import pytest

from skycast import __version__
from skycast.cli import main


def test_main_prints_hello_world(capsys: pytest.CaptureFixture[str]) -> None:
    """``main()`` greets the user and includes the installed version."""
    main()

    captured = capsys.readouterr()

    assert captured.out == f"Hello from SkyCast v{__version__}!\n"
    assert captured.err == ""


def test_version_is_set() -> None:
    """The package exposes a real version from installed metadata."""
    assert __version__
    assert __version__ != "0.0.0+unknown"
