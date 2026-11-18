"""Command-line entry point for SkyCast."""

from skycast import __version__


def main() -> None:
    """Print a hello-world greeting with the installed SkyCast version.

    This is the target of the ``skycast`` console script defined in
    ``pyproject.toml``.
    """
    print(f"Hello from SkyCast v{__version__}!")


if __name__ == "__main__":
    main()
