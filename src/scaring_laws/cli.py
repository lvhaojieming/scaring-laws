"""Command-line entry point for the starter project."""

from __future__ import annotations

from typing import Sequence

from . import __version__


def build_status() -> str:
    """Return a short human-readable project status summary."""
    lines = [
        "scaring-laws is ready.",
        f"Version: {__version__}",
        "",
        "Next moves:",
        "- add your domain logic under src/scaring_laws/",
        "- extend tests under tests/",
        "- update README.md with product goals",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the starter CLI."""
    del argv
    print(build_status())
    return 0
