"""Entry point for the ``mleval`` CLI.

At MVP stage this is a stub: the only subcommand is ``version``. As Layer 1
analyzer (#62), state-predicate runner (#63), and orchestrator (#75) land,
they will register subcommands here.
"""

from __future__ import annotations

import argparse
import sys

from mleval import __version__


def build_parser() -> argparse.ArgumentParser:
    """Return the top-level argparse parser.

    Returns:
        Configured parser with one subparser group.
    """
    parser = argparse.ArgumentParser(prog="mleval", description="Skill-evaluation harness CLI.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version", help="Print the package version and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Optional argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        print(__version__)
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2  # unreachable; parser.error sys.exits


if __name__ == "__main__":
    sys.exit(main())
