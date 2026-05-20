"""Smoke tests for the CLI."""

from __future__ import annotations

import subprocess
import sys

from mleval import __version__
from mleval._cli.main import main


def test_version_command_via_function(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_version_command_via_subprocess():
    result = subprocess.run(
        [sys.executable, "-m", "mleval._cli.main", "version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == __version__
