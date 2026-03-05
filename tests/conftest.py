"""Shared test fixtures."""

import io
import runpy
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def _run_main(script_name, *args):
    """Run a script's __main__ block in-process for coverage tracking.

    Returns an object with .returncode and .stdout, mimicking
    subprocess.CompletedProcess.
    """
    script = str(SCRIPTS_DIR / script_name)
    old_argv = sys.argv[:]
    sys.argv = [script] + list(args)
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            runpy.run_path(script, run_name="__main__")
        return type("R", (), {"returncode": 0, "stdout": out.getvalue()})
    except SystemExit as e:
        return type("R", (), {"returncode": e.code or 0, "stdout": out.getvalue()})
    finally:
        sys.argv = old_argv


@pytest.fixture
def run_main():
    """Fixture that returns the _run_main helper."""
    return _run_main
