"""Shared test fixtures."""

import io
import json
import runpy
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pytest

# Make scripts importable from all test files
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(SCRIPTS_DIR))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


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
        return type("R", (), {"returncode": 0, "stdout": out.getvalue(), "stderr": err.getvalue()})
    except SystemExit as e:
        return type("R", (), {"returncode": e.code or 0, "stdout": out.getvalue(), "stderr": err.getvalue()})
    finally:
        sys.argv = old_argv


@pytest.fixture
def run_main():
    """Fixture that returns the _run_main helper."""
    return _run_main


def _write_result(results_dir, exp_id, status, config, metrics, **extra):
    """Helper to write a minimal experiment result JSON."""
    data = {"exp_id": exp_id, "status": status, "config": config, "metrics": metrics}
    data.update(extra)
    (results_dir / f"{exp_id}.json").write_text(json.dumps(data))


def _write_results(results_dir, experiments: dict):
    """Write experiment dicts as JSON files, keyed by filename stem."""
    for name, data in experiments.items():
        (results_dir / f"{name}.json").write_text(json.dumps(data))
