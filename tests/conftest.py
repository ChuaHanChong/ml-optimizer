"""Shared test fixtures and common paths for the ml-optimizer test suite."""

import io
import json
import os
import re
import runpy
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Shared paths — the single source for all test modules. scripts/ is placed on
# sys.path here (the ONE place), so test files import scripts modules flat and
# never need their own sys.path manipulation.
# ---------------------------------------------------------------------------
PLUGIN_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PLUGIN_ROOT / "scripts"
AGENTS_DIR = PLUGIN_ROOT / "agents"
SKILLS_DIR = PLUGIN_ROOT / "skills"
HOOKS_DIR = PLUGIN_ROOT / "hooks"
REFERENCES_DIR = SKILLS_DIR / "orchestrate" / "references"
PLUGIN_JSON = PLUGIN_ROOT / ".claude-plugin" / "plugin.json"
FIXTURES = PLUGIN_ROOT / "tests" / "fixtures"
sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_setup import next_experiment_id, generate_script


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


def _write_result(results_dir, exp_id, status, config, metrics,
                   method_tier=None, proposal_source=None, code_branch=None,
                   **extra):
    """Helper to write a minimal experiment result JSON.

    Optional three-tier tracking fields: method_tier, proposal_source, code_branch.
    """
    data = {"exp_id": exp_id, "status": status, "config": config, "metrics": metrics}
    if method_tier is not None:
        data["method_tier"] = method_tier
    if proposal_source is not None:
        data["proposal_source"] = proposal_source
    if code_branch is not None:
        data["code_branch"] = code_branch
    data.update(extra)
    (results_dir / f"{exp_id}.json").write_text(json.dumps(data))


def _write_results(results_dir, experiments: dict):
    """Write experiment dicts as JSON files, keyed by filename stem."""
    for name, data in experiments.items():
        (results_dir / f"{name}.json").write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# Test helpers moved from scripts/experiment_setup.py — these are only used
# by the test suite, not by the plugin runtime.
# ---------------------------------------------------------------------------

_EXP_SUBDIRS = ["logs", "reports", "scripts", "results", "artifacts"]


def create_experiment_dirs(project_root: str) -> str:
    """Create a canonical `<exp_root>` directory structure under project_root.

    Test-only helper. The plugin runtime does NOT call this — the orchestrator
    chooses `<exp_root>` at Phase 0 and creates subdirs lazily.
    """
    exp_root = Path(project_root) / "experiments"
    for subdir in _EXP_SUBDIRS:
        (exp_root / subdir).mkdir(parents=True, exist_ok=True)
    dev_notes = exp_root / "dev_notes.md"
    if not dev_notes.exists():
        dev_notes.write_text("# Dev Notes\n\nSession task log.\n\n")
    return str(exp_root)


def write_experiment_config(results_dir: str, exp_id: str, config: dict, exclusive: bool = False) -> str:
    """Write an experiment config JSON file.

    When *exclusive* is True, the file is created atomically using O_CREAT|O_EXCL.
    Raises FileExistsError if the file already exists.
    """
    path = Path(results_dir) / f"{exp_id}.json"
    content = json.dumps(config, indent=2)
    if exclusive:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, content.encode())
        finally:
            os.close(fd)
    else:
        path.write_text(content)
    return str(path)


def cleanup_stale_experiments(results_dir: str, timeout_hours: float = 2.0) -> list:
    """Mark stale running/pending experiments as failed.

    An experiment is considered stale when its JSON file has not been modified
    for longer than *timeout_hours*.
    """
    path = Path(results_dir)
    if not path.exists():
        return []

    exp_pattern = re.compile(r"^exp-\d+\.json$")
    now = time.time()
    cutoff = now - timeout_hours * 3600
    cleaned = []

    exp_files = []
    for f in sorted(path.iterdir()):
        if f.is_file() and exp_pattern.match(f.name):
            exp_files.append(f)
    for rdir in sorted(path.iterdir()):
        if rdir.is_dir() and rdir.name.startswith("round-"):
            for f in sorted(rdir.iterdir()):
                if f.is_file() and exp_pattern.match(f.name):
                    exp_files.append(f)

    for f in exp_files:
        if f.stat().st_mtime > cutoff:
            continue
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        status = data.get("status")
        if status not in ("running", "pending"):
            continue
        data["status"] = "failed"
        data["notes"] = (
            f"Marked as failed: stale experiment (no updates for {timeout_hours}h)"
        )
        f.write_text(json.dumps(data, indent=2))
        cleaned.append(f.stem)

    return cleaned


def setup_experiment(project_root: str, train_command: str, gpu_id: int = 0,
                     config: dict = None, checkpoint_path: str = None,
                     method_tier: str = None, iteration: int = None,
                     round_dir: str = None) -> dict:
    """Full test setup: create dirs, generate ID, write config and script.

    Test-only wrapper. Renamed from `setup` to `setup_experiment` to avoid
    shadowing pytest's own `setup` fixture.
    """
    exp_root = create_experiment_dirs(project_root)
    results_dir = str(Path(exp_root) / "results")
    if round_dir:
        write_dir = str(Path(exp_root) / "results" / round_dir)
        Path(write_dir).mkdir(parents=True, exist_ok=True)
    else:
        write_dir = results_dir

    config = config or {}
    max_retries = 10
    for attempt in range(max_retries):
        exp_id = next_experiment_id(results_dir)
        try:
            placeholder = {
                "exp_id": exp_id,
                "config": config,
                "status": "pending",
            }
            if method_tier is not None:
                placeholder["method_tier"] = method_tier
            if iteration is not None:
                placeholder["iteration"] = iteration
            config_path = write_experiment_config(write_dir, exp_id, placeholder, exclusive=True)
            break
        except FileExistsError:
            if attempt == max_retries - 1:
                raise
            continue

    if round_dir:
        log_file = str(Path(exp_root) / "logs" / round_dir / exp_id / "train.log")
        scripts_base = str(Path(exp_root) / "scripts" / round_dir)
    else:
        log_file = str(Path(exp_root) / "logs" / exp_id / "train.log")
        scripts_base = str(Path(exp_root) / "scripts")
    script_path = generate_script(
        scripts_base, exp_id, train_command, gpu_id, log_file,
        checkpoint_path=checkpoint_path,
    )

    return {
        "exp_id": exp_id,
        "exp_root": exp_root,
        "config_path": config_path,
        "script_path": script_path,
        "log_file": log_file,
    }
