#!/usr/bin/env python3
"""Set up experiment directory structure and generate configs/scripts."""

import json
import os
import re
import shlex
import sys
import time
from pathlib import Path


SUBDIRS = ["logs", "reports", "scripts", "results", "artifacts"]


def create_experiment_dirs(project_root: str) -> str:
    """Create the experiments/ directory structure in a project."""
    exp_root = Path(project_root) / "experiments"
    for subdir in SUBDIRS:
        (exp_root / subdir).mkdir(parents=True, exist_ok=True)
    dev_notes = exp_root / "dev_notes.md"
    if not dev_notes.exists():
        dev_notes.write_text("# Dev Notes\n\nSession task log.\n\n")
    return str(exp_root)


def next_experiment_id(results_dir: str) -> str:
    """Generate the next sequential experiment ID (exp-001, exp-002, ...).

    Scans both flat ``results/`` and round directories
    (``results/round-*/``) to ensure globally unique IDs.
    Only files matching the strict ``exp-\\d+\\.json`` pattern are considered.
    """
    path = Path(results_dir)
    if not path.exists():
        return "exp-001"

    exp_pattern = re.compile(r"^exp-(\d+)\.json$")
    nums: list[int] = []

    # Scan flat results directory
    for f in path.iterdir():
        if f.is_file():
            m = exp_pattern.match(f.name)
            if m:
                nums.append(int(m.group(1)))

    # Scan round directories (round-N-type/)
    for rdir in path.iterdir():
        if rdir.is_dir() and rdir.name.startswith("round-"):
            for f in rdir.iterdir():
                if f.is_file():
                    m = exp_pattern.match(f.name)
                    if m:
                        nums.append(int(m.group(1)))

    if not nums:
        return "exp-001"
    return f"exp-{max(nums) + 1:03d}"


def write_experiment_config(results_dir: str, exp_id: str, config: dict, exclusive: bool = False) -> str:
    """Write an experiment config JSON file.

    When *exclusive* is True, the file is created atomically using O_CREAT|O_EXCL.
    Raises FileExistsError if the file already exists (used to prevent race conditions).
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


def generate_train_script(
    scripts_dir: str,
    exp_id: str,
    train_command: str,
    gpu_id: int = 0,
    log_file: str | None = None,
    env_vars: dict | None = None,
    time_budget: int | None = None,
    checkpoint_path: str | None = None,
) -> str:
    """Generate a bash training script from parameters.

    When *time_budget* is set (seconds), the training command is wrapped
    with ``timeout --signal=SIGTERM`` so all experiments train for the same
    wall-clock duration.  Exit code 124 (budget reached) is treated as
    success, not failure.

    When *checkpoint_path* is provided, it is exported as the ``CHECKPOINT_PATH``
    environment variable for the training script to load for warm-starting.

    ``log_file`` is required — callers must compute a round-based path
    (e.g., ``experiments/logs/round-1-hp/exp-001/train.log``) or the
    baseline-only flat path (``experiments/logs/baseline/train.log``).
    """
    if log_file is None:
        raise ValueError(
            "log_file is required — pass a round-based path like "
            "'experiments/logs/round-N-<type>/<exp-id>/train.log' "
            "(or 'experiments/logs/baseline/train.log' for baseline)."
        )

    lines = ["#!/bin/bash", f"# Experiment: {exp_id}", "set -e", ""]

    # Environment variables
    lines.append(f"export CUDA_VISIBLE_DEVICES={gpu_id}")
    if env_vars:
        for key, value in env_vars.items():
            lines.append(f"export {key}={shlex.quote(str(value))}")
    if checkpoint_path:
        lines.append(f"export CHECKPOINT_PATH={shlex.quote(str(checkpoint_path))}")
        lines.append(f"echo {shlex.quote(f'Warm-starting from checkpoint: {checkpoint_path}')}")
    lines.append("")

    # Create log directory and record PID
    log_dir = str(Path(log_file).parent)
    lines.append(f"mkdir -p {shlex.quote(log_dir)}")
    lines.append(f"echo $$ > {shlex.quote(log_dir + '/pid')}")
    lines.append("")

    # Training command with logging
    lines.append(f"echo {shlex.quote(f'Starting experiment {exp_id} on GPU {gpu_id}')}")
    if time_budget is not None and time_budget > 0:
        lines.append(f"echo {shlex.quote(f'Time budget: {time_budget}s')}")
        lines.append(
            f"timeout --signal=SIGTERM --kill-after=60 {time_budget} "
            f"{train_command} 2>&1 | tee {shlex.quote(log_file)}"
        )
        lines.append("EXIT_CODE=${PIPESTATUS[0]}")
        lines.append(f"if [ $EXIT_CODE -eq 124 ]; then")
        lines.append(f"    echo {shlex.quote(f'Time budget reached ({time_budget}s) — training stopped normally')}")
        lines.append(f"fi")
    else:
        lines.append(f"{train_command} 2>&1 | tee {shlex.quote(log_file)}")
    lines.append("")
    lines.append(f"echo {shlex.quote(f'Experiment {exp_id} completed')}")

    script_dir = Path(scripts_dir) / exp_id
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / "train.sh"
    script_path.write_text("\n".join(lines))
    script_path.chmod(0o755)
    return str(script_path)


def setup(project_root: str, train_command: str, gpu_id: int = 0, config: dict | None = None, checkpoint_path: str | None = None, method_tier: str | None = None, iteration: int | None = None, round_dir: str | None = None) -> dict:
    """Full setup: create dirs, generate ID, write config and script.

    Uses atomic file creation to prevent race conditions when multiple
    agents call setup() concurrently.

    When *round_dir* is provided (e.g. ``"round-1-hp"``), the result JSON
    is created inside ``results/<round_dir>/`` instead of flat ``results/``.
    """
    exp_root = create_experiment_dirs(project_root)
    results_dir = str(Path(exp_root) / "results")
    # Determine where to write the result file
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
    script_path = generate_train_script(
        scripts_base,
        exp_id,
        train_command,
        gpu_id,
        log_file,
        checkpoint_path=checkpoint_path,
    )

    return {
        "exp_id": exp_id,
        "exp_root": exp_root,
        "config_path": config_path,
        "script_path": script_path,
        "log_file": log_file,
    }


def cleanup_stale_experiments(results_dir: str, timeout_hours: float = 2.0) -> list[str]:
    """Mark stale running/pending experiments as failed.

    An experiment is considered stale when its JSON file has not been modified
    for longer than *timeout_hours*.

    Returns:
        List of experiment IDs that were cleaned up.
    """
    path = Path(results_dir)
    if not path.exists():
        return []

    exp_pattern = re.compile(r"^exp-\d+\.json$")
    now = time.time()
    cutoff = now - timeout_hours * 3600
    cleaned: list[str] = []

    # Collect experiment files from flat dir and round directories
    exp_files: list[Path] = []
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


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: experiment_setup.py <project_root> <train_command> [gpu_id] [config_json]')
        sys.exit(1)
    project_root = sys.argv[1]
    train_command = sys.argv[2]
    try:
        gpu_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    except ValueError:
        print(f"Error: invalid gpu_id '{sys.argv[3]}' (expected integer)")
        print('Usage: experiment_setup.py <project_root> <train_command> [gpu_id] [config_json] [round_dir]')
        sys.exit(1)
    try:
        config = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    except json.JSONDecodeError:
        print(f"Error: invalid config JSON '{sys.argv[4]}'")
        print('Usage: experiment_setup.py <project_root> <train_command> [gpu_id] [config_json] [round_dir]')
        sys.exit(1)
    round_dir = sys.argv[5] if len(sys.argv) > 5 else None
    print(json.dumps(setup(project_root, train_command, gpu_id, config, round_dir=round_dir), indent=2))
