#!/usr/bin/env python3
"""Set up experiment directory structure and generate configs/scripts."""

import json
import os
import re
import shlex
import sys
import time
from pathlib import Path


SUBDIRS = ["logs", "reports", "scripts", "results"]


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

    Only files matching the strict ``exp-\\d+\\.json`` pattern are considered,
    so unrelated JSON files (e.g. ``experiment-summary.json``) are ignored.
    """
    path = Path(results_dir)
    if not path.exists():
        return "exp-001"

    exp_pattern = re.compile(r"^exp-(\d+)\.json$")
    nums: list[int] = []
    for f in path.iterdir():
        m = exp_pattern.match(f.name)
        if m:
            nums.append(int(m.group(1)))

    if not nums:
        return "exp-001"
    return f"exp-{max(nums) + 1:03d}"


def write_experiment_config(results_dir: str, exp_id: str, config: dict) -> str:
    """Write an experiment config JSON file."""
    path = Path(results_dir) / f"{exp_id}.json"
    path.write_text(json.dumps(config, indent=2))
    return str(path)


def generate_train_script(
    scripts_dir: str,
    exp_id: str,
    train_command: str,
    gpu_id: int = 0,
    log_file: str | None = None,
    env_vars: dict | None = None,
) -> str:
    """Generate a bash training script from parameters."""
    if log_file is None:
        log_file = f"experiments/logs/{exp_id}/train.log"

    lines = ["#!/bin/bash", f"# Experiment: {exp_id}", "set -e", ""]

    # Environment variables
    lines.append(f"export CUDA_VISIBLE_DEVICES={gpu_id}")
    if env_vars:
        for key, value in env_vars.items():
            lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.append("")

    # Create log directory and record PID
    log_dir = str(Path(log_file).parent)
    lines.append(f"mkdir -p {shlex.quote(log_dir)}")
    lines.append(f"echo $$ > {shlex.quote(log_dir + '/pid')}")
    lines.append("")

    # Training command with logging
    lines.append(f"echo {shlex.quote(f'Starting experiment {exp_id} on GPU {gpu_id}')}")
    lines.append(f"{train_command} 2>&1 | tee {shlex.quote(log_file)}")
    lines.append("")
    lines.append(f"echo {shlex.quote(f'Experiment {exp_id} completed')}")

    script_path = Path(scripts_dir) / f"{exp_id}.sh"
    script_path.write_text("\n".join(lines))
    script_path.chmod(0o755)
    return str(script_path)


def setup(project_root: str, train_command: str, gpu_id: int = 0, config: dict | None = None) -> dict:
    """Full setup: create dirs, generate ID, write config and script."""
    exp_root = create_experiment_dirs(project_root)
    results_dir = str(Path(exp_root) / "results")
    exp_id = next_experiment_id(results_dir)

    config = config or {}
    config_path = write_experiment_config(results_dir, exp_id, {
        "exp_id": exp_id,
        "config": config,
        "status": "pending",
    })

    log_file = str(Path(exp_root) / "logs" / exp_id / "train.log")
    script_path = generate_train_script(
        str(Path(exp_root) / "scripts"),
        exp_id,
        train_command,
        gpu_id,
        log_file,
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

    for f in sorted(path.iterdir()):
        if not exp_pattern.match(f.name):
            continue
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
    gpu_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    config = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    print(json.dumps(setup(project_root, train_command, gpu_id, config), indent=2))
