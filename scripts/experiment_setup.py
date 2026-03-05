#!/usr/bin/env python3
"""Set up experiment directory structure and generate configs/scripts."""

import json
import shlex
import sys
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
    """Generate the next sequential experiment ID (exp-001, exp-002, ...)."""
    path = Path(results_dir)
    if not path.exists():
        return "exp-001"
    existing = sorted(path.glob("exp-*.json"))
    if not existing:
        return "exp-001"
    last = existing[-1].stem  # e.g., "exp-005"
    try:
        num = int(last.split("-")[1])
    except (IndexError, ValueError):
        num = 0
    return f"exp-{num + 1:03d}"


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

    # Create log directory
    log_dir = str(Path(log_file).parent)
    lines.append(f"mkdir -p {shlex.quote(log_dir)}")
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


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: experiment_setup.py <project_root> <train_command> [gpu_id] [config_json]')
        sys.exit(1)
    project_root = sys.argv[1]
    train_command = sys.argv[2]
    gpu_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    config = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    print(json.dumps(setup(project_root, train_command, gpu_id, config), indent=2))
