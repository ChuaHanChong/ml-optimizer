#!/usr/bin/env python3
"""Set up experiment directory structure and generate configs/scripts."""

import json
import re
import shlex
import sys
from pathlib import Path


def next_experiment_id(results_dir: str) -> str:
    """Generate the next sequential experiment ID (exp-001, exp-002, ...).

    Scans both flat `results/` and round directories
    (`results/round-*/`) to ensure globally unique IDs.
    Only files matching the strict `exp-\\d+\\.json` pattern are considered.
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


def generate_script(
    scripts_dir: str,
    exp_id: str,
    command: str,
    gpu_id: int = 0,
    log_file: str | None = None,
    env_vars: dict | None = None,
    time_budget: int | None = None,
    checkpoint_path: str | None = None,
    script_name: str = "train.sh",
) -> str:
    """Generate a bash experiment script (training, evaluation, or any command).

    Writes a self-contained bash script at `<scripts_dir>/<exp_id>/<script_name>`
    with GPU assignment, logging, PID tracking, and optional timeout.

    Use `script_name` to distinguish multiple scripts per experiment:
      - `"train.sh"` for training (default)
      - `"eval.sh"` for evaluation
      - `"preprocess.sh"` for data preparation

    When *time_budget* is set (seconds), the command is wrapped with
    `timeout --signal=SIGTERM`. Exit code 124 (budget reached) is treated
    as success, not failure.

    When *checkpoint_path* is provided, it is exported as `CHECKPOINT_PATH`
    for warm-starting.

    `log_file` is required — callers must compute a round-based path
    (e.g., `<exp_root>/logs/round-1-hp/exp-001/train.log`).
    """
    if log_file is None:
        raise ValueError(
            "log_file is required — pass a round-based path like "
            "'<exp_root>/logs/round-N-<type>/<exp-id>/train.log' "
            "(or '<exp_root>/logs/baseline/train.log' for baseline)."
        )

    label = script_name.removesuffix(".sh")
    lines = ["#!/bin/bash", f"# Experiment: {exp_id} ({label})", "set -e", ""]

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

    # Command with logging
    lines.append(f"echo {shlex.quote(f'Starting {label} for {exp_id} on GPU {gpu_id}')}")
    if time_budget is not None and time_budget > 0:
        lines.append(f"echo {shlex.quote(f'Time budget: {time_budget}s')}")
        lines.append(
            f"timeout --signal=SIGTERM --kill-after=60 {time_budget} "
            f"{command} 2>&1 | tee {shlex.quote(log_file)}"
        )
        lines.append("EXIT_CODE=${PIPESTATUS[0]}")
        lines.append(f"if [ $EXIT_CODE -eq 124 ]; then")
        lines.append(f"    echo {shlex.quote(f'Time budget reached ({time_budget}s) — stopped normally')}")
        lines.append(f"fi")
    else:
        lines.append(f"{command} 2>&1 | tee {shlex.quote(log_file)}")
    lines.append("")
    lines.append(f"echo {shlex.quote(f'{label} for {exp_id} completed')}")

    script_dir = Path(scripts_dir) / exp_id
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / script_name
    script_path.write_text("\n".join(lines))
    script_path.chmod(0o755)
    return str(script_path)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: experiment_setup.py <exp_root> <train_command> [gpu_id] [config_json] [round_dir]')
        sys.exit(1)
    exp_root = sys.argv[1]
    train_command = sys.argv[2]
    try:
        gpu_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    except ValueError:
        print(f"Error: invalid gpu_id '{sys.argv[3]}' (expected integer)")
        sys.exit(1)
    try:
        config = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    except json.JSONDecodeError:
        print(f"Error: invalid config JSON '{sys.argv[4]}'")
        sys.exit(1)
    round_dir = sys.argv[5] if len(sys.argv) > 5 else None

    # Create subdirs under the user-provided exp_root
    root = Path(exp_root)
    for subdir in ["logs", "reports", "scripts", "results", "artifacts"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)

    results_dir = str(root / "results")
    exp_id = next_experiment_id(results_dir)
    if round_dir:
        write_dir = str(root / "results" / round_dir)
        Path(write_dir).mkdir(parents=True, exist_ok=True)
        log_file = str(root / "logs" / round_dir / exp_id / "train.log")
        scripts_base = str(root / "scripts" / round_dir)
    else:
        write_dir = results_dir
        log_file = str(root / "logs" / exp_id / "train.log")
        scripts_base = str(root / "scripts")

    config_path = Path(write_dir) / f"{exp_id}.json"
    config_path.write_text(json.dumps({"exp_id": exp_id, "config": config, "status": "pending"}, indent=2))
    script_path = generate_script(scripts_base, exp_id, train_command, gpu_id, log_file)

    print(json.dumps({
        "exp_id": exp_id,
        "exp_root": exp_root,
        "config_path": str(config_path),
        "script_path": script_path,
        "log_file": log_file,
    }, indent=2))
