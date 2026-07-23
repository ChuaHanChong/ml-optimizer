#!/usr/bin/env python3
"""Set up experiment directory structure and generate configs/scripts.

Allocates the next globally-unique experiment ID (scanning flat and round
directories), creates the `<exp_root>/` subdirectory layout, writes a pending
config JSON, and generates a self-contained bash training script.

Usage:
    python3 experiment_setup.py <exp_root> <train_command>                              # Next exp on GPU 0, no config, flat results/
    python3 experiment_setup.py <exp_root> <train_command> <gpu_id>                     # Pin to a specific GPU index
    python3 experiment_setup.py <exp_root> <train_command> <gpu_id> <config_json>       # Embed an HP config dict in the result JSON
    python3 experiment_setup.py <exp_root> <train_command> <gpu_id> <config_json> <round_dir>  # Write under results/<round_dir>/
    python3 experiment_setup.py <exp_root> <train_command> <gpu_id> <config_json> <round_dir> <env_vars_json>  # Also export env vars in the script

gpu_id defaults to 0; config_json defaults to {}; round_dir routes output into
the round-based layout (results/<round_dir>/, logs/<round_dir>/, scripts/<round_dir>/);
env_vars_json (optional JSON dict, e.g. '{"MUJOCO_GL": "egl"}') is exported in the
generated script — used for headless-simulator env vars from baseline.json profiling.sim_env.

Examples:
    python3 experiment_setup.py <exp_root> "python train.py --lr 0.01"
    python3 experiment_setup.py <exp_root> "python train.py" 1 '{"lr": 0.001, "batch_size": 64}'
    python3 experiment_setup.py <exp_root> "python train.py" 0 '{"lr": 0.01}' round-1-hp
"""

import json
import re
import shlex
import sys
from pathlib import Path


def next_experiment_id(results_dir: str) -> str:
    """Next sequential exp id (exp-001, ...), scanning flat results/ and round-*/ dirs for globally unique ids (strict `exp-\\d+\\.json` only)."""
    path = Path(results_dir)
    if not path.exists():
        return "exp-001"

    exp_pattern = re.compile(r"^exp-(\d+)\.json$")
    nums: list[int] = []

    def _scan(d: Path) -> None:
        for f in d.iterdir():
            if f.is_file() and (m := exp_pattern.match(f.name)):
                nums.append(int(m.group(1)))

    _scan(path)
    for rdir in path.iterdir():
        if rdir.is_dir() and rdir.name.startswith("round-"):
            _scan(rdir)

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
    """Generate a self-contained bash script at `<scripts_dir>/<exp_id>/<script_name>` (GPU, logging, PID, optional timeout).

    `script_name` distinguishes scripts per experiment (train.sh default, eval.sh, preprocess.sh).
    The (optionally `timeout`-wrapped) command runs in the BACKGROUND with output to `log_file`;
    the training PID (`$!`) is written to the pid file (monitor kills training, not the wrapper),
    then the script `wait`s and propagates the REAL exit code. Exit code 124 is rewritten to success
    ONLY when *time_budget* is set (budget reached = normal stop). *checkpoint_path*, when given, is
    exported as `CHECKPOINT_PATH` for warm-starting. `log_file` is required (a round-based path).
    """
    if log_file is None:
        raise ValueError(
            "log_file is required — pass a round-based path like "
            "'<exp_root>/logs/round-N-<type>/<exp-id>/train.log' "
            "(or '<exp_root>/logs/baseline/train.log' for baseline)."
        )

    label = script_name.removesuffix(".sh")
    lines = [
        "#!/bin/bash",
        f"# Experiment: {exp_id} ({label})",
        "set -e",
        "set -o pipefail",
        "",
    ]

    # Environment variables
    lines.append(f"export CUDA_VISIBLE_DEVICES={gpu_id}")
    if env_vars:
        for key, value in env_vars.items():
            lines.append(f"export {key}={shlex.quote(str(value))}")
    if checkpoint_path:
        lines.append(f"export CHECKPOINT_PATH={shlex.quote(str(checkpoint_path))}")
        lines.append(f"echo {shlex.quote(f'Warm-starting from checkpoint: {checkpoint_path}')}")
    lines.append("")

    # Create log directory
    log_dir = str(Path(log_file).parent)
    lines.append(f"mkdir -p {shlex.quote(log_dir)}")
    lines.append("")

    # Background launch: redirect to log, record the TRAINING pid ($!), wait, propagate real exit code.
    # (`echo $$` would record the wrapper's pid; `cmd | tee` would hide the command's exit code.)
    lines.append(f"echo {shlex.quote(f'Starting {label} for {exp_id} on GPU {gpu_id}')}")
    if time_budget is not None and time_budget > 0:
        lines.append(f"echo {shlex.quote(f'Time budget: {time_budget}s')}")
        lines.append(
            f"timeout --signal=SIGTERM --kill-after=60 {time_budget} "
            f"{command} > {shlex.quote(log_file)} 2>&1 &"
        )
    else:
        lines.append(f"{command} > {shlex.quote(log_file)} 2>&1 &")
    lines.append(f"echo $! > {shlex.quote(log_dir + '/pid')}")
    lines.append("EXIT_CODE=0")
    lines.append("wait $! || EXIT_CODE=$?")
    if time_budget is not None and time_budget > 0:
        lines.append("if [ $EXIT_CODE -eq 124 ]; then")
        lines.append(f"    echo {shlex.quote(f'Time budget reached ({time_budget}s) — stopped normally')}")
        lines.append("    EXIT_CODE=0")
        lines.append("fi")
    lines.append("")
    lines.append(f"echo {shlex.quote(f'{label} for {exp_id} finished with exit code')} \"$EXIT_CODE\"")
    lines.append("exit $EXIT_CODE")

    script_dir = Path(scripts_dir) / exp_id
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / script_name
    script_path.write_text("\n".join(lines))
    script_path.chmod(0o755)
    return str(script_path)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: experiment_setup.py <exp_root> <train_command> [gpu_id] [config_json] [round_dir] [env_vars_json]')
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
    try:
        env_vars = json.loads(sys.argv[6]) if len(sys.argv) > 6 else None
    except json.JSONDecodeError:
        print(f"Error: invalid env_vars JSON '{sys.argv[6]}'")
        sys.exit(1)

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
    script_path = generate_script(scripts_base, exp_id, train_command, gpu_id, log_file,
                                  env_vars=env_vars)

    print(json.dumps({
        "exp_id": exp_id,
        "exp_root": exp_root,
        "config_path": str(config_path),
        "script_path": script_path,
        "log_file": log_file,
    }, indent=2))
