#!/usr/bin/env python3
"""State validation and pipeline resumption utilities."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def validate_phase_requirements(phase: int, exp_root: str) -> dict:
    """Validate that prerequisites for a given pipeline phase are met.

    Phase 2 (baseline): exp_root/results/ directory must exist.
    Phase 3 (checkpoint): exp_root/results/baseline.json must exist with
        "metrics" and "config" keys.
    Phase 4 (research): exp_root/results/baseline.json must exist.
    Phase 5 (experiment loop): baseline.json must exist with metrics+config,
        and if implementation-manifest.json exists it must have a "proposals" key.
    """
    root = Path(exp_root)
    missing: list[str] = []
    warnings: list[str] = []

    if phase == 2:
        results_dir = root / "results"
        if not results_dir.is_dir():
            missing.append("results/ directory does not exist")

    elif phase == 3:
        baseline_path = root / "results" / "baseline.json"
        if not baseline_path.is_file():
            missing.append("results/baseline.json does not exist")
        else:
            try:
                data = json.loads(baseline_path.read_text())
            except (json.JSONDecodeError, OSError):
                missing.append("results/baseline.json is not valid JSON")
                data = {}
            if "metrics" not in data:
                missing.append("results/baseline.json missing 'metrics' key")
            if "config" not in data:
                missing.append("results/baseline.json missing 'config' key")

    elif phase == 4:
        baseline_path = root / "results" / "baseline.json"
        if not baseline_path.is_file():
            missing.append("results/baseline.json does not exist")

    elif phase == 5:
        baseline_path = root / "results" / "baseline.json"
        if not baseline_path.is_file():
            missing.append("results/baseline.json does not exist")
        else:
            try:
                data = json.loads(baseline_path.read_text())
            except (json.JSONDecodeError, OSError):
                missing.append("results/baseline.json is not valid JSON")
                data = {}
            if "metrics" not in data:
                missing.append("results/baseline.json missing 'metrics' key")
            if "config" not in data:
                missing.append("results/baseline.json missing 'config' key")

        manifest_path = root / "results" / "implementation-manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                warnings.append("implementation-manifest.json is not valid JSON")
                manifest = {}
            if "proposals" not in manifest:
                warnings.append("implementation-manifest.json missing 'proposals' key")

    return {
        "valid": len(missing) == 0,
        "phase": phase,
        "missing": missing,
        "warnings": warnings,
    }


def save_state(phase: int, iteration: int, running_exp_ids: list[str], exp_root: str) -> str:
    """Write pipeline-state.json to exp_root.

    Returns the path to the state file.
    """
    root = Path(exp_root)
    root.mkdir(parents=True, exist_ok=True)

    state = {
        "phase": phase,
        "iteration": iteration,
        "running_experiments": running_exp_ids,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }

    state_path = root / "pipeline-state.json"
    state_path.write_text(json.dumps(state, indent=2))
    return str(state_path)


def load_state(exp_root: str) -> dict | None:
    """Read pipeline-state.json if it exists.

    Returns the state dict, or None if no state file exists.
    """
    state_path = Path(exp_root) / "pipeline-state.json"
    if not state_path.is_file():
        return None
    try:
        return json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def cleanup_stale(exp_root: str, timeout_hours: float = 2.0) -> list[str]:
    """Mark stale running items as interrupted/failed.

    Reads pipeline-state.json and checks if status is "running" with a
    timestamp older than *timeout_hours* ago.  Also scans
    experiments/results/ for any exp-*.json with status "running" older
    than the timeout.

    Returns a list of cleaned-up item descriptions.
    """
    cleaned: list[str] = []
    now = datetime.now(timezone.utc)
    root = Path(exp_root)

    # --- pipeline-state.json ---
    state_path = root / "pipeline-state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            state = {}

        if state.get("status") == "running":
            ts_str = state.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                elapsed = (now - ts).total_seconds() / 3600.0
                if elapsed > timeout_hours:
                    state["status"] = "interrupted"
                    state["interrupted_at"] = now.isoformat()
                    state_path.write_text(json.dumps(state, indent=2))
                    cleaned.append("pipeline-state.json marked as interrupted")
            except (ValueError, TypeError):
                pass

    # --- experiments/results/exp-*.json ---
    results_dir = root / "results"
    if results_dir.is_dir():
        for exp_file in sorted(results_dir.glob("exp-*.json")):
            try:
                data = json.loads(exp_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if data.get("status") != "running":
                continue
            ts_str = data.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                elapsed = (now - ts).total_seconds() / 3600.0
                if elapsed > timeout_hours:
                    data["status"] = "failed"
                    data["note"] = "Marked as stale: exceeded timeout"
                    exp_file.write_text(json.dumps(data, indent=2))
                    cleaned.append(f"{exp_file.name} marked as failed (stale)")
            except (ValueError, TypeError):
                continue

    return cleaned


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: pipeline_state.py <exp_root> "
            "[validate <phase>|save <phase> <iteration>|load|cleanup]"
        )
        sys.exit(1)

    exp_root = sys.argv[1]
    action = sys.argv[2]

    if action == "validate":
        if len(sys.argv) < 4:
            print("Usage: pipeline_state.py <exp_root> validate <phase>")
            sys.exit(1)
        phase = int(sys.argv[3])
        print(json.dumps(validate_phase_requirements(phase, exp_root), indent=2))

    elif action == "save":
        if len(sys.argv) < 5:
            print("Usage: pipeline_state.py <exp_root> save <phase> <iteration> [running_ids_json]")
            sys.exit(1)
        phase = int(sys.argv[3])
        iteration = int(sys.argv[4])
        running_ids = json.loads(sys.argv[5]) if len(sys.argv) > 5 else []
        path = save_state(phase, iteration, running_ids, exp_root)
        print(f"State saved to {path}")

    elif action == "load":
        state = load_state(exp_root)
        if state is None:
            print("No pipeline state found.")
        else:
            print(json.dumps(state, indent=2))

    elif action == "cleanup":
        cleaned = cleanup_stale(exp_root)
        if cleaned:
            print("Cleaned up:")
            for item in cleaned:
                print(f"  - {item}")
        else:
            print("Nothing to clean up.")

    else:
        print(f"Unknown action: {action}")
        sys.exit(1)
