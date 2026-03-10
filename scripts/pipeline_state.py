#!/usr/bin/env python3
"""State validation and pipeline resumption utilities."""

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def validate_phase_requirements(phase: int, exp_root: str) -> dict:
    """Validate that prerequisites for a given pipeline phase are met.

    Phase 2 (prerequisites): no file requirements.
    Phase 3 (baseline): exp_root/results/ directory must exist.
        If prerequisites.json exists and ready_for_baseline is false, fail.
    Phase 4 (checkpoint): exp_root/results/baseline.json must exist with
        "metrics" and "config" keys.
    Phase 5 (research): exp_root/results/baseline.json must exist.
    Phase 6 (experiment loop): baseline.json must exist with metrics+config,
        and if implementation-manifest.json exists it must have a "proposals" key.
    """
    root = Path(exp_root)
    missing: list[str] = []
    warnings: list[str] = []

    if phase == 2:
        # Prerequisites phase — no file-based requirements
        pass

    elif phase == 3:
        results_dir = root / "results"
        if not results_dir.is_dir():
            missing.append("results/ directory does not exist")
        else:
            prereq_path = results_dir / "prerequisites.json"
            if prereq_path.is_file():
                try:
                    prereq = json.loads(prereq_path.read_text())
                    if prereq.get("ready_for_baseline") is False:
                        missing.append(
                            "prerequisites.json indicates ready_for_baseline=false"
                        )
                except (json.JSONDecodeError, OSError):
                    warnings.append("prerequisites.json is not valid JSON")

    elif phase == 4:
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

    elif phase == 5:
        baseline_path = root / "results" / "baseline.json"
        if not baseline_path.is_file():
            missing.append("results/baseline.json does not exist")

    elif phase == 6:
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

    else:
        warnings.append(f"No validation rules defined for phase {phase}")

    return {
        "valid": len(missing) == 0,
        "phase": phase,
        "missing": missing,
        "warnings": warnings,
    }


def save_state(
    phase: int,
    iteration: int,
    running_exp_ids: list[str],
    exp_root: str,
    *,
    user_choices: dict | None = None,
) -> str:
    """Write pipeline-state.json to exp_root.

    Args:
        user_choices: Optional dict of Phase 0 user choices to persist
            (e.g., primary_metric, divergence_metric, divergence_lower_is_better,
            lower_is_better, target_value, train_command, eval_command,
            train_data_path, val_data_path, prepared_train_path,
            prepared_val_path, env_manager, env_name, model_category,
            user_papers, budget_mode, difficulty, difficulty_multiplier,
            method_proposal_scope, method_proposal_iterations,
            hp_batches_per_round). These are preserved across pipeline
            resumptions.

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
    if user_choices:
        state["user_choices"] = user_choices

    state_path = root / "pipeline-state.json"
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(root), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, str(state_path))
    except BaseException:
        os.unlink(tmp_path)
        raise

    # Backup user_choices separately for recovery if main state corrupts
    if user_choices:
        backup_path = root / "user-choices-backup.json"
        try:
            tmp_fd2, tmp_path2 = tempfile.mkstemp(dir=str(root), suffix=".tmp")
            with os.fdopen(tmp_fd2, "w") as f:
                json.dump(user_choices, f, indent=2)
            os.replace(tmp_path2, str(backup_path))
        except OSError:
            pass  # Best-effort backup — don't fail the main save

    return str(state_path)


def load_state(exp_root: str) -> dict | None:
    """Read pipeline-state.json if it exists.

    Returns the state dict, or None if no state file exists.
    If the main state file is corrupt but user-choices-backup.json exists,
    returns a minimal state dict with the recovered user_choices.
    """
    root = Path(exp_root)
    state_path = root / "pipeline-state.json"
    if not state_path.is_file():
        return None
    try:
        return json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        # Main state corrupt — attempt to recover user_choices from backup
        backup_path = root / "user-choices-backup.json"
        if backup_path.is_file():
            try:
                user_choices = json.loads(backup_path.read_text())
                return {
                    "phase": 0,
                    "iteration": 0,
                    "running_experiments": [],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "recovered",
                    "user_choices": user_choices,
                }
            except (json.JSONDecodeError, OSError):
                pass
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
                    tmp_fd, tmp_path = tempfile.mkstemp(
                        dir=str(root), suffix=".tmp"
                    )
                    try:
                        with os.fdopen(tmp_fd, "w") as f:
                            json.dump(state, f, indent=2)
                        os.replace(tmp_path, str(state_path))
                    except BaseException:
                        os.unlink(tmp_path)
                        raise
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
                    tmp_fd, tmp_path = tempfile.mkstemp(
                        dir=str(results_dir), suffix=".tmp"
                    )
                    try:
                        with os.fdopen(tmp_fd, "w") as f:
                            json.dump(data, f, indent=2)
                        os.replace(tmp_path, str(exp_file))
                    except BaseException:
                        os.unlink(tmp_path)
                        raise
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
        try:
            phase = int(sys.argv[3])
        except ValueError:
            print(f"Error: invalid phase '{sys.argv[3]}' (expected integer)")
            sys.exit(1)
        print(json.dumps(validate_phase_requirements(phase, exp_root), indent=2))

    elif action == "save":
        if len(sys.argv) < 5:
            print("Usage: pipeline_state.py <exp_root> save <phase> <iteration> [running_ids_json]")
            sys.exit(1)
        try:
            phase = int(sys.argv[3])
        except ValueError:
            print(f"Error: invalid phase '{sys.argv[3]}' (expected integer)")
            sys.exit(1)
        try:
            iteration = int(sys.argv[4])
        except ValueError:
            print(f"Error: invalid iteration '{sys.argv[4]}' (expected integer)")
            sys.exit(1)
        try:
            running_ids = json.loads(sys.argv[5]) if len(sys.argv) > 5 else []
        except json.JSONDecodeError:
            print(f"Error: invalid running_ids JSON '{sys.argv[5]}'")
            sys.exit(1)
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
