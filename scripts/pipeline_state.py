#!/usr/bin/env python3
"""State validation and pipeline resumption utilities.

Manages pipeline-state.json (phase, iteration, user choices, baseline checksum),
phase-gate validation/logging, baseline integrity verification, stale-experiment
cleanup, and the decision log (with replay checks).

CLI: python3 pipeline_state.py <exp_root> <action> [args] where <action> is one of
validate <phase> | save <phase> <iteration> [running_ids_json] | load | cleanup |
verify-baseline (exit 1 on mismatch) | gate <current_phase> <next_phase> |
log-gate <phase> <status> <summary> | log-decision <json_data> |
replay-check <decision_id> <current_decision> |
decisions [--phase N] [--agent NAME] [--type TYPE].
"""

import fcntl
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Make sibling scripts (error_tracker) importable when pipeline_state.py is
# loaded via `python3 scripts/pipeline_state.py ...` or `from pipeline_state
# import ...` inside the plugin scripts/ dir.
sys.path.insert(0, str(Path(__file__).parent))
from error_tracker import create_event, log_event  # noqa: E402


def _atomic_write_json(target_path, data, tmp_dir) -> None:
    """Atomically write `data` as JSON to target_path (temp file + os.replace)."""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(tmp_dir), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, str(target_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _compute_baseline_checksum(metrics: dict) -> str:
    """Deterministic SHA-256 of a baseline metrics dict (canonical JSON, reproducible)."""
    canonical = json.dumps(metrics, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_baseline_integrity(exp_root: str) -> dict:
    """Verify baseline.json metrics haven't changed since checksum was stored.

    Returns {"valid": bool, "error": str|None, "checksum": str|None}.
    Legacy pipelines without a stored checksum return valid=True with a warning.
    """
    state = load_state(exp_root)

    if state is None:
        return {"valid": True, "error": None, "checksum": None,
                "warning": "No pipeline state found"}

    stored = state.get("baseline_checksum")
    if stored is None:
        return {"valid": True, "error": None, "checksum": None,
                "warning": "No baseline_checksum in pipeline state (legacy pipeline)"}

    baseline_path = Path(exp_root) / "results" / "baseline.json"
    if not baseline_path.is_file():
        return {"valid": False, "error": "baseline.json does not exist",
                "checksum": stored}

    try:
        data = json.loads(baseline_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return {"valid": False, "error": f"baseline.json is not valid JSON: {e}",
                "checksum": stored}

    metrics = data.get("metrics")
    if metrics is None:
        return {"valid": False, "error": "baseline.json missing 'metrics' key",
                "checksum": stored}

    current = _compute_baseline_checksum(metrics)
    if current != stored:
        return {
            "valid": False,
            "error": (f"Baseline integrity check FAILED: metrics checksum mismatch. "
                      f"Stored: {stored[:16]}..., Current: {current[:16]}..."),
            "checksum": stored,
            "current_checksum": current,
        }

    return {"valid": True, "error": None, "checksum": stored}


def _check_baseline_metrics_config(root: Path, missing: list) -> None:
    """Require results/baseline.json to exist with 'metrics' and 'config' keys."""
    baseline_path = root / "results" / "baseline.json"
    if not baseline_path.is_file():
        missing.append("results/baseline.json does not exist")
        return
    try:
        data = json.loads(baseline_path.read_text())
    except (json.JSONDecodeError, OSError):
        missing.append("results/baseline.json is not valid JSON")
        data = {}
    if "metrics" not in data:
        missing.append("results/baseline.json missing 'metrics' key")
    if "config" not in data:
        missing.append("results/baseline.json missing 'config' key")


def _warn_manifest_proposals(manifest_path: Path, warnings: list) -> None:
    """Warn if an existing implementation-manifest.json is invalid or lacks 'proposals'."""
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        warnings.append("implementation-manifest.json is not valid JSON")
        manifest = {}
    if "proposals" not in manifest:
        warnings.append("implementation-manifest.json missing 'proposals' key")


def validate_phase_requirements(phase: int, exp_root: str) -> dict:
    """Validate that prerequisites for a given pipeline phase are met.

    Phase 2: none. Phase 3: results/ dir (fail if prerequisites.json has
    ready_for_baseline=false). Phases 4/6/7: baseline.json with metrics+config
    (7 also needs optimization-goals.json). Phase 5: baseline.json exists.
    Phase 6/8: manifest 'proposals' check (8 requires the manifest). Phase 9:
    baseline.json + rounds-manifest + a round dir (last two are warnings).
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
        _check_baseline_metrics_config(root, missing)

    elif phase == 5:
        if not (root / "results" / "baseline.json").is_file():
            missing.append("results/baseline.json does not exist")

    elif phase == 6:
        _check_baseline_metrics_config(root, missing)
        manifest_path = root / "results" / "implementation-manifest.json"
        if manifest_path.is_file():
            _warn_manifest_proposals(manifest_path, warnings)

    elif phase == 7:
        _check_baseline_metrics_config(root, missing)
        if not (root / "optimization-goals.json").is_file():
            missing.append("optimization-goals.json does not exist")

    elif phase == 8:
        # Stacking: baseline.json + implementation-manifest.json required
        if not (root / "results" / "baseline.json").is_file():
            missing.append("results/baseline.json does not exist")
        manifest_path = root / "results" / "implementation-manifest.json"
        if not manifest_path.is_file():
            missing.append(
                "results/implementation-manifest.json does not exist"
                " (stacking requires method branches)"
            )
        else:
            _warn_manifest_proposals(manifest_path, warnings)

    elif phase == 9:
        # Report: baseline.json + rounds-manifest.json must exist
        baseline_path = root / "results" / "baseline.json"
        if not baseline_path.is_file():
            missing.append("results/baseline.json does not exist")
        manifest_path = root / "results" / "rounds-manifest.json"
        if not manifest_path.is_file():
            warnings.append("results/rounds-manifest.json does not exist (no round history)")
        # At least one round directory with results should exist
        results_path = root / "results"
        if results_path.is_dir():
            round_dirs = [d for d in results_path.iterdir()
                          if d.is_dir() and d.name.startswith("round-")]
            if not round_dirs:
                warnings.append("No round directories found in results/")

    else:
        warnings.append(f"No validation rules defined for phase {phase}")

    return {
        "valid": len(missing) == 0,
        "phase": phase,
        "missing": missing,
        "warnings": warnings,
    }


def _check_goal_metric_consistency(exp_root: str, user_choices: dict) -> None:
    """Warn (stderr + error tracker) if user_choices.primary_metric disagrees with
    optimization-goals.json — a stale dispatch metric would drift silently otherwise.

    No-op when either side is absent (early Phase 0, no goals yet).
    """
    metric = user_choices.get("primary_metric") if isinstance(user_choices, dict) else None
    if not metric:
        return

    goals_path = Path(exp_root) / "optimization-goals.json"
    if not goals_path.is_file():
        return
    try:
        goals = json.loads(goals_path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    goal_metric = (goals.get("objective") or {}).get("primary_metric")
    if not goal_metric or goal_metric == metric:
        return

    msg = (
        f"goal-metric mismatch: optimization-goals.json primary_metric="
        f"{goal_metric!r} but user_choices primary_metric={metric!r}. "
        "Analysis/hp-tune will key off user_choices; the goal anchor will drift."
    )
    print(f"[pipeline_state] WARNING: {msg}", file=sys.stderr)

    # Closest existing category/source (error_tracker rejects unknown values).
    try:
        event = create_event(
            category="pipeline_inefficiency",
            severity="warning",
            source="orchestrate",
            message=f"goal_metric_mismatch: {msg}",
            config={"goal_metric": goal_metric, "user_choices_metric": metric},
        )
        log_event(exp_root, event)
    except Exception:
        pass  # never let logging block save_state — stderr warning is the contract


def save_state(
    phase: int,
    iteration: int,
    running_exp_ids: list[str],
    exp_root: str,
    *,
    user_choices: dict | None = None,
    consecutive_stop_count: int | None = None,
    stuck_protocol_triggered: bool | None = None,
    baseline_checksum: str | None = None,
) -> str:
    """Write pipeline-state.json to exp_root, returning its path.

    user_choices: Phase 0 choices (primary_metric, secondary_metrics, eval_tasks,
        budgets, commands, etc.), preserved across resumptions. consecutive_stop_count: orchestrator
        exit-judgment signal (NOT a hardcoded threshold). stuck_protocol_triggered:
        prevents infinite recovery loops. Any arg left None keeps its prior value.
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
    # Load existing state once if needed for preserving fields
    existing = (
        load_state(exp_root)
        if (user_choices is None or consecutive_stop_count is None
            or stuck_protocol_triggered is None or baseline_checksum is None)
        else None
    )

    # Preserve each field: use the passed value, else carry the existing one.
    if user_choices is not None:
        state["user_choices"] = user_choices
        _check_goal_metric_consistency(exp_root, user_choices)
    elif existing and existing.get("user_choices"):
        state["user_choices"] = existing["user_choices"]
    for field, value in (("consecutive_stop_count", consecutive_stop_count),
                         ("stuck_protocol_triggered", stuck_protocol_triggered),
                         ("baseline_checksum", baseline_checksum)):
        if value is not None:
            state[field] = value
        elif existing and field in existing:
            state[field] = existing[field]

    state_path = root / "pipeline-state.json"
    _atomic_write_json(state_path, state, root)

    # Backup user_choices separately for recovery if main state corrupts (best-effort)
    if user_choices is not None:
        try:
            _atomic_write_json(root / "user-choices-backup.json", user_choices, root)
        except OSError:
            pass

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


def _elapsed_hours(ts_str: str, now: datetime) -> float:
    """Hours between an ISO timestamp (assumed UTC if naive) and now; raises on bad input."""
    ts = datetime.fromisoformat(ts_str)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).total_seconds() / 3600.0


def cleanup_stale(exp_root: str, timeout_hours: float = 2.0) -> list[str]:
    """Mark stale "running" items as interrupted/failed.

    Checks pipeline-state.json against *timeout_hours*, and scans
    results/round-*/exp-*.json (plus legacy flat results/exp-*.json) against each
    result's own timeout — 3x time_budget_seconds when present (the existing
    baseline_training_time*3 rule), else *timeout_hours*. Returns cleaned descriptions.
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
            try:
                if _elapsed_hours(state.get("timestamp", ""), now) > timeout_hours:
                    state["status"] = "interrupted"
                    state["interrupted_at"] = now.isoformat()
                    _atomic_write_json(state_path, state, root)
                    cleaned.append("pipeline-state.json marked as interrupted")
            except (ValueError, TypeError):
                pass

    # --- <exp_root>/results/round-*/exp-*.json (+ legacy flat exp-*.json) ---
    results_dir = root / "results"
    if results_dir.is_dir():
        exp_files = sorted(results_dir.glob("exp-*.json")) + sorted(
            results_dir.glob("round-*/exp-*.json")
        )
        for exp_file in exp_files:
            try:
                data = json.loads(exp_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if data.get("status") != "running":
                continue
            # Threshold from the experiment's own timeout when known, else default.
            threshold_hours = timeout_hours
            tb = data.get("time_budget_seconds")
            if isinstance(tb, (int, float)) and not isinstance(tb, bool) and tb > 0:
                threshold_hours = max(timeout_hours, 3.0 * tb / 3600.0)
            try:
                if _elapsed_hours(data.get("timestamp", ""), now) > threshold_hours:
                    data["status"] = "failed"
                    data["note"] = "Marked as stale: exceeded timeout"
                    _atomic_write_json(exp_file, data, exp_file.parent)
                    cleaned.append(f"{exp_file.name} marked as failed (stale)")
            except (ValueError, TypeError):
                continue

    return cleaned



# ---------------------------------------------------------------------------
# Phase Gate Protocol
# ---------------------------------------------------------------------------

PHASE_TRANSITIONS = {
    0: [1], 1: [2], 2: [3], 3: [4], 4: [5, 7], 5: [6], 6: [7], 7: [8, 9], 8: [7, 9], 9: []
}

RECOVERY_INSTRUCTIONS = {
    (2, "dependency_missing"): "Install missing packages and retry prerequisites",
    (3, "training_crash"): "Check training command, reduce batch size, retry baseline",
    (3, "eval_failed"): "Verify eval command and data paths, retry baseline",
    (5, "search_failed"): "Retry with source='knowledge' (LLM-only fallback)",
    (6, "implementation_failed"): "Check implementation errors, try from_scratch strategy",
    (7, "all_diverged"): "Halve learning rates and retry batch",
    (7, "all_timeout"): "Reduce training budget or increase timeout multiplier",
    (7, "stuck"): "Dispatch research for fresh ideas, or try code_evolution",
}

_RECOVERY_DEFAULT = "Check error logs and retry the current phase"


def validate_phase_gate(current_phase, next_phase, exp_root):
    """Validate a phase transition is allowed and prerequisites met -> {valid, errors}."""
    errors = []

    # 1. Check transition legality
    allowed = PHASE_TRANSITIONS.get(current_phase)
    if allowed is None:
        return {"valid": False, "errors": [f"Unknown current phase: {current_phase}"]}
    if next_phase not in allowed:
        errors.append(
            f"Transition {current_phase} -> {next_phase} is not allowed "
            f"(valid targets: {allowed})"
        )

    # 2. Check that current_phase has status "completed" in gate log
    #    (phase 0 is exempt — no gate log requirement)
    if current_phase != 0:
        gate_path = Path(exp_root) / "phase-gates.json"
        if not gate_path.is_file():
            errors.append(
                f"Phase gate log does not exist; phase {current_phase} "
                f"has no completion record"
            )
        else:
            try:
                entries = json.loads(gate_path.read_text())
            except (json.JSONDecodeError, OSError):
                entries = []
            # Find the latest entry for current_phase
            completed = False
            for entry in reversed(entries):
                if entry.get("phase") == current_phase:
                    if entry.get("status") == "completed":
                        completed = True
                    break
            if not completed:
                errors.append(
                    f"Phase {current_phase} has not been marked as completed "
                    f"in the gate log"
                )

    # 3. Validate next_phase requirements via existing function
    req = validate_phase_requirements(next_phase, exp_root)
    if not req["valid"]:
        for m in req["missing"]:
            errors.append(f"Phase {next_phase} requirement: {m}")

    return {"valid": len(errors) == 0, "errors": errors}


def log_phase_gate(exp_root, phase, status, summary):
    """Append {phase, status, summary, timestamp} to phase-gates.json (atomic + flock)."""
    root = Path(exp_root)
    root.mkdir(parents=True, exist_ok=True)
    gate_path = root / "phase-gates.json"
    lock_path = gate_path.with_suffix(".lock")

    entry = {
        "phase": phase,
        "status": status,
        "summary": summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            if gate_path.is_file():
                try:
                    entries = json.loads(gate_path.read_text())
                except (json.JSONDecodeError, OSError):
                    entries = []
            else:
                entries = []

            entries.append(entry)
            _atomic_write_json(gate_path, entries, root)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def get_recovery_instruction(phase, failure_type):
    """Recovery instruction for a (phase, failure_type), or a generic default."""
    return RECOVERY_INSTRUCTIONS.get((phase, failure_type), _RECOVERY_DEFAULT)


# ---------------------------------------------------------------------------
# Decision Logging
# ---------------------------------------------------------------------------

_DECISION_REQUIRED_FIELDS = ("phase", "agent", "decision_type", "decision")


def _compute_inputs_hash(decision: dict) -> str:
    """SHA-256 of canonical JSON of the input-context fields present in the decision."""
    input_keys = ("phase", "iteration", "agent", "decision_type", "context_summary")
    inputs = {k: decision[k] for k in input_keys if k in decision}
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_decision_log(exp_root: str) -> list[dict]:
    """Load decision-log.json, returning [] if absent or corrupt."""
    log_path = Path(exp_root) / "decision-log.json"
    if not log_path.is_file():
        return []
    try:
        data = json.loads(log_path.read_text())
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, OSError):
        return []


def _save_decision_log(exp_root: str, entries: list[dict]) -> None:
    """Atomically write decision-log.json."""
    root = Path(exp_root)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(root / "decision-log.json", entries, root)




def log_decision(exp_root: str, decision: dict) -> str:
    """Append a decision to decision-log.json and return its auto-incremented id.

    *decision* requires phase, agent, decision_type, decision (raises ValueError if
    missing); optional iteration, reasoning, context_summary. Adds id, ISO8601 UTC
    timestamp, and inputs_hash. Atomic write under flock.
    """
    missing = [f for f in _DECISION_REQUIRED_FIELDS if f not in decision]
    if missing:
        raise ValueError(f"Missing required decision fields: {', '.join(missing)}")

    root = Path(exp_root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "decision-log.lock"

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            entries = _load_decision_log(exp_root)

            # Auto-increment id from existing entries
            max_id = 0
            for entry in entries:
                try:
                    entry_id = int(entry.get("id", 0))
                    if entry_id > max_id:
                        max_id = entry_id
                except (ValueError, TypeError):
                    pass
            new_id = str(max_id + 1)

            entry = dict(decision)
            entry["id"] = new_id
            entry["timestamp"] = datetime.now(timezone.utc).isoformat()
            entry["inputs_hash"] = _compute_inputs_hash(decision)

            entries.append(entry)
            _save_decision_log(exp_root, entries)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)

    return new_id


def get_decisions(
    exp_root: str,
    phase: int | None = None,
    agent: str | None = None,
    decision_type: str | None = None,
) -> list[dict]:
    """Load decision-log.json, filtered by any of phase/agent/decision_type."""
    entries = _load_decision_log(exp_root)
    result = []
    for entry in entries:
        if phase is not None and entry.get("phase") != phase:
            continue
        if agent is not None and entry.get("agent") != agent:
            continue
        if decision_type is not None and entry.get("decision_type") != decision_type:
            continue
        result.append(entry)
    return result


def replay_check(exp_root: str, decision_id: str, current_decision: str) -> dict:
    """Compare *current_decision* against a prior logged decision with the same inputs_hash.

    Finds the entry with *decision_id*, then any other entry sharing its inputs_hash.
    Returns {match, original_decision, current_decision, divergence}; if no such prior
    exists, {"match": None, "divergence": "no_prior_decision_with_matching_inputs"}.
    """
    no_prior = {"match": None, "divergence": "no_prior_decision_with_matching_inputs"}
    entries = _load_decision_log(exp_root)

    # Find the target entry by id
    target = None
    for entry in entries:
        if str(entry.get("id")) == str(decision_id):
            target = entry
            break

    if target is None:
        return no_prior

    target_hash = target.get("inputs_hash")
    if target_hash is None:
        return no_prior

    # Search for another entry with the same inputs_hash
    original = None
    for entry in entries:
        if (entry.get("inputs_hash") == target_hash
                and str(entry.get("id")) != str(decision_id)):
            original = entry
            break

    if original is None:
        return no_prior

    original_decision = original.get("decision", "")
    if original_decision == current_decision:
        return {
            "match": True,
            "original_decision": original_decision,
            "current_decision": current_decision,
            "divergence": None,
        }

    return {
        "match": False,
        "original_decision": original_decision,
        "current_decision": current_decision,
        "divergence": (
            f"Decision diverged: original='{original_decision}', "
            f"current='{current_decision}'"
        ),
    }


def _int_arg(raw: str, name: str) -> int:
    """Parse a CLI integer arg or print an error and exit 1."""
    try:
        return int(raw)
    except ValueError:
        print(f"Error: invalid {name} '{raw}' (expected integer)")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: pipeline_state.py <exp_root> "
            "[validate <phase>|save <phase> <iteration>|load|cleanup"
            "|gate <current_phase> <next_phase>|log-gate <phase> <status> <summary>"
            "|log-decision <json_data>|replay-check <decision_id> <current_decision>"
            "|decisions [--phase N] [--agent NAME] [--type TYPE]]"
        )
        sys.exit(1)

    exp_root = sys.argv[1]
    action = sys.argv[2]

    if action == "validate":
        if len(sys.argv) < 4:
            print("Usage: pipeline_state.py <exp_root> validate <phase>")
            sys.exit(1)
        phase = _int_arg(sys.argv[3], "phase")
        print(json.dumps(validate_phase_requirements(phase, exp_root), indent=2))

    elif action == "save":
        if len(sys.argv) < 5:
            print("Usage: pipeline_state.py <exp_root> save <phase> <iteration> [running_ids_json]")
            sys.exit(1)
        phase = _int_arg(sys.argv[3], "phase")
        iteration = _int_arg(sys.argv[4], "iteration")
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

    elif action == "verify-baseline":
        result = verify_baseline_integrity(exp_root)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["valid"] else 1)

    elif action == "gate":
        if len(sys.argv) < 5:
            print("Usage: pipeline_state.py <exp_root> gate <current_phase> <next_phase>")
            sys.exit(1)
        current_phase = _int_arg(sys.argv[3], "current_phase")
        next_phase = _int_arg(sys.argv[4], "next_phase")
        print(json.dumps(validate_phase_gate(current_phase, next_phase, exp_root)))

    elif action == "log-gate":
        if len(sys.argv) < 6:
            print("Usage: pipeline_state.py <exp_root> log-gate <phase> <status> <summary>")
            sys.exit(1)
        phase = _int_arg(sys.argv[3], "phase")
        log_phase_gate(exp_root, phase, sys.argv[4], sys.argv[5])
        print(json.dumps({"logged": True, "phase": phase, "status": sys.argv[4]}))

    elif action == "log-decision":
        if len(sys.argv) < 4:
            print("Usage: pipeline_state.py <exp_root> log-decision <json_data>")
            sys.exit(1)
        try:
            decision_data = json.loads(sys.argv[3])
        except json.JSONDecodeError as e:
            print(f"Error: invalid JSON: {e}")
            sys.exit(1)
        try:
            decision_id = log_decision(exp_root, decision_data)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(json.dumps({"logged": True, "id": decision_id}))

    elif action == "decisions":
        kwargs_d: dict = {}
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--phase" and i + 1 < len(sys.argv):
                try:
                    kwargs_d["phase"] = int(sys.argv[i + 1])
                except ValueError:
                    print(f"Error: invalid phase '{sys.argv[i + 1]}'")
                    sys.exit(1)
                i += 2
            elif sys.argv[i] == "--agent" and i + 1 < len(sys.argv):
                kwargs_d["agent"] = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--type" and i + 1 < len(sys.argv):
                kwargs_d["decision_type"] = sys.argv[i + 1]
                i += 2
            else:
                print(f"Error: unexpected argument '{sys.argv[i]}'")
                sys.exit(1)
        result = get_decisions(exp_root, **kwargs_d)
        print(json.dumps(result, indent=2))
    elif action == "replay-check":
        if len(sys.argv) < 5:
            print("Usage: pipeline_state.py <exp_root> replay-check <decision_id> <current_decision>")
            sys.exit(1)
        result = replay_check(exp_root, sys.argv[3], sys.argv[4])
        print(json.dumps(result, indent=2))

    else:
        print(f"Unknown action: {action}")
        sys.exit(1)
