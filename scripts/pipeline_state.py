#!/usr/bin/env python3
"""State validation and pipeline resumption utilities."""

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


def init_hyperagent_state(enabled: bool = True) -> dict:
    """Create initial hyperagent_state structure for pipeline-state.json.

    Called by the orchestrator in Phase 0. Hyperagent mode is ON by default —
    the hyperagent enables self-improvement and helps Phase 7 and Phase 8.
    All fields are initialized to defaults.
    """
    return {
        "enabled": enabled,
        "current_phase": "hp_tuning",
        "archive_generation": 0,
        "strategy_history": [],
        "meta_improvement_count": 0,
        "active_meta_patches": [],
        "operator_stats": {
            "hp_tune": {"attempts": 0, "improvements": 0},
            "llm_patch": {"attempts": 0, "improvements": 0},
            "shinka_evolve": {"attempts": 0, "improvements": 0},
            "research_implement": {"attempts": 0, "improvements": 0},
            "meta_improve": {"attempts": 0, "improvements": 0},
        },
    }


def _compute_baseline_checksum(metrics: dict) -> str:
    """Compute a deterministic SHA-256 checksum of a baseline metrics dict.

    Uses canonical JSON serialization (sorted keys, compact separators)
    so the hash is reproducible across runs.
    """
    canonical = json.dumps(metrics, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_baseline_integrity(exp_root: str) -> dict:
    """Verify baseline.json metrics haven't changed since checksum was stored.

    Returns {"valid": bool, "error": str|None, "checksum": str|None}.
    Backward-compatible: returns valid=True with warning for legacy pipelines
    without checksums.
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

    elif phase == 7:
        # Experiment loop: baseline.json must exist with metrics+config
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
        # Goals file must exist
        goals_path = root / "optimization-goals.json"
        if not goals_path.is_file():
            missing.append("optimization-goals.json does not exist")

    elif phase == 8:
        # Stacking: baseline.json + implementation-manifest.json required
        baseline_path = root / "results" / "baseline.json"
        if not baseline_path.is_file():
            missing.append("results/baseline.json does not exist")
        manifest_path = root / "results" / "implementation-manifest.json"
        if not manifest_path.is_file():
            missing.append(
                "results/implementation-manifest.json does not exist"
                " (stacking requires method branches)"
            )
        else:
            try:
                manifest = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                warnings.append("implementation-manifest.json is not valid JSON")
                manifest = {}
            if "proposals" not in manifest:
                warnings.append(
                    "implementation-manifest.json missing 'proposals' key"
                )

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
    """Warn if user_choices.primary_metric disagrees with optimization-goals.json.

    The orchestrator writes ``optimization-goals.json`` at Phase 0 and persists
    ``user_choices`` to ``pipeline-state.json`` separately. Nothing else checks
    they agree — so a stale dispatch metric can propagate silently through the
    whole pipeline. This helper surfaces the mismatch via stderr and, if
    available, logs a warning to the error tracker.

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

    # Log to error tracker. Use the closest existing category/source —
    # error_tracker rejects unknown values by design.
    # "pipeline_inefficiency" covers configuration drift;
    # "orchestrate" is the pipeline-level source.
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
    agent_registry: dict | None = None,
    hyperagent_state: dict | None = None,
) -> str:
    """Write pipeline-state.json to exp_root.

    Args:
        user_choices: Optional dict of Phase 0 user choices to persist
            (e.g., primary_metric, divergence_metric, divergence_lower_is_better,
            lower_is_better, target_value, train_command, eval_command,
            train_data_path, val_data_path, prepared_train_path,
            prepared_val_path, env_manager, env_name, model_category,
            user_papers, method_proposal_scope, method_proposal_iterations,
            hp_batches_per_round). These are preserved across pipeline
            resumptions.
        consecutive_stop_count: Optional counter for the
            3-consecutive-stop exit rule. Persisted at root level.
        stuck_protocol_triggered: Whether the stuck protocol has been
            triggered this session. Prevents infinite recovery loops.
        agent_registry: Optional dict mapping persistent agent roles to
            their agentIds (e.g. {"research": "abc123", "tuning": "def456"}).
            Used for resuming agents via SendMessage instead of fresh dispatch.
            Session-scoped — cleared on new session, preserved within session.
        hyperagent_state: Optional dict for Hyperagent mode state. Created
            via init_hyperagent_state(). Tracks archive generation,
            strategy history, meta-improvement count, operator stats,
            and active meta-patches. Preserved across pipeline resumptions.

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
    # Load existing state once if needed for preserving fields
    existing = (
        load_state(exp_root)
        if (user_choices is None or consecutive_stop_count is None
            or stuck_protocol_triggered is None or baseline_checksum is None
            or agent_registry is None or hyperagent_state is None)
        else None
    )

    # Preserve existing user_choices if not explicitly provided
    if user_choices is not None:
        state["user_choices"] = user_choices
        _check_goal_metric_consistency(exp_root, user_choices)
    elif existing and existing.get("user_choices"):
        state["user_choices"] = existing["user_choices"]

    # Preserve consecutive_stop_count for 3-consecutive-stop exit rule
    if consecutive_stop_count is not None:
        state["consecutive_stop_count"] = consecutive_stop_count
    elif existing and "consecutive_stop_count" in existing:
        state["consecutive_stop_count"] = existing["consecutive_stop_count"]

    # Preserve stuck_protocol_triggered (prevents infinite recovery loops)
    if stuck_protocol_triggered is not None:
        state["stuck_protocol_triggered"] = stuck_protocol_triggered
    elif existing and "stuck_protocol_triggered" in existing:
        state["stuck_protocol_triggered"] = existing["stuck_protocol_triggered"]

    # Preserve baseline_checksum (immutable baseline integrity)
    if baseline_checksum is not None:
        state["baseline_checksum"] = baseline_checksum
    elif existing and "baseline_checksum" in existing:
        state["baseline_checksum"] = existing["baseline_checksum"]

    # Preserve agent_registry (resumable subagent IDs, session-scoped)
    if agent_registry is not None:
        state["agent_registry"] = agent_registry
    elif existing and "agent_registry" in existing:
        state["agent_registry"] = existing["agent_registry"]

    # Preserve hyperagent_state (Hyperagent mode: evolutionary code search)
    if hyperagent_state is not None:
        state["hyperagent_state"] = hyperagent_state
    elif existing and "hyperagent_state" in existing:
        state["hyperagent_state"] = existing["hyperagent_state"]

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
    if user_choices is not None:
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
    """Validate that a phase transition is allowed and prerequisites are met.

    Returns {"valid": bool, "errors": list[str]}.
    """
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
    """Append a phase gate entry to experiments/phase-gates.json.

    Each entry: {"phase": int, "status": str, "summary": str, "timestamp": ISO8601}.
    Creates the file if it does not exist. Uses atomic write with file locking.
    """
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

            tmp_fd, tmp_path = tempfile.mkstemp(dir=str(root), suffix=".tmp")
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(entries, f, indent=2)
                os.replace(tmp_path, str(gate_path))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def get_recovery_instruction(phase, failure_type):
    """Return a recovery instruction for a given phase and failure type.

    Falls back to a default instruction if no specific match is found.
    """
    return RECOVERY_INSTRUCTIONS.get((phase, failure_type), _RECOVERY_DEFAULT)


# ---------------------------------------------------------------------------
# Decision Logging
# ---------------------------------------------------------------------------

_DECISION_REQUIRED_FIELDS = ("phase", "agent", "decision_type", "decision")


def _compute_inputs_hash(decision: dict) -> str:
    """Compute SHA-256 hash of the input state that led to a decision.

    Hashes the canonical JSON of the subset of fields that represent the
    input context: phase, iteration, agent, decision_type, context_summary.
    Only includes fields that are actually present in the decision dict.
    """
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
    """Atomically write decision-log.json (temp file + os.replace)."""
    root = Path(exp_root)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "decision-log.json"
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(root), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(entries, f, indent=2)
        os.replace(tmp_path, str(log_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Meta-Patch Management
# ---------------------------------------------------------------------------

META_PATCH_MAX_PER_SESSION = 3
META_PATCH_FORBIDDEN_SKILLS = {
    "orchestrate", "hyperagent-generate", "hyperagent-select",
    "hyperagent-eval", "hyperagent-archive", "hyperagent-init",
    "hyperagent-inspect",
}

_META_PATCH_REQUIRED_FIELDS = ("skill", "change", "reason", "expected_impact")


def validate_meta_patch(exp_root: str, patch: dict) -> dict:
    """Validate a meta-patch dict before logging.

    Checks:
    1. All required fields present and non-empty strings.
    2. ``skill`` not in ``META_PATCH_FORBIDDEN_SKILLS``.
    3. Current logged patch count < ``META_PATCH_MAX_PER_SESSION``.

    Returns ``{"valid": bool, "errors": list[str]}``.
    """
    errors: list[str] = []

    # 1. Required fields
    for field in _META_PATCH_REQUIRED_FIELDS:
        val = patch.get(field)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"Missing or empty required field: '{field}'")

    # 2. Forbidden skill check
    skill = patch.get("skill", "")
    if skill in META_PATCH_FORBIDDEN_SKILLS:
        errors.append(f"Skill '{skill}' is forbidden for meta-patches")

    # 3. Counter enforcement
    existing = get_meta_patches(exp_root)
    if len(existing) >= META_PATCH_MAX_PER_SESSION:
        errors.append(
            f"Max {META_PATCH_MAX_PER_SESSION} meta-patches per session "
            f"exceeded (current: {len(existing)})"
        )

    return {"valid": len(errors) == 0, "errors": errors}


def log_meta_patch(exp_root: str, patch: dict) -> dict:
    """Validate and log a meta-patch to the changelog.

    Returns ``{"logged": True, "id": N, "count": N}`` on success, or
    ``{"logged": False, "errors": [...]}`` on validation failure.
    """
    result = validate_meta_patch(exp_root, patch)
    if not result["valid"]:
        return {"logged": False, "errors": result["errors"]}

    meta_dir = Path(exp_root) / "meta-patches"
    meta_dir.mkdir(parents=True, exist_ok=True)
    changelog_path = meta_dir / "meta-changelog.json"
    lock_path = meta_dir / "meta-changelog.lock"

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            # Read existing entries
            if changelog_path.is_file():
                try:
                    entries = json.loads(changelog_path.read_text())
                except (json.JSONDecodeError, OSError):
                    entries = []
            else:
                entries = []

            # Defend against corrupt/legacy entries: changelog must be a list
            # of dicts. Drop anything else (old string entries, accidental
            # top-level dict/None). See tests/test_pipeline.py::
            # TestMetaPatchManagement::test_log_handles_corrupt_string_entries.
            if not isinstance(entries, list):
                entries = []
            entries = [e for e in entries if isinstance(e, dict)]

            # Compute next ID (auto-increment from 1)
            next_id = max((e.get("id", 0) for e in entries), default=0) + 1

            entry = {
                **patch,
                "id": next_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            entries.append(entry)

            # Atomic write
            tmp_fd, tmp_path = tempfile.mkstemp(dir=str(meta_dir), suffix=".tmp")
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(entries, f, indent=2)
                os.replace(tmp_path, str(changelog_path))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)

    return {"logged": True, "id": next_id, "count": len(entries)}


def log_decision(exp_root: str, decision: dict) -> str:
    """Append a decision entry to experiments/decision-log.json.

    The *decision* dict must contain at minimum: phase (int), agent (str),
    decision_type (str), decision (str). Optional fields: iteration (int),
    reasoning (str), context_summary (str).

    The function:
    1. Validates required fields — raises ValueError if missing.
    2. Adds timestamp (ISO8601 UTC) and id (auto-incremented).
    3. Computes inputs_hash: SHA-256 of canonical JSON of the input state.
    4. Appends to the log file using atomic write.
    5. Returns the decision id (str).
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
    """Load decision-log.json and filter by optional parameters.

    Returns matching entries (all entries if no filters provided).
    """
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
    """Compare a new decision against a logged one with the same inputs_hash.

    Finds the entry with the given *decision_id*, reads its stored inputs_hash,
    then searches for any prior entry with the same inputs_hash to compare
    the decision text.

    Returns {"match": bool, "original_decision": str, "current_decision": str,
             "divergence": str | None}.
    If no matching inputs_hash found, returns
    {"match": None, "divergence": "no_prior_decision_with_matching_inputs"}.
    """
    entries = _load_decision_log(exp_root)

    # Find the target entry by id
    target = None
    for entry in entries:
        if str(entry.get("id")) == str(decision_id):
            target = entry
            break

    if target is None:
        return {
            "match": None,
            "divergence": "no_prior_decision_with_matching_inputs",
        }

    target_hash = target.get("inputs_hash")
    if target_hash is None:
        return {
            "match": None,
            "divergence": "no_prior_decision_with_matching_inputs",
        }

    # Search for other entries with the same inputs_hash (excluding this one)
    original = None
    for entry in entries:
        if (entry.get("inputs_hash") == target_hash
                and str(entry.get("id")) != str(decision_id)):
            original = entry
            break

    if original is None:
        return {
            "match": None,
            "divergence": "no_prior_decision_with_matching_inputs",
        }

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


def get_meta_patches(exp_root: str) -> list[dict]:
    """Return all logged meta-patch changelog entries.

    Returns an empty list if the changelog file does not exist or is corrupt.
    """
    changelog_path = Path(exp_root) / "meta-patches" / "meta-changelog.json"
    if not changelog_path.is_file():
        return []
    try:
        entries = json.loads(changelog_path.read_text())
        if not isinstance(entries, list):
            return []
        # Drop corrupt/legacy non-dict entries so callers (promote_meta_patch)
        # can .get() safely.
        return [e for e in entries if isinstance(e, dict)]
    except (json.JSONDecodeError, OSError):
        return []


def promote_meta_patch(
    exp_root: str, plugin_root: str, patch_index: int
) -> dict:
    """Promote a meta-patch by copying its patched skill file into the plugin.

    1. Load changelog entry at *patch_index*.
    2. Look for patched file at ``experiments/meta-patches/<skill>-SKILL.md``.
    3. Ensure content starts with ``# [meta-improvement]``.
    4. Write to ``<plugin_root>/skills/<skill>/SKILL.md``.

    Returns ``{"promoted": True, "skill": str, "path": str}`` on success,
    or ``{"promoted": False, "error": str}`` on failure.
    """
    entries = get_meta_patches(exp_root)
    if not isinstance(patch_index, int) or patch_index < 0 or patch_index >= len(entries):
        return {
            "promoted": False,
            "error": f"Invalid patch index: {patch_index} (have {len(entries)} entries)",
        }

    entry = entries[patch_index]
    skill = entry.get("skill", "")

    # Locate the patched file
    patched_path = Path(exp_root) / "meta-patches" / f"{skill}-SKILL.md"
    if not patched_path.is_file():
        return {
            "promoted": False,
            "error": f"Patched file not found: {patched_path}",
        }

    content = patched_path.read_text()

    # Ensure the meta-improvement marker is present
    if not content.startswith("# [meta-improvement]"):
        content = "# [meta-improvement]\n" + content

    # Atomic write to plugin skill directory
    dest = Path(plugin_root) / "skills" / skill / "SKILL.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(content)
        os.replace(tmp_path, str(dest))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return {"promoted": True, "skill": skill, "path": str(dest)}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: pipeline_state.py <exp_root> "
            "[validate <phase>|save <phase> <iteration>|load|cleanup"
            "|gate <current_phase> <next_phase>|log-gate <phase> <status> <summary>"
            "|log-decision <json_data>|replay-check <decision_id> <current_decision>"
            "|decisions [--phase N] [--agent NAME] [--type TYPE]"
            "|meta-patch validate <json>|meta-patch log <json>"
            "|meta-patch list|meta-patch promote <index> <plugin_root>]"
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

    elif action == "verify-baseline":
        result = verify_baseline_integrity(exp_root)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["valid"] else 1)

    elif action == "gate":
        if len(sys.argv) < 5:
            print("Usage: pipeline_state.py <exp_root> gate <current_phase> <next_phase>")
            sys.exit(1)
        try:
            current_phase = int(sys.argv[3])
        except ValueError:
            print(f"Error: invalid current_phase '{sys.argv[3]}' (expected integer)")
            sys.exit(1)
        try:
            next_phase = int(sys.argv[4])
        except ValueError:
            print(f"Error: invalid next_phase '{sys.argv[4]}' (expected integer)")
            sys.exit(1)
        print(json.dumps(validate_phase_gate(current_phase, next_phase, exp_root)))

    elif action == "log-gate":
        if len(sys.argv) < 6:
            print("Usage: pipeline_state.py <exp_root> log-gate <phase> <status> <summary>")
            sys.exit(1)
        try:
            phase = int(sys.argv[3])
        except ValueError:
            print(f"Error: invalid phase '{sys.argv[3]}' (expected integer)")
            sys.exit(1)
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

    elif action == "meta-patch":
        if len(sys.argv) < 4:
            print("Usage: pipeline_state.py <exp_root> meta-patch "
                  "[validate <json>|log <json>|list|promote <index> <plugin_root>]")
            sys.exit(1)
        sub = sys.argv[3]
        if sub == "validate":
            if len(sys.argv) < 5:
                print("Usage: pipeline_state.py <exp_root> meta-patch validate <json_data>")
                sys.exit(1)
            try:
                patch = json.loads(sys.argv[4])
            except json.JSONDecodeError:
                print(f"Error: invalid JSON '{sys.argv[4]}'")
                sys.exit(1)
            print(json.dumps(validate_meta_patch(exp_root, patch), indent=2))
        elif sub == "log":
            if len(sys.argv) < 5:
                print("Usage: pipeline_state.py <exp_root> meta-patch log <json_data>")
                sys.exit(1)
            try:
                patch = json.loads(sys.argv[4])
            except json.JSONDecodeError:
                print(f"Error: invalid JSON '{sys.argv[4]}'")
                sys.exit(1)
            print(json.dumps(log_meta_patch(exp_root, patch), indent=2))
        elif sub == "list":
            print(json.dumps(get_meta_patches(exp_root), indent=2))
        elif sub == "promote":
            if len(sys.argv) < 6:
                print("Usage: pipeline_state.py <exp_root> meta-patch promote <index> <plugin_root>")
                sys.exit(1)
            try:
                idx = int(sys.argv[4])
            except ValueError:
                print(f"Error: invalid index '{sys.argv[4]}' (expected integer)")
                sys.exit(1)
            print(json.dumps(promote_meta_patch(exp_root, sys.argv[5], idx), indent=2))
        else:
            print(f"Unknown meta-patch subcommand: {sub}")
            sys.exit(1)

    else:
        print(f"Unknown action: {action}")
        sys.exit(1)
