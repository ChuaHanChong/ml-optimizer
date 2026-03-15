#!/usr/bin/env python3
"""Error tracking and self-improvement for the ML optimizer plugin.

Captures, persists, and analyzes errors from skills and agents.
Supports per-project error logs and cross-project pattern memory.

Dependency-free — uses only the Python standard library.
"""

import fcntl
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

ERROR_EVENT_REQUIRED = ["event_id", "timestamp", "category", "severity", "source", "message"]
ERROR_EVENT_OPTIONAL = [
    "exp_id", "skill", "agent", "phase", "iteration",
    "code_branch", "config", "metrics", "stack_trace",
    "context", "resolution", "project_id", "duration_seconds",
]

VALID_CATEGORIES = [
    "agent_failure",
    "training_failure",
    "divergence",
    "implementation_error",
    "pipeline_inefficiency",
    "config_error",
    "research_failure",
    "timeout",
    "resource_error",
]

VALID_SEVERITIES = ["critical", "warning", "info"]

VALID_SOURCES = [
    "orchestrate", "baseline", "research", "implement",
    "hp-tune", "experiment", "monitor", "analyze", "report", "review",
    "prerequisites", "hook:detect-critical-errors", "hook:bash-safety",
    "hook:file-guardrail", "hook:subagent-stop", "hook:pre-compact",
]

# Monotonic counter for unique event IDs within a process
_event_counter = 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_event(event: dict) -> dict:
    """Validate an error event dict.

    Returns {"valid": bool, "errors": list[str]}.
    """
    errors: list[str] = []

    if not isinstance(event, dict):
        return {"valid": False, "errors": ["Event must be a dict"]}

    for field in ERROR_EVENT_REQUIRED:
        if field not in event:
            errors.append(f"Missing required field: {field}")

    if "category" in event and event["category"] not in VALID_CATEGORIES:
        errors.append(f"Invalid category: {event['category']}")

    if "severity" in event and event["severity"] not in VALID_SEVERITIES:
        errors.append(f"Invalid severity: {event['severity']}")

    if "source" in event and event["source"] not in VALID_SOURCES:
        errors.append(f"Invalid source: {event['source']}")

    return {"valid": len(errors) == 0, "errors": errors}


# ---------------------------------------------------------------------------
# Event creation
# ---------------------------------------------------------------------------

def create_event(
    category: str, severity: str, source: str, message: str,
    *, exp_id=None, skill=None, agent=None, phase=None, iteration=None,
    code_branch=None, config=None, metrics=None, stack_trace=None,
    context=None, resolution=None, project_id=None, duration_seconds=None,
) -> dict:
    """Create a new error event with auto-generated event_id and timestamp."""
    global _event_counter
    _event_counter += 1

    event = {
        "event_id": f"err-{_event_counter}-{int(time.time())}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "severity": severity,
        "source": source,
        "message": message,
    }

    optional = {
        "exp_id": exp_id, "skill": skill, "agent": agent,
        "phase": phase, "iteration": iteration, "code_branch": code_branch,
        "config": config, "metrics": metrics, "stack_trace": stack_trace,
        "context": context, "resolution": resolution, "project_id": project_id,
        "duration_seconds": duration_seconds,
    }
    for k, v in optional.items():
        if v is not None:
            event[k] = v

    result = validate_event(event)
    if not result["valid"]:
        raise ValueError(f"Invalid event: {'; '.join(result['errors'])}")

    return event


# ---------------------------------------------------------------------------
# Per-project storage
# ---------------------------------------------------------------------------

def _error_log_path(exp_root: str) -> Path:
    """Return the path to the per-project error log file."""
    return Path(exp_root) / "reports" / "error-log.json"


def _compute_summary(events: list[dict]) -> dict:
    """Compute summary counts from an events list."""
    by_cat: dict[str, int] = {}
    by_sev: dict[str, int] = {}
    for ev in events:
        cat = ev.get("category", "unknown")
        sev = ev.get("severity", "unknown")
        by_cat[cat] = by_cat.get(cat, 0) + 1
        by_sev[sev] = by_sev.get(sev, 0) + 1
    return {
        "total_events": len(events),
        "by_category": by_cat,
        "by_severity": by_sev,
    }


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def log_event(exp_root: str, event: dict) -> str:
    """Append an event to the per-project error log. Returns the log path."""
    path = _error_log_path(exp_root)
    lock_path = path.with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            log_data = load_error_log(exp_root)

            if log_data is None:
                log_data = {
                    "project_id": _project_id(exp_root),
                    "session_start": datetime.now(timezone.utc).isoformat(),
                    "events": [],
                    "summary": {},
                }

            log_data["events"].append(event)
            log_data["summary"] = _compute_summary(log_data["events"])
            _atomic_write_json(path, log_data)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    return str(path)


def load_error_log(exp_root: str) -> dict | None:
    """Load the per-project error log. Returns None if not found or corrupt."""
    path = _error_log_path(exp_root)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def get_events(
    exp_root: str, category: str | None = None, severity: str | None = None,
) -> list[dict]:
    """Get events from the per-project log, optionally filtered."""
    log_data = load_error_log(exp_root)
    if log_data is None:
        return []
    events = log_data.get("events", [])
    if category is not None:
        events = [e for e in events if e.get("category") == category]
    if severity is not None:
        events = [e for e in events if e.get("severity") == severity]
    return events


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

def _project_id(path: str) -> str:
    """Compute a short deterministic project ID from a path."""
    return hashlib.md5(path.encode()).hexdigest()[:12]


def detect_patterns(events: list[dict]) -> list[dict]:
    """Analyze events for recurring patterns. Returns pattern dicts."""
    if not events:
        return []

    patterns: list[dict] = []

    # --- high_lr_divergence: 3+ divergence events ---
    divergence_events = [e for e in events if e.get("category") == "divergence"]
    if len(divergence_events) >= 3:
        lrs = []
        for e in divergence_events:
            cfg = e.get("config", {})
            if isinstance(cfg, dict) and "lr" in cfg:
                try:
                    lr_val = float(cfg["lr"])
                    if math.isfinite(lr_val):
                        lrs.append(lr_val)
                except (ValueError, TypeError):
                    pass
        if lrs:
            avg_lr = sum(lrs) / len(lrs)
            patterns.append({
                "pattern_id": "high_lr_divergence",
                "description": f"Divergence cluster: {len(divergence_events)} events, avg LR={avg_lr:.4f}",
                "occurrences": len(divergence_events),
                "suggested_action": f"Start LR search below {min(lrs):.4f}",
            })

    # --- oom_batch_size: 2+ OOM events with same batch_size ---
    oom_events = [
        e for e in events
        if e.get("category") == "training_failure"
        and ("oom" in e.get("message", "").lower() or
             (isinstance(e.get("context"), dict) and e["context"].get("error_type") == "oom"))
    ]
    if len(oom_events) >= 2:
        batch_sizes: Counter = Counter()
        for e in oom_events:
            cfg = e.get("config", {})
            if isinstance(cfg, dict) and "batch_size" in cfg:
                batch_sizes[cfg["batch_size"]] += 1
        for bs, count in batch_sizes.items():
            if count >= 2:
                patterns.append({
                    "pattern_id": "oom_batch_size",
                    "description": f"OOM with batch_size={bs} occurred {count} times",
                    "occurrences": count,
                    "suggested_action": f"Reduce batch_size below {bs}",
                })

    # --- wasted_budget: pipeline_inefficiency events about wasted experiments ---
    waste_events = [
        e for e in events
        if e.get("category") == "pipeline_inefficiency"
        and isinstance(e.get("context"), dict)
        and e["context"].get("experiments_wasted")
    ]
    if len(waste_events) >= 2:
        total_wasted = sum(e["context"]["experiments_wasted"] for e in waste_events)
        patterns.append({
            "pattern_id": "wasted_budget",
            "description": f"{total_wasted} experiments wasted across {len(waste_events)} batches",
            "occurrences": len(waste_events),
            "suggested_action": "Tighten HP search space or add pre-flight validation",
        })

    # --- redundant_configs: pipeline_inefficiency about duplication ---
    dup_events = [
        e for e in events
        if e.get("category") == "pipeline_inefficiency"
        and "duplicat" in e.get("message", "").lower()
    ]
    if len(dup_events) >= 2:
        patterns.append({
            "pattern_id": "redundant_configs",
            "description": f"HP proposal duplication occurred {len(dup_events)} times",
            "occurrences": len(dup_events),
            "suggested_action": "Widen HP search space to avoid exhaustion",
        })

    # --- branch_underperformance: pipeline_inefficiency about underperforming branches ---
    branch_events = [
        e for e in events
        if e.get("category") == "pipeline_inefficiency"
        and "underperform" in e.get("message", "").lower()
    ]
    if len(branch_events) >= 1:
        branches = [e.get("code_branch", "unknown") for e in branch_events]
        patterns.append({
            "pattern_id": "branch_underperformance",
            "description": f"Underperforming branches: {', '.join(branches)}",
            "occurrences": len(branch_events),
            "suggested_action": "Prune these branches earlier to save budget",
        })

    # --- early_failure_cluster: 3+ failures in same non-Phase-5 phase ---
    failure_cats = {"training_failure", "divergence", "config_error",
                    "implementation_error", "research_failure", "resource_error",
                    "agent_failure", "timeout"}
    phase_failures: dict[int, int] = {}
    for e in events:
        if e.get("category") in failure_cats and e.get("phase") is not None:
            ph = e["phase"]
            if ph != 5:
                phase_failures[ph] = phase_failures.get(ph, 0) + 1
    for ph, count in phase_failures.items():
        if count >= 3:
            patterns.append({
                "pattern_id": "early_failure_cluster",
                "description": f"{count} failures concentrated in Phase {ph}",
                "occurrences": count,
                "suggested_action": f"Investigate Phase {ph} setup — failures are concentrated there",
            })
            break  # report at most one

    # --- hp_interaction_failure: 3+ failures with same LR-bucket × batch_size ---
    fail_events = [
        e for e in events
        if e.get("category") in ("divergence", "training_failure")
    ]
    interaction_keys: Counter = Counter()
    for e in fail_events:
        cfg = e.get("config", {})
        if not isinstance(cfg, dict):
            continue
        lr_val = cfg.get("lr")
        bs_val = cfg.get("batch_size")
        if lr_val is None or bs_val is None:
            continue
        try:
            lr_f = float(lr_val)
        except (ValueError, TypeError):
            continue
        if not math.isfinite(lr_f):
            continue
        if lr_f < 0.001:
            bucket = "low"
        elif lr_f <= 0.01:
            bucket = "medium"
        else:
            bucket = "high"
        interaction_keys[(bucket, bs_val)] += 1
    for (bucket, bs), count in interaction_keys.items():
        if count >= 3:
            patterns.append({
                "pattern_id": "hp_interaction_failure",
                "description": f"{bucket} LR + batch_size={bs} failed {count} times",
                "occurrences": count,
                "suggested_action": f"Avoid {bucket} LR with batch_size={bs}",
            })
            break  # report worst combo only

    # --- temporal_failure_cluster: 60%+ of failures in iteration ≤1 ---
    iter_events = [
        e for e in events
        if e.get("category") in ("training_failure", "divergence")
        and e.get("iteration") is not None
    ]
    if len(iter_events) >= 4:
        early = sum(1 for e in iter_events if e["iteration"] <= 1)
        ratio = early / len(iter_events)
        if ratio >= 0.6:
            patterns.append({
                "pattern_id": "temporal_failure_cluster",
                "description": f"{early}/{len(iter_events)} failures in iteration 1 ({ratio:.0%})",
                "occurrences": early,
                "suggested_action": "Most failures occur early — initial HP search space may be too aggressive",
            })

    # --- timeout_pattern: 2+ timeout events ---
    timeout_events = [e for e in events if e.get("category") == "timeout"]
    if len(timeout_events) >= 2:
        patterns.append({
            "pattern_id": "timeout_pattern",
            "description": f"{len(timeout_events)} timeout events recorded",
            "occurrences": len(timeout_events),
            "suggested_action": "Consider splitting long-running tasks or increasing time limits",
        })

    return patterns


# ---------------------------------------------------------------------------
# Cross-project storage
# ---------------------------------------------------------------------------

def _cross_project_path(plugin_root: str) -> Path:
    """Return the path to the cross-project memory file."""
    return Path(plugin_root) / "memory" / "cross-project-errors.json"


def update_cross_project(
    plugin_root: str, project_path: str, exp_root: str,
) -> str:
    """Sync per-project error log into cross-project memory. Returns memory path."""
    mem_path = _cross_project_path(plugin_root)

    # Load or create cross-project memory
    memory = load_cross_project(plugin_root)
    if memory is None:
        memory = {
            "version": 1,
            "last_updated": None,
            "projects": {},
            "cross_project_patterns": [],
        }

    # Load per-project data
    log_data = load_error_log(exp_root)
    if log_data is None:
        log_data = {"events": [], "session_start": datetime.now(timezone.utc).isoformat()}

    proj_id = _project_id(project_path)
    events = log_data.get("events", [])
    detected = detect_patterns(events)
    pattern_ids = [p["pattern_id"] for p in detected]

    session_entry = {
        "session_start": log_data.get("session_start", datetime.now(timezone.utc).isoformat()),
        "event_count": len(events),
        "categories": _compute_summary(events)["by_category"],
        "patterns_detected": pattern_ids,
    }

    if proj_id not in memory["projects"]:
        memory["projects"][proj_id] = {
            "project_path": project_path,
            "sessions": [],
        }
    sessions = memory["projects"][proj_id]["sessions"]
    if sessions and sessions[-1].get("session_start") == session_entry["session_start"]:
        sessions[-1] = session_entry  # update existing session
    else:
        sessions.append(session_entry)

    # Recompute cross-project patterns
    memory["cross_project_patterns"] = detect_cross_project_patterns(memory)
    memory["last_updated"] = datetime.now(timezone.utc).isoformat()

    # Ensure memory dir exists
    mem_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(mem_path, memory)
    return str(mem_path)


def load_cross_project(plugin_root: str) -> dict | None:
    """Load cross-project error memory. Returns None if not found or corrupt."""
    path = _cross_project_path(plugin_root)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def detect_cross_project_patterns(memory: dict) -> list[dict]:
    """Detect patterns spanning multiple projects."""
    projects = memory.get("projects", {})
    if not projects:
        return []

    # Count how many projects have each pattern
    pattern_projects: dict[str, set] = {}
    for proj_id, proj_data in projects.items():
        for session in proj_data.get("sessions", []):
            for pat_id in session.get("patterns_detected", []):
                if pat_id not in pattern_projects:
                    pattern_projects[pat_id] = set()
                pattern_projects[pat_id].add(proj_id)

    patterns = []
    for pat_id, proj_ids in pattern_projects.items():
        if len(proj_ids) >= 2:
            patterns.append({
                "pattern_id": pat_id,
                "projects_affected": len(proj_ids),
                "description": f"Pattern '{pat_id}' observed across {len(proj_ids)} projects",
                "suggested_action": f"Persistent issue — consider updating plugin defaults",
            })

    return patterns


# ---------------------------------------------------------------------------
# Suggestion ranking
# ---------------------------------------------------------------------------

# Severity weights: blocking = 3, quality-degrading = 2, informational = 1
_PATTERN_WEIGHTS: dict[str, int] = {
    "oom_batch_size": 3,
    "timeout_pattern": 3,
    "high_lr_divergence": 2,
    "hp_interaction_failure": 2,
    "early_failure_cluster": 2,
    "wasted_budget": 1,
    "redundant_configs": 1,
    "branch_underperformance": 1,
    "temporal_failure_cluster": 1,
}


def rank_suggestions(
    patterns: list[dict],
    cross_project_patterns: list[dict] | None = None,
    total_experiments: int | None = None,
) -> list[dict]:
    """Rank detected patterns by impact score.

    Score = severity_weight × occurrences × cross_project_boost.
    Returns patterns sorted by score descending, with ``score`` added.
    When *total_experiments* is provided, each entry also gets a
    ``significance`` field (occurrences / total_experiments).
    """
    if not patterns:
        return []

    cross_ids = set()
    if cross_project_patterns:
        cross_ids = {p["pattern_id"] for p in cross_project_patterns}

    ranked = []
    for p in patterns:
        pid = p["pattern_id"]
        weight = _PATTERN_WEIGHTS.get(pid, 1)
        boost = 1.5 if pid in cross_ids else 1.0
        occ = p.get("occurrences", 1)
        score = weight * occ * boost
        entry = {**p, "score": score}
        if total_experiments is not None and total_experiments > 0:
            entry["significance"] = round(occ / total_experiments, 3)
        ranked.append(entry)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------

def summarize_session(exp_root: str) -> dict:
    """Generate a session summary from the error log."""
    log_data = load_error_log(exp_root)
    if log_data is None:
        return {
            "total_events": 0,
            "by_category": {},
            "by_severity": {},
            "patterns_detected": [],
        }

    events = log_data.get("events", [])
    summary = _compute_summary(events)
    detected = detect_patterns(events)
    summary["patterns_detected"] = [p["pattern_id"] for p in detected]
    return summary


# ---------------------------------------------------------------------------
# Success metrics and proposal outcomes
# ---------------------------------------------------------------------------

def _load_results(exp_root: str) -> tuple[dict | None, list[dict]]:
    """Load baseline and experiment results from results/ directory.

    Returns (baseline_data_or_none, list_of_experiment_dicts).
    """
    results_dir = Path(exp_root) / "results"
    if not results_dir.is_dir():
        return None, []

    baseline = None
    experiments = []

    for p in results_dir.glob("*.json"):
        if p.name in ("implementation-manifest.json",):
            continue
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict) or "exp_id" not in data:
            continue
        if p.name == "baseline.json" or data.get("exp_id") == "baseline":
            baseline = data
        elif p.name.startswith("exp-"):
            experiments.append(data)

    return baseline, experiments


def _is_better(value: float, baseline_value: float, lower_is_better: bool) -> bool:
    """Check if value is better than baseline."""
    if lower_is_better:
        return value < baseline_value
    return value > baseline_value


def compute_success_metrics(
    exp_root: str, primary_metric: str, lower_is_better: bool,
) -> dict:
    """Analyze experiment results for success signals (not just errors)."""
    baseline, experiments = _load_results(exp_root)

    if not experiments:
        return {
            "total_experiments": 0,
            "completed": 0,
            "failed": 0,
            "diverged": 0,
            "success_rate": 0.0,
            "improvement_rate": None,
            "best_improvement_pct": None,
            "avg_duration_completed": None,
            "avg_duration_failed": None,
            "time_wasted_on_failures_pct": None,
            "top_configs": [],
            "worst_configs": [],
        }

    completed = [e for e in experiments if e.get("status") == "completed"]
    failed = [e for e in experiments if e.get("status") == "failed"]
    diverged = [e for e in experiments if e.get("status") == "diverged"]
    non_completed = failed + diverged

    total = len(experiments)
    success_rate = len(completed) / total if total > 0 else 0.0

    # Improvement rate: how many completed experiments beat baseline?
    improvement_rate = None
    best_improvement_pct = None
    top_configs = []
    worst_configs = []

    if baseline is not None:
        baseline_val = baseline.get("metrics", {}).get(primary_metric)
        if baseline_val is not None and isinstance(baseline_val, (int, float)) and math.isfinite(baseline_val):
            beat_count = 0
            improvements = []
            for e in completed:
                val = e.get("metrics", {}).get(primary_metric)
                if val is not None and isinstance(val, (int, float)) and math.isfinite(val):
                    if _is_better(val, baseline_val, lower_is_better):
                        beat_count += 1
                        if baseline_val != 0:
                            delta = baseline_val - val if lower_is_better else val - baseline_val
                            pct = round(delta / abs(baseline_val) * 100, 2)
                        else:
                            pct = None
                        improvements.append({
                            "exp_id": e["exp_id"],
                            "config": e.get("config", {}),
                            "metric_value": val,
                            "improvement_pct": pct,
                        })

            improvement_rate = beat_count / len(completed) if completed else None

            # Sort improvements: best first
            if lower_is_better:
                improvements.sort(key=lambda x: x["metric_value"])
            else:
                improvements.sort(key=lambda x: x["metric_value"], reverse=True)
            top_configs = improvements

            if improvements:
                best_improvement_pct = improvements[0].get("improvement_pct")

    # Worst configs: failed/diverged experiments
    worst_configs = [
        {"exp_id": e["exp_id"], "config": e.get("config", {}), "status": e.get("status")}
        for e in non_completed
    ]

    # Duration analysis
    completed_durations = [
        e["duration_seconds"] for e in completed
        if "duration_seconds" in e and isinstance(e["duration_seconds"], (int, float))
    ]
    failed_durations = [
        e["duration_seconds"] for e in non_completed
        if "duration_seconds" in e and isinstance(e["duration_seconds"], (int, float))
    ]

    avg_dur_completed = (
        sum(completed_durations) / len(completed_durations)
        if completed_durations else None
    )
    avg_dur_failed = (
        sum(failed_durations) / len(failed_durations)
        if failed_durations else None
    )

    total_time = sum(completed_durations) + sum(failed_durations)
    time_wasted = sum(failed_durations)
    time_wasted_pct = (
        (time_wasted / total_time * 100) if total_time > 0 else None
    )

    return {
        "total_experiments": total,
        "completed": len(completed),
        "failed": len(failed),
        "diverged": len(diverged),
        "success_rate": success_rate,
        "improvement_rate": improvement_rate,
        "best_improvement_pct": best_improvement_pct,
        "avg_duration_completed": avg_dur_completed,
        "avg_duration_failed": avg_dur_failed,
        "time_wasted_on_failures_pct": time_wasted_pct,
        "top_configs": top_configs,
        "worst_configs": worst_configs,
    }


def compute_proposal_outcomes(
    exp_root: str, primary_metric: str, lower_is_better: bool,
) -> dict:
    """Cross-reference proposals with experiment outcomes."""
    results_dir = Path(exp_root) / "results"
    baseline, experiments = _load_results(exp_root)

    # Implementation manifest
    impl_stats = {
        "total_proposals": 0,
        "validated": 0,
        "validation_failed": 0,
        "implementation_error": 0,
    }
    proposal_branches: dict[str, str] = {}  # branch -> proposal name

    manifest_path = results_dir / "implementation-manifest.json" if results_dir.is_dir() else None
    if manifest_path and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            proposals = manifest.get("proposals", [])
            impl_stats["total_proposals"] = len(proposals)
            for p in proposals:
                status = p.get("status", "")
                if status == "validated":
                    impl_stats["validated"] += 1
                    branch = p.get("branch", "")
                    if branch:
                        proposal_branches[branch] = p.get("name", "unknown")
                elif status == "validation_failed":
                    impl_stats["validation_failed"] += 1
                elif status == "implementation_error":
                    impl_stats["implementation_error"] += 1
        except (json.JSONDecodeError, OSError):
            pass

    # Research proposal outcomes: group experiments by code_branch
    baseline_val = None
    if baseline:
        baseline_val = baseline.get("metrics", {}).get(primary_metric)

    research_proposals = []
    branch_exps: dict[str, list[dict]] = {}
    for e in experiments:
        branch = e.get("code_branch")
        if branch and branch in proposal_branches:
            if branch not in branch_exps:
                branch_exps[branch] = []
            branch_exps[branch].append(e)

    for branch, exps in branch_exps.items():
        name = proposal_branches[branch]
        beat = 0
        best_imp = None
        for e in exps:
            val = e.get("metrics", {}).get(primary_metric)
            if (
                val is not None
                and baseline_val is not None
                and isinstance(val, (int, float))
                and isinstance(baseline_val, (int, float))
                and _is_better(val, baseline_val, lower_is_better)
            ):
                beat += 1
                if baseline_val != 0:
                    delta = baseline_val - val if lower_is_better else val - baseline_val
                    pct = round(delta / abs(baseline_val) * 100, 2)
                    if best_imp is None or pct > (best_imp or 0):
                        best_imp = pct

        research_proposals.append({
            "name": name,
            "branch": branch,
            "experiments": len(exps),
            "beat_baseline": beat,
            "best_improvement": f"{best_imp:.1f}%" if best_imp is not None else None,
        })

    # HP proposal stats
    configs_dir = results_dir / "proposed-configs" if results_dir.is_dir() else None
    total_proposed = 0
    if configs_dir and configs_dir.is_dir():
        total_proposed = len(list(configs_dir.glob("*.json")))

    hp_proposals = {
        "total_proposed": total_proposed,
        "total_run": len(experiments),
        "total_completed": len([e for e in experiments if e.get("status") == "completed"]),
        "total_beat_baseline": sum(
            1 for e in experiments
            if e.get("status") == "completed"
            and baseline_val is not None
            and isinstance(baseline_val, (int, float))
            and isinstance(e.get("metrics", {}).get(primary_metric), (int, float))
            and _is_better(e["metrics"][primary_metric], baseline_val, lower_is_better)
        ),
    }

    return {
        "research_proposals": research_proposals,
        "hp_proposals": hp_proposals,
        "implementation_stats": impl_stats,
    }


# ---------------------------------------------------------------------------
# Cross-project memory cleanup
# ---------------------------------------------------------------------------


def cleanup_memory(
    plugin_root: str, max_sessions_per_project: int = 10,
) -> dict:
    """Remove old sessions from cross-project memory.

    Keeps the last *max_sessions_per_project* sessions per project,
    removes projects with zero sessions, and recomputes cross-project
    patterns.  Returns ``{"cleaned": N, "projects_remaining": M}``.
    """
    memory = load_cross_project(plugin_root)
    if memory is None:
        return {"cleaned": 0, "projects_remaining": 0}

    cleaned = 0
    for proj_data in memory["projects"].values():
        sessions = proj_data.get("sessions", [])
        if len(sessions) > max_sessions_per_project:
            excess = len(sessions) - max_sessions_per_project
            proj_data["sessions"] = sessions[-max_sessions_per_project:]
            cleaned += excess

    # Remove empty projects
    empty = [pid for pid, pd in memory["projects"].items() if not pd.get("sessions")]
    for pid in empty:
        del memory["projects"][pid]

    if cleaned or empty:
        memory["cross_project_patterns"] = detect_cross_project_patterns(memory)
        mem_path = _cross_project_path(plugin_root)
        _atomic_write_json(mem_path, memory)

    return {"cleaned": cleaned, "projects_remaining": len(memory["projects"])}


# ---------------------------------------------------------------------------
# Dead-end catalog
# ---------------------------------------------------------------------------


def _dead_ends_path(exp_root: str) -> Path:
    """Return the path to the dead-ends JSON file."""
    return Path(exp_root) / "reports" / "dead-ends.json"


def _dead_ends_md_path(exp_root: str) -> Path:
    """Return the path to the dead-ends Markdown companion."""
    return Path(exp_root) / "reports" / "dead-ends.md"


def _normalize_technique(name: str) -> str:
    """Normalize a technique name for fuzzy matching."""
    return name.lower().strip().replace("-", " ").replace("_", " ")


def log_dead_end(exp_root: str, entry: dict) -> str:
    """Append a dead-end entry. Returns the JSON file path.

    *entry* should contain at minimum ``technique`` and ``reason``.
    Optional fields: ``branch``, ``experiments_tried``, ``best_result``,
    ``source``.  ``timestamp`` is auto-added.
    """
    path = _dead_ends_path(exp_root)
    lock_path = path.with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)

    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    entry.setdefault("source", "unknown")

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            dead_ends = get_dead_ends(exp_root)
            dead_ends.append(entry)
            _atomic_write_json(path, {"dead_ends": dead_ends})
            _generate_dead_ends_md(exp_root, dead_ends)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    return str(path)


def get_dead_ends(exp_root: str) -> list[dict]:
    """Return all dead-end entries."""
    path = _dead_ends_path(exp_root)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
        return data.get("dead_ends", [])
    except (json.JSONDecodeError, OSError):
        return []


def is_dead_end(exp_root: str, technique_name: str) -> bool:
    """Check if a technique is in the dead-end catalog (fuzzy match).

    Matches when the normalized query is a substring of a cataloged
    technique name, or vice versa.
    """
    query = _normalize_technique(technique_name)
    if not query:
        return False
    for entry in get_dead_ends(exp_root):
        cataloged = _normalize_technique(entry.get("technique", ""))
        if not cataloged:
            continue
        if query in cataloged or cataloged in query:
            return True
    return False


def _generate_dead_ends_md(exp_root: str, dead_ends: list[dict]) -> None:
    """Write the human-readable dead-ends.md companion file."""
    lines = ["# Dead Ends\n", "Techniques that were tried and conclusively failed.\n"]
    for i, de in enumerate(dead_ends, 1):
        tech = de.get("technique", "unknown")
        reason = de.get("reason", "no reason given")
        branch = de.get("branch", "")
        tried = de.get("experiments_tried", "?")
        ts = de.get("timestamp", "")
        source = de.get("source", "")
        lines.append(f"## {i}. {tech}\n")
        lines.append(f"- **Reason**: {reason}")
        if branch:
            lines.append(f"- **Branch**: `{branch}`")
        lines.append(f"- **Experiments tried**: {tried}")
        if source:
            lines.append(f"- **Source**: {source}")
        best = de.get("best_result")
        if isinstance(best, dict):
            metric = best.get("metric", "?")
            val = best.get("value", "?")
            base = best.get("baseline", "?")
            lines.append(f"- **Best result**: {metric}={val} (baseline={base})")
        if ts:
            lines.append(f"- **Logged**: {ts}")
        lines.append("")
    md_path = _dead_ends_md_path(exp_root)
    md_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Research agenda
# ---------------------------------------------------------------------------


def _agenda_path(exp_root: str) -> Path:
    """Return the path to the research agenda JSON file."""
    return Path(exp_root) / "reports" / "research-agenda.json"


def _agenda_md_path(exp_root: str) -> Path:
    """Return the path to the research agenda Markdown companion."""
    return Path(exp_root) / "reports" / "research-agenda.md"


def init_agenda(exp_root: str, ideas: list[dict]) -> str:
    """Initialize the research agenda from a list of ideas. Returns JSON path.

    Each idea should have at minimum ``id`` and ``name``.
    Optional fields: ``priority``, ``source``, ``scope``, ``status``.
    """
    path = _agenda_path(exp_root)
    lock_path = path.with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)

    for idea in ideas:
        idea.setdefault("status", "untried")
        idea.setdefault("priority", 5.0)
        idea.setdefault("initial_priority", idea["priority"])
        idea.setdefault("source", "unknown")
        idea.setdefault("scope", "training")
        idea.setdefault("evidence", [])
        idea.setdefault("lessons", "")
        idea.setdefault("related_dead_end", None)

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            _atomic_write_json(path, {"ideas": ideas})
            _generate_agenda_md(exp_root, ideas)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    return str(path)


def get_agenda(exp_root: str) -> list[dict]:
    """Return all agenda ideas."""
    path = _agenda_path(exp_root)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
        return data.get("ideas", [])
    except (json.JSONDecodeError, OSError):
        return []


def update_agenda_item(exp_root: str, idea_id: str, updates: dict) -> bool:
    """Update a single agenda item by id. Returns True if found and updated.

    *updates* can include any fields: ``status``, ``priority``,
    ``evidence`` (appended, not replaced), ``lessons``,
    ``related_dead_end``.
    """
    path = _agenda_path(exp_root)
    lock_path = path.with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            ideas = get_agenda(exp_root)
            found = False
            for idea in ideas:
                if idea.get("id") == idea_id:
                    # Append evidence rather than replace (don't mutate caller's dict)
                    if "evidence" in updates:
                        existing = idea.get("evidence", [])
                        new_ev = updates["evidence"]
                        if isinstance(new_ev, list):
                            existing.extend(new_ev)
                        else:
                            existing.append(new_ev)
                        idea["evidence"] = existing
                    # Apply all other fields (skip evidence since it was handled above)
                    for k, v in updates.items():
                        if k != "evidence":
                            idea[k] = v
                    found = True
                    break
            if found:
                _atomic_write_json(path, {"ideas": ideas})
                _generate_agenda_md(exp_root, ideas)
            return found
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def add_agenda_idea(exp_root: str, idea: dict) -> str:
    """Add a new idea to the agenda mid-session. Returns JSON path."""
    path = _agenda_path(exp_root)
    lock_path = path.with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)

    idea.setdefault("status", "untried")
    idea.setdefault("priority", 5.0)
    idea.setdefault("initial_priority", idea["priority"])
    idea.setdefault("source", "unknown")
    idea.setdefault("scope", "training")
    idea.setdefault("evidence", [])
    idea.setdefault("lessons", "")
    idea.setdefault("related_dead_end", None)

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            ideas = get_agenda(exp_root)
            ideas.append(idea)
            _atomic_write_json(path, {"ideas": ideas})
            _generate_agenda_md(exp_root, ideas)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    return str(path)


def _generate_agenda_md(exp_root: str, ideas: list[dict]) -> None:
    """Write the human-readable research-agenda.md companion file."""
    active = [i for i in ideas if i.get("status") == "untried"]
    tried = [i for i in ideas if i.get("status") == "tried"]
    improved = [i for i in ideas if i.get("status") == "improved"]
    dead = [i for i in ideas if i.get("status") == "dead-end"]

    active.sort(key=lambda x: x.get("priority", 0), reverse=True)

    lines = ["# Research Agenda\n"]

    if improved:
        lines.append("## Successful Techniques\n")
        for idea in improved:
            lines.append(f"- **{idea.get('name', '?')}** (priority: {idea.get('priority', '?')}, source: {idea.get('source', '?')})")
            for ev in idea.get("evidence", []):
                if isinstance(ev, dict):
                    lines.append(f"  - Batch {ev.get('batch', '?')}: {ev.get('result', '?')}")
            if idea.get("lessons"):
                lines.append(f"  - Lesson: {idea['lessons']}")
        lines.append("")

    if active:
        lines.append("## Active Ideas (sorted by priority)\n")
        for idea in active:
            lines.append(f"- **{idea.get('name', '?')}** (priority: {idea.get('priority', '?')}, source: {idea.get('source', '?')}, scope: {idea.get('scope', '?')})")
        lines.append("")

    if tried:
        lines.append("## Tried (mixed or neutral results)\n")
        for idea in tried:
            lines.append(f"- **{idea.get('name', '?')}** (priority: {idea.get('priority', '?')})")
            for ev in idea.get("evidence", []):
                if isinstance(ev, dict):
                    lines.append(f"  - Batch {ev.get('batch', '?')}: {ev.get('result', '?')}")
            if idea.get("lessons"):
                lines.append(f"  - Lesson: {idea['lessons']}")
        lines.append("")

    if dead:
        lines.append("## Dead Ends\n")
        for idea in dead:
            lines.append(f"- ~~{idea.get('name', '?')}~~ — {idea.get('lessons', 'no details')}")
        lines.append("")

    # Lessons learned section
    all_lessons = [i.get("lessons", "") for i in ideas if i.get("lessons")]
    if all_lessons:
        lines.append("## Lessons Learned\n")
        for lesson in all_lessons:
            lines.append(f"- {lesson}")
        lines.append("")

    md_path = _agenda_md_path(exp_root)
    md_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Suggestion history / feedback loop
# ---------------------------------------------------------------------------


def _suggestion_history_path(exp_root: str) -> Path:
    """Return the path to the suggestion history file."""
    return Path(exp_root) / "reports" / "suggestion-history.json"


def log_suggestion(
    exp_root: str, pattern_id: str, scope: str = "session",
) -> None:
    """Log that a suggestion was generated for a pattern."""
    path = _suggestion_history_path(exp_root)
    history = get_suggestion_history(exp_root)

    # Compute iteration: count of existing entries with same pattern_id + 1
    iteration = sum(1 for s in history if s["pattern_id"] == pattern_id) + 1

    history.append({
        "pattern_id": pattern_id,
        "scope": scope,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iteration": iteration,
    })

    _atomic_write_json(path, {"suggestions": history})


def get_suggestion_history(exp_root: str) -> list[dict]:
    """Return list of previously generated suggestions."""
    path = _suggestion_history_path(exp_root)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
        return data.get("suggestions", [])
    except (json.JSONDecodeError, OSError):
        return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 3:
        print("Usage: error_tracker.py <exp_root> <action> [args...]", file=sys.stderr)
        print("Actions: log <event_json>, show [category], patterns, summary, sync <plugin_root>, success <metric> <lower>, proposals <metric> <lower>, rank [total], cleanup <plugin_root> [max_sessions], log-suggestion <pattern_id> [scope], suggestion-history, dead-end <add|list|check> [args], agenda <init|update|list|add> [args]", file=sys.stderr)
        sys.exit(1)

    exp_root = sys.argv[1]
    action = sys.argv[2]

    if action == "log":
        if len(sys.argv) < 4:
            print("Usage: error_tracker.py <exp_root> log <event_json>", file=sys.stderr)
            sys.exit(1)
        try:
            raw = json.loads(sys.argv[3])
        except json.JSONDecodeError:
            print("Error: invalid JSON", file=sys.stderr)
            sys.exit(1)
        # Create a proper event from the raw input
        try:
            ev = create_event(
                raw.get("category", ""),
                raw.get("severity", ""),
                raw.get("source", ""),
                raw.get("message", ""),
                exp_id=raw.get("exp_id"),
                skill=raw.get("skill"),
                agent=raw.get("agent"),
                phase=raw.get("phase"),
                iteration=raw.get("iteration"),
                code_branch=raw.get("code_branch"),
                config=raw.get("config"),
                metrics=raw.get("metrics"),
                stack_trace=raw.get("stack_trace"),
                context=raw.get("context"),
                resolution=raw.get("resolution"),
                project_id=raw.get("project_id"),
                duration_seconds=raw.get("duration_seconds"),
            )
        except ValueError as e:
            print(json.dumps({"error": str(e)}), file=sys.stderr)
            sys.exit(1)
        path = log_event(exp_root, ev)
        print(json.dumps({"logged": True, "path": path}))

    elif action == "show":
        category = sys.argv[3] if len(sys.argv) > 3 else None
        events = get_events(exp_root, category=category)
        print(json.dumps(events, indent=2))

    elif action == "patterns":
        events = get_events(exp_root)
        patterns = detect_patterns(events)
        print(json.dumps(patterns, indent=2))

    elif action == "summary":
        summary = summarize_session(exp_root)
        print(json.dumps(summary, indent=2))

    elif action == "sync":
        if len(sys.argv) < 4:
            print("Usage: error_tracker.py <exp_root> sync <plugin_root>", file=sys.stderr)
            sys.exit(1)
        plugin_root = sys.argv[3]
        # Derive project_path from exp_root (go up from experiments/)
        project_path = str(Path(exp_root).parent) if Path(exp_root).name == "experiments" else exp_root
        path = update_cross_project(plugin_root, project_path, exp_root)
        print(json.dumps({"synced": True, "path": path}))

    elif action == "success":
        if len(sys.argv) < 5:
            print("Usage: error_tracker.py <exp_root> success <metric> <lower_is_better>", file=sys.stderr)
            sys.exit(1)
        metric = sys.argv[3]
        lower = sys.argv[4].lower() in ("true", "1", "yes")
        result = compute_success_metrics(exp_root, metric, lower)
        print(json.dumps(result, indent=2))

    elif action == "proposals":
        if len(sys.argv) < 5:
            print("Usage: error_tracker.py <exp_root> proposals <metric> <lower_is_better>", file=sys.stderr)
            sys.exit(1)
        metric = sys.argv[3]
        lower = sys.argv[4].lower() in ("true", "1", "yes")
        result = compute_proposal_outcomes(exp_root, metric, lower)
        print(json.dumps(result, indent=2))

    elif action == "rank":
        events = get_events(exp_root)
        pats = detect_patterns(events)
        total = int(sys.argv[3]) if len(sys.argv) > 3 else None
        cross = None
        if len(sys.argv) > 4:
            memory = load_cross_project(sys.argv[4])
            if memory:
                cross = memory.get("cross_project_patterns", [])
        ranked = rank_suggestions(pats, total_experiments=total, cross_project_patterns=cross)
        print(json.dumps(ranked, indent=2))

    elif action == "cleanup":
        if len(sys.argv) < 4:
            print("Usage: error_tracker.py <exp_root> cleanup <plugin_root> [max_sessions]", file=sys.stderr)
            sys.exit(1)
        plugin_root = sys.argv[3]
        max_sessions = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        result = cleanup_memory(plugin_root, max_sessions)
        print(json.dumps(result))

    elif action == "log-suggestion":
        if len(sys.argv) < 4:
            print("Usage: error_tracker.py <exp_root> log-suggestion <pattern_id> [scope]", file=sys.stderr)
            sys.exit(1)
        pattern_id = sys.argv[3]
        scope = sys.argv[4] if len(sys.argv) > 4 else "session"
        log_suggestion(exp_root, pattern_id, scope)
        print(json.dumps({"logged": True, "pattern_id": pattern_id, "scope": scope}))

    elif action == "suggestion-history":
        history = get_suggestion_history(exp_root)
        print(json.dumps(history, indent=2))

    elif action == "dead-end":
        if len(sys.argv) < 4:
            print("Usage: error_tracker.py <exp_root> dead-end <add|list|check> [args]", file=sys.stderr)
            sys.exit(1)
        sub = sys.argv[3]
        if sub == "add":
            if len(sys.argv) < 5:
                print("Usage: error_tracker.py <exp_root> dead-end add <entry_json>", file=sys.stderr)
                sys.exit(1)
            try:
                entry = json.loads(sys.argv[4])
            except json.JSONDecodeError:
                print("Error: invalid JSON", file=sys.stderr)
                sys.exit(1)
            path = log_dead_end(exp_root, entry)
            print(json.dumps({"logged": True, "path": path}))
        elif sub == "list":
            dead_ends = get_dead_ends(exp_root)
            print(json.dumps(dead_ends, indent=2))
        elif sub == "check":
            if len(sys.argv) < 5:
                print("Usage: error_tracker.py <exp_root> dead-end check <technique_name>", file=sys.stderr)
                sys.exit(1)
            name = sys.argv[4]
            result = is_dead_end(exp_root, name)
            print(json.dumps({"technique": name, "is_dead_end": result}))
        else:
            print(f"Unknown dead-end sub-action: {sub}", file=sys.stderr)
            sys.exit(1)

    elif action == "agenda":
        if len(sys.argv) < 4:
            print("Usage: error_tracker.py <exp_root> agenda <init|update|list|add> [args]", file=sys.stderr)
            sys.exit(1)
        sub = sys.argv[3]
        if sub == "init":
            if len(sys.argv) < 5:
                print("Usage: error_tracker.py <exp_root> agenda init <ideas_json>", file=sys.stderr)
                sys.exit(1)
            try:
                ideas = json.loads(sys.argv[4])
            except json.JSONDecodeError:
                print("Error: invalid JSON", file=sys.stderr)
                sys.exit(1)
            if not isinstance(ideas, list):
                ideas = [ideas]
            path = init_agenda(exp_root, ideas)
            print(json.dumps({"initialized": True, "count": len(ideas), "path": path}))
        elif sub == "update":
            if len(sys.argv) < 6:
                print("Usage: error_tracker.py <exp_root> agenda update <idea_id> <updates_json>", file=sys.stderr)
                sys.exit(1)
            idea_id = sys.argv[4]
            try:
                updates = json.loads(sys.argv[5])
            except json.JSONDecodeError:
                print("Error: invalid JSON", file=sys.stderr)
                sys.exit(1)
            found = update_agenda_item(exp_root, idea_id, updates)
            print(json.dumps({"updated": found, "idea_id": idea_id}))
        elif sub == "list":
            ideas = get_agenda(exp_root)
            print(json.dumps(ideas, indent=2))
        elif sub == "add":
            if len(sys.argv) < 5:
                print("Usage: error_tracker.py <exp_root> agenda add <idea_json>", file=sys.stderr)
                sys.exit(1)
            try:
                idea = json.loads(sys.argv[4])
            except json.JSONDecodeError:
                print("Error: invalid JSON", file=sys.stderr)
                sys.exit(1)
            path = add_agenda_idea(exp_root, idea)
            print(json.dumps({"added": True, "path": path}))
        else:
            print(f"Unknown agenda sub-action: {sub}", file=sys.stderr)
            sys.exit(1)

    else:
        print(f"Unknown action: {action}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
