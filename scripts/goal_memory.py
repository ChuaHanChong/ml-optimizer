#!/usr/bin/env python3
"""Goal anchoring and behavioral memory for the ML optimizer plugin.

Maintains optimization-goals.json (goal anchor, written once at Phase 0) and
learned-behaviors.json (constraints, method outcomes, divergence patterns) under
`<exp_root>/`. Validates agent output against goals to catch drift (frozen-param
changes, scope breaches, dead-end re-proposals, metric mismatches) and emits a
compact agent briefing. See _USAGE for CLI actions/categories.

validate-output exits: 0 = valid, 1 = script error, 2 = violations found.
"""

import fcntl
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# --- Constants ---

BEHAVIOR_CATEGORIES = [
    "hp_constraint",
    "method_outcome",
    "divergence_pattern",
    "resource_constraint",
    "training_insight",
    "scope_violation",
    "goal_update",
    "evolve_hp",
]

GOALS_REQUIRED_PATHS = [
    ("objective", "primary_metric"),
    ("objective", "lower_is_better"),
]

# File-pattern heuristics for scope validation (warnings, not violations).
# Architecture-scope indicators: supervised model files + RL policy/value nets.
ARCH_FILE_PATTERNS = [
    "model", "network", "backbone", "encoder", "decoder", "arch",
    "policy", "critic", "actor",
]
# Env-dynamics indicators — env dynamics / curriculum / domain randomization
# are FULL scope only (RL scope rows). Reward shaping stays training-scope,
# so "reward" is intentionally NOT in this list.
# ponytail: substring heuristic ("env" also matches e.g. "venv/") — acceptable
# for warning-only output; tighten to path-segment matching if noise appears.
ENV_FILE_PATTERNS = ["env", "environment", "curriculum", "randomization", "domain_rand"]

# Domain-randomization parameters are proposed as a center/width scalar pair
# rather than an explicit [lo, hi] range: hp-tune proposes scalars, and a pair
# can never produce an inverted range no matter what values it picks (there is
# no cross-parameter constraint mechanism to catch lo > hi).
DR_SUFFIXES = ("_center", "_width")


def dr_params(config: dict) -> list[str]:
    """Base names in `config` having BOTH a `_center` and a `_width` key.

    Requiring both avoids false positives on unrelated names (e.g. a
    `hidden_width` layer size is not domain randomization).
    """
    names = {
        key[: -len(suffix)]
        for key in config
        for suffix in DR_SUFFIXES
        if key.endswith(suffix)
    }
    return sorted(n for n in names if f"{n}_center" in config and f"{n}_width" in config)


def dr_effective_range(config: dict, name: str) -> tuple[float, float] | None:
    """`(low, high)` that `name`'s center/width pair expands to, or None.

    Returns None when the pair is missing or non-numeric. `|width|` is used so a
    negative width proposal still yields an ordered range instead of an inverted one.
    """
    center, width = config.get(f"{name}_center"), config.get(f"{name}_width")
    if isinstance(center, bool) or isinstance(width, bool):
        return None
    if not isinstance(center, (int, float)) or not isinstance(width, (int, float)):
        return None
    half = abs(float(width)) / 2.0
    return float(center) - half, float(center) + half

# --- File paths ---

def _goals_path(exp_root: str) -> Path:
    return Path(exp_root) / "optimization-goals.json"


def _behaviors_path(exp_root: str) -> Path:
    return Path(exp_root) / "learned-behaviors.json"


def _ensure_scripts_path() -> None:
    """Put this scripts/ dir on sys.path for lazy error_tracker imports."""
    d = str(Path(__file__).parent)
    if d not in sys.path:
        sys.path.insert(0, d)


# --- Atomic writes (same pattern as error_tracker.py) ---

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


# --- Goals: init / load / validate schema ---

def validate_goals(goals: dict) -> dict:
    """Validate a goals dict. Returns {"valid": bool, "errors": [...]}."""
    errors: list[str] = []
    if not isinstance(goals, dict):
        return {"valid": False, "errors": ["Goals must be a dict"]}
    for keys in GOALS_REQUIRED_PATHS:
        obj = goals
        for k in keys:
            if not isinstance(obj, dict) or k not in obj:
                errors.append(f"Missing required field: {'.'.join(keys)}")
                break
            obj = obj[k]
    return {"valid": len(errors) == 0, "errors": errors}


def init_goals(exp_root: str, goals: dict) -> str:
    """Write optimization-goals.json. Returns file path."""
    result = validate_goals(goals)
    if not result["valid"]:
        raise ValueError(f"Invalid goals: {'; '.join(result['errors'])}")

    goals = dict(goals)  # shallow copy to avoid mutating caller's dict
    goals.setdefault("version", 1)
    goals.setdefault("created_at", datetime.now(timezone.utc).isoformat())

    path = _goals_path(exp_root)
    _atomic_write_json(path, goals)
    return str(path)


def update_goals(exp_root: str, updates: dict) -> dict:
    """Merge partial updates into optimization-goals.json (nested dicts merged,
    not replaced) and log the change under category 'goal_update'.

    Returns {"updated": True, "changes": [...], "path": str} on success,
    {"updated": False, "error": str} on failure.
    """
    goals = load_goals(exp_root)
    if goals is None:
        return {"updated": False, "error": "No optimization-goals.json found"}

    changes: list[str] = []

    def _merge(target: dict, source: dict, prefix: str = "") -> None:
        for key, value in source.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                _merge(target[key], value, full_key)
            elif key in target and target[key] != value:
                changes.append(f"{full_key}: {target[key]} → {value}")
                target[key] = value
            elif key not in target:
                changes.append(f"{full_key}: (new) → {value}")
                target[key] = value

    _merge(goals, updates)

    if not changes:
        return {"updated": False, "error": "No changes detected"}

    result = validate_goals(goals)
    if not result["valid"]:
        return {"updated": False, "error": f"Invalid after update: {'; '.join(result['errors'])}"}

    goals["updated_at"] = datetime.now(timezone.utc).isoformat()
    path = _goals_path(exp_root)
    _atomic_write_json(path, goals)

    log_behavior(exp_root, "goal_update", {
        "changes": changes,
        "timestamp": goals["updated_at"],
    })

    return {"updated": True, "changes": changes, "path": str(path)}


def load_goals(exp_root: str) -> dict | None:
    """Load optimization-goals.json. Returns None if missing or corrupt."""
    path = _goals_path(exp_root)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# --- Behaviors: log / query ---

def load_behaviors(exp_root: str) -> dict:
    """Load learned-behaviors.json, returning empty structure if absent.

    Public alongside load_goals — validate_experiment_write.py's PreToolUse hook
    reads both to run check_goal_compliance at write time.
    """
    path = _behaviors_path(exp_root)
    if not path.is_file():
        return _empty_behaviors()
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return _empty_behaviors()
        return data
    except (json.JSONDecodeError, OSError):
        return _empty_behaviors()


def _empty_behaviors() -> dict:
    return {
        "version": 1,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "hp_constraints": [],
        "method_outcomes": [],
        "divergence_patterns": [],
        "resource_constraints": [],
        "training_insights": [],
        "scope_violations": [],
        "evolve_hp": [],
    }


def _category_key(category: str) -> str:
    """Map singular CLI category to plural JSON key."""
    mapping = {
        "hp_constraint": "hp_constraints",
        "method_outcome": "method_outcomes",
        "divergence_pattern": "divergence_patterns",
        "resource_constraint": "resource_constraints",
        "training_insight": "training_insights",
        "scope_violation": "scope_violations",
    }
    return mapping.get(category, category)


def log_behavior(exp_root: str, category: str, entry: dict) -> str:
    """Append a learned behavior. Returns file path."""
    if category not in BEHAVIOR_CATEGORIES:
        raise ValueError(
            f"Invalid category: {category}. "
            f"Must be one of: {', '.join(BEHAVIOR_CATEGORIES)}"
        )

    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    path = _behaviors_path(exp_root)
    lock_path = path.with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            data = load_behaviors(exp_root)
            key = _category_key(category)
            if key not in data:
                data[key] = []
            data[key].append(entry)
            data["last_updated"] = datetime.now(timezone.utc).isoformat()
            _atomic_write_json(path, data)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    return str(path)


def get_behaviors(
    exp_root: str,
    category: str | None = None,
    recent: int | None = None,
) -> list[dict]:
    """Query learned behaviors, optionally filtered by category."""
    data = load_behaviors(exp_root)
    if category is not None:
        key = _category_key(category)
        items = data.get(key, [])
    else:
        items = []
        for cat in BEHAVIOR_CATEGORIES:
            key = _category_key(cat)
            for entry in data.get(key, []):
                entry_copy = dict(entry)
                entry_copy["_category"] = cat
                items.append(entry_copy)
        # Sort by timestamp
        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    if recent is not None and recent > 0:
        items = items[-recent:] if category else items[:recent]
    return items


# --- Validation ---

def max_batch_size(behaviors: dict) -> int | None:
    """Tightest OOM batch-size ceiling across resource_constraints, or None."""
    best = None
    for rc in behaviors.get("resource_constraints", []):
        mbs = rc.get("max_batch_size")
        if mbs is not None:
            best = mbs if best is None else min(best, mbs)
    return best


def check_goal_compliance(
    config: dict, frozen: list, max_bs: int | None, scope: str | None = None
) -> list[str]:
    """Return violations for one config: frozen-parameter edits, OOM breaches, and
    (when scope == "training") domain-randomization parameters that change environment
    dynamics — `training` scope must leave the environment untouched, the same intent
    ENV_FILE_PATTERNS enforces for env-file edits.

    Single owner of this rule set — used by the hp-tune/experiment validators
    below and by validate_experiment_write.py's PreToolUse hook (the actual live
    enforcement point: every Write to proposed-configs/round-N-*/exp-*.json AND
    results/round-N-*/exp-*.json).
    """
    violations: list[str] = []
    for fp in frozen:
        if fp in config:
            violations.append(
                f"modifies frozen parameter '{fp}' — this parameter cannot be changed"
            )
    if max_bs is not None and "batch_size" in config:
        try:
            bs = int(config["batch_size"])
            if bs > int(max_bs):
                violations.append(f"batch_size={bs} exceeds OOM limit {max_bs}")
        except (ValueError, TypeError):
            pass
    if scope == "training":
        for name in dr_params(config):
            violations.append(
                f"tunes domain-randomization parameter '{name}' "
                f"(needs scope_level 'architecture' or 'full', got 'training')"
            )
    return violations


def validate_agent_output(exp_root: str, agent: str, output: dict) -> dict:
    """Validate agent output against goals. Returns {valid, violations, warnings}."""
    violations: list[str] = []
    warnings: list[str] = []

    goals = load_goals(exp_root)
    if goals is None:
        return {"valid": True, "violations": [], "warnings": ["No optimization-goals.json found"]}

    behaviors = load_behaviors(exp_root)
    constraints = goals.get("constraints", {})
    objective = goals.get("objective", {})
    frozen = constraints.get("frozen_parameters", [])
    scope = constraints.get("scope_level", "full")

    if agent == "hp-tune":
        _validate_hp_tune(output, frozen, behaviors, exp_root, scope, violations, warnings)
    elif agent == "research":
        _validate_research(output, scope, exp_root, violations, warnings)
    elif agent == "analyze":
        _validate_analyze(output, objective, violations, warnings)
    elif agent == "implement":
        _validate_implement(output, scope, violations, warnings)
    elif agent == "experiment":
        _validate_experiment(output, frozen, behaviors, scope, violations, warnings)

    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "warnings": warnings,
    }


def _validate_hp_tune(output, frozen, behaviors, exp_root, scope, violations, warnings):
    """Validate hp-tune proposed configs."""
    configs = output.get("configs", [])
    if not isinstance(configs, list):
        configs = [output] if isinstance(output, dict) and "lr" in output else []

    max_bs = max_batch_size(behaviors)

    # HP upper bounds from hp_constraints
    hp_bounds: dict[str, float] = {}
    for hc in behaviors.get("hp_constraints", []):
        param = hc.get("parameter")
        if param and hc.get("constraint_type") == "upper_bound":
            val = hc.get("value")
            if val is not None:
                if param not in hp_bounds or val < hp_bounds[param]:
                    hp_bounds[param] = val

    for i, cfg in enumerate(configs):
        if not isinstance(cfg, dict):
            continue
        violations.extend(
            f"Config {i}: {v}" for v in check_goal_compliance(cfg, frozen, max_bs, scope)
        )
        # HP bounds (warnings, not violations — agents may intentionally explore)
        for param, bound in hp_bounds.items():
            if param in cfg:
                try:
                    val = float(cfg[param])
                    if val > bound:
                        warnings.append(
                            f"Config {i}: {param}={val} exceeds learned bound {bound}"
                        )
                except (ValueError, TypeError):
                    pass

    # Dead-end branch check (import lazily to avoid circular deps)
    try:
        _ensure_scripts_path()
        from error_tracker import is_dead_end  # lazy: avoid circular dep
        for i, cfg in enumerate(configs):
            if not isinstance(cfg, dict):
                continue
            branch = cfg.get("code_branch") or cfg.get("branch")
            if branch and is_dead_end(exp_root, branch):
                violations.append(
                    f"Config {i}: targets dead-end branch '{branch}'"
                )
    except ImportError:
        warnings.append("Could not import error_tracker for dead-end check")


def _validate_research(output, scope, exp_root, violations, warnings):
    """Validate research proposals against scope and dead-ends."""
    proposals = output.get("proposals", [])
    if not isinstance(proposals, list):
        return

    for i, prop in enumerate(proposals):
        if not isinstance(prop, dict):
            continue

        # Scope check
        prop_scope = prop.get("scope", "")
        if prop_scope == "architecture" and scope == "training":
            violations.append(
                f"Proposal {i} ('{prop.get('name', '?')}'): "
                f"architecture change violates scope_level='training'"
            )
        elif prop.get("type") == "code_change" and scope == "training":
            # Heuristic: code_change could be training-scope or architecture-scope
            # Check files_to_modify for architecture patterns
            files = prop.get("files_to_modify", [])
            arch_files = [f for f in files if any(p in str(f).lower() for p in ARCH_FILE_PATTERNS)]
            if arch_files:
                warnings.append(
                    f"Proposal {i} ('{prop.get('name', '?')}'): "
                    f"code_change modifies possible architecture files: {arch_files}"
                )

        # Env-dynamics heuristic: env/curriculum/randomization files are
        # full-scope only (RL scope rows) — warn at training AND architecture.
        if prop.get("type") == "code_change" and scope in ("training", "architecture"):
            files = prop.get("files_to_modify", [])
            env_files = [f for f in files if any(p in str(f).lower() for p in ENV_FILE_PATTERNS)]
            if env_files:
                warnings.append(
                    f"Proposal {i} ('{prop.get('name', '?')}'): "
                    f"code_change modifies possible env-dynamics files "
                    f"(full scope only): {env_files}"
                )

        # Dead-end check
        technique = prop.get("name", prop.get("technique", ""))
        if technique:
            try:
                _ensure_scripts_path()
                from error_tracker import is_dead_end  # lazy: avoid circular dep
                if is_dead_end(exp_root, technique):
                    violations.append(
                        f"Proposal {i}: re-proposes dead-end technique '{technique}'"
                    )
            except ImportError:
                warnings.append("Could not import error_tracker for dead-end check")

        # Confidence cap for knowledge mode
        confidence = prop.get("confidence")
        source = prop.get("proposal_source", prop.get("source", ""))
        if source == "llm_knowledge" and confidence is not None:
            try:
                if float(confidence) > 7:
                    warnings.append(
                        f"Proposal {i}: knowledge-mode confidence {confidence} > 7 cap"
                    )
            except (ValueError, TypeError):
                pass


def _validate_analyze(output, objective, violations, warnings):
    """Validate analyze output matches goal metrics."""
    # Check metric name alignment
    out_metric = output.get("primary_metric")
    goal_metric = objective.get("primary_metric")
    if out_metric and goal_metric and out_metric != goal_metric:
        violations.append(
            f"Metric mismatch: analysis uses '{out_metric}' "
            f"but goal is '{goal_metric}'"
        )

    # Check polarity alignment
    out_polarity = output.get("lower_is_better")
    goal_polarity = objective.get("lower_is_better")
    if out_polarity is not None and goal_polarity is not None:
        if out_polarity != goal_polarity:
            violations.append(
                f"Polarity mismatch: analysis uses lower_is_better={out_polarity} "
                f"but goal is {goal_polarity}"
            )


def _validate_implement(output, scope, violations, warnings):
    """Validate implement changes stay within scope."""
    if scope == "full":
        return  # full scope allows all changes

    changes = output.get("files_modified", output.get("changes", []))
    if not isinstance(changes, list):
        return

    for f in changes:
        fname = str(f).lower() if isinstance(f, str) else ""
        # Heuristic: model/policy definition files suggest architecture changes
        if scope == "training" and any(p in fname for p in ARCH_FILE_PATTERNS):
            warnings.append(
                f"File '{f}' may contain architecture changes "
                f"(scope_level='training')"
            )
        # Env dynamics / curriculum / randomization are full-scope only
        if any(p in fname for p in ENV_FILE_PATTERNS):
            warnings.append(
                f"File '{f}' may contain env-dynamics changes "
                f"(full scope only; scope_level='{scope}')"
            )


def _validate_experiment(output, frozen, behaviors, scope, violations, warnings):
    """Validate experiment config respects frozen params, OOM limits, and (at
    training scope) domain-randomization parameters."""
    cfg = output.get("config", {})
    if not isinstance(cfg, dict):
        return

    violations.extend(
        f"Experiment {v}"
        for v in check_goal_compliance(cfg, frozen, max_batch_size(behaviors), scope)
    )


# --- Summary generation ---

def generate_summary(exp_root: str) -> str:
    """Generate a compact agent briefing combining goals + behaviors + dead-ends."""
    lines: list[str] = []

    goals = load_goals(exp_root)
    behaviors = load_behaviors(exp_root)

    # --- Goals section ---
    if goals:
        obj = goals.get("objective", {})
        con = goals.get("constraints", {})
        metric = obj.get("primary_metric", "?")
        lib = obj.get("lower_is_better", True)
        direction = "lower is better" if lib else "higher is better"
        target = obj.get("target_value")
        scope = con.get("scope_level", "?")
        lines.append("=== OPTIMIZATION GOALS ===")
        target_str = f", Target: {target}" if target is not None else ""
        lines.append(f"Metric: {metric} ({direction}){target_str}")
        lines.append(f"Scope: {scope}")

        frozen = con.get("frozen_parameters", [])
        if frozen:
            lines.append(f"Frozen parameters: {', '.join(str(f) for f in frozen)}")

        desc = obj.get("problem_description")
        if desc:
            lines.append(f"Goal: {desc}")

        lines.append("")
    else:
        lines.append("=== OPTIMIZATION GOALS ===")
        lines.append("WARNING: No optimization-goals.json found")
        lines.append("")

    # --- Learned constraints ---
    hp_constraints = behaviors.get("hp_constraints", [])
    resource_constraints = behaviors.get("resource_constraints", [])
    frozen_from_goals = (goals or {}).get("constraints", {}).get("frozen_parameters", [])

    if hp_constraints or resource_constraints or frozen_from_goals:
        lines.append("=== LEARNED CONSTRAINTS ===")
        for hc in hp_constraints:
            param = hc.get("parameter", "?")
            ctype = hc.get("constraint_type", "bound")
            val = hc.get("value", "?")
            reason = hc.get("reason", "")
            ev_count = hc.get("evidence_count", "")
            ev_str = f" ({ev_count} events)" if ev_count else ""
            lines.append(f"- {param} {ctype}: {val}{ev_str}")
            if reason:
                lines.append(f"  Reason: {reason}")
        for rc in resource_constraints:
            mbs = rc.get("max_batch_size")
            if mbs:
                lines.append(f"- max batch_size: {mbs} (OOM above this)")
            notes = rc.get("notes")
            if notes:
                lines.append(f"  {notes}")
        for fp in frozen_from_goals:
            lines.append(f"- {fp} is FROZEN by user")
        lines.append("")

    # --- Dead ends ---
    dead_ends: list[dict] = []
    try:
        _ensure_scripts_path()
        from error_tracker import get_dead_ends  # lazy: avoid circular dep
        dead_ends = get_dead_ends(exp_root)
    except ImportError:
        pass

    method_dead_ends = [
        mo for mo in behaviors.get("method_outcomes", [])
        if mo.get("outcome") == "dead_end"
    ]

    if dead_ends or method_dead_ends:
        lines.append("=== DEAD ENDS (do not re-propose) ===")
        seen = set()
        for de in dead_ends:
            tech = de.get("technique", "?")
            reason = de.get("reason", "")
            if tech.lower() not in seen:
                lines.append(f"- {tech}: {reason}")
                seen.add(tech.lower())
        for mo in method_dead_ends:
            method = mo.get("method", "?")
            if method.lower() not in seen:
                reason = mo.get("reason", mo.get("hp_sensitivity", ""))
                lines.append(f"- {method}: {reason}")
                seen.add(method.lower())
        lines.append("")

    # --- What works ---
    good_outcomes = [
        mo for mo in behaviors.get("method_outcomes", [])
        if mo.get("outcome") == "improved"
    ]
    good_insights = behaviors.get("training_insights", [])

    if good_outcomes or good_insights:
        lines.append("=== WHAT WORKS ===")
        for mo in good_outcomes:
            method = mo.get("method", "?")
            pct = mo.get("improvement_pct")
            hp_note = mo.get("hp_sensitivity", "")
            try:
                pct_str = f": +{float(pct):.2f}%" if pct is not None else ""
            except (ValueError, TypeError):
                pct_str = f": +{pct}%" if pct is not None else ""
            hp_str = f" ({hp_note})" if hp_note else ""
            lines.append(f"- {method}{pct_str}{hp_str}")
        for ti in good_insights[-5:]:  # last 5 insights
            insight = ti.get("insight", "?")
            lines.append(f"- {insight}")
        lines.append("")

    # --- Divergence patterns ---
    div_patterns = behaviors.get("divergence_patterns", [])
    if div_patterns:
        lines.append("=== DIVERGENCE PATTERNS ===")
        for dp in div_patterns[-3:]:  # last 3
            desc = dp.get("description", dp.get("pattern", "?"))
            lines.append(f"- {desc}")
        lines.append("")

    # --- Research agenda (top untried) ---
    try:
        from error_tracker import get_agenda  # lazy: avoid circular dep with error_tracker
        agenda = get_agenda(exp_root)
        untried = [a for a in agenda if a.get("status") == "untried"]
        untried.sort(key=lambda x: x.get("priority", 0), reverse=True)
        if untried:
            lines.append("=== HIGH-PRIORITY UNTRIED IDEAS ===")
            for item in untried[:3]:
                name = item.get("name", "?")
                pri = item.get("priority", "?")
                lines.append(f"- {name} (priority: {pri})")
            lines.append("")
    except ImportError:
        pass

    return "\n".join(lines).strip()


# --- Sync from error tracker ---

def sync_from_errors(exp_root: str) -> dict:
    """Pull OOM / divergence patterns from error_tracker into behaviors.

    Returns {"synced": int, "skipped": int}.
    """
    synced = 0
    skipped = 0

    try:
        _ensure_scripts_path()
        from error_tracker import get_events, get_dead_ends  # lazy: avoid circular dep
    except ImportError:
        return {"synced": 0, "skipped": 0, "error": "error_tracker not found"}

    behaviors = load_behaviors(exp_root)

    # --- OOM events → resource_constraints ---
    oom_events = [
        e for e in get_events(exp_root, category="training_failure")
        if "oom" in e.get("message", "").lower()
        or "out of memory" in e.get("message", "").lower()
        or (isinstance(e.get("context"), dict)
            and e["context"].get("error_type") == "oom")
    ]

    existing_bs = {
        rc.get("max_batch_size")
        for rc in behaviors.get("resource_constraints", [])
    }

    for e in oom_events:
        cfg = e.get("config", {})
        if isinstance(cfg, dict) and "batch_size" in cfg:
            bs = cfg["batch_size"]
            if bs not in existing_bs:
                log_behavior(exp_root, "resource_constraint", {
                    "resource": "gpu_memory",
                    "max_batch_size": bs,
                    "notes": f"OOM at batch_size={bs}",
                    "source": "sync_from_errors",
                })
                existing_bs.add(bs)
                synced += 1
            else:
                skipped += 1

    # --- Divergence events → hp_constraints ---
    div_events = get_events(exp_root, category="divergence")
    existing_hp = {
        (hc.get("parameter"), hc.get("value"))
        for hc in behaviors.get("hp_constraints", [])
        if hc.get("source") == "sync_from_errors"
    }

    # Find the min LR that diverged — used as an upper-bound signal (LRs at/above this risk divergence)
    div_lrs = []
    for e in div_events:
        cfg = e.get("config", {})
        if isinstance(cfg, dict) and "lr" in cfg:
            try:
                lr_val = float(cfg["lr"])
                if math.isfinite(lr_val):
                    div_lrs.append(lr_val)
            except (ValueError, TypeError):
                pass

    if div_lrs and ("lr", min(div_lrs)) not in existing_hp:
        min_div_lr = min(div_lrs)
        log_behavior(exp_root, "hp_constraint", {
            "parameter": "lr",
            "constraint_type": "upper_bound",
            "value": min_div_lr,
            "reason": f"All LRs >= {min_div_lr} diverged ({len(div_lrs)} events)",
            "evidence_count": len(div_lrs),
            "source": "sync_from_errors",
        })
        synced += 1
    elif div_lrs:
        skipped += 1

    # --- Dead ends → method_outcomes ---
    dead_ends = get_dead_ends(exp_root)
    existing_de = {
        mo.get("method", "").lower()
        for mo in behaviors.get("method_outcomes", [])
        if mo.get("outcome") == "dead_end" and mo.get("source") == "sync_from_errors"
    }

    for de in dead_ends:
        tech = de.get("technique", "")
        if tech.lower() not in existing_de:
            log_behavior(exp_root, "method_outcome", {
                "method": tech,
                "outcome": "dead_end",
                "reason": de.get("reason", ""),
                "branch": de.get("branch", ""),
                "source": "sync_from_errors",
            })
            existing_de.add(tech.lower())
            synced += 1
        else:
            skipped += 1

    return {"synced": synced, "skipped": skipped}


# --- CLI ---

_USAGE = """\
Usage: python3 goal_memory.py <exp_root> <action> [args...]

Actions:
  init-goals <goals_json>              Write optimization-goals.json
  update-goals <updates_json>          Merge partial updates into goals (mid-run)
  read-goals                           Print goals to stdout
  log-behavior <category> <entry_json> Append a learned behavior
  query-behaviors [category] [recent]  Query behaviors (optionally filtered, recent=N for last N)
  validate-output <agent> <output_json> Validate agent output against goals
  summary                              Compact agent briefing
  sync-from-errors                     Pull OOM/divergence from error_tracker

Categories: hp_constraint, method_outcome, divergence_pattern,
            resource_constraint, training_insight, scope_violation,
            goal_update, evolve_hp
"""


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]

    if len(args) < 2:
        print(_USAGE, file=sys.stderr)
        return 1

    exp_root = args[0]
    action = args[1]

    try:
        if action == "init-goals":
            if len(args) < 3:
                print("Error: init-goals requires a JSON string argument", file=sys.stderr)
                return 1
            goals = json.loads(args[2])
            path = init_goals(exp_root, goals)
            print(json.dumps({"status": "ok", "path": path}))

        elif action == "update-goals":
            if len(args) < 3:
                print("Error: update-goals requires a JSON string argument", file=sys.stderr)
                return 1
            updates = json.loads(args[2])
            result = update_goals(exp_root, updates)
            print(json.dumps(result, indent=2))
            return 0 if result.get("updated") else 1

        elif action == "read-goals":
            goals = load_goals(exp_root)
            if goals is None:
                print(json.dumps({"error": "not found"}))
                return 1
            print(json.dumps(goals, indent=2))

        elif action == "log-behavior":
            if len(args) < 4:
                print("Error: log-behavior requires <category> <entry_json>", file=sys.stderr)
                return 1
            category = args[2]
            entry = json.loads(args[3])
            path = log_behavior(exp_root, category, entry)
            print(json.dumps({"status": "ok", "path": path}))

        elif action == "query-behaviors":
            category = args[2] if len(args) > 2 else None
            recent = None
            if len(args) > 3:
                try:
                    recent = int(args[3])
                except ValueError:
                    pass
            items = get_behaviors(exp_root, category=category, recent=recent)
            print(json.dumps(items, indent=2))

        elif action == "validate-output":
            if len(args) < 4:
                print("Error: validate-output requires <agent> <output_json>", file=sys.stderr)
                return 1
            agent = args[2]
            output = json.loads(args[3])
            result = validate_agent_output(exp_root, agent, output)
            print(json.dumps(result, indent=2))
            return 0 if result["valid"] else 2

        elif action == "summary":
            text = generate_summary(exp_root)
            print(text)

        elif action == "sync-from-errors":
            result = sync_from_errors(exp_root)
            print(json.dumps(result, indent=2))

        else:
            print(f"Unknown action: {action}", file=sys.stderr)
            print(_USAGE, file=sys.stderr)
            return 1

    except json.JSONDecodeError as exc:
        print(f"Invalid JSON: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
