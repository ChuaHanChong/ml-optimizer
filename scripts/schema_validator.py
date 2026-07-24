#!/usr/bin/env python3
"""JSON schema validation for pipeline data.

Validates pipeline JSON files (experiment results, baseline, manifest, prerequisites,
HP proposals, rounds manifest) and inter-agent relay messages against their schemas.

Usage:
    python3 schema_validator.py <filepath> result                # Validate an experiment result JSON
    python3 schema_validator.py <filepath> baseline              # Validate baseline.json
    python3 schema_validator.py <filepath> manifest              # Validate implementation-manifest.json
    python3 schema_validator.py <filepath> prerequisites         # Validate prerequisites.json
    python3 schema_validator.py <filepath> hp_proposal           # Validate an HP tuning proposal JSON
    python3 schema_validator.py <filepath> rounds_manifest       # Validate rounds-manifest.json
    python3 schema_validator.py <filepath> result --strict       # Result validation with completeness enforced
    python3 schema_validator.py relay <route> <json_data>        # Validate an inter-agent relay message

--strict (result only) promotes completeness warnings to blocking errors. Relay
routes: analyze_to_tuning, analyze_to_research, monitor_to_tuning,
research_to_implement, research_to_tuning, experiments_to_analyze.
Exit code 0 = valid, 1 = invalid.
"""

import json
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

EXPERIMENT_RESULT_REQUIRED = ["exp_id", "status", "config", "metrics"]
EXPERIMENT_RESULT_OPTIONAL = [
    "gpu_id", "duration_seconds", "log_file", "script_file",
    "code_branch", "code_proposal", "notes",
    "method_tier", "proposal_source", "iteration",
    "code_branches", "stacking_order", "stack_base_exp",
    "artifacts_dir", "time_budget_seconds",
    "checkpoint_source", "warm_started",
    "reproducibility", "random_seed",
]
VALID_METHOD_TIERS = ["baseline", "method_default_hp", "method_tuned_hp", "stacked_default_hp", "stacked_tuned_hp"]
VALID_STATUSES = ["completed", "failed", "diverged", "running", "pending", "timeout"]

BASELINE_REQUIRED = ["exp_id", "status", "config", "metrics"]
BASELINE_OPTIONAL = ["profiling", "eval_command", "train_command", "notes", "model_category"]
VALID_MODEL_CATEGORIES = ["supervised", "rl", "generative"]

MANIFEST_REQUIRED = ["original_branch", "strategy", "proposals"]
MANIFEST_OPTIONAL = ["conflicts", "new_dependencies"]
VALID_STRATEGIES = ["git_branch", "file_backup"]

PROPOSAL_REQUIRED = ["name", "slug", "status"]
PROPOSAL_OPTIONAL = [
    "branch", "files_modified", "complexity", "validation",
    "commit_sha", "notes", "type",
    "implementation_strategy", "reference_repo", "reference_files_used",
    "adaptation_notes", "files_created", "license_warning", "new_dependencies",
    "proposal_source", "test_file", "explanation", "diff_summary",
]
VALID_PROPOSAL_STATUSES = ["validated", "validation_failed", "implementation_error", "preflight_failed"]
VALID_IMPLEMENTATION_STRATEGIES = ["from_scratch", "from_reference"]

PREREQUISITES_REQUIRED = ["status", "dataset", "environment", "ready_for_baseline"]
VALID_PREREQ_STATUSES = ["ready", "partial", "failed"]

HP_PROPOSAL_REQUIRED = ["exp_id", "config"]
HP_PROPOSAL_OPTIONAL = [
    "code_branch", "code_proposal", "proposal_source",
    "method_tier", "gpu_id", "reasoning", "iteration",
    "checkpoint_source", "warm_started", "random_seed",
]

ROUNDS_MANIFEST_REQUIRED = ["rounds", "current_round", "total_experiments"]
VALID_ROUND_TYPES = ["hp", "evolved", "research", "stacked"]

SEARCH_SPACE_ENTRY_REQUIRED = ["param", "range", "scale", "source"]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_required(data: dict, required: list[str]) -> list[str]:
    """Return error strings for any missing required fields."""
    errors = []
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    return errors


def _check_numeric_metrics(data: dict, errors: list[str]) -> None:
    """Append errors for any non-numeric or non-finite values in data["metrics"]."""
    if "metrics" in data and isinstance(data["metrics"], dict):
        for mk, mv in data["metrics"].items():
            if isinstance(mv, bool) or not isinstance(mv, (int, float)):
                errors.append(f"Metric '{mk}' must be numeric, got {type(mv).__name__}")
            elif isinstance(mv, float) and not math.isfinite(mv):
                errors.append(f"Metric '{mk}' must be finite, got {mv}")


def _check_random_seed(data: dict, errors: list[str]) -> None:
    """Append an error when `random_seed` is present but not an integer."""
    if "random_seed" in data:
        rs = data["random_seed"]
        if isinstance(rs, bool) or not isinstance(rs, int):
            errors.append("'random_seed' must be an integer")


# ---------------------------------------------------------------------------
# Public validation functions
# ---------------------------------------------------------------------------

def validate_result(data: dict) -> dict:
    """Validate an experiment result dict -> {"valid": bool, "errors": [...], "warnings": [...]}."""
    errors: list[str] = []

    if not isinstance(data, dict):
        return {"valid": False, "errors": ["Data must be a dict"]}

    errors.extend(_check_required(data, EXPERIMENT_RESULT_REQUIRED))

    if "status" in data and data["status"] not in VALID_STATUSES:
        errors.append(
            f"Invalid status '{data['status']}': must be one of {VALID_STATUSES}"
        )

    if "metrics" in data and not isinstance(data["metrics"], dict):
        errors.append("'metrics' must be a dict")
    else:
        _check_numeric_metrics(data, errors)

    if "config" in data and not isinstance(data["config"], dict):
        errors.append("'config' must be a dict")

    if "method_tier" in data and data["method_tier"] not in VALID_METHOD_TIERS:
        errors.append(
            f"Invalid method_tier '{data['method_tier']}': must be one of {VALID_METHOD_TIERS}"
        )

    if "code_branches" in data:
        if not isinstance(data["code_branches"], list):
            errors.append("'code_branches' must be a list")
        elif not all(isinstance(b, str) for b in data["code_branches"]):
            errors.append("'code_branches' elements must be strings")

    if "stacking_order" in data:
        if not isinstance(data["stacking_order"], int) or data["stacking_order"] < 1:
            errors.append("'stacking_order' must be a positive integer")

    if "stack_base_exp" in data:
        if not isinstance(data["stack_base_exp"], str):
            errors.append("'stack_base_exp' must be a string")

    if "checkpoint_source" in data:
        cs = data["checkpoint_source"]
        if cs is not None:
            if not isinstance(cs, dict):
                errors.append("'checkpoint_source' must be a dict or null")
            elif "exp_id" not in cs or "checkpoint_path" not in cs:
                errors.append("'checkpoint_source' must have 'exp_id' and 'checkpoint_path' keys")

    if "warm_started" in data:
        if not isinstance(data["warm_started"], bool):
            errors.append("'warm_started' must be a boolean")
        elif data["warm_started"] and not isinstance(data.get("checkpoint_source"), dict):
            errors.append("'warm_started' is true but 'checkpoint_source' is missing or null")

    _check_random_seed(data, errors)

    warnings = check_completeness(data) if not errors else []
    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def validate_baseline(data: dict) -> dict:
    """Validate a baseline result dict -> {"valid": bool, "errors": [...]}."""
    errors: list[str] = []

    if not isinstance(data, dict):
        return {"valid": False, "errors": ["Data must be a dict"]}

    errors.extend(_check_required(data, BASELINE_REQUIRED))

    if "status" in data and data["status"] not in VALID_STATUSES:
        errors.append(
            f"Invalid status '{data['status']}': must be one of {VALID_STATUSES}"
        )

    if "metrics" in data and not isinstance(data["metrics"], dict):
        errors.append("'metrics' must be a dict")
    else:
        _check_numeric_metrics(data, errors)

    if "config" in data and not isinstance(data["config"], dict):
        errors.append("'config' must be a dict")

    if "model_category" in data and data["model_category"] is not None:
        if data["model_category"] not in VALID_MODEL_CATEGORIES:
            errors.append(
                f"Invalid model_category '{data['model_category']}': "
                f"must be one of {VALID_MODEL_CATEGORIES} or null"
            )

    if "profiling" in data:
        prof = data["profiling"]
        if not isinstance(prof, dict):
            errors.append("'profiling' must be a dict")
        else:
            for key in ("estimated_timeout_seconds", "training_duration_seconds"):
                if key in prof and prof[key] is not None:
                    v = prof[key]
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        errors.append(f"'profiling.{key}' must be numeric or null")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": []}


def validate_manifest(data: dict) -> dict:
    """Validate a manifest dict, including each proposal in the proposals list."""
    errors: list[str] = []

    if not isinstance(data, dict):
        return {"valid": False, "errors": ["Data must be a dict"]}

    errors.extend(_check_required(data, MANIFEST_REQUIRED))

    if "strategy" in data and data["strategy"] not in VALID_STRATEGIES:
        errors.append(
            f"Invalid strategy '{data['strategy']}': must be one of {VALID_STRATEGIES}"
        )

    if "proposals" in data:
        if not isinstance(data["proposals"], list):
            errors.append("'proposals' must be a list")
        else:
            for i, proposal in enumerate(data["proposals"]):
                if not isinstance(proposal, dict):
                    errors.append(f"Proposal at index {i} must be a dict")
                    continue
                for field in PROPOSAL_REQUIRED:
                    if field not in proposal:
                        errors.append(
                            f"Proposal at index {i}: missing required field: {field}"
                        )
                if "status" in proposal and proposal["status"] not in VALID_PROPOSAL_STATUSES:
                    errors.append(
                        f"Proposal at index {i}: invalid status '{proposal['status']}': "
                        f"must be one of {VALID_PROPOSAL_STATUSES}"
                    )
                if "implementation_strategy" in proposal and proposal["implementation_strategy"] not in VALID_IMPLEMENTATION_STRATEGIES:
                    errors.append(
                        f"Proposal at index {i}: invalid implementation_strategy "
                        f"'{proposal['implementation_strategy']}': "
                        f"must be one of {VALID_IMPLEMENTATION_STRATEGIES}"
                    )

    return {"valid": len(errors) == 0, "errors": errors, "warnings": []}


def validate_prerequisites(data: dict) -> dict:
    """Validate a prerequisites report dict.

    Inner-field checks (missing dataset.train_path, prepared_train_path when
    prepared is True, environment.manager) are soft warnings, not errors.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, dict):
        return {"valid": False, "errors": ["Data must be a dict"], "warnings": []}

    errors.extend(_check_required(data, PREREQUISITES_REQUIRED))

    if "status" in data and data["status"] not in VALID_PREREQ_STATUSES:
        errors.append(
            f"Invalid status '{data['status']}': must be one of {VALID_PREREQ_STATUSES}"
        )

    if "dataset" in data and not isinstance(data["dataset"], dict):
        errors.append("'dataset' must be a dict")

    if "environment" in data and not isinstance(data["environment"], dict):
        errors.append("'environment' must be a dict")

    if "ready_for_baseline" in data and not isinstance(data["ready_for_baseline"], bool):
        errors.append("'ready_for_baseline' must be a boolean")

    # Inner field warnings (soft checks — don't block validation)
    ds = data.get("dataset")
    if isinstance(ds, dict):
        if "train_path" not in ds:
            warnings.append("dataset.train_path is missing")
        if ds.get("prepared") is True and "prepared_train_path" not in ds:
            warnings.append(
                "dataset.prepared is True but prepared_train_path is missing"
            )
        ptpath = ds.get("prepared_train_path")
        if ptpath is not None and not isinstance(ptpath, str):
            warnings.append("dataset.prepared_train_path should be a string")
        elif isinstance(ptpath, str) and not ptpath:
            warnings.append("dataset.prepared_train_path is empty")
        pvpath = ds.get("prepared_val_path")
        if pvpath is not None and not isinstance(pvpath, str):
            warnings.append("dataset.prepared_val_path should be a string")
        elif isinstance(pvpath, str) and not pvpath:
            warnings.append("dataset.prepared_val_path is empty")

    env = data.get("environment")
    if isinstance(env, dict):
        if "manager" not in env:
            warnings.append("environment.manager is missing")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def validate_hp_proposal(data: dict) -> dict:
    """Validate an HP tuning proposal dict (exp_id + config required)."""
    errors: list[str] = []

    if not isinstance(data, dict):
        return {"valid": False, "errors": ["Data must be a dict"]}

    errors.extend(_check_required(data, HP_PROPOSAL_REQUIRED))

    if "config" in data and not isinstance(data["config"], dict):
        errors.append("'config' must be a dict")

    if "method_tier" in data and data["method_tier"] not in VALID_METHOD_TIERS:
        errors.append(
            f"Invalid method_tier '{data['method_tier']}': must be one of {VALID_METHOD_TIERS}"
        )

    if "exp_id" in data and not isinstance(data["exp_id"], str):
        errors.append("'exp_id' must be a string")

    _check_random_seed(data, errors)

    return {"valid": len(errors) == 0, "errors": errors, "warnings": []}


def validate_rounds_manifest(data: dict) -> dict:
    """Validate a rounds-manifest.json dict (rounds list, current_round int, total_experiments >= 0)."""
    errors: list[str] = []

    if not isinstance(data, dict):
        return {"valid": False, "errors": ["Data must be a dict"]}

    errors.extend(_check_required(data, ROUNDS_MANIFEST_REQUIRED))

    if "current_round" in data and not isinstance(data["current_round"], int):
        errors.append("'current_round' must be an integer")

    if "total_experiments" in data:
        te = data["total_experiments"]
        if not isinstance(te, int) or te < 0:
            errors.append("'total_experiments' must be a non-negative integer")

    if "rounds" in data:
        if not isinstance(data["rounds"], list):
            errors.append("'rounds' must be a list")
        else:
            for i, rnd in enumerate(data["rounds"]):
                if not isinstance(rnd, dict):
                    errors.append(f"Round at index {i} must be a dict")
                    continue
                for field in ["id", "dir", "type"]:
                    if field not in rnd:
                        errors.append(f"Round at index {i}: missing required field: {field}")
                if "type" in rnd and rnd["type"] not in VALID_ROUND_TYPES:
                    errors.append(
                        f"Round at index {i}: invalid type '{rnd['type']}': "
                        f"must be one of {VALID_ROUND_TYPES}"
                    )
                if "experiments" in rnd and not isinstance(rnd["experiments"], list):
                    errors.append(f"Round at index {i}: 'experiments' must be a list")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": []}


def validate_file(filepath: str, schema_type: str) -> dict:
    """Read a JSON file and validate it against the named schema; adds 'filepath' to the result."""
    result: dict = {"filepath": filepath}
    path = Path(filepath)

    if not path.exists():
        result["valid"] = False
        result["errors"] = [f"File not found: {filepath}"]
        return result

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        result["valid"] = False
        result["errors"] = [f"Invalid JSON: {exc}"]
        return result

    validators = {
        "result": validate_result,
        "baseline": validate_baseline,
        "manifest": validate_manifest,
        "prerequisites": validate_prerequisites,
        "hp_proposal": validate_hp_proposal,
        "rounds_manifest": validate_rounds_manifest,
    }

    if schema_type not in validators:
        result["valid"] = False
        result["errors"] = [
            f"Unknown schema type '{schema_type}': must be one of {list(validators.keys())}"
        ]
        return result

    validation = validators[schema_type](data)
    result["valid"] = validation["valid"]
    result["errors"] = validation["errors"]
    if "warnings" in validation:
        result["warnings"] = validation["warnings"]
    return result


# ---------------------------------------------------------------------------
# Completeness checks (status-aware)
# ---------------------------------------------------------------------------

def check_completeness(data: dict) -> list[str]:
    """Return warnings for missing fields that should be present given the status.

    These are not structural errors — the result is valid but incomplete.
    In `--strict` mode, these become blocking errors. Also the single owner of
    this rule set for validate_experiment_write.py's PreToolUse hook.
    """
    warnings: list[str] = []
    status = data.get("status")

    if status == "completed":
        metrics = data.get("metrics", {})
        config = data.get("config", {})
        if isinstance(metrics, dict) and not metrics:
            warnings.append("Completed experiment has empty metrics")
        if isinstance(config, dict) and not config:
            warnings.append("Completed experiment has empty config")
        if "iteration" not in data:
            warnings.append("Completed experiment missing 'iteration'")
        if "method_tier" not in data:
            warnings.append("Completed experiment missing 'method_tier'")
        if "duration_seconds" not in data:
            warnings.append("Completed experiment missing 'duration_seconds'")
        if "eval_protocol" not in data:
            warnings.append("Completed experiment missing 'eval_protocol' (held_out_eval|train_report|rl_final_eval)")
        # Stacking completeness
        tier = data.get("method_tier", "")
        if isinstance(tier, str) and tier.startswith("stacked_"):
            if "code_branches" not in data:
                warnings.append("Stacked experiment missing 'code_branches'")
            if "stacking_order" not in data:
                warnings.append("Stacked experiment missing 'stacking_order'")

    elif status in ("failed", "diverged"):
        if "notes" not in data:
            warnings.append(f"{status.capitalize()} experiment missing 'notes' (explain what went wrong)")

    return warnings


def validate_result_strict(data: dict) -> dict:
    """Validate structure AND completeness. Warnings become errors."""
    result = validate_result(data)
    completeness = check_completeness(data)
    if completeness:
        result["errors"].extend(completeness)
        result["valid"] = False
    return result


def validate_search_space(entries) -> list[str]:
    """Validate a research-derived HP prior list [{param, range, scale, source}] -> list of error strings.

    Used for hp_only search_space payloads, research-agenda entries, and the
    research_to_tuning relay route; `source` cites the paper/URL the range came from.
    """
    errors: list[str] = []
    if not isinstance(entries, list):
        return ["'search_space' must be a list of {param, range, scale, source} entries"]
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"search_space[{i}] must be a dict")
            continue
        for field in SEARCH_SPACE_ENTRY_REQUIRED:
            if field not in entry:
                errors.append(f"search_space[{i}]: missing required field: {field}")
        if "param" in entry and not isinstance(entry["param"], str):
            errors.append(f"search_space[{i}]: 'param' must be a string")
        if "range" in entry and not isinstance(entry["range"], list):
            errors.append(f"search_space[{i}]: 'range' must be a list")
        if "scale" in entry and not isinstance(entry["scale"], str):
            errors.append(f"search_space[{i}]: 'scale' must be a string")
        if "source" in entry and not isinstance(entry["source"], str):
            errors.append(f"search_space[{i}]: 'source' must be a string")
    return errors


# ---------------------------------------------------------------------------
# Relay message schemas (inter-agent communication contracts)
# ---------------------------------------------------------------------------

RELAY_SCHEMAS = {
    "analyze_to_tuning": {
        "required": ["recommendation", "batch_number"],
        "optional": ["correlations", "branch_scores", "best_metric_value", "pivot_type", "dead_ends_summary"],
        "types": {"recommendation": str, "batch_number": int, "best_metric_value": (int, float)},
    },
    "analyze_to_research": {
        "required": ["pivot_reason"],
        "optional": ["dead_end_catalog", "improvement_gaps", "best_metric_value", "scope_level"],
        "types": {"pivot_reason": str},
    },
    "monitor_to_tuning": {
        "required": [],
        "optional": ["max_batch_size", "divergence_patterns", "oom_batch_sizes"],
        "types": {"max_batch_size": int},
    },
    "research_to_implement": {
        "required": ["findings_path"],
        "optional": ["scope_level", "selected_indices", "implementation_strategies"],
        "types": {"findings_path": str, "selected_indices": list},
    },
    "research_to_tuning": {
        "required": ["search_space"],
        "optional": ["findings_path", "model_category"],
        "types": {"search_space": list, "findings_path": str, "model_category": str},
    },
    "experiments_to_analyze": {
        "required": ["completed_ids"],
        "optional": ["best_metric_value", "diverged_count", "timeout_count", "batch_number"],
        "types": {"completed_ids": list, "diverged_count": int},
    },
}


def validate_relay(route: str, data: dict) -> dict:
    """Validate a relay message against its route schema -> {"valid": bool, "errors": [...], "warnings": [...]}."""
    errors: list[str] = []
    warnings: list[str] = []

    if route not in RELAY_SCHEMAS:
        errors.append(
            f"Unknown relay route '{route}': must be one of {sorted(RELAY_SCHEMAS.keys())}"
        )
        return {"valid": False, "errors": errors, "warnings": warnings}

    schema = RELAY_SCHEMAS[route]

    # Check required fields
    for field in schema["required"]:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # Check type constraints
    for field, expected_type in schema.get("types", {}).items():
        if field in data:
            if not isinstance(data[field], expected_type):
                errors.append(
                    f"Type mismatch for '{field}': expected {expected_type}, "
                    f"got {type(data[field]).__name__}"
                )

    # Element-level validation for research-derived HP priors (list form only —
    # analyze→tuning's dict-form search_space is a plain range dict, not priors).
    if isinstance(data.get("search_space"), list):
        errors.extend(validate_search_space(data["search_space"]))

    # Check for unexpected fields
    all_known = set(schema["required"]) | set(schema["optional"])
    for field in data:
        if field not in all_known:
            warnings.append(f"Unexpected field: {field}")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "relay":
        if len(sys.argv) < 4:
            print("Usage: schema_validator.py relay <route> <json_data>")
            sys.exit(1)
        route = sys.argv[2]
        try:
            data = json.loads(sys.argv[3])
        except json.JSONDecodeError as exc:
            output = {"valid": False, "errors": [f"Invalid JSON: {exc}"], "warnings": []}
            print(json.dumps(output, indent=2))
            sys.exit(1)
        output = validate_relay(route, data)
        print(json.dumps(output, indent=2))
        sys.exit(0 if output["valid"] else 1)

    if len(sys.argv) < 3:
        print("Usage: schema_validator.py <filepath> result|baseline|manifest|prerequisites|hp_proposal|rounds_manifest [--strict]")
        print("       schema_validator.py relay <route> <json_data>")
        sys.exit(1)
    filepath = sys.argv[1]
    schema_type = sys.argv[2]
    strict = "--strict" in sys.argv

    if strict and schema_type == "result":
        try:
            data = json.loads(Path(filepath).read_text())
        except (json.JSONDecodeError, OSError) as exc:
            output = {"filepath": filepath, "valid": False, "errors": [str(exc)]}
            print(json.dumps(output, indent=2))
            sys.exit(1)
        output = validate_result_strict(data)
        output["filepath"] = filepath
    else:
        output = validate_file(filepath, schema_type)

    print(json.dumps(output, indent=2))
    sys.exit(0 if output["valid"] else 1)
