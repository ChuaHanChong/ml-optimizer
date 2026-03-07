#!/usr/bin/env python3
"""JSON schema validation for pipeline data.

Dependency-free validation of experiment results, baselines, and manifests.
Schemas are defined as plain Python data structures.
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
]
VALID_STATUSES = ["completed", "failed", "diverged", "running", "pending"]

BASELINE_REQUIRED = ["exp_id", "status", "config", "metrics"]
BASELINE_OPTIONAL = ["profiling", "eval_command", "train_command", "notes"]

MANIFEST_REQUIRED = ["original_branch", "strategy", "proposals"]
MANIFEST_OPTIONAL = ["conflicts", "new_dependencies"]
VALID_STRATEGIES = ["git_branch", "file_backup"]

PROPOSAL_REQUIRED = ["name", "slug", "status"]
PROPOSAL_OPTIONAL = [
    "branch", "files_modified", "complexity", "validation",
    "commit_sha", "notes", "type",
    "implementation_strategy", "reference_repo", "reference_files_used",
    "adaptation_notes", "files_created", "license_warning", "new_dependencies",
]
VALID_PROPOSAL_STATUSES = ["validated", "validation_failed", "implementation_error"]
VALID_IMPLEMENTATION_STRATEGIES = ["from_scratch", "from_reference"]

PREREQUISITES_REQUIRED = ["status", "dataset", "environment", "ready_for_baseline"]
VALID_PREREQ_STATUSES = ["ready", "partial", "failed"]


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
            if not isinstance(mv, (int, float)):
                errors.append(f"Metric '{mk}' must be numeric, got {type(mv).__name__}")
            elif isinstance(mv, float) and not math.isfinite(mv):
                errors.append(f"Metric '{mk}' must be finite, got {mv}")


# ---------------------------------------------------------------------------
# Public validation functions
# ---------------------------------------------------------------------------

def validate_result(data: dict) -> dict:
    """Validate an experiment result dict.

    Checks:
    - All required fields are present.
    - ``status`` is one of the valid status values.
    - ``metrics`` is a dict.
    - ``config`` is a dict.

    Returns ``{"valid": True/False, "errors": [...]}``.
    """
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

    return {"valid": len(errors) == 0, "errors": errors}


def validate_baseline(data: dict) -> dict:
    """Validate a baseline result dict.

    Checks that all required baseline fields are present.

    Returns ``{"valid": True/False, "errors": [...]}``.
    """
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

    return {"valid": len(errors) == 0, "errors": errors}


def validate_manifest(data: dict) -> dict:
    """Validate a manifest dict, including each proposal in the proposals list.

    Checks:
    - All required manifest fields are present.
    - ``strategy`` is one of the valid strategies.
    - ``proposals`` is a list and each proposal has the required fields.
    - Each proposal ``status`` is valid.

    Returns ``{"valid": True/False, "errors": [...]}``.
    """
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

    return {"valid": len(errors) == 0, "errors": errors}


def validate_prerequisites(data: dict) -> dict:
    """Validate a prerequisites report dict.

    Checks:
    - All required fields are present.
    - ``status`` is one of the valid prerequisite statuses.
    - ``dataset`` and ``environment`` are dicts.
    - ``ready_for_baseline`` is a boolean.
    - Inner field warnings for missing ``dataset.train_path``,
      ``dataset.prepared_train_path`` (when ``prepared`` is True),
      and ``environment.manager``.

    Returns ``{"valid": True/False, "errors": [...], "warnings": [...]}``.
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


def validate_file(filepath: str, schema_type: str) -> dict:
    """Read a JSON file and validate it against the specified schema.

    Args:
        filepath: Path to the JSON file.
        schema_type: One of ``"result"``, ``"baseline"``, or ``"manifest"``.

    Returns ``{"valid": True/False, "errors": [...], "filepath": ...}``.
    """
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
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: schema_validator.py <filepath> <schema_type>")
        sys.exit(1)
    filepath = sys.argv[1]
    schema_type = sys.argv[2]
    output = validate_file(filepath, schema_type)
    print(json.dumps(output, indent=2))
    sys.exit(0 if output["valid"] else 1)
