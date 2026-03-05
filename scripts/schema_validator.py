#!/usr/bin/env python3
"""JSON schema validation for pipeline data.

Dependency-free validation of experiment results, baselines, and manifests.
Schemas are defined as plain Python data structures.
"""

import json
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
]
VALID_PROPOSAL_STATUSES = ["validated", "validation_failed", "implementation_error"]


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

    return {"valid": len(errors) == 0, "errors": errors}


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
