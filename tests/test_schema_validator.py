"""Tests for schema_validator.py."""

import json

import pytest

from schema_validator import validate_result, validate_baseline, validate_manifest, validate_file, validate_prerequisites


def test_validate_result_valid():
    """A fully valid experiment result passes validation."""
    data = {
        "exp_id": "exp-001",
        "status": "completed",
        "config": {"lr": 0.001, "batch_size": 16},
        "metrics": {"loss": 0.67, "accuracy": 82.5},
        "gpu_id": 0,
        "duration_seconds": 3600,
        "log_file": "experiments/logs/exp-001/train.log",
        "notes": "baseline run",
    }
    result = validate_result(data)
    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_result_missing_field():
    """A result without 'metrics' fails validation."""
    data = {
        "exp_id": "exp-001",
        "status": "completed",
        "config": {"lr": 0.001},
    }
    result = validate_result(data)
    assert result["valid"] is False
    assert any("metrics" in e for e in result["errors"])


def test_validate_result_invalid_status():
    """A result with status 'unknown' fails validation."""
    data = {
        "exp_id": "exp-001",
        "status": "unknown",
        "config": {"lr": 0.001},
        "metrics": {"loss": 0.5},
    }
    result = validate_result(data)
    assert result["valid"] is False
    assert any("unknown" in e for e in result["errors"])


def test_validate_result_timeout_status():
    """A result with status 'timeout' passes validation."""
    data = {
        "exp_id": "exp-001",
        "status": "timeout",
        "config": {"lr": 0.001},
        "metrics": {},
    }
    result = validate_result(data)
    assert result["valid"] is True


def test_validate_result_with_iteration():
    """A result with an iteration field passes validation."""
    data = {
        "exp_id": "exp-001",
        "status": "completed",
        "config": {"lr": 0.001},
        "metrics": {"loss": 0.5},
        "iteration": 3,
    }
    result = validate_result(data)
    assert result["valid"] is True


def test_validate_baseline_valid():
    """A valid baseline passes validation."""
    data = {
        "exp_id": "baseline",
        "status": "completed",
        "config": {"lr": 0.001, "batch_size": 32},
        "metrics": {"loss": 1.0, "accuracy": 70.0},
        "profiling": {"gpu_util": 0.85},
        "train_command": "python train.py",
    }
    result = validate_baseline(data)
    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_manifest_valid():
    """A valid manifest with proposals passes validation."""
    data = {
        "original_branch": "main",
        "strategy": "git_branch",
        "proposals": [
            {
                "name": "Increase learning rate",
                "slug": "increase-lr",
                "status": "validated",
                "branch": "opt/increase-lr",
                "files_modified": ["train.py"],
            },
            {
                "name": "Add warmup",
                "slug": "add-warmup",
                "status": "validation_failed",
                "notes": "Warmup caused instability",
            },
        ],
    }
    result = validate_manifest(data)
    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_manifest_invalid_proposal():
    """A proposal without 'name' fails validation."""
    data = {
        "original_branch": "main",
        "strategy": "git_branch",
        "proposals": [
            {
                "slug": "missing-name",
                "status": "validated",
            },
        ],
    }
    result = validate_manifest(data)
    assert result["valid"] is False
    assert any("name" in e for e in result["errors"])


def test_validate_result_empty_metrics():
    """A result with an empty metrics dict should still pass validation."""
    data = {
        "exp_id": "exp-001",
        "status": "completed",
        "config": {"lr": 0.001},
        "metrics": {},
    }
    result = validate_result(data)
    assert result["valid"] is True
    assert result["errors"] == []


@pytest.mark.parametrize("validator,bad_input", [
    (validate_result, "not a dict"),
    (validate_baseline, "string"),
    (validate_manifest, [1, 2, 3]),
    (validate_prerequisites, "string"),
])
def test_validate_non_dict_input(validator, bad_input):
    """Non-dict input should fail validation for all validators."""
    result = validator(bad_input)
    assert result["valid"] is False
    assert any("dict" in e.lower() for e in result["errors"])


def test_validate_manifest_empty_proposals():
    """A manifest with an empty proposals list should pass validation."""
    data = {
        "original_branch": "main",
        "strategy": "git_branch",
        "proposals": [],
    }
    result = validate_manifest(data)
    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_manifest_invalid_strategy():
    """A manifest with an invalid strategy should fail validation."""
    data = {
        "original_branch": "main",
        "strategy": "invalid_strategy",
        "proposals": [],
    }
    result = validate_manifest(data)
    assert result["valid"] is False
    assert any("strategy" in e.lower() for e in result["errors"])


def test_validate_file_invalid_json(tmp_path):
    """Validating a file with invalid JSON should fail gracefully."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{this is not valid json")
    result = validate_file(str(bad_file), "result")
    assert result["valid"] is False
    assert any("json" in e.lower() for e in result["errors"])


def test_validate_file_nonexistent():
    """Validating a nonexistent file returns an error."""
    result = validate_file("/nonexistent/path/to/file.json", "result")
    assert result["valid"] is False
    assert result["filepath"] == "/nonexistent/path/to/file.json"
    assert any("not found" in e.lower() for e in result["errors"])


# --- Type-check branches ---


def test_validate_result_non_dict_metrics():
    """Result with non-dict metrics fails."""
    data = {"exp_id": "exp-001", "status": "completed", "config": {}, "metrics": "not-a-dict"}
    result = validate_result(data)
    assert result["valid"] is False
    assert any("metrics" in e and "dict" in e for e in result["errors"])


def test_validate_result_non_dict_config():
    """Result with non-dict config fails."""
    data = {"exp_id": "exp-001", "status": "completed", "config": [1, 2], "metrics": {}}
    result = validate_result(data)
    assert result["valid"] is False
    assert any("config" in e and "dict" in e for e in result["errors"])


def test_validate_baseline_invalid_status():
    """Baseline with invalid status fails."""
    data = {"exp_id": "baseline", "status": "bogus", "config": {}, "metrics": {}}
    result = validate_baseline(data)
    assert result["valid"] is False
    assert any("bogus" in e for e in result["errors"])


def test_validate_baseline_non_dict_metrics():
    """Baseline with non-dict metrics fails."""
    data = {"exp_id": "baseline", "status": "completed", "config": {}, "metrics": 42}
    result = validate_baseline(data)
    assert result["valid"] is False
    assert any("metrics" in e and "dict" in e for e in result["errors"])


def test_validate_baseline_non_dict_config():
    """Baseline with non-dict config fails."""
    data = {"exp_id": "baseline", "status": "completed", "config": "bad", "metrics": {}}
    result = validate_baseline(data)
    assert result["valid"] is False
    assert any("config" in e and "dict" in e for e in result["errors"])


def test_validate_manifest_non_list_proposals():
    """Manifest with non-list proposals fails."""
    data = {"original_branch": "main", "strategy": "git_branch", "proposals": "not-a-list"}
    result = validate_manifest(data)
    assert result["valid"] is False
    assert any("proposals" in e and "list" in e for e in result["errors"])


def test_validate_manifest_non_dict_proposal_item():
    """Manifest with non-dict proposal item fails."""
    data = {"original_branch": "main", "strategy": "git_branch", "proposals": ["string-item"]}
    result = validate_manifest(data)
    assert result["valid"] is False
    assert any("dict" in e for e in result["errors"])


def test_validate_manifest_invalid_proposal_status():
    """Manifest with invalid proposal status fails."""
    data = {
        "original_branch": "main",
        "strategy": "git_branch",
        "proposals": [{"name": "Test", "slug": "test", "status": "bogus"}],
    }
    result = validate_manifest(data)
    assert result["valid"] is False
    assert any("bogus" in e for e in result["errors"])


# --- validate_file dispatch ---


def test_validate_file_valid_result(tmp_path):
    """validate_file dispatches to validate_result for result schema."""
    f = tmp_path / "exp.json"
    f.write_text(json.dumps({"exp_id": "exp-001", "status": "completed", "config": {}, "metrics": {}}))
    result = validate_file(str(f), "result")
    assert result["valid"] is True


def test_validate_file_valid_baseline(tmp_path):
    """validate_file dispatches to validate_baseline for baseline schema."""
    f = tmp_path / "baseline.json"
    f.write_text(json.dumps({"exp_id": "baseline", "status": "completed", "config": {}, "metrics": {}}))
    result = validate_file(str(f), "baseline")
    assert result["valid"] is True


def test_validate_file_valid_manifest(tmp_path):
    """validate_file dispatches to validate_manifest for manifest schema."""
    f = tmp_path / "manifest.json"
    f.write_text(json.dumps({
        "original_branch": "main",
        "strategy": "git_branch",
        "proposals": [{"name": "test", "slug": "test", "status": "validated"}],
    }))
    result = validate_file(str(f), "manifest")
    assert result["valid"] is True


def test_validate_file_unknown_schema(tmp_path):
    """validate_file returns error for unknown schema type."""
    f = tmp_path / "data.json"
    f.write_text(json.dumps({"key": "value"}))
    result = validate_file(str(f), "unknown_schema")
    assert result["valid"] is False
    assert any("unknown" in e.lower() for e in result["errors"])


# --- CLI tests ---


def test_validate_manifest_invalid_implementation_strategy():
    """Proposal with invalid implementation_strategy should fail."""
    data = {
        "original_branch": "main",
        "strategy": "git_branch",
        "proposals": [{
            "name": "Test",
            "slug": "test",
            "status": "validated",
            "implementation_strategy": "from_nowhere",
        }],
    }
    result = validate_manifest(data)
    assert result["valid"] is False
    assert any("implementation_strategy" in e for e in result["errors"])


def test_validate_manifest_valid_implementation_strategies():
    """Both valid implementation strategies should pass."""
    for strategy in ["from_scratch", "from_reference"]:
        data = {
            "original_branch": "main",
            "strategy": "git_branch",
            "proposals": [{
                "name": "Test",
                "slug": "test",
                "status": "validated",
                "implementation_strategy": strategy,
            }],
        }
        result = validate_manifest(data)
        assert result["valid"] is True, f"Strategy '{strategy}' should be valid"


def test_validate_manifest_from_reference_without_repo():
    """from_reference proposal without reference_repo still passes (field is optional)."""
    data = {
        "original_branch": "main",
        "strategy": "git_branch",
        "proposals": [{
            "name": "X",
            "slug": "x",
            "status": "validated",
            "implementation_strategy": "from_reference",
        }],
    }
    assert validate_manifest(data)["valid"] is True


def test_validate_result_non_numeric_metric():
    """A result with a non-numeric metric value should fail."""
    data = {
        "exp_id": "exp-001",
        "status": "completed",
        "config": {"lr": 0.001},
        "metrics": {"loss": 0.5, "model_name": "resnet"},
    }
    result = validate_result(data)
    assert result["valid"] is False
    assert any("model_name" in e and "numeric" in e for e in result["errors"])


def test_validate_result_bool_metric():
    """Boolean metric values are rejected (bool is subclass of int)."""
    data = {
        "exp_id": "exp-001",
        "status": "completed",
        "config": {},
        "metrics": {"converged": True},
    }
    result = validate_result(data)
    assert result["valid"] is False
    assert any("numeric" in e for e in result["errors"])


def test_validate_baseline_non_numeric_metric():
    """A baseline with a non-numeric metric value should fail."""
    data = {
        "exp_id": "baseline",
        "status": "completed",
        "config": {"lr": 0.001},
        "metrics": {"loss": 0.5, "notes": "good run"},
    }
    result = validate_baseline(data)
    assert result["valid"] is False
    assert any("notes" in e and "numeric" in e for e in result["errors"])


def test_validate_result_nan_metric():
    """A result with NaN metric value should fail."""
    data = {
        "exp_id": "exp-001",
        "status": "completed",
        "config": {"lr": 0.001},
        "metrics": {"loss": float("nan")},
    }
    result = validate_result(data)
    assert result["valid"] is False
    assert any("loss" in e and "finite" in e for e in result["errors"])


def test_validate_result_inf_metric():
    """A result with Inf metric value should fail."""
    data = {
        "exp_id": "exp-001",
        "status": "completed",
        "config": {"lr": 0.001},
        "metrics": {"loss": float("inf")},
    }
    result = validate_result(data)
    assert result["valid"] is False
    assert any("loss" in e and "finite" in e for e in result["errors"])


def test_validate_baseline_nan_metric():
    """A baseline with NaN metric value should fail."""
    data = {
        "exp_id": "baseline",
        "status": "completed",
        "config": {"lr": 0.001},
        "metrics": {"loss": float("nan")},
    }
    result = validate_baseline(data)
    assert result["valid"] is False
    assert any("loss" in e and "finite" in e for e in result["errors"])


def test_validate_result_neg_inf_metric():
    """A result with -Inf metric value should fail."""
    data = {
        "exp_id": "exp-001",
        "status": "completed",
        "config": {"lr": 0.001},
        "metrics": {"loss": float("-inf")},
    }
    result = validate_result(data)
    assert result["valid"] is False
    assert any("loss" in e and "finite" in e for e in result["errors"])


def test_validate_result_valid_method_tier():
    """A result with valid method_tier passes validation."""
    for tier in ["baseline", "method_default_hp", "method_tuned_hp"]:
        data = {
            "exp_id": "exp-001",
            "status": "completed",
            "config": {"lr": 0.001},
            "metrics": {"loss": 0.5},
            "method_tier": tier,
        }
        result = validate_result(data)
        assert result["valid"] is True, f"method_tier '{tier}' should be valid"


def test_validate_result_invalid_method_tier():
    """A result with invalid method_tier fails validation."""
    data = {
        "exp_id": "exp-001",
        "status": "completed",
        "config": {"lr": 0.001},
        "metrics": {"loss": 0.5},
        "method_tier": "invalid_tier",
    }
    result = validate_result(data)
    assert result["valid"] is False
    assert any("method_tier" in e for e in result["errors"])


def test_validate_result_without_method_tier():
    """A result without method_tier still passes (field is optional)."""
    data = {
        "exp_id": "exp-001",
        "status": "completed",
        "config": {"lr": 0.001},
        "metrics": {"loss": 0.5},
    }
    result = validate_result(data)
    assert result["valid"] is True


def test_cli_validate_valid(run_main, tmp_path):
    """CLI validates a valid result file."""
    f = tmp_path / "exp.json"
    f.write_text(json.dumps({"exp_id": "exp-001", "status": "completed", "config": {}, "metrics": {}}))
    r = run_main("schema_validator.py", str(f), "result")
    assert r.returncode == 0
    output = json.loads(r.stdout)
    assert output["valid"] is True


def test_cli_no_args(run_main):
    """CLI with no args prints usage and exits 1."""
    r = run_main("schema_validator.py")
    assert r.returncode == 1
    assert "Usage" in r.stdout


# --- validate_prerequisites tests ---


def test_validate_prerequisites_valid():
    """A fully valid prerequisites report passes validation."""
    data = {
        "status": "ready",
        "dataset": {
            "train_path": "/data/train",
            "format_detected": "image_folder",
            "validation_passed": True,
        },
        "environment": {
            "manager": "conda",
            "packages_installed": ["torch"],
            "all_imports_resolved": True,
        },
        "ready_for_baseline": True,
    }
    result = validate_prerequisites(data)
    assert result["valid"] is True
    assert result["errors"] == []
    assert result["warnings"] == []


def test_validate_prerequisites_partial_status():
    """A prerequisites report with partial status passes validation."""
    data = {
        "status": "partial",
        "dataset": {"notes": "format unknown"},
        "environment": {"notes": "missing packages"},
        "ready_for_baseline": False,
    }
    result = validate_prerequisites(data)
    assert result["valid"] is True


def test_validate_prerequisites_missing_required_field():
    """A prerequisites report missing 'dataset' fails validation."""
    data = {
        "status": "ready",
        "environment": {},
        "ready_for_baseline": True,
    }
    result = validate_prerequisites(data)
    assert result["valid"] is False
    assert any("dataset" in e for e in result["errors"])


def test_validate_prerequisites_invalid_status():
    """A prerequisites report with invalid status fails."""
    data = {
        "status": "unknown",
        "dataset": {},
        "environment": {},
        "ready_for_baseline": True,
    }
    result = validate_prerequisites(data)
    assert result["valid"] is False
    assert any("unknown" in e for e in result["errors"])


def test_validate_prerequisites_non_dict_dataset():
    """A prerequisites report with non-dict dataset fails."""
    data = {
        "status": "ready",
        "dataset": "not a dict",
        "environment": {},
        "ready_for_baseline": True,
    }
    result = validate_prerequisites(data)
    assert result["valid"] is False
    assert any("dataset" in e and "dict" in e for e in result["errors"])


def test_validate_prerequisites_non_dict_environment():
    """A prerequisites report with non-dict environment fails."""
    data = {
        "status": "ready",
        "dataset": {},
        "environment": ["not", "a", "dict"],
        "ready_for_baseline": True,
    }
    result = validate_prerequisites(data)
    assert result["valid"] is False
    assert any("environment" in e and "dict" in e for e in result["errors"])


def test_validate_prerequisites_non_bool_ready_for_baseline():
    """A prerequisites report with non-bool ready_for_baseline fails."""
    data = {
        "status": "ready",
        "dataset": {},
        "environment": {},
        "ready_for_baseline": "yes",
    }
    result = validate_prerequisites(data)
    assert result["valid"] is False
    assert any("ready_for_baseline" in e and "boolean" in e for e in result["errors"])


def test_validate_prerequisites_warns_missing_train_path():
    """Missing dataset.train_path produces a warning, not an error."""
    data = {
        "status": "ready",
        "dataset": {"format_detected": "csv"},
        "environment": {"manager": "pip"},
        "ready_for_baseline": True,
    }
    result = validate_prerequisites(data)
    assert result["valid"] is True
    assert any("train_path" in w for w in result["warnings"])


def test_validate_prerequisites_warns_missing_prepared_path():
    """When dataset.prepared=True but no prepared_train_path, produce a warning."""
    data = {
        "status": "ready",
        "dataset": {"train_path": "/data/train", "prepared": True},
        "environment": {"manager": "conda"},
        "ready_for_baseline": True,
    }
    result = validate_prerequisites(data)
    assert result["valid"] is True
    assert any("prepared_train_path" in w for w in result["warnings"])


def test_validate_prerequisites_warns_missing_env_manager():
    """Missing environment.manager produces a warning."""
    data = {
        "status": "ready",
        "dataset": {"train_path": "/data/train"},
        "environment": {"packages_installed": ["torch"]},
        "ready_for_baseline": True,
    }
    result = validate_prerequisites(data)
    assert result["valid"] is True
    assert any("manager" in w for w in result["warnings"])


def test_validate_prerequisites_no_warnings_when_complete():
    """Fully populated prerequisites report has no warnings."""
    data = {
        "status": "ready",
        "dataset": {
            "train_path": "/data/train",
            "val_path": "/data/val",
            "prepared": False,
        },
        "environment": {"manager": "conda", "packages_installed": ["torch"]},
        "ready_for_baseline": True,
    }
    result = validate_prerequisites(data)
    assert result["valid"] is True
    assert result["warnings"] == []


def test_validate_prerequisites_warns_empty_prepared_path():
    """Empty prepared_train_path triggers a warning."""
    data = {
        "status": "ready",
        "dataset": {
            "train_path": "/data/train",
            "prepared": True,
            "prepared_train_path": "",
        },
        "environment": {"manager": "conda"},
        "ready_for_baseline": True,
    }
    result = validate_prerequisites(data)
    assert result["valid"] is True
    assert "dataset.prepared_train_path is empty" in result["warnings"]


def test_validate_prerequisites_warns_empty_prepared_val_path():
    """Empty prepared_val_path triggers a warning."""
    data = {
        "status": "ready",
        "dataset": {
            "train_path": "/data/train",
            "prepared": True,
            "prepared_train_path": "/prep/train",
            "prepared_val_path": "",
        },
        "environment": {"manager": "conda"},
        "ready_for_baseline": True,
    }
    result = validate_prerequisites(data)
    assert result["valid"] is True
    assert "dataset.prepared_val_path is empty" in result["warnings"]


def test_validate_prerequisites_warns_non_string_prepared_train_path():
    """Non-string prepared_train_path triggers a warning."""
    data = {
        "status": "ready",
        "dataset": {
            "train_path": "/data/train",
            "prepared": True,
            "prepared_train_path": 123,
        },
        "environment": {"manager": "conda"},
        "ready_for_baseline": True,
    }
    result = validate_prerequisites(data)
    assert result["valid"] is True
    assert "dataset.prepared_train_path should be a string" in result["warnings"]


def test_validate_file_valid_prerequisites(tmp_path):
    """validate_file dispatches to validate_prerequisites for prerequisites schema."""
    f = tmp_path / "prerequisites.json"
    f.write_text(json.dumps({
        "status": "ready",
        "dataset": {},
        "environment": {},
        "ready_for_baseline": True,
    }))
    result = validate_file(str(f), "prerequisites")
    assert result["valid"] is True


def test_validate_all_validators_return_warnings_key():
    """All validators return a 'warnings' key for API consistency."""
    for name, validator, data in [
        ("validate_result", validate_result, {"exp_id": "e", "status": "completed", "config": {}, "metrics": {}}),
        ("validate_baseline", validate_baseline, {"exp_id": "b", "status": "completed", "config": {}, "metrics": {}}),
        ("validate_manifest", validate_manifest, {"original_branch": "main", "strategy": "git_branch", "proposals": []}),
        ("validate_prerequisites", validate_prerequisites, {"status": "ready", "dataset": {}, "environment": {}, "ready_for_baseline": True}),
    ]:
        out = validator(data)
        assert "warnings" in out, f"{name} missing 'warnings' key"
        assert isinstance(out["warnings"], list), f"{name} 'warnings' not a list"
