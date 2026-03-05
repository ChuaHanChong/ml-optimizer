"""Tests for schema_validator.py."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from schema_validator import validate_result, validate_baseline, validate_manifest, validate_file


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


def test_validate_result_non_dict_input():
    """A non-dict input should fail validation."""
    result = validate_result("not a dict")
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


def test_validate_baseline_non_dict_input():
    """Non-dict baseline input fails."""
    result = validate_baseline("string")
    assert result["valid"] is False
    assert any("dict" in e.lower() for e in result["errors"])


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


def test_validate_manifest_non_dict_input():
    """Non-dict manifest input fails."""
    result = validate_manifest([1, 2, 3])
    assert result["valid"] is False
    assert any("dict" in e.lower() for e in result["errors"])


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


def test_validate_file_unknown_schema(tmp_path):
    """validate_file returns error for unknown schema type."""
    f = tmp_path / "data.json"
    f.write_text(json.dumps({"key": "value"}))
    result = validate_file(str(f), "unknown_schema")
    assert result["valid"] is False
    assert any("unknown" in e.lower() for e in result["errors"])


# --- CLI tests ---


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
