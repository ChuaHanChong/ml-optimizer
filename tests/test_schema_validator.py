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
