"""Tests for pipeline_state.py."""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from pipeline_state import validate_phase_requirements, save_state, load_state, cleanup_stale


def test_validate_phase_requirements_phase5_valid(tmp_path):
    """Phase 5 validates when baseline.json has proper schema."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    baseline = {
        "metrics": {"loss": 0.5, "accuracy": 85.0},
        "config": {"lr": 0.001, "batch_size": 32},
    }
    (results_dir / "baseline.json").write_text(json.dumps(baseline))

    result = validate_phase_requirements(5, str(tmp_path))
    assert result["valid"] is True
    assert result["phase"] == 5
    assert result["missing"] == []
    assert result["warnings"] == []


def test_validate_phase_requirements_phase5_missing_baseline(tmp_path):
    """Phase 5 fails when baseline.json does not exist."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    result = validate_phase_requirements(5, str(tmp_path))
    assert result["valid"] is False
    assert result["phase"] == 5
    assert len(result["missing"]) > 0
    assert any("baseline.json" in m for m in result["missing"])


def test_validate_phase_requirements_phase5_invalid_manifest(tmp_path):
    """Phase 5 warns when implementation-manifest.json lacks 'proposals' key."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    baseline = {
        "metrics": {"loss": 0.5},
        "config": {"lr": 0.001},
    }
    (results_dir / "baseline.json").write_text(json.dumps(baseline))
    manifest = {"items": ["something"]}
    (results_dir / "implementation-manifest.json").write_text(json.dumps(manifest))

    result = validate_phase_requirements(5, str(tmp_path))
    assert result["valid"] is True
    assert result["phase"] == 5
    assert result["missing"] == []
    assert len(result["warnings"]) > 0
    assert any("proposals" in w for w in result["warnings"])


def test_save_and_load_state(tmp_path):
    """Save state, load it back, and verify fields match."""
    exp_ids = ["exp-001", "exp-002"]
    path = save_state(3, 2, exp_ids, str(tmp_path))
    assert Path(path).exists()

    state = load_state(str(tmp_path))
    assert state is not None
    assert state["phase"] == 3
    assert state["iteration"] == 2
    assert state["running_experiments"] == exp_ids
    assert state["status"] == "running"
    assert "timestamp" in state
    # Verify timestamp is valid ISO format
    datetime.fromisoformat(state["timestamp"])


def test_cleanup_stale(tmp_path):
    """A stale pipeline-state.json (3 hours old) gets marked as interrupted."""
    stale_time = datetime.now(timezone.utc) - timedelta(hours=3)
    state = {
        "phase": 4,
        "iteration": 1,
        "running_experiments": ["exp-001"],
        "timestamp": stale_time.isoformat(),
        "status": "running",
    }
    (tmp_path / "pipeline-state.json").write_text(json.dumps(state))

    cleaned = cleanup_stale(str(tmp_path), timeout_hours=2.0)
    assert len(cleaned) > 0
    assert any("interrupted" in item for item in cleaned)

    # Verify the file was updated
    updated = json.loads((tmp_path / "pipeline-state.json").read_text())
    assert updated["status"] == "interrupted"
    assert "interrupted_at" in updated
