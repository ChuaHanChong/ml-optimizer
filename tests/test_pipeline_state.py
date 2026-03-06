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


def test_save_and_load_state_with_user_choices(tmp_path):
    """Save state with user_choices, load it back, verify choices persist."""
    choices = {
        "primary_metric": "accuracy",
        "divergence_metric": "loss",
        "lower_is_better": False,
        "target_value": 0.95,
    }
    save_state(5, 1, ["exp-001"], str(tmp_path), user_choices=choices)
    state = load_state(str(tmp_path))
    assert state is not None
    assert state["user_choices"] == choices
    assert state["user_choices"]["primary_metric"] == "accuracy"
    assert state["user_choices"]["lower_is_better"] is False


def test_save_state_without_user_choices_has_no_key(tmp_path):
    """Save state without user_choices omits the key entirely."""
    save_state(3, 1, [], str(tmp_path))
    state = load_state(str(tmp_path))
    assert state is not None
    assert "user_choices" not in state


def test_validate_phase2_valid(tmp_path):
    """Phase 2 passes when results/ directory exists."""
    (tmp_path / "results").mkdir()
    result = validate_phase_requirements(2, str(tmp_path))
    assert result["valid"] is True
    assert result["phase"] == 2
    assert result["missing"] == []


def test_validate_phase2_missing_results(tmp_path):
    """Phase 2 fails when results/ directory does not exist."""
    result = validate_phase_requirements(2, str(tmp_path))
    assert result["valid"] is False
    assert any("results" in m for m in result["missing"])


def test_validate_phase3_valid(tmp_path):
    """Phase 3 passes when baseline.json has metrics and config."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    baseline = {"metrics": {"loss": 0.5}, "config": {"lr": 0.001}}
    (results_dir / "baseline.json").write_text(json.dumps(baseline))

    result = validate_phase_requirements(3, str(tmp_path))
    assert result["valid"] is True
    assert result["missing"] == []


def test_validate_phase3_missing_keys(tmp_path):
    """Phase 3 fails when baseline.json is missing required keys."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    baseline = {"metrics": {"loss": 0.5}}  # missing 'config'
    (results_dir / "baseline.json").write_text(json.dumps(baseline))

    result = validate_phase_requirements(3, str(tmp_path))
    assert result["valid"] is False
    assert any("config" in m for m in result["missing"])


def test_validate_phase4_valid(tmp_path):
    """Phase 4 passes when baseline.json exists."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "baseline.json").write_text(json.dumps({"metrics": {}, "config": {}}))

    result = validate_phase_requirements(4, str(tmp_path))
    assert result["valid"] is True
    assert result["missing"] == []


def test_validate_phase4_missing_baseline(tmp_path):
    """Phase 4 fails when baseline.json does not exist."""
    (tmp_path / "results").mkdir()
    result = validate_phase_requirements(4, str(tmp_path))
    assert result["valid"] is False
    assert any("baseline.json" in m for m in result["missing"])


def test_load_state_corrupt_json(tmp_path):
    """Loading corrupt JSON returns None."""
    (tmp_path / "pipeline-state.json").write_text("{invalid json")
    state = load_state(str(tmp_path))
    assert state is None


def test_cleanup_stale_skips_recent(tmp_path):
    """A recently-updated running state should NOT be cleaned up."""
    recent_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    state = {
        "phase": 5,
        "iteration": 1,
        "running_experiments": ["exp-001"],
        "timestamp": recent_time.isoformat(),
        "status": "running",
    }
    (tmp_path / "pipeline-state.json").write_text(json.dumps(state))

    cleaned = cleanup_stale(str(tmp_path), timeout_hours=2.0)
    assert len(cleaned) == 0

    # Verify the file was NOT modified
    updated = json.loads((tmp_path / "pipeline-state.json").read_text())
    assert updated["status"] == "running"


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


# --- Validation error paths ---


def test_validate_phase3_corrupt_json(tmp_path):
    """Phase 3 fails when baseline.json is corrupt JSON."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "baseline.json").write_text("{bad json")
    result = validate_phase_requirements(3, str(tmp_path))
    assert result["valid"] is False
    assert any("not valid JSON" in m for m in result["missing"])


def test_validate_phase3_missing_baseline_file(tmp_path):
    """Phase 3 fails when baseline.json does not exist."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    result = validate_phase_requirements(3, str(tmp_path))
    assert result["valid"] is False
    assert any("baseline.json" in m for m in result["missing"])


def test_validate_phase3_missing_metrics_key(tmp_path):
    """Phase 3 fails when baseline.json has config but not metrics."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "baseline.json").write_text(json.dumps({"config": {"lr": 0.001}}))
    result = validate_phase_requirements(3, str(tmp_path))
    assert result["valid"] is False
    assert any("metrics" in m for m in result["missing"])


def test_validate_phase5_corrupt_json(tmp_path):
    """Phase 5 fails when baseline.json is corrupt JSON."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "baseline.json").write_text("{bad")
    result = validate_phase_requirements(5, str(tmp_path))
    assert result["valid"] is False
    assert any("not valid JSON" in m for m in result["missing"])


def test_validate_phase5_missing_metrics_key(tmp_path):
    """Phase 5 fails when baseline.json is missing metrics."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "baseline.json").write_text(json.dumps({"config": {"lr": 0.001}}))
    result = validate_phase_requirements(5, str(tmp_path))
    assert result["valid"] is False
    assert any("metrics" in m for m in result["missing"])


def test_validate_phase5_missing_config_key(tmp_path):
    """Phase 5 fails when baseline.json is missing config."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "baseline.json").write_text(json.dumps({"metrics": {"loss": 0.5}}))
    result = validate_phase_requirements(5, str(tmp_path))
    assert result["valid"] is False
    assert any("config" in m for m in result["missing"])


def test_validate_phase5_corrupt_manifest(tmp_path):
    """Phase 5 warns when manifest is corrupt JSON."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "baseline.json").write_text(json.dumps({"metrics": {"loss": 0.5}, "config": {"lr": 0.001}}))
    (results_dir / "implementation-manifest.json").write_text("{bad")
    result = validate_phase_requirements(5, str(tmp_path))
    assert result["valid"] is True
    assert any("not valid JSON" in w for w in result["warnings"])


# --- load_state / cleanup_stale ---


def test_load_state_no_file(tmp_path):
    """Loading state when no file exists returns None."""
    assert load_state(str(tmp_path)) is None


def test_cleanup_stale_corrupt_state_json(tmp_path):
    """Corrupt pipeline-state.json is handled gracefully."""
    (tmp_path / "pipeline-state.json").write_text("{bad")
    cleaned = cleanup_stale(str(tmp_path))
    assert cleaned == []


def test_cleanup_stale_naive_timestamp(tmp_path):
    """Naive timestamp (no tzinfo) is treated as UTC."""
    naive_time = (datetime.now(timezone.utc) - timedelta(hours=3)).replace(tzinfo=None)
    state = {
        "phase": 5,
        "iteration": 1,
        "running_experiments": [],
        "timestamp": naive_time.isoformat(),
        "status": "running",
    }
    (tmp_path / "pipeline-state.json").write_text(json.dumps(state))
    cleaned = cleanup_stale(str(tmp_path), timeout_hours=2.0)
    assert len(cleaned) > 0
    assert any("interrupted" in c for c in cleaned)


def test_cleanup_stale_invalid_timestamp(tmp_path):
    """Invalid timestamp string is handled gracefully (no crash)."""
    state = {
        "phase": 5,
        "iteration": 1,
        "running_experiments": [],
        "timestamp": "not-a-date",
        "status": "running",
    }
    (tmp_path / "pipeline-state.json").write_text(json.dumps(state))
    cleaned = cleanup_stale(str(tmp_path), timeout_hours=2.0)
    # Invalid timestamp => ValueError caught, state left as-is
    assert cleaned == []


def test_cleanup_stale_missing_timestamp(tmp_path):
    """State file with status=running but no timestamp key: no crash, no cleanup."""
    state = {
        "phase": 5,
        "iteration": 1,
        "running_experiments": [],
        "status": "running",
        # No "timestamp" key at all
    }
    (tmp_path / "pipeline-state.json").write_text(json.dumps(state))
    cleaned = cleanup_stale(str(tmp_path), timeout_hours=2.0)
    # Missing timestamp → .get returns "" → fromisoformat raises → caught → skip
    assert cleaned == []


def test_cleanup_stale_exp_files_stale(tmp_path):
    """Stale running exp-*.json in results/ are marked as failed."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    stale_time = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    exp = {"status": "running", "timestamp": stale_time, "exp_id": "exp-001"}
    (results_dir / "exp-001.json").write_text(json.dumps(exp))
    cleaned = cleanup_stale(str(tmp_path), timeout_hours=2.0)
    assert any("exp-001" in c for c in cleaned)
    data = json.loads((results_dir / "exp-001.json").read_text())
    assert data["status"] == "failed"


def test_cleanup_stale_exp_files_corrupt(tmp_path):
    """Corrupt exp-*.json files are skipped without error."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "exp-001.json").write_text("{bad")
    cleaned = cleanup_stale(str(tmp_path), timeout_hours=2.0)
    assert cleaned == []


def test_cleanup_stale_exp_files_not_running(tmp_path):
    """Completed experiments are not cleaned up."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    stale_time = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    exp = {"status": "completed", "timestamp": stale_time, "exp_id": "exp-001"}
    (results_dir / "exp-001.json").write_text(json.dumps(exp))
    cleaned = cleanup_stale(str(tmp_path), timeout_hours=2.0)
    assert cleaned == []


def test_cleanup_stale_exp_files_bad_timestamp(tmp_path):
    """Experiment with invalid timestamp is skipped."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    exp = {"status": "running", "timestamp": "not-a-date", "exp_id": "exp-001"}
    (results_dir / "exp-001.json").write_text(json.dumps(exp))
    cleaned = cleanup_stale(str(tmp_path), timeout_hours=2.0)
    assert cleaned == []


# --- CLI tests ---


def test_cli_no_args(run_main):
    """CLI with no args prints usage and exits 1."""
    r = run_main("pipeline_state.py")
    assert r.returncode == 1
    assert "Usage" in r.stdout


def test_cli_validate(run_main, tmp_path):
    """CLI validate action works."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "baseline.json").write_text(json.dumps({"metrics": {"loss": 0.5}, "config": {"lr": 0.001}}))
    r = run_main("pipeline_state.py", str(tmp_path), "validate", "3")
    assert r.returncode == 0
    output = json.loads(r.stdout)
    assert output["valid"] is True


def test_cli_save(run_main, tmp_path):
    """CLI save action writes state file."""
    r = run_main("pipeline_state.py", str(tmp_path), "save", "3", "1")
    assert r.returncode == 0
    assert "saved" in r.stdout.lower()
    assert (tmp_path / "pipeline-state.json").exists()


def test_cli_load_exists(run_main, tmp_path):
    """CLI load action returns state when file exists."""
    save_state(3, 1, [], str(tmp_path))
    r = run_main("pipeline_state.py", str(tmp_path), "load")
    assert r.returncode == 0
    output = json.loads(r.stdout)
    assert output["phase"] == 3


def test_cli_load_missing(run_main, tmp_path):
    """CLI load action reports no state when file missing."""
    r = run_main("pipeline_state.py", str(tmp_path), "load")
    assert r.returncode == 0
    assert "no pipeline state" in r.stdout.lower()


def test_cli_cleanup(run_main, tmp_path):
    """CLI cleanup action runs successfully."""
    r = run_main("pipeline_state.py", str(tmp_path), "cleanup")
    assert r.returncode == 0
    assert "nothing to clean" in r.stdout.lower()


def test_cli_validate_non_integer_phase(run_main, tmp_path):
    """CLI validate with non-integer phase exits cleanly."""
    r = run_main("pipeline_state.py", str(tmp_path), "validate", "abc")
    assert r.returncode == 1
    assert "Error" in r.stdout
    assert "phase" in r.stdout.lower()


def test_cli_save_invalid_args(run_main, tmp_path):
    """CLI save with non-integer iteration exits cleanly."""
    r = run_main("pipeline_state.py", str(tmp_path), "save", "5", "not_int")
    assert r.returncode == 1
    assert "Error" in r.stdout
    assert "iteration" in r.stdout.lower()


def test_cli_unknown_action(run_main, tmp_path):
    """CLI with unknown action exits 1."""
    r = run_main("pipeline_state.py", str(tmp_path), "bogus")
    assert r.returncode == 1
    assert "unknown" in r.stdout.lower()
