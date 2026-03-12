"""Tests for pipeline_state.py.

Phase numbering (after prerequisites addition):
  Phase 2: prerequisites (no file requirements)
  Phase 3: baseline (results/ dir, prerequisites.json warning)
  Phase 4: checkpoint (baseline.json with metrics+config)
  Phase 5: research (baseline.json exists)
  Phase 6: experiment loop (baseline.json valid + manifest check)
"""

import json
import os as _os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

from pipeline_state import validate_phase_requirements, save_state, load_state, cleanup_stale


# --- Phase 2: Prerequisites (no file requirements) ---


def test_validate_phase2_always_valid(tmp_path):
    """Phase 2 (prerequisites) has no file-based requirements."""
    result = validate_phase_requirements(2, str(tmp_path))
    assert result["valid"] is True
    assert result["phase"] == 2
    assert result["missing"] == []


# --- Phase 3: Baseline (results/ dir + prerequisites warning) ---


def test_validate_phase3_valid(tmp_path):
    """Phase 3 passes when results/ directory exists."""
    (tmp_path / "results").mkdir()
    result = validate_phase_requirements(3, str(tmp_path))
    assert result["valid"] is True
    assert result["phase"] == 3
    assert result["missing"] == []


def test_validate_phase3_missing_results(tmp_path):
    """Phase 3 fails when results/ directory does not exist."""
    result = validate_phase_requirements(3, str(tmp_path))
    assert result["valid"] is False
    assert any("results" in m for m in result["missing"])


def test_validate_phase3_prerequisites_not_ready_blocks(tmp_path):
    """Phase 3 blocks when prerequisites.json says not ready for baseline."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    prereq = {"status": "failed", "dataset": {}, "environment": {}, "ready_for_baseline": False}
    (results_dir / "prerequisites.json").write_text(json.dumps(prereq))
    result = validate_phase_requirements(3, str(tmp_path))
    assert result["valid"] is False
    assert any("prerequisites" in m.lower() or "ready_for_baseline" in m.lower() for m in result["missing"])


def test_validate_phase3_prerequisites_ready_no_warning(tmp_path):
    """Phase 3 has no warning when prerequisites.json says ready."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    prereq = {"status": "ready", "dataset": {}, "environment": {}, "ready_for_baseline": True}
    (results_dir / "prerequisites.json").write_text(json.dumps(prereq))
    result = validate_phase_requirements(3, str(tmp_path))
    assert result["valid"] is True
    assert result["warnings"] == []


def test_validate_phase3_prerequisites_corrupt_json_warns(tmp_path):
    """Phase 3 warns when prerequisites.json is corrupt JSON."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "prerequisites.json").write_text("{bad json")
    result = validate_phase_requirements(3, str(tmp_path))
    assert result["valid"] is True
    assert any("prerequisites.json" in w for w in result["warnings"])


def test_validate_phase3_no_prerequisites_file_ok(tmp_path):
    """Phase 3 is fine without prerequisites.json (backward compat)."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    result = validate_phase_requirements(3, str(tmp_path))
    assert result["valid"] is True
    assert result["warnings"] == []


# --- Phase 4: Checkpoint (baseline.json with metrics+config) ---


def test_validate_phase4_valid(tmp_path):
    """Phase 4 passes when baseline.json has metrics and config."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    baseline = {"metrics": {"loss": 0.5}, "config": {"lr": 0.001}}
    (results_dir / "baseline.json").write_text(json.dumps(baseline))

    result = validate_phase_requirements(4, str(tmp_path))
    assert result["valid"] is True
    assert result["missing"] == []


def test_validate_phase4_missing_keys(tmp_path):
    """Phase 4 fails when baseline.json is missing required keys."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    baseline = {"metrics": {"loss": 0.5}}  # missing 'config'
    (results_dir / "baseline.json").write_text(json.dumps(baseline))

    result = validate_phase_requirements(4, str(tmp_path))
    assert result["valid"] is False
    assert any("config" in m for m in result["missing"])


def test_validate_phase4_corrupt_json(tmp_path):
    """Phase 4 fails when baseline.json is corrupt JSON."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "baseline.json").write_text("{bad json")
    result = validate_phase_requirements(4, str(tmp_path))
    assert result["valid"] is False
    assert any("not valid JSON" in m for m in result["missing"])


def test_validate_phase4_missing_baseline_file(tmp_path):
    """Phase 4 fails when baseline.json does not exist."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    result = validate_phase_requirements(4, str(tmp_path))
    assert result["valid"] is False
    assert any("baseline.json" in m for m in result["missing"])


def test_validate_phase4_missing_metrics_key(tmp_path):
    """Phase 4 fails when baseline.json has config but not metrics."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "baseline.json").write_text(json.dumps({"config": {"lr": 0.001}}))
    result = validate_phase_requirements(4, str(tmp_path))
    assert result["valid"] is False
    assert any("metrics" in m for m in result["missing"])


# --- Phase 5: Research (baseline.json exists) ---


def test_validate_phase5_valid(tmp_path):
    """Phase 5 passes when baseline.json exists."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "baseline.json").write_text(json.dumps({"metrics": {}, "config": {}}))

    result = validate_phase_requirements(5, str(tmp_path))
    assert result["valid"] is True
    assert result["missing"] == []


def test_validate_phase5_missing_baseline(tmp_path):
    """Phase 5 fails when baseline.json does not exist."""
    (tmp_path / "results").mkdir()
    result = validate_phase_requirements(5, str(tmp_path))
    assert result["valid"] is False
    assert any("baseline.json" in m for m in result["missing"])


# --- Phase 6: Experiment loop (baseline valid + manifest check) ---


def test_validate_phase6_valid(tmp_path):
    """Phase 6 validates when baseline.json has proper schema."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    baseline = {
        "metrics": {"loss": 0.5, "accuracy": 85.0},
        "config": {"lr": 0.001, "batch_size": 32},
    }
    (results_dir / "baseline.json").write_text(json.dumps(baseline))

    result = validate_phase_requirements(6, str(tmp_path))
    assert result["valid"] is True
    assert result["phase"] == 6
    assert result["missing"] == []
    assert result["warnings"] == []


def test_validate_phase6_missing_baseline(tmp_path):
    """Phase 6 fails when baseline.json does not exist."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    result = validate_phase_requirements(6, str(tmp_path))
    assert result["valid"] is False
    assert result["phase"] == 6
    assert len(result["missing"]) > 0
    assert any("baseline.json" in m for m in result["missing"])


def test_validate_phase6_invalid_manifest(tmp_path):
    """Phase 6 warns when implementation-manifest.json lacks 'proposals' key."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    baseline = {
        "metrics": {"loss": 0.5},
        "config": {"lr": 0.001},
    }
    (results_dir / "baseline.json").write_text(json.dumps(baseline))
    manifest = {"items": ["something"]}
    (results_dir / "implementation-manifest.json").write_text(json.dumps(manifest))

    result = validate_phase_requirements(6, str(tmp_path))
    assert result["valid"] is True
    assert result["phase"] == 6
    assert result["missing"] == []
    assert len(result["warnings"]) > 0
    assert any("proposals" in w for w in result["warnings"])


def test_validate_phase6_corrupt_json(tmp_path):
    """Phase 6 fails when baseline.json is corrupt JSON."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "baseline.json").write_text("{bad")
    result = validate_phase_requirements(6, str(tmp_path))
    assert result["valid"] is False
    assert any("not valid JSON" in m for m in result["missing"])


def test_validate_phase6_missing_metrics_key(tmp_path):
    """Phase 6 fails when baseline.json is missing metrics."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "baseline.json").write_text(json.dumps({"config": {"lr": 0.001}}))
    result = validate_phase_requirements(6, str(tmp_path))
    assert result["valid"] is False
    assert any("metrics" in m for m in result["missing"])


def test_validate_phase6_missing_config_key(tmp_path):
    """Phase 6 fails when baseline.json is missing config."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "baseline.json").write_text(json.dumps({"metrics": {"loss": 0.5}}))
    result = validate_phase_requirements(6, str(tmp_path))
    assert result["valid"] is False
    assert any("config" in m for m in result["missing"])


def test_validate_phase6_corrupt_manifest(tmp_path):
    """Phase 6 warns when manifest is corrupt JSON."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "baseline.json").write_text(json.dumps({"metrics": {"loss": 0.5}, "config": {"lr": 0.001}}))
    (results_dir / "implementation-manifest.json").write_text("{bad")
    result = validate_phase_requirements(6, str(tmp_path))
    assert result["valid"] is True
    assert any("not valid JSON" in w for w in result["warnings"])


def test_validate_undefined_phase(tmp_path):
    """Undefined phase returns valid=True with a warning."""
    result = validate_phase_requirements(99, str(tmp_path))
    assert result["valid"] is True
    assert any("No validation rules" in w for w in result["warnings"])


# --- save_state / load_state ---


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
    save_state(6, 1, ["exp-001"], str(tmp_path), user_choices=choices)
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


def test_load_state_corrupt_json(tmp_path):
    """Loading corrupt JSON returns None."""
    (tmp_path / "pipeline-state.json").write_text("{invalid json")
    state = load_state(str(tmp_path))
    assert state is None


def test_load_state_no_file(tmp_path):
    """Loading state when no file exists returns None."""
    assert load_state(str(tmp_path)) is None


# --- cleanup_stale ---


def test_cleanup_stale_skips_recent(tmp_path):
    """A recently-updated running state should NOT be cleaned up."""
    recent_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    state = {
        "phase": 6,
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
        "phase": 5,
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


def test_cleanup_stale_corrupt_state_json(tmp_path):
    """Corrupt pipeline-state.json is handled gracefully."""
    (tmp_path / "pipeline-state.json").write_text("{bad")
    cleaned = cleanup_stale(str(tmp_path))
    assert cleaned == []


def test_cleanup_stale_naive_timestamp(tmp_path):
    """Naive timestamp (no tzinfo) is treated as UTC."""
    naive_time = (datetime.now(timezone.utc) - timedelta(hours=3)).replace(tzinfo=None)
    state = {
        "phase": 6,
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
        "phase": 6,
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
        "phase": 6,
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


def test_cleanup_stale_exp_files_naive_timestamp(tmp_path):
    """Timezone-naive timestamps in exp-*.json files are treated as UTC."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    naive_time = (datetime.now(timezone.utc) - timedelta(hours=3)).replace(tzinfo=None)
    exp = {"status": "running", "timestamp": naive_time.isoformat(), "exp_id": "exp-002"}
    (results_dir / "exp-002.json").write_text(json.dumps(exp))
    cleaned = cleanup_stale(str(tmp_path), timeout_hours=2.0)
    assert any("exp-002" in c for c in cleaned)
    data = json.loads((results_dir / "exp-002.json").read_text())
    assert data["status"] == "failed"


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
    r = run_main("pipeline_state.py", str(tmp_path), "validate", "4")
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
    r = run_main("pipeline_state.py", str(tmp_path), "save", "6", "not_int")
    assert r.returncode == 1
    assert "Error" in r.stdout
    assert "iteration" in r.stdout.lower()


def test_cli_unknown_action(run_main, tmp_path):
    """CLI with unknown action exits 1."""
    r = run_main("pipeline_state.py", str(tmp_path), "bogus")
    assert r.returncode == 1
    assert "unknown" in r.stdout.lower()


# --- Phase 3: prerequisites.json blocks when ready_for_baseline=false ---


def test_validate_phase3_blocks_on_failed_prerequisites(tmp_path):
    """Phase 3 should BLOCK (not just warn) when prerequisites say not ready."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    prereq = {"status": "failed", "dataset": {}, "environment": {}, "ready_for_baseline": False}
    (results_dir / "prerequisites.json").write_text(json.dumps(prereq))
    result = validate_phase_requirements(3, str(tmp_path))
    assert result["valid"] is False
    assert any("prerequisites" in m.lower() or "ready_for_baseline" in m.lower() for m in result["missing"])


# --- User choices backup and recovery ---


def test_save_state_creates_user_choices_backup(tmp_path):
    """save_state writes a separate user-choices-backup.json file."""
    choices = {"primary_metric": "accuracy", "lower_is_better": False}
    save_state(6, 1, [], str(tmp_path), user_choices=choices)
    backup_path = tmp_path / "user-choices-backup.json"
    assert backup_path.is_file()
    backup = json.loads(backup_path.read_text())
    assert backup == choices


def test_save_state_no_backup_without_user_choices(tmp_path):
    """save_state without user_choices does not create a backup file."""
    save_state(3, 1, [], str(tmp_path))
    backup_path = tmp_path / "user-choices-backup.json"
    assert not backup_path.is_file()


def test_load_state_recovers_from_corrupt_state_with_backup(tmp_path):
    """If pipeline-state.json is corrupt but backup exists, recover user_choices."""
    choices = {"primary_metric": "loss", "lower_is_better": True, "target_value": 0.01}
    # First save normally to create the backup
    save_state(6, 2, [], str(tmp_path), user_choices=choices)
    # Now corrupt the main state file
    (tmp_path / "pipeline-state.json").write_text("{corrupt json!!!")
    state = load_state(str(tmp_path))
    assert state is not None
    assert state["status"] == "recovered"
    assert state["user_choices"] == choices


def test_load_state_corrupt_no_backup_returns_none(tmp_path):
    """If both state and backup are missing/corrupt, returns None."""
    (tmp_path / "pipeline-state.json").write_text("{bad}")
    state = load_state(str(tmp_path))
    assert state is None


# --- Full user_choices roundtrip (Task 3.2) ---


class TestFullUserChoicesRoundtrip:
    """All 20+ documented user_choices fields survive save/load (Task 3.2)."""

    def test_all_user_choices_fields_persist(self, tmp_path):
        all_choices = {
            "primary_metric": "accuracy",
            "divergence_metric": "loss",
            "divergence_lower_is_better": True,
            "lower_is_better": False,
            "target_value": 0.95,
            "train_command": "python train.py",
            "eval_command": "python eval.py",
            "train_data_path": "/data/train.csv",
            "val_data_path": "/data/val.csv",
            "prepared_train_path": "/data/prepared/train.csv",
            "prepared_val_path": "/data/prepared/val.csv",
            "env_manager": "conda",
            "env_name": "ml-env",
            "model_category": "supervised",
            "user_papers": ["paper1.pdf", "paper2.pdf"],
            "budget_mode": "autonomous",
            "difficulty": "moderate",
            "difficulty_multiplier": 15,
            "method_proposal_scope": "training",
            "method_proposal_iterations": 3,
            "hp_batches_per_round": 3,
        }

        save_state(4, 1, [], str(tmp_path), user_choices=all_choices)
        loaded = load_state(str(tmp_path))

        assert loaded is not None
        loaded_choices = loaded.get("user_choices", {})

        for key, value in all_choices.items():
            assert key in loaded_choices, f"Missing key: {key}"
            assert loaded_choices[key] == value, f"Mismatch for {key}: {loaded_choices[key]} != {value}"


# --- Coverage tests for save_state / load_state edge cases ---


def test_save_state_write_failure(tmp_path):
    """save_state should propagate exception and clean up temp on write failure."""
    exp_root = str(tmp_path / "exp")
    (tmp_path / "exp").mkdir()
    with pytest.raises(OSError):
        with mock.patch("pipeline_state.os.fdopen", side_effect=OSError("disk full")):
            save_state(phase=1, iteration=0, running_exp_ids=[], exp_root=exp_root)
    # Verify no temp files left behind
    assert not list((tmp_path / "exp").glob("*.tmp"))


def test_save_state_backup_write_failure(tmp_path):
    """Main save succeeds even when user_choices backup fails."""
    exp_root = str(tmp_path / "exp")
    original_mkstemp = tempfile.mkstemp
    call_count = [0]

    def mock_mkstemp(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 2:  # Second call is the backup
            raise OSError("backup failed")
        return original_mkstemp(*args, **kwargs)

    with mock.patch("pipeline_state.tempfile.mkstemp", side_effect=mock_mkstemp):
        path = save_state(phase=1, iteration=0, running_exp_ids=[],
                          exp_root=exp_root, user_choices={"metric": "loss"})
    assert Path(path).exists()
    state = json.loads(Path(path).read_text())
    assert state["user_choices"]["metric"] == "loss"


def test_load_state_both_corrupt(tmp_path):
    """Returns None when both main state and backup are corrupt."""
    exp_root = tmp_path / "exp"
    exp_root.mkdir()
    (exp_root / "pipeline-state.json").write_text("NOT JSON")
    (exp_root / "user-choices-backup.json").write_text("ALSO NOT JSON")
    result = load_state(str(exp_root))
    assert result is None


def test_cli_save_invalid_json_running_ids(run_main, tmp_path):
    """CLI save with invalid JSON for running_ids should error."""
    r = run_main("pipeline_state.py", str(tmp_path), "save", "1", "0", "not_json")
    assert r.returncode == 1
    assert "invalid" in r.stdout.lower() or "error" in r.stdout.lower()


def test_cli_load_no_state(run_main, tmp_path):
    """CLI load with no state should print 'No pipeline state found.'"""
    r = run_main("pipeline_state.py", str(tmp_path), "load")
    assert r.returncode == 0
    assert "no pipeline state" in r.stdout.lower()


def test_cli_cleanup_nothing(run_main, tmp_path):
    """CLI cleanup with nothing stale should report 'Nothing to clean up.'"""
    r = run_main("pipeline_state.py", str(tmp_path), "cleanup")
    assert r.returncode == 0
    assert "nothing to clean" in r.stdout.lower()


def test_cleanup_stale_pipeline_write_failure(tmp_path):
    """cleanup_stale re-raises after cleaning temp file on write failure (lines 235-237)."""
    exp_root = tmp_path / "exp"
    exp_root.mkdir()
    stale_time = (datetime.now(tz=timezone.utc) - timedelta(hours=10)).isoformat()
    state = {
        "phase": 6, "iteration": 1,
        "status": "running",
        "running_exp_ids": ["exp-old"],
        "timestamp": stale_time,
    }
    (exp_root / "pipeline-state.json").write_text(json.dumps(state))
    original_fdopen = _os.fdopen

    def mock_fdopen(fd, *args, **kwargs):
        _os.close(fd)  # Close fd to prevent leak
        raise OSError("disk full during cleanup")

    with mock.patch("pipeline_state.os.fdopen", side_effect=mock_fdopen):
        with pytest.raises(OSError, match="disk full"):
            cleanup_stale(str(exp_root))


def test_cleanup_stale_exp_result_write_failure(tmp_path):
    """cleanup_stale re-raises when marking stale exp-result fails (lines 268-270)."""
    exp_root = tmp_path / "exp"
    results = exp_root / "results"
    results.mkdir(parents=True)
    stale_time = (datetime.now(tz=timezone.utc) - timedelta(hours=10)).isoformat()
    (results / "exp-001.json").write_text(json.dumps({
        "exp_id": "exp-001", "status": "running", "timestamp": stale_time,
        "config": {}, "metrics": {},
    }))
    # No pipeline-state.json so skip that path — go directly to exp results
    def mock_fdopen(fd, *args, **kwargs):
        _os.close(fd)
        raise OSError("disk full during exp rewrite")

    with mock.patch("pipeline_state.os.fdopen", side_effect=mock_fdopen):
        with pytest.raises(OSError, match="disk full"):
            cleanup_stale(str(exp_root))


# --- Stacking state persistence ---


def test_save_and_load_stacking_state(tmp_path):
    """Stacking state is preserved through save/load cycle."""
    stacking = {
        "ranked_methods": ["perceptual-loss", "cosine-scheduler", "mixup"],
        "current_stack_order": 2,
        "stack_base_branch": "ml-opt/stack-2",
        "stack_base_exp": "exp-015",
        "skipped_methods": ["mixup"],
        "stacked_methods": ["perceptual-loss", "cosine-scheduler"],
    }
    save_state(6, 5, [], str(tmp_path), user_choices={"stacking": stacking})
    loaded = load_state(str(tmp_path))
    assert loaded is not None
    assert loaded["user_choices"]["stacking"] == stacking
