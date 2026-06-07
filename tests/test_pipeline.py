"""Consolidated tests for pipeline_state.py and experiment_setup.py.

Covers phase validation, state persistence, baseline checksum integrity,
stale experiment cleanup, experiment setup, and CLI interfaces.
"""

import concurrent.futures
import json
import os
import stat
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

from pipeline_state import (
    PHASE_TRANSITIONS,
    RECOVERY_INSTRUCTIONS,
    _RECOVERY_DEFAULT,
    _compute_baseline_checksum,
    cleanup_stale,
    get_decisions,
    get_recovery_instruction,
    load_state,
    log_decision,
    log_phase_gate,
    replay_check,
    save_state,
    validate_phase_gate,
    validate_phase_requirements,
    verify_baseline_integrity,
)
from experiment_setup import generate_script, next_experiment_id
from conftest import (
    cleanup_stale_experiments,
    create_experiment_dirs,
    setup_experiment,
    write_experiment_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_results_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir(exist_ok=True)
    return d


def _write_baseline(tmp_path, metrics=None, config=None):
    d = _make_results_dir(tmp_path)
    data = {}
    if metrics is not None:
        data["metrics"] = metrics
    if config is not None:
        data["config"] = config
    (d / "baseline.json").write_text(json.dumps(data))
    return d


def _write_prerequisites(tmp_path, ready, status="ready"):
    d = _make_results_dir(tmp_path)
    (d / "prerequisites.json").write_text(json.dumps(
        {"status": status, "dataset": {}, "environment": {},
         "ready_for_baseline": ready}))
    return d


# ---------------------------------------------------------------------------
# TestPhaseValidation
# ---------------------------------------------------------------------------

class TestPhaseValidation:
    """Phase requirement validation (pipeline_state.validate_phase_requirements)."""

    @pytest.mark.parametrize("phase", [5, 6, 7, 9])
    def test_phases_requiring_baseline_fail_without_it(self, tmp_path, phase):
        """Phases 5/6/7/9 all fail when baseline.json is absent."""
        _make_results_dir(tmp_path)
        result = validate_phase_requirements(phase, str(tmp_path))
        assert result["valid"] is False
        assert any("baseline.json" in m for m in result["missing"])

    @pytest.mark.parametrize("phase", [5, 6, 7, 9])
    def test_phases_requiring_baseline_pass_with_it(self, tmp_path, phase):
        """Phases 5/6/7/9 pass with a valid baseline (+ goals for Phase 7)."""
        _write_baseline(tmp_path, {"loss": 0.5}, {"lr": 0.001})
        if phase == 7:
            # Phase 7 also requires optimization-goals.json
            goals_path = tmp_path / "optimization-goals.json"
            goals_path.write_text('{"objective": {"primary_metric": "loss"}}')
        result = validate_phase_requirements(phase, str(tmp_path))
        assert result["valid"] is True

    def test_phase2_always_valid(self, tmp_path):
        result = validate_phase_requirements(2, str(tmp_path))
        assert result["valid"] is True and result["missing"] == []

    def test_phase3_results_dir_required(self, tmp_path):
        """Phase 3 fails without results/, passes with it."""
        assert validate_phase_requirements(3, str(tmp_path))["valid"] is False
        _make_results_dir(tmp_path)
        r = validate_phase_requirements(3, str(tmp_path))
        assert r["valid"] is True and r["warnings"] == []

    @pytest.mark.parametrize("ready, expect_valid", [(True, True), (False, False)])
    def test_phase3_prerequisites_gating(self, tmp_path, ready, expect_valid):
        _write_prerequisites(tmp_path, ready, status="ready" if ready else "failed")
        assert validate_phase_requirements(3, str(tmp_path))["valid"] is expect_valid

    def test_phase3_prereq_corrupt_warns(self, tmp_path):
        d = _make_results_dir(tmp_path)
        (d / "prerequisites.json").write_text("{bad json")
        r = validate_phase_requirements(3, str(tmp_path))
        assert r["valid"] is True
        assert any("prerequisites.json" in w for w in r["warnings"])

    @pytest.mark.parametrize("data, missing_key", [
        ({"metrics": {"loss": 0.5}}, "config"),
        ({"config": {"lr": 0.001}}, "metrics"),
    ])
    def test_phase4_missing_keys(self, tmp_path, data, missing_key):
        d = _make_results_dir(tmp_path)
        (d / "baseline.json").write_text(json.dumps(data))
        r = validate_phase_requirements(4, str(tmp_path))
        assert r["valid"] is False
        assert any(missing_key in m for m in r["missing"])

    def test_phase4_corrupt_and_absent(self, tmp_path):
        """Corrupt baseline.json fails; absent baseline.json also fails."""
        d = _make_results_dir(tmp_path)
        (d / "baseline.json").write_text("{bad")
        r = validate_phase_requirements(4, str(tmp_path))
        assert r["valid"] is False and any("not valid JSON" in m for m in r["missing"])
        (d / "baseline.json").unlink()
        r = validate_phase_requirements(4, str(tmp_path))
        assert r["valid"] is False and any("baseline.json" in m for m in r["missing"])

    def test_phase6_manifest_warnings(self, tmp_path):
        """Phase 6 warns on invalid/corrupt manifest but still valid."""
        _write_baseline(tmp_path, {"loss": 0.5}, {"lr": 0.001})
        d = tmp_path / "results"
        # Invalid manifest (no 'proposals' key)
        (d / "implementation-manifest.json").write_text(json.dumps({"items": []}))
        r = validate_phase_requirements(6, str(tmp_path))
        assert r["valid"] is True and any("proposals" in w for w in r["warnings"])
        # Corrupt manifest
        (d / "implementation-manifest.json").write_text("{bad")
        r = validate_phase_requirements(6, str(tmp_path))
        assert r["valid"] is True and any("not valid JSON" in w for w in r["warnings"])

    def test_phase8_requires_manifest(self, tmp_path):
        _write_baseline(tmp_path, {"loss": 1.0}, {"lr": 0.001})
        r = validate_phase_requirements(8, str(tmp_path))
        assert r["valid"] is False
        assert any("implementation-manifest" in m for m in r["missing"])

    def test_undefined_phase_warns(self, tmp_path):
        r = validate_phase_requirements(99, str(tmp_path))
        assert r["valid"] is True
        assert any("No validation rules" in w for w in r["warnings"])


# ---------------------------------------------------------------------------
# TestStatePersistence
# ---------------------------------------------------------------------------

class TestStatePersistence:
    """save_state / load_state roundtrips and edge cases."""

    def test_save_and_load_roundtrip(self, tmp_path):
        exp_ids = ["exp-001", "exp-002"]
        path = save_state(3, 2, exp_ids, str(tmp_path))
        assert Path(path).exists()
        state = load_state(str(tmp_path))
        assert state["phase"] == 3 and state["iteration"] == 2
        assert state["running_experiments"] == exp_ids
        assert state["status"] == "running"
        datetime.fromisoformat(state["timestamp"])

    def test_user_choices_full_roundtrip(self, tmp_path):
        """All documented user_choices fields survive save/load."""
        all_choices = {
            "primary_metric": "accuracy", "divergence_metric": "loss",
            "divergence_lower_is_better": True, "lower_is_better": False,
            "target_value": 0.95, "train_command": "python train.py",
            "eval_command": "python eval.py", "train_data_path": "/data/train.csv",
            "val_data_path": "/data/val.csv",
            "prepared_train_path": "/data/prepared/train.csv",
            "prepared_val_path": "/data/prepared/val.csv",
            "env_manager": "conda", "env_name": "ml-env",
            "model_category": "supervised",
            "user_papers": ["paper1.pdf", "paper2.pdf"],
            "method_proposal_scope": "training",
            "method_proposal_iterations": 3, "hp_batches_per_round": 3,
        }
        save_state(4, 1, [], str(tmp_path), user_choices=all_choices)
        loaded = load_state(str(tmp_path))["user_choices"]
        for key, value in all_choices.items():
            assert loaded[key] == value, f"Mismatch for {key}"

    def test_user_choices_lifecycle(self, tmp_path):
        """Absent when never set; preserved across saves; cleared with {}."""
        save_state(3, 1, [], str(tmp_path))
        assert "user_choices" not in load_state(str(tmp_path))
        save_state(0, 0, [], str(tmp_path), user_choices={"primary_metric": "acc"})
        save_state(3, 1, [], str(tmp_path))  # no user_choices arg
        assert load_state(str(tmp_path))["user_choices"]["primary_metric"] == "acc"
        save_state(3, 2, [], str(tmp_path), user_choices={})
        assert load_state(str(tmp_path)).get("user_choices") == {}

    @pytest.mark.parametrize("field, value, reset_val", [
        ("consecutive_stop_count", 2, 0),
        ("stuck_protocol_triggered", True, False),
    ])
    def test_root_field_persist_preserve_reset(self, tmp_path, field, value, reset_val):
        save_state(7, 3, [], str(tmp_path), **{field: value})
        assert load_state(str(tmp_path))[field] == value
        save_state(7, 4, [], str(tmp_path))  # preserved
        assert load_state(str(tmp_path))[field] == value
        save_state(7, 5, [], str(tmp_path), **{field: reset_val})
        assert load_state(str(tmp_path))[field] == reset_val

    def test_stuck_protocol_absent_when_never_set(self, tmp_path):
        save_state(7, 1, [], str(tmp_path))
        assert "stuck_protocol_triggered" not in load_state(str(tmp_path))

    def test_agent_registry_persist_preserve_reset(self, tmp_path):
        """Agent registry roundtrips, preserves across saves, and can be cleared."""
        registry = {"research": "agent-abc123", "tuning": "agent-def456"}
        save_state(7, 1, [], str(tmp_path), agent_registry=registry)
        assert load_state(str(tmp_path))["agent_registry"] == registry
        # Preserved when not explicitly provided
        save_state(7, 2, [], str(tmp_path))
        assert load_state(str(tmp_path))["agent_registry"] == registry
        # Updated with new entries
        updated = {**registry, "analysis": "agent-ghi789"}
        save_state(7, 3, [], str(tmp_path), agent_registry=updated)
        assert load_state(str(tmp_path))["agent_registry"] == updated
        # Cleared with empty dict
        save_state(7, 4, [], str(tmp_path), agent_registry={})
        assert load_state(str(tmp_path))["agent_registry"] == {}

    def test_agent_registry_absent_when_never_set(self, tmp_path):
        save_state(7, 1, [], str(tmp_path))
        assert "agent_registry" not in load_state(str(tmp_path))

    def test_agent_registry_and_user_choices_simultaneous(self, tmp_path):
        """Both agent_registry and user_choices survive when saved together."""
        registry = {"research": "agent-abc", "tuning": "agent-def"}
        choices = {"primary_metric": "accuracy", "lower_is_better": False}
        save_state(7, 1, [], str(tmp_path),
                   agent_registry=registry, user_choices=choices)
        state = load_state(str(tmp_path))
        assert state["agent_registry"] == registry
        assert state["user_choices"] == choices

    def test_agent_registry_preserved_when_user_choices_updated(self, tmp_path):
        """Updating user_choices alone doesn't clobber agent_registry."""
        save_state(7, 1, [], str(tmp_path),
                   agent_registry={"research": "agent-abc"})
        save_state(7, 2, [], str(tmp_path),
                   user_choices={"primary_metric": "loss"})
        state = load_state(str(tmp_path))
        assert state["agent_registry"] == {"research": "agent-abc"}
        assert state["user_choices"]["primary_metric"] == "loss"

    def test_agent_registry_clearing_simulates_resumption(self, tmp_path):
        """On pipeline resumption (new session), registry must be clearable
        while preserving user_choices — simulates the orchestrate skill's
        session-start clearing logic."""
        full_registry = {
            "research": "agent-old-1", "implement": "agent-old-2",
            "tuning": "agent-old-3", "analysis": "agent-old-4",
            "monitor": "agent-old-5",
        }
        choices = {"primary_metric": "accuracy"}
        save_state(7, 5, [], str(tmp_path),
                   agent_registry=full_registry, user_choices=choices)
        # Simulate new session: load state, clear registry, re-save
        state = load_state(str(tmp_path))
        assert state["agent_registry"] == full_registry
        save_state(state["phase"], state["iteration"], [], str(tmp_path),
                   agent_registry={})
        state = load_state(str(tmp_path))
        assert state["agent_registry"] == {}
        assert state["user_choices"] == choices  # preserved

    def test_stacking_state_roundtrip(self, tmp_path):
        stacking = {
            "ranked_methods": ["perceptual-loss", "cosine-scheduler", "mixup"],
            "current_stack_order": 2, "stack_base_branch": "ml-opt/stack-2",
            "stack_base_exp": "exp-015", "skipped_methods": ["mixup"],
            "stacked_methods": ["perceptual-loss", "cosine-scheduler"],
        }
        save_state(6, 5, [], str(tmp_path), user_choices={"stacking": stacking})
        assert load_state(str(tmp_path))["user_choices"]["stacking"] == stacking

    def test_load_returns_none_on_corrupt_or_missing(self, tmp_path):
        """Returns None when file is missing, corrupt, or both state+backup corrupt."""
        assert load_state(str(tmp_path)) is None  # missing
        (tmp_path / "pipeline-state.json").write_text("{invalid")
        assert load_state(str(tmp_path)) is None  # corrupt, no backup
        (tmp_path / "user-choices-backup.json").write_text("ALSO BAD")
        assert load_state(str(tmp_path)) is None  # both corrupt

    def test_backup_lifecycle(self, tmp_path):
        """Backup created with user_choices; absent without; recovery works."""
        save_state(3, 1, [], str(tmp_path))
        assert not (tmp_path / "user-choices-backup.json").is_file()
        choices = {"primary_metric": "loss", "lower_is_better": True}
        save_state(6, 2, [], str(tmp_path), user_choices=choices)
        assert json.loads((tmp_path / "user-choices-backup.json").read_text()) == choices
        (tmp_path / "pipeline-state.json").write_text("{corrupt!!!")
        state = load_state(str(tmp_path))
        assert state["status"] == "recovered" and state["user_choices"] == choices

    def test_save_state_write_failure(self, tmp_path):
        exp_root = tmp_path / "exp"
        exp_root.mkdir()
        with pytest.raises(OSError):
            with mock.patch("pipeline_state.os.fdopen", side_effect=OSError("disk full")):
                save_state(1, 0, [], str(exp_root))
        assert not list(exp_root.glob("*.tmp"))

    def test_backup_write_failure_main_still_succeeds(self, tmp_path):
        exp_root = tmp_path / "exp"
        original_mkstemp = tempfile.mkstemp
        call_count = [0]
        def mock_mkstemp(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise OSError("backup failed")
            return original_mkstemp(*args, **kwargs)
        with mock.patch("pipeline_state.tempfile.mkstemp", side_effect=mock_mkstemp):
            path = save_state(1, 0, [], str(exp_root), user_choices={"metric": "loss"})
        assert json.loads(Path(path).read_text())["user_choices"]["metric"] == "loss"


# ---------------------------------------------------------------------------
# TestBaselineChecksum
# ---------------------------------------------------------------------------

class TestBaselineChecksum:
    """Baseline checksum computation and integrity verification."""

    @pytest.mark.parametrize("ma, mb, eq", [
        ({"loss": 0.5}, {"loss": 0.5}, True),
        ({"accuracy": 85.0, "loss": 0.5}, {"loss": 0.5, "accuracy": 85.0}, True),
        ({"loss": 0.5}, {"loss": 0.6}, False),
    ], ids=["deterministic", "key_order_independent", "different_values"])
    def test_checksum_properties(self, ma, mb, eq):
        assert (_compute_baseline_checksum(ma) == _compute_baseline_checksum(mb)) is eq

    def test_checksum_format_and_persistence(self, tmp_path):
        """SHA-256 format; persists in state; survives across saves."""
        c = _compute_baseline_checksum({"loss": 0.5})
        assert len(c) == 64 and all(ch in "0123456789abcdef" for ch in c)
        save_state(3, 0, [], str(tmp_path), baseline_checksum="abc123")
        assert load_state(str(tmp_path))["baseline_checksum"] == "abc123"
        save_state(7, 1, [], str(tmp_path))
        assert load_state(str(tmp_path))["baseline_checksum"] == "abc123"

    def test_verify_integrity_match_and_mismatch(self, tmp_path):
        """Verify passes with correct checksum; fails on tampered metrics."""
        metrics = {"loss": 0.5, "accuracy": 85.0}
        checksum = _compute_baseline_checksum(metrics)
        save_state(3, 0, [], str(tmp_path), baseline_checksum=checksum)
        d = _make_results_dir(tmp_path)
        (d / "baseline.json").write_text(json.dumps(
            {"exp_id": "baseline", "config": {}, "metrics": metrics}))
        assert verify_baseline_integrity(str(tmp_path))["valid"] is True
        # Tamper
        (d / "baseline.json").write_text(json.dumps(
            {"exp_id": "baseline", "config": {}, "metrics": {"loss": 0.3}}))
        r = verify_baseline_integrity(str(tmp_path))
        assert r["valid"] is False and "mismatch" in r["error"].lower()

    @pytest.mark.parametrize("setup_fn, expect_valid", [
        (lambda p: None, True),
        (lambda p: save_state(3, 0, [], str(p)), True),
        (lambda p: save_state(3, 0, [], str(p), baseline_checksum="abc"), False),
        (lambda p: (save_state(3, 0, [], str(p), baseline_checksum="abc"),
                    (p / "results").mkdir(exist_ok=True),
                    (p / "results" / "baseline.json").write_text("{bad")),
         False),
    ], ids=["no_state", "legacy_no_checksum", "missing_file", "corrupt_json"])
    def test_verify_edge_cases(self, tmp_path, setup_fn, expect_valid):
        setup_fn(tmp_path)
        result = verify_baseline_integrity(str(tmp_path))
        assert result["valid"] is expect_valid
        if expect_valid:
            assert result.get("warning") is not None


# ---------------------------------------------------------------------------
# TestCleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    """Stale experiment cleanup (pipeline_state.cleanup_stale and experiment_setup)."""

    def test_cleanup_stale_recent_vs_old(self, tmp_path):
        """Recent state untouched; stale state marked interrupted."""
        recent = datetime.now(timezone.utc) - timedelta(minutes=5)
        state = {"phase": 6, "iteration": 1, "running_experiments": ["exp-001"],
                 "timestamp": recent.isoformat(), "status": "running"}
        (tmp_path / "pipeline-state.json").write_text(json.dumps(state))
        assert cleanup_stale(str(tmp_path), timeout_hours=2.0) == []
        assert json.loads((tmp_path / "pipeline-state.json").read_text())["status"] == "running"
        # Now stale
        stale = datetime.now(timezone.utc) - timedelta(hours=3)
        state["timestamp"] = stale.isoformat()
        (tmp_path / "pipeline-state.json").write_text(json.dumps(state))
        cleaned = cleanup_stale(str(tmp_path), timeout_hours=2.0)
        assert any("interrupted" in c for c in cleaned)
        updated = json.loads((tmp_path / "pipeline-state.json").read_text())
        assert updated["status"] == "interrupted" and "interrupted_at" in updated

    @pytest.mark.parametrize("body", [
        "{bad",
        json.dumps({"phase": 6, "iteration": 1, "running_experiments": [],
                     "status": "running", "timestamp": "not-a-date"}),
        json.dumps({"phase": 6, "iteration": 1, "running_experiments": [],
                     "status": "running"}),
    ], ids=["corrupt", "invalid_ts", "missing_ts"])
    def test_cleanup_stale_graceful_on_bad_state(self, tmp_path, body):
        (tmp_path / "pipeline-state.json").write_text(body)
        assert cleanup_stale(str(tmp_path), timeout_hours=2.0) == []

    def test_cleanup_stale_naive_timestamp(self, tmp_path):
        naive = (datetime.now(timezone.utc) - timedelta(hours=3)).replace(tzinfo=None)
        state = {"phase": 6, "iteration": 1, "running_experiments": [],
                 "timestamp": naive.isoformat(), "status": "running"}
        (tmp_path / "pipeline-state.json").write_text(json.dumps(state))
        assert len(cleanup_stale(str(tmp_path), timeout_hours=2.0)) > 0

    def test_cleanup_stale_exp_files(self, tmp_path):
        """Stale running exp marked failed; completed/bad-ts/corrupt skipped; naive-ts handled."""
        results = tmp_path / "results"
        results.mkdir()
        stale_ts = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        naive_ts = (datetime.now(timezone.utc) - timedelta(hours=3)).replace(tzinfo=None).isoformat()
        (results / "exp-001.json").write_text(json.dumps(
            {"status": "running", "timestamp": stale_ts, "exp_id": "exp-001"}))
        (results / "exp-002.json").write_text(json.dumps(
            {"status": "completed", "timestamp": stale_ts, "exp_id": "exp-002"}))
        (results / "exp-003.json").write_text(json.dumps(
            {"status": "running", "timestamp": "not-a-date", "exp_id": "exp-003"}))
        (results / "exp-004.json").write_text("{bad")
        (results / "exp-005.json").write_text(json.dumps(
            {"status": "running", "timestamp": naive_ts, "exp_id": "exp-005"}))
        cleaned = cleanup_stale(str(tmp_path), timeout_hours=2.0)
        cleaned_ids = " ".join(cleaned)
        assert "exp-001" in cleaned_ids and "exp-005" in cleaned_ids
        assert "exp-002" not in cleaned_ids
        assert json.loads((results / "exp-001.json").read_text())["status"] == "failed"
        assert json.loads((results / "exp-005.json").read_text())["status"] == "failed"
        assert json.loads((results / "exp-002.json").read_text())["status"] == "completed"

    @pytest.mark.parametrize("target", ["pipeline", "exp_result"])
    def test_cleanup_write_failure_reraises(self, tmp_path, target):
        exp = tmp_path / "exp"
        exp.mkdir()
        stale_ts = (datetime.now(tz=timezone.utc) - timedelta(hours=10)).isoformat()
        if target == "pipeline":
            (exp / "pipeline-state.json").write_text(json.dumps(
                {"phase": 6, "iteration": 1, "status": "running",
                 "running_exp_ids": ["exp-old"], "timestamp": stale_ts}))
        else:
            results = exp / "results"
            results.mkdir()
            (results / "exp-001.json").write_text(json.dumps({
                "exp_id": "exp-001", "status": "running", "timestamp": stale_ts,
                "config": {}, "metrics": {}}))
        def mock_fdopen(fd, *a, **kw):
            os.close(fd)
            raise OSError("disk full")
        with mock.patch("pipeline_state.os.fdopen", side_effect=mock_fdopen):
            with pytest.raises(OSError, match="disk full"):
                cleanup_stale(str(exp))

    def test_cleanup_stale_experiments(self, tmp_path):
        """Stale running/pending marked failed; fresh/completed untouched; nonexistent/corrupt safe."""
        assert cleanup_stale_experiments("/nonexistent/dir") == []
        stale_mtime = time.time() - 3 * 3600
        for eid, status in [("exp-001", "running"), ("exp-004", "pending")]:
            f = tmp_path / f"{eid}.json"
            f.write_text(json.dumps({"exp_id": eid, "status": status}))
            os.utime(str(f), (stale_mtime, stale_mtime))
        (tmp_path / "exp-002.json").write_text(
            json.dumps({"exp_id": "exp-002", "status": "running"}))
        done = tmp_path / "exp-003.json"
        done.write_text(json.dumps({"exp_id": "exp-003", "status": "completed"}))
        os.utime(str(done), (stale_mtime, stale_mtime))
        (tmp_path / "exp-005.json").write_text("{bad")
        os.utime(str(tmp_path / "exp-005.json"), (stale_mtime, stale_mtime))
        cleaned = cleanup_stale_experiments(str(tmp_path), timeout_hours=2.0)
        assert sorted(cleaned) == ["exp-001", "exp-004"]
        assert json.loads((tmp_path / "exp-002.json").read_text())["status"] == "running"
        assert json.loads(done.read_text())["status"] == "completed"


# ---------------------------------------------------------------------------
# TestExperimentSetup
# ---------------------------------------------------------------------------

class TestExperimentSetup:
    """Experiment ID generation, directory structure, config, and script generation."""

    def test_create_experiment_dirs_idempotent(self, tmp_path):
        for _ in range(2):
            exp_root = create_experiment_dirs(str(tmp_path))
        for subdir in ["logs", "reports", "scripts", "results", "artifacts"]:
            assert (Path(exp_root) / subdir).exists()
        dev_notes = Path(exp_root) / "dev_notes.md"
        assert dev_notes.exists() and "# Dev Notes" in dev_notes.read_text()

    @pytest.mark.parametrize("existing, expected", [
        ([], "exp-001"),
        (["exp-001.json", "exp-002.json"], "exp-003"),
        (["exp-001.json", "experiment-summary.json", "baseline.json"], "exp-002"),
    ], ids=["empty", "sequential", "ignores_non_exp"])
    def test_next_experiment_id(self, tmp_path, existing, expected):
        for f in existing:
            (tmp_path / f).write_text("{}")
        assert next_experiment_id(str(tmp_path)) == expected

    def test_next_experiment_id_nonexistent_dir(self):
        assert next_experiment_id("/nonexistent/dir") == "exp-001"

    def test_write_experiment_config(self, tmp_path):
        path = write_experiment_config(str(tmp_path), "exp-001", {"lr": 0.001})
        assert json.loads(Path(path).read_text())["lr"] == 0.001

    def test_generate_script_features(self, tmp_path):
        """GPU, command, env vars, log piping, executable, PID."""
        path = generate_script(
            str(tmp_path), "exp-001", "python train.py --lr 0.001",
            gpu_id=2, log_file="logs/round-1-hp/exp-001/train.log",
            env_vars={"WANDB_DISABLED": "true"})
        content = Path(path).read_text()
        for expected in ["CUDA_VISIBLE_DEVICES=2", "python train.py --lr 0.001",
                         "WANDB_DISABLED=true", "tee logs/round-1-hp/exp-001/train.log",
                         "pid", "$$"]:
            assert expected in content
        assert Path(path).stat().st_mode & stat.S_IXUSR

    def test_generate_script_requires_log_file(self, tmp_path):
        """Missing log_file raises ValueError — no stale flat-path fallback."""
        with pytest.raises(ValueError, match="log_file is required"):
            generate_script(str(tmp_path), "exp-002", "python train.py", gpu_id=0)

    def test_generate_script_path_with_spaces(self, tmp_path):
        path = generate_script(str(tmp_path), "exp-001", "python train.py",
                                     gpu_id=0, log_file="logs/my experiment/train.log")
        content = Path(path).read_text()
        assert "'logs/my experiment'" in content or "'logs/my experiment/train.log'" in content

    @pytest.mark.parametrize("budget, expect_timeout", [
        (120, True), (None, False), (0, False),
    ], ids=["with_budget", "none", "zero"])
    def test_generate_script_time_budget(self, tmp_path, budget, expect_timeout):
        path = generate_script(str(tmp_path), "exp-001", "python train.py",
                                     gpu_id=0, log_file="logs/round-1-hp/exp-001/train.log",
                                     time_budget=budget)
        content = Path(path).read_text()
        if expect_timeout:
            assert "timeout --signal=SIGTERM --kill-after=60 120" in content
        else:
            assert "timeout --signal=SIGTERM" not in content

    def test_setup_increments_ids(self, tmp_path):
        r1 = setup_experiment(str(tmp_path), "python train.py", gpu_id=0, config={"lr": 0.001})
        assert r1["exp_id"] == "exp-001"
        assert Path(r1["config_path"]).exists() and Path(r1["script_path"]).exists()
        assert setup_experiment(str(tmp_path), "python train.py", gpu_id=1)["exp_id"] == "exp-002"

    def test_concurrent_setup_unique_ids(self, tmp_path):
        def do_setup(i):
            return setup_experiment(str(tmp_path), f"python train.py --seed {i}", config={"seed": i})
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(do_setup, i) for i in range(8)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        ids = [r["exp_id"] for r in results]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------

class TestCLI:
    """CLI tests for pipeline_state.py and experiment_setup.py."""

    @pytest.mark.parametrize("script, args, rc, check", [
        ("pipeline_state.py", [], 1, "Usage"),
        ("pipeline_state.py", ["{tmp}", "bogus"], 1, "unknown"),
        ("pipeline_state.py", ["{tmp}", "validate", "abc"], 1, "phase"),
        ("pipeline_state.py", ["{tmp}", "save", "6", "not_int"], 1, "iteration"),
        ("pipeline_state.py", ["{tmp}", "save", "1", "0", "not_json"], 1, ""),
        ("experiment_setup.py", [], 1, "Usage"),
        ("experiment_setup.py", ["{tmp}", "echo hi", "abc"], 1, "gpu_id"),
        ("experiment_setup.py", ["{tmp}", "echo hi", "0", "{bad"], 1, "Error"),
    ], ids=["pipe_no_args", "pipe_unknown", "pipe_bad_phase", "pipe_bad_iter",
            "pipe_bad_json", "setup_no_args", "setup_bad_gpu", "setup_bad_config"])
    def test_cli_errors(self, run_main, tmp_path, script, args, rc, check):
        resolved = [str(tmp_path) if a == "{tmp}" else a for a in args]
        r = run_main(script, *resolved)
        assert r.returncode == rc
        if check:
            assert check.lower() in r.stdout.lower()

    def test_cli_pipeline_state_happy_path(self, run_main, tmp_path):
        """Validate, save, load, cleanup, verify-baseline all work."""
        _write_baseline(tmp_path, {"loss": 0.5}, {"lr": 0.001})
        r = run_main("pipeline_state.py", str(tmp_path), "validate", "4")
        assert r.returncode == 0 and json.loads(r.stdout)["valid"] is True
        r = run_main("pipeline_state.py", str(tmp_path), "save", "3", "1")
        assert r.returncode == 0 and "saved" in r.stdout.lower()
        r = run_main("pipeline_state.py", str(tmp_path), "load")
        assert json.loads(r.stdout)["phase"] == 3
        r = run_main("pipeline_state.py", str(tmp_path), "cleanup")
        assert r.returncode == 0

    def test_cli_load_missing(self, run_main, tmp_path):
        r = run_main("pipeline_state.py", str(tmp_path), "load")
        assert r.returncode == 0 and "no pipeline state" in r.stdout.lower()

    @pytest.mark.parametrize("checksum, expect_rc", [("__REAL__", 0), ("wrong", 1)])
    def test_cli_verify_baseline(self, run_main, tmp_path, checksum, expect_rc):
        metrics = {"loss": 0.5}
        cs = _compute_baseline_checksum(metrics) if checksum == "__REAL__" else checksum
        save_state(3, 0, [], str(tmp_path), baseline_checksum=cs)
        d = _make_results_dir(tmp_path)
        (d / "baseline.json").write_text(json.dumps({"metrics": metrics}))
        assert run_main("pipeline_state.py", str(tmp_path), "verify-baseline").returncode == expect_rc

    def test_cli_experiment_setup(self, run_main, tmp_path):
        r = run_main("experiment_setup.py", str(tmp_path), "python train.py")
        assert r.returncode == 0 and json.loads(r.stdout)["exp_id"] == "exp-001"


# ---------------------------------------------------------------------------
# TestPhaseGates
# ---------------------------------------------------------------------------

class TestPhaseGates:
    """Phase gate protocol: transition validation, gate logging, recovery instructions."""

    def test_valid_transitions(self, tmp_path):
        """All edges in PHASE_TRANSITIONS pass validation."""
        for src, dsts in PHASE_TRANSITIONS.items():
            for dst in dsts:
                # Log the source phase as completed (unless phase 0)
                if src != 0:
                    log_phase_gate(str(tmp_path), src, "completed", f"Phase {src} done")
                # Set up file prerequisites for the destination phase
                _setup_phase_prereqs(tmp_path, dst)
                result = validate_phase_gate(src, dst, str(tmp_path))
                assert result["valid"] is True, (
                    f"Transition {src} -> {dst} should be valid but got errors: "
                    f"{result['errors']}"
                )

    def test_invalid_transitions(self, tmp_path):
        """Disallowed jumps (e.g., 2->7) are rejected."""
        invalid_pairs = [(2, 7), (0, 5), (3, 9), (1, 4), (5, 8), (9, 0)]
        for src, dst in invalid_pairs:
            # Even if we log completion, the transition itself is illegal
            if src != 0:
                log_phase_gate(str(tmp_path), src, "completed", f"Phase {src} done")
            result = validate_phase_gate(src, dst, str(tmp_path))
            assert result["valid"] is False, (
                f"Transition {src} -> {dst} should be invalid"
            )
            assert any("not allowed" in e for e in result["errors"])

    def test_gate_requires_completion(self, tmp_path):
        """Gate fails if previous phase status is not 'completed'."""
        # Log phase 3 as "failed"
        log_phase_gate(str(tmp_path), 3, "failed", "Phase 3 crashed")
        # Set up prereqs for phase 4
        _write_baseline(tmp_path, {"loss": 0.5}, {"lr": 0.001})
        result = validate_phase_gate(3, 4, str(tmp_path))
        assert result["valid"] is False
        assert any("not been marked as completed" in e for e in result["errors"])

    def test_phase_zero_no_completion_required(self, tmp_path):
        """Phase 0 doesn't need a prior gate log entry."""
        # gate 0->1 should work without any prior logs
        # Phase 1 -> validate_phase_requirements for phase 1 has no rules
        # (undefined phase warns but is still valid)
        result = validate_phase_gate(0, 1, str(tmp_path))
        assert result["valid"] is True

    def test_recovery_instructions(self):
        """Each phase x failure type has a non-empty recovery string."""
        for (phase, failure_type), instruction in RECOVERY_INSTRUCTIONS.items():
            assert isinstance(instruction, str)
            assert len(instruction) > 0
            result = get_recovery_instruction(phase, failure_type)
            assert result == instruction

    def test_recovery_default_fallback(self):
        """Unknown failure types get the default recovery instruction."""
        result = get_recovery_instruction(99, "unknown_failure")
        assert result == _RECOVERY_DEFAULT
        assert len(result) > 0
        # Also check a known phase with unknown failure type
        result2 = get_recovery_instruction(7, "some_other_error")
        assert result2 == _RECOVERY_DEFAULT

    def test_log_phase_gate_creates_file(self, tmp_path):
        """First call creates phase-gates.json."""
        gate_path = tmp_path / "phase-gates.json"
        assert not gate_path.exists()
        log_phase_gate(str(tmp_path), 1, "completed", "Phase 1 finished")
        assert gate_path.exists()
        entries = json.loads(gate_path.read_text())
        assert len(entries) == 1
        assert entries[0]["phase"] == 1
        assert entries[0]["status"] == "completed"
        assert entries[0]["summary"] == "Phase 1 finished"

    def test_log_phase_gate_appends(self, tmp_path):
        """Multiple calls append entries."""
        log_phase_gate(str(tmp_path), 1, "completed", "Phase 1 done")
        log_phase_gate(str(tmp_path), 2, "completed", "Phase 2 done")
        log_phase_gate(str(tmp_path), 3, "failed", "Phase 3 crashed")
        entries = json.loads((tmp_path / "phase-gates.json").read_text())
        assert len(entries) == 3
        assert entries[0]["phase"] == 1
        assert entries[1]["phase"] == 2
        assert entries[2]["phase"] == 3
        assert entries[2]["status"] == "failed"

    def test_log_phase_gate_has_timestamp(self, tmp_path):
        """Each entry has an ISO8601 timestamp."""
        log_phase_gate(str(tmp_path), 5, "completed", "Research done")
        entries = json.loads((tmp_path / "phase-gates.json").read_text())
        ts = entries[0]["timestamp"]
        # Verify it parses as ISO8601
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None  # Should be timezone-aware


def _setup_phase_prereqs(tmp_path, phase):
    """Set up minimal file prerequisites so validate_phase_requirements passes."""
    if phase in (4, 5, 6, 7, 9):
        _write_baseline(tmp_path, {"loss": 0.5}, {"lr": 0.001})
    elif phase == 3:
        _make_results_dir(tmp_path)
    if phase == 7:
        # Phase 7 also requires optimization-goals.json
        goals_path = tmp_path / "optimization-goals.json"
        if not goals_path.exists():
            goals_path.write_text(json.dumps({"objective": {"primary_metric": "loss"}}))
    if phase == 8:
        _write_baseline(tmp_path, {"loss": 0.5}, {"lr": 0.001})
        d = tmp_path / "results"
        d.mkdir(exist_ok=True)
        (d / "implementation-manifest.json").write_text(
            json.dumps({"proposals": []})
        )


# ---------------------------------------------------------------------------
# TestDecisionLogging
# ---------------------------------------------------------------------------

class TestDecisionLogging:
    """Decision logging (pipeline_state decision logging functions)."""

    def _make_decision(self, **overrides):
        """Helper to create a valid decision dict with optional overrides."""
        base = {
            "phase": 7,
            "agent": "analysis",
            "decision_type": "pivot",
            "decision": "switch to code_evolution",
        }
        base.update(overrides)
        return base

    def test_log_decision_creates_file(self, tmp_path):
        """First log call creates decision-log.json."""
        log_path = tmp_path / "decision-log.json"
        assert not log_path.exists()
        log_decision(str(tmp_path), self._make_decision())
        assert log_path.exists()
        entries = json.loads(log_path.read_text())
        assert len(entries) == 1

    def test_log_decision_appends(self, tmp_path):
        """Multiple calls append entries."""
        log_decision(str(tmp_path), self._make_decision(decision="first"))
        log_decision(str(tmp_path), self._make_decision(decision="second"))
        log_decision(str(tmp_path), self._make_decision(decision="third"))
        entries = json.loads((tmp_path / "decision-log.json").read_text())
        assert len(entries) == 3
        assert entries[0]["decision"] == "first"
        assert entries[1]["decision"] == "second"
        assert entries[2]["decision"] == "third"

    def test_log_decision_validates_required_fields(self, tmp_path):
        """Missing required fields raise ValueError."""
        with pytest.raises(ValueError, match="phase"):
            log_decision(str(tmp_path), {"agent": "a", "decision_type": "t", "decision": "d"})
        with pytest.raises(ValueError, match="agent"):
            log_decision(str(tmp_path), {"phase": 1, "decision_type": "t", "decision": "d"})
        with pytest.raises(ValueError, match="decision_type"):
            log_decision(str(tmp_path), {"phase": 1, "agent": "a", "decision": "d"})
        with pytest.raises(ValueError, match="decision"):
            log_decision(str(tmp_path), {"phase": 1, "agent": "a", "decision_type": "t"})

    def test_log_decision_adds_timestamp(self, tmp_path):
        """Each entry gets an ISO8601 timestamp."""
        log_decision(str(tmp_path), self._make_decision())
        entries = json.loads((tmp_path / "decision-log.json").read_text())
        ts = entries[0]["timestamp"]
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None

    def test_log_decision_auto_increments_id(self, tmp_path):
        """IDs auto-increment from 1."""
        id1 = log_decision(str(tmp_path), self._make_decision(decision="first"))
        id2 = log_decision(str(tmp_path), self._make_decision(decision="second"))
        id3 = log_decision(str(tmp_path), self._make_decision(decision="third"))
        assert id1 == "1"
        assert id2 == "2"
        assert id3 == "3"

    def test_log_decision_computes_inputs_hash(self, tmp_path):
        """Entries have a SHA-256 inputs_hash."""
        log_decision(str(tmp_path), self._make_decision())
        entries = json.loads((tmp_path / "decision-log.json").read_text())
        h = entries[0]["inputs_hash"]
        assert len(h) == 64
        assert all(ch in "0123456789abcdef" for ch in h)

    def test_inputs_hash_deterministic(self, tmp_path):
        """Same inputs produce the same hash."""
        d1 = self._make_decision()
        d2 = self._make_decision()
        log_decision(str(tmp_path), d1)
        log_decision(str(tmp_path), d2)
        entries = json.loads((tmp_path / "decision-log.json").read_text())
        assert entries[0]["inputs_hash"] == entries[1]["inputs_hash"]

    def test_inputs_hash_differs_on_different_inputs(self, tmp_path):
        """Different inputs produce different hashes."""
        log_decision(str(tmp_path), self._make_decision(agent="analysis"))
        log_decision(str(tmp_path), self._make_decision(agent="tuning"))
        entries = json.loads((tmp_path / "decision-log.json").read_text())
        assert entries[0]["inputs_hash"] != entries[1]["inputs_hash"]

    def test_get_decisions_all(self, tmp_path):
        """No filters returns all decisions."""
        log_decision(str(tmp_path), self._make_decision(decision="a"))
        log_decision(str(tmp_path), self._make_decision(decision="b"))
        log_decision(str(tmp_path), self._make_decision(decision="c"))
        result = get_decisions(str(tmp_path))
        assert len(result) == 3

    def test_get_decisions_filter_by_phase(self, tmp_path):
        """Filter by phase returns correct subset."""
        log_decision(str(tmp_path), self._make_decision(phase=5, decision="research"))
        log_decision(str(tmp_path), self._make_decision(phase=7, decision="pivot"))
        log_decision(str(tmp_path), self._make_decision(phase=7, decision="stop"))
        result = get_decisions(str(tmp_path), phase=7)
        assert len(result) == 2
        assert all(e["phase"] == 7 for e in result)

    def test_get_decisions_filter_by_agent(self, tmp_path):
        """Filter by agent returns correct subset."""
        log_decision(str(tmp_path), self._make_decision(agent="analysis"))
        log_decision(str(tmp_path), self._make_decision(agent="tuning"))
        log_decision(str(tmp_path), self._make_decision(agent="analysis"))
        result = get_decisions(str(tmp_path), agent="tuning")
        assert len(result) == 1
        assert result[0]["agent"] == "tuning"

    def test_get_decisions_empty(self, tmp_path):
        """Returns empty list when no log exists."""
        result = get_decisions(str(tmp_path))
        assert result == []

    def test_replay_check_match(self, tmp_path):
        """Identical decision matches."""
        decision = self._make_decision(
            context_summary="loss plateaued at 0.3",
            decision="switch to code_evolution",
        )
        id1 = log_decision(str(tmp_path), decision)
        # Log the same decision again (same inputs, same decision text)
        id2 = log_decision(str(tmp_path), decision)
        result = replay_check(str(tmp_path), id2, "switch to code_evolution")
        assert result["match"] is True
        assert result["divergence"] is None
        assert result["original_decision"] == "switch to code_evolution"
        assert result["current_decision"] == "switch to code_evolution"

    def test_replay_check_divergence(self, tmp_path):
        """Different decision flags divergence."""
        decision = self._make_decision(
            context_summary="loss plateaued at 0.3",
            decision="switch to code_evolution",
        )
        id1 = log_decision(str(tmp_path), decision)
        # Log same inputs but different decision
        decision2 = self._make_decision(
            context_summary="loss plateaued at 0.3",
            decision="try research_implement",
        )
        id2 = log_decision(str(tmp_path), decision2)
        result = replay_check(str(tmp_path), id2, "try research_implement")
        assert result["match"] is False
        assert "diverged" in result["divergence"]
        assert result["original_decision"] == "switch to code_evolution"
        assert result["current_decision"] == "try research_implement"

    def test_replay_check_no_prior(self, tmp_path):
        """No matching inputs returns match=None."""
        decision = self._make_decision(
            context_summary="unique context",
            decision="unique decision",
        )
        id1 = log_decision(str(tmp_path), decision)
        result = replay_check(str(tmp_path), id1, "unique decision")
        assert result["match"] is None
        assert result["divergence"] == "no_prior_decision_with_matching_inputs"

