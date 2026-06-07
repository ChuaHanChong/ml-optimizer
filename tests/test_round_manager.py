#!/usr/bin/env python3
"""Tests for round_manager.py — round lifecycle and completeness checking."""

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from round_manager import (
    VALID_ROUND_TYPES,
    check_baseline,
    check_manifest,
    check_prerequisites,
    check_proposals,
    check_round,
    close_round,
    create_round,
    current_round,
    next_experiment_id,
    register_experiment,
)
from schema_validator import validate_hp_proposal, validate_rounds_manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _make_exp_result(exp_id, status="completed", metrics=None, config=None):
    return {
        "exp_id": exp_id,
        "status": status,
        "metrics": metrics or {"loss": 0.5},
        "config": config or {"lr": 0.001},
    }


# ---------------------------------------------------------------------------
# TestCreateRound
# ---------------------------------------------------------------------------

class TestCreateRound:
    def test_creates_round_directory(self, tmp_path):
        result = create_round(str(tmp_path), "hp")
        assert result["id"] == 1
        assert result["dir"] == "round-1-hp"
        assert (tmp_path / "results" / "round-1-hp").is_dir()

    def test_creates_proposed_configs_directory(self, tmp_path):
        create_round(str(tmp_path), "hp")
        assert (tmp_path / "proposed-configs" / "round-1-hp").is_dir()

    def test_creates_logs_scripts_artifacts_dirs(self, tmp_path):
        create_round(str(tmp_path), "hp")
        assert (tmp_path / "logs" / "round-1-hp").is_dir()
        assert (tmp_path / "scripts" / "round-1-hp").is_dir()
        assert (tmp_path / "artifacts" / "round-1-hp").is_dir()

    def test_creates_manifest(self, tmp_path):
        create_round(str(tmp_path), "hp")
        manifest_path = tmp_path / "results" / "rounds-manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["current_round"] == 1
        assert len(manifest["rounds"]) == 1
        assert manifest["rounds"][0]["type"] == "hp"

    def test_increments_round_id(self, tmp_path):
        create_round(str(tmp_path), "hp")
        result = create_round(str(tmp_path), "evolved")
        assert result["id"] == 2
        assert result["dir"] == "round-2-evolved"

    def test_all_round_types(self, tmp_path):
        for i, rtype in enumerate(VALID_ROUND_TYPES):
            result = create_round(str(tmp_path), rtype)
            assert result["id"] == i + 1
            assert result["type"] == rtype

    def test_invalid_round_type(self, tmp_path):
        result = create_round(str(tmp_path), "invalid")
        assert "error" in result

    def test_branch(self, tmp_path):
        result = create_round(str(tmp_path), "hp", branch="ml-opt/test")
        manifest = json.loads((tmp_path / "results" / "rounds-manifest.json").read_text())
        assert manifest["rounds"][0]["code_branch"] == "ml-opt/test"
        assert result["id"] == 1


# ---------------------------------------------------------------------------
# TestCurrentRound
# ---------------------------------------------------------------------------

class TestCurrentRound:
    def test_none_when_empty(self, tmp_path):
        assert current_round(str(tmp_path)) is None

    def test_returns_latest(self, tmp_path):
        create_round(str(tmp_path), "hp")
        create_round(str(tmp_path), "evolved")
        result = current_round(str(tmp_path))
        assert result["id"] == 2
        assert result["type"] == "evolved"


# ---------------------------------------------------------------------------
# TestRegisterExperiment
# ---------------------------------------------------------------------------

class TestRegisterExperiment:
    def test_registers_experiment(self, tmp_path):
        create_round(str(tmp_path), "hp")
        result = register_experiment(str(tmp_path), "round-1-hp", "exp-001")
        assert result["registered"] is True

    def test_updates_manifest(self, tmp_path):
        create_round(str(tmp_path), "hp")
        register_experiment(str(tmp_path), "round-1-hp", "exp-001")
        register_experiment(str(tmp_path), "round-1-hp", "exp-002")
        manifest = json.loads((tmp_path / "results" / "rounds-manifest.json").read_text())
        assert manifest["rounds"][0]["experiments"] == ["exp-001", "exp-002"]
        assert manifest["total_experiments"] == 2

    def test_no_duplicate(self, tmp_path):
        create_round(str(tmp_path), "hp")
        register_experiment(str(tmp_path), "round-1-hp", "exp-001")
        register_experiment(str(tmp_path), "round-1-hp", "exp-001")
        manifest = json.loads((tmp_path / "results" / "rounds-manifest.json").read_text())
        assert manifest["rounds"][0]["experiments"] == ["exp-001"]

    def test_unknown_round(self, tmp_path):
        create_round(str(tmp_path), "hp")
        result = register_experiment(str(tmp_path), "round-99-hp", "exp-001")
        assert "error" in result


# ---------------------------------------------------------------------------
# TestCloseRound
# ---------------------------------------------------------------------------

class TestCloseRound:
    def test_closes_round(self, tmp_path):
        create_round(str(tmp_path), "hp")
        result = close_round(str(tmp_path), summary="Initial exploration")
        assert result["closed"] is True
        manifest = json.loads((tmp_path / "results" / "rounds-manifest.json").read_text())
        assert manifest["rounds"][0]["closed_at"] is not None
        assert manifest["rounds"][0]["summary"] == "Initial exploration"

    def test_no_rounds(self, tmp_path):
        result = close_round(str(tmp_path))
        assert "error" in result


# ---------------------------------------------------------------------------
# TestNextExperimentId
# ---------------------------------------------------------------------------

class TestNextExperimentId:
    def test_first_id(self, tmp_path):
        (tmp_path / "results").mkdir(parents=True)
        assert next_experiment_id(str(tmp_path)) == "exp-001"

    def test_increments_across_rounds(self, tmp_path):
        create_round(str(tmp_path), "hp")
        _write_json(tmp_path / "results" / "round-1-hp" / "exp-001.json", {})
        _write_json(tmp_path / "results" / "round-1-hp" / "exp-002.json", {})
        create_round(str(tmp_path), "evolved")
        _write_json(tmp_path / "results" / "round-2-evolved" / "exp-003.json", {})
        assert next_experiment_id(str(tmp_path)) == "exp-004"

    def test_handles_flat_and_round_dirs(self, tmp_path):
        (tmp_path / "results").mkdir(parents=True)
        # Flat (backwards compat)
        _write_json(tmp_path / "results" / "exp-005.json", {})
        # Round dir
        create_round(str(tmp_path), "hp")
        _write_json(tmp_path / "results" / "round-1-hp" / "exp-003.json", {})
        assert next_experiment_id(str(tmp_path)) == "exp-006"


# ---------------------------------------------------------------------------
# TestCheckBaseline
# ---------------------------------------------------------------------------

class TestCheckBaseline:
    def test_missing(self, tmp_path):
        result = check_baseline(str(tmp_path))
        assert result["complete"] is False

    def test_valid(self, tmp_path):
        _write_json(tmp_path / "results" / "baseline.json", {
            "exp_id": "baseline", "status": "completed",
            "config": {"lr": 0.001}, "metrics": {"loss": 0.5},
        })
        result = check_baseline(str(tmp_path))
        assert result["complete"] is True

    def test_invalid_schema(self, tmp_path):
        _write_json(tmp_path / "results" / "baseline.json", {"bad": "data"})
        result = check_baseline(str(tmp_path))
        assert result["complete"] is False


# ---------------------------------------------------------------------------
# TestCheckPrerequisites
# ---------------------------------------------------------------------------


class TestCheckPrerequisites:
    """Tests for round_manager.check_prerequisites.

    Exercises the prerequisites.json existence + schema-validation path via
    the same _check_file helper used by check_baseline and check_manifest.
    """

    def test_missing(self, tmp_path):
        """prerequisites.json does not exist -> not complete, clear error."""
        result = check_prerequisites(str(tmp_path))
        assert result["complete"] is False
        assert any("prerequisites.json" in err for err in result["errors"])

    def test_valid(self, tmp_path):
        """Valid prerequisites.json with all required fields -> complete."""
        _write_json(tmp_path / "results" / "prerequisites.json", {
            "status": "ready",
            "dataset": {
                "train_path": "/data/train",
                "val_path": "/data/val",
                "prepared": False,
            },
            "environment": {
                "manager": "conda",
                "name": "base",
            },
            "ready_for_baseline": True,
        })
        result = check_prerequisites(str(tmp_path))
        assert result["complete"] is True
        assert result["errors"] == []

    def test_invalid_schema(self, tmp_path):
        """Missing required fields -> not complete, errors list is populated."""
        _write_json(tmp_path / "results" / "prerequisites.json", {
            "status": "ready",
            # missing: dataset, environment, ready_for_baseline
        })
        result = check_prerequisites(str(tmp_path))
        assert result["complete"] is False
        assert len(result["errors"]) > 0

    def test_malformed_json(self, tmp_path):
        """Corrupt JSON file -> not complete, error mentions parse failure."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "prerequisites.json").write_text("{not valid json")
        result = check_prerequisites(str(tmp_path))
        assert result["complete"] is False
        assert len(result["errors"]) > 0


# ---------------------------------------------------------------------------
# TestCheckManifest
# ---------------------------------------------------------------------------


class TestCheckManifest:
    """Tests for round_manager.check_manifest.

    Exercises the implementation-manifest.json existence + schema-validation
    path. The manifest is produced by the implement-agent at Phase 6 and
    checked by the orchestrator before Phase 7 experiment dispatch.
    """

    def test_missing(self, tmp_path):
        """implementation-manifest.json does not exist -> not complete."""
        result = check_manifest(str(tmp_path))
        assert result["complete"] is False
        assert any("implementation-manifest.json" in err for err in result["errors"])

    def test_valid_git_branch_strategy(self, tmp_path):
        """Valid manifest with git_branch strategy and one validated proposal."""
        _write_json(tmp_path / "results" / "implementation-manifest.json", {
            "original_branch": "main",
            "strategy": "git_branch",
            "proposals": [
                {
                    "name": "Label Smoothing",
                    "slug": "label-smoothing",
                    "status": "validated",
                    "branch": "ml-opt/label-smoothing",
                }
            ],
        })
        result = check_manifest(str(tmp_path))
        assert result["complete"] is True
        assert result["errors"] == []

    def test_valid_file_backup_strategy(self, tmp_path):
        """Valid manifest with file_backup strategy (non-git projects)."""
        _write_json(tmp_path / "results" / "implementation-manifest.json", {
            "original_branch": "main",
            "strategy": "file_backup",
            "proposals": [],
        })
        result = check_manifest(str(tmp_path))
        assert result["complete"] is True

    def test_invalid_strategy(self, tmp_path):
        """Invalid strategy value -> schema validation fails."""
        _write_json(tmp_path / "results" / "implementation-manifest.json", {
            "original_branch": "main",
            "strategy": "bogus_strategy",
            "proposals": [],
        })
        result = check_manifest(str(tmp_path))
        assert result["complete"] is False
        assert any("strategy" in err.lower() for err in result["errors"])

    def test_missing_required_fields(self, tmp_path):
        """Missing original_branch / strategy / proposals -> not complete."""
        _write_json(tmp_path / "results" / "implementation-manifest.json", {
            "proposals": [],
        })
        result = check_manifest(str(tmp_path))
        assert result["complete"] is False
        assert len(result["errors"]) >= 2  # missing original_branch + strategy

    def test_proposal_missing_required_fields(self, tmp_path):
        """Proposal missing required fields (name/slug/status) -> not complete."""
        _write_json(tmp_path / "results" / "implementation-manifest.json", {
            "original_branch": "main",
            "strategy": "git_branch",
            "proposals": [
                {"name": "Incomplete"},  # missing slug + status
            ],
        })
        result = check_manifest(str(tmp_path))
        assert result["complete"] is False
        assert any("slug" in err for err in result["errors"])
        assert any("status" in err for err in result["errors"])


# ---------------------------------------------------------------------------
# TestCheckRound
# ---------------------------------------------------------------------------

class TestCheckRound:
    def test_nonexistent_round(self, tmp_path):
        result = check_round(str(tmp_path), "round-99-hp")
        assert result["complete"] is False

    def test_empty_round(self, tmp_path):
        create_round(str(tmp_path), "hp")
        result = check_round(str(tmp_path), "round-1-hp")
        assert result["complete"] is False
        assert result["total"] == 0

    def test_valid_round(self, tmp_path):
        create_round(str(tmp_path), "hp")
        exp = _make_exp_result("exp-001")
        _write_json(tmp_path / "results" / "round-1-hp" / "exp-001.json", exp)
        # Create log file (round-based)
        (tmp_path / "logs" / "round-1-hp" / "exp-001").mkdir(parents=True)
        (tmp_path / "logs" / "round-1-hp" / "exp-001" / "train.log").write_text("training log")
        result = check_round(str(tmp_path), "round-1-hp")
        assert result["complete"] is True
        assert result["total"] == 1
        assert result["valid"] == 1

    def test_non_terminal_status(self, tmp_path):
        create_round(str(tmp_path), "hp")
        exp = _make_exp_result("exp-001", status="running")
        _write_json(tmp_path / "results" / "round-1-hp" / "exp-001.json", exp)
        result = check_round(str(tmp_path), "round-1-hp")
        assert result["complete"] is False
        assert result["non_terminal"] == ["exp-001"]

    def test_invalid_schema(self, tmp_path):
        create_round(str(tmp_path), "hp")
        _write_json(tmp_path / "results" / "round-1-hp" / "exp-001.json", {"bad": True})
        result = check_round(str(tmp_path), "round-1-hp")
        assert result["complete"] is False
        assert len(result["invalid"]) == 1

    def test_missing_log(self, tmp_path):
        create_round(str(tmp_path), "hp")
        exp = _make_exp_result("exp-001")
        _write_json(tmp_path / "results" / "round-1-hp" / "exp-001.json", exp)
        result = check_round(str(tmp_path), "round-1-hp")
        assert result["missing_logs"] == ["exp-001"]


# ---------------------------------------------------------------------------
# TestCheckProposals
# ---------------------------------------------------------------------------

class TestCheckProposals:
    def test_nonexistent(self, tmp_path):
        result = check_proposals(str(tmp_path), "round-99-hp")
        assert result["complete"] is False

    def test_valid_proposals(self, tmp_path):
        create_round(str(tmp_path), "hp")
        _write_json(
            tmp_path / "proposed-configs" / "round-1-hp" / "exp-001.json",
            {"exp_id": "exp-001", "config": {"lr": 0.01}},
        )
        result = check_proposals(str(tmp_path), "round-1-hp")
        assert result["complete"] is True
        assert result["total"] == 1
        assert result["valid"] == 1

    def test_mismatched_exp_id(self, tmp_path):
        create_round(str(tmp_path), "hp")
        _write_json(
            tmp_path / "proposed-configs" / "round-1-hp" / "exp-001.json",
            {"exp_id": "exp-999", "config": {"lr": 0.01}},
        )
        result = check_proposals(str(tmp_path), "round-1-hp")
        assert result["complete"] is False
        assert len(result["invalid"]) == 1


# ---------------------------------------------------------------------------
# TestSchemaValidation (new schemas)
# ---------------------------------------------------------------------------

class TestHpProposalSchema:
    def test_valid_minimal(self):
        result = validate_hp_proposal({"exp_id": "exp-001", "config": {"lr": 0.01}})
        assert result["valid"] is True

    def test_missing_config(self):
        result = validate_hp_proposal({"exp_id": "exp-001"})
        assert result["valid"] is False

    def test_missing_exp_id(self):
        result = validate_hp_proposal({"config": {"lr": 0.01}})
        assert result["valid"] is False

    def test_invalid_method_tier(self):
        result = validate_hp_proposal({
            "exp_id": "exp-001", "config": {"lr": 0.01},
            "method_tier": "invalid",
        })
        assert result["valid"] is False


class TestRoundsManifestSchema:
    def test_valid(self):
        result = validate_rounds_manifest({
            "rounds": [{"id": 1, "dir": "round-1-hp", "type": "hp", "experiments": []}],
            "current_round": 1,
            "total_experiments": 0,
        })
        assert result["valid"] is True

    def test_missing_rounds(self):
        result = validate_rounds_manifest({"current_round": 1, "total_experiments": 0})
        assert result["valid"] is False

    def test_invalid_round_type(self):
        result = validate_rounds_manifest({
            "rounds": [{"id": 1, "dir": "round-1-bad", "type": "bad", "experiments": []}],
            "current_round": 1,
            "total_experiments": 0,
        })
        assert result["valid"] is False

    def test_negative_total(self):
        result = validate_rounds_manifest({
            "rounds": [],
            "current_round": 0,
            "total_experiments": -1,
        })
        assert result["valid"] is False


# ---------------------------------------------------------------------------
# TestWriteHookCompleteness — unit tests for _check_completeness()
# ---------------------------------------------------------------------------

from validate_experiment_write import _check_completeness, _check_goal_compliance, _find_exp_root


class TestWriteHookCompleteness:
    """Test completeness enforcement for experiment result writes."""

    def test_blocks_completed_missing_iteration(self):
        """Completed experiment without 'iteration' field is blocked."""
        data = {
            "exp_id": "exp-001",
            "status": "completed",
            "config": {"lr": 0.001},
            "metrics": {"loss": 0.5},
            "method_tier": "baseline",
            "duration_seconds": 120.0,
            # missing: iteration
        }
        reason = _check_completeness(data)
        assert reason is not None
        assert "iteration" in reason

    def test_allows_completed_with_all_fields(self):
        """Completed experiment with all mandatory fields is approved."""
        data = {
            "exp_id": "exp-001",
            "status": "completed",
            "config": {"lr": 0.001},
            "metrics": {"loss": 0.5},
            "iteration": 1,
            "method_tier": "baseline",
            "duration_seconds": 120.0,
        }
        reason = _check_completeness(data)
        assert reason is None

    def test_allows_running_placeholder(self):
        """Running (placeholder) experiment with minimal fields is approved."""
        data = {
            "exp_id": "exp-001",
            "status": "running",
            "config": {"lr": 0.001},
        }
        reason = _check_completeness(data)
        assert reason is None

    def test_allows_pending_placeholder(self):
        """Pending (placeholder) experiment with minimal fields is approved."""
        data = {
            "exp_id": "exp-001",
            "status": "pending",
        }
        reason = _check_completeness(data)
        assert reason is None

    def test_blocks_failed_missing_notes(self):
        """Failed experiment without 'notes' field is blocked."""
        data = {
            "exp_id": "exp-001",
            "status": "failed",
            "config": {"lr": 0.001},
            "metrics": {},
        }
        reason = _check_completeness(data)
        assert reason is not None
        assert "notes" in reason
        assert "Failed" in reason

    def test_allows_failed_with_notes(self):
        """Failed experiment with 'notes' field is approved."""
        data = {
            "exp_id": "exp-001",
            "status": "failed",
            "config": {"lr": 0.001},
            "metrics": {},
            "notes": "OOM at batch size 64",
        }
        reason = _check_completeness(data)
        assert reason is None

    def test_blocks_diverged_missing_notes(self):
        """Diverged experiment without 'notes' field is blocked."""
        data = {
            "exp_id": "exp-001",
            "status": "diverged",
            "config": {"lr": 0.1},
            "metrics": {"loss": float("inf")},
        }
        reason = _check_completeness(data)
        assert reason is not None
        assert "notes" in reason
        assert "Diverged" in reason

    def test_blocks_stacked_missing_code_branches(self):
        """Completed stacked experiment without 'code_branches' is blocked."""
        data = {
            "exp_id": "exp-010",
            "status": "completed",
            "config": {"lr": 0.001},
            "metrics": {"loss": 0.3},
            "iteration": 5,
            "method_tier": "stacked_tuned_hp",
            "duration_seconds": 300.0,
            # missing: code_branches, stacking_order
        }
        reason = _check_completeness(data)
        assert reason is not None
        assert "code_branches" in reason
        assert "stacking_order" in reason

    def test_allows_stacked_with_all_fields(self):
        """Completed stacked experiment with all fields is approved."""
        data = {
            "exp_id": "exp-010",
            "status": "completed",
            "config": {"lr": 0.001},
            "metrics": {"loss": 0.3},
            "iteration": 5,
            "method_tier": "stacked_tuned_hp",
            "duration_seconds": 300.0,
            "code_branches": ["ml-opt/branch-a", "ml-opt/branch-b"],
            "stacking_order": 2,
        }
        reason = _check_completeness(data)
        assert reason is None

    def test_blocks_completed_missing_multiple_fields(self):
        """Completed experiment missing all three mandatory fields lists them all."""
        data = {
            "exp_id": "exp-001",
            "status": "completed",
            "config": {"lr": 0.001},
            "metrics": {"loss": 0.5},
            # missing: iteration, method_tier, duration_seconds
        }
        reason = _check_completeness(data)
        assert reason is not None
        assert "iteration" in reason
        assert "method_tier" in reason
        assert "duration_seconds" in reason


# ---------------------------------------------------------------------------
# TestWriteHookGoalCompliance — unit tests for _check_goal_compliance()
# ---------------------------------------------------------------------------


class TestWriteHookGoalCompliance:
    """Test goal compliance enforcement (frozen params, OOM limits)."""

    def _setup_goals(self, tmp_path, frozen=None):
        """Create optimization-goals.json + breadcrumb."""
        exp_root = tmp_path / "experiments"
        exp_root.mkdir(parents=True, exist_ok=True)
        goals = {
            "objective": {"primary_metric": "loss"},
            "constraints": {"frozen_parameters": frozen or []},
        }
        (exp_root / "optimization-goals.json").write_text(json.dumps(goals))
        (tmp_path / ".claude").mkdir(exist_ok=True)
        (tmp_path / ".claude" / "ml-optimizer.json").write_text(
            json.dumps({"exp_root": str(exp_root)})
        )
        return exp_root

    def test_blocks_frozen_param(self, tmp_path):
        exp_root = self._setup_goals(tmp_path, frozen=["lr"])
        file_path = str(exp_root / "results" / "round-1-hp" / "exp-001.json")
        data = {"status": "completed", "config": {"lr": 0.01, "batch_size": 32}}
        reason = _check_goal_compliance(data, file_path)
        assert reason is not None
        assert "frozen" in reason.lower()

    def test_allows_without_frozen(self, tmp_path):
        exp_root = self._setup_goals(tmp_path, frozen=["lr"])
        file_path = str(exp_root / "results" / "round-1-hp" / "exp-001.json")
        data = {"status": "completed", "config": {"batch_size": 32}}
        reason = _check_goal_compliance(data, file_path)
        assert reason is None

    def test_skips_without_breadcrumb(self, tmp_path):
        file_path = str(tmp_path / "experiments" / "results" / "round-1-hp" / "exp-001.json")
        data = {"status": "completed", "config": {"lr": 0.01}}
        reason = _check_goal_compliance(data, file_path)
        assert reason is None

    def test_skips_running_status(self, tmp_path):
        exp_root = self._setup_goals(tmp_path, frozen=["lr"])
        file_path = str(exp_root / "results" / "round-1-hp" / "exp-001.json")
        data = {"status": "running", "config": {"lr": 0.01}}
        reason = _check_goal_compliance(data, file_path)
        assert reason is None

    def test_blocks_oom_batch_size(self, tmp_path):
        exp_root = self._setup_goals(tmp_path)
        behaviors = {"resource_constraints": [{"max_batch_size": 32}]}
        (exp_root / "learned-behaviors.json").write_text(json.dumps(behaviors))
        file_path = str(exp_root / "results" / "round-1-hp" / "exp-001.json")
        data = {"status": "completed", "config": {"batch_size": 64}}
        reason = _check_goal_compliance(data, file_path)
        assert reason is not None
        assert "batch_size" in reason

    def test_find_exp_root(self, tmp_path):
        exp_root = tmp_path / "project" / "experiments"
        exp_root.mkdir(parents=True)
        (tmp_path / "project" / ".claude").mkdir()
        (tmp_path / "project" / ".claude" / "ml-optimizer.json").write_text(
            json.dumps({"exp_root": str(exp_root)})
        )
        file_path = str(exp_root / "results" / "round-1-hp" / "exp-001.json")
        assert _find_exp_root(file_path) == str(exp_root)

    def test_find_exp_root_missing(self, tmp_path):
        file_path = str(tmp_path / "nowhere" / "exp-001.json")
        assert _find_exp_root(file_path) is None


# ---------------------------------------------------------------------------
# TestAgentOutputValidation — SubagentStop hook output-file checks
# ---------------------------------------------------------------------------

from validate_agent_output import check_agent_output


class TestAgentOutputValidation:
    """SubagentStop hook blocks agents that didn't produce expected output."""

    def _setup_exp(self, tmp_path):
        exp_root = tmp_path / "experiments"
        exp_root.mkdir(parents=True)
        (exp_root / "results").mkdir()
        (exp_root / "reports").mkdir()
        (tmp_path / ".claude").mkdir(exist_ok=True)
        (tmp_path / ".claude" / "ml-optimizer.json").write_text(
            json.dumps({"exp_root": str(exp_root)})
        )
        return exp_root

    def test_blocks_baseline_agent_without_output(self, tmp_path):
        self._setup_exp(tmp_path)
        result = check_agent_output(str(tmp_path), "baseline-agent")
        assert result["decision"] == "block"

    def test_blocks_baseline_agent_missing_log(self, tmp_path):
        exp_root = self._setup_exp(tmp_path)
        (exp_root / "results" / "baseline.json").write_text('{"exp_id": "baseline"}')
        result = check_agent_output(str(tmp_path), "baseline-agent")
        assert result["decision"] == "block"
        assert "train.log" in result["reason"]

    def test_allows_baseline_agent_with_all_output(self, tmp_path):
        exp_root = self._setup_exp(tmp_path)
        (exp_root / "results" / "baseline.json").write_text('{"exp_id": "baseline"}')
        (exp_root / "logs" / "baseline").mkdir(parents=True)
        (exp_root / "logs" / "baseline" / "train.log").write_text("training log")
        result = check_agent_output(str(tmp_path), "baseline-agent")
        assert result["decision"] == "approve"

    def test_blocks_prerequisites_agent_without_output(self, tmp_path):
        self._setup_exp(tmp_path)
        result = check_agent_output(str(tmp_path), "prerequisites-agent")
        assert result["decision"] == "block"

    def test_allows_prerequisites_agent_with_output(self, tmp_path):
        exp_root = self._setup_exp(tmp_path)
        (exp_root / "results" / "prerequisites.json").write_text('{"status": "ready"}')
        result = check_agent_output(str(tmp_path), "prerequisites-agent")
        assert result["decision"] == "approve"

    def test_blocks_research_agent_without_output(self, tmp_path):
        self._setup_exp(tmp_path)
        result = check_agent_output(str(tmp_path), "research-agent")
        assert result["decision"] == "block"

    def test_allows_research_agent_with_output(self, tmp_path):
        exp_root = self._setup_exp(tmp_path)
        (exp_root / "reports" / "research-findings.md").write_text("# Research")
        result = check_agent_output(str(tmp_path), "research-agent")
        assert result["decision"] == "approve"

    def test_allows_research_agent_with_variant_filename(self, tmp_path):
        exp_root = self._setup_exp(tmp_path)
        (exp_root / "reports" / "research-findings-method-proposals.md").write_text(
            "# Method Proposals"
        )
        result = check_agent_output(str(tmp_path), "research-agent")
        assert result["decision"] == "approve"

    def test_blocks_implement_agent_without_output(self, tmp_path):
        self._setup_exp(tmp_path)
        result = check_agent_output(str(tmp_path), "implement-agent")
        assert result["decision"] == "block"

    def test_allows_implement_agent_with_output(self, tmp_path):
        exp_root = self._setup_exp(tmp_path)
        (exp_root / "results" / "implementation-manifest.json").write_text(
            '{"proposals": []}'
        )
        result = check_agent_output(str(tmp_path), "implementation-manifest.json")
        # Unknown agent name -- auto-approve
        assert result["decision"] == "approve"

    def test_allows_implement_agent_with_manifest(self, tmp_path):
        exp_root = self._setup_exp(tmp_path)
        (exp_root / "results" / "implementation-manifest.json").write_text(
            '{"proposals": []}'
        )
        result = check_agent_output(str(tmp_path), "implement-agent")
        assert result["decision"] == "approve"

    def test_blocks_report_agent_without_output(self, tmp_path):
        self._setup_exp(tmp_path)
        result = check_agent_output(str(tmp_path), "report-agent")
        assert result["decision"] == "block"

    def test_blocks_report_agent_missing_chart(self, tmp_path):
        exp_root = self._setup_exp(tmp_path)
        (exp_root / "reports" / "final-report.md").write_text("# Final Report")
        # Missing: progress_chart.png
        result = check_agent_output(str(tmp_path), "report-agent")
        assert result["decision"] == "block"
        assert "progress_chart" in result["reason"]

    def test_allows_report_agent_with_all_output(self, tmp_path):
        exp_root = self._setup_exp(tmp_path)
        (exp_root / "reports" / "final-report.md").write_text("# Final Report")
        (exp_root / "reports" / "progress_chart.png").write_text("PNG data")
        result = check_agent_output(str(tmp_path), "report-agent")
        assert result["decision"] == "approve"

    def test_allows_unknown_agent(self, tmp_path):
        self._setup_exp(tmp_path)
        result = check_agent_output(str(tmp_path), "unknown-agent")
        assert result["decision"] == "approve"

    def test_allows_monitor_agent(self, tmp_path):
        self._setup_exp(tmp_path)
        result = check_agent_output(str(tmp_path), "monitor-agent")
        assert result["decision"] == "approve"

    def test_blocks_experiment_agent_without_output(self, tmp_path):
        self._setup_exp(tmp_path)
        result = check_agent_output(str(tmp_path), "experiment-agent")
        assert result["decision"] == "block"

    def test_allows_experiment_agent_with_all_output(self, tmp_path):
        exp_root = self._setup_exp(tmp_path)
        # Result in round dir
        rd = exp_root / "results" / "round-1-hp"
        rd.mkdir(parents=True)
        (rd / "exp-001.json").write_text('{"exp_id": "exp-001"}')
        # Log (round-based)
        (exp_root / "logs" / "round-1-hp" / "exp-001").mkdir(parents=True)
        (exp_root / "logs" / "round-1-hp" / "exp-001" / "train.log").write_text("training log")
        # Script dir (round-based)
        (exp_root / "scripts" / "round-1-hp" / "exp-001").mkdir(parents=True)
        # Artifacts dir (round-based)
        (exp_root / "artifacts" / "round-1-hp" / "exp-001").mkdir(parents=True)
        result = check_agent_output(str(tmp_path), "experiment-agent")
        assert result["decision"] == "approve"

    def test_blocks_experiment_agent_missing_log(self, tmp_path):
        exp_root = self._setup_exp(tmp_path)
        rd = exp_root / "results" / "round-1-hp"
        rd.mkdir(parents=True)
        (rd / "exp-001.json").write_text('{"exp_id": "exp-001"}')
        (exp_root / "scripts" / "round-1-hp" / "exp-001").mkdir(parents=True)
        (exp_root / "artifacts" / "round-1-hp" / "exp-001").mkdir(parents=True)
        # Missing: logs/round-1-hp/exp-001/train.log
        result = check_agent_output(str(tmp_path), "experiment-agent")
        assert result["decision"] == "block"
        assert "train.log" in result["reason"]

    def test_blocks_tuning_agent_without_output(self, tmp_path):
        self._setup_exp(tmp_path)
        result = check_agent_output(str(tmp_path), "tuning-agent")
        assert result["decision"] == "block"

    def test_allows_tuning_agent_with_output(self, tmp_path):
        exp_root = self._setup_exp(tmp_path)
        pc = exp_root / "proposed-configs" / "round-1-hp"
        pc.mkdir(parents=True)
        (pc / "exp-001.json").write_text('{"exp_id": "exp-001", "config": {}}')
        result = check_agent_output(str(tmp_path), "tuning-agent")
        assert result["decision"] == "approve"

    def test_blocks_analysis_agent_without_report(self, tmp_path):
        self._setup_exp(tmp_path)
        result = check_agent_output(str(tmp_path), "analysis-agent")
        assert result["decision"] == "block"
        assert "batch-" in result["reason"] or "session-review" in result["reason"]

    def test_allows_analysis_agent_with_batch_report(self, tmp_path):
        exp_root = self._setup_exp(tmp_path)
        (exp_root / "reports" / "batch-1-analysis.md").write_text("# Batch 1")
        result = check_agent_output(str(tmp_path), "analysis-agent")
        assert result["decision"] == "approve"

    def test_allows_analysis_agent_with_session_review(self, tmp_path):
        exp_root = self._setup_exp(tmp_path)
        (exp_root / "reports" / "session-review.md").write_text("# Session Review")
        result = check_agent_output(str(tmp_path), "analysis-agent")
        assert result["decision"] == "approve"

    def test_graceful_without_breadcrumb(self, tmp_path):
        result = check_agent_output(str(tmp_path), "baseline-agent")
        assert result["decision"] == "approve"

    def test_strips_plugin_prefix(self, tmp_path):
        """Agent names with ml-optimizer: prefix are normalized."""
        self._setup_exp(tmp_path)
        result = check_agent_output(str(tmp_path), "ml-optimizer:monitor-agent")
        # monitor-agent after stripping prefix -> auto-approve (no expected outputs)
        assert result["decision"] == "approve"

    def test_block_reason_includes_missing_path(self, tmp_path):
        self._setup_exp(tmp_path)
        result = check_agent_output(str(tmp_path), "baseline-agent")
        assert result["decision"] == "block"
        assert "results/baseline.json" in result["reason"]
        assert "baseline-agent" in result["reason"]


class TestDevNotesAgentIdCheck:
    """SubagentStop blocks agents when last dev_notes.md entry's agent_id doesn't match."""

    def _setup(self, tmp_path):
        exp_root = tmp_path / "experiments"
        exp_root.mkdir(parents=True)
        (exp_root / "results").mkdir()
        (exp_root / "reports").mkdir()
        (tmp_path / ".claude").mkdir(exist_ok=True)
        (tmp_path / ".claude" / "ml-optimizer.json").write_text(
            json.dumps({"exp_root": str(exp_root)})
        )
        return exp_root

    def test_blocks_agent_without_dev_notes_update(self, tmp_path):
        exp_root = self._setup(tmp_path)
        agent_id = "test-agent-001"
        # Create dev_notes with NO entries — agent didn't append
        (exp_root / "dev_notes.md").write_text("# Dev Notes\n")
        # Create required output so contract check passes
        (exp_root / "results" / "prerequisites.json").write_text('{"status":"ready","dataset":{},"environment":{},"ready_for_baseline":true}')
        # Agent "forgot" to call dev_notes.py
        result = check_agent_output(str(tmp_path), "prerequisites-agent", agent_id=agent_id)
        assert result["decision"] == "block"
        assert "dev_notes" in result["reason"]

    def test_allows_agent_with_dev_notes_update(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from dev_notes import append as dev_notes_append
        exp_root = self._setup(tmp_path)
        agent_id = "test-agent-002"
        (exp_root / "dev_notes.md").write_text("# Dev Notes\n\n")
        (exp_root / "results" / "prerequisites.json").write_text('{"status":"ready","dataset":{},"environment":{},"ready_for_baseline":true}')
        # Agent uses the script with its agent_id
        dev_notes_append(str(exp_root), "prerequisites-agent", "All dependencies installed.", agent_id=agent_id)
        result = check_agent_output(str(tmp_path), "prerequisites-agent", agent_id=agent_id)
        assert result["decision"] == "approve"

    def test_blocks_wrong_agent_id(self, tmp_path):
        """Agent with wrong agent_id in the last entry is blocked."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from dev_notes import append as dev_notes_append
        exp_root = self._setup(tmp_path)
        (exp_root / "dev_notes.md").write_text("# Dev Notes\n\n")
        (exp_root / "results" / "prerequisites.json").write_text('{"status":"ready","dataset":{},"environment":{},"ready_for_baseline":true}')
        # Another agent wrote with a DIFFERENT agent_id
        dev_notes_append(str(exp_root), "prerequisites-agent", "Some other invocation.", agent_id="different-agent-id")
        # Current agent invocation checks — last entry has wrong agent_id
        result = check_agent_output(str(tmp_path), "prerequisites-agent", agent_id="my-agent-id")
        assert result["decision"] == "block"
        assert "agent_id" in result["reason"]

    def test_skips_without_agent_id(self, tmp_path):
        """Without agent_id, dev_notes check is skipped (backward compat)."""
        exp_root = self._setup(tmp_path)
        (exp_root / "dev_notes.md").write_text("# Dev Notes\n")
        (exp_root / "results" / "prerequisites.json").write_text('{"status":"ready","dataset":{},"environment":{},"ready_for_baseline":true}')
        # No agent_id passed
        result = check_agent_output(str(tmp_path), "prerequisites-agent")
        assert result["decision"] == "approve"

    def test_parallel_agents_same_type_different_ids(self, tmp_path):
        """Two experiment-agent invocations with different agent_ids — only the matching one passes."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from dev_notes import append as dev_notes_append
        exp_root = self._setup(tmp_path)
        (exp_root / "dev_notes.md").write_text("# Dev Notes\n\n")
        (exp_root / "results" / "prerequisites.json").write_text('{"status":"ready","dataset":{},"environment":{},"ready_for_baseline":true}')
        # Agent A writes first
        dev_notes_append(str(exp_root), "prerequisites-agent", "Agent A done.", agent_id="agent-A")
        # Agent B (different invocation) checks — last entry is A's, should BLOCK B
        result_b = check_agent_output(str(tmp_path), "prerequisites-agent", agent_id="agent-B")
        assert result_b["decision"] == "block"
        # Agent A checks — last entry matches A's agent_id, should APPROVE
        result_a = check_agent_output(str(tmp_path), "prerequisites-agent", agent_id="agent-A")
        assert result_a["decision"] == "approve"
