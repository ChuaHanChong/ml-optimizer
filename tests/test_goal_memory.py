"""Tests for goal_memory.py — goal anchoring and behavioral memory."""

import json
import threading
from pathlib import Path

import pytest

from conftest import FIXTURES

from goal_memory import (
    generate_summary,
    get_behaviors,
    init_goals,
    load_goals,
    log_behavior,
    main,
    sync_from_errors,
    update_goals,
    validate_agent_output,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_GOALS = json.loads((FIXTURES / "sample_goals.json").read_text())


def _minimal_goals(**overrides):
    """Return a minimal valid goals dict, with optional overrides."""
    g = {
        "objective": {"primary_metric": "accuracy", "lower_is_better": False},
        "constraints": {"scope_level": "training"},
        "divergence": {"metric": "loss", "lower_is_better": True},
    }
    g.update(overrides)
    return g


def _setup_goals(tmp_path, goals=None):
    """Create experiments dir and write goals."""
    exp = tmp_path / "experiments"
    exp.mkdir(exist_ok=True)
    init_goals(str(exp), goals or SAMPLE_GOALS)
    return str(exp)


def _run_cli(*args):
    """Run the CLI main() and return exit code."""
    return main(list(args))


# ---------------------------------------------------------------------------
# TestGoalInitialization
# ---------------------------------------------------------------------------


class TestGoalInitialization:
    """Tests for goal file initialization and loading."""

    def test_init_creates_file(self, tmp_path):
        """Initializing goals writes a readable optimization-goals.json."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        path = init_goals(exp, SAMPLE_GOALS)
        assert Path(path).is_file()
        data = json.loads(Path(path).read_text())
        assert data["objective"]["primary_metric"] == "accuracy"

    def test_required_fields_missing(self, tmp_path):
        """Initializing goals without required fields raises ValueError."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        with pytest.raises(ValueError, match="Missing required field"):
            init_goals(exp, {"objective": {"primary_metric": "loss"}})

    def test_read_missing(self, tmp_path):
        """Loading goals when the file is absent returns None."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        assert load_goals(exp) is None

    def test_read_corrupt(self, tmp_path):
        """Loading a corrupt goals file returns None."""
        exp = tmp_path / "experiments"
        exp.mkdir()
        (exp / "optimization-goals.json").write_text("{bad json")
        assert load_goals(str(exp)) is None


# ---------------------------------------------------------------------------
# TestGoalUpdates
# ---------------------------------------------------------------------------


class TestGoalUpdates:
    """Mid-run goal updates via update_goals()."""

    def test_update_target_value(self, tmp_path):
        """Partial update merges into existing goals."""
        init_goals(str(tmp_path), SAMPLE_GOALS)
        result = update_goals(str(tmp_path), {"objective": {"target_value": 85.0}})
        assert result["updated"] is True
        assert len(result["changes"]) == 1
        assert "85.0" in result["changes"][0]
        goals = load_goals(str(tmp_path))
        assert goals["objective"]["target_value"] == 85.0
        assert goals["objective"]["primary_metric"] == "accuracy"  # unchanged

    def test_update_frozen_parameters(self, tmp_path):
        """Updating frozen_parameters merges the new constraint into goals."""
        init_goals(str(tmp_path), SAMPLE_GOALS)
        result = update_goals(str(tmp_path), {"constraints": {"frozen_parameters": ["lr"]}})
        assert result["updated"] is True
        assert load_goals(str(tmp_path))["constraints"]["frozen_parameters"] == ["lr"]

    def test_update_primary_metric(self, tmp_path):
        """Updating the primary metric and its polarity is persisted."""
        init_goals(str(tmp_path), SAMPLE_GOALS)
        result = update_goals(str(tmp_path), {
            "objective": {"primary_metric": "f1", "lower_is_better": False}
        })
        assert result["updated"] is True
        goals = load_goals(str(tmp_path))
        assert goals["objective"]["primary_metric"] == "f1"
        assert goals["objective"]["lower_is_better"] is False

    def test_update_no_changes(self, tmp_path):
        """A no-op update reports not-updated with an explanatory error."""
        init_goals(str(tmp_path), SAMPLE_GOALS)
        goals = load_goals(str(tmp_path))
        result = update_goals(str(tmp_path), {
            "objective": {"primary_metric": goals["objective"]["primary_metric"]}
        })
        assert result["updated"] is False
        assert "No changes" in result["error"]

    def test_update_logs_to_behaviors(self, tmp_path):
        """A goal update records a goal_update behavior entry."""
        init_goals(str(tmp_path), SAMPLE_GOALS)
        update_goals(str(tmp_path), {"objective": {"target_value": 80.0}})
        behaviors = get_behaviors(str(tmp_path), category="goal_update")
        assert len(behaviors) == 1
        assert "changes" in behaviors[0]

    def test_update_no_goals_file(self, tmp_path):
        """Updating goals with no existing file reports not-updated."""
        result = update_goals(str(tmp_path), {"objective": {"target_value": 80.0}})
        assert result["updated"] is False
        assert "No optimization-goals.json" in result["error"]

    def test_update_adds_timestamp(self, tmp_path):
        """A goal update stamps the goals with an updated_at field."""
        init_goals(str(tmp_path), SAMPLE_GOALS)
        update_goals(str(tmp_path), {"objective": {"target_value": 80.0}})
        goals = load_goals(str(tmp_path))
        assert "updated_at" in goals


# ---------------------------------------------------------------------------
# TestBehaviorLogging
# ---------------------------------------------------------------------------


class TestBehaviorLogging:
    """Tests for logging and querying learned behaviors."""

    def test_log_and_query(self, tmp_path):
        """Core test: log a behavior, query it back, verify fields + auto-timestamp."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        log_behavior(exp, "hp_constraint", {
            "parameter": "lr", "constraint_type": "upper_bound", "value": 0.01
        })
        items = get_behaviors(exp, category="hp_constraint")
        assert len(items) == 1
        assert items[0]["parameter"] == "lr"
        assert "timestamp" in items[0]

    @pytest.mark.parametrize("category,entry", [
        ("method_outcome", {"method": "mixup", "outcome": "improved"}),
        ("divergence_pattern", {"pattern": "high_lr"}),
        ("resource_constraint", {"max_batch_size": 128}),
        ("training_insight", {"insight": "warmup helps"}),
        ("scope_violation", {"agent": "hp-tune", "detail": "froze batch_size"}),
    ])
    def test_log_all_categories(self, tmp_path, category, entry):
        """Each valid behavior category can be logged and queried back."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        log_behavior(exp, category, entry)
        items = get_behaviors(exp, category=category)
        assert len(items) == 1

    def test_invalid_category(self, tmp_path):
        """Logging an unknown behavior category raises ValueError."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        with pytest.raises(ValueError, match="Invalid category"):
            log_behavior(exp, "bad_category", {"foo": "bar"})

    def test_concurrent_logging(self, tmp_path):
        """4 threads x 5 entries = 20 total; verify none lost."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()

        def writer(thread_id):
            """Log five training-insight behaviors for one thread."""
            for i in range(5):
                log_behavior(exp, "training_insight", {
                    "insight": f"thread-{thread_id}-{i}"
                })

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        items = get_behaviors(exp, category="training_insight")
        assert len(items) == 20

    def test_query_all_categories(self, tmp_path):
        """Querying with no category returns behaviors across all categories."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        log_behavior(exp, "hp_constraint", {"parameter": "lr"})
        log_behavior(exp, "training_insight", {"insight": "test"})
        items = get_behaviors(exp)
        assert len(items) == 2

    def test_query_recent(self, tmp_path):
        """The recent parameter limits a query to the most recent N behaviors."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        for i in range(5):
            log_behavior(exp, "training_insight", {"insight": f"item-{i}"})
        items = get_behaviors(exp, category="training_insight", recent=2)
        assert len(items) == 2

    def test_query_empty(self, tmp_path):
        """Querying behaviors with none logged returns an empty list."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        assert get_behaviors(exp) == []

    def test_load_corrupt_behaviors(self, tmp_path):
        """Corrupt learned-behaviors.json returns empty structure."""
        exp = tmp_path / "experiments"
        exp.mkdir()
        (exp / "learned-behaviors.json").write_text("{bad json")
        items = get_behaviors(str(exp))
        assert items == []

    def test_load_non_dict_behaviors(self, tmp_path):
        """Non-dict learned-behaviors.json returns empty structure."""
        exp = tmp_path / "experiments"
        exp.mkdir()
        (exp / "learned-behaviors.json").write_text('"just a string"')
        items = get_behaviors(str(exp))
        assert items == []


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for agent output validation against goals and learned behaviors."""

    def test_hp_tune_frozen_param(self, tmp_path):
        """A hp-tune config touching a frozen parameter is a violation."""
        exp = _setup_goals(tmp_path)
        result = validate_agent_output(exp, "hp-tune", {
            "configs": [{"batch_size": 64, "lr": 0.001}]
        })
        assert result["valid"] is False
        assert any("frozen parameter" in v for v in result["violations"])

    def test_hp_tune_oom_limit(self, tmp_path):
        """A hp-tune batch size above the learned OOM limit is a violation."""
        exp = _setup_goals(tmp_path)
        log_behavior(exp, "resource_constraint", {"max_batch_size": 128})
        result = validate_agent_output(exp, "hp-tune", {
            "configs": [{"lr": 0.001, "batch_size": 256}]
        })
        assert result["valid"] is False
        assert any("OOM limit" in v for v in result["violations"])

    def test_hp_tune_hp_bound_is_warning_not_violation(self, tmp_path):
        """Exceeding a learned HP bound warns but does not invalidate the output."""
        exp = _setup_goals(tmp_path)
        log_behavior(exp, "hp_constraint", {
            "parameter": "lr", "constraint_type": "upper_bound", "value": 0.01
        })
        result = validate_agent_output(exp, "hp-tune", {
            "configs": [{"lr": 0.05}]
        })
        assert result["valid"] is True
        assert any("learned bound" in w for w in result["warnings"])

    def test_hp_tune_valid(self, tmp_path):
        """A hp-tune config within all constraints is valid."""
        exp = _setup_goals(tmp_path)
        result = validate_agent_output(exp, "hp-tune", {
            "configs": [{"lr": 0.001}]
        })
        assert result["valid"] is True

    def test_research_scope_violation(self, tmp_path):
        """An architecture proposal under training scope is a violation."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        init_goals(exp, _minimal_goals())
        result = validate_agent_output(exp, "research", {
            "proposals": [{"name": "ViT", "scope": "architecture"}]
        })
        assert result["valid"] is False
        assert any("scope_level='training'" in v for v in result["violations"])

    def test_research_dead_end(self, tmp_path):
        """Re-proposing a cataloged dead-end technique is a violation."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        init_goals(exp, _minimal_goals())
        reports = Path(exp) / "reports"
        reports.mkdir()
        (reports / "dead-ends.json").write_text(json.dumps({
            "dead_ends": [{"technique": "focal-loss", "reason": "5% worse"}]
        }))
        result = validate_agent_output(exp, "research", {
            "proposals": [{"name": "focal-loss", "type": "code_change"}]
        })
        assert result["valid"] is False
        assert any("dead-end" in v for v in result["violations"])

    def test_research_valid(self, tmp_path):
        """An in-scope, non-dead-end research proposal is valid."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        init_goals(exp, _minimal_goals())
        result = validate_agent_output(exp, "research", {
            "proposals": [{"name": "cosine_annealing", "type": "hp_only"}]
        })
        assert result["valid"] is True

    def test_research_code_change_training_scope_allowed(self, tmp_path):
        """code_change on training files should NOT be rejected (regression test)."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        init_goals(exp, _minimal_goals())
        result = validate_agent_output(exp, "research", {
            "proposals": [
                {"name": "label_smoothing", "type": "code_change",
                 "files_to_modify": ["train.py", "loss.py"]},
            ]
        })
        assert result["valid"] is True

    def test_research_code_change_arch_files_warns(self, tmp_path):
        """code_change modifying model files in training scope → warning."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        init_goals(exp, _minimal_goals())
        result = validate_agent_output(exp, "research", {
            "proposals": [
                {"name": "attention_replacement", "type": "code_change",
                 "files_to_modify": ["model.py", "train.py"]},
            ]
        })
        assert result["valid"] is True
        assert any("architecture files" in w for w in result["warnings"])

    def test_analyze_metric_mismatch(self, tmp_path):
        """An analyze metric differing from the goal metric is a violation."""
        exp = _setup_goals(tmp_path)
        result = validate_agent_output(exp, "analyze", {
            "primary_metric": "loss", "lower_is_better": True
        })
        assert result["valid"] is False
        assert any("Metric mismatch" in v for v in result["violations"])

    def test_analyze_polarity_mismatch(self, tmp_path):
        """An analyze polarity differing from the goal polarity is a violation."""
        exp = _setup_goals(tmp_path)
        result = validate_agent_output(exp, "analyze", {
            "primary_metric": "accuracy", "lower_is_better": True
        })
        assert result["valid"] is False
        assert any("Polarity mismatch" in v for v in result["violations"])

    def test_analyze_valid(self, tmp_path):
        """An analyze output matching the goal metric and polarity is valid."""
        exp = _setup_goals(tmp_path)
        result = validate_agent_output(exp, "analyze", {
            "primary_metric": "accuracy", "lower_is_better": False
        })
        assert result["valid"] is True

    def test_implement_scope_warning(self, tmp_path):
        """Modifying model files under training scope warns but stays valid."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        init_goals(exp, _minimal_goals())
        result = validate_agent_output(exp, "implement", {
            "files_modified": ["model.py", "train.py"]
        })
        assert result["valid"] is True
        assert any("architecture changes" in w for w in result["warnings"])

    def test_experiment_frozen_param(self, tmp_path):
        """An experiment config touching a frozen parameter is a violation."""
        exp = _setup_goals(tmp_path)
        result = validate_agent_output(exp, "experiment", {
            "config": {"batch_size": 64, "lr": 0.001}
        })
        assert result["valid"] is False
        assert any("frozen parameter" in v for v in result["violations"])

    def test_experiment_oom_limit(self, tmp_path):
        """An experiment batch size above the learned OOM limit is a violation."""
        exp = _setup_goals(tmp_path)
        log_behavior(exp, "resource_constraint", {"max_batch_size": 128})
        result = validate_agent_output(exp, "experiment", {
            "config": {"lr": 0.001, "batch_size": 256}
        })
        assert result["valid"] is False
        assert any("OOM limit" in v for v in result["violations"])

    def test_hp_tune_flat_config(self, tmp_path):
        """Flat dict with 'lr' key is treated as a single config."""
        exp = _setup_goals(tmp_path)
        result = validate_agent_output(exp, "hp-tune", {"lr": 0.001})
        assert result["valid"] is True

    def test_hp_tune_non_dict_in_list(self, tmp_path):
        """Non-dict items in configs list are skipped, not crashed."""
        exp = _setup_goals(tmp_path)
        result = validate_agent_output(exp, "hp-tune", {
            "configs": ["not_a_dict", {"lr": 0.001}]
        })
        assert result["valid"] is True

    def test_hp_tune_bad_batch_size_type(self, tmp_path):
        """String batch_size doesn't crash OOM check."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        init_goals(exp, _minimal_goals())  # no frozen params
        log_behavior(exp, "resource_constraint", {"max_batch_size": 128})
        result = validate_agent_output(exp, "hp-tune", {
            "configs": [{"lr": 0.001, "batch_size": "not_a_number"}]
        })
        assert result["valid"] is True  # bad type is skipped, not violation

    def test_implement_full_scope(self, tmp_path):
        """Full scope returns no warnings even for model files."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        init_goals(exp, _minimal_goals(constraints={"scope_level": "full"}))
        result = validate_agent_output(exp, "implement", {
            "files_modified": ["model.py", "backbone.py"]
        })
        assert result["valid"] is True
        assert result["warnings"] == []

    def test_experiment_non_dict_config(self, tmp_path):
        """Non-dict config returns early, no crash."""
        exp = _setup_goals(tmp_path)
        result = validate_agent_output(exp, "experiment", {
            "config": "not_a_dict"
        })
        assert result["valid"] is True

    def test_research_non_list_proposals(self, tmp_path):
        """Non-list proposals returns early, no crash."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        init_goals(exp, _minimal_goals())
        result = validate_agent_output(exp, "research", {
            "proposals": "not_a_list"
        })
        assert result["valid"] is True

    def test_implement_non_list_changes(self, tmp_path):
        """Non-list files_modified returns early, no crash."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        init_goals(exp, _minimal_goals())
        result = validate_agent_output(exp, "implement", {
            "files_modified": "not_a_list"
        })
        assert result["valid"] is True

    def test_hp_tune_bad_lr_bound_type(self, tmp_path):
        """Non-numeric lr value doesn't crash HP bound check."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        init_goals(exp, _minimal_goals())
        log_behavior(exp, "hp_constraint", {
            "parameter": "lr", "constraint_type": "upper_bound", "value": 0.01
        })
        result = validate_agent_output(exp, "hp-tune", {
            "configs": [{"lr": "bad_value"}]
        })
        assert result["valid"] is True  # bad type skipped

    def test_no_goals_file(self, tmp_path):
        """Validation without a goals file passes but warns it is missing."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        result = validate_agent_output(exp, "hp-tune", {"configs": [{"lr": 0.1}]})
        assert result["valid"] is True
        assert any("No optimization-goals" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# TestSyncFromErrors
# ---------------------------------------------------------------------------


class TestSyncFromErrors:
    """Tests for syncing learned behaviors from the error log and dead-end catalog."""

    def test_sync_oom_events(self, tmp_path):
        """OOM error events sync into a resource_constraint behavior."""
        exp = tmp_path / "experiments"
        exp.mkdir()
        reports = exp / "reports"
        reports.mkdir()
        (reports / "error-log.json").write_text(json.dumps({
            "events": [{
                "event_id": "e1", "timestamp": "2026-01-01T00:00:00Z",
                "category": "training_failure", "severity": "critical",
                "source": "experiment", "message": "CUDA out of memory (OOM)",
                "config": {"batch_size": 256, "lr": 0.001},
            }],
            "summary": {},
        }))
        result = sync_from_errors(str(exp))
        assert result["synced"] >= 1
        items = get_behaviors(str(exp), category="resource_constraint")
        assert any(rc.get("max_batch_size") == 256 for rc in items)

    def test_sync_divergence_events(self, tmp_path):
        """Divergence events sync into a learning-rate hp_constraint behavior."""
        exp = tmp_path / "experiments"
        exp.mkdir()
        reports = exp / "reports"
        reports.mkdir()
        (reports / "error-log.json").write_text(json.dumps({
            "events": [
                {"event_id": "e1", "timestamp": "2026-01-01T00:00:00Z",
                 "category": "divergence", "severity": "warning",
                 "source": "monitor", "message": "NaN detected",
                 "config": {"lr": 0.05}},
                {"event_id": "e2", "timestamp": "2026-01-01T00:01:00Z",
                 "category": "divergence", "severity": "warning",
                 "source": "monitor", "message": "NaN detected",
                 "config": {"lr": 0.1}},
            ],
            "summary": {},
        }))
        result = sync_from_errors(str(exp))
        assert result["synced"] >= 1
        items = get_behaviors(str(exp), category="hp_constraint")
        assert any(hc.get("parameter") == "lr" for hc in items)

    def test_sync_dead_ends(self, tmp_path):
        """Dead-end catalog entries sync into method_outcome behaviors."""
        exp = tmp_path / "experiments"
        exp.mkdir()
        reports = exp / "reports"
        reports.mkdir()
        (reports / "dead-ends.json").write_text(json.dumps({
            "dead_ends": [{"technique": "mixup", "reason": "no improvement"}]
        }))
        (reports / "error-log.json").write_text(json.dumps({
            "events": [], "summary": {}
        }))
        result = sync_from_errors(str(exp))
        assert result["synced"] >= 1
        items = get_behaviors(str(exp), category="method_outcome")
        assert any(mo.get("method") == "mixup" for mo in items)

    def test_sync_oom_already_exists(self, tmp_path):
        """OOM sync skips batch_size already in behaviors."""
        exp = tmp_path / "experiments"
        exp.mkdir()
        reports = exp / "reports"
        reports.mkdir()
        # Pre-populate a resource constraint
        log_behavior(str(exp), "resource_constraint", {
            "max_batch_size": 256, "source": "manual"
        })
        # Log OOM with same batch_size
        (reports / "error-log.json").write_text(json.dumps({
            "events": [{
                "event_id": "e1", "timestamp": "2026-01-01T00:00:00Z",
                "category": "training_failure", "severity": "critical",
                "source": "experiment", "message": "CUDA out of memory (OOM)",
                "config": {"batch_size": 256},
            }],
            "summary": {},
        }))
        result = sync_from_errors(str(exp))
        assert result["skipped"] >= 1

    def test_sync_divergence_already_exists(self, tmp_path):
        """Divergence sync skips LR bound already in behaviors."""
        exp = tmp_path / "experiments"
        exp.mkdir()
        reports = exp / "reports"
        reports.mkdir()
        # Pre-populate hp constraint
        log_behavior(str(exp), "hp_constraint", {
            "parameter": "lr", "value": 0.05, "source": "sync_from_errors"
        })
        (reports / "error-log.json").write_text(json.dumps({
            "events": [
                {"event_id": "e1", "timestamp": "2026-01-01T00:00:00Z",
                 "category": "divergence", "severity": "warning",
                 "source": "monitor", "message": "NaN",
                 "config": {"lr": 0.05}},
            ],
            "summary": {},
        }))
        result = sync_from_errors(str(exp))
        assert result["skipped"] >= 1

    def test_sync_idempotent(self, tmp_path):
        """Running sync twice does not duplicate already-synced behaviors."""
        exp = tmp_path / "experiments"
        exp.mkdir()
        reports = exp / "reports"
        reports.mkdir()
        (reports / "dead-ends.json").write_text(json.dumps({
            "dead_ends": [{"technique": "mixup", "reason": "no improvement"}]
        }))
        (reports / "error-log.json").write_text(json.dumps({
            "events": [], "summary": {}
        }))
        sync_from_errors(str(exp))
        result = sync_from_errors(str(exp))
        assert result["skipped"] >= 1
        items = get_behaviors(str(exp), category="method_outcome")
        assert len([mo for mo in items if mo.get("method") == "mixup"]) == 1


# ---------------------------------------------------------------------------
# TestSummary
# ---------------------------------------------------------------------------


class TestSummary:
    """Tests for the compact goals-and-behaviors summary briefing."""

    def test_full_summary(self, tmp_path):
        """The summary includes goals, learned constraints, and what works."""
        exp = _setup_goals(tmp_path)
        log_behavior(exp, "hp_constraint", {
            "parameter": "lr", "constraint_type": "upper_bound",
            "value": 0.01, "evidence_count": 4
        })
        log_behavior(exp, "method_outcome", {
            "method": "perceptual-loss", "outcome": "improved",
            "improvement_pct": 12.14
        })
        summary = generate_summary(exp)
        assert "OPTIMIZATION GOALS" in summary
        assert "accuracy" in summary
        assert "LEARNED CONSTRAINTS" in summary
        assert "WHAT WORKS" in summary
        assert "perceptual-loss" in summary

    def test_summary_all_sections(self, tmp_path):
        """Summary with all behavior categories populated."""
        exp = _setup_goals(tmp_path)
        log_behavior(exp, "hp_constraint", {
            "parameter": "lr", "constraint_type": "upper_bound",
            "value": 0.01, "evidence_count": 4, "reason": "diverged above this"
        })
        log_behavior(exp, "resource_constraint", {
            "max_batch_size": 128, "notes": "OOM at 256"
        })
        log_behavior(exp, "method_outcome", {
            "method": "perceptual-loss", "outcome": "improved",
            "improvement_pct": 12.14, "hp_sensitivity": "best at lr=0.0005"
        })
        log_behavior(exp, "method_outcome", {
            "method": "focal-loss", "outcome": "dead_end",
            "reason": "5% worse than baseline"
        })
        log_behavior(exp, "divergence_pattern", {
            "description": "LR > 0.01 causes NaN within 50 steps"
        })
        log_behavior(exp, "training_insight", {
            "insight": "warmup=500 helps convergence"
        })
        summary = generate_summary(exp)
        assert "LEARNED CONSTRAINTS" in summary
        assert "diverged above this" in summary
        assert "OOM" in summary
        assert "DEAD ENDS" in summary
        assert "focal-loss" in summary
        assert "WHAT WORKS" in summary
        assert "perceptual-loss" in summary
        assert "DIVERGENCE PATTERNS" in summary
        assert "NaN" in summary
        assert "warmup" in summary

    def test_no_goals(self, tmp_path):
        """The summary warns when no goals file is present."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        summary = generate_summary(exp)
        assert "WARNING" in summary

    def test_includes_dead_ends(self, tmp_path):
        """The summary surfaces cataloged dead ends."""
        exp = _setup_goals(tmp_path)
        reports = Path(exp) / "reports"
        reports.mkdir()
        (reports / "dead-ends.json").write_text(json.dumps({
            "dead_ends": [{"technique": "focal-loss", "reason": "5% worse"}]
        }))
        summary = generate_summary(exp)
        assert "DEAD ENDS" in summary
        assert "focal-loss" in summary


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests for the goal_memory.py CLI interface."""

    def test_cli_init_goals(self, tmp_path):
        """The init-goals CLI writes a loadable goals file."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        code = _run_cli(exp, "init-goals", json.dumps(SAMPLE_GOALS))
        assert code == 0
        assert load_goals(exp) is not None

    def test_cli_validate_output(self, tmp_path):
        """The validate-output CLI exits zero for a compliant output."""
        exp = _setup_goals(tmp_path)
        code = _run_cli(exp, "validate-output", "hp-tune",
                        json.dumps({"configs": [{"lr": 0.001}]}))
        assert code == 0

    def test_cli_validate_output_violation(self, tmp_path):
        """The validate-output CLI exits 2 when a violation is found."""
        exp = _setup_goals(tmp_path)
        code = _run_cli(exp, "validate-output", "hp-tune",
                        json.dumps({"configs": [{"batch_size": 64}]}))
        assert code == 2  # exit code 2 = violations

    def test_cli_no_args(self):
        """The CLI with no arguments exits with code 1."""
        assert _run_cli() == 1

    def test_cli_missing_args(self, tmp_path):
        """CLI subcommands missing required arguments exit with code 1."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        assert _run_cli(exp, "init-goals") == 1
        assert _run_cli(exp, "log-behavior") == 1
        assert _run_cli(exp, "validate-output") == 1

    def test_cli_invalid_json(self, tmp_path):
        """The init-goals CLI exits with code 1 on malformed JSON."""
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        assert _run_cli(exp, "init-goals", "{bad json") == 1


# ---------------------------------------------------------------------------
# TestRLScopeHeuristics (Batch E, Task 23)
# ---------------------------------------------------------------------------


def _rl_goals(scope):
    return {
        "objective": {"primary_metric": "episode_return", "lower_is_better": False},
        "constraints": {"scope_level": scope},
    }


class TestRLScopeHeuristics:
    """Env/reward file-pattern heuristics: env dynamics files are full-scope only."""

    def _exp(self, tmp_path, scope):
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        init_goals(exp, _rl_goals(scope))
        return exp

    def test_research_env_file_warns_at_training_scope(self, tmp_path):
        exp = self._exp(tmp_path, "training")
        result = validate_agent_output(exp, "research", {"proposals": [
            {"name": "harder curriculum", "type": "code_change", "scope": "training",
             "files_to_modify": ["envs/cartpole_env.py"]},
        ]})
        assert result["valid"] is True  # heuristic = warning, never violation
        assert any("env-dynamics" in w for w in result["warnings"])

    def test_research_env_file_warns_at_architecture_scope(self, tmp_path):
        exp = self._exp(tmp_path, "architecture")
        result = validate_agent_output(exp, "research", {"proposals": [
            {"name": "domain randomization", "type": "code_change", "scope": "architecture",
             "files_to_modify": ["tasks/domain_rand.py"]},
        ]})
        assert result["valid"] is True
        assert any("env-dynamics" in w for w in result["warnings"])

    def test_implement_policy_file_warns_at_training_scope(self, tmp_path):
        exp = self._exp(tmp_path, "training")
        result = validate_agent_output(exp, "implement", {
            "files_modified": ["agents/policy_net.py"]})
        assert result["valid"] is True
        assert any("architecture changes" in w for w in result["warnings"])

    def test_implement_env_file_warns_at_architecture_scope(self, tmp_path):
        exp = self._exp(tmp_path, "architecture")
        result = validate_agent_output(exp, "implement", {
            "files_modified": ["tasks/curriculum.py"]})
        assert result["valid"] is True
        assert any("env-dynamics" in w for w in result["warnings"])

    def test_implement_env_file_ok_at_full_scope(self, tmp_path):
        exp = self._exp(tmp_path, "full")
        result = validate_agent_output(exp, "implement", {
            "files_modified": ["envs/cartpole_env.py", "tasks/curriculum.py"]})
        assert result["valid"] is True
        assert result["warnings"] == []


class TestDomainRandomization:
    """DR center/width parameterization and its scope gate."""

    def test_dr_params_needs_both_center_and_width(self):
        from goal_memory import dr_params
        assert dr_params({"friction_center": 1.0, "friction_width": 0.4}) == ["friction"]
        # a lone _center or _width is not a DR pair
        assert dr_params({"friction_center": 1.0}) == []
        assert dr_params({"hidden_width": 64}) == []

    def test_dr_params_finds_multiple_sorted(self):
        from goal_memory import dr_params
        cfg = {"mass_center": 2.0, "mass_width": 0.5, "friction_center": 1.0, "friction_width": 0.4}
        assert dr_params(cfg) == ["friction", "mass"]

    def test_effective_range_never_inverted(self):
        from goal_memory import dr_effective_range
        lo, hi = dr_effective_range({"friction_center": 1.0, "friction_width": 0.4}, "friction")
        assert (lo, hi) == pytest.approx((0.8, 1.2))
        # width 0 collapses to a point, still ordered
        lo, hi = dr_effective_range({"friction_center": 1.0, "friction_width": 0.0}, "friction")
        assert lo == hi == pytest.approx(1.0)
        # negative width still yields a non-inverted range (the abs() guarantee)
        lo, hi = dr_effective_range({"friction_center": 1.0, "friction_width": -0.4}, "friction")
        assert (lo, hi) == pytest.approx((0.8, 1.2))

    def test_effective_range_none_when_absent_or_bad_type(self):
        from goal_memory import dr_effective_range
        assert dr_effective_range({"friction_center": 1.0}, "friction") is None
        assert dr_effective_range({"friction_center": "x", "friction_width": 0.4}, "friction") is None

    def test_dr_tuning_blocked_at_training_scope(self, tmp_path):
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        init_goals(exp, {
            "objective": {"primary_metric": "reward", "lower_is_better": False},
            "constraints": {"scope_level": "training"},
        })
        result = validate_agent_output(exp, "hp-tune", {
            "configs": [{"lr": 0.001, "friction_center": 1.0, "friction_width": 0.4}]
        })
        assert result["valid"] is False
        assert any("friction" in v and "scope" in v.lower() for v in result["violations"])

    def test_dr_tuning_allowed_at_architecture_scope(self, tmp_path):
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        init_goals(exp, {
            "objective": {"primary_metric": "reward", "lower_is_better": False},
            "constraints": {"scope_level": "architecture"},
        })
        result = validate_agent_output(exp, "hp-tune", {
            "configs": [{"lr": 0.001, "friction_center": 1.0, "friction_width": 0.4}]
        })
        assert result["valid"] is True

    def test_non_dr_config_unaffected_at_training_scope(self, tmp_path):
        exp = str(tmp_path / "experiments")
        Path(exp).mkdir()
        init_goals(exp, {
            "objective": {"primary_metric": "reward", "lower_is_better": False},
            "constraints": {"scope_level": "training"},
        })
        result = validate_agent_output(exp, "hp-tune", {"configs": [{"lr": 0.001}]})
        assert result["valid"] is True
