"""Consolidated tests for error_tracker.py and review workflow."""

import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

from conftest import FIXTURES, _write_result

from error_tracker import (
    _atomic_write_json,
    _cli_main,
    _load_results,
    _normalize_technique,
    add_agenda_idea,
    cleanup_memory,
    compute_proposal_outcomes,
    compute_success_metrics,
    create_event,
    detect_cross_project_patterns,
    detect_patterns,
    get_agenda,
    get_dead_ends,
    get_events,
    get_suggestion_history,
    init_agenda,
    is_dead_end,
    load_cross_project,
    load_error_log,
    log_dead_end,
    log_event,
    log_suggestion,
    rank_suggestions,
    summarize_session,
    update_agenda_item,
    update_cross_project,
    validate_event,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_full_session(tmp_path):
    """Create a realistic session with mixed results and errors."""
    exp_root = tmp_path / "experiments"
    exp_root.mkdir()
    results = exp_root / "results"
    results.mkdir()
    _write_result(results, "baseline", "completed",
                  {"lr": 0.001, "batch_size": 64}, {"acc": 70.0})
    _write_result(results, "exp-001", "completed",
                  {"lr": 0.0005, "batch_size": 64}, {"acc": 78.0},
                  code_branch="ml-opt/perceptual-loss", duration_seconds=3600)
    _write_result(results, "exp-002", "completed",
                  {"lr": 0.001, "batch_size": 32}, {"acc": 72.5},
                  duration_seconds=1800)
    _write_result(results, "exp-003", "failed",
                  {"lr": 0.1, "batch_size": 256}, {},
                  duration_seconds=120)
    _write_result(results, "exp-004", "diverged",
                  {"lr": 0.05, "batch_size": 256}, {},
                  duration_seconds=60)
    manifest = {
        "original_branch": "main",
        "strategy": "git_branch",
        "proposals": [
            {"name": "perceptual-loss", "slug": "perceptual-loss",
             "status": "validated", "branch": "ml-opt/perceptual-loss"},
            {"name": "mixup-aug", "slug": "mixup-aug",
             "status": "validation_failed"},
        ],
    }
    (results / "implementation-manifest.json").write_text(json.dumps(manifest))
    configs_dir = results / "proposed-configs"
    configs_dir.mkdir()
    for i in range(4):
        (configs_dir / f"exp-{i+1:03d}.json").write_text(
            json.dumps({"lr": 0.001 * (i + 1)}))
    log_event(str(exp_root), create_event(
        "training_failure", "critical", "experiment", "GPU OOM",
        exp_id="exp-003", config={"lr": 0.1, "batch_size": 256},
        context={"error_type": "oom"}, iteration=1))
    log_event(str(exp_root), create_event(
        "divergence", "warning", "monitor", "NaN at step 5",
        exp_id="exp-004", config={"lr": 0.05, "batch_size": 64}, iteration=1))
    log_event(str(exp_root), create_event(
        "divergence", "warning", "monitor", "Explosion at step 10",
        exp_id="exp-005", config={"lr": 0.08, "batch_size": 32}, iteration=1))
    log_event(str(exp_root), create_event(
        "divergence", "warning", "monitor", "NaN at step 2",
        exp_id="exp-006", config={"lr": 0.2, "batch_size": 64}, iteration=1))
    log_event(str(exp_root), create_event(
        "pipeline_inefficiency", "warning", "analyze",
        "All 3 experiments in batch 2 diverged",
        context={"experiments_wasted": 3}, iteration=2))
    log_event(str(exp_root), create_event(
        "pipeline_inefficiency", "warning", "analyze",
        "All 2 experiments in batch 3 diverged",
        context={"experiments_wasted": 2}, iteration=3))
    log_event(str(exp_root), create_event(
        "research_failure", "warning", "research",
        "No relevant results for query: CIFAR optimization",
        phase=4, context={"query": "CIFAR optimization"}))
    log_event(str(exp_root), create_event(
        "resource_error", "info", "baseline",
        "GPU profiling failed", phase=2))
    return exp_root


# ===========================================================================
# TestEventCreation
# ===========================================================================


class TestEventCreation:
    """Tests for create_event and validate_event."""

    def test_create_event_fields(self):
        """create_event produces valid events with required and optional fields."""
        ev = create_event("training_failure", "critical", "experiment", "OOM")
        assert ev["category"] == "training_failure"
        assert ev["severity"] == "critical"
        assert ev["event_id"].startswith("err-")
        datetime.fromisoformat(ev["timestamp"])

        # Optional fields
        ev2 = create_event(
            "divergence", "warning", "monitor", "NaN at step 5",
            exp_id="exp-003", phase=5, iteration=2,
            config={"lr": 0.1}, context={"step": 5},
        )
        assert ev2["exp_id"] == "exp-003"
        assert ev2["phase"] == 5
        assert ev2["config"] == {"lr": 0.1}

        # Unique IDs
        assert ev["event_id"] != ev2["event_id"]

    def test_duration_field(self):
        """duration_seconds included when provided, omitted otherwise."""
        ev = create_event("timeout", "critical", "orchestrate", "Timed out",
                          duration_seconds=3600.5)
        assert ev["duration_seconds"] == 3600.5
        ev2 = create_event("agent_failure", "critical", "orchestrate", "crash")
        assert "duration_seconds" not in ev2

    @pytest.mark.parametrize("category", ["research_failure", "timeout", "resource_error"])
    def test_new_categories_valid(self, category):
        ev = create_event(category, "warning", "orchestrate", "msg")
        assert validate_event(ev)["valid"] is True

    def test_validate_event(self):
        """Valid event passes, missing-field/non-dict/empty fail."""
        valid = create_event("agent_failure", "critical", "orchestrate", "timeout")
        assert validate_event(valid)["valid"] is True
        assert validate_event(valid)["errors"] == []

        missing = {"category": "agent_failure", "severity": "critical"}
        assert validate_event(missing)["valid"] is False

        assert validate_event("not a dict")["valid"] is False
        assert validate_event({})["valid"] is False

    @pytest.mark.parametrize("field,args", [
        ("category", ("unknown_cat", "critical", "experiment", "x")),
        ("severity", ("agent_failure", "fatal", "orchestrate", "x")),
        ("source", ("agent_failure", "critical", "unknown_src", "x")),
    ], ids=["bad_category", "bad_severity", "bad_source"])
    def test_invalid_field_raises(self, field, args):
        with pytest.raises(ValueError, match=field):
            create_event(*args)


# ===========================================================================
# TestEventLogging
# ===========================================================================


class TestEventLogging:
    """Tests for log_event, load_error_log, get_events."""

    def test_log_creates_appends_and_updates_summary(self, tmp_path):
        """log_event creates file, appends events, and updates summary counts."""
        ev1 = create_event("training_failure", "critical", "experiment", "first")
        path = log_event(str(tmp_path), ev1)
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert len(data["events"]) == 1

        ev2 = create_event("divergence", "warning", "monitor", "second")
        log_event(str(tmp_path), ev2)
        ev3 = create_event("training_failure", "warning", "experiment", "third")
        log_event(str(tmp_path), ev3)
        data = load_error_log(str(tmp_path))
        assert data["summary"]["total_events"] == 3
        assert data["summary"]["by_category"]["training_failure"] == 2
        assert data["summary"]["by_severity"]["warning"] == 2

    def test_log_creates_reports_dir(self, tmp_path):
        """log_event creates reports/ subdirectory if it doesn't exist."""
        exp_root = tmp_path / "experiments"
        exp_root.mkdir()
        ev = create_event("config_error", "info", "experiment", "test")
        path = log_event(str(exp_root), ev)
        assert (exp_root / "reports").is_dir()
        assert Path(path).exists()

    def test_load_edge_cases(self, tmp_path):
        """load_error_log: valid returns data, not-found returns None, corrupt returns None."""
        assert load_error_log(str(tmp_path)) is None
        log_event(str(tmp_path), create_event("agent_failure", "critical", "orchestrate", "x"))
        assert load_error_log(str(tmp_path)) is not None

        reports = tmp_path / "corrupt"
        reports.mkdir()
        (reports / "reports").mkdir()
        ((reports / "reports") / "error-log.json").write_text("{bad json")
        assert load_error_log(str(reports)) is None

    def test_get_events_filtering(self, tmp_path):
        """get_events: no filter, category filter, severity filter, both."""
        log_event(str(tmp_path), create_event("training_failure", "critical", "experiment", "a"))
        log_event(str(tmp_path), create_event("divergence", "warning", "monitor", "b"))
        log_event(str(tmp_path), create_event("training_failure", "warning", "experiment", "c"))

        assert len(get_events(str(tmp_path))) == 3
        assert len(get_events(str(tmp_path), category="training_failure")) == 2
        assert len(get_events(str(tmp_path), severity="critical")) == 1
        filtered = get_events(str(tmp_path), category="training_failure", severity="critical")
        assert len(filtered) == 1 and filtered[0]["message"] == "a"
        assert get_events(str(tmp_path / "nonexistent")) == []

    def test_concurrent_log_events_no_loss(self, tmp_path):
        """4 threads x 5 events each = 20 events, verify none lost."""
        errors = []

        def log_events(thread_id):
            try:
                for i in range(5):
                    ev = create_event(
                        category="training_failure", severity="warning",
                        source="experiment", message=f"Thread {thread_id} event {i}",
                        exp_id=f"exp-t{thread_id}-{i}", phase=6)
                    log_event(str(tmp_path), ev)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=log_events, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(get_events(str(tmp_path))) == 20

    def test_duration_stored_and_retrievable(self, tmp_path):
        log_event(str(tmp_path), create_event(
            "timeout", "critical", "orchestrate", "Timed out",
            duration_seconds=3600.5))
        assert load_error_log(str(tmp_path))["events"][0]["duration_seconds"] == 3600.5

    def test_atomic_write_json_exception(self, tmp_path):
        path = tmp_path / "reports" / "error-log.json"
        path.parent.mkdir(parents=True)
        with pytest.raises(TypeError):
            _atomic_write_json(path, {"bad": set()})

    def test_load_results_edge_cases(self, tmp_path):
        """_load_results: no dir, corrupt JSON, missing exp_id."""
        baseline, experiments = _load_results(str(tmp_path))
        assert baseline is None and experiments == []

        results = tmp_path / "results"
        results.mkdir()
        (results / "exp-001.json").write_text("CORRUPT")
        (results / "exp-002.json").write_text(json.dumps({
            "exp_id": "exp-002", "status": "completed",
            "config": {}, "metrics": {"loss": 0.5}
        }))
        (results / "exp-003.json").write_text(json.dumps({"no_id": True}))
        _, experiments = _load_results(str(tmp_path))
        assert len(experiments) == 1 and experiments[0]["exp_id"] == "exp-002"


# ===========================================================================
# TestPatternDetection
# ===========================================================================


_DIVERGENCE_HIGH_LR = [
    create_event("divergence", "warning", "monitor", "NaN",
                 config={"lr": lr, "batch_size": 32})
    for lr in [0.1, 0.2, 0.05]
]

_OOM_SAME_BS = [
    create_event("training_failure", "critical", "experiment", "OOM",
                 config={"lr": lr, "batch_size": 256},
                 context={"error_type": "oom"})
    for lr in [0.01, 0.001]
]

_WASTED_BUDGET = [
    create_event("pipeline_inefficiency", "warning", "analyze",
                 f"All {n} experiments in batch diverged",
                 context={"experiments_wasted": n})
    for n in [3, 2]
]

_EARLY_FAILURE = [
    create_event("training_failure", "critical", "baseline", f"fail{i}", phase=2)
    for i in range(3)
]

_HP_INTERACTION = [
    create_event(cat, sev, src, msg,
                 config={"lr": lr, "batch_size": 256})
    for cat, sev, src, msg, lr in [
        ("divergence", "warning", "monitor", "NaN", 0.05),
        ("training_failure", "critical", "experiment", "OOM", 0.02),
        ("divergence", "warning", "monitor", "explosion", 0.03),
    ]
]

_TEMPORAL_EARLY = [
    create_event("training_failure", "critical", "experiment", f"f{i}", iteration=it)
    for i, it in enumerate([1, 1, 1, 2, 1], 1)
]

_TIMEOUT = [
    create_event("timeout", "critical", "orchestrate", msg, duration_seconds=dur)
    for msg, dur in [("Agent timed out", 600), ("Training timed out", 7200)]
]

_REDUNDANT_CONFIGS = [
    create_event("pipeline_inefficiency", "info", "hp-tune",
                 f"Regenerated {i+1} proposals due to duplication")
    for i in range(2)
]

_BRANCH_UNDERPERFORMANCE = [
    create_event("pipeline_inefficiency", "info", "analyze",
                 "Branch ml-opt/foo underperforms baseline across all HP configs",
                 code_branch="ml-opt/foo")
]


class TestPatternDetection:
    """Tests for detect_patterns."""

    def test_empty(self):
        assert detect_patterns([]) == []

    @pytest.mark.parametrize("events,expected_pattern", [
        (_DIVERGENCE_HIGH_LR, "high_lr_divergence"),
        (_OOM_SAME_BS, "oom_batch_size"),
        (_WASTED_BUDGET, "wasted_budget"),
        (_EARLY_FAILURE, "early_failure_cluster"),
        (_HP_INTERACTION, "hp_interaction_failure"),
        (_TEMPORAL_EARLY, "temporal_failure_cluster"),
        (_TIMEOUT, "timeout_pattern"),
        (_REDUNDANT_CONFIGS, "redundant_configs"),
        (_BRANCH_UNDERPERFORMANCE, "branch_underperformance"),
    ], ids=[
        "high_lr_divergence", "oom_batch_size", "wasted_budget",
        "early_failure_cluster", "hp_interaction_failure",
        "temporal_failure_cluster", "timeout_pattern",
        "redundant_configs", "branch_underperformance",
    ])
    def test_pattern_detected(self, events, expected_pattern):
        ids = [p["pattern_id"] for p in detect_patterns(events)]
        assert expected_pattern in ids

    @pytest.mark.parametrize("events,absent_pattern", [
        # Single divergence => no pattern
        ([create_event("divergence", "warning", "monitor", "NaN",
                        config={"lr": 0.001, "batch_size": 32})],
         "high_lr_divergence"),
        # Phase 5 failures => no early_failure
        ([create_event("training_failure", "critical", "experiment", f"f{i}", phase=5)
          for i in range(3)],
         "early_failure_cluster"),
        # Evenly spread iterations => no temporal cluster
        ([create_event("training_failure", "critical", "experiment", f"f{i}", iteration=i)
          for i in range(1, 5)],
         "temporal_failure_cluster"),
        # 1 timeout => no pattern
        ([create_event("timeout", "critical", "orchestrate", "Agent timed out")],
         "timeout_pattern"),
    ], ids=["single_divergence", "phase5_excluded", "spread_iters", "single_timeout"])
    def test_no_false_positive(self, events, absent_pattern):
        ids = [p["pattern_id"] for p in detect_patterns(events)]
        assert absent_pattern not in ids

    def test_from_fixture(self):
        data = json.loads((FIXTURES / "sample_error_log.json").read_text())
        ids = [p["pattern_id"] for p in detect_patterns(data["events"])]
        assert "high_lr_divergence" in ids
        assert "oom_batch_size" in ids

    def test_nan_inf_lr_excluded(self):
        events = [
            create_event("divergence", "warning", "monitor", "NaN",
                         config={"lr": lr, "batch_size": 32})
            for lr in [0.1, 0.2, float("nan"), float("inf")]
        ]
        patterns = detect_patterns(events)
        lr_pattern = [p for p in patterns if p["pattern_id"] == "high_lr_divergence"]
        assert len(lr_pattern) == 1
        assert "0.1500" in lr_pattern[0]["description"]

    def test_bad_config_skipped(self, tmp_path):
        """Events with missing/non-numeric/non-dict config are skipped in interaction check."""
        exp_root = str(tmp_path / "exp")
        for i in range(3):
            log_event(exp_root, create_event(
                "divergence", "warning", "monitor", f"Diverged {i}",
                config={"some_hp": 1}))
        ids = [p["pattern_id"]
               for p in detect_patterns(load_error_log(exp_root)["events"])]
        assert "hp_interaction_failure" not in ids

    def test_summarize_session(self, tmp_path):
        """Session summary: correct counts, empty returns zeros, includes patterns."""
        # Empty
        summary = summarize_session(str(tmp_path))
        assert summary["total_events"] == 0 and summary["by_category"] == {}

        # With events
        for lr in [0.1, 0.2, 0.05]:
            log_event(str(tmp_path), create_event(
                "divergence", "warning", "monitor", "NaN",
                config={"lr": lr, "batch_size": 32}))
        summary = summarize_session(str(tmp_path))
        assert summary["total_events"] == 3
        assert summary["by_category"]["divergence"] == 3
        assert "high_lr_divergence" in summary["patterns_detected"]


# ===========================================================================
# TestCrossProject
# ===========================================================================


class TestCrossProject:
    """Tests for cross-project storage and pattern detection."""

    def test_update_and_load(self, tmp_path):
        """First sync creates file; load returns data; load missing returns None."""
        exp_root = tmp_path / "project" / "experiments"
        exp_root.mkdir(parents=True)
        plugin_root = tmp_path / "plugin"
        plugin_root.mkdir()
        (plugin_root / "memory").mkdir()
        ev = create_event("training_failure", "critical", "experiment", "crash")
        log_event(str(exp_root), ev)
        path = update_cross_project(str(plugin_root), str(tmp_path / "project"), str(exp_root))
        assert Path(path).exists()
        data = load_cross_project(str(plugin_root))
        assert data is not None and len(data["projects"]) == 1

        assert load_cross_project(str(tmp_path / "nonexistent")) is None

    def test_load_corrupt(self, tmp_path):
        mem = tmp_path / "memory"
        mem.mkdir()
        (mem / "cross-project-errors.json").write_text("not json{")
        assert load_cross_project(str(tmp_path)) is None

    def test_update_creates_memory_dir_and_handles_no_log(self, tmp_path):
        """update_cross_project creates memory/ dir; handles missing error-log.json."""
        exp_root = tmp_path / "project" / "experiments"
        exp_root.mkdir(parents=True)
        plugin_root = tmp_path / "plugin"
        plugin_root.mkdir()
        log_event(str(exp_root), create_event("config_error", "info", "experiment", "x"))
        path = update_cross_project(str(plugin_root), str(tmp_path / "project"), str(exp_root))
        assert (plugin_root / "memory").is_dir()

        # No error log case
        plugin2 = tmp_path / "plugin2"
        exp2 = tmp_path / "project2" / "experiments"
        exp2.mkdir(parents=True)
        update_cross_project(str(plugin2), str(tmp_path / "project2"), str(exp2))
        assert (Path(plugin2) / "memory" / "cross-project-errors.json").exists()

    def test_sync_dedup_and_update(self, tmp_path):
        """Same session deduplicates; more events update count; different projects kept."""
        plugin_root = tmp_path / "plugin"
        exp_root = tmp_path / "project"
        exp_root.mkdir()
        log_event(str(exp_root), create_event("training_failure", "warning", "experiment", "fail"))
        update_cross_project(str(plugin_root), str(exp_root), str(exp_root))
        update_cross_project(str(plugin_root), str(exp_root), str(exp_root))
        memory = load_cross_project(str(plugin_root))
        for proj_data in memory["projects"].values():
            assert len(proj_data["sessions"]) == 1

        log_event(str(exp_root), create_event("divergence", "warning", "monitor", "NaN"))
        update_cross_project(str(plugin_root), str(exp_root), str(exp_root))
        memory = load_cross_project(str(plugin_root))
        for proj_data in memory["projects"].values():
            assert proj_data["sessions"][0]["event_count"] == 2

    def test_detect_patterns(self):
        """No projects => empty; shared pattern across 2+ projects detected."""
        assert detect_cross_project_patterns(
            {"version": 1, "projects": {}, "cross_project_patterns": []}) == []

        memory = {
            "version": 1,
            "projects": {
                "p1": {"project_path": "/a",
                       "sessions": [{"patterns_detected": ["high_lr_divergence", "oom_batch_size"]}]},
                "p2": {"project_path": "/b",
                       "sessions": [{"patterns_detected": ["high_lr_divergence"]}]},
            },
            "cross_project_patterns": [],
        }
        ids = [p["pattern_id"] for p in detect_cross_project_patterns(memory)]
        assert "high_lr_divergence" in ids
        assert "oom_batch_size" not in ids

    def test_cross_project_integration(self, tmp_path):
        """Two projects synced detect shared patterns."""
        plugin_root = tmp_path / "plugin"
        (plugin_root / "memory").mkdir(parents=True)
        for proj_name, lrs in [("project_a", [0.1, 0.2, 0.05]),
                                ("project_b", [0.3, 0.15, 0.08])]:
            exp = tmp_path / proj_name / "experiments"
            exp.mkdir(parents=True)
            for lr in lrs:
                log_event(str(exp), create_event(
                    "divergence", "warning", "monitor", "NaN",
                    config={"lr": lr, "batch_size": 32}))
            update_cross_project(str(plugin_root), str(tmp_path / proj_name), str(exp))
        ids = [p["pattern_id"]
               for p in detect_cross_project_patterns(load_cross_project(str(plugin_root)))]
        assert "high_lr_divergence" in ids

    def test_cleanup(self, tmp_path):
        """cleanup_memory: trims sessions, removes empty projects, no-op under limit, missing file."""
        plugin_root = tmp_path / "plugin"
        mem_dir = plugin_root / "memory"
        mem_dir.mkdir(parents=True)
        sessions = [
            {"session_start": f"2026-01-{i+1:02d}T00:00:00Z", "event_count": i,
             "categories": {}, "patterns_detected": []}
            for i in range(15)
        ]
        memory = {
            "version": 1, "last_updated": "2026-03-07T00:00:00Z",
            "projects": {
                "proj1": {"project_path": "/tmp/proj", "sessions": sessions},
                "empty": {"project_path": "/a", "sessions": []},
            },
            "cross_project_patterns": [],
        }
        (mem_dir / "cross-project-errors.json").write_text(json.dumps(memory))
        result = cleanup_memory(str(plugin_root), max_sessions_per_project=10)
        assert result["cleaned"] == 5
        updated = load_cross_project(str(plugin_root))
        assert len(updated["projects"]["proj1"]["sessions"]) == 10
        assert "empty" not in updated["projects"]

        # No file
        assert cleanup_memory(str(tmp_path / "missing")) == {"cleaned": 0, "projects_remaining": 0}


# ===========================================================================
# TestSuccessMetrics
# ===========================================================================


class TestSuccessMetrics:
    """Tests for compute_success_metrics and compute_proposal_outcomes."""

    def test_basic_metrics(self, tmp_path):
        """Mixed results produce correct success/failure counts and improvement."""
        results = tmp_path / "results"
        results.mkdir()
        _write_result(results, "baseline", "completed", {"lr": 0.001}, {"acc": 70.0})
        _write_result(results, "exp-001", "completed", {"lr": 0.001}, {"acc": 75.0})
        _write_result(results, "exp-002", "completed", {"lr": 0.01}, {"acc": 68.0})
        _write_result(results, "exp-003", "failed", {"lr": 0.1}, {"acc": 0.0})
        _write_result(results, "exp-004", "diverged", {"lr": 0.5}, {})
        m = compute_success_metrics(str(tmp_path), "acc", lower_is_better=False)
        assert m["total_experiments"] == 4
        assert m["completed"] == 2
        assert m["failed"] == 1
        assert m["diverged"] == 1
        assert m["improvement_rate"] > 0
        # Top configs sorted
        assert m["top_configs"][0]["exp_id"] == "exp-001"

    def test_no_baseline_and_empty(self, tmp_path):
        """Missing baseline returns improvement_rate=None; empty returns zero counts."""
        results = tmp_path / "results"
        results.mkdir()
        _write_result(results, "exp-001", "completed", {"lr": 0.001}, {"acc": 75.0})
        m = compute_success_metrics(str(tmp_path), "acc", lower_is_better=False)
        assert m["improvement_rate"] is None

        empty = tmp_path / "empty" / "results"
        empty.mkdir(parents=True)
        m2 = compute_success_metrics(str(tmp_path / "empty"), "acc", lower_is_better=False)
        assert m2["total_experiments"] == 0

    def test_all_failed(self, tmp_path):
        results = tmp_path / "results"
        results.mkdir()
        _write_result(results, "baseline", "completed", {"lr": 0.001}, {"loss": 1.0})
        _write_result(results, "exp-001", "failed", {"lr": 0.1}, {})
        m = compute_success_metrics(str(tmp_path), "loss", lower_is_better=True)
        assert m["success_rate"] == 0.0

    def test_duration_analysis(self, tmp_path):
        results = tmp_path / "results"
        results.mkdir()
        _write_result(results, "baseline", "completed", {}, {"loss": 1.0})
        _write_result(results, "exp-001", "completed", {}, {"loss": 0.5}, duration_seconds=3600)
        _write_result(results, "exp-002", "completed", {}, {"loss": 0.8}, duration_seconds=1800)
        _write_result(results, "exp-003", "failed", {}, {}, duration_seconds=300)
        m = compute_success_metrics(str(tmp_path), "loss", lower_is_better=True)
        assert m["avg_duration_completed"] == 2700.0
        assert m["time_wasted_on_failures_pct"] > 0

    def test_edge_cases_nan_inf_zero(self, tmp_path):
        """NaN baseline, Inf experiment, zero baseline handled correctly."""
        # NaN baseline
        r1 = tmp_path / "nan" / "results"
        r1.mkdir(parents=True)
        _write_result(r1, "baseline", "completed", {}, {"acc": float("nan")})
        _write_result(r1, "exp-001", "completed", {}, {"acc": 80.0})
        assert compute_success_metrics(str(tmp_path / "nan"), "acc", False)["improvement_rate"] is None

        # Inf experiment
        r2 = tmp_path / "inf" / "results"
        r2.mkdir(parents=True)
        _write_result(r2, "baseline", "completed", {}, {"acc": 70.0})
        _write_result(r2, "exp-001", "completed", {}, {"acc": float("inf")})
        _write_result(r2, "exp-002", "completed", {}, {"acc": 75.0})
        m = compute_success_metrics(str(tmp_path / "inf"), "acc", False)
        assert m["best_improvement_pct"] is not None

        # Zero baseline
        r3 = tmp_path / "zero" / "results"
        r3.mkdir(parents=True)
        (r3 / "baseline.json").write_text(json.dumps({
            "exp_id": "baseline", "status": "completed", "config": {}, "metrics": {"loss": 0}
        }))
        (r3 / "exp-001.json").write_text(json.dumps({
            "exp_id": "exp-001", "status": "completed", "config": {}, "metrics": {"loss": 0.5}
        }))
        assert compute_success_metrics(str(tmp_path / "zero"), "loss", True) is not None

    @pytest.mark.parametrize("metric,lower,baseline_val,exp_val,expected_pct", [
        ("loss", True, 1.0, 0.5, 50.0),
        ("acc", False, 70.0, 77.0, 10.0),
    ], ids=["lower_is_better", "higher_is_better"])
    def test_improvement_pct_sign(self, tmp_path, metric, lower, baseline_val, exp_val, expected_pct):
        exp_root = tmp_path / "experiments"
        results = exp_root / "results"
        results.mkdir(parents=True)
        (results / "baseline.json").write_text(json.dumps({
            "exp_id": "baseline", "status": "completed",
            "config": {}, "metrics": {metric: baseline_val}
        }))
        (results / "exp-001.json").write_text(json.dumps({
            "exp_id": "exp-001", "status": "completed",
            "config": {"lr": 0.001}, "metrics": {metric: exp_val}
        }))
        m = compute_success_metrics(str(exp_root), metric, lower)
        assert m["improvement_rate"] == 1.0
        assert m["top_configs"][0]["improvement_pct"] == expected_pct

    def test_proposal_outcomes_with_manifest(self, tmp_path):
        """Cross-references manifest proposals with experiment results."""
        results = tmp_path / "results"
        results.mkdir()
        manifest = {
            "original_branch": "main", "strategy": "git_branch",
            "proposals": [
                {"name": "perceptual-loss", "slug": "perceptual-loss",
                 "status": "validated", "branch": "ml-opt/perceptual-loss"},
                {"name": "mixup-aug", "slug": "mixup-aug",
                 "status": "validation_failed"},
            ],
        }
        (results / "implementation-manifest.json").write_text(json.dumps(manifest))
        _write_result(results, "baseline", "completed", {"lr": 0.001}, {"acc": 70.0})
        _write_result(results, "exp-001", "completed", {"lr": 0.001}, {"acc": 78.0},
                      code_branch="ml-opt/perceptual-loss", code_proposal="perceptual-loss")
        _write_result(results, "exp-002", "completed", {"lr": 0.01}, {"acc": 72.0},
                      code_branch="ml-opt/perceptual-loss", code_proposal="perceptual-loss")
        p = compute_proposal_outcomes(str(tmp_path), "acc", lower_is_better=False)
        assert p["implementation_stats"]["validated"] == 1
        assert p["implementation_stats"]["validation_failed"] == 1
        assert len(p["research_proposals"]) == 1
        assert p["research_proposals"][0]["beat_baseline"] >= 1

    def test_proposal_outcomes_edge_cases(self, tmp_path):
        """No manifest, HP stats, lower-is-better sign, corrupt manifest, impl error status."""
        # No manifest
        r1 = tmp_path / "no_manifest" / "results"
        r1.mkdir(parents=True)
        _write_result(r1, "baseline", "completed", {}, {"acc": 70.0})
        p = compute_proposal_outcomes(str(tmp_path / "no_manifest"), "acc", False)
        assert p["implementation_stats"]["total_proposals"] == 0

        # HP stats
        r2 = tmp_path / "hp" / "results"
        r2.mkdir(parents=True)
        configs_dir = r2 / "proposed-configs"
        configs_dir.mkdir()
        for i in range(3):
            (configs_dir / f"exp-{i+1:03d}.json").write_text(json.dumps({"lr": 0.001 * (i + 1)}))
        _write_result(r2, "baseline", "completed", {}, {"loss": 1.0})
        _write_result(r2, "exp-001", "completed", {"lr": 0.001}, {"loss": 0.5})
        _write_result(r2, "exp-002", "diverged", {"lr": 0.002}, {})
        p2 = compute_proposal_outcomes(str(tmp_path / "hp"), "loss", True)
        assert p2["hp_proposals"]["total_proposed"] == 3
        assert p2["hp_proposals"]["total_run"] == 2

        # Corrupt manifest
        r3 = tmp_path / "corrupt" / "results"
        r3.mkdir(parents=True)
        (r3 / "implementation-manifest.json").write_text("NOT VALID JSON")
        (r3 / "baseline.json").write_text(json.dumps({
            "exp_id": "baseline", "status": "completed", "config": {}, "metrics": {"loss": 1.0}
        }))
        assert compute_proposal_outcomes(str(tmp_path / "corrupt"), "loss", True)["implementation_stats"]["total_proposals"] == 0

        # Impl error status
        r4 = tmp_path / "impl_err" / "results"
        r4.mkdir(parents=True)
        (r4 / "implementation-manifest.json").write_text(json.dumps({
            "proposals": [
                {"name": "p1", "branch": "ml-opt/p1", "status": "validated"},
                {"name": "p2", "branch": "", "status": "implementation_error"},
            ]
        }))
        (r4 / "baseline.json").write_text(json.dumps({
            "exp_id": "baseline", "status": "completed", "config": {}, "metrics": {"loss": 1.0}
        }))
        p4 = compute_proposal_outcomes(str(tmp_path / "impl_err"), "loss", True)
        assert p4["implementation_stats"]["implementation_error"] == 1


# ===========================================================================
# TestSuggestions
# ===========================================================================


class TestSuggestions:
    """Tests for log_suggestion, get_suggestion_history, rank_suggestions."""

    def test_suggestion_lifecycle(self, tmp_path):
        """Log, retrieve, increment iteration, handle empty and corrupt."""
        assert get_suggestion_history(str(tmp_path)) == []

        log_suggestion(str(tmp_path), "high_lr_divergence", scope="session")
        history = get_suggestion_history(str(tmp_path))
        assert len(history) == 1
        assert history[0]["pattern_id"] == "high_lr_divergence"
        assert history[0]["iteration"] == 1

        log_suggestion(str(tmp_path), "oom_batch_size")
        log_suggestion(str(tmp_path), "high_lr_divergence")
        history = get_suggestion_history(str(tmp_path))
        assert len(history) == 3
        lr_suggestions = [s for s in history if s["pattern_id"] == "high_lr_divergence"]
        assert lr_suggestions[1]["iteration"] == 2

        # Corrupt file
        reports = tmp_path / "corrupt" / "reports"
        reports.mkdir(parents=True)
        (reports / "suggestion-history.json").write_text("CORRUPT")
        assert get_suggestion_history(str(tmp_path / "corrupt")) == []

    def test_rank_suggestions(self):
        """Ranking order, cross-project boost, empty input, significance."""
        patterns = [
            {"pattern_id": "redundant_configs", "description": "dup",
             "occurrences": 5, "suggested_action": "widen"},
            {"pattern_id": "oom_batch_size", "description": "oom",
             "occurrences": 2, "suggested_action": "reduce bs"},
        ]
        ranked = rank_suggestions(patterns)
        assert ranked[0]["pattern_id"] == "oom_batch_size"
        assert "score" in ranked[0]

        # Cross-project boost
        cross = [{"pattern_id": "wasted_budget", "projects_affected": 3}]
        boosted_patterns = [
            {"pattern_id": "wasted_budget", "description": "waste",
             "occurrences": 2, "suggested_action": "tighten"},
        ]
        ranked_no = rank_suggestions(boosted_patterns)
        ranked_yes = rank_suggestions(boosted_patterns, cross_project_patterns=cross)
        assert ranked_yes[0]["score"] == ranked_no[0]["score"] * 1.5

        # Empty
        assert rank_suggestions([]) == []

        # Significance
        ranked_with = rank_suggestions([patterns[1]], total_experiments=100)
        assert ranked_with[0]["significance"] == 0.02
        ranked_without = rank_suggestions([patterns[1]])
        assert "significance" not in ranked_without[0]


# ===========================================================================
# TestDeadEndCatalog
# ===========================================================================


class TestDeadEndCatalog:
    """Tests for dead-end catalog: log, get, is_dead_end, fuzzy matching."""

    def test_log_basic(self, tmp_path):
        path = log_dead_end(str(tmp_path), {"technique": "perceptual-loss", "reason": "5% worse"})
        assert "dead-ends.json" in path
        de = get_dead_ends(str(tmp_path))
        assert len(de) == 1 and "timestamp" in de[0]

    def test_log_multiple_with_details(self, tmp_path):
        log_dead_end(str(tmp_path), {"technique": "mixup", "reason": "no improvement"})
        log_dead_end(str(tmp_path), {
            "technique": "focal-loss", "reason": "worse on all configs",
            "branch": "ml-opt/focal-loss", "experiments_tried": 4,
            "best_result": {"metric": "accuracy", "value": 0.82, "baseline": 0.85},
        })
        de = get_dead_ends(str(tmp_path))
        assert len(de) == 2
        assert de[1]["branch"] == "ml-opt/focal-loss"

    def test_get_empty_and_corrupt(self, tmp_path):
        assert get_dead_ends(str(tmp_path)) == []
        reports = tmp_path / "reports"
        reports.mkdir(parents=True)
        (reports / "dead-ends.json").write_text("NOT JSON")
        assert get_dead_ends(str(tmp_path)) == []

    @pytest.mark.parametrize("query,expected", [
        ("perceptual-loss", True),
        ("PERCEPTUAL-LOSS", True),
        ("perceptual_loss", True),
        ("perceptual", True),
        ("deep perceptual loss function", True),
        ("label-smoothing", False),
    ], ids=[
        "case_match", "upper_case", "underscore",
        "substring_of_cataloged", "reverse_substring", "no_match",
    ])
    def test_fuzzy_match(self, tmp_path, query, expected):
        log_dead_end(str(tmp_path), {"technique": "Perceptual-Loss", "reason": "bad"})
        assert is_dead_end(str(tmp_path), query) is expected

    def test_empty_query(self, tmp_path):
        log_dead_end(str(tmp_path), {"technique": "test", "reason": "bad"})
        assert is_dead_end(str(tmp_path), "") is False
        assert is_dead_end(str(tmp_path), "  ") is False

    def test_normalize_technique(self):
        assert _normalize_technique("Perceptual-Loss") == "perceptual loss"
        assert _normalize_technique("  label_smoothing  ") == "label smoothing"

    def test_generates_markdown(self, tmp_path):
        log_dead_end(str(tmp_path), {
            "technique": "mixup", "reason": "no improvement on this dataset",
            "branch": "ml-opt/mixup", "experiments_tried": 3,
        })
        md = (tmp_path / "reports" / "dead-ends.md").read_text()
        assert "mixup" in md and "no improvement" in md


# ===========================================================================
# TestResearchAgenda
# ===========================================================================


class TestResearchAgenda:
    """Tests for research agenda: init, get, update, add."""

    def test_init(self, tmp_path):
        ideas = [
            {"id": "idea-1", "name": "CutMix augmentation", "priority": 8},
            {"id": "idea-2", "name": "Cosine annealing", "priority": 6},
        ]
        path = init_agenda(str(tmp_path), ideas)
        assert "research-agenda.json" in path
        agenda = get_agenda(str(tmp_path))
        assert len(agenda) == 2
        assert agenda[0]["status"] == "untried"
        assert agenda[0]["initial_priority"] == 8
        assert agenda[0]["evidence"] == []

    def test_get_empty_and_corrupt(self, tmp_path):
        assert get_agenda(str(tmp_path)) == []
        reports = tmp_path / "reports"
        reports.mkdir(parents=True)
        (reports / "research-agenda.json").write_text("CORRUPT")
        assert get_agenda(str(tmp_path)) == []

    def test_update_status_and_evidence(self, tmp_path):
        init_agenda(str(tmp_path), [{"id": "idea-1", "name": "Test", "priority": 7}])
        assert update_agenda_item(str(tmp_path), "idea-1", {
            "status": "tried", "priority": 4,
            "evidence": {"batch": 3, "result": "2% worse"},
            "lessons": "Did not help",
        }) is True
        idea = get_agenda(str(tmp_path))[0]
        assert idea["status"] == "tried" and idea["priority"] == 4
        assert len(idea["evidence"]) == 1

        # Evidence appends
        update_agenda_item(str(tmp_path), "idea-1", {
            "evidence": {"batch": 4, "result": "still worse"}})
        assert len(get_agenda(str(tmp_path))[0]["evidence"]) == 2

        # Not found
        assert update_agenda_item(str(tmp_path), "nonexistent", {"status": "tried"}) is False

    def test_add_idea(self, tmp_path):
        init_agenda(str(tmp_path), [{"id": "idea-1", "name": "Original"}])
        add_agenda_idea(str(tmp_path), {"id": "idea-2", "name": "New idea", "priority": 9})
        agenda = get_agenda(str(tmp_path))
        assert len(agenda) == 2 and agenda[1]["priority"] == 9

        # Add to non-existing
        add_agenda_idea(str(tmp_path / "new"), {"id": "idea-1", "name": "First"})
        assert len(get_agenda(str(tmp_path / "new"))) == 1

    def test_generates_markdown_with_sections(self, tmp_path):
        init_agenda(str(tmp_path), [
            {"id": "a", "name": "Active", "status": "untried"},
            {"id": "b", "name": "Tried", "status": "tried"},
            {"id": "c", "name": "Winner", "status": "improved"},
            {"id": "d", "name": "Failed", "status": "dead-end", "lessons": "too slow"},
        ])
        md = (tmp_path / "reports" / "research-agenda.md").read_text()
        assert "Active Ideas" in md
        assert "Successful Techniques" in md
        assert "Dead Ends" in md
        assert "too slow" in md

    def test_priority_preserved_on_update(self, tmp_path):
        init_agenda(str(tmp_path), [{"id": "idea-1", "name": "Test", "priority": 8}])
        update_agenda_item(str(tmp_path), "idea-1", {"status": "tried"})
        idea = get_agenda(str(tmp_path))[0]
        assert idea["priority"] == 8 and idea["initial_priority"] == 8


# ===========================================================================
# TestReviewWorkflow
# ===========================================================================


class TestReviewWorkflow:
    """End-to-end tests for the review skill's data flow."""

    def test_full_session(self, tmp_path):
        """Full session: summary, patterns, success metrics, proposal outcomes, ranked suggestions."""
        exp_root = _create_full_session(tmp_path)

        summary = summarize_session(str(exp_root))
        assert summary["total_events"] == 8
        assert summary["by_category"]["divergence"] == 3

        log_data = load_error_log(str(exp_root))
        ids = [p["pattern_id"] for p in detect_patterns(log_data["events"])]
        assert "high_lr_divergence" in ids and "wasted_budget" in ids

        m = compute_success_metrics(str(exp_root), "acc", lower_is_better=False)
        assert m["completed"] == 2 and m["improvement_rate"] > 0

        p = compute_proposal_outcomes(str(exp_root), "acc", lower_is_better=False)
        assert p["implementation_stats"]["validated"] == 1
        assert p["research_proposals"][0]["beat_baseline"] >= 1

        ranked = rank_suggestions(detect_patterns(log_data["events"]))
        for i in range(len(ranked) - 1):
            assert ranked[i]["score"] >= ranked[i + 1]["score"]

    def test_new_categories(self, tmp_path):
        exp_root = tmp_path / "experiments"
        exp_root.mkdir()
        log_event(str(exp_root), create_event("research_failure", "warning", "research", "No results", phase=4))
        log_event(str(exp_root), create_event("timeout", "critical", "orchestrate", "Timed out", duration_seconds=600))
        log_event(str(exp_root), create_event("resource_error", "info", "baseline", "No GPU", phase=2))
        summary = summarize_session(str(exp_root))
        for cat in ("research_failure", "timeout", "resource_error"):
            assert summary["by_category"][cat] == 1

    def test_empty_session(self, tmp_path):
        exp_root = tmp_path / "experiments"
        exp_root.mkdir()
        (exp_root / "results").mkdir()
        assert summarize_session(str(exp_root))["total_events"] == 0
        assert compute_success_metrics(str(exp_root), "acc", False)["total_experiments"] == 0


# ===========================================================================
# TestCLI
# ===========================================================================


class TestCLI:
    """Tests for the CLI interface of error_tracker.py."""

    def test_no_args(self, run_main):
        r = run_main("error_tracker.py")
        assert r.returncode == 1

    def test_log_and_show(self, run_main, tmp_path):
        """CLI log creates entry; show returns it; show with filter works."""
        ev_json = json.dumps({
            "category": "training_failure", "severity": "critical",
            "source": "experiment", "message": "test crash",
        })
        r = run_main("error_tracker.py", str(tmp_path), "log", ev_json)
        assert r.returncode == 0

        log_event(str(tmp_path), create_event("divergence", "warning", "monitor", "nan"))
        r = run_main("error_tracker.py", str(tmp_path), "show")
        assert r.returncode == 0 and len(json.loads(r.stdout)) >= 2

        r = run_main("error_tracker.py", str(tmp_path), "show", "divergence")
        events = json.loads(r.stdout)
        assert all(e["category"] == "divergence" for e in events)

    def test_patterns_and_summary(self, run_main, tmp_path):
        for lr in [0.1, 0.2, 0.05]:
            log_event(str(tmp_path), create_event(
                "divergence", "warning", "monitor", "NaN",
                config={"lr": lr, "batch_size": 32}))
        r = run_main("error_tracker.py", str(tmp_path), "patterns")
        assert r.returncode == 0 and len(json.loads(r.stdout)) > 0

        r = run_main("error_tracker.py", str(tmp_path), "summary")
        assert r.returncode == 0 and json.loads(r.stdout)["total_events"] == 3

    def test_sync(self, run_main, tmp_path):
        exp_root = tmp_path / "project" / "experiments"
        exp_root.mkdir(parents=True)
        plugin_root = tmp_path / "plugin"
        (plugin_root / "memory").mkdir(parents=True)
        log_event(str(exp_root), create_event("divergence", "warning", "monitor", "x"))
        r = run_main("error_tracker.py", str(exp_root), "sync", str(plugin_root))
        assert r.returncode == 0

    def test_success_and_proposals(self, run_main, tmp_path):
        results = tmp_path / "results"
        results.mkdir()
        _write_result(results, "baseline", "completed", {}, {"acc": 70.0})
        _write_result(results, "exp-001", "completed", {}, {"acc": 75.0})
        r = run_main("error_tracker.py", str(tmp_path), "success", "acc", "false")
        assert r.returncode == 0 and json.loads(r.stdout)["total_experiments"] == 1

        r = run_main("error_tracker.py", str(tmp_path), "proposals", "acc", "false")
        assert r.returncode == 0 and "implementation_stats" in json.loads(r.stdout)

    def test_rank(self, run_main, tmp_path):
        for lr in [0.1, 0.2, 0.05]:
            log_event(str(tmp_path), create_event(
                "divergence", "warning", "monitor", "NaN",
                config={"lr": lr, "batch_size": 32}))
        r = run_main("error_tracker.py", str(tmp_path), "rank")
        assert r.returncode == 0 and "score" in json.loads(r.stdout)[0]

        r = run_main("error_tracker.py", str(tmp_path), "rank", "50")
        assert "significance" in json.loads(r.stdout)[0]

    def test_rank_with_cross_project(self, run_main, tmp_path):
        plugin_root = tmp_path / "plugin"
        plugin_root.mkdir()
        for proj_name in ["proj_a", "proj_b"]:
            proj = tmp_path / proj_name
            proj.mkdir()
            for lr in [0.1, 0.2, 0.05]:
                log_event(str(proj), create_event(
                    "divergence", "warning", "monitor", "NaN",
                    config={"lr": lr, "batch_size": 32}))
            update_cross_project(str(plugin_root), str(proj), str(proj))
        r_boosted = run_main("error_tracker.py", str(tmp_path / "proj_a"), "rank", "50", str(plugin_root))
        r_plain = run_main("error_tracker.py", str(tmp_path / "proj_a"), "rank", "50")
        assert json.loads(r_boosted.stdout)[0]["score"] > json.loads(r_plain.stdout)[0]["score"]

    def test_cleanup_cli(self, run_main, tmp_path):
        plugin_root = tmp_path / "plugin"
        mem_dir = plugin_root / "memory"
        mem_dir.mkdir(parents=True)
        sessions = [
            {"session_start": f"2026-01-{i+1:02d}T00:00:00Z", "event_count": i,
             "categories": {}, "patterns_detected": []}
            for i in range(5)
        ]
        memory = {
            "version": 1, "last_updated": "2026-03-07T00:00:00Z",
            "projects": {"proj1": {"project_path": "/tmp/p", "sessions": sessions}},
            "cross_project_patterns": [],
        }
        (mem_dir / "cross-project-errors.json").write_text(json.dumps(memory))
        r = run_main("error_tracker.py", str(tmp_path), "cleanup", str(plugin_root), "3")
        assert r.returncode == 0 and json.loads(r.stdout)["cleaned"] == 2

    def test_suggestion_cli(self, run_main, tmp_path):
        """log-suggestion and suggestion-history CLI actions."""
        log_event(str(tmp_path), create_event("training_failure", "warning", "experiment", "fail"))
        r = run_main("error_tracker.py", str(tmp_path), "log-suggestion", "oom_batch_size", "session")
        assert r.returncode == 0

        log_suggestion(str(tmp_path), "high_lr_divergence", scope="session")
        r = run_main("error_tracker.py", str(tmp_path), "suggestion-history")
        assert r.returncode == 0 and len(json.loads(r.stdout)) >= 2

    def test_dead_end_cli(self, run_main, tmp_path):
        """dead-end add, list, check CLI actions."""
        entry_json = json.dumps({"technique": "test-tech", "reason": "failed"})
        r = run_main("error_tracker.py", str(tmp_path), "dead-end", "add", entry_json)
        assert r.returncode == 0

        log_dead_end(str(tmp_path), {"technique": "focal-loss", "reason": "bad"})
        r = run_main("error_tracker.py", str(tmp_path), "dead-end", "list")
        assert len(json.loads(r.stdout)) == 2

        r = run_main("error_tracker.py", str(tmp_path), "dead-end", "check", "focal-loss")
        assert json.loads(r.stdout)["is_dead_end"] is True

    def test_agenda_cli(self, run_main, tmp_path):
        """agenda init, list, update, add CLI actions."""
        ideas = json.dumps([{"id": "i1", "name": "Test idea", "priority": 7}])
        r = run_main("error_tracker.py", str(tmp_path), "agenda", "init", ideas)
        assert r.returncode == 0 and json.loads(r.stdout)["count"] == 1

        r = run_main("error_tracker.py", str(tmp_path), "agenda", "list")
        assert len(json.loads(r.stdout)) == 1

        updates = json.dumps({"status": "tried", "priority": 3})
        r = run_main("error_tracker.py", str(tmp_path), "agenda", "update", "i1", updates)
        assert json.loads(r.stdout)["updated"] is True

        idea = json.dumps({"id": "i2", "name": "Added", "priority": 9})
        r = run_main("error_tracker.py", str(tmp_path), "agenda", "add", idea)
        assert r.returncode == 0 and len(get_agenda(str(tmp_path))) == 2

    @pytest.mark.parametrize("action,extra_args", [
        ("log", ["not valid json{"]),
        ("nonexistent", []),
        ("log", []),
        ("sync", []),
        ("dead-end", ["add", "NOT_JSON"]),
        ("agenda", ["init"]),
    ], ids=[
        "log_invalid_json", "unknown_action", "log_missing_arg",
        "sync_missing_arg", "dead_end_add_invalid_json", "agenda_init_missing_json",
    ])
    def test_cli_error_handling(self, run_main, tmp_path, action, extra_args):
        r = run_main("error_tracker.py", str(tmp_path), action, *extra_args)
        assert r.returncode == 1

    def test_cli_log_invalid_validation(self, tmp_path):
        with mock.patch.object(sys, "argv", [
            "et", str(tmp_path), "log",
            '{"category":"divergence","severity":"bad_sev","source":"monitor","message":"m"}'
        ]):
            with pytest.raises(SystemExit) as exc:
                _cli_main()
            assert exc.value.code == 1
