"""Tests for error_tracker.py."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from conftest import FIXTURES, _write_result

from error_tracker import (
    validate_event,
    create_event,
    log_event,
    load_error_log,
    get_events,
    update_cross_project,
    load_cross_project,
    detect_patterns,
    detect_cross_project_patterns,
    summarize_session,
    compute_success_metrics,
    compute_proposal_outcomes,
    rank_suggestions,
    cleanup_memory,
    log_suggestion,
    get_suggestion_history,
    VALID_CATEGORIES,
    VALID_SEVERITIES,
    VALID_SOURCES,
)


# ---------------------------------------------------------------------------
# Event creation and validation
# ---------------------------------------------------------------------------


def test_create_event_minimal():
    """create_event with required args produces a valid event."""
    ev = create_event("training_failure", "critical", "experiment", "OOM")
    assert ev["category"] == "training_failure"
    assert ev["severity"] == "critical"
    assert ev["source"] == "experiment"
    assert ev["message"] == "OOM"
    assert ev["event_id"].startswith("err-")
    # timestamp should be valid ISO
    datetime.fromisoformat(ev["timestamp"])


def test_create_event_with_optional_fields():
    """create_event passes through optional keyword arguments."""
    ev = create_event(
        "divergence", "warning", "monitor", "NaN at step 5",
        exp_id="exp-003", phase=5, iteration=2,
        config={"lr": 0.1}, context={"step": 5},
    )
    assert ev["exp_id"] == "exp-003"
    assert ev["phase"] == 5
    assert ev["iteration"] == 2
    assert ev["config"] == {"lr": 0.1}
    assert ev["context"] == {"step": 5}


def test_create_event_auto_generates_unique_ids():
    """Two successive create_event calls produce different event_ids."""
    ev1 = create_event("config_error", "info", "experiment", "a")
    ev2 = create_event("config_error", "info", "experiment", "b")
    assert ev1["event_id"] != ev2["event_id"]


def test_validate_event_valid():
    """A well-formed event passes validation."""
    ev = create_event("agent_failure", "critical", "orchestrate", "timeout")
    result = validate_event(ev)
    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_event_missing_field():
    """An event missing a required field fails validation."""
    ev = {"category": "agent_failure", "severity": "critical"}
    result = validate_event(ev)
    assert result["valid"] is False
    assert any("source" in e for e in result["errors"])


@pytest.mark.parametrize("field,args", [
    ("category", ("unknown_cat", "critical", "experiment", "x")),
    ("severity", ("agent_failure", "fatal", "orchestrate", "x")),
    ("source", ("agent_failure", "critical", "unknown_src", "x")),
])
def test_create_event_invalid_field_raises(field, args):
    """create_event raises ValueError on unknown category/severity/source."""
    with pytest.raises(ValueError, match=field):
        create_event(*args)


def test_validate_event_non_dict():
    """A non-dict input fails validation."""
    result = validate_event("not a dict")
    assert result["valid"] is False
    assert len(result["errors"]) > 0


# ---------------------------------------------------------------------------
# Per-project storage
# ---------------------------------------------------------------------------


def test_log_event_creates_file(tmp_path):
    """First log_event call creates error-log.json."""
    ev = create_event("training_failure", "critical", "experiment", "crash")
    path = log_event(str(tmp_path), ev)
    assert Path(path).exists()
    data = json.loads(Path(path).read_text())
    assert len(data["events"]) == 1
    assert data["events"][0]["message"] == "crash"


def test_log_event_appends(tmp_path):
    """Second log_event appends to events list."""
    ev1 = create_event("training_failure", "critical", "experiment", "first")
    ev2 = create_event("divergence", "warning", "monitor", "second")
    log_event(str(tmp_path), ev1)
    path = log_event(str(tmp_path), ev2)
    data = json.loads(Path(path).read_text())
    assert len(data["events"]) == 2
    assert data["events"][1]["message"] == "second"


def test_log_event_updates_summary(tmp_path):
    """Summary counts are correct after multiple events."""
    log_event(str(tmp_path), create_event("training_failure", "critical", "experiment", "a"))
    log_event(str(tmp_path), create_event("divergence", "warning", "monitor", "b"))
    log_event(str(tmp_path), create_event("training_failure", "warning", "experiment", "c"))
    data = load_error_log(str(tmp_path))
    assert data["summary"]["total_events"] == 3
    assert data["summary"]["by_category"]["training_failure"] == 2
    assert data["summary"]["by_category"]["divergence"] == 1
    assert data["summary"]["by_severity"]["critical"] == 1
    assert data["summary"]["by_severity"]["warning"] == 2


def test_log_event_creates_reports_dir(tmp_path):
    """log_event creates reports/ subdirectory if it doesn't exist."""
    exp_root = tmp_path / "experiments"
    exp_root.mkdir()
    ev = create_event("config_error", "info", "experiment", "test")
    path = log_event(str(exp_root), ev)
    assert (exp_root / "reports").is_dir()
    assert Path(path).exists()


def test_load_error_log_valid(tmp_path):
    """load_error_log returns correct data after logging."""
    ev = create_event("agent_failure", "critical", "orchestrate", "timeout")
    log_event(str(tmp_path), ev)
    data = load_error_log(str(tmp_path))
    assert data is not None
    assert len(data["events"]) == 1


def test_load_error_log_not_found(tmp_path):
    """load_error_log returns None when no log file exists."""
    assert load_error_log(str(tmp_path)) is None


def test_load_error_log_corrupt(tmp_path):
    """load_error_log returns None on corrupt JSON."""
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "error-log.json").write_text("{bad json")
    assert load_error_log(str(tmp_path)) is None


def test_get_events_no_filter(tmp_path):
    """get_events with no filter returns all events."""
    log_event(str(tmp_path), create_event("training_failure", "critical", "experiment", "a"))
    log_event(str(tmp_path), create_event("divergence", "warning", "monitor", "b"))
    events = get_events(str(tmp_path))
    assert len(events) == 2


def test_get_events_filter_category(tmp_path):
    """get_events filters by category correctly."""
    log_event(str(tmp_path), create_event("training_failure", "critical", "experiment", "a"))
    log_event(str(tmp_path), create_event("divergence", "warning", "monitor", "b"))
    log_event(str(tmp_path), create_event("training_failure", "warning", "experiment", "c"))
    events = get_events(str(tmp_path), category="training_failure")
    assert len(events) == 2
    assert all(e["category"] == "training_failure" for e in events)


def test_get_events_filter_severity(tmp_path):
    """get_events filters by severity correctly."""
    log_event(str(tmp_path), create_event("training_failure", "critical", "experiment", "a"))
    log_event(str(tmp_path), create_event("divergence", "warning", "monitor", "b"))
    events = get_events(str(tmp_path), severity="critical")
    assert len(events) == 1
    assert events[0]["severity"] == "critical"


def test_get_events_filter_both(tmp_path):
    """get_events filters by both category and severity."""
    log_event(str(tmp_path), create_event("training_failure", "critical", "experiment", "a"))
    log_event(str(tmp_path), create_event("training_failure", "warning", "experiment", "b"))
    log_event(str(tmp_path), create_event("divergence", "critical", "monitor", "c"))
    events = get_events(str(tmp_path), category="training_failure", severity="critical")
    assert len(events) == 1
    assert events[0]["message"] == "a"


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def test_detect_patterns_empty():
    """No events returns empty patterns list."""
    assert detect_patterns([]) == []


def test_detect_patterns_divergence_cluster():
    """3+ divergence events with high LR detected as pattern."""
    events = [
        create_event("divergence", "warning", "monitor", "NaN",
                      config={"lr": 0.1, "batch_size": 32}),
        create_event("divergence", "warning", "monitor", "NaN",
                      config={"lr": 0.2, "batch_size": 32}),
        create_event("divergence", "warning", "monitor", "explosion",
                      config={"lr": 0.05, "batch_size": 64}),
    ]
    patterns = detect_patterns(events)
    ids = [p["pattern_id"] for p in patterns]
    assert "high_lr_divergence" in ids


def test_detect_patterns_oom_cluster():
    """2+ OOM events with same batch_size detected as pattern."""
    events = [
        create_event("training_failure", "critical", "experiment", "OOM",
                      config={"lr": 0.01, "batch_size": 256},
                      context={"error_type": "oom"}),
        create_event("training_failure", "critical", "experiment", "OOM",
                      config={"lr": 0.001, "batch_size": 256},
                      context={"error_type": "oom"}),
    ]
    patterns = detect_patterns(events)
    ids = [p["pattern_id"] for p in patterns]
    assert "oom_batch_size" in ids


def test_detect_patterns_wasted_budget():
    """All-failed batch detected as wasted budget pattern."""
    events = [
        create_event("pipeline_inefficiency", "warning", "analyze",
                      "All 3 experiments in batch diverged",
                      context={"experiments_wasted": 3}),
        create_event("pipeline_inefficiency", "warning", "analyze",
                      "All 2 experiments in batch diverged",
                      context={"experiments_wasted": 2}),
    ]
    patterns = detect_patterns(events)
    ids = [p["pattern_id"] for p in patterns]
    assert "wasted_budget" in ids


def test_detect_patterns_no_false_positives():
    """Single divergence event should not trigger a pattern."""
    events = [
        create_event("divergence", "warning", "monitor", "NaN",
                      config={"lr": 0.001, "batch_size": 32}),
    ]
    patterns = detect_patterns(events)
    # Should not produce high_lr_divergence with only 1 event
    ids = [p["pattern_id"] for p in patterns]
    assert "high_lr_divergence" not in ids


def test_detect_patterns_from_fixture():
    """Pattern detection works on the sample fixture file."""
    data = json.loads((FIXTURES / "sample_error_log.json").read_text())
    patterns = detect_patterns(data["events"])
    # Fixture has 3 divergence events (lr=0.1, 0.05, 0.2) and 2 OOM (batch=256)
    ids = [p["pattern_id"] for p in patterns]
    assert "high_lr_divergence" in ids
    assert "oom_batch_size" in ids


# ---------------------------------------------------------------------------
# Cross-project storage
# ---------------------------------------------------------------------------


def test_update_cross_project_creates_file(tmp_path):
    """First sync creates the cross-project memory file."""
    exp_root = tmp_path / "project" / "experiments"
    exp_root.mkdir(parents=True)
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()
    memory_dir = plugin_root / "memory"
    memory_dir.mkdir()

    ev = create_event("training_failure", "critical", "experiment", "crash")
    log_event(str(exp_root), ev)
    path = update_cross_project(str(plugin_root), str(tmp_path / "project"), str(exp_root))
    assert Path(path).exists()
    data = json.loads(Path(path).read_text())
    assert "projects" in data
    assert len(data["projects"]) == 1


def test_update_cross_project_appends_session(tmp_path):
    """Second sync for same project appends a session entry."""
    exp_root = tmp_path / "project" / "experiments"
    exp_root.mkdir(parents=True)
    plugin_root = tmp_path / "plugin"
    (plugin_root / "memory").mkdir(parents=True)

    log_event(str(exp_root), create_event("training_failure", "critical", "experiment", "a"))
    update_cross_project(str(plugin_root), str(tmp_path / "project"), str(exp_root))

    # Simulate a new session by updating session_start
    log_data = load_error_log(str(exp_root))
    log_data["session_start"] = "2026-03-07T10:00:00+00:00"
    reports = Path(exp_root) / "reports"
    (reports / "error-log.json").write_text(json.dumps(log_data))

    path = update_cross_project(str(plugin_root), str(tmp_path / "project"), str(exp_root))
    data = json.loads(Path(path).read_text())
    project_id = list(data["projects"].keys())[0]
    assert len(data["projects"][project_id]["sessions"]) == 2


def test_load_cross_project_valid(tmp_path):
    """load_cross_project returns data after sync."""
    exp_root = tmp_path / "project" / "experiments"
    exp_root.mkdir(parents=True)
    plugin_root = tmp_path / "plugin"
    (plugin_root / "memory").mkdir(parents=True)

    log_event(str(exp_root), create_event("divergence", "warning", "monitor", "nan"))
    update_cross_project(str(plugin_root), str(tmp_path / "project"), str(exp_root))
    data = load_cross_project(str(plugin_root))
    assert data is not None
    assert "projects" in data


def test_load_cross_project_not_found(tmp_path):
    """load_cross_project returns None when no memory file exists."""
    assert load_cross_project(str(tmp_path)) is None


def test_load_cross_project_corrupt(tmp_path):
    """load_cross_project returns None on corrupt JSON."""
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "cross-project-errors.json").write_text("not json{")
    assert load_cross_project(str(tmp_path)) is None


def test_update_cross_project_creates_memory_dir(tmp_path):
    """update_cross_project creates memory/ directory if missing."""
    exp_root = tmp_path / "project" / "experiments"
    exp_root.mkdir(parents=True)
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()
    # No memory/ dir yet

    log_event(str(exp_root), create_event("config_error", "info", "experiment", "x"))
    path = update_cross_project(str(plugin_root), str(tmp_path / "project"), str(exp_root))
    assert (plugin_root / "memory").is_dir()
    assert Path(path).exists()


# ---------------------------------------------------------------------------
# Cross-project pattern detection
# ---------------------------------------------------------------------------


def test_detect_cross_project_patterns_empty():
    """No projects returns empty patterns."""
    memory = {"version": 1, "projects": {}, "cross_project_patterns": []}
    assert detect_cross_project_patterns(memory) == []


def test_detect_cross_project_patterns_shared():
    """Same pattern across 2+ projects is detected."""
    memory = {
        "version": 1,
        "projects": {
            "proj1": {
                "project_path": "/a",
                "sessions": [{"patterns_detected": ["high_lr_divergence", "oom_batch_size"]}],
            },
            "proj2": {
                "project_path": "/b",
                "sessions": [{"patterns_detected": ["high_lr_divergence"]}],
            },
            "proj3": {
                "project_path": "/c",
                "sessions": [{"patterns_detected": ["wasted_budget"]}],
            },
        },
        "cross_project_patterns": [],
    }
    patterns = detect_cross_project_patterns(memory)
    ids = [p["pattern_id"] for p in patterns]
    assert "high_lr_divergence" in ids
    # oom_batch_size only in 1 project, should not appear
    assert "oom_batch_size" not in ids


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------


def test_summarize_session_basic(tmp_path):
    """Session summary has correct counts and structure."""
    log_event(str(tmp_path), create_event("training_failure", "critical", "experiment", "a"))
    log_event(str(tmp_path), create_event("divergence", "warning", "monitor", "b"))
    summary = summarize_session(str(tmp_path))
    assert summary["total_events"] == 2
    assert summary["by_category"]["training_failure"] == 1
    assert summary["by_severity"]["critical"] == 1
    assert "patterns_detected" in summary


def test_summarize_session_empty(tmp_path):
    """Empty log returns zero-count summary."""
    summary = summarize_session(str(tmp_path))
    assert summary["total_events"] == 0
    assert summary["by_category"] == {}
    assert summary["by_severity"] == {}


def test_summarize_session_includes_patterns(tmp_path):
    """Summary includes detected pattern IDs."""
    # Log enough divergences to trigger a pattern
    for lr in [0.1, 0.2, 0.05]:
        log_event(str(tmp_path), create_event(
            "divergence", "warning", "monitor", "NaN",
            config={"lr": lr, "batch_size": 32},
        ))
    summary = summarize_session(str(tmp_path))
    assert "high_lr_divergence" in summary["patterns_detected"]


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_cli_no_args(run_main):
    """No arguments prints usage and exits 1."""
    r = run_main("error_tracker.py")
    assert r.returncode == 1
    assert "usage" in r.stderr.lower() or "usage" in r.stdout.lower()


def test_cli_log_event(run_main, tmp_path):
    """CLI log action creates error-log.json."""
    ev_json = json.dumps({
        "category": "training_failure",
        "severity": "critical",
        "source": "experiment",
        "message": "test crash",
    })
    r = run_main("error_tracker.py", str(tmp_path), "log", ev_json)
    assert r.returncode == 0
    assert load_error_log(str(tmp_path)) is not None


def test_cli_show_events(run_main, tmp_path):
    """CLI show action outputs events as JSON."""
    log_event(str(tmp_path), create_event("divergence", "warning", "monitor", "nan"))
    r = run_main("error_tracker.py", str(tmp_path), "show")
    assert r.returncode == 0
    events = json.loads(r.stdout)
    assert len(events) == 1


def test_cli_show_filtered(run_main, tmp_path):
    """CLI show with category filter works."""
    log_event(str(tmp_path), create_event("divergence", "warning", "monitor", "a"))
    log_event(str(tmp_path), create_event("training_failure", "critical", "experiment", "b"))
    r = run_main("error_tracker.py", str(tmp_path), "show", "divergence")
    assert r.returncode == 0
    events = json.loads(r.stdout)
    assert len(events) == 1
    assert events[0]["category"] == "divergence"


def test_cli_patterns(run_main, tmp_path):
    """CLI patterns action outputs detected patterns as JSON."""
    for lr in [0.1, 0.2, 0.05]:
        log_event(str(tmp_path), create_event(
            "divergence", "warning", "monitor", "NaN",
            config={"lr": lr, "batch_size": 32},
        ))
    r = run_main("error_tracker.py", str(tmp_path), "patterns")
    assert r.returncode == 0
    patterns = json.loads(r.stdout)
    assert len(patterns) > 0


def test_cli_summary(run_main, tmp_path):
    """CLI summary action outputs session summary as JSON."""
    log_event(str(tmp_path), create_event("agent_failure", "critical", "orchestrate", "x"))
    r = run_main("error_tracker.py", str(tmp_path), "summary")
    assert r.returncode == 0
    summary = json.loads(r.stdout)
    assert summary["total_events"] == 1


def test_cli_sync(run_main, tmp_path):
    """CLI sync action creates cross-project memory file."""
    exp_root = tmp_path / "project" / "experiments"
    exp_root.mkdir(parents=True)
    plugin_root = tmp_path / "plugin"
    (plugin_root / "memory").mkdir(parents=True)

    log_event(str(exp_root), create_event("divergence", "warning", "monitor", "x"))
    r = run_main("error_tracker.py", str(exp_root), "sync", str(plugin_root))
    assert r.returncode == 0
    assert load_cross_project(str(plugin_root)) is not None


def test_cli_invalid_json(run_main, tmp_path):
    """CLI log with invalid JSON exits 1."""
    r = run_main("error_tracker.py", str(tmp_path), "log", "not valid json{")
    assert r.returncode == 1


# ---------------------------------------------------------------------------
# Success metrics
# ---------------------------------------------------------------------------


def test_compute_success_metrics_basic(tmp_path):
    """Mixed results produce correct success/failure counts."""
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results, "baseline", "completed", {"lr": 0.001}, {"acc": 70.0})
    _write_result(results, "exp-001", "completed", {"lr": 0.001}, {"acc": 75.0})
    _write_result(results, "exp-002", "completed", {"lr": 0.01}, {"acc": 68.0})
    _write_result(results, "exp-003", "failed", {"lr": 0.1}, {"acc": 0.0})
    _write_result(results, "exp-004", "diverged", {"lr": 0.5}, {})

    m = compute_success_metrics(str(tmp_path), "acc", lower_is_better=False)
    assert m["total_experiments"] == 4  # excludes baseline
    assert m["completed"] == 2
    assert m["failed"] == 1
    assert m["diverged"] == 1
    assert m["improvement_rate"] > 0  # exp-001 beat baseline


def test_compute_success_metrics_no_baseline(tmp_path):
    """Missing baseline returns metrics with improvement_rate=None."""
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results, "exp-001", "completed", {"lr": 0.001}, {"acc": 75.0})

    m = compute_success_metrics(str(tmp_path), "acc", lower_is_better=False)
    assert m["total_experiments"] == 1
    assert m["improvement_rate"] is None


def test_compute_success_metrics_all_failed(tmp_path):
    """All-failed experiments return 0% success rate."""
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results, "baseline", "completed", {"lr": 0.001}, {"loss": 1.0})
    _write_result(results, "exp-001", "failed", {"lr": 0.1}, {})
    _write_result(results, "exp-002", "diverged", {"lr": 0.5}, {})

    m = compute_success_metrics(str(tmp_path), "loss", lower_is_better=True)
    assert m["success_rate"] == 0.0
    assert m["completed"] == 0


def test_compute_success_metrics_duration(tmp_path):
    """Duration analysis computes averages for completed vs failed."""
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results, "baseline", "completed", {}, {"loss": 1.0})
    _write_result(results, "exp-001", "completed", {}, {"loss": 0.5},
                  duration_seconds=3600)
    _write_result(results, "exp-002", "completed", {}, {"loss": 0.8},
                  duration_seconds=1800)
    _write_result(results, "exp-003", "failed", {}, {},
                  duration_seconds=300)
    _write_result(results, "exp-004", "diverged", {}, {},
                  duration_seconds=120)

    m = compute_success_metrics(str(tmp_path), "loss", lower_is_better=True)
    assert m["avg_duration_completed"] == 2700.0  # (3600+1800)/2
    assert m["avg_duration_failed"] == 210.0  # (300+120)/2
    assert m["time_wasted_on_failures_pct"] > 0


def test_compute_success_metrics_empty(tmp_path):
    """No experiment results returns zero counts."""
    results = tmp_path / "results"
    results.mkdir()

    m = compute_success_metrics(str(tmp_path), "acc", lower_is_better=False)
    assert m["total_experiments"] == 0
    assert m["completed"] == 0


def test_compute_success_metrics_top_configs(tmp_path):
    """Top configs are sorted by improvement over baseline."""
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results, "baseline", "completed", {"lr": 0.001}, {"acc": 70.0})
    _write_result(results, "exp-001", "completed", {"lr": 0.0005}, {"acc": 80.0})
    _write_result(results, "exp-002", "completed", {"lr": 0.0003}, {"acc": 75.0})

    m = compute_success_metrics(str(tmp_path), "acc", lower_is_better=False)
    assert len(m["top_configs"]) == 2
    # Best improvement first
    assert m["top_configs"][0]["exp_id"] == "exp-001"


def test_compute_success_metrics_nan_baseline(tmp_path):
    """NaN baseline metric value is treated as missing (no improvement calc)."""
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results, "baseline", "completed", {"lr": 0.001}, {"acc": float("nan")})
    _write_result(results, "exp-001", "completed", {"lr": 0.01}, {"acc": 80.0})

    m = compute_success_metrics(str(tmp_path), "acc", lower_is_better=False)
    assert m["improvement_rate"] is None


def test_compute_success_metrics_inf_experiment(tmp_path):
    """Inf experiment metric values are excluded from improvement calculation."""
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results, "baseline", "completed", {"lr": 0.001}, {"acc": 70.0})
    _write_result(results, "exp-001", "completed", {"lr": 0.01}, {"acc": float("inf")})
    _write_result(results, "exp-002", "completed", {"lr": 0.005}, {"acc": 75.0})

    m = compute_success_metrics(str(tmp_path), "acc", lower_is_better=False)
    # Only exp-002 should count (exp-001 has inf, should be skipped)
    assert m["improvement_rate"] is not None
    assert m["best_improvement_pct"] is not None


# ---------------------------------------------------------------------------
# Proposal outcomes
# ---------------------------------------------------------------------------


def test_compute_proposal_outcomes_with_manifest(tmp_path):
    """Cross-references manifest proposals with experiment results."""
    results = tmp_path / "results"
    results.mkdir()
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
    _write_result(results, "baseline", "completed", {"lr": 0.001}, {"acc": 70.0})
    _write_result(results, "exp-001", "completed", {"lr": 0.001}, {"acc": 78.0},
                  code_branch="ml-opt/perceptual-loss",
                  code_proposal="perceptual-loss")
    _write_result(results, "exp-002", "completed", {"lr": 0.01}, {"acc": 72.0},
                  code_branch="ml-opt/perceptual-loss",
                  code_proposal="perceptual-loss")

    p = compute_proposal_outcomes(str(tmp_path), "acc", lower_is_better=False)
    assert p["implementation_stats"]["validated"] == 1
    assert p["implementation_stats"]["validation_failed"] == 1
    assert len(p["research_proposals"]) == 1
    assert p["research_proposals"][0]["name"] == "perceptual-loss"
    assert p["research_proposals"][0]["experiments"] == 2
    assert p["research_proposals"][0]["beat_baseline"] >= 1


def test_compute_proposal_outcomes_no_manifest(tmp_path):
    """Missing manifest returns empty implementation stats."""
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results, "baseline", "completed", {}, {"acc": 70.0})
    _write_result(results, "exp-001", "completed", {}, {"acc": 75.0})

    p = compute_proposal_outcomes(str(tmp_path), "acc", lower_is_better=False)
    assert p["implementation_stats"]["total_proposals"] == 0
    assert p["research_proposals"] == []


def test_compute_proposal_outcomes_hp_stats(tmp_path):
    """HP proposal stats count proposed vs run configs."""
    results = tmp_path / "results"
    results.mkdir()
    configs_dir = results / "proposed-configs"
    configs_dir.mkdir()
    # 3 proposed configs
    for i in range(3):
        (configs_dir / f"exp-{i+1:03d}.json").write_text(
            json.dumps({"lr": 0.001 * (i + 1)}))
    _write_result(results, "baseline", "completed", {}, {"loss": 1.0})
    _write_result(results, "exp-001", "completed", {"lr": 0.001}, {"loss": 0.5})
    _write_result(results, "exp-002", "diverged", {"lr": 0.002}, {})

    p = compute_proposal_outcomes(str(tmp_path), "loss", lower_is_better=True)
    assert p["hp_proposals"]["total_proposed"] == 3
    assert p["hp_proposals"]["total_run"] == 2


# ---------------------------------------------------------------------------
# CLI tests for new modes
# ---------------------------------------------------------------------------


def test_cli_success(run_main, tmp_path):
    """CLI success action returns valid JSON."""
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results, "baseline", "completed", {}, {"acc": 70.0})
    _write_result(results, "exp-001", "completed", {}, {"acc": 75.0})

    r = run_main("error_tracker.py", str(tmp_path), "success", "acc", "false")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["total_experiments"] == 1


def test_cli_proposals(run_main, tmp_path):
    """CLI proposals action returns valid JSON."""
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results, "baseline", "completed", {}, {"acc": 70.0})

    r = run_main("error_tracker.py", str(tmp_path), "proposals", "acc", "false")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert "implementation_stats" in data


# ---------------------------------------------------------------------------
# New categories (A1)
# ---------------------------------------------------------------------------


def test_validate_event_research_failure_valid():
    """research_failure is a valid category."""
    ev = create_event("research_failure", "warning", "research", "No results")
    result = validate_event(ev)
    assert result["valid"] is True


def test_validate_event_timeout_valid():
    """timeout is a valid category."""
    ev = create_event("timeout", "critical", "orchestrate", "Agent timed out")
    result = validate_event(ev)
    assert result["valid"] is True


def test_validate_event_resource_error_valid():
    """resource_error is a valid category."""
    ev = create_event("resource_error", "warning", "baseline", "No GPU detected")
    result = validate_event(ev)
    assert result["valid"] is True


# ---------------------------------------------------------------------------
# Duration field (A2)
# ---------------------------------------------------------------------------


def test_create_event_with_duration():
    """duration_seconds is included when provided."""
    ev = create_event("timeout", "critical", "orchestrate", "Timed out",
                      duration_seconds=3600.5)
    assert ev["duration_seconds"] == 3600.5


def test_create_event_without_duration():
    """duration_seconds is omitted when not provided."""
    ev = create_event("agent_failure", "critical", "orchestrate", "crash")
    assert "duration_seconds" not in ev


# ---------------------------------------------------------------------------
# New pattern detectors (A3)
# ---------------------------------------------------------------------------


def test_detect_patterns_early_failure_cluster():
    """3+ failures in same non-Phase-5 phase triggers early_failure_cluster."""
    events = [
        create_event("training_failure", "critical", "baseline", "fail1", phase=2),
        create_event("config_error", "warning", "baseline", "fail2", phase=2),
        create_event("training_failure", "critical", "baseline", "fail3", phase=2),
    ]
    patterns = detect_patterns(events)
    ids = [p["pattern_id"] for p in patterns]
    assert "early_failure_cluster" in ids


def test_detect_patterns_no_early_failure_phase5():
    """Failures in Phase 5 do NOT trigger early_failure_cluster."""
    events = [
        create_event("training_failure", "critical", "experiment", "f1", phase=5),
        create_event("divergence", "warning", "monitor", "f2", phase=5),
        create_event("training_failure", "critical", "experiment", "f3", phase=5),
    ]
    patterns = detect_patterns(events)
    ids = [p["pattern_id"] for p in patterns]
    assert "early_failure_cluster" not in ids


def test_detect_patterns_early_failure_below_threshold():
    """2 failures in Phase 2 does not trigger early_failure_cluster."""
    events = [
        create_event("training_failure", "critical", "baseline", "f1", phase=2),
        create_event("config_error", "warning", "baseline", "f2", phase=2),
    ]
    patterns = detect_patterns(events)
    ids = [p["pattern_id"] for p in patterns]
    assert "early_failure_cluster" not in ids


def test_detect_patterns_hp_interaction():
    """3+ failures with same LR-bucket + batch_size triggers hp_interaction_failure."""
    events = [
        create_event("divergence", "warning", "monitor", "NaN",
                      config={"lr": 0.05, "batch_size": 256}),
        create_event("training_failure", "critical", "experiment", "OOM",
                      config={"lr": 0.02, "batch_size": 256}),
        create_event("divergence", "warning", "monitor", "explosion",
                      config={"lr": 0.03, "batch_size": 256}),
    ]
    patterns = detect_patterns(events)
    ids = [p["pattern_id"] for p in patterns]
    assert "hp_interaction_failure" in ids


def test_detect_patterns_hp_interaction_different_combos():
    """Failures with varied LR-bucket × batch_size do not trigger."""
    events = [
        create_event("divergence", "warning", "monitor", "NaN",
                      config={"lr": 0.05, "batch_size": 256}),  # high, 256
        create_event("divergence", "warning", "monitor", "NaN",
                      config={"lr": 0.0005, "batch_size": 32}),  # low, 32
        create_event("divergence", "warning", "monitor", "NaN",
                      config={"lr": 0.005, "batch_size": 64}),  # medium, 64
    ]
    patterns = detect_patterns(events)
    ids = [p["pattern_id"] for p in patterns]
    assert "hp_interaction_failure" not in ids


def test_detect_patterns_hp_interaction_missing_config():
    """Events without config fields are gracefully skipped."""
    events = [
        create_event("divergence", "warning", "monitor", "NaN"),
        create_event("divergence", "warning", "monitor", "NaN"),
        create_event("divergence", "warning", "monitor", "NaN"),
    ]
    patterns = detect_patterns(events)
    ids = [p["pattern_id"] for p in patterns]
    assert "hp_interaction_failure" not in ids


def test_detect_patterns_temporal_early():
    """60%+ of failures in iteration 1 triggers temporal_failure_cluster."""
    events = [
        create_event("training_failure", "critical", "experiment", "f1", iteration=1),
        create_event("divergence", "warning", "monitor", "f2", iteration=1),
        create_event("training_failure", "critical", "experiment", "f3", iteration=1),
        create_event("divergence", "warning", "monitor", "f4", iteration=2),
        create_event("training_failure", "critical", "experiment", "f5", iteration=1),
    ]
    patterns = detect_patterns(events)
    ids = [p["pattern_id"] for p in patterns]
    assert "temporal_failure_cluster" in ids


def test_detect_patterns_temporal_spread():
    """Evenly spread failures do not trigger temporal_failure_cluster."""
    events = [
        create_event("training_failure", "critical", "experiment", "f1", iteration=1),
        create_event("training_failure", "critical", "experiment", "f2", iteration=2),
        create_event("training_failure", "critical", "experiment", "f3", iteration=3),
        create_event("training_failure", "critical", "experiment", "f4", iteration=4),
    ]
    patterns = detect_patterns(events)
    ids = [p["pattern_id"] for p in patterns]
    assert "temporal_failure_cluster" not in ids


def test_detect_patterns_temporal_too_few():
    """Fewer than 4 events do not trigger temporal_failure_cluster."""
    events = [
        create_event("training_failure", "critical", "experiment", "f1", iteration=1),
        create_event("training_failure", "critical", "experiment", "f2", iteration=1),
        create_event("training_failure", "critical", "experiment", "f3", iteration=1),
    ]
    patterns = detect_patterns(events)
    ids = [p["pattern_id"] for p in patterns]
    assert "temporal_failure_cluster" not in ids


def test_detect_patterns_timeout():
    """2+ timeout events triggers timeout_pattern."""
    events = [
        create_event("timeout", "critical", "orchestrate", "Agent timed out",
                      duration_seconds=600),
        create_event("timeout", "warning", "orchestrate", "Training timed out",
                      duration_seconds=7200),
    ]
    patterns = detect_patterns(events)
    ids = [p["pattern_id"] for p in patterns]
    assert "timeout_pattern" in ids


def test_detect_patterns_timeout_single():
    """1 timeout event does not trigger timeout_pattern."""
    events = [
        create_event("timeout", "critical", "orchestrate", "Agent timed out"),
    ]
    patterns = detect_patterns(events)
    ids = [p["pattern_id"] for p in patterns]
    assert "timeout_pattern" not in ids


def test_detect_patterns_nan_inf_lr_excluded():
    """NaN/Inf LR values in divergence events are excluded from pattern stats."""
    events = []
    for lr in [0.1, 0.2, float("nan"), float("inf")]:
        events.append(create_event(
            "divergence", "warning", "monitor", "NaN",
            config={"lr": lr, "batch_size": 32},
        ))
    patterns = detect_patterns(events)
    lr_pattern = [p for p in patterns if p["pattern_id"] == "high_lr_divergence"]
    assert len(lr_pattern) == 1
    # avg should be computed from only 0.1 and 0.2 (the finite values)
    assert "0.1500" in lr_pattern[0]["description"]


# ---------------------------------------------------------------------------
# Suggestion ranking (A4)
# ---------------------------------------------------------------------------


def test_rank_suggestions_order():
    """Higher severity patterns rank first."""
    patterns = [
        {"pattern_id": "redundant_configs", "description": "dup", "occurrences": 5,
         "suggested_action": "widen"},
        {"pattern_id": "oom_batch_size", "description": "oom", "occurrences": 2,
         "suggested_action": "reduce bs"},
    ]
    ranked = rank_suggestions(patterns)
    # oom_batch_size (weight 3, occ 2 = 6) > redundant_configs (weight 1, occ 5 = 5)
    assert ranked[0]["pattern_id"] == "oom_batch_size"
    assert "score" in ranked[0]


def test_rank_suggestions_cross_project_boost():
    """Patterns in cross-project get 1.5x boost."""
    patterns = [
        {"pattern_id": "high_lr_divergence", "description": "div", "occurrences": 2,
         "suggested_action": "lower lr"},
        {"pattern_id": "wasted_budget", "description": "waste", "occurrences": 3,
         "suggested_action": "tighten"},
    ]
    cross = [{"pattern_id": "wasted_budget", "projects_affected": 3}]
    ranked = rank_suggestions(patterns, cross_project_patterns=cross)
    # wasted_budget: weight 1 * 3 * 1.5 = 4.5
    # high_lr_divergence: weight 2 * 2 * 1.0 = 4.0
    assert ranked[0]["pattern_id"] == "wasted_budget"


def test_rank_suggestions_cross_project_boost_score_value():
    """Cross-project boost applies 1.5x multiplier to score."""
    patterns = [
        {"pattern_id": "wasted_budget", "description": "waste",
         "occurrences": 2, "suggested_action": "tighten"},
    ]
    ranked_no_boost = rank_suggestions(patterns)
    cross = [{"pattern_id": "wasted_budget", "projects_affected": 2}]
    ranked_boosted = rank_suggestions(patterns, cross_project_patterns=cross)
    assert ranked_boosted[0]["score"] == ranked_no_boost[0]["score"] * 1.5


def test_rank_suggestions_empty():
    """Empty input returns empty list."""
    assert rank_suggestions([]) == []


def test_cli_rank(run_main, tmp_path):
    """CLI rank action returns sorted JSON."""
    # Create enough events to trigger patterns
    for lr in [0.1, 0.2, 0.05]:
        log_event(str(tmp_path), create_event(
            "divergence", "warning", "monitor", "NaN",
            config={"lr": lr, "batch_size": 32},
        ))
    r = run_main("error_tracker.py", str(tmp_path), "rank")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert len(data) > 0
    assert "score" in data[0]


# ---------------------------------------------------------------------------
# Significance in rank_suggestions (Phase A)
# ---------------------------------------------------------------------------


def test_rank_with_total_experiments_adds_significance():
    """rank_suggestions with total_experiments adds significance field."""
    patterns = [
        {"pattern_id": "oom_batch_size", "description": "oom", "occurrences": 3,
         "suggested_action": "reduce bs"},
    ]
    ranked = rank_suggestions(patterns, total_experiments=100)
    assert "significance" in ranked[0]
    assert ranked[0]["significance"] == 0.03


def test_rank_without_total_experiments_no_significance():
    """rank_suggestions without total_experiments omits significance field."""
    patterns = [
        {"pattern_id": "oom_batch_size", "description": "oom", "occurrences": 3,
         "suggested_action": "reduce bs"},
    ]
    ranked = rank_suggestions(patterns)
    assert "significance" not in ranked[0]


def test_rank_significance_calculation():
    """significance = occurrences / total_experiments, rounded to 3 places."""
    patterns = [
        {"pattern_id": "high_lr_divergence", "description": "div", "occurrences": 7,
         "suggested_action": "lower lr"},
    ]
    ranked = rank_suggestions(patterns, total_experiments=30)
    assert ranked[0]["significance"] == round(7 / 30, 3)


def test_cli_rank_with_total_experiments(run_main, tmp_path):
    """CLI rank action with total_experiments passes it through."""
    for lr in [0.1, 0.2, 0.05]:
        log_event(str(tmp_path), create_event(
            "divergence", "warning", "monitor", "NaN",
            config={"lr": lr, "batch_size": 32},
        ))
    r = run_main("error_tracker.py", str(tmp_path), "rank", "50")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert len(data) > 0
    assert "significance" in data[0]


def test_cli_rank_with_cross_project(run_main, tmp_path):
    """CLI rank action with plugin_root loads cross-project patterns for boost."""
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()
    # Create two "projects" that share the same pattern to trigger cross-project detection
    for proj_name in ["proj_a", "proj_b"]:
        proj = tmp_path / proj_name
        proj.mkdir()
        for lr in [0.1, 0.2, 0.05]:
            log_event(str(proj), create_event(
                "divergence", "warning", "monitor", "NaN",
                config={"lr": lr, "batch_size": 32},
            ))
        update_cross_project(str(plugin_root), str(proj), str(proj))
    # Now rank from proj_a with cross-project data
    proj_a = tmp_path / "proj_a"
    r_boosted = run_main("error_tracker.py", str(proj_a), "rank", "50", str(plugin_root))
    assert r_boosted.returncode == 0
    boosted = json.loads(r_boosted.stdout)
    # Rank without cross-project
    r_plain = run_main("error_tracker.py", str(proj_a), "rank", "50")
    assert r_plain.returncode == 0
    plain = json.loads(r_plain.stdout)
    # Cross-project boost should make scores higher
    assert len(boosted) > 0
    assert len(plain) > 0
    assert boosted[0]["score"] > plain[0]["score"]


# ---------------------------------------------------------------------------
# Session deduplication (Phase B)
# ---------------------------------------------------------------------------


def _sync_project(tmp_path, plugin_root, n_events=2):
    """Helper: log events and sync to cross-project memory."""
    for i in range(n_events):
        log_event(str(tmp_path), create_event(
            "training_failure", "warning", "experiment", f"fail-{i}",
        ))
    return update_cross_project(str(plugin_root), str(tmp_path), str(tmp_path))


def test_sync_deduplicates_same_session(tmp_path):
    """Syncing the same session twice should not create duplicate entries."""
    plugin_root = tmp_path / "plugin"
    exp_root = tmp_path / "project"
    exp_root.mkdir()
    log_event(str(exp_root), create_event(
        "training_failure", "warning", "experiment", "fail",
    ))
    # Sync twice
    update_cross_project(str(plugin_root), str(exp_root), str(exp_root))
    update_cross_project(str(plugin_root), str(exp_root), str(exp_root))
    memory = load_cross_project(str(plugin_root))
    # Should have exactly 1 session, not 2
    for proj_data in memory["projects"].values():
        assert len(proj_data["sessions"]) == 1


def test_sync_updates_existing_session(tmp_path):
    """Syncing with more events should update the existing session entry."""
    plugin_root = tmp_path / "plugin"
    exp_root = tmp_path / "project"
    exp_root.mkdir()
    log_event(str(exp_root), create_event(
        "training_failure", "warning", "experiment", "fail-1",
    ))
    update_cross_project(str(plugin_root), str(exp_root), str(exp_root))
    # Add another event and re-sync
    log_event(str(exp_root), create_event(
        "divergence", "warning", "monitor", "NaN",
    ))
    update_cross_project(str(plugin_root), str(exp_root), str(exp_root))
    memory = load_cross_project(str(plugin_root))
    for proj_data in memory["projects"].values():
        assert len(proj_data["sessions"]) == 1
        assert proj_data["sessions"][0]["event_count"] == 2


def test_sync_different_sessions_not_deduped(tmp_path):
    """Sessions with different session_start should both be kept."""
    plugin_root = tmp_path / "plugin"
    exp_root1 = tmp_path / "project1"
    exp_root1.mkdir()
    exp_root2 = tmp_path / "project2"
    exp_root2.mkdir()
    log_event(str(exp_root1), create_event(
        "training_failure", "warning", "experiment", "fail-a",
    ))
    log_event(str(exp_root2), create_event(
        "training_failure", "warning", "experiment", "fail-b",
    ))
    # Sync two different projects
    update_cross_project(str(plugin_root), str(exp_root1), str(exp_root1))
    update_cross_project(str(plugin_root), str(exp_root2), str(exp_root2))
    memory = load_cross_project(str(plugin_root))
    assert len(memory["projects"]) == 2


# ---------------------------------------------------------------------------
# Cross-project memory cleanup (Phase C)
# ---------------------------------------------------------------------------


def test_cleanup_respects_max_sessions(tmp_path):
    """cleanup_memory keeps only the last max_sessions entries per project."""
    plugin_root = tmp_path / "plugin"
    mem_dir = plugin_root / "memory"
    mem_dir.mkdir(parents=True)
    # Build memory with 15 sessions for one project
    sessions = [
        {"session_start": f"2026-01-{i+1:02d}T00:00:00Z", "event_count": i,
         "categories": {}, "patterns_detected": []}
        for i in range(15)
    ]
    memory = {
        "version": 1,
        "last_updated": "2026-03-07T00:00:00Z",
        "projects": {"proj123": {"project_path": "/tmp/proj", "sessions": sessions}},
        "cross_project_patterns": [],
    }
    (mem_dir / "cross-project-errors.json").write_text(json.dumps(memory))
    result = cleanup_memory(str(plugin_root), max_sessions_per_project=10)
    assert result["cleaned"] == 5
    # Verify the most recent sessions are kept
    updated = load_cross_project(str(plugin_root))
    assert len(updated["projects"]["proj123"]["sessions"]) == 10
    assert updated["projects"]["proj123"]["sessions"][0]["session_start"] == "2026-01-06T00:00:00Z"


def test_cleanup_removes_empty_projects(tmp_path):
    """Projects with 0 sessions after cleanup are removed."""
    plugin_root = tmp_path / "plugin"
    mem_dir = plugin_root / "memory"
    mem_dir.mkdir(parents=True)
    memory = {
        "version": 1,
        "last_updated": "2026-03-07T00:00:00Z",
        "projects": {
            "proj_a": {"project_path": "/a", "sessions": []},
            "proj_b": {"project_path": "/b", "sessions": [
                {"session_start": "2026-01-01T00:00:00Z", "event_count": 3,
                 "categories": {}, "patterns_detected": []}
            ]},
        },
        "cross_project_patterns": [],
    }
    (mem_dir / "cross-project-errors.json").write_text(json.dumps(memory))
    result = cleanup_memory(str(plugin_root))
    assert result["projects_remaining"] == 1
    updated = load_cross_project(str(plugin_root))
    assert "proj_a" not in updated["projects"]
    assert "proj_b" in updated["projects"]


def test_cleanup_no_op_under_limit(tmp_path):
    """cleanup_memory does nothing when sessions count is under the limit."""
    plugin_root = tmp_path / "plugin"
    mem_dir = plugin_root / "memory"
    mem_dir.mkdir(parents=True)
    sessions = [
        {"session_start": f"2026-01-{i+1:02d}T00:00:00Z", "event_count": i,
         "categories": {}, "patterns_detected": []}
        for i in range(5)
    ]
    memory = {
        "version": 1,
        "last_updated": "2026-03-07T00:00:00Z",
        "projects": {"proj1": {"project_path": "/tmp/p", "sessions": sessions}},
        "cross_project_patterns": [],
    }
    (mem_dir / "cross-project-errors.json").write_text(json.dumps(memory))
    result = cleanup_memory(str(plugin_root), max_sessions_per_project=10)
    assert result["cleaned"] == 0


def test_cli_cleanup(run_main, tmp_path):
    """CLI cleanup action works."""
    plugin_root = tmp_path / "plugin"
    mem_dir = plugin_root / "memory"
    mem_dir.mkdir(parents=True)
    sessions = [
        {"session_start": f"2026-01-{i+1:02d}T00:00:00Z", "event_count": i,
         "categories": {}, "patterns_detected": []}
        for i in range(5)
    ]
    memory = {
        "version": 1,
        "last_updated": "2026-03-07T00:00:00Z",
        "projects": {"proj1": {"project_path": "/tmp/p", "sessions": sessions}},
        "cross_project_patterns": [],
    }
    (mem_dir / "cross-project-errors.json").write_text(json.dumps(memory))
    # Use a dummy exp_root (cleanup uses plugin_root from arg)
    r = run_main("error_tracker.py", str(tmp_path), "cleanup", str(plugin_root), "3")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["cleaned"] == 2


# ---------------------------------------------------------------------------
# Suggestion history / feedback loop (Phase E)
# ---------------------------------------------------------------------------


def test_log_suggestion_creates_history(tmp_path):
    """log_suggestion adds a suggestion entry to the error log."""
    log_suggestion(str(tmp_path), "high_lr_divergence", scope="session")
    history = get_suggestion_history(str(tmp_path))
    assert len(history) == 1
    assert history[0]["pattern_id"] == "high_lr_divergence"
    assert history[0]["scope"] == "session"
    assert history[0]["iteration"] == 1


def test_get_suggestion_history_empty(tmp_path):
    """get_suggestion_history returns empty list when no suggestions logged."""
    history = get_suggestion_history(str(tmp_path))
    assert history == []


def test_get_suggestion_history_returns_logged(tmp_path):
    """get_suggestion_history returns all previously logged suggestions."""
    log_suggestion(str(tmp_path), "oom_batch_size")
    log_suggestion(str(tmp_path), "wasted_budget")
    history = get_suggestion_history(str(tmp_path))
    assert len(history) == 2
    pattern_ids = [s["pattern_id"] for s in history]
    assert "oom_batch_size" in pattern_ids
    assert "wasted_budget" in pattern_ids


def test_log_suggestion_increments_iteration(tmp_path):
    """Logging the same pattern_id twice increments the iteration counter."""
    log_suggestion(str(tmp_path), "high_lr_divergence")
    log_suggestion(str(tmp_path), "high_lr_divergence")
    history = get_suggestion_history(str(tmp_path))
    lr_suggestions = [s for s in history if s["pattern_id"] == "high_lr_divergence"]
    assert len(lr_suggestions) == 2
    assert lr_suggestions[0]["iteration"] == 1
    assert lr_suggestions[1]["iteration"] == 2


def test_cli_log_suggestion(run_main, tmp_path):
    """CLI log-suggestion action stores a suggestion."""
    # First create the error log so exp_root is valid
    log_event(str(tmp_path), create_event(
        "training_failure", "warning", "experiment", "fail",
    ))
    r = run_main("error_tracker.py", str(tmp_path), "log-suggestion", "oom_batch_size", "session")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["logged"] is True
    # Verify it was stored
    history = get_suggestion_history(str(tmp_path))
    assert len(history) == 1


def test_cli_suggestion_history(run_main, tmp_path):
    """CLI suggestion-history action returns logged suggestions."""
    log_suggestion(str(tmp_path), "high_lr_divergence", scope="session")
    log_suggestion(str(tmp_path), "oom_batch_size", scope="cross-project")
    r = run_main("error_tracker.py", str(tmp_path), "suggestion-history")
    assert r.returncode == 0
    history = json.loads(r.stdout)
    assert len(history) == 2
    pattern_ids = [s["pattern_id"] for s in history]
    assert "high_lr_divergence" in pattern_ids
    assert "oom_batch_size" in pattern_ids


class TestConcurrentLogEvent:
    """Test concurrent log_event calls don't lose events (Task 3.1)."""

    def test_concurrent_log_events_no_loss(self, tmp_path):
        """4 threads x 5 events each = 20 events, verify none lost."""
        import threading

        errors = []

        def log_events(thread_id):
            try:
                for i in range(5):
                    ev = create_event(
                        category="training_failure",
                        severity="warning",
                        source="experiment",
                        message=f"Thread {thread_id} event {i}",
                        exp_id=f"exp-t{thread_id}-{i}",
                        phase=6,
                    )
                    log_event(str(tmp_path), ev)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=log_events, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Threads raised errors: {errors}"

        # Read all events and verify count
        events = get_events(str(tmp_path))
        assert len(events) == 20, f"Expected 20 events, got {len(events)}"


def test_create_event_invalid_category_raises():
    """create_event with invalid category raises ValueError."""
    with pytest.raises(ValueError, match="Invalid.*category"):
        create_event("bad_category", "critical", "experiment", "msg")


def test_create_event_invalid_severity_raises():
    """create_event with invalid severity raises ValueError."""
    with pytest.raises(ValueError, match="Invalid.*severity"):
        create_event("training_failure", "bad_severity", "experiment", "msg")


def test_create_event_invalid_source_raises():
    """create_event with invalid source raises ValueError."""
    with pytest.raises(ValueError, match="Invalid.*source"):
        create_event("training_failure", "critical", "bad_source", "msg")


class TestEmptyInputEdgeCases:
    """Edge case tests for empty inputs (Task 3.5)."""

    def test_validate_event_empty(self):
        # Should handle empty dict gracefully (return errors or False, not crash)
        result = validate_event({})
        assert result is not None
        assert result["valid"] is False
        assert len(result["errors"]) > 0
