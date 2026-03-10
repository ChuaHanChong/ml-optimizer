"""End-to-end tests for the review skill's data flow.

Simulates full sessions and validates that error_tracker functions
produce correct outputs for the review skill to consume.
"""

import json

from conftest import _write_result

from error_tracker import (
    create_event,
    load_cross_project,
    log_event,
    load_error_log,
    detect_patterns,
    summarize_session,
    compute_success_metrics,
    compute_proposal_outcomes,
    rank_suggestions,
    update_cross_project,
    detect_cross_project_patterns,
)


def _create_full_session(tmp_path):
    """Create a realistic session with mixed results and errors.

    Returns exp_root path.
    """
    exp_root = tmp_path / "experiments"
    exp_root.mkdir()

    # Results
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

    # Implementation manifest
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

    # Proposed configs
    configs_dir = results / "proposed-configs"
    configs_dir.mkdir()
    for i in range(4):
        (configs_dir / f"exp-{i+1:03d}.json").write_text(
            json.dumps({"lr": 0.001 * (i + 1)}))

    # Error events — diverse mix
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


# ---------------------------------------------------------------------------
# Full session tests
# ---------------------------------------------------------------------------


def test_full_session_summary(tmp_path):
    """Session summary counts all categories correctly."""
    exp_root = _create_full_session(tmp_path)
    summary = summarize_session(str(exp_root))
    assert summary["total_events"] == 8
    assert summary["by_category"]["divergence"] == 3
    assert summary["by_category"]["pipeline_inefficiency"] == 2
    assert summary["by_category"]["research_failure"] == 1
    assert summary["by_category"]["resource_error"] == 1


def test_full_session_patterns(tmp_path):
    """Expected patterns are detected from full session."""
    exp_root = _create_full_session(tmp_path)
    log_data = load_error_log(str(exp_root))
    patterns = detect_patterns(log_data["events"])
    ids = [p["pattern_id"] for p in patterns]
    assert "high_lr_divergence" in ids
    assert "wasted_budget" in ids


def test_full_session_success_metrics(tmp_path):
    """Success metrics are computed correctly from full session."""
    exp_root = _create_full_session(tmp_path)
    m = compute_success_metrics(str(exp_root), "acc", lower_is_better=False)
    assert m["total_experiments"] == 4
    assert m["completed"] == 2
    assert m["failed"] == 1
    assert m["diverged"] == 1
    assert m["improvement_rate"] is not None
    assert m["improvement_rate"] > 0  # exp-001 (78.0) beats baseline (70.0)
    assert len(m["top_configs"]) >= 1


def test_full_session_proposal_outcomes(tmp_path):
    """Proposal outcomes cross-reference correctly."""
    exp_root = _create_full_session(tmp_path)
    p = compute_proposal_outcomes(str(exp_root), "acc", lower_is_better=False)
    assert p["implementation_stats"]["validated"] == 1
    assert p["implementation_stats"]["validation_failed"] == 1
    assert len(p["research_proposals"]) == 1
    assert p["research_proposals"][0]["name"] == "perceptual-loss"
    assert p["research_proposals"][0]["beat_baseline"] >= 1


def test_full_session_ranked_suggestions(tmp_path):
    """Ranked suggestions are sorted by score descending."""
    exp_root = _create_full_session(tmp_path)
    log_data = load_error_log(str(exp_root))
    patterns = detect_patterns(log_data["events"])
    ranked = rank_suggestions(patterns)
    assert len(ranked) >= 2
    # Verify sorted by score
    for i in range(len(ranked) - 1):
        assert ranked[i]["score"] >= ranked[i + 1]["score"]


def test_cross_project_integration(tmp_path):
    """Two projects synced to cross-project memory detect shared patterns."""
    plugin_root = tmp_path / "plugin"
    (plugin_root / "memory").mkdir(parents=True)

    # Project A: has divergence pattern
    exp_a = tmp_path / "project_a" / "experiments"
    exp_a.mkdir(parents=True)
    for lr in [0.1, 0.2, 0.05]:
        log_event(str(exp_a), create_event(
            "divergence", "warning", "monitor", "NaN",
            config={"lr": lr, "batch_size": 32}))
    update_cross_project(str(plugin_root), str(tmp_path / "project_a"), str(exp_a))

    # Project B: also has divergence pattern
    exp_b = tmp_path / "project_b" / "experiments"
    exp_b.mkdir(parents=True)
    for lr in [0.3, 0.15, 0.08]:
        log_event(str(exp_b), create_event(
            "divergence", "warning", "monitor", "NaN",
            config={"lr": lr, "batch_size": 64}))
    update_cross_project(str(plugin_root), str(tmp_path / "project_b"), str(exp_b))

    # Cross-project should detect shared divergence pattern
    memory = load_cross_project(str(plugin_root))
    cross_patterns = detect_cross_project_patterns(memory)
    ids = [p["pattern_id"] for p in cross_patterns]
    assert "high_lr_divergence" in ids


def test_review_with_new_categories(tmp_path):
    """New categories (research_failure, timeout, resource_error) tracked correctly."""
    exp_root = tmp_path / "experiments"
    exp_root.mkdir()
    log_event(str(exp_root), create_event(
        "research_failure", "warning", "research", "No results", phase=4))
    log_event(str(exp_root), create_event(
        "timeout", "critical", "orchestrate", "Agent timed out",
        duration_seconds=600))
    log_event(str(exp_root), create_event(
        "resource_error", "info", "baseline", "No GPU", phase=2))

    summary = summarize_session(str(exp_root))
    assert summary["by_category"]["research_failure"] == 1
    assert summary["by_category"]["timeout"] == 1
    assert summary["by_category"]["resource_error"] == 1


def test_review_with_duration_field(tmp_path):
    """Events with duration_seconds are stored and retrievable."""
    exp_root = tmp_path / "experiments"
    exp_root.mkdir()
    log_event(str(exp_root), create_event(
        "timeout", "critical", "orchestrate", "Timed out",
        duration_seconds=3600.5))
    log_data = load_error_log(str(exp_root))
    assert log_data["events"][0]["duration_seconds"] == 3600.5


def test_review_empty_session(tmp_path):
    """Review functions handle empty session gracefully."""
    exp_root = tmp_path / "experiments"
    exp_root.mkdir()
    (exp_root / "results").mkdir()

    summary = summarize_session(str(exp_root))
    assert summary["total_events"] == 0

    m = compute_success_metrics(str(exp_root), "acc", lower_is_better=False)
    assert m["total_experiments"] == 0

    p = compute_proposal_outcomes(str(exp_root), "acc", lower_is_better=False)
    assert p["hp_proposals"]["total_run"] == 0


def test_review_session_no_baseline(tmp_path):
    """Review metrics with no baseline returns improvement_rate=None."""
    exp_root = tmp_path / "experiments"
    exp_root.mkdir()
    results = exp_root / "results"
    results.mkdir()
    _write_result(results, "exp-001", "completed", {"lr": 0.001}, {"acc": 75.0})

    m = compute_success_metrics(str(exp_root), "acc", lower_is_better=False)
    assert m["total_experiments"] == 1
    assert m["improvement_rate"] is None
