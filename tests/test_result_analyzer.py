"""Tests for result_analyzer.py."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from result_analyzer import load_results, rank_by_metric, compute_deltas, identify_correlations, analyze


def _write_results(tmp_path, experiments: dict):
    """Helper to write experiment result files."""
    for name, data in experiments.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(data))


def test_load_results(tmp_path):
    _write_results(tmp_path, {
        "baseline": {"metrics": {"loss": 1.0}},
        "exp-001": {"metrics": {"loss": 0.8}},
    })
    results = load_results(str(tmp_path))
    assert "baseline" in results
    assert "exp-001" in results


def test_load_results_empty(tmp_path):
    assert load_results(str(tmp_path)) == {}


def test_load_results_nonexistent():
    assert load_results("/nonexistent/dir") == {}


def test_rank_by_metric():
    results = {
        "baseline": {"metrics": {"loss": 1.0}},
        "exp-001": {"metrics": {"loss": 0.5}},
        "exp-002": {"metrics": {"loss": 0.8}},
    }
    ranked = rank_by_metric(results, "loss", lower_is_better=True)
    assert len(ranked) == 3
    assert ranked[0]["exp_id"] == "exp-001"
    assert ranked[1]["exp_id"] == "exp-002"


def test_rank_by_metric_higher_better():
    results = {
        "baseline": {"metrics": {"accuracy": 70.0}},
        "exp-001": {"metrics": {"accuracy": 85.0}},
        "exp-002": {"metrics": {"accuracy": 78.0}},
    }
    ranked = rank_by_metric(results, "accuracy", lower_is_better=False)
    assert ranked[0]["exp_id"] == "exp-001"


def test_compute_deltas():
    results = {
        "baseline": {"metrics": {"loss": 1.0}},
        "exp-001": {"metrics": {"loss": 0.8}, "config": {"lr": 0.001}},
        "exp-002": {"metrics": {"loss": 0.6}, "config": {"lr": 0.0001}},
    }
    deltas = compute_deltas(results, "baseline", "loss")
    assert len(deltas) == 2
    # Find exp-002
    exp2 = next(d for d in deltas if d["exp_id"] == "exp-002")
    assert exp2["delta"] == -0.4
    assert exp2["delta_pct"] == -40.0


def test_compute_deltas_missing_baseline():
    results = {"exp-001": {"metrics": {"loss": 0.8}}}
    assert compute_deltas(results, "baseline", "loss") == []


def test_analyze_missing_baseline_warning(tmp_path):
    _write_results(tmp_path, {
        "exp-001": {"metrics": {"loss": 0.8}, "config": {"lr": 0.001}},
    })
    result = analyze(str(tmp_path), "loss", baseline_id="baseline")
    assert "warning" in result
    assert "baseline" in result["warning"]


def test_identify_correlations():
    results = {
        "exp-001": {"metrics": {"loss": 0.3}, "config": {"lr": 0.0001, "batch_size": 32}},
        "exp-002": {"metrics": {"loss": 0.5}, "config": {"lr": 0.001, "batch_size": 16}},
        "exp-003": {"metrics": {"loss": 0.4}, "config": {"lr": 0.0005, "batch_size": 32}},
        "exp-004": {"metrics": {"loss": 0.7}, "config": {"lr": 0.01, "batch_size": 8}},
    }
    corr = identify_correlations(results, "loss", lower_is_better=True)
    assert "correlations" in corr
    assert len(corr["correlations"]) > 0


def test_identify_correlations_too_few():
    results = {
        "exp-001": {"metrics": {"loss": 0.3}, "config": {"lr": 0.0001}},
        "exp-002": {"metrics": {"loss": 0.5}, "config": {"lr": 0.001}},
    }
    corr = identify_correlations(results, "loss", lower_is_better=True)
    assert corr["correlations"] == []
    assert "note" in corr


def test_analyze(tmp_path):
    _write_results(tmp_path, {
        "baseline": {"metrics": {"loss": 1.0}, "config": {"lr": 0.001}},
        "exp-001": {"metrics": {"loss": 0.5}, "config": {"lr": 0.0001}},
    })
    result = analyze(str(tmp_path), "loss")
    assert result["num_experiments"] == 2
    assert len(result["ranking"]) == 2
    assert len(result["deltas"]) == 1


def test_identify_correlations_exactly_three():
    results = {
        "exp-001": {"metrics": {"loss": 0.3}, "config": {"lr": 0.0001}},
        "exp-002": {"metrics": {"loss": 0.5}, "config": {"lr": 0.001}},
        "exp-003": {"metrics": {"loss": 0.4}, "config": {"lr": 0.0005}},
    }
    corr = identify_correlations(results, "loss", lower_is_better=True)
    assert corr["correlations"] == []
    assert "note" in corr


def test_analyze_empty(tmp_path):
    result = analyze(str(tmp_path / "nonexistent"), "loss")
    assert "error" in result
