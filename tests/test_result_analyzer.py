"""Tests for result_analyzer.py."""

import json

import pytest

from conftest import _write_results

from result_analyzer import load_results, rank_by_metric, compute_deltas, identify_correlations, analyze, spearman_correlation, group_by_method_tier, build_experiment_description, rank_methods_for_stacking


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


def test_load_results_mixed_valid_and_corrupt(tmp_path):
    """load_results silently skips corrupt JSON, loads valid files."""
    (tmp_path / "exp-001.json").write_text('{"metrics": {"loss": 0.5}}')
    (tmp_path / "exp-002.json").write_text("{bad json")
    (tmp_path / "exp-003.json").write_text('{"metrics": {"loss": 0.3}}')
    results = load_results(str(tmp_path))
    assert len(results) == 2
    assert "exp-001" in results
    assert "exp-003" in results
    assert "exp-002" not in results


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


def test_rank_by_metric_includes_status():
    """Ranked entries include the experiment status field."""
    results = {
        "baseline": {"metrics": {"loss": 1.0}, "status": "completed"},
        "exp-001": {"metrics": {"loss": 0.5}, "status": "completed"},
        "exp-002": {"metrics": {"loss": 0.7}, "status": "failed"},
        "exp-003": {"metrics": {"loss": 0.3}, "status": "diverged"},
    }
    ranked = rank_by_metric(results, "loss", lower_is_better=True)
    assert len(ranked) == 4
    # Each entry has a status field
    for entry in ranked:
        assert "status" in entry
    # Find the failed one
    failed = next(r for r in ranked if r["exp_id"] == "exp-002")
    assert failed["status"] == "failed"
    diverged = next(r for r in ranked if r["exp_id"] == "exp-003")
    assert diverged["status"] == "diverged"


def test_rank_by_metric_status_none_when_missing():
    """Status is None when experiment data has no status field."""
    results = {
        "exp-001": {"metrics": {"loss": 0.5}},
    }
    ranked = rank_by_metric(results, "loss")
    assert ranked[0]["status"] is None


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


def test_compute_deltas_zero_baseline():
    """Verify no division error when baseline metric is 0."""
    results = {
        "baseline": {"metrics": {"loss": 0}},
        "exp-001": {"metrics": {"loss": 0.5}, "config": {"lr": 0.001}},
        "exp-002": {"metrics": {"loss": -0.3}, "config": {"lr": 0.0001}},
    }
    deltas = compute_deltas(results, "baseline", "loss")
    assert len(deltas) == 2
    # When baseline is zero, delta_pct should be None (undefined percentage)
    for d in deltas:
        assert d["delta_pct"] is None
    exp1 = next(d for d in deltas if d["exp_id"] == "exp-001")
    assert exp1["delta"] == 0.5


def test_compute_deltas_near_zero_baseline():
    """Baseline within 1e-8 threshold returns delta_pct=None."""
    results = {
        "baseline": {"metrics": {"loss": 5e-10}},
        "exp-001": {"metrics": {"loss": 0.5}, "config": {"lr": 0.001}},
    }
    deltas = compute_deltas(results, "baseline", "loss")
    assert len(deltas) == 1
    assert deltas[0]["delta_pct"] is None
    assert abs(deltas[0]["delta"] - 0.5) < 1e-6


def test_identify_correlations_with_status():
    """Experiments with status 'diverged' should be excluded from correlation analysis."""
    results = {
        "exp-001": {"metrics": {"loss": 0.3}, "config": {"lr": 0.0001}, "status": "completed"},
        "exp-002": {"metrics": {"loss": 0.5}, "config": {"lr": 0.001}, "status": "completed"},
        "exp-003": {"metrics": {"loss": 0.4}, "config": {"lr": 0.0005}, "status": "completed"},
        "exp-004": {"metrics": {"loss": 0.7}, "config": {"lr": 0.01}, "status": "completed"},
        "exp-005": {"metrics": {"loss": 99.0}, "config": {"lr": 0.0001}, "status": "diverged"},
        "exp-006": {"metrics": {"loss": 50.0}, "config": {"lr": 0.0001}, "status": "failed"},
    }
    corr = identify_correlations(results, "loss", lower_is_better=True)
    assert "correlations" in corr
    assert len(corr["correlations"]) > 0
    # The diverged/failed experiments should not affect the correlations
    # With only 4 completed experiments, we should still get results
    # Check that spearman_rho is present for numeric params
    lr_corr = next(c for c in corr["correlations"] if c["param"] == "lr")
    assert "spearman_rho" in lr_corr

    # Now verify that if we remove completed experiments so fewer than 4 remain,
    # we get the "too few" response
    results_few = {
        "exp-001": {"metrics": {"loss": 0.3}, "config": {"lr": 0.0001}, "status": "completed"},
        "exp-002": {"metrics": {"loss": 0.5}, "config": {"lr": 0.001}, "status": "completed"},
        "exp-003": {"metrics": {"loss": 0.4}, "config": {"lr": 0.0005}, "status": "diverged"},
        "exp-004": {"metrics": {"loss": 0.7}, "config": {"lr": 0.01}, "status": "diverged"},
    }
    corr_few = identify_correlations(results_few, "loss", lower_is_better=True)
    assert corr_few["correlations"] == []
    assert "note" in corr_few


def test_spearman_correlation():
    """Test spearman_correlation with known values."""
    # Perfect positive correlation
    rho = spearman_correlation([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    assert abs(rho - 1.0) < 1e-9

    # Perfect negative correlation
    rho = spearman_correlation([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
    assert abs(rho - (-1.0)) < 1e-9

    # No correlation (orthogonal ranks)
    # [1,2,3,4,5] vs [3,5,1,4,2] — ranks are shuffled
    rho = spearman_correlation([1, 2, 3, 4, 5], [3, 5, 1, 4, 2])
    assert abs(rho) < 0.5  # Should be weak/no correlation

    # Edge case: fewer than 2 elements
    rho = spearman_correlation([1], [2])
    assert rho == 0.0

    # Edge case: mismatched lengths
    rho = spearman_correlation([1, 2, 3], [1, 2])
    assert rho == 0.0

    # Tied values
    rho = spearman_correlation([1, 1, 2, 3], [10, 10, 20, 30])
    assert rho > 0.9  # Should still be strongly positive


def test_compute_deltas_baseline_missing_metric():
    """When baseline exists but lacks the requested metric, return empty list."""
    results = {
        "baseline": {"metrics": {"accuracy": 85.0}},
        "exp-001": {"metrics": {"loss": 0.5}, "config": {"lr": 0.001}},
    }
    deltas = compute_deltas(results, "baseline", "loss")
    assert deltas == []


def test_identify_correlations_categorical_values():
    """Categorical HP values trigger the except branch with top_common/bottom_common."""
    results = {
        "exp-001": {"metrics": {"loss": 0.3}, "config": {"optimizer": "adam"}, "status": "completed"},
        "exp-002": {"metrics": {"loss": 0.5}, "config": {"optimizer": "sgd"}, "status": "completed"},
        "exp-003": {"metrics": {"loss": 0.4}, "config": {"optimizer": "adam"}, "status": "completed"},
        "exp-004": {"metrics": {"loss": 0.7}, "config": {"optimizer": "sgd"}, "status": "completed"},
    }
    corr = identify_correlations(results, "loss", lower_is_better=True)
    assert len(corr["correlations"]) > 0
    opt_corr = next(c for c in corr["correlations"] if c["param"] == "optimizer")
    assert "top_common" in opt_corr
    assert "bottom_common" in opt_corr


# --- CLI tests ---


def test_cli_analyze(run_main, tmp_path):
    """CLI analyzes results directory."""
    _write_results(tmp_path, {
        "baseline": {"metrics": {"loss": 1.0}, "config": {"lr": 0.001}},
        "exp-001": {"metrics": {"loss": 0.5}, "config": {"lr": 0.0001}},
    })
    r = run_main("result_analyzer.py", str(tmp_path), "loss")
    assert r.returncode == 0
    output = json.loads(r.stdout)
    assert output["num_experiments"] == 2


def test_spearman_constant_x():
    """All-identical x values should return 0.0 (no variance)."""
    rho = spearman_correlation([5, 5, 5, 5, 5], [1, 2, 3, 4, 5])
    assert rho == 0.0
    # Also test constant y
    rho2 = spearman_correlation([1, 2, 3, 4, 5], [7, 7, 7, 7, 7])
    assert rho2 == 0.0


@pytest.mark.parametrize("label,results,expected_last_id,all_have_note", [
    ("nan_last", {"e1": {"metrics": {"loss": 0.5}}, "e2": {"metrics": {"loss": float("nan")}}, "e3": {"metrics": {"loss": 0.3}}}, "e2", False),
    ("inf_last", {"e1": {"metrics": {"loss": 0.5}}, "e2": {"metrics": {"loss": float("inf")}}, "e3": {"metrics": {"loss": 0.3}}}, "e2", False),
    ("all_nan", {"e1": {"metrics": {"loss": float("nan")}}, "e2": {"metrics": {"loss": float("nan")}}}, None, True),
])
def test_rank_by_metric_non_finite(label, results, expected_last_id, all_have_note):
    """Non-finite metric values (NaN, Inf) are sorted to end with note."""
    ranked = rank_by_metric(results, "loss", lower_is_better=True)
    assert len(ranked) == len(results)
    if expected_last_id:
        assert ranked[-1]["exp_id"] == expected_last_id
        assert "note" in ranked[-1]
    if all_have_note:
        assert all("note" in r for r in ranked)


def test_rank_by_metric_partial_metric_coverage():
    """rank_by_metric returns only experiments that have the requested metric."""
    results = {
        "exp-001": {"metrics": {"loss": 0.5, "accuracy": 90.0}},
        "exp-002": {"metrics": {"loss": 0.3}},
        "exp-003": {"metrics": {"loss": 0.4}},
        "exp-004": {"metrics": {"loss": 0.7, "accuracy": 80.0}},
    }
    ranked = rank_by_metric(results, "accuracy", lower_is_better=False)
    assert len(ranked) == 2
    assert ranked[0]["exp_id"] == "exp-001"
    assert ranked[1]["exp_id"] == "exp-004"


def test_cli_lower_is_better_parsing(run_main, tmp_path):
    """CLI correctly parses 'false' as lower_is_better=False."""
    _write_results(tmp_path, {
        "baseline": {"metrics": {"accuracy": 70.0}, "config": {"lr": 0.001}},
        "exp-001": {"metrics": {"accuracy": 85.0}, "config": {"lr": 0.0001}},
    })
    r = run_main("result_analyzer.py", str(tmp_path), "accuracy", "baseline", "false")
    assert r.returncode == 0
    output = json.loads(r.stdout)
    # With lower_is_better=False, higher accuracy should rank first
    assert output["ranking"][0]["exp_id"] == "exp-001"


def test_identify_correlations_mixed_numeric_string():
    """Mixed numeric/string HP values: numeric majority gets numeric correlation."""
    results = {
        "exp-001": {"metrics": {"loss": 0.3}, "config": {"lr": 0.0001}, "status": "completed"},
        "exp-002": {"metrics": {"loss": 0.5}, "config": {"lr": 0.001}, "status": "completed"},
        "exp-003": {"metrics": {"loss": 0.4}, "config": {"lr": 0.0005}, "status": "completed"},
        "exp-004": {"metrics": {"loss": 0.7}, "config": {"lr": 0.01}, "status": "completed"},
        "exp-005": {"metrics": {"loss": 0.6}, "config": {"lr": "adaptive"}, "status": "completed"},
    }
    corr = identify_correlations(results, "loss", lower_is_better=True)
    assert len(corr["correlations"]) > 0
    lr_corr = next(c for c in corr["correlations"] if c["param"] == "lr")
    # Should compute numeric correlation (not categorical)
    assert "spearman_rho" in lr_corr
    assert "top_common" not in lr_corr
    # Should note the excluded non-numeric value
    assert "note" in lr_corr
    assert "1 non-numeric" in lr_corr["note"]


def test_identify_correlations_mostly_string():
    """Mostly-string HP values fall to categorical treatment."""
    results = {
        "exp-001": {"metrics": {"loss": 0.3}, "config": {"optim": "adam"}, "status": "completed"},
        "exp-002": {"metrics": {"loss": 0.5}, "config": {"optim": "sgd"}, "status": "completed"},
        "exp-003": {"metrics": {"loss": 0.4}, "config": {"optim": "adam"}, "status": "completed"},
        "exp-004": {"metrics": {"loss": 0.7}, "config": {"optim": "rmsprop"}, "status": "completed"},
        "exp-005": {"metrics": {"loss": 0.6}, "config": {"optim": "1"}, "status": "completed"},
    }
    corr = identify_correlations(results, "loss", lower_is_better=True)
    opt_corr = next(c for c in corr["correlations"] if c["param"] == "optim")
    # Only 1 out of 5 is numeric — should be categorical
    assert "top_common" in opt_corr
    assert "spearman_rho" not in opt_corr


def test_identify_correlations_constant_metric():
    """When all experiments have the same metric value, Spearman rho should be 0."""
    results = {
        "exp-001": {"metrics": {"loss": 0.5}, "config": {"lr": 0.0001}, "status": "completed"},
        "exp-002": {"metrics": {"loss": 0.5}, "config": {"lr": 0.001}, "status": "completed"},
        "exp-003": {"metrics": {"loss": 0.5}, "config": {"lr": 0.0005}, "status": "completed"},
        "exp-004": {"metrics": {"loss": 0.5}, "config": {"lr": 0.01}, "status": "completed"},
    }
    corr = identify_correlations(results, "loss", lower_is_better=True)
    assert "correlations" in corr
    if corr["correlations"]:
        lr_corr = next(c for c in corr["correlations"] if c["param"] == "lr")
        assert lr_corr["spearman_rho"] == 0.0


def test_cli_no_args(run_main):
    """CLI with no args prints usage and exits 1."""
    r = run_main("result_analyzer.py")
    assert r.returncode == 1
    assert "Usage" in r.stdout


# --- group_by_method_tier tests ---


def test_group_by_method_tier_mixed():
    """Groups experiments by method_tier field."""
    results = {
        "baseline": {"method_tier": "baseline", "metrics": {"loss": 1.0}},
        "exp-001": {"method_tier": "method_default_hp", "metrics": {"loss": 0.8}},
        "exp-002": {"method_tier": "method_default_hp", "metrics": {"loss": 0.7}},
        "exp-003": {"method_tier": "method_tuned_hp", "metrics": {"loss": 0.5}},
    }
    groups = group_by_method_tier(results)
    assert len(groups["baseline"]) == 1
    assert len(groups["method_default_hp"]) == 2
    assert len(groups["method_tuned_hp"]) == 1
    assert "unknown" not in groups


def test_group_by_method_tier_missing_field():
    """Experiments without method_tier go to 'unknown'."""
    results = {
        "baseline": {"metrics": {"loss": 1.0}},
        "exp-001": {"method_tier": "method_default_hp", "metrics": {"loss": 0.8}},
        "exp-002": {"metrics": {"loss": 0.6}},
    }
    groups = group_by_method_tier(results)
    assert len(groups["unknown"]) == 2
    assert len(groups["method_default_hp"]) == 1


def test_group_by_method_tier_empty():
    """Empty results returns empty dict."""
    groups = group_by_method_tier({})
    assert groups == {}


def test_group_by_method_tier_preserves_exp_id():
    """Each grouped entry includes exp_id."""
    results = {
        "exp-001": {"method_tier": "baseline", "metrics": {"loss": 0.5}},
    }
    groups = group_by_method_tier(results)
    assert groups["baseline"][0]["exp_id"] == "exp-001"


# --- load_results file filter tests ---


class TestLoadResultsFileFilter:
    """Verify load_results only loads baseline.json and exp-*.json files."""

    def test_filters_non_result_files(self, tmp_path):
        """Only baseline.json and exp-*.json are loaded; other JSON files are skipped."""
        # Valid result files
        (tmp_path / "baseline.json").write_text(
            json.dumps({"experiment_id": "baseline", "metrics": {"accuracy": 0.85}})
        )
        (tmp_path / "exp-001.json").write_text(
            json.dumps({"experiment_id": "exp-001", "metrics": {"accuracy": 0.90}})
        )
        # Files that should be filtered out
        (tmp_path / "prerequisites.json").write_text(
            json.dumps({"experiment_id": "prerequisites", "metrics": {"accuracy": 0.0}})
        )
        (tmp_path / "implementation-manifest.json").write_text(
            json.dumps({"experiment_id": "manifest", "metrics": {"accuracy": 0.0}})
        )

        results = load_results(str(tmp_path))
        assert len(results) == 2
        assert "baseline" in results
        assert "exp-001" in results
        assert "prerequisites" not in results
        assert "implementation-manifest" not in results

    def test_multiple_exp_files(self, tmp_path):
        """Multiple exp-*.json files are all loaded."""
        for i in range(5):
            (tmp_path / f"exp-{i:03d}.json").write_text(
                json.dumps({"experiment_id": f"exp-{i:03d}", "metrics": {"loss": 0.5 - i * 0.05}})
            )
        (tmp_path / "some-other-file.json").write_text(
            json.dumps({"experiment_id": "other", "metrics": {"loss": 1.0}})
        )
        results = load_results(str(tmp_path))
        assert len(results) == 5
        assert "some-other-file" not in results

    def test_no_matching_files(self, tmp_path):
        """Directory with only non-matching JSON files returns empty dict."""
        (tmp_path / "prerequisites.json").write_text('{"status": "ok"}')
        (tmp_path / "implementation-manifest.json").write_text('{"proposals": []}')
        results = load_results(str(tmp_path))
        assert results == {}


class TestEmptyInputEdgeCases:
    """Edge case tests for empty inputs (Task 3.5)."""

    def test_rank_by_metric_empty(self):
        result = rank_by_metric({}, "accuracy")
        assert result == []


# ---------- build_experiment_description ----------


def test_build_description_with_code_proposal():
    """Description includes code_proposal name."""
    data = {"code_proposal": "perceptual-loss", "config": {"lr": 0.001}}
    desc = build_experiment_description("exp-001", data)
    assert "perceptual-loss" in desc


def test_build_description_strips_branch_prefix():
    """Branch prefix ml-opt/ is stripped from description."""
    data = {"code_branch": "ml-opt/cosine-scheduler", "config": {}}
    desc = build_experiment_description("exp-001", data)
    assert "cosine-scheduler" in desc
    assert "ml-opt/" not in desc


def test_build_description_hp_diff_vs_baseline():
    """Description shows HP changes relative to baseline."""
    baseline_config = {"lr": 0.01, "batch_size": 32}
    data = {"config": {"lr": 0.003, "batch_size": 32}}
    desc = build_experiment_description("exp-002", data, baseline_config)
    assert "lr=0.003" in desc
    # batch_size unchanged, should not appear
    assert "batch_size" not in desc


def test_build_description_method_and_hp_diff():
    """Description combines method name and HP diff."""
    baseline_config = {"lr": 0.01}
    data = {"code_proposal": "mixup", "config": {"lr": 0.005}}
    desc = build_experiment_description("exp-003", data, baseline_config)
    assert "mixup" in desc
    assert "lr=0.005" in desc


def test_build_description_fallback_to_exp_id():
    """Falls back to exp_id when no rich fields present."""
    data = {"config": {}}
    desc = build_experiment_description("exp-007", data)
    assert desc == "exp-007"


def test_build_description_truncation():
    """Long descriptions are truncated to max_len."""
    data = {"code_proposal": "very-long-proposal-name-that-exceeds-limit",
            "config": {"learning_rate": 0.0001, "weight_decay": 0.01}}
    desc = build_experiment_description("exp-001", data, max_len=30)
    assert len(desc) <= 30
    assert desc.endswith("...")


def test_build_description_no_baseline_shows_first_hp():
    """Without baseline config, shows first interesting HP."""
    data = {"config": {"lr": 0.01, "batch_size": 64}}
    desc = build_experiment_description("exp-001", data, baseline_config=None)
    # Should show at least one HP value
    assert "=" in desc


def test_build_description_code_proposal_preferred_over_branch():
    """code_proposal is preferred over code_branch."""
    data = {"code_proposal": "label-smoothing",
            "code_branch": "ml-opt/label-smoothing",
            "config": {}}
    desc = build_experiment_description("exp-001", data)
    assert "label-smoothing" in desc


# ---------- build_experiment_description — stacked experiments ----------


def test_build_description_stacked_experiment():
    """Stacked experiments show combined method names."""
    data = {
        "code_branches": ["ml-opt/perceptual-loss", "ml-opt/cosine-scheduler"],
        "stacking_order": 2,
        "config": {"lr": 0.003},
    }
    desc = build_experiment_description("exp-stack-001", data)
    assert "perceptual-loss" in desc
    assert "cosine-scheduler" in desc


def test_build_description_stacked_truncation():
    """Long stacked descriptions are truncated."""
    data = {
        "code_branches": [f"ml-opt/method-{c}" for c in "abcdefghij"],
        "config": {},
    }
    desc = build_experiment_description("exp-stack-010", data, max_len=30)
    assert len(desc) <= 30
    assert desc.endswith("...")


# ---------- group_by_method_tier — stacked tiers ----------


def test_group_by_method_tier_stacked(tmp_path):
    """group_by_method_tier handles stacked tier values."""
    _write_results(tmp_path, {
        "baseline": {"metrics": {"loss": 1.0}, "method_tier": "baseline"},
        "exp-001": {"metrics": {"loss": 0.8}, "method_tier": "method_tuned_hp"},
        "exp-stack-001": {"metrics": {"loss": 0.6}, "method_tier": "stacked_default_hp",
                          "code_branches": ["ml-opt/a", "ml-opt/b"], "stacking_order": 2},
        "exp-stack-002": {"metrics": {"loss": 0.5}, "method_tier": "stacked_tuned_hp",
                          "code_branches": ["ml-opt/a", "ml-opt/b"], "stacking_order": 2},
    })
    results = load_results(str(tmp_path))
    groups = group_by_method_tier(results)
    assert "stacked_default_hp" in groups
    assert "stacked_tuned_hp" in groups
    assert len(groups["stacked_default_hp"]) == 1
    assert len(groups["stacked_tuned_hp"]) == 1


# ---------- rank_methods_for_stacking ----------


def test_rank_methods_for_stacking_basic(tmp_path):
    """Methods are ranked by improvement magnitude over baseline."""
    _write_results(tmp_path, {
        "baseline": {"metrics": {"loss": 1.0}, "config": {"lr": 0.01}},
        "exp-001": {"metrics": {"loss": 0.8}, "config": {"lr": 0.01},
                    "code_branch": "ml-opt/method-a", "code_proposal": "method-a",
                    "method_tier": "method_tuned_hp", "status": "completed"},
        "exp-002": {"metrics": {"loss": 0.6}, "config": {"lr": 0.001},
                    "code_branch": "ml-opt/method-b", "code_proposal": "method-b",
                    "method_tier": "method_tuned_hp", "status": "completed"},
        "exp-003": {"metrics": {"loss": 0.9}, "config": {"lr": 0.01},
                    "code_branch": "ml-opt/method-c", "code_proposal": "method-c",
                    "method_tier": "method_tuned_hp", "status": "completed"},
    })
    results = load_results(str(tmp_path))
    ranked = rank_methods_for_stacking(results, "loss", lower_is_better=True)
    # method-b improved most (0.6 vs 1.0), then method-a (0.8), then method-c (0.9)
    assert len(ranked) == 3
    assert ranked[0]["code_proposal"] == "method-b"
    assert ranked[1]["code_proposal"] == "method-a"
    assert ranked[2]["code_proposal"] == "method-c"


def test_rank_methods_excludes_non_improvements(tmp_path):
    """Methods that didn't improve over baseline are excluded."""
    _write_results(tmp_path, {
        "baseline": {"metrics": {"loss": 1.0}, "config": {"lr": 0.01}},
        "exp-001": {"metrics": {"loss": 0.8}, "config": {"lr": 0.01},
                    "code_branch": "ml-opt/method-a", "code_proposal": "method-a",
                    "method_tier": "method_tuned_hp", "status": "completed"},
        "exp-002": {"metrics": {"loss": 1.2}, "config": {"lr": 0.001},
                    "code_branch": "ml-opt/method-b", "code_proposal": "method-b",
                    "method_tier": "method_tuned_hp", "status": "completed"},
    })
    results = load_results(str(tmp_path))
    ranked = rank_methods_for_stacking(results, "loss", lower_is_better=True)
    assert len(ranked) == 1
    assert ranked[0]["code_proposal"] == "method-a"


def test_rank_methods_picks_best_per_branch(tmp_path):
    """When a branch has multiple experiments, uses the best result."""
    _write_results(tmp_path, {
        "baseline": {"metrics": {"loss": 1.0}, "config": {"lr": 0.01}},
        "exp-001": {"metrics": {"loss": 0.9}, "config": {"lr": 0.01},
                    "code_branch": "ml-opt/method-a", "code_proposal": "method-a",
                    "method_tier": "method_default_hp", "status": "completed"},
        "exp-002": {"metrics": {"loss": 0.7}, "config": {"lr": 0.001},
                    "code_branch": "ml-opt/method-a", "code_proposal": "method-a",
                    "method_tier": "method_tuned_hp", "status": "completed"},
    })
    results = load_results(str(tmp_path))
    ranked = rank_methods_for_stacking(results, "loss", lower_is_better=True)
    assert len(ranked) == 1
    assert ranked[0]["best_metric"] == 0.7
    assert ranked[0]["best_config"] == {"lr": 0.001}


def test_rank_methods_higher_is_better(tmp_path):
    """Ranking works with higher-is-better metrics."""
    _write_results(tmp_path, {
        "baseline": {"metrics": {"accuracy": 0.7}, "config": {}},
        "exp-001": {"metrics": {"accuracy": 0.9}, "config": {},
                    "code_branch": "ml-opt/method-a", "code_proposal": "method-a",
                    "method_tier": "method_tuned_hp", "status": "completed"},
        "exp-002": {"metrics": {"accuracy": 0.8}, "config": {},
                    "code_branch": "ml-opt/method-b", "code_proposal": "method-b",
                    "method_tier": "method_tuned_hp", "status": "completed"},
    })
    results = load_results(str(tmp_path))
    ranked = rank_methods_for_stacking(results, "accuracy", lower_is_better=False)
    assert ranked[0]["code_proposal"] == "method-a"  # 0.9 > 0.8


def test_rank_methods_empty_results(tmp_path):
    """Returns empty list when no methods improved."""
    _write_results(tmp_path, {
        "baseline": {"metrics": {"loss": 1.0}, "config": {}},
    })
    results = load_results(str(tmp_path))
    ranked = rank_methods_for_stacking(results, "loss", lower_is_better=True)
    assert ranked == []


def test_rank_methods_non_finite_excluded(tmp_path):
    """Experiments with NaN/Inf metrics are excluded from stacking ranking."""
    _write_results(tmp_path, {
        "baseline": {"metrics": {"loss": 1.0}, "config": {}},
        "exp-001": {"metrics": {"loss": 0.8}, "config": {},
                    "code_branch": "ml-opt/method-a", "code_proposal": "method-a",
                    "status": "completed"},
        "exp-002": {"metrics": {"loss": float("nan")}, "config": {},
                    "code_branch": "ml-opt/method-b", "code_proposal": "method-b",
                    "status": "completed"},
        "exp-003": {"metrics": {"loss": float("inf")}, "config": {},
                    "code_branch": "ml-opt/method-c", "code_proposal": "method-c",
                    "status": "completed"},
    })
    results = load_results(str(tmp_path))
    ranked = rank_methods_for_stacking(results, "loss", lower_is_better=True)
    assert len(ranked) == 1
    assert ranked[0]["code_proposal"] == "method-a"


def test_rank_methods_zero_baseline(tmp_path):
    """Ranking works when baseline metric is near zero (improvement_pct=None)."""
    _write_results(tmp_path, {
        "baseline": {"metrics": {"loss": 0.0}, "config": {}},
        "exp-001": {"metrics": {"loss": -0.5}, "config": {},
                    "code_branch": "ml-opt/method-a", "code_proposal": "method-a",
                    "status": "completed"},
    })
    results = load_results(str(tmp_path))
    ranked = rank_methods_for_stacking(results, "loss", lower_is_better=True)
    assert len(ranked) == 1
    assert ranked[0]["improvement_pct"] is None
