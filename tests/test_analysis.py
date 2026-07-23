"""Tests for result_analyzer.py, schema_validator.py, and plot_results.py."""

import json

import pytest

from conftest import _write_result, _write_results
from result_analyzer import (
    analyze,
    build_experiment_description,
    compute_branch_scores,
    compute_deltas,
    detect_hp_interactions,
    group_by_method_tier,
    identify_correlations,
    load_results,
    rank_by_metric,
    rank_methods_for_stacking,
    spearman_correlation,
    compare_experiments,
    format_comparison_table,
)
from schema_validator import (
    validate_baseline,
    validate_file,
    validate_manifest,
    validate_prerequisites,
    validate_result,
    validate_result_strict,
)
from plot_results import (
    ascii_bar_chart,
    ascii_line_chart,
    plot_hp_sensitivity,
    plot_improvement_timeline,
    plot_metric_comparison,
    plot_progress_chart,
)


# ======================================================================
# TestResultAnalyzer
# ======================================================================


class TestResultAnalyzer:
    """Tests for result_analyzer.py loading, ranking, correlation, grouping, stacking, and CLI."""

    # ---------- load_results ----------

    def test_load_results_normal_and_filtering(self, tmp_path):
        """Normal load and filtering of non-result files."""
        _write_results(tmp_path, {
            "baseline": {"metrics": {"loss": 1.0}},
            "exp-001": {"metrics": {"loss": 0.8}},
            "prerequisites": {"metrics": {"loss": 0.0}},
            "implementation-manifest": {"metrics": {"loss": 0.0}},
        })
        results = load_results(str(tmp_path))
        assert set(results.keys()) == {"baseline", "exp-001"}

    def test_load_results_empty_and_nonexistent(self, tmp_path):
        """Empty and nonexistent directories both yield an empty dict."""
        assert load_results(str(tmp_path)) == {}
        assert load_results("/nonexistent/dir") == {}

    def test_load_results_skips_corrupt_json(self, tmp_path):
        """Corrupt result JSON is skipped while valid files load."""
        (tmp_path / "exp-001.json").write_text('{"metrics": {"loss": 0.5}}')
        (tmp_path / "exp-002.json").write_text("{bad json")
        results = load_results(str(tmp_path))
        assert len(results) == 1
        assert "exp-002" not in results

    # ---------- rank_by_metric ----------

    def test_rank_by_metric_ordering(self):
        """Lowest-loss experiment ranks first when lower is better."""
        results = {
            "baseline": {"metrics": {"loss": 1.0}},
            "exp-001": {"metrics": {"loss": 0.5}},
            "exp-002": {"metrics": {"loss": 0.8}},
        }
        ranked = rank_by_metric(results, "loss", lower_is_better=True)
        assert ranked[0]["exp_id"] == "exp-001"

    def test_rank_by_metric_status_and_nonfinite(self):
        """Status preserved; NaN/inf sorted last."""
        results = {
            "e1": {"metrics": {"loss": 0.5}, "status": "completed"},
            "e2": {"metrics": {"loss": float("nan")}, "status": "diverged"},
            "e3": {"metrics": {"loss": 0.3}, "status": "completed"},
        }
        ranked = rank_by_metric(results, "loss", lower_is_better=True)
        assert ranked[-1]["exp_id"] == "e2"
        assert "note" in ranked[-1]
        assert ranked[0]["status"] == "completed"

    def test_rank_by_metric_partial_and_empty(self):
        """Experiments lacking the metric are dropped; empty input ranks empty."""
        results = {
            "exp-001": {"metrics": {"loss": 0.5, "accuracy": 90.0}},
            "exp-002": {"metrics": {"loss": 0.3}},
        }
        ranked = rank_by_metric(results, "accuracy", lower_is_better=False)
        assert len(ranked) == 1
        assert rank_by_metric({}, "accuracy") == []

    # ---------- compute_deltas ----------

    @pytest.mark.parametrize("baseline_val,exp_vals,expected_pct_none", [
        (1.0, [("exp-001", 0.8, {"lr": 0.001}), ("exp-002", 0.6, {"lr": 0.0001})], False),
        (0, [("exp-001", 0.5, {"lr": 0.001})], True),
    ])
    def test_compute_deltas(self, baseline_val, exp_vals, expected_pct_none):
        """Deltas vs baseline computed; percentage is None when baseline is zero."""
        results = {"baseline": {"metrics": {"loss": baseline_val}}}
        for eid, val, cfg in exp_vals:
            results[eid] = {"metrics": {"loss": val}, "config": cfg}
        deltas = compute_deltas(results, "baseline", "loss")
        assert len(deltas) == len(exp_vals)
        if expected_pct_none:
            assert all(d["delta_pct"] is None for d in deltas)
        else:
            exp2 = next(d for d in deltas if d["exp_id"] == "exp-002")
            assert exp2["delta"] == pytest.approx(-0.4)
            assert exp2["delta_pct"] == pytest.approx(-40.0)

    def test_compute_deltas_missing_baseline(self):
        """Absent baseline yields no deltas."""
        assert compute_deltas({"exp-001": {"metrics": {"loss": 0.8}}}, "baseline", "loss") == []

    # ---------- spearman_correlation (3 cases) ----------

    @pytest.mark.parametrize("x,y,check", [
        ([1, 2, 3, 4, 5], [10, 20, 30, 40, 50], lambda r: abs(r - 1.0) < 1e-9),
        ([1, 2, 3, 4, 5], [3, 5, 1, 4, 2], lambda r: abs(r) < 0.5),
        ([1, 1, 2, 3], [10, 10, 20, 30], lambda r: r > 0.9),
    ])
    def test_spearman_correlation(self, x, y, check):
        """Rank correlation matches expectations for monotonic, random, and tied data."""
        assert check(spearman_correlation(x, y))

    # ---------- identify_correlations ----------

    def test_identify_correlations_basic(self):
        """Numeric HP-metric correlations are surfaced when enough data exists."""
        results = {
            "exp-001": {"metrics": {"loss": 0.3}, "config": {"lr": 0.0001, "batch_size": 32}},
            "exp-002": {"metrics": {"loss": 0.5}, "config": {"lr": 0.001, "batch_size": 16}},
            "exp-003": {"metrics": {"loss": 0.4}, "config": {"lr": 0.0005, "batch_size": 32}},
            "exp-004": {"metrics": {"loss": 0.7}, "config": {"lr": 0.01, "batch_size": 8}},
        }
        corr = identify_correlations(results, "loss", lower_is_better=True)
        assert len(corr["correlations"]) > 0

    def test_identify_correlations_too_few(self):
        """Too few experiments returns no correlations plus an explanatory note."""
        results = {f"exp-{i:03d}": {"metrics": {"loss": 0.3 + i * 0.1},
                                     "config": {"lr": 0.0001 * (i + 1)}}
                   for i in range(2)}
        corr = identify_correlations(results, "loss", lower_is_better=True)
        assert corr["correlations"] == []
        assert "note" in corr

    def test_identify_correlations_categorical(self):
        """Categorical HPs report top and bottom common values instead of correlation."""
        results = {
            "exp-001": {"metrics": {"loss": 0.3}, "config": {"optimizer": "adam"}, "status": "completed"},
            "exp-002": {"metrics": {"loss": 0.5}, "config": {"optimizer": "sgd"}, "status": "completed"},
            "exp-003": {"metrics": {"loss": 0.4}, "config": {"optimizer": "adam"}, "status": "completed"},
            "exp-004": {"metrics": {"loss": 0.7}, "config": {"optimizer": "sgd"}, "status": "completed"},
        }
        corr = identify_correlations(results, "loss", lower_is_better=True)
        opt_corr = next(c for c in corr["correlations"] if c["param"] == "optimizer")
        assert "top_common" in opt_corr and "bottom_common" in opt_corr

    # ---------- analyze ----------

    def test_analyze_basic(self, tmp_path):
        """Full analysis reports experiment count, ranking, and deltas."""
        _write_results(tmp_path, {
            "baseline": {"metrics": {"loss": 1.0}, "config": {"lr": 0.001}},
            "exp-001": {"metrics": {"loss": 0.5}, "config": {"lr": 0.0001}},
        })
        result = analyze(str(tmp_path), "loss")
        assert result["num_experiments"] == 2
        assert len(result["ranking"]) == 2
        assert len(result["deltas"]) == 1

    def test_analyze_empty(self, tmp_path):
        """Analyzing a directory with no results returns an error."""
        assert "error" in analyze(str(tmp_path / "nonexistent"), "loss")

    # ---------- group_by_method_tier ----------

    def test_group_by_method_tier(self):
        """Experiments are bucketed by their method_tier field."""
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

    def test_group_by_method_tier_empty_and_unknown(self):
        """Empty input yields no groups; tier-less experiments fall under 'unknown'."""
        assert group_by_method_tier({}) == {}
        groups = group_by_method_tier({"e1": {"metrics": {"loss": 0.5}}})
        assert len(groups["unknown"]) == 1
        assert groups["unknown"][0]["exp_id"] == "e1"

    def test_group_by_method_tier_stacked(self, tmp_path):
        """Stacked experiments are grouped under their stacked tier."""
        _write_results(tmp_path, {
            "baseline": {"metrics": {"loss": 1.0}, "method_tier": "baseline"},
            "exp-stack-001": {"metrics": {"loss": 0.6}, "method_tier": "stacked_default_hp",
                              "code_branches": ["ml-opt/a", "ml-opt/b"], "stacking_order": 2},
        })
        groups = group_by_method_tier(load_results(str(tmp_path)))
        assert len(groups["stacked_default_hp"]) == 1

    # ---------- build_experiment_description (4 cases) ----------

    @pytest.mark.parametrize("data,baseline_cfg,max_len,check", [
        ({"code_proposal": "perceptual-loss", "config": {"lr": 0.001}},
         None, 80, lambda d: "perceptual-loss" in d),
        ({"code_branch": "ml-opt/cosine-scheduler", "config": {}},
         None, 80, lambda d: "cosine-scheduler" in d and "ml-opt/" not in d),
        ({"config": {"lr": 0.003, "batch_size": 32}},
         {"lr": 0.01, "batch_size": 32}, 80,
         lambda d: "lr=0.003" in d and "batch_size" not in d),
        ({"code_proposal": "very-long-proposal-name-that-exceeds-limit",
          "config": {"learning_rate": 0.0001, "weight_decay": 0.01}},
         None, 30, lambda d: len(d) <= 30 and d.endswith("...")),
    ])
    def test_build_experiment_description(self, data, baseline_cfg, max_len, check):
        """Description reflects proposal/branch/config diff and respects max length."""
        desc = build_experiment_description("exp-001", data, baseline_cfg, max_len=max_len)
        assert check(desc)

    # ---------- rank_methods_for_stacking ----------

    def test_rank_methods_for_stacking_basic(self, tmp_path):
        """Improving methods are ranked best-first for stacking."""
        _write_results(tmp_path, {
            "baseline": {"metrics": {"loss": 1.0}, "config": {"lr": 0.01}},
            "exp-001": {"metrics": {"loss": 0.8}, "config": {"lr": 0.01},
                        "code_branch": "ml-opt/method-a", "code_proposal": "method-a",
                        "method_tier": "method_tuned_hp", "status": "completed"},
            "exp-002": {"metrics": {"loss": 0.6}, "config": {"lr": 0.001},
                        "code_branch": "ml-opt/method-b", "code_proposal": "method-b",
                        "method_tier": "method_tuned_hp", "status": "completed"},
        })
        ranked = rank_methods_for_stacking(load_results(str(tmp_path)), "loss", lower_is_better=True)
        assert len(ranked) == 2
        assert ranked[0]["code_proposal"] == "method-b"

    def test_rank_methods_excludes_non_improvements(self, tmp_path):
        """Methods that do not beat baseline are excluded from stacking."""
        _write_results(tmp_path, {
            "baseline": {"metrics": {"loss": 1.0}, "config": {"lr": 0.01}},
            "exp-001": {"metrics": {"loss": 1.2}, "config": {"lr": 0.001},
                        "code_branch": "ml-opt/method-b", "code_proposal": "method-b",
                        "method_tier": "method_tuned_hp", "status": "completed"},
        })
        ranked = rank_methods_for_stacking(load_results(str(tmp_path)), "loss", lower_is_better=True)
        assert ranked == []

    def test_rank_methods_picks_best_per_branch(self, tmp_path):
        """Each branch contributes only its single best experiment to the ranking."""
        _write_results(tmp_path, {
            "baseline": {"metrics": {"loss": 1.0}, "config": {"lr": 0.01}},
            "exp-001": {"metrics": {"loss": 0.9}, "config": {"lr": 0.01},
                        "code_branch": "ml-opt/method-a", "code_proposal": "method-a",
                        "method_tier": "method_default_hp", "status": "completed"},
            "exp-002": {"metrics": {"loss": 0.7}, "config": {"lr": 0.001},
                        "code_branch": "ml-opt/method-a", "code_proposal": "method-a",
                        "method_tier": "method_tuned_hp", "status": "completed"},
        })
        ranked = rank_methods_for_stacking(load_results(str(tmp_path)), "loss", lower_is_better=True)
        assert len(ranked) == 1
        assert ranked[0]["best_metric"] == 0.7

    # ---------- CLI ----------

    def test_cli_analyze(self, run_main, tmp_path):
        """CLI analyze prints valid JSON with the experiment count."""
        _write_results(tmp_path, {
            "baseline": {"metrics": {"loss": 1.0}, "config": {"lr": 0.001}},
            "exp-001": {"metrics": {"loss": 0.5}, "config": {"lr": 0.0001}},
        })
        r = run_main("result_analyzer.py", str(tmp_path), "loss")
        assert r.returncode == 0
        assert json.loads(r.stdout)["num_experiments"] == 2

    def test_cli_no_args(self, run_main):
        """CLI with no arguments exits non-zero and prints usage."""
        r = run_main("result_analyzer.py")
        assert r.returncode == 1
        assert "Usage" in r.stdout


# ======================================================================
# TestSchemaValidator
# ======================================================================


class TestSchemaValidator:
    """Tests for schema_validator.py result, baseline, manifest, and prerequisites validation."""

    # ---------- validate_result (6 cases) ----------

    @pytest.mark.parametrize("data,expect_valid", [
        # valid minimal
        ({"exp_id": "exp-001", "status": "completed", "config": {"lr": 0.001},
          "metrics": {"loss": 0.5}},
         True),
        # valid with optionals
        ({"exp_id": "exp-001", "status": "completed",
          "config": {"lr": 0.001, "batch_size": 16},
          "metrics": {"loss": 0.67, "accuracy": 82.5},
          "gpu_id": 0, "duration_seconds": 3600},
         True),
        # missing required field (metrics)
        ({"exp_id": "exp-001", "status": "completed", "config": {"lr": 0.001}},
         False),
        # invalid status
        ({"exp_id": "exp-001", "status": "unknown", "config": {"lr": 0.001},
          "metrics": {"loss": 0.5}},
         False),
        # invalid method_tier
        ({"exp_id": "exp-001", "status": "completed", "config": {"lr": 0.001},
          "metrics": {"loss": 0.5}, "method_tier": "invalid_tier"},
         False),
        # NaN metric
        ({"exp_id": "exp-001", "status": "completed", "config": {"lr": 0.001},
          "metrics": {"loss": float("nan")}},
         False),
        # valid with checkpoint fields
        ({"exp_id": "exp-ckpt", "status": "completed", "config": {"lr": 0.001},
          "metrics": {"loss": 0.5},
          "checkpoint_source": {"exp_id": "exp-001", "checkpoint_path": "/tmp/ckpt.pt"},
          "warm_started": True},
         True),
        # invalid: checkpoint_source as string (not dict)
        ({"exp_id": "exp-bad", "status": "completed", "config": {"lr": 0.001},
          "metrics": {"loss": 0.5}, "checkpoint_source": "/tmp/ckpt.pt"},
         False),
        # invalid: warm_started=true without checkpoint_source
        ({"exp_id": "exp-bad2", "status": "completed", "config": {"lr": 0.001},
          "metrics": {"loss": 0.5}, "warm_started": True},
         False),
        # valid: reproducibility metadata
        ({"exp_id": "exp-repro", "status": "completed", "config": {"lr": 0.01},
          "metrics": {"loss": 0.4},
          "reproducibility": {"random_seed": 42, "environment_file": "pip_freeze.txt",
                               "git_sha": "abc123", "framework_version": "torch==2.1.0"}},
         True),
    ])
    def test_validate_result(self, data, expect_valid):
        """Result validity matches expectations across required fields and constraints."""
        assert validate_result(data)["valid"] is expect_valid

    def test_validate_result_valid_method_tiers(self):
        """All valid method tiers accepted."""
        for tier in ["baseline", "method_tuned_hp", "stacked_default_hp"]:
            data = {"exp_id": "exp-001", "status": "completed",
                    "config": {"lr": 0.001}, "metrics": {"loss": 0.5},
                    "method_tier": tier}
            if tier.startswith("stacked"):
                data["code_branches"] = ["ml-opt/a", "ml-opt/b"]
                data["stacking_order"] = 2
                data["stack_base_exp"] = "exp-012"
            assert validate_result(data)["valid"] is True

    # ---------- validate_baseline (3 cases) ----------

    @pytest.mark.parametrize("data,expect_valid", [
        ({"exp_id": "baseline", "status": "completed",
          "config": {"lr": 0.001, "batch_size": 32},
          "metrics": {"loss": 1.0, "accuracy": 70.0}},
         True),
        ({"exp_id": "baseline", "status": "bogus", "config": {}, "metrics": {}},
         False),
        ({"exp_id": "baseline", "status": "completed", "config": {"lr": 0.001},
          "metrics": {"loss": float("nan")}},
         False),
    ])
    def test_validate_baseline(self, data, expect_valid):
        """Baseline validity matches expectations for status and metric values."""
        assert validate_baseline(data)["valid"] is expect_valid

    # ---------- validate_manifest (4 cases) ----------

    @pytest.mark.parametrize("data,expect_valid", [
        ({"original_branch": "main", "strategy": "git_branch",
          "proposals": [{"name": "LR", "slug": "lr", "status": "validated"}]},
         True),
        ({"original_branch": "main", "strategy": "git_branch",
          "proposals": [{"slug": "missing-name", "status": "validated"}]},
         False),
        ({"original_branch": "main", "strategy": "invalid_strategy", "proposals": []},
         False),
        ({"original_branch": "main", "strategy": "git_branch",
          "proposals": [{"name": "T", "slug": "t", "status": "validated",
                         "implementation_strategy": "from_nowhere"}]},
         False),
    ])
    def test_validate_manifest(self, data, expect_valid):
        """Manifest validity matches expectations for proposals, strategy, and fields."""
        assert validate_manifest(data)["valid"] is expect_valid

    # ---------- validate_prerequisites (3 cases) ----------

    @pytest.mark.parametrize("data,expect_valid", [
        ({"status": "ready",
          "dataset": {"train_path": "/data/train"},
          "environment": {"manager": "conda"},
          "ready_for_baseline": True},
         True),
        ({"status": "ready", "environment": {}, "ready_for_baseline": True},
         False),
        ({"status": "unknown", "dataset": {}, "environment": {},
          "ready_for_baseline": True},
         False),
    ])
    def test_validate_prerequisites(self, data, expect_valid):
        """Prerequisites validity matches expectations for required sections."""
        assert validate_prerequisites(data)["valid"] is expect_valid

    def test_validate_prerequisites_warnings(self):
        """A missing train_path produces a warning but remains valid."""
        data = {"status": "ready", "dataset": {"format_detected": "csv"},
                "environment": {"manager": "conda"}, "ready_for_baseline": True}
        result = validate_prerequisites(data)
        assert result["valid"] is True
        assert any("train_path" in w for w in result["warnings"])

    # ---------- non-dict input ----------

    def test_validate_non_dict_input(self):
        """Every validator rejects non-dict input with a dict-related error."""
        for validator in [validate_result, validate_baseline, validate_manifest, validate_prerequisites]:
            result = validator("not a dict")
            assert result["valid"] is False
            assert any("dict" in e.lower() for e in result["errors"])

    # ---------- validate_file ----------

    @pytest.mark.parametrize("schema,content", [
        ("result", {"exp_id": "exp-001", "status": "completed", "config": {}, "metrics": {}}),
        ("baseline", {"exp_id": "baseline", "status": "completed", "config": {}, "metrics": {}}),
        ("manifest", {"original_branch": "main", "strategy": "git_branch",
                       "proposals": [{"name": "test", "slug": "test", "status": "validated"}]}),
        ("prerequisites", {"status": "ready", "dataset": {}, "environment": {},
                            "ready_for_baseline": True}),
    ])
    def test_validate_file_dispatch(self, tmp_path, schema, content):
        """validate_file dispatches to the correct schema validator for each type."""
        f = tmp_path / f"{schema}.json"
        f.write_text(json.dumps(content))
        assert validate_file(str(f), schema)["valid"] is True

    def test_validate_file_errors(self, tmp_path):
        """Invalid JSON, nonexistent file, unknown schema."""
        bad = tmp_path / "bad.json"
        bad.write_text("{bad json")
        assert validate_file(str(bad), "result")["valid"] is False
        assert validate_file("/nonexistent/file.json", "result")["valid"] is False
        good = tmp_path / "good.json"
        good.write_text('{"key": "value"}')
        assert validate_file(str(good), "unknown_schema")["valid"] is False

    # ---------- all validators return warnings key ----------

    def test_all_validators_return_warnings_key(self):
        """Every validator returns a list-valued 'warnings' key."""
        cases = [
            (validate_result, {"exp_id": "e", "status": "completed", "config": {}, "metrics": {}}),
            (validate_baseline, {"exp_id": "b", "status": "completed", "config": {}, "metrics": {}}),
            (validate_manifest, {"original_branch": "main", "strategy": "git_branch", "proposals": []}),
            (validate_prerequisites, {"status": "ready", "dataset": {}, "environment": {},
                                       "ready_for_baseline": True}),
        ]
        for validator, data in cases:
            out = validator(data)
            assert "warnings" in out and isinstance(out["warnings"], list)

    # ---------- CLI ----------

    def test_cli_validate_valid(self, run_main, tmp_path):
        """CLI validates a well-formed result file and reports valid."""
        f = tmp_path / "exp.json"
        f.write_text(json.dumps({"exp_id": "exp-001", "status": "completed",
                                 "config": {}, "metrics": {}}))
        r = run_main("schema_validator.py", str(f), "result")
        assert r.returncode == 0
        assert json.loads(r.stdout)["valid"] is True

    def test_cli_no_args(self, run_main):
        """CLI with no arguments exits non-zero."""
        r = run_main("schema_validator.py")
        assert r.returncode == 1


# ======================================================================
# TestPlotResults
# ======================================================================


class TestPlotResults:
    """Tests for plot_results.py ASCII charts, comparison, timeline, sensitivity, and progress chart."""

    # ---------- ascii_bar_chart (3 cases) ----------

    @pytest.mark.parametrize("labels,values,title,check", [
        (["alpha", "beta", "gamma"], [10.0, 20.0, 15.0], "Test Chart",
         lambda c: all(l in c for l in ["alpha", "beta", "gamma"]) and "\u2588" in c),
        ([], [], None, lambda c: c == ""),
        (["a", "b", "c"], [-5.0, -10.0, 3.0], None,
         lambda c: "a" in c and "-10" in c),
    ])
    def test_ascii_bar_chart(self, labels, values, title, check):
        """Bar chart renders labels and bars; empty input yields an empty string."""
        chart = ascii_bar_chart(labels, values, title=title) if title else ascii_bar_chart(labels, values)
        assert check(chart)

    # ---------- ascii_line_chart (3 cases) ----------

    @pytest.mark.parametrize("values,height,check", [
        ([1, 2, 4, 3, 5, 4, 6, 7, 8, 10], 10,
         lambda c: "*" in c and len(c.strip().split("\n")) >= 12),
        ([], None, lambda c: c == ""),
        ([5.0, 5.0, 5.0, 5.0], 5, lambda c: "*" in c),
    ])
    def test_ascii_line_chart(self, values, height, check):
        """Line chart plots points; empty input yields an empty string."""
        kwargs = {}
        if height is not None:
            kwargs["height"] = height
        chart = ascii_line_chart(values, **kwargs)
        assert check(chart)

    def test_ascii_line_chart_resampling(self):
        """Long series are resampled to fit the requested width."""
        values = [i * 0.1 for i in range(100)]
        chart = ascii_line_chart(values, width=60)
        assert chart and "*" in chart

    # ---------- combined chart types ----------

    def test_plot_all_chart_types(self, tmp_path):
        """Test comparison, timeline, and sensitivity in one go."""
        _write_results(tmp_path, {
            "baseline": {"metrics": {"loss": 1.0}, "config": {"lr": 0.01}},
            "exp-001": {"metrics": {"loss": 0.7}, "config": {"lr": 0.001}},
            "exp-002": {"metrics": {"loss": 0.5}, "config": {"lr": 0.0001}},
        })
        chart = plot_metric_comparison(str(tmp_path), "loss")
        assert chart and "[B]" in chart
        chart = plot_improvement_timeline(str(tmp_path), "loss")
        assert chart and "*" in chart
        chart = plot_hp_sensitivity(str(tmp_path), "loss", "lr")
        assert chart and "*" in chart

    def test_plot_no_results(self, tmp_path):
        """Plots over an empty directory report 'No results'."""
        assert "No results" in plot_metric_comparison(str(tmp_path / "nonexistent"), "loss")
        assert "No results" in plot_improvement_timeline(str(tmp_path / "nonexistent"), "loss")

    # ---------- plot_progress_chart ----------

    def test_progress_chart_basic(self, tmp_path):
        """Progress chart writes a non-empty PNG to the requested path."""
        _write_results(tmp_path, {
            "baseline": {"metrics": {"loss": 1.0}, "config": {"lr": 0.01}},
            "exp-001": {"metrics": {"loss": 0.7}, "config": {"lr": 0.001}},
            "exp-002": {"metrics": {"loss": 0.5}, "config": {"lr": 0.0001}},
        })
        out = tmp_path / "chart.png"
        path = plot_progress_chart(str(tmp_path), "loss", output_path=str(out))
        assert path == str(out) and out.exists() and out.stat().st_size > 0

    def test_progress_chart_no_results(self, tmp_path):
        """Progress chart returns None when there are no results."""
        assert plot_progress_chart(str(tmp_path / "nonexistent"), "loss") is None

    def test_progress_chart_annotations(self, tmp_path):
        """Progress chart annotates proposal and stacking experiments."""
        _write_results(tmp_path, {
            "baseline": {"metrics": {"loss": 1.0}, "config": {"lr": 0.01}},
            "exp-001": {"metrics": {"loss": 0.7}, "config": {"lr": 0.01},
                        "code_proposal": "perceptual-loss",
                        "code_branch": "ml-opt/perceptual-loss"},
            "exp-stack-001": {
                "metrics": {"loss": 0.5}, "config": {"lr": 0.001},
                "code_branches": ["ml-opt/method-a", "ml-opt/method-b"],
                "stacking_order": 2, "status": "completed"},
        })
        out = tmp_path / "annotated.png"
        path = plot_progress_chart(str(tmp_path), "loss", output_path=str(out))
        assert path == str(out) and out.exists()

    # ---------- CLI ----------

    def test_cli_comparison(self, run_main, tmp_path):
        """CLI comparison chart runs successfully on a results directory."""
        _write_results(tmp_path, {"exp-001": {"metrics": {"loss": 0.5}, "config": {}}})
        r = run_main("plot_results.py", str(tmp_path), "loss", "comparison")
        assert r.returncode == 0

    def test_cli_no_args(self, run_main):
        """CLI with no arguments exits non-zero and prints usage."""
        r = run_main("plot_results.py")
        assert r.returncode == 1 and "Usage" in r.stdout


# ======================================================================
# TestHPInteractions
# ======================================================================


class TestHPInteractions:
    """Tests for detect_hp_interactions()."""

    def test_clear_interaction(self, tmp_path):
        """An lr x batch_size interaction scenario returns an interactions key."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        # Create experiments where lr*batch_size interaction matters
        configs = [
            (0.01, 16, 0.3),   # high lr + small batch = good
            (0.01, 64, 1.5),   # high lr + large batch = bad
            (0.001, 16, 0.5),  # low lr + small batch = ok
            (0.001, 64, 0.4),  # low lr + large batch = ok
            (0.01, 16, 0.35),  # repeat high lr + small batch = good
            (0.001, 64, 0.45), # repeat low lr + large batch = ok
        ]
        _write_result(results_dir, "baseline", "completed", {"lr": 0.01}, {"loss": 1.0})
        for i, (lr, bs, loss) in enumerate(configs):
            _write_result(results_dir, f"exp-{i+1}", "completed",
                         {"lr": lr, "batch_size": bs}, {"loss": loss})
        results = load_results(str(results_dir))
        out = detect_hp_interactions(results, "loss", lower_is_better=True, min_experiments=5)
        # May or may not detect interaction depending on correlation strength
        assert "interactions" in out

    def test_insufficient_data(self, tmp_path):
        """Too few experiments returns no interactions with an explanatory note."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_result(results_dir, "baseline", "completed", {"lr": 0.01}, {"loss": 1.0})
        _write_result(results_dir, "exp-1", "completed", {"lr": 0.001}, {"loss": 0.8})
        results = load_results(str(results_dir))
        out = detect_hp_interactions(results, "loss", min_experiments=5)
        assert out["interactions"] == []
        assert "Need at least" in out["note"]

    def test_categorical_excluded(self, tmp_path):
        """Interactions need two numeric HPs; a lone numeric HP yields none."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_result(results_dir, "baseline", "completed", {"lr": 0.01}, {"loss": 1.0})
        for i in range(6):
            _write_result(results_dir, f"exp-{i}", "completed",
                         {"lr": 0.001 * (i+1), "optimizer": "adam"}, {"loss": 0.5 + i*0.1})
        results = load_results(str(results_dir))
        out = detect_hp_interactions(results, "loss")
        # Only 1 numeric HP (lr) — need 2+ for interactions
        assert out["interactions"] == []

    def test_analyze_includes_interactions(self, tmp_path):
        """analyze() output includes interactions and branch_scores keys."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_result(results_dir, "baseline", "completed", {"lr": 0.01}, {"loss": 1.0})
        for i in range(6):
            _write_result(results_dir, f"exp-{i}", "completed",
                         {"lr": 0.001*(i+1), "batch_size": 16*(i+1)}, {"loss": 0.5+i*0.05})
        result = analyze(str(results_dir), "loss")
        assert "interactions" in result
        assert "branch_scores" in result
        assert isinstance(result["branch_scores"], dict)


# ======================================================================
# TestBranchScores
# ======================================================================


class TestBranchScores:
    """Tests for compute_branch_scores()."""

    def test_basic_scoring(self, tmp_path):
        """Better-performing branches score higher; sample counts are tracked."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_result(results_dir, "baseline", "completed", {"lr": 0.01}, {"loss": 1.0})
        _write_result(results_dir, "exp-1", "completed", {"lr": 0.001}, {"loss": 0.7},
                     code_branch="ml-opt/method-a")
        _write_result(results_dir, "exp-2", "completed", {"lr": 0.001}, {"loss": 0.6},
                     code_branch="ml-opt/method-a")
        _write_result(results_dir, "exp-3", "completed", {"lr": 0.001}, {"loss": 0.9},
                     code_branch="ml-opt/method-b")
        scores = compute_branch_scores(load_results(str(results_dir)), "loss")
        assert scores["ml-opt/method-a"]["score"] > scores["ml-opt/method-b"]["score"]
        assert scores["ml-opt/method-a"]["sample_count"] == 2

    def test_null_branch(self, tmp_path):
        """Branchless experiments are grouped under the synthetic baseline branch."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_result(results_dir, "baseline", "completed", {"lr": 0.01}, {"loss": 1.0})
        _write_result(results_dir, "exp-1", "completed", {"lr": 0.001}, {"loss": 0.85})
        scores = compute_branch_scores(load_results(str(results_dir)), "loss")
        assert "__baseline__" in scores

    def test_all_worse(self, tmp_path):
        """A branch worse than baseline scores zero."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_result(results_dir, "baseline", "completed", {"lr": 0.01}, {"loss": 1.0})
        _write_result(results_dir, "exp-1", "completed", {"lr": 0.001}, {"loss": 1.2},
                     code_branch="ml-opt/a")
        scores = compute_branch_scores(load_results(str(results_dir)), "loss")
        assert scores["ml-opt/a"]["score"] == 0.0  # worse than baseline gets zero

    def test_excludes_failed(self, tmp_path):
        """Diverged experiments are excluded from a branch's sample count."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_result(results_dir, "baseline", "completed", {"lr": 0.01}, {"loss": 1.0})
        _write_result(results_dir, "exp-1", "diverged", {"lr": 0.001}, {"loss": 0.5},
                     code_branch="ml-opt/a")
        _write_result(results_dir, "exp-2", "completed", {"lr": 0.001}, {"loss": 0.8},
                     code_branch="ml-opt/a")
        scores = compute_branch_scores(load_results(str(results_dir)), "loss")
        assert scores["ml-opt/a"]["sample_count"] == 1

    def test_higher_is_better(self, tmp_path):
        """A higher-is-better metric yields positive improvement for a better branch."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_result(results_dir, "baseline", "completed", {"lr": 0.01}, {"accuracy": 70.0})
        _write_result(results_dir, "exp-1", "completed", {"lr": 0.001}, {"accuracy": 80.0},
                     code_branch="ml-opt/a")
        scores = compute_branch_scores(load_results(str(results_dir)), "accuracy", lower_is_better=False)
        assert scores["ml-opt/a"]["improvement_pct"] > 0

    def test_stacking_excluded(self, tmp_path):
        """Stacking experiments (method_tier=stacked_*) don't pollute branch scores."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_result(results_dir, "baseline", "completed", {"lr": 0.01}, {"loss": 1.0})
        _write_result(results_dir, "exp-1", "completed", {"lr": 0.001}, {"loss": 0.7},
                     code_branch="ml-opt/a")
        # Stacking experiment — should NOT appear in branch scores
        stack = {"exp_id": "exp-stack", "status": "completed", "config": {"lr": 0.001},
                 "metrics": {"loss": 0.5}, "method_tier": "stacked_default_hp",
                 "code_branches": ["ml-opt/a", "ml-opt/b"]}
        (results_dir / "exp-stack.json").write_text(json.dumps(stack))
        scores = compute_branch_scores(load_results(str(results_dir)), "loss")
        assert "exp-stack" not in str(scores)
        assert scores.get("__baseline__") is None  # no phantom baseline group

    def test_empty_results(self, tmp_path):
        """Only a baseline (no branch experiments) yields no branch scores."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_result(results_dir, "baseline", "completed", {"lr": 0.01}, {"loss": 1.0})
        scores = compute_branch_scores(load_results(str(results_dir)), "loss")
        assert scores == {}


# ======================================================================
# TestExperimentComparison
# ======================================================================


class TestExperimentComparison:
    """Tests for compare_experiments() and format_comparison_table()."""

    def test_basic_comparison(self, tmp_path):
        """Pairwise comparison reports config diffs, metric deltas, and the winner."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_result(results_dir, "baseline", "completed", {"lr": 0.01}, {"loss": 1.0})
        _write_result(results_dir, "exp-1", "completed",
                     {"lr": 0.001, "batch_size": 32}, {"loss": 0.5, "accuracy": 80.0})
        _write_result(results_dir, "exp-2", "completed",
                     {"lr": 0.005, "batch_size": 64}, {"loss": 0.4, "accuracy": 85.0})
        result = compare_experiments(str(results_dir), ["exp-1", "exp-2"], "loss")
        assert result["experiments"] == ["exp-1", "exp-2"]
        assert result["config_diff"]["lr"]["differs"] is True
        assert result["config_diff"]["batch_size"]["differs"] is True
        assert result["metrics_comparison"]["loss"]["delta"] < 0  # exp-2 lower
        assert result["winner"]["exp_id"] == "exp-2"  # lower loss wins

    def test_higher_is_better(self, tmp_path):
        """For a higher-is-better metric, the larger value wins the comparison."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_result(results_dir, "exp-1", "completed", {"lr": 0.001}, {"accuracy": 80.0})
        _write_result(results_dir, "exp-2", "completed", {"lr": 0.005}, {"accuracy": 85.0})
        result = compare_experiments(str(results_dir), ["exp-1", "exp-2"], "accuracy",
                                    lower_is_better=False)
        assert result["winner"]["exp_id"] == "exp-2"

    def test_missing_experiment(self, tmp_path):
        """Comparing against a nonexistent experiment returns an error."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_result(results_dir, "exp-1", "completed", {"lr": 0.001}, {"loss": 0.5})
        result = compare_experiments(str(results_dir), ["exp-1", "exp-999"], "loss")
        assert "error" in result

    def test_same_config_no_diff(self, tmp_path):
        """Identical config values are reported as not differing."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_result(results_dir, "exp-1", "completed", {"lr": 0.001}, {"loss": 0.5})
        _write_result(results_dir, "exp-2", "completed", {"lr": 0.001}, {"loss": 0.4})
        result = compare_experiments(str(results_dir), ["exp-1", "exp-2"], "loss")
        assert result["config_diff"]["lr"]["differs"] is False

    def test_format_table(self, tmp_path):
        """The comparison table renders experiment IDs, header, and winner."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_result(results_dir, "exp-1", "completed",
                     {"lr": 0.001}, {"loss": 0.5})
        _write_result(results_dir, "exp-2", "completed",
                     {"lr": 0.005}, {"loss": 0.4})
        result = compare_experiments(str(results_dir), ["exp-1", "exp-2"], "loss")
        table = format_comparison_table(result)
        assert "EXPERIMENT COMPARISON" in table
        assert "exp-1" in table
        assert "exp-2" in table
        assert "Winner" in table

    def test_format_table_error(self):
        """Formatting an error result returns the error message verbatim."""
        table = format_comparison_table({"error": "Not found"})
        assert table == "Not found"


# ======================================================================
# TestCompletenessWarnings
# ======================================================================


class TestCompletenessWarnings:
    """Tests for status-aware completeness checks in validate_result()."""

    def test_completed_with_all_fields(self):
        """A completed result with all completeness fields produces no warnings."""
        data = {"exp_id": "exp-1", "status": "completed",
                "config": {"lr": 0.001}, "metrics": {"loss": 0.5},
                "iteration": 1, "method_tier": "baseline", "duration_seconds": 60}
        r = validate_result(data)
        assert r["valid"] is True
        assert r["warnings"] == []

    def test_completed_empty_metrics_warns(self):
        """A completed result with empty metrics is valid but warns."""
        data = {"exp_id": "exp-1", "status": "completed",
                "config": {"lr": 0.001}, "metrics": {}}
        r = validate_result(data)
        assert r["valid"] is True  # structurally valid
        assert any("empty metrics" in w for w in r["warnings"])

    def test_completed_missing_iteration_warns(self):
        """A completed result missing iteration/method_tier warns on both."""
        data = {"exp_id": "exp-1", "status": "completed",
                "config": {"lr": 0.001}, "metrics": {"loss": 0.5}}
        r = validate_result(data)
        assert r["valid"] is True
        assert any("iteration" in w for w in r["warnings"])
        assert any("method_tier" in w for w in r["warnings"])

    def test_failed_missing_notes_warns(self):
        """A failed result without notes warns."""
        data = {"exp_id": "exp-1", "status": "failed",
                "config": {"lr": 0.001}, "metrics": {}}
        r = validate_result(data)
        assert r["valid"] is True
        assert any("notes" in w for w in r["warnings"])

    def test_diverged_missing_notes_warns(self):
        """A diverged result without notes warns."""
        data = {"exp_id": "exp-1", "status": "diverged",
                "config": {"lr": 0.001}, "metrics": {}}
        r = validate_result(data)
        assert any("notes" in w for w in r["warnings"])

    def test_stacked_missing_fields_warns(self):
        """A stacked result missing code_branches/stacking_order warns on both."""
        data = {"exp_id": "exp-1", "status": "completed",
                "config": {"lr": 0.001}, "metrics": {"loss": 0.5},
                "method_tier": "stacked_default_hp", "iteration": 1,
                "duration_seconds": 60}
        r = validate_result(data)
        assert any("code_branches" in w for w in r["warnings"])
        assert any("stacking_order" in w for w in r["warnings"])

    def test_strict_mode_fails(self):
        """Strict validation upgrades empty-metrics warnings to errors."""
        data = {"exp_id": "exp-1", "status": "completed",
                "config": {}, "metrics": {}}
        r = validate_result_strict(data)
        assert r["valid"] is False
        assert any("empty metrics" in e for e in r["errors"])

    def test_running_no_warnings(self):
        """A running placeholder result is exempt from completeness warnings."""
        data = {"exp_id": "exp-1", "status": "running",
                "config": {"lr": 0.001}, "metrics": {}}
        r = validate_result(data)
        assert r["valid"] is True
        assert r["warnings"] == []


# ---------------------------------------------------------------------------
# TestBaselineBatchBFields — model_category + profiling timeout/duration
# ---------------------------------------------------------------------------


class TestBaselineBatchBFields:
    """baseline.json Batch B fields validate correctly."""

    def _base(self, **extra):
        data = {"exp_id": "baseline", "status": "completed",
                "config": {"lr": 0.001}, "metrics": {"loss": 0.5}}
        data.update(extra)
        return data

    def test_accepts_new_fields(self):
        data = self._base(
            model_category="rl",
            profiling={"estimated_timeout_seconds": 1200,
                       "training_duration_seconds": 400.5,
                       "steps_per_second": 800.0},
        )
        assert validate_baseline(data)["valid"] is True

    def test_null_model_category_ok(self):
        assert validate_baseline(self._base(model_category=None))["valid"] is True

    def test_rejects_bad_model_category(self):
        result = validate_baseline(self._base(model_category="robotics"))
        assert result["valid"] is False
        assert any("model_category" in e for e in result["errors"])

    def test_rejects_non_numeric_profiling_timeout(self):
        result = validate_baseline(
            self._base(profiling={"estimated_timeout_seconds": "soon"}))
        assert result["valid"] is False
        assert any("estimated_timeout_seconds" in e for e in result["errors"])

    def test_rejects_non_dict_profiling(self):
        result = validate_baseline(self._base(profiling="fast"))
        assert result["valid"] is False


# ======================================================================
# TestSeedReplicates (Batch C, Task 15)
# ======================================================================


class TestSeedReplicates:
    """random_seed schema field + replicate aggregation in result_analyzer."""

    def test_validate_result_accepts_integer_random_seed(self):
        data = {"exp_id": "exp-001", "status": "completed",
                "config": {"lr": 0.001}, "metrics": {"loss": 0.5},
                "iteration": 1, "method_tier": "baseline", "duration_seconds": 10,
                "random_seed": 42}
        result = validate_result(data)
        assert result["valid"] is True
        assert result["warnings"] == []

    def test_validate_result_rejects_non_integer_random_seed(self):
        data = {"exp_id": "exp-001", "status": "completed",
                "config": {}, "metrics": {"loss": 0.5}, "random_seed": "42"}
        result = validate_result(data)
        assert result["valid"] is False
        assert any("random_seed" in e for e in result["errors"])

    def test_validate_hp_proposal_rejects_bool_random_seed(self):
        from schema_validator import validate_hp_proposal
        result = validate_hp_proposal({"exp_id": "exp-001", "config": {}, "random_seed": True})
        assert result["valid"] is False
        assert any("random_seed" in e for e in result["errors"])

    def test_aggregate_replicates_pools_seed_replicates(self, tmp_path):
        from result_analyzer import aggregate_replicates
        for i, (seed, loss) in enumerate([(1, 0.8), (2, 0.6), (3, 0.7)]):
            _write_result(tmp_path, f"exp-{i+1}", "completed",
                          {"lr": 0.001}, {"loss": loss},
                          code_branch="ml-opt/a", random_seed=seed)
        groups = aggregate_replicates(load_results(str(tmp_path)), "loss")
        assert len(groups) == 1
        g = groups[0]
        assert g["n"] == 3
        assert abs(g["mean"] - 0.7) < 1e-9
        assert g["std"] is not None and g["std"] > 0
        assert g["best_exp_id"] == "exp-2"

    def test_aggregate_replicates_never_pools_without_seed(self, tmp_path):
        from result_analyzer import aggregate_replicates
        _write_result(tmp_path, "exp-1", "completed", {"lr": 0.001}, {"loss": 0.8})
        _write_result(tmp_path, "exp-2", "completed", {"lr": 0.001}, {"loss": 0.6})
        groups = aggregate_replicates(load_results(str(tmp_path)), "loss")
        assert len(groups) == 2

    def test_rank_methods_uses_replicate_mean(self, tmp_path):
        _write_results(tmp_path, {
            "baseline": {"metrics": {"loss": 1.0}, "config": {"lr": 0.01}},
            # branch a: lucky seed 0.5 but replicate mean 0.75
            "exp-001": {"metrics": {"loss": 0.5}, "config": {"lr": 0.001},
                        "random_seed": 1, "code_branch": "ml-opt/a",
                        "code_proposal": "a", "method_tier": "method_tuned_hp",
                        "status": "completed"},
            "exp-002": {"metrics": {"loss": 1.0}, "config": {"lr": 0.001},
                        "random_seed": 2, "code_branch": "ml-opt/a",
                        "code_proposal": "a", "method_tier": "method_tuned_hp",
                        "status": "completed"},
            # branch b: consistent 0.7
            "exp-003": {"metrics": {"loss": 0.7}, "config": {"lr": 0.001},
                        "random_seed": 1, "code_branch": "ml-opt/b",
                        "code_proposal": "b", "method_tier": "method_tuned_hp",
                        "status": "completed"},
            "exp-004": {"metrics": {"loss": 0.7}, "config": {"lr": 0.001},
                        "random_seed": 2, "code_branch": "ml-opt/b",
                        "code_proposal": "b", "method_tier": "method_tuned_hp",
                        "status": "completed"},
        })
        ranked = rank_methods_for_stacking(load_results(str(tmp_path)), "loss", lower_is_better=True)
        # b (mean 0.7) ranks above a (mean 0.75) despite a's lucky 0.5 seed
        assert ranked[0]["code_proposal"] == "b"
        assert ranked[0]["replicates"]["n"] == 2
        assert ranked[1]["replicates"]["std"] is not None

    def test_branch_scores_expose_replicates(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_result(results_dir, "baseline", "completed", {"lr": 0.01}, {"loss": 1.0})
        _write_result(results_dir, "exp-1", "completed", {"lr": 0.001}, {"loss": 0.7},
                      code_branch="ml-opt/a", random_seed=1)
        _write_result(results_dir, "exp-2", "completed", {"lr": 0.001}, {"loss": 0.9},
                      code_branch="ml-opt/a", random_seed=2)
        scores = compute_branch_scores(load_results(str(results_dir)), "loss")
        entry = scores["ml-opt/a"]
        assert entry["sample_count"] == 2
        assert entry["replicates"]["n"] == 2
        assert abs(entry["best_metric"] - 0.8) < 1e-9  # replicate mean, not lucky 0.7


# ======================================================================
# TestLowNStats (Batch C, Task 16)
# ======================================================================


class TestLowNStats:
    """n/low_n on correlations, small-n significance gating, zero-centered deltas."""

    def test_correlations_carry_n_and_low_n(self):
        results = {
            f"exp-{i}": {"status": "completed", "config": {"lr": 0.001 * (i + 1)},
                         "metrics": {"loss": 0.5 + i * 0.1}}
            for i in range(4)
        }
        out = identify_correlations(results, "loss")
        entry = next(c for c in out["correlations"] if c["param"] == "lr")
        assert entry["n"] == 4
        assert entry["low_n"] is True

    def test_interaction_not_significant_at_n5(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_result(results_dir, "baseline", "completed", {"lr": 0.01}, {"loss": 1.0})
        # XOR-like lr x batch_size pattern: interaction rho = 0.975, but the
        # exact critical value at n=5 is 1.0 -> annotated not significant.
        configs = [
            (0.01, 16, 0.2), (0.01, 64, 1.0), (0.001, 16, 1.0),
            (0.001, 64, 0.25), (0.005, 32, 0.6),
        ]
        for i, (lr, bs, loss) in enumerate(configs):
            _write_result(results_dir, f"exp-{i+1}", "completed",
                          {"lr": lr, "batch_size": bs}, {"loss": loss})
        out = detect_hp_interactions(load_results(str(results_dir)), "loss")
        assert len(out["interactions"]) == 1
        entry = out["interactions"][0]
        assert entry["n_experiments"] == 5
        assert entry["significant"] is False
        assert "not significant at this n" in entry["note"]

    def test_compute_deltas_zero_centered_fallback(self):
        results = {
            "baseline": {"metrics": {"score": 0.01}, "config": {}},
            "exp-1": {"metrics": {"score": 1.0}, "config": {"lr": 0.1}},
            "exp-2": {"metrics": {"score": -1.0}, "config": {"lr": 0.01}},
        }
        deltas = compute_deltas(results, "baseline", "score")
        assert all(d["batch_std"] is not None for d in deltas)
        assert all(d["delta_pct"] is None for d in deltas)
        assert all("delta_vs_spread" in d for d in deltas)

    def test_compute_deltas_normal_baseline_keeps_pct(self):
        results = {
            "baseline": {"metrics": {"loss": 1.0}, "config": {}},
            "exp-1": {"metrics": {"loss": 0.9}, "config": {"lr": 0.1}},
            "exp-2": {"metrics": {"loss": 0.8}, "config": {"lr": 0.01}},
        }
        deltas = compute_deltas(results, "baseline", "loss")
        assert all(d["delta_pct"] is not None for d in deltas)
        assert all(d["batch_std"] is not None for d in deltas)
        assert not any("delta_vs_spread" in d for d in deltas)
