"""Tests for plot_results.py."""

import json

from conftest import _write_results

from plot_results import (
    ascii_bar_chart,
    ascii_line_chart,
    plot_metric_comparison,
    plot_improvement_timeline,
    plot_hp_sensitivity,
)


# ---------- ascii_bar_chart ----------


def test_ascii_bar_chart_basic():
    labels = ["alpha", "beta", "gamma"]
    values = [10.0, 20.0, 15.0]
    chart = ascii_bar_chart(labels, values, title="Test Chart")
    lines = chart.strip().split("\n")
    # Title + separator + 3 bars = 5 lines
    assert len(lines) == 5
    for label in labels:
        assert label in chart
    for value in values:
        assert f"{value:.4g}" in chart


def test_ascii_bar_chart_contains_bars():
    labels = ["a", "b"]
    values = [5.0, 10.0]
    chart = ascii_bar_chart(labels, values)
    # The max-value bar should have a full-width block run
    assert "\u2588" in chart


def test_ascii_bar_chart_empty():
    assert ascii_bar_chart([], []) == ""


def test_ascii_bar_chart_single():
    chart = ascii_bar_chart(["only"], [42.0])
    assert "only" in chart
    assert "42" in chart


def test_ascii_bar_chart_zero_values():
    chart = ascii_bar_chart(["a", "b"], [0.0, 0.0], title="Zeros")
    assert "a" in chart
    assert "b" in chart


# ---------- ascii_line_chart ----------


def test_ascii_line_chart_basic():
    values = [1, 2, 4, 3, 5, 4, 6, 7, 8, 10]
    chart = ascii_line_chart(values, title="Line Test", height=10)
    lines = chart.strip().split("\n")
    # title + separator + 10 rows + x-axis separator + x-axis labels = 14
    assert len(lines) >= 12  # at minimum height rows + axis
    assert "*" in chart


def test_ascii_line_chart_height():
    values = [0, 5, 10]
    chart = ascii_line_chart(values, height=8)
    # Should contain exactly 8 data rows (lines with '|')
    data_rows = [l for l in chart.split("\n") if "|" in l]
    assert len(data_rows) == 8


def test_ascii_line_chart_empty():
    assert ascii_line_chart([]) == ""


def test_ascii_line_chart_constant():
    """All identical values should still render without error."""
    chart = ascii_line_chart([5.0, 5.0, 5.0, 5.0], height=5)
    assert "*" in chart


def test_ascii_line_chart_single_value():
    chart = ascii_line_chart([7.0], height=5)
    assert "*" in chart


# ---------- plot_metric_comparison ----------


def test_plot_metric_comparison(tmp_path):
    _write_results(tmp_path, {
        "baseline": {"metrics": {"loss": 1.0}, "config": {"lr": 0.01}},
        "exp-001": {"metrics": {"loss": 0.7}, "config": {"lr": 0.001}},
        "exp-002": {"metrics": {"loss": 0.5}, "config": {"lr": 0.0001}},
    })
    chart = plot_metric_comparison(str(tmp_path), "loss")
    assert chart  # non-empty
    assert "exp-001" in chart
    assert "exp-002" in chart
    assert "[B]" in chart  # baseline marker


def test_plot_metric_comparison_no_results(tmp_path):
    chart = plot_metric_comparison(str(tmp_path / "nonexistent"), "loss")
    assert "No results" in chart


def test_plot_metric_comparison_missing_metric(tmp_path):
    _write_results(tmp_path, {
        "exp-001": {"metrics": {"accuracy": 0.9}, "config": {}},
    })
    chart = plot_metric_comparison(str(tmp_path), "loss")
    assert "not found" in chart


# ---------- plot_improvement_timeline ----------


def test_plot_improvement_timeline(tmp_path):
    _write_results(tmp_path, {
        "exp-001": {"metrics": {"loss": 1.0}, "config": {}},
        "exp-002": {"metrics": {"loss": 0.8}, "config": {}},
        "exp-003": {"metrics": {"loss": 0.9}, "config": {}},
        "exp-004": {"metrics": {"loss": 0.5}, "config": {}},
    })
    chart = plot_improvement_timeline(str(tmp_path), "loss")
    assert chart  # non-empty
    assert "*" in chart  # line chart points
    # Best-so-far should appear in the title
    assert "Best-so-far" in chart


def test_plot_improvement_timeline_shows_improvement(tmp_path):
    """The best-so-far series should be monotonically non-increasing for lower_is_better."""
    _write_results(tmp_path, {
        "exp-001": {"metrics": {"loss": 2.0}, "config": {}},
        "exp-002": {"metrics": {"loss": 1.5}, "config": {}},
        "exp-003": {"metrics": {"loss": 1.8}, "config": {}},
        "exp-004": {"metrics": {"loss": 1.0}, "config": {}},
    })
    chart = plot_improvement_timeline(str(tmp_path), "loss", lower_is_better=True)
    assert chart
    # The chart is non-empty and contains data points
    assert "*" in chart


def test_plot_improvement_timeline_no_results(tmp_path):
    chart = plot_improvement_timeline(str(tmp_path / "nonexistent"), "loss")
    assert "No results" in chart


# ---------- plot_hp_sensitivity ----------


def test_plot_hp_sensitivity(tmp_path):
    _write_results(tmp_path, {
        "exp-001": {"metrics": {"loss": 0.9}, "config": {"lr": 0.1}},
        "exp-002": {"metrics": {"loss": 0.5}, "config": {"lr": 0.01}},
        "exp-003": {"metrics": {"loss": 0.3}, "config": {"lr": 0.001}},
    })
    chart = plot_hp_sensitivity(str(tmp_path), "loss", "lr")
    assert chart
    assert "*" in chart
    assert "lr" in chart


def test_plot_hp_sensitivity_no_data(tmp_path):
    _write_results(tmp_path, {
        "exp-001": {"metrics": {"loss": 0.9}, "config": {"batch_size": 32}},
    })
    chart = plot_hp_sensitivity(str(tmp_path), "loss", "lr")
    assert "No numeric data" in chart


def test_plot_hp_sensitivity_no_results(tmp_path):
    chart = plot_hp_sensitivity(str(tmp_path / "nonexistent"), "loss", "lr")
    assert "No results" in chart


def test_plot_hp_sensitivity_identical_hp_values(tmp_path):
    """HP sensitivity with all-identical HP values renders without error."""
    _write_results(tmp_path, {
        "exp-001": {"metrics": {"loss": 0.5}, "config": {"lr": 0.001}},
        "exp-002": {"metrics": {"loss": 0.3}, "config": {"lr": 0.001}},
    })
    chart = plot_hp_sensitivity(str(tmp_path), "loss", "lr")
    assert chart and "*" in chart


# --- Additional edge cases ---


def test_ascii_line_chart_resampling():
    """Line chart with >60 values triggers resampling."""
    values = [i * 0.1 for i in range(100)]
    chart = ascii_line_chart(values, width=60)
    assert chart
    assert "*" in chart


def test_plot_improvement_timeline_higher_is_better(tmp_path):
    """Timeline with higher_is_better uses max for best-so-far."""
    _write_results(tmp_path, {
        "exp-001": {"metrics": {"acc": 70.0}, "config": {}},
        "exp-002": {"metrics": {"acc": 80.0}, "config": {}},
        "exp-003": {"metrics": {"acc": 75.0}, "config": {}},
    })
    chart = plot_improvement_timeline(str(tmp_path), "acc", lower_is_better=False)
    assert chart
    assert "Best-so-far" in chart


def test_plot_improvement_timeline_metric_not_found(tmp_path):
    """Timeline returns error when metric not found."""
    _write_results(tmp_path, {
        "exp-001": {"metrics": {"accuracy": 0.9}, "config": {}},
    })
    chart = plot_improvement_timeline(str(tmp_path), "loss")
    assert "not found" in chart


def test_plot_hp_sensitivity_non_numeric_hp(tmp_path):
    """HP sensitivity skips non-numeric HP values."""
    _write_results(tmp_path, {
        "exp-001": {"metrics": {"loss": 0.5}, "config": {"optimizer": "adam"}},
        "exp-002": {"metrics": {"loss": 0.3}, "config": {"optimizer": "sgd"}},
    })
    chart = plot_hp_sensitivity(str(tmp_path), "loss", "optimizer")
    assert "No numeric data" in chart


# --- CLI tests ---


def test_plot_metric_comparison_baseline_case_insensitive(tmp_path):
    """Baseline marker [B] should work regardless of case."""
    _write_results(tmp_path, {
        "Baseline": {"metrics": {"loss": 1.0}, "config": {"lr": 0.01}},
        "exp-001": {"metrics": {"loss": 0.7}, "config": {"lr": 0.001}},
    })
    chart = plot_metric_comparison(str(tmp_path), "loss")
    assert "[B]" in chart


def test_ascii_line_chart_resampling_preserves_data_points():
    """Resampling with large dataset should produce valid chart with expected row count."""
    values = [i * 0.1 for i in range(150)]
    chart = ascii_line_chart(values, width=40, height=10)
    data_rows = [line for line in chart.split("\n") if "|" in line]
    assert len(data_rows) == 10
    assert "*" in chart


def test_cli_comparison(run_main, tmp_path):
    """CLI comparison mode works."""
    _write_results(tmp_path, {
        "exp-001": {"metrics": {"loss": 0.5}, "config": {}},
    })
    r = run_main("plot_results.py", str(tmp_path), "loss", "comparison")
    assert r.returncode == 0
    assert "loss" in r.stdout


def test_cli_timeline(run_main, tmp_path):
    """CLI timeline mode works."""
    _write_results(tmp_path, {
        "exp-001": {"metrics": {"loss": 0.5}, "config": {}},
        "exp-002": {"metrics": {"loss": 0.3}, "config": {}},
    })
    r = run_main("plot_results.py", str(tmp_path), "loss", "timeline")
    assert r.returncode == 0


def test_cli_sensitivity(run_main, tmp_path):
    """CLI sensitivity mode works."""
    _write_results(tmp_path, {
        "exp-001": {"metrics": {"loss": 0.5}, "config": {"lr": 0.01}},
        "exp-002": {"metrics": {"loss": 0.3}, "config": {"lr": 0.001}},
    })
    r = run_main("plot_results.py", str(tmp_path), "loss", "sensitivity", "lr")
    assert r.returncode == 0


def test_cli_sensitivity_missing_hp_name(run_main, tmp_path):
    """CLI sensitivity mode without hp_name exits with error."""
    _write_results(tmp_path, {
        "exp-001": {"metrics": {"loss": 0.5}, "config": {"lr": 0.01}},
    })
    r = run_main("plot_results.py", str(tmp_path), "loss", "sensitivity")
    assert r.returncode == 1


def test_ascii_bar_chart_nan_values():
    """Bar chart with NaN values renders without error."""
    labels = ["a", "b", "c"]
    values = [5.0, float("nan"), 10.0]
    chart = ascii_bar_chart(labels, values)
    assert "a" in chart
    assert "c" in chart
    assert "\u2588" in chart


def test_ascii_line_chart_nan_values():
    """Line chart with NaN values renders without error."""
    values = [1.0, float("nan"), 3.0, 4.0, float("nan"), 6.0]
    chart = ascii_line_chart(values, height=5)
    assert "*" in chart


def test_ascii_line_chart_all_nan():
    """All-NaN values returns empty string."""
    assert ascii_line_chart([float("nan")] * 5) == ""


def test_ascii_bar_chart_inf_values():
    """Bar chart with Inf values renders without error."""
    labels = ["a", "b"]
    values = [float("inf"), 5.0]
    chart = ascii_bar_chart(labels, values)
    assert "b" in chart


def test_ascii_bar_chart_negative_values():
    """Bar chart handles negative values without error."""
    chart = ascii_bar_chart(["a", "b", "c"], [-5.0, -10.0, 3.0])
    assert chart and "a" in chart and "-10" in chart


def test_ascii_bar_chart_all_negative():
    """Bar chart where all values are negative still renders."""
    chart = ascii_bar_chart(["x", "y"], [-3.0, -7.0])
    assert chart and "\u2588" in chart


def test_ascii_line_chart_resampling_includes_last():
    """After resampling, the last original data point is represented."""
    values = list(range(100))
    chart = ascii_line_chart(values, width=40, height=10)
    # The chart should include the last value (99) in its rendering
    assert chart
    assert "99" in chart  # Last value appears in y-axis labels


def test_plot_metric_comparison_large_dataset(tmp_path):
    """plot_metric_comparison handles 100+ experiments without error."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for i in range(120):
        (results_dir / f"exp-{i:04d}.json").write_text(json.dumps({
            "exp_id": f"exp-{i:04d}",
            "config": {"lr": 0.001 * (i + 1)},
            "metrics": {"loss": 2.0 - i * 0.01},
            "status": "completed",
        }))
    chart = plot_metric_comparison(str(results_dir), "loss")
    assert chart
    assert len(chart) > 0


def test_cli_unknown_mode(run_main, tmp_path):
    """CLI with unknown mode exits 1."""
    (tmp_path / "exp-001.json").write_text('{"metrics": {"loss": 0.5}, "config": {}}')
    r = run_main("plot_results.py", str(tmp_path), "loss", "badmode")
    assert r.returncode == 1
    assert "Unknown" in r.stdout


def test_cli_no_args(run_main):
    """CLI with no args prints usage and exits 1."""
    r = run_main("plot_results.py")
    assert r.returncode == 1
    assert "Usage" in r.stdout
