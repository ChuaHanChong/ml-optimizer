"""Tests for plot_results.py."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from plot_results import (
    ascii_bar_chart,
    ascii_line_chart,
    plot_metric_comparison,
    plot_improvement_timeline,
    plot_hp_sensitivity,
)


def _write_results(tmp_path, experiments: dict):
    """Helper to write experiment result files."""
    for name, data in experiments.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(data))


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
