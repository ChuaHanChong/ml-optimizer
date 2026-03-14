"""Tests for dashboard.py."""

import json
import sys
from pathlib import Path

import pytest
from unittest import mock

from dashboard import generate_dashboard, _load_dashboard_data, _format_value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_experiments(exp_root: Path, num=3, metric="loss"):
    """Create baseline + experiments + pipeline state."""
    results = exp_root / "results"
    results.mkdir(parents=True, exist_ok=True)
    reports = exp_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    baseline = {
        "exp_id": "baseline", "status": "completed",
        "config": {"lr": 0.001, "batch_size": 32},
        "metrics": {metric: 1.0},
    }
    (results / "baseline.json").write_text(json.dumps(baseline))

    for i in range(1, num + 1):
        status = "completed" if i <= num - 1 else "failed"
        exp = {
            "exp_id": f"exp-{i:03d}", "status": status,
            "config": {"lr": 0.001 * i, "batch_size": 32},
            "metrics": {metric: 1.0 - 0.1 * i} if status == "completed" else {},
            "iteration": 1,
        }
        (results / f"exp-{i:03d}.json").write_text(json.dumps(exp))

    state = {
        "phase": 7, "iteration": 3,
        "running_experiments": [],
        "user_choices": {
            "primary_metric": metric,
            "lower_is_better": True,
            "budget_mode": "auto",
        },
    }
    (exp_root / "pipeline-state.json").write_text(json.dumps(state))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def test_load_dashboard_data_basic(tmp_path):
    """_load_dashboard_data loads experiments, baseline, and state."""
    _create_experiments(tmp_path)
    data = _load_dashboard_data(str(tmp_path))
    assert data["baseline"] is not None
    assert len(data["experiments"]) > 0
    assert data["primary_metric"] == "loss"
    assert data["lower_is_better"] is True


def test_load_dashboard_data_empty(tmp_path):
    """_load_dashboard_data handles empty directory."""
    data = _load_dashboard_data(str(tmp_path))
    assert data["experiments"] == []
    assert data["baseline"] is None


def test_load_dashboard_data_with_dead_ends(tmp_path):
    """_load_dashboard_data loads dead ends."""
    _create_experiments(tmp_path)
    reports = tmp_path / "reports"
    de = {"dead_ends": [{"technique": "test", "reason": "bad"}]}
    (reports / "dead-ends.json").write_text(json.dumps(de))
    data = _load_dashboard_data(str(tmp_path))
    assert len(data["dead_ends"]) == 1


def test_load_dashboard_data_with_agenda(tmp_path):
    """_load_dashboard_data loads research agenda."""
    _create_experiments(tmp_path)
    reports = tmp_path / "reports"
    ag = {"ideas": [{"id": "i1", "name": "Test", "status": "untried", "priority": 7}]}
    (reports / "research-agenda.json").write_text(json.dumps(ag))
    data = _load_dashboard_data(str(tmp_path))
    assert len(data["agenda"]) == 1


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def test_dashboard_generates_html(tmp_path):
    """generate_dashboard creates dashboard.html."""
    _create_experiments(tmp_path)
    path = generate_dashboard(str(tmp_path))
    assert Path(path).exists()
    assert path.endswith("dashboard.html")
    html = Path(path).read_text()
    assert "<!DOCTYPE html>" in html
    assert "ML Optimizer Dashboard" in html


def test_dashboard_html_contains_experiments(tmp_path):
    """Dashboard HTML includes experiment data."""
    _create_experiments(tmp_path, num=3)
    path = generate_dashboard(str(tmp_path))
    html = Path(path).read_text()
    assert "exp-001" in html
    assert "exp-002" in html


def test_dashboard_html_contains_baseline(tmp_path):
    """Dashboard HTML shows baseline reference."""
    _create_experiments(tmp_path)
    path = generate_dashboard(str(tmp_path))
    html = Path(path).read_text()
    assert "baseline" in html.lower()


def test_dashboard_empty_results(tmp_path):
    """Dashboard handles empty experiment directory."""
    (tmp_path / "results").mkdir(parents=True)
    path = generate_dashboard(str(tmp_path))
    html = Path(path).read_text()
    assert "ML Optimizer Dashboard" in html
    assert "0" in html  # 0 experiments


def test_dashboard_auto_infer_direction(tmp_path):
    """Dashboard reads metric direction from pipeline state."""
    _create_experiments(tmp_path)
    path = generate_dashboard(str(tmp_path))
    html = Path(path).read_text()
    assert "lower is better" in html


def test_dashboard_with_agenda_and_dead_ends(tmp_path):
    """Dashboard includes agenda and dead-end sections."""
    _create_experiments(tmp_path)
    reports = tmp_path / "reports"
    de = {"dead_ends": [{"technique": "mixup", "reason": "no improvement"}]}
    (reports / "dead-ends.json").write_text(json.dumps(de))
    ag = {"ideas": [{"id": "i1", "name": "CutMix", "status": "untried", "priority": 8, "source": "paper"}]}
    (reports / "research-agenda.json").write_text(json.dumps(ag))
    path = generate_dashboard(str(tmp_path))
    html = Path(path).read_text()
    assert "Research Agenda" in html
    assert "CutMix" in html
    assert "Dead Ends" in html
    assert "mixup" in html


def test_dashboard_with_errors(tmp_path):
    """Dashboard includes error summary when errors exist."""
    _create_experiments(tmp_path)
    reports = tmp_path / "reports"
    error_log = {
        "events": [],
        "summary": {
            "total_events": 3,
            "by_category": {"training_failure": 2, "divergence": 1},
            "by_severity": {"critical": 1, "warning": 2},
        },
    }
    (reports / "error-log.json").write_text(json.dumps(error_log))
    path = generate_dashboard(str(tmp_path))
    html = Path(path).read_text()
    assert "Error Summary" in html
    assert "training_failure" in html


def test_dashboard_timeline_svg(tmp_path):
    """Dashboard generates SVG timeline."""
    _create_experiments(tmp_path, num=5)
    path = generate_dashboard(str(tmp_path))
    html = Path(path).read_text()
    assert "<svg" in html
    assert "circle" in html


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def test_format_value_none():
    assert _format_value(None) == "—"


def test_format_value_float():
    assert _format_value(0.12345) == "0.1235"


def test_format_value_int():
    assert _format_value(42) == "42"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_generates_html(tmp_path):
    """CLI generates dashboard.html."""
    _create_experiments(tmp_path)
    from dashboard import _cli_main
    with mock.patch.object(sys, "argv", ["db", str(tmp_path)]):
        _cli_main()
    assert (tmp_path / "reports" / "dashboard.html").exists()


def test_cli_no_args():
    """CLI with no args exits 1."""
    from dashboard import _cli_main
    with mock.patch.object(sys, "argv", ["db"]):
        with pytest.raises(SystemExit) as exc_info:
            _cli_main()
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Live dashboard
# ---------------------------------------------------------------------------


def test_dashboard_live_mode(tmp_path):
    """Live mode adds auto-refresh meta tag."""
    _create_experiments(tmp_path)
    path = generate_dashboard(str(tmp_path), live=True)
    html = Path(path).read_text()
    assert 'http-equiv="refresh"' in html
    assert 'content="30"' in html


def test_dashboard_no_auto_refresh_by_default(tmp_path):
    """Non-live mode omits auto-refresh."""
    _create_experiments(tmp_path)
    # Mark pipeline as not running
    state = json.loads((tmp_path / "pipeline-state.json").read_text())
    state["status"] = "completed"
    (tmp_path / "pipeline-state.json").write_text(json.dumps(state))
    path = generate_dashboard(str(tmp_path), live=False)
    html = Path(path).read_text()
    assert 'http-equiv="refresh"' not in html


def test_dashboard_running_experiments(tmp_path):
    """Dashboard shows running experiments section."""
    _create_experiments(tmp_path)
    state = json.loads((tmp_path / "pipeline-state.json").read_text())
    state["running_experiments"] = ["exp-010", "exp-011"]
    state["status"] = "running"
    (tmp_path / "pipeline-state.json").write_text(json.dumps(state))
    path = generate_dashboard(str(tmp_path))
    html = Path(path).read_text()
    assert "Running Experiments" in html
    assert "exp-010" in html
    assert "exp-011" in html


def test_dashboard_method_explanations(tmp_path):
    """Dashboard shows method explanations for code-change proposals."""
    _create_experiments(tmp_path)
    results = tmp_path / "results"
    manifest = {
        "original_branch": "main",
        "strategy": "git_branch",
        "proposals": [{
            "name": "FocalLoss",
            "slug": "focal-loss",
            "status": "validated",
            "files_modified": ["train.py", "loss.py"],
            "implementation_strategy": "from_scratch",
            "proposal_source": "paper",
            "explanation": "Replaces CE with Focal Loss for class imbalance",
            "diff_summary": {
                "files_changed": 2, "lines_added": 30, "lines_removed": 5,
                "changed_functions": ["compute_loss"],
            },
        }],
    }
    (results / "implementation-manifest.json").write_text(json.dumps(manifest))
    path = generate_dashboard(str(tmp_path))
    html = Path(path).read_text()
    assert "FocalLoss" in html
    assert "Replaces CE with Focal Loss" in html
    assert "compute_loss" in html
    assert "+30" in html


def test_dashboard_no_methods_for_hp_only(tmp_path):
    """Dashboard omits method section when no code-change proposals."""
    _create_experiments(tmp_path)
    path = generate_dashboard(str(tmp_path))
    html = Path(path).read_text()
    assert "Method Implementation Details" not in html


def test_dashboard_html_escaping(tmp_path):
    """Dashboard escapes HTML in proposal names and explanations."""
    _create_experiments(tmp_path)
    results = tmp_path / "results"
    manifest = {
        "original_branch": "main", "strategy": "git_branch",
        "proposals": [{
            "name": "<script>alert('xss')</script>",
            "slug": "xss", "status": "validated",
            "explanation": "Uses <b>bold</b> & special chars",
            "proposal_source": "paper",
        }],
    }
    (results / "implementation-manifest.json").write_text(json.dumps(manifest))
    path = generate_dashboard(str(tmp_path))
    html = Path(path).read_text()
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;b&gt;" in html


def test_cli_live_flag(tmp_path):
    """CLI --live flag produces auto-refresh HTML."""
    _create_experiments(tmp_path)
    from dashboard import _cli_main
    with mock.patch.object(sys, "argv", ["db", str(tmp_path), "--live"]):
        _cli_main()
    html = (tmp_path / "reports" / "dashboard.html").read_text()
    assert 'http-equiv="refresh"' in html
