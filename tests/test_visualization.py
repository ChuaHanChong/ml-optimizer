"""Tests for dashboard.py and excalidraw_gen.py (visualization utilities)."""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from dashboard import generate_dashboard, _load_dashboard_data, _format_value, _cli_main as dashboard_cli
from excalidraw_gen import (
    _rect,
    _text,
    _arrow,
    _write_excalidraw,
    generate_pipeline_diagram,
    generate_comparison_diagram,
    generate_hp_landscape,
    generate_architecture_diagram,
    _cli_main as excalidraw_cli,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _create_experiments(exp_root: Path, num=3, metric="loss"):
    """Create baseline + experiments + pipeline state for dashboard tests."""
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


def _create_mock_experiments(exp_root: Path, num_exps=3, metric="loss"):
    """Create baseline + experiment results for excalidraw tests."""
    results = exp_root / "results"
    results.mkdir(parents=True, exist_ok=True)

    baseline = {
        "exp_id": "baseline", "status": "completed",
        "config": {"lr": 0.001, "batch_size": 32},
        "metrics": {metric: 1.0},
    }
    (results / "baseline.json").write_text(json.dumps(baseline))

    for i in range(1, num_exps + 1):
        exp = {
            "exp_id": f"exp-{i:03d}", "status": "completed",
            "config": {"lr": 0.001 * i, "batch_size": 32 * i},
            "metrics": {metric: 1.0 - 0.1 * i},
        }
        (results / f"exp-{i:03d}.json").write_text(json.dumps(exp))


# ===========================================================================
# TestDashboard
# ===========================================================================


class TestDashboard:
    """Tests for dashboard.py -- data loading, HTML generation, CLI."""

    # --- Data loading ---

    def test_load_data_basic(self, tmp_path):
        """_load_dashboard_data loads experiments, baseline, and state."""
        _create_experiments(tmp_path)
        data = _load_dashboard_data(str(tmp_path))
        assert data["baseline"] is not None
        assert len(data["experiments"]) > 0
        assert data["primary_metric"] == "loss"
        assert data["lower_is_better"] is True

    def test_load_data_empty(self, tmp_path):
        """_load_dashboard_data handles empty directory."""
        data = _load_dashboard_data(str(tmp_path))
        assert data["experiments"] == []
        assert data["baseline"] is None

    @pytest.mark.parametrize("extra_file,extra_key,setup_data,expected_len", [
        (
            "dead-ends.json", "dead_ends",
            {"dead_ends": [{"technique": "test", "reason": "bad"}]},
            1,
        ),
        (
            "research-agenda.json", "agenda",
            {"ideas": [{"id": "i1", "name": "Test", "status": "untried", "priority": 7}]},
            1,
        ),
    ], ids=["dead_ends", "agenda"])
    def test_load_data_with_extra_files(self, tmp_path, extra_file, extra_key, setup_data, expected_len):
        """_load_dashboard_data loads optional report files (dead ends, agenda)."""
        _create_experiments(tmp_path)
        reports = tmp_path / "reports"
        (reports / extra_file).write_text(json.dumps(setup_data))
        data = _load_dashboard_data(str(tmp_path))
        assert len(data[extra_key]) == expected_len

    # --- HTML generation ---

    def test_generates_html_with_content(self, tmp_path):
        """generate_dashboard creates valid HTML with experiments, baseline, and metric direction."""
        _create_experiments(tmp_path, num=3)
        path = generate_dashboard(str(tmp_path))
        assert Path(path).exists()
        assert path.endswith("dashboard.html")
        html = Path(path).read_text()
        assert "<!DOCTYPE html>" in html
        assert "ML Optimizer Dashboard" in html
        assert "exp-001" in html
        assert "exp-002" in html
        assert "baseline" in html.lower()
        assert "lower is better" in html

    def test_empty_results(self, tmp_path):
        """Dashboard handles empty experiment directory."""
        (tmp_path / "results").mkdir(parents=True)
        path = generate_dashboard(str(tmp_path))
        html = Path(path).read_text()
        assert "ML Optimizer Dashboard" in html
        assert "0" in html

    def test_with_agenda_and_dead_ends(self, tmp_path):
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

    def test_with_errors(self, tmp_path):
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

    def test_timeline_svg(self, tmp_path):
        """Dashboard generates SVG timeline."""
        _create_experiments(tmp_path, num=5)
        path = generate_dashboard(str(tmp_path))
        html = Path(path).read_text()
        assert "<svg" in html
        assert "circle" in html

    def test_method_explanations_and_hp_only(self, tmp_path):
        """Dashboard shows method details when manifest exists, omits when absent."""
        _create_experiments(tmp_path)
        # Without manifest: no method section
        path = generate_dashboard(str(tmp_path))
        html = Path(path).read_text()
        assert "Method Implementation Details" not in html

        # With manifest: method section present
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

    def test_html_escaping(self, tmp_path):
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

    # --- Live mode ---

    @pytest.mark.parametrize("live,completed,expect_refresh", [
        (True, False, True),
        (False, True, False),
    ], ids=["live_on", "live_off_completed"])
    def test_auto_refresh(self, tmp_path, live, completed, expect_refresh):
        """Auto-refresh meta tag depends on live flag and pipeline status."""
        _create_experiments(tmp_path)
        if completed:
            state = json.loads((tmp_path / "pipeline-state.json").read_text())
            state["status"] = "completed"
            (tmp_path / "pipeline-state.json").write_text(json.dumps(state))
        path = generate_dashboard(str(tmp_path), live=live)
        html = Path(path).read_text()
        if expect_refresh:
            assert 'http-equiv="refresh"' in html
            assert 'content="30"' in html
        else:
            assert 'http-equiv="refresh"' not in html

    def test_running_experiments(self, tmp_path):
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

    # --- _format_value ---

    @pytest.mark.parametrize("input_val,expected", [
        (None, "\u2014"),
        (0.12345, "0.1235"),
        (42, "42"),
    ], ids=["none", "float", "int"])
    def test_format_value(self, input_val, expected):
        """_format_value formats various types correctly."""
        assert _format_value(input_val) == expected

    # --- CLI ---

    def test_cli_generates_html(self, tmp_path):
        """CLI generates dashboard.html."""
        _create_experiments(tmp_path)
        with mock.patch.object(sys, "argv", ["db", str(tmp_path)]):
            dashboard_cli()
        assert (tmp_path / "reports" / "dashboard.html").exists()

    def test_cli_no_args(self):
        """CLI with no args exits 1."""
        with mock.patch.object(sys, "argv", ["db"]):
            with pytest.raises(SystemExit) as exc_info:
                dashboard_cli()
            assert exc_info.value.code == 1

    def test_cli_live_flag(self, tmp_path):
        """CLI --live flag produces auto-refresh HTML."""
        _create_experiments(tmp_path)
        with mock.patch.object(sys, "argv", ["db", str(tmp_path), "--live"]):
            dashboard_cli()
        html = (tmp_path / "reports" / "dashboard.html").read_text()
        assert 'http-equiv="refresh"' in html


# ===========================================================================
# TestExcalidraw
# ===========================================================================


class TestExcalidraw:
    """Tests for excalidraw_gen.py -- element helpers, file output, diagrams, CLI."""

    # --- Element helpers (parametrized) ---

    @pytest.mark.parametrize("fn,args,kwargs,checks", [
        (
            _rect, (10, 20, 100, 50), {},
            [("len", 1), ("type", "rectangle"), ("x", 10), ("width", 100)],
        ),
        (
            _rect, (0, 0, 200, 60), {"label": "Hello"},
            [("len", 2), ("type", "rectangle"), ("text_type", "text"),
             ("text_value", "Hello"), ("container_link", True)],
        ),
        (
            _text, (50, 100, "Test text"), {"font_size": 20},
            [("single_type", "text"), ("text_value", "Test text"), ("fontSize", 20)],
        ),
        (
            _arrow, (0, 0, 100, 50), {},
            [("single_type", "arrow"), ("points", [[0, 0], [100, 50]]),
             ("endArrowhead", "arrow")],
        ),
    ], ids=["rect_plain", "rect_with_label", "text", "arrow"])
    def test_element_helpers(self, fn, args, kwargs, checks):
        """Element helper functions produce correct Excalidraw elements."""
        result = fn(*args, **kwargs)
        for check in checks:
            key, val = check
            if key == "len":
                assert len(result) == val
            elif key == "type":
                assert result[0]["type"] == val
            elif key == "x":
                assert result[0]["x"] == val
            elif key == "width":
                assert result[0]["width"] == val
            elif key == "text_type":
                assert result[1]["type"] == val
            elif key == "text_value":
                if isinstance(result, list):
                    assert result[1]["text"] == val
                else:
                    assert result["text"] == val
            elif key == "container_link":
                assert result[1]["containerId"] == result[0]["id"]
            elif key == "single_type":
                assert result["type"] == val
            elif key == "fontSize":
                assert result["fontSize"] == val
            elif key == "points":
                assert result["points"] == val
            elif key == "endArrowhead":
                assert result["endArrowhead"] == val

    # --- File output ---

    def test_write_excalidraw(self, tmp_path):
        """_write_excalidraw produces valid JSON and creates parent directories."""
        # Valid JSON output
        out = tmp_path / "test.excalidraw"
        elems = [_text(0, 0, "hello")]
        path = _write_excalidraw(out, elems)
        data = json.loads(Path(path).read_text())
        assert data["type"] == "excalidraw"
        assert data["version"] == 2
        assert len(data["elements"]) == 1

        # Creates parent dirs
        nested = tmp_path / "a" / "b" / "test.excalidraw"
        _write_excalidraw(nested, [])
        assert nested.exists()

    # --- Pipeline diagram ---

    def test_pipeline_diagram(self, tmp_path):
        """Pipeline diagram produces valid excalidraw file."""
        _create_mock_experiments(tmp_path, num_exps=3)
        path = generate_pipeline_diagram(str(tmp_path), "loss")
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert data["type"] == "excalidraw"
        assert len(data["elements"]) > 0

    def test_pipeline_empty(self, tmp_path):
        """Pipeline diagram handles empty results gracefully."""
        (tmp_path / "results").mkdir()
        path = generate_pipeline_diagram(str(tmp_path), "loss")
        data = json.loads(Path(path).read_text())
        texts = [e for e in data["elements"] if e.get("type") == "text"]
        assert any("No experiment" in t.get("text", "") for t in texts)

    # --- Comparison diagram ---

    def test_comparison_diagram(self, tmp_path):
        """Comparison diagram shows two experiments side by side."""
        _create_mock_experiments(tmp_path, num_exps=2)
        path = generate_comparison_diagram(str(tmp_path), "exp-001", "exp-002")
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        texts = [e.get("text", "") for e in data["elements"] if e.get("type") == "text"]
        text_joined = " ".join(texts)
        assert "exp-001" in text_joined
        assert "exp-002" in text_joined

    def test_comparison_missing_exp(self, tmp_path):
        """Comparison diagram handles missing experiment gracefully."""
        _create_mock_experiments(tmp_path, num_exps=1)
        path = generate_comparison_diagram(str(tmp_path), "exp-001", "nonexistent")
        assert Path(path).exists()

    # --- HP landscape ---

    def test_hp_landscape(self, tmp_path):
        """HP landscape creates scatter-style diagram."""
        _create_mock_experiments(tmp_path, num_exps=5)
        path = generate_hp_landscape(str(tmp_path), "lr", "loss")
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert len(data["elements"]) > 3

    def test_hp_landscape_unknown_hp(self, tmp_path):
        """HP landscape handles HP not in any config."""
        _create_mock_experiments(tmp_path, num_exps=2)
        path = generate_hp_landscape(str(tmp_path), "nonexistent_hp", "loss")
        data = json.loads(Path(path).read_text())
        texts = [e.get("text", "") for e in data["elements"] if e.get("type") == "text"]
        assert any("No experiments" in t for t in texts)

    # --- Architecture diagram ---

    def test_architecture_diagram(self, tmp_path):
        """Architecture diagram reads from manifest."""
        results = tmp_path / "results"
        results.mkdir(parents=True)
        manifest = {
            "original_branch": "main",
            "strategy": "git_branch",
            "proposals": [{
                "name": "CutMix",
                "slug": "cutmix",
                "branch": "ml-opt/cutmix",
                "status": "validated",
                "files_modified": ["train.py", "augment.py"],
                "complexity": "low",
                "implementation_strategy": "from_scratch",
                "notes": "Added CutMix augmentation",
            }],
        }
        (results / "implementation-manifest.json").write_text(json.dumps(manifest))
        path = generate_architecture_diagram(str(tmp_path), "CutMix")
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        texts = [e.get("text", "") for e in data["elements"] if e.get("type") == "text"]
        text_joined = " ".join(texts)
        assert "CutMix" in text_joined
        assert "BEFORE" in text_joined
        assert "AFTER" in text_joined

    def test_architecture_not_found(self, tmp_path):
        """Architecture diagram handles missing proposal."""
        (tmp_path / "results").mkdir()
        path = generate_architecture_diagram(str(tmp_path), "nonexistent")
        data = json.loads(Path(path).read_text())
        texts = [e.get("text", "") for e in data["elements"] if e.get("type") == "text"]
        assert any("not found" in t for t in texts)

    # --- CLI ---

    def test_cli_pipeline(self, tmp_path):
        """CLI pipeline mode generates file."""
        _create_mock_experiments(tmp_path, num_exps=2)
        with mock.patch.object(sys, "argv", ["eg", str(tmp_path), "pipeline", "loss"]):
            excalidraw_cli()
        assert (tmp_path / "artifacts" / "pipeline-overview.excalidraw").exists()

    @pytest.mark.parametrize("argv_extra", [
        ["unknown", "x"],
        [],
    ], ids=["unknown_mode", "no_args"])
    def test_cli_error_exits(self, tmp_path, argv_extra):
        """CLI exits with code 1 on invalid args or unknown mode."""
        argv = ["eg"] + ([str(tmp_path)] + argv_extra if argv_extra else [])
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit) as exc_info:
                excalidraw_cli()
            assert exc_info.value.code == 1
