"""Tests for excalidraw_gen.py."""

import json
import sys
from pathlib import Path

import pytest

from excalidraw_gen import (
    _rect,
    _text,
    _arrow,
    _write_excalidraw,
    generate_pipeline_diagram,
    generate_comparison_diagram,
    generate_hp_landscape,
    generate_architecture_diagram,
)


# ---------------------------------------------------------------------------
# Element helpers
# ---------------------------------------------------------------------------


def test_rect_creates_rectangle():
    """_rect produces a rectangle element."""
    elems = _rect(10, 20, 100, 50)
    assert len(elems) == 1
    assert elems[0]["type"] == "rectangle"
    assert elems[0]["x"] == 10
    assert elems[0]["width"] == 100


def test_rect_with_label():
    """_rect with label produces rectangle + text."""
    elems = _rect(0, 0, 200, 60, label="Hello")
    assert len(elems) == 2
    assert elems[0]["type"] == "rectangle"
    assert elems[1]["type"] == "text"
    assert elems[1]["text"] == "Hello"
    assert elems[1]["containerId"] == elems[0]["id"]


def test_text_element():
    """_text creates a text element."""
    elem = _text(50, 100, "Test text", font_size=20)
    assert elem["type"] == "text"
    assert elem["text"] == "Test text"
    assert elem["fontSize"] == 20


def test_arrow_element():
    """_arrow creates an arrow with points."""
    elem = _arrow(0, 0, 100, 50)
    assert elem["type"] == "arrow"
    assert elem["points"] == [[0, 0], [100, 50]]
    assert elem["endArrowhead"] == "arrow"


# ---------------------------------------------------------------------------
# Excalidraw file output
# ---------------------------------------------------------------------------


def test_write_excalidraw_valid_json(tmp_path):
    """_write_excalidraw produces valid Excalidraw JSON."""
    out = tmp_path / "test.excalidraw"
    elems = [_text(0, 0, "hello")]
    path = _write_excalidraw(out, elems)
    data = json.loads(Path(path).read_text())
    assert data["type"] == "excalidraw"
    assert data["version"] == 2
    assert len(data["elements"]) == 1


def test_write_excalidraw_creates_dirs(tmp_path):
    """_write_excalidraw creates parent directories."""
    out = tmp_path / "a" / "b" / "test.excalidraw"
    _write_excalidraw(out, [])
    assert out.exists()


# ---------------------------------------------------------------------------
# Helper to create mock experiment data
# ---------------------------------------------------------------------------


def _create_mock_experiments(exp_root: Path, num_exps=3, metric="loss"):
    """Create baseline + experiment results for testing."""
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


# ---------------------------------------------------------------------------
# Pipeline diagram
# ---------------------------------------------------------------------------


def test_generate_pipeline_diagram(tmp_path):
    """Pipeline diagram produces valid excalidraw file."""
    _create_mock_experiments(tmp_path, num_exps=3)
    path = generate_pipeline_diagram(str(tmp_path), "loss")
    assert Path(path).exists()
    data = json.loads(Path(path).read_text())
    assert data["type"] == "excalidraw"
    assert len(data["elements"]) > 0


def test_generate_pipeline_empty(tmp_path):
    """Pipeline diagram handles empty results gracefully."""
    (tmp_path / "results").mkdir()
    path = generate_pipeline_diagram(str(tmp_path), "loss")
    data = json.loads(Path(path).read_text())
    # Should have at least a "no results" message
    texts = [e for e in data["elements"] if e.get("type") == "text"]
    assert any("No experiment" in t.get("text", "") for t in texts)


# ---------------------------------------------------------------------------
# Comparison diagram
# ---------------------------------------------------------------------------


def test_generate_comparison_diagram(tmp_path):
    """Comparison diagram shows two experiments side by side."""
    _create_mock_experiments(tmp_path, num_exps=2)
    path = generate_comparison_diagram(str(tmp_path), "exp-001", "exp-002")
    assert Path(path).exists()
    data = json.loads(Path(path).read_text())
    texts = [e.get("text", "") for e in data["elements"] if e.get("type") == "text"]
    text_joined = " ".join(texts)
    assert "exp-001" in text_joined
    assert "exp-002" in text_joined


def test_generate_comparison_missing_exp(tmp_path):
    """Comparison diagram handles missing experiment gracefully."""
    _create_mock_experiments(tmp_path, num_exps=1)
    path = generate_comparison_diagram(str(tmp_path), "exp-001", "nonexistent")
    assert Path(path).exists()


# ---------------------------------------------------------------------------
# HP landscape
# ---------------------------------------------------------------------------


def test_generate_hp_landscape(tmp_path):
    """HP landscape creates scatter-style diagram."""
    _create_mock_experiments(tmp_path, num_exps=5)
    path = generate_hp_landscape(str(tmp_path), "lr", "loss")
    assert Path(path).exists()
    data = json.loads(Path(path).read_text())
    assert len(data["elements"]) > 3  # At least axes + some dots


def test_generate_hp_landscape_unknown_hp(tmp_path):
    """HP landscape handles HP not in any config."""
    _create_mock_experiments(tmp_path, num_exps=2)
    path = generate_hp_landscape(str(tmp_path), "nonexistent_hp", "loss")
    data = json.loads(Path(path).read_text())
    texts = [e.get("text", "") for e in data["elements"] if e.get("type") == "text"]
    assert any("No experiments" in t for t in texts)


# ---------------------------------------------------------------------------
# Architecture diagram
# ---------------------------------------------------------------------------


def test_generate_architecture_diagram(tmp_path):
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


def test_generate_architecture_not_found(tmp_path):
    """Architecture diagram handles missing proposal."""
    (tmp_path / "results").mkdir()
    path = generate_architecture_diagram(str(tmp_path), "nonexistent")
    data = json.loads(Path(path).read_text())
    texts = [e.get("text", "") for e in data["elements"] if e.get("type") == "text"]
    assert any("not found" in t for t in texts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_pipeline(tmp_path):
    """CLI pipeline mode generates file."""
    _create_mock_experiments(tmp_path, num_exps=2)
    from unittest import mock
    from excalidraw_gen import _cli_main
    with mock.patch.object(sys, "argv", ["eg", str(tmp_path), "pipeline", "loss"]):
        _cli_main()
    assert (tmp_path / "artifacts" / "pipeline-overview.excalidraw").exists()


def test_cli_unknown_mode(tmp_path):
    """CLI unknown mode exits 1."""
    from unittest import mock
    from excalidraw_gen import _cli_main
    with mock.patch.object(sys, "argv", ["eg", str(tmp_path), "unknown", "x"]):
        with pytest.raises(SystemExit) as exc_info:
            _cli_main()
        assert exc_info.value.code == 1


def test_cli_no_args():
    """CLI no args exits 1."""
    from unittest import mock
    from excalidraw_gen import _cli_main
    with mock.patch.object(sys, "argv", ["eg"]):
        with pytest.raises(SystemExit) as exc_info:
            _cli_main()
        assert exc_info.value.code == 1
