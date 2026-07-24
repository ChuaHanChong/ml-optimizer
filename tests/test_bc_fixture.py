"""Smoke tests for the behavior-cloning fixture — demo-format detection + transitions counting.

Covers: robomimic-style script detection (D2), HDF5 data-path validation,
transitions-vs-episodes ground truth (D3), the research skill's D3 contract,
and that the tiny BC trainer runs and learns. Data/run tests skip cleanly
without numpy/h5py/torch.
"""

import subprocess
import sys

import pytest

from conftest import FIXTURES, SKILLS_DIR

from parse_logs import extract_metric_trajectory, parse_log
from prerequisites_check import detect_dataset_format, validate_data_path

BC_FIXTURE = FIXTURES / "bc-demos"
MAKE_DEMOS = BC_FIXTURE / "make_demos.py"
TRAIN_BC = BC_FIXTURE / "train_bc.py"


def test_fixture_scripts_exist():
    for script, flags in ((MAKE_DEMOS, ("--demos", "--steps", "--seed")),
                          (TRAIN_BC, ("--demos", "--epochs", "--lr", "--seed"))):
        assert script.is_file(), f"missing fixture script: {script}"
        text = script.read_text()
        for flag in flags:
            assert flag in text, f"{script.name} missing {flag} flag"


def test_script_detected_as_robomimic():
    """D2: _FORMAT_PATTERNS recognizes the robomimic-style loader (AST-level, no deps)."""
    res = detect_dataset_format(str(TRAIN_BC))
    assert res["format"] == "robomimic" or "robomimic" in res["patterns_found"], res


@pytest.fixture(scope="module")
def demos_file(tmp_path_factory):
    """Generate the synthetic demo file once per module."""
    pytest.importorskip("numpy")
    pytest.importorskip("h5py")
    out = tmp_path_factory.mktemp("bc") / "demos.hdf5"
    result = subprocess.run(
        [sys.executable, str(MAKE_DEMOS), str(out),
         "--demos", "20", "--steps", "50", "--seed", "0"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"make_demos failed: {result.stderr}"
    return out


class TestDemoData:

    def test_hdf5_data_path_validates(self, demos_file):
        res = validate_data_path(str(demos_file), "hdf5")
        assert res["exists"] and res["readable"] and res["non_empty"]
        assert res["format_matches"] is True

    def test_transitions_counted_not_episodes(self, demos_file):
        """D3 ground truth: this dataset is 1000 transitions, not 20 samples."""
        h5py = pytest.importorskip("h5py")
        with h5py.File(demos_file, "r") as f:
            demos = list(f["data"].keys())
            total = int(f["data"].attrs["total"])
            per_demo = [int(f["data"][d].attrs["num_samples"]) for d in demos]
        assert len(demos) == 20
        assert total == sum(per_demo) == 1000
        assert total != len(demos)  # transitions, not episodes


@pytest.mark.slow
def test_bc_training_runs_and_learns(demos_file, tmp_path):
    pytest.importorskip("torch")
    result = subprocess.run(
        [sys.executable, str(TRAIN_BC), "--demos", str(demos_file),
         "--epochs", "50", "--lr", "1e-2", "--seed", "0"],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"train_bc failed: {result.stderr}"
    log = tmp_path / "train.log"
    log.write_text(result.stdout)
    records = parse_log(str(log))
    losses = extract_metric_trajectory(records, "loss")
    assert len(losses) == 50
    assert losses[-1] < losses[0]  # BC learned the linear expert
    assert records[0]["transitions"] == 1000.0


def test_research_skill_counts_transitions_not_episodes():
    """D3: Small Dataset Awareness counts transitions for demo data, routes to IL techniques."""
    text = (SKILLS_DIR / "research" / "SKILL.md").read_text()
    section = text.split("### Small Dataset Awareness", 1)[1].split("\n### ", 1)[0]
    assert "transitions" in section
    assert "action chunking" in section
