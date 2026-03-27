"""Tests for Hyperagent per-skill scripts — archive management and parent selection.

Tests the Hyperagents submodule-backed skills that replace hyperagent_adapter.py.
All operations use gl_utils.py directly via the native gen_X/ directory layout.
"""

import json
import os
import subprocess
import sys

import pytest

PLUGIN_ROOT = os.path.join(os.path.dirname(__file__), "..")
SKILLS_DIR = os.path.join(PLUGIN_ROOT, "skills", "hyperagent", "Hyperagents", "skills")
INIT_SCRIPT = os.path.join(SKILLS_DIR, "hyperagent-init", "scripts", "init_archive.py")
SELECT_SCRIPT = os.path.join(SKILLS_DIR, "hyperagent-select", "scripts", "select_parent.py")
EVAL_SCRIPT = os.path.join(SKILLS_DIR, "hyperagent-eval", "scripts", "run_eval.py")
ARCHIVE_SCRIPT = os.path.join(SKILLS_DIR, "hyperagent-archive", "scripts", "archive_utils.py")
SCRIPTS_DIR_PY = os.path.join(PLUGIN_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR_PY)

from pipeline_state import init_hyperagent_state


def run_init(output_dir, *extra_args, env=None):
    """Run init_archive.py."""
    cmd = [sys.executable, INIT_SCRIPT, "--output-dir", output_dir] + list(extra_args)
    run_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=run_env)
    assert result.returncode == 0, f"Init failed: {result.stderr}"
    return json.loads(result.stdout.strip())


def run_select(output_dir, *extra_args, env=None):
    """Run select_parent.py."""
    cmd = [sys.executable, SELECT_SCRIPT, "--output-dir", output_dir] + list(extra_args)
    run_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=run_env)
    assert result.returncode == 0, f"Select failed: {result.stderr}"
    return json.loads(result.stdout.strip())


def run_archive(output_dir, subcmd, *extra_args, env=None):
    """Run archive_utils.py subcommand."""
    cmd = [sys.executable, ARCHIVE_SCRIPT, subcmd, "--output-dir", output_dir] + list(extra_args)
    run_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=run_env)
    assert result.returncode == 0, f"Archive {subcmd} failed: {result.stderr}"
    return json.loads(result.stdout.strip())


@pytest.fixture
def tmp_hyperagent_dir(tmp_path):
    """Create experiments/hyperagent dir with baseline and pipeline state."""
    exp = tmp_path / "experiments"
    results = exp / "results"
    results.mkdir(parents=True)
    baseline = {"metrics": {"accuracy": 0.823, "loss": 0.45}, "config": {"lr": 0.001}}
    (results / "baseline.json").write_text(json.dumps(baseline))
    state = {"user_choices": {"primary_metric": "accuracy", "lower_is_better": False}}
    (exp / "pipeline-state.json").write_text(json.dumps(state))
    h_dir = exp / "hyperagent"
    h_dir.mkdir()
    return str(h_dir)


@pytest.fixture
def initialized_dir(tmp_hyperagent_dir):
    """Return an initialized hyperagent dir with gen_initial."""
    run_init(tmp_hyperagent_dir, env={"HYPERAGENT_METRIC": "accuracy"})
    return tmp_hyperagent_dir


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_creates_archive(self, tmp_hyperagent_dir):
        out = run_init(tmp_hyperagent_dir, env={"HYPERAGENT_METRIC": "accuracy"})
        assert out["status"] == "initialized"
        assert out["entries"] >= 1
        assert os.path.exists(os.path.join(tmp_hyperagent_dir, "archive.jsonl"))

    def test_init_creates_gen_initial(self, tmp_hyperagent_dir):
        run_init(tmp_hyperagent_dir, env={"HYPERAGENT_METRIC": "accuracy"})
        gen_dir = os.path.join(tmp_hyperagent_dir, "gen_initial")
        assert os.path.isdir(gen_dir)
        assert os.path.exists(os.path.join(gen_dir, "metadata.json"))
        assert os.path.exists(os.path.join(gen_dir, "ml_eval", "report.json"))

    def test_init_reads_baseline_metric(self, tmp_hyperagent_dir):
        run_init(tmp_hyperagent_dir, env={"HYPERAGENT_METRIC": "accuracy"})
        report_path = os.path.join(tmp_hyperagent_dir, "gen_initial", "ml_eval", "report.json")
        with open(report_path) as f:
            report = json.load(f)
        assert report["accuracy"] == 0.823

    def test_init_warns_when_baseline_missing(self, tmp_path):
        exp = tmp_path / "experiments"
        exp.mkdir()
        h_dir = exp / "hyperagent"
        h_dir.mkdir()
        out = run_init(str(h_dir), env={"HYPERAGENT_METRIC": "loss"})
        assert out["status"] == "initialized"
        assert "warning" in out

    def test_init_idempotent(self, initialized_dir):
        out = run_init(initialized_dir, env={"HYPERAGENT_METRIC": "accuracy"})
        assert out["status"] == "already_initialized"

    def test_init_seeds_from_manifest(self, tmp_hyperagent_dir):
        exp_dir = os.path.dirname(tmp_hyperagent_dir)
        manifest = {
            "proposals": [
                {"name": "Label Smoothing", "slug": "label-smoothing",
                 "branch": "ml-opt/label-smoothing", "status": "validated"},
                {"name": "Failed Prop", "slug": "failed",
                 "branch": "ml-opt/failed", "status": "validation_failed"},
            ]
        }
        results_dir = os.path.join(exp_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "implementation-manifest.json"), "w") as f:
            json.dump(manifest, f)
        out = run_init(tmp_hyperagent_dir, env={"HYPERAGENT_METRIC": "accuracy"})
        assert out["entries"] == 2  # initial + 1 validated proposal


# ---------------------------------------------------------------------------
# Select Parent
# ---------------------------------------------------------------------------


class TestSelectParent:
    def test_select_best(self, initialized_dir):
        out = run_select(initialized_dir, "--strategy", "best",
                         env={"HYPERAGENT_METRIC": "accuracy"})
        assert out["strategy"] == "best"
        assert out["genid"] == "initial"

    def test_select_random(self, initialized_dir):
        out = run_select(initialized_dir, "--strategy", "random",
                         env={"HYPERAGENT_METRIC": "accuracy"})
        assert out["strategy"] == "random"

    def test_select_score_child_prop(self, initialized_dir):
        out = run_select(initialized_dir, "--strategy", "score_child_prop",
                         env={"HYPERAGENT_METRIC": "accuracy"})
        assert out["strategy"] == "score_child_prop"


# ---------------------------------------------------------------------------
# Archive: Add
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_entry(self, initialized_dir):
        entry = {
            "genid": 1,
            "code_branch": "ml-opt/test",
            "mutation_type": "llm_patch",
            "fitness_score": 0.85,
            "parent_genid": "initial",
            "status": "evaluated",
        }
        out = run_archive(initialized_dir, "add", json.dumps(entry),
                          env={"HYPERAGENT_METRIC": "accuracy"})
        assert out["status"] == "added"

    def test_add_creates_gen_dir(self, initialized_dir):
        entry = {
            "genid": 1,
            "code_branch": "ml-opt/test",
            "mutation_type": "llm_patch",
            "fitness_score": 0.85,
            "parent_genid": "initial",
            "status": "evaluated",
        }
        run_archive(initialized_dir, "add", json.dumps(entry),
                    env={"HYPERAGENT_METRIC": "accuracy"})
        gen_dir = os.path.join(initialized_dir, "gen_1")
        assert os.path.isdir(gen_dir)
        assert os.path.exists(os.path.join(gen_dir, "metadata.json"))


# ---------------------------------------------------------------------------
# Archive: Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_after_init(self, initialized_dir):
        out = run_archive(initialized_dir, "stats",
                          env={"HYPERAGENT_METRIC": "accuracy"})
        assert out["total_entries"] >= 1

    def test_stats_empty(self, tmp_path):
        h_dir = tmp_path / "empty"
        h_dir.mkdir()
        out = run_archive(str(h_dir), "stats",
                          env={"HYPERAGENT_METRIC": "loss"})
        assert out["entries"] == 0


# ---------------------------------------------------------------------------
# Archive: Best
# ---------------------------------------------------------------------------


class TestBest:
    def test_best_after_init(self, initialized_dir):
        out = run_archive(initialized_dir, "best",
                          env={"HYPERAGENT_METRIC": "accuracy"})
        assert isinstance(out, list)
        assert len(out) >= 1

    def test_best_n(self, initialized_dir):
        out = run_archive(initialized_dir, "best", "-n", "1",
                          env={"HYPERAGENT_METRIC": "accuracy"})
        assert len(out) == 1


# ---------------------------------------------------------------------------
# Archive: Lineage
# ---------------------------------------------------------------------------


class TestLineage:
    def test_lineage_initial(self, initialized_dir):
        out = run_archive(initialized_dir, "lineage", "initial",
                          env={"HYPERAGENT_METRIC": "accuracy"})
        assert len(out["lineage"]) == 1
        assert out["lineage"][0]["genid"] == "initial"


# ---------------------------------------------------------------------------
# Archive: Update Fitness
# ---------------------------------------------------------------------------


class TestUpdateFitness:
    def test_update_fitness(self, initialized_dir):
        out = run_archive(initialized_dir, "update-fitness", "initial", "0.95",
                          env={"HYPERAGENT_METRIC": "accuracy"})
        assert out["status"] == "updated"
        assert out["fitness_score"] == 0.95

    def test_update_fitness_not_found(self, initialized_dir):
        cmd = [sys.executable, ARCHIVE_SCRIPT, "update-fitness",
               "--output-dir", initialized_dir, "nonexistent", "0.5"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                                env={**os.environ, "HYPERAGENT_METRIC": "accuracy"})
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Archive: Operator Stats & Prune
# ---------------------------------------------------------------------------


class TestOperatorStats:
    def test_operator_stats_empty(self, initialized_dir):
        out = run_archive(initialized_dir, "operator-stats",
                          env={"HYPERAGENT_METRIC": "accuracy"})
        assert isinstance(out, dict)


class TestPrune:
    def test_prune_after_init(self, initialized_dir):
        out = run_archive(initialized_dir, "prune",
                          env={"HYPERAGENT_METRIC": "accuracy"})
        assert "pruned" in out
        assert "remaining" in out


# ---------------------------------------------------------------------------
# Pipeline State Integration
# ---------------------------------------------------------------------------


class TestPipelineStateInit:
    def test_init_hyperagent_state(self):
        result = init_hyperagent_state()
        assert result["enabled"] is True

    def test_init_hyperagent_state_can_disable(self):
        result = init_hyperagent_state(enabled=False)
        assert result["enabled"] is False
