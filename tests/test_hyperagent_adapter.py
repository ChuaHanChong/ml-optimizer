"""Tests for scripts/hyperagent_adapter.py — archive management and parent selection."""

import json
import os
import shutil
import subprocess
import sys
import pytest

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, SCRIPTS_DIR)
ADAPTER = os.path.join(SCRIPTS_DIR, "hyperagent_adapter.py")
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

from pipeline_state import init_hyperagent_state


def run_adapter(exp_root, *args):
    """Run hyperagent_adapter.py and return parsed JSON output."""
    cmd = [sys.executable, ADAPTER, exp_root] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, f"Adapter failed: {result.stderr}"
    return json.loads(result.stdout.strip())


@pytest.fixture
def tmp_exp(tmp_path):
    """Create a temporary experiments directory with baseline."""
    exp = tmp_path / "experiments"
    results = exp / "results"
    results.mkdir(parents=True)
    baseline = {"metrics": {"accuracy": 0.823, "loss": 0.45}, "config": {"lr": 0.001}}
    (results / "baseline.json").write_text(json.dumps(baseline))
    return str(exp)


@pytest.fixture
def tmp_exp_with_archive(tmp_path):
    """Create experiments dir with a pre-built archive."""
    exp = tmp_path / "experiments"
    exp.mkdir(parents=True)
    fixture = os.path.join(FIXTURES_DIR, "sample_code_archive.jsonl")
    shutil.copy2(fixture, str(exp / "code-archive.jsonl"))
    return str(exp)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_creates_archive(self, tmp_exp):
        out = run_adapter(tmp_exp, "init")
        assert out["status"] == "initialized"
        assert out["entries"] >= 1
        assert os.path.exists(os.path.join(tmp_exp, "code-archive.jsonl"))

    def test_init_reads_baseline_metric(self, tmp_exp):
        # Write pipeline state with primary_metric
        state = {"user_choices": {"primary_metric": "accuracy"}}
        with open(os.path.join(tmp_exp, "pipeline-state.json"), "w") as f:
            json.dump(state, f)
        run_adapter(tmp_exp, "init")
        # Read archive and check fitness
        with open(os.path.join(tmp_exp, "code-archive.jsonl")) as f:
            entry = json.loads(f.readline())
        assert entry["fitness_score"] == 0.823
        assert entry["genid"] == "gen-000"

    def test_init_warns_when_baseline_missing(self, tmp_path):
        """Init without baseline.json should return warning and fitness=0.0."""
        exp = tmp_path / "experiments"
        exp.mkdir(parents=True)
        out = run_adapter(str(exp), "init")
        assert out["status"] == "initialized"
        assert "warning" in out
        assert "baseline.json not found" in out["warning"]
        # Check gen-000 has fitness=0.0
        with open(str(exp / "code-archive.jsonl")) as f:
            entry = json.loads(f.readline())
        assert entry["fitness_score"] == 0.0

    def test_init_idempotent(self, tmp_exp):
        run_adapter(tmp_exp, "init")
        out = run_adapter(tmp_exp, "init")
        assert out["status"] == "already_initialized"

    def test_init_seeds_from_manifest(self, tmp_exp):
        manifest = {
            "proposals": [
                {"name": "Label Smoothing", "slug": "label-smoothing",
                 "branch": "ml-opt/label-smoothing", "status": "validated"},
                {"name": "Failed Prop", "slug": "failed",
                 "branch": "ml-opt/failed", "status": "validation_failed"},
            ]
        }
        with open(os.path.join(tmp_exp, "results", "implementation-manifest.json"), "w") as f:
            json.dump(manifest, f)
        out = run_adapter(tmp_exp, "init")
        assert out["entries"] == 2  # gen-000 + 1 validated proposal


# ---------------------------------------------------------------------------
# Add
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_entry(self, tmp_exp):
        run_adapter(tmp_exp, "init")
        entry = {
            "code_branch": "ml-opt/test",
            "mutation_type": "llm_patch",
            "mutation_description": "Test mutation",
            "fitness_score": 0.85,
            "parent_genid": "gen-000",
            "status": "evaluated",
        }
        out = run_adapter(tmp_exp, "add", json.dumps(entry))
        assert out["status"] == "added"
        assert out["genid"] == "gen-001"

    def test_add_computes_lineage(self, tmp_exp):
        run_adapter(tmp_exp, "init")
        entry = {
            "code_branch": "ml-opt/test",
            "mutation_type": "llm_patch",
            "fitness_score": 0.85,
            "parent_genid": "gen-000",
            "status": "evaluated",
        }
        run_adapter(tmp_exp, "add", json.dumps(entry))
        # Read the added entry
        with open(os.path.join(tmp_exp, "code-archive.jsonl")) as f:
            lines = f.readlines()
        added = json.loads(lines[-1])
        assert added["lineage"] == ["gen-000", "gen-001"]
        assert added["lineage_depth"] == 1

    def test_add_increments_parent_children(self, tmp_exp):
        run_adapter(tmp_exp, "init")
        entry = {
            "code_branch": "ml-opt/test",
            "mutation_type": "llm_patch",
            "fitness_score": 0.85,
            "parent_genid": "gen-000",
            "status": "evaluated",
        }
        run_adapter(tmp_exp, "add", json.dumps(entry))
        # Check parent's num_children
        with open(os.path.join(tmp_exp, "code-archive.jsonl")) as f:
            lines = f.readlines()
        parent = json.loads(lines[0])
        assert parent["num_children"] == 1


# ---------------------------------------------------------------------------
# Select Parent
# ---------------------------------------------------------------------------


class TestSelectParent:
    def test_select_best(self, tmp_exp_with_archive):
        out = run_adapter(tmp_exp_with_archive, "select-parent", "best")
        assert out["genid"] == "gen-005"  # highest fitness (0.872)
        assert out["strategy"] == "best"

    def test_select_latest(self, tmp_exp_with_archive):
        out = run_adapter(tmp_exp_with_archive, "select-parent", "latest")
        # Latest evaluated entry (gen-006 is filtered, gen-005 is last evaluated)
        assert out["genid"] == "gen-005"
        assert out["strategy"] == "latest"

    def test_select_random(self, tmp_exp_with_archive):
        out = run_adapter(tmp_exp_with_archive, "select-parent", "random")
        assert out["strategy"] == "random"
        assert out["genid"].startswith("gen-")

    def test_select_score_prop(self, tmp_exp_with_archive):
        out = run_adapter(tmp_exp_with_archive, "select-parent", "score_prop")
        assert out["strategy"] == "score_prop"
        assert out["genid"].startswith("gen-")

    def test_select_score_child_prop(self, tmp_exp_with_archive):
        out = run_adapter(tmp_exp_with_archive, "select-parent", "score_child_prop")
        assert out["strategy"] == "score_child_prop"
        assert out["genid"].startswith("gen-")

    def test_score_child_prop_favors_high_score_low_children(self, tmp_exp_with_archive):
        """score_child_prop should prefer high-scoring entries with few children."""
        # Run 20 times to check statistical tendency
        selections = {}
        for _ in range(20):
            out = run_adapter(tmp_exp_with_archive, "select-parent", "score_child_prop")
            gid = out["genid"]
            selections[gid] = selections.get(gid, 0) + 1
        # gen-005 (score=0.872, 0 children) should be selected frequently
        # gen-000 (score=0.823, 2 children) should be selected less often
        assert "gen-005" in selections, "gen-005 should be selected at least once"

    def test_empty_archive_fallback(self, tmp_exp):
        run_adapter(tmp_exp, "init")
        out = run_adapter(tmp_exp, "select-parent", "best")
        assert out["genid"] == "gen-000"


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------


class TestLineage:
    def test_trace_lineage(self, tmp_exp_with_archive):
        out = run_adapter(tmp_exp_with_archive, "lineage", "gen-005")
        assert len(out["lineage"]) == 4
        genids = [e["genid"] for e in out["lineage"]]
        assert genids == ["gen-000", "gen-001", "gen-003", "gen-005"]

    def test_lineage_root(self, tmp_exp_with_archive):
        out = run_adapter(tmp_exp_with_archive, "lineage", "gen-000")
        assert len(out["lineage"]) == 1
        assert out["lineage"][0]["genid"] == "gen-000"

    def test_lineage_not_found(self, tmp_exp_with_archive):
        cmd = [sys.executable, ADAPTER, tmp_exp_with_archive, "lineage", "gen-999"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats(self, tmp_exp_with_archive):
        out = run_adapter(tmp_exp_with_archive, "stats")
        assert out["total_entries"] == 7
        assert out["evaluated"] == 6
        assert out["filtered"] == 1
        assert out["best_score"] == 0.872
        assert out["max_lineage_depth"] == 3
        assert out["mutation_types"]["llm_patch"] == 3
        assert out["mutation_types"]["research_implement"] == 2
        assert out["mutation_types"]["shinka_evolve"] == 1

    def test_stats_empty(self, tmp_exp):
        out = run_adapter(tmp_exp, "stats")
        assert out["entries"] == 0


# ---------------------------------------------------------------------------
# Best
# ---------------------------------------------------------------------------


class TestBest:
    def test_best_default(self, tmp_exp_with_archive):
        out = run_adapter(tmp_exp_with_archive, "best")
        assert len(out) == 5  # default top 5
        assert out[0]["genid"] == "gen-005"
        assert out[0]["fitness_score"] == 0.872

    def test_best_n(self, tmp_exp_with_archive):
        out = run_adapter(tmp_exp_with_archive, "best", "3")
        assert len(out) == 3
        scores = [e["fitness_score"] for e in out]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Operator Stats
# ---------------------------------------------------------------------------


class TestOperatorStats:
    def test_operator_stats(self, tmp_exp_with_archive):
        out = run_adapter(tmp_exp_with_archive, "operator-stats")
        assert "llm_patch" in out
        assert "research_implement" in out
        assert "shinka_evolve" in out
        assert out["llm_patch"]["attempts"] == 3
        assert out["research_implement"]["attempts"] == 2
        # llm_patch: gen-003 (0.867 > 0.823), gen-005 (0.872 > 0.823), gen-006 (filtered, no score)
        assert out["llm_patch"]["improvements"] == 2


# ---------------------------------------------------------------------------
# Prune
# ---------------------------------------------------------------------------


class TestPrune:
    def test_prune_removes_dead_lineages(self, tmp_exp):
        run_adapter(tmp_exp, "init")
        # Add an evaluated entry and a failed entry on different lineages
        run_adapter(tmp_exp, "add", json.dumps({
            "code_branch": "ml-opt/good", "mutation_type": "llm_patch",
            "fitness_score": 0.85, "parent_genid": "gen-000", "status": "evaluated",
        }))
        run_adapter(tmp_exp, "add", json.dumps({
            "code_branch": "ml-opt/bad", "mutation_type": "llm_patch",
            "fitness_score": None, "parent_genid": "gen-000", "status": "failed",
        }))
        out = run_adapter(tmp_exp, "prune")
        assert "gen-002" in out["pruned"]  # failed, no evaluated descendants
        assert out["remaining"] == 2  # gen-000 and gen-001


# ---------------------------------------------------------------------------
# Sigmoid and diversity math validation
# ---------------------------------------------------------------------------


class TestMath:
    """Verify the parent selection math matches Hyperagents' exact formulas."""

    def test_sigmoid_concentrates_probability(self, tmp_exp):
        """Higher scores should get higher sigmoid-transformed probability."""
        run_adapter(tmp_exp, "init")
        # Add entries with different scores
        for i, score in enumerate([0.5, 0.7, 0.9], start=1):
            run_adapter(tmp_exp, "add", json.dumps({
                "code_branch": f"ml-opt/test-{i}",
                "mutation_type": "llm_patch",
                "fitness_score": score,
                "parent_genid": "gen-000",
                "status": "evaluated",
            }))
        # Score_prop should heavily favor the 0.9 entry
        selections = {}
        for _ in range(50):
            out = run_adapter(tmp_exp, "select-parent", "score_prop")
            gid = out["genid"]
            selections[gid] = selections.get(gid, 0) + 1
        # gen-003 (0.9) should dominate due to sigmoid concentration
        assert selections.get("gen-003", 0) > selections.get("gen-001", 0)


# ---------------------------------------------------------------------------
# CLI edge cases
# ---------------------------------------------------------------------------


class TestCLI:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, ADAPTER, "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "hyperagent_adapter" in result.stdout

    def test_unknown_command(self, tmp_exp):
        cmd = [sys.executable, ADAPTER, tmp_exp, "unknown-cmd"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        assert result.returncode != 0

    def test_unknown_strategy(self, tmp_exp):
        run_adapter(tmp_exp, "init")
        cmd = [sys.executable, ADAPTER, tmp_exp, "select-parent", "nonexistent"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Extended coverage: strategy distributions, concurrent safety, deep lineage
# ---------------------------------------------------------------------------


class TestScorePropDistribution:
    """Verify score_prop statistically favors higher-scored entries."""

    def test_score_prop_favors_high_scores(self, tmp_exp):
        run_adapter(tmp_exp, "init")
        for i, score in enumerate([0.50, 0.60, 0.70, 0.80, 0.95], start=1):
            run_adapter(tmp_exp, "add", json.dumps({
                "code_branch": f"ml-opt/t{i}", "mutation_type": "llm_patch",
                "fitness_score": score, "parent_genid": "gen-000", "status": "evaluated",
            }))
        counts = {}
        for _ in range(100):
            out = run_adapter(tmp_exp, "select-parent", "score_prop")
            counts[out["genid"]] = counts.get(out["genid"], 0) + 1
        # gen-005 (0.95) should be selected most often
        assert counts.get("gen-005", 0) > counts.get("gen-001", 0)


class TestDiversityPenalty:
    """Verify score_child_prop penalizes parents with many children."""

    def test_diversity_penalty_reduces_selection(self, tmp_exp):
        run_adapter(tmp_exp, "init")
        # Add a high-score parent
        run_adapter(tmp_exp, "add", json.dumps({
            "code_branch": "ml-opt/prolific", "mutation_type": "llm_patch",
            "fitness_score": 0.90, "parent_genid": "gen-000", "status": "evaluated",
        }))
        # Add 8 children to gen-001 (prolific parent) — penalty = exp(-(8/8)^3) ≈ 0.37
        for i in range(8):
            run_adapter(tmp_exp, "add", json.dumps({
                "code_branch": f"ml-opt/child-{i}", "mutation_type": "llm_patch",
                "fitness_score": 0.85, "parent_genid": "gen-001", "status": "evaluated",
            }))
        # Add a lower-score parent with no children
        run_adapter(tmp_exp, "add", json.dumps({
            "code_branch": "ml-opt/fresh", "mutation_type": "llm_patch",
            "fitness_score": 0.88, "parent_genid": "gen-000", "status": "evaluated",
        }))
        # score_child_prop should favor the fresh parent despite lower score
        counts = {}
        for _ in range(100):
            out = run_adapter(tmp_exp, "select-parent", "score_child_prop")
            counts[out["genid"]] = counts.get(out["genid"], 0) + 1
        # gen-010 (fresh, 0.88, 0 children) should be selected more than gen-001 (prolific, 0.90, 8 children)
        assert counts.get("gen-010", 0) >= counts.get("gen-001", 0)


class TestConcurrentSafety:
    """Verify concurrent archive writes don't lose entries."""

    def test_concurrent_adds(self, tmp_exp):
        run_adapter(tmp_exp, "init")
        # Spawn 5 parallel subprocesses each adding one entry
        import multiprocessing
        def add_entry(idx):
            run_adapter(tmp_exp, "add", json.dumps({
                "code_branch": f"ml-opt/concurrent-{idx}",
                "mutation_type": "llm_patch",
                "fitness_score": 0.80 + idx * 0.01,
                "parent_genid": "gen-000",
                "status": "evaluated",
            }))
        procs = []
        for i in range(5):
            p = multiprocessing.Process(target=add_entry, args=(i,))
            procs.append(p)
            p.start()
        for p in procs:
            p.join(timeout=10)
        # All 5 entries should be in the archive (plus gen-000)
        stats = run_adapter(tmp_exp, "stats")
        assert stats["total_entries"] == 6  # gen-000 + 5 concurrent adds


class TestDeepLineage:
    """Verify lineage tracking through multiple generations."""

    def test_three_level_lineage(self, tmp_exp):
        run_adapter(tmp_exp, "init")
        # gen-000 → gen-001 → gen-002 → gen-003
        for i in range(1, 4):
            parent = f"gen-{i-1:03d}"
            run_adapter(tmp_exp, "add", json.dumps({
                "code_branch": f"ml-opt/level-{i}",
                "mutation_type": "llm_patch",
                "fitness_score": 0.80 + i * 0.03,
                "parent_genid": parent,
                "status": "evaluated",
            }))
        out = run_adapter(tmp_exp, "lineage", "gen-003")
        genids = [e["genid"] for e in out["lineage"]]
        assert genids == ["gen-000", "gen-001", "gen-002", "gen-003"]
        assert len(out["lineage"]) == 4

    def test_prune_preserves_evaluated_lineage(self, tmp_exp):
        """If a leaf is evaluated, its entire lineage should survive pruning."""
        run_adapter(tmp_exp, "init")
        # Build: gen-000 → gen-001 (failed) → gen-002 (evaluated)
        run_adapter(tmp_exp, "add", json.dumps({
            "code_branch": "ml-opt/middle", "mutation_type": "llm_patch",
            "fitness_score": None, "parent_genid": "gen-000", "status": "failed",
        }))
        run_adapter(tmp_exp, "add", json.dumps({
            "code_branch": "ml-opt/leaf", "mutation_type": "llm_patch",
            "fitness_score": 0.85, "parent_genid": "gen-001", "status": "evaluated",
        }))
        out = run_adapter(tmp_exp, "prune")
        # gen-001 is failed but has an evaluated descendant (gen-002) — should be kept
        assert out["remaining"] == 3  # gen-000, gen-001, gen-002
        assert len(out["pruned"]) == 0


class TestPipelineStateInit:
    """Verify init_hyperagent_state function."""

    def test_init_hyperagent_state(self):

        state = init_hyperagent_state(enabled=True)
        assert state["enabled"] is True
        assert state["current_phase"] == "hp_tuning"
        assert state["archive_generation"] == 0
        assert state["meta_improvement_count"] == 0
        assert state["active_meta_patches"] == []
        assert "llm_patch" in state["operator_stats"]
        assert "shinka_evolve" in state["operator_stats"]
        assert "research_implement" in state["operator_stats"]

    def test_init_hyperagent_state_enabled_by_default(self):
        state = init_hyperagent_state()
        assert state["enabled"] is True

    def test_init_hyperagent_state_can_disable(self):
        state = init_hyperagent_state(enabled=False)
        assert state["enabled"] is False


# ---------------------------------------------------------------------------
# Update fitness
# ---------------------------------------------------------------------------


class TestUpdateFitness:
    """Test update-fitness command for HP-tuning feedback."""

    def test_update_fitness(self, tmp_exp):
        run_adapter(tmp_exp, "init")
        run_adapter(tmp_exp, "add", json.dumps({
            "code_branch": "ml-opt/t1", "mutation_type": "llm_patch",
            "fitness_score": 0.85, "parent_genid": "gen-000", "status": "evaluated",
        }))
        # Update fitness after HP tuning improved it
        out = run_adapter(tmp_exp, "update-fitness", "gen-001", "0.91")
        assert out["status"] == "updated"
        assert out["fitness_score"] == 0.91
        # Verify in archive
        best = run_adapter(tmp_exp, "best", "1")
        assert best[0]["fitness_score"] == 0.91

    def test_update_fitness_with_exp_id(self, tmp_exp):
        run_adapter(tmp_exp, "init")
        run_adapter(tmp_exp, "add", json.dumps({
            "code_branch": "ml-opt/t1", "mutation_type": "llm_patch",
            "fitness_score": 0.85, "parent_genid": "gen-000", "status": "evaluated",
        }))
        cmd = [sys.executable, ADAPTER, tmp_exp, "update-fitness", "gen-001", "0.92", "exp-042"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["fitness_score"] == 0.92

    def test_update_fitness_pending_to_evaluated(self, tmp_exp):
        """Updating fitness on a pending entry should set status to evaluated."""
        run_adapter(tmp_exp, "init")
        run_adapter(tmp_exp, "add", json.dumps({
            "code_branch": "ml-opt/t1", "mutation_type": "research_implement",
            "fitness_score": None, "parent_genid": "gen-000", "status": "pending",
        }))
        run_adapter(tmp_exp, "update-fitness", "gen-001", "0.88")
        stats = run_adapter(tmp_exp, "stats")
        assert stats["evaluated"] == 2  # gen-000 + gen-001 now evaluated
        assert stats["pending"] == 0

    def test_update_fitness_not_found(self, tmp_exp):
        run_adapter(tmp_exp, "init")
        cmd = [sys.executable, ADAPTER, tmp_exp, "update-fitness", "gen-999", "0.5"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        assert result.returncode != 0
