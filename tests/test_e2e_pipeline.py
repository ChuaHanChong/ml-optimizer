"""End-to-end pipeline integration tests using Tiny ResNet on CIFAR-10.

Exercises all 6 pipeline phases against a real ML model, verifying that the
existing Python scripts (parse_logs, detect_divergence, experiment_setup,
result_analyzer, gpu_check) work correctly with real training output.
"""

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from parse_logs import parse_log, extract_metric_trajectory
from detect_divergence import check_divergence
from experiment_setup import create_experiment_dirs, next_experiment_id, setup
from result_analyzer import analyze, rank_by_metric, compute_deltas
from pipeline_state import save_state, load_state, validate_phase_requirements, cleanup_stale
from schema_validator import validate_result, validate_baseline, validate_manifest, validate_file
from plot_results import plot_metric_comparison, plot_improvement_timeline, plot_hp_sensitivity
from conftest import FIXTURES, _write_result
from implement_utils import (
    parse_research_proposals, detect_conflicts, validate_syntax,
    write_manifest, backup_files, is_git_repo,
)
RESNET_FIXTURE = FIXTURES / "tiny_resnet_cifar10"


def has_torch():
    """Check if PyTorch is available in the current environment."""
    try:
        import torch
        return True
    except ImportError:
        return False


def has_gpu():
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_python():
    """Get Python executable with PyTorch available.

    Searches conda environments for one with torch, falls back to sys.executable.
    """
    candidates = []
    try:
        result = subprocess.run(
            ["conda", "info", "--base"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            conda_base = Path(result.stdout.strip())
            # Check base env first
            candidates.append(conda_base / "bin" / "python")
            # Then check all envs
            envs_dir = conda_base / "envs"
            if envs_dir.exists():
                for env_dir in sorted(envs_dir.iterdir()):
                    p = env_dir / "bin" / "python"
                    if p.exists():
                        candidates.append(p)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    for candidate in candidates:
        if candidate.exists():
            try:
                check = subprocess.run(
                    [str(candidate), "-c", "import torch"],
                    capture_output=True, timeout=10,
                )
                if check.returncode == 0:
                    return str(candidate)
            except (subprocess.TimeoutExpired, OSError):
                continue
    return sys.executable


# Module-level detection for skip markers
_has_torch = has_torch()
_has_gpu = has_gpu()
_python = get_python()


@pytest.fixture
def project_dir(tmp_path):
    """Copy the tiny_resnet_cifar10 fixture into a tmp_path project directory."""
    project = tmp_path / "project"
    shutil.copytree(RESNET_FIXTURE, project)
    return project


@pytest.fixture
def shared_data_dir(tmp_path):
    """Shared CIFAR-10 data directory to avoid re-downloading per test."""
    data_dir = tmp_path / "cifar_data"
    data_dir.mkdir()
    return data_dir


def run_training(project_dir, output_dir, data_dir, extra_args=None, timeout=300):
    """Run train.py and return (returncode, stdout, stderr)."""
    cmd = [
        _python, str(project_dir / "train.py"),
        "--data_dir", str(data_dir),
        "--output_dir", str(output_dir),
    ]
    if not _has_gpu:
        cmd.extend(["--subset_size", "500", "--epochs", "2"])
    else:
        cmd.extend(["--subset_size", "1000", "--epochs", "3"])
    if extra_args:
        cmd.extend(extra_args)

    env = None
    if _has_gpu:
        import os
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env,
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Phase 1: Model Understanding
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_torch, reason="PyTorch not available")
class TestPhase1ModelUnderstanding:
    """Phase 1: Verify project files are discoverable and parseable."""

    def test_model_file_exists(self):
        assert (RESNET_FIXTURE / "model.py").exists()

    def test_model_has_nn_module(self):
        content = (RESNET_FIXTURE / "model.py").read_text()
        assert "nn.Module" in content
        assert "class TinyResNet" in content

    def test_config_parseable(self):
        import yaml
        config = yaml.safe_load((RESNET_FIXTURE / "config.yaml").read_text())
        assert config["model"]["type"] == "tiny_resnet"
        assert config["training"]["lr"] == 0.01
        assert config["data"]["dataset"] == "cifar10"

    def test_model_instantiates(self):
        import torch
        sys.path.insert(0, str(RESNET_FIXTURE))
        from model import get_model
        model = get_model()
        # Verify parameter count is in expected range
        num_params = sum(p.numel() for p in model.parameters())
        assert 50_000 < num_params < 150_000, f"Param count {num_params} outside expected range"
        # Verify forward pass works
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_gpu_check_script(self):
        """gpu_check.py should return valid JSON with gpus key."""
        result = subprocess.run(
            [_python, str(Path(__file__).parent.parent / "scripts" / "gpu_check.py")],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "gpus" in data


# ---------------------------------------------------------------------------
# Phase 2: Baseline
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not _has_torch, reason="PyTorch not available")
class TestPhase2Baseline:
    """Phase 2: Run baseline training and verify log parsing + divergence check."""

    def test_baseline_training(self, project_dir, shared_data_dir, tmp_path):
        # Create experiment dirs
        exp_root = create_experiment_dirs(str(tmp_path / "exp_project"))
        results_dir = str(Path(exp_root) / "results")

        # Run baseline training
        output_dir = tmp_path / "baseline_output"
        returncode, stdout, stderr = run_training(
            project_dir, output_dir, shared_data_dir,
            extra_args=["--seed", "42"],
        )
        assert returncode == 0, f"Training failed:\nstdout: {stdout}\nstderr: {stderr}"

        # Save log to file for parse_logs
        log_path = tmp_path / "baseline_train.log"
        log_path.write_text(stdout)

        # parse_logs should extract metrics
        records = parse_log(str(log_path))
        assert len(records) >= 2, f"Expected at least 2 log records, got {len(records)}"
        assert "loss" in records[0]
        assert "accuracy" in records[0]
        assert "lr" in records[0]

        # Extract loss trajectory
        losses = extract_metric_trajectory(records, "loss")
        assert len(losses) >= 2

        # Divergence check should show healthy
        div_result = check_divergence(losses)
        assert div_result["diverged"] is False

        # Write baseline.json with correct schema
        baseline_data = {
            "exp_id": "baseline",
            "config": {"lr": 0.01, "batch_size": 64},
            "metrics": {
                "loss": losses[-1],
                "accuracy": extract_metric_trajectory(records, "accuracy")[-1],
            },
            "status": "completed",
        }
        baseline_path = Path(results_dir) / "baseline.json"
        baseline_path.write_text(json.dumps(baseline_data, indent=2))
        assert baseline_path.exists()

        # Verify checkpoint was saved
        assert (output_dir / "model.pth").exists()


# ---------------------------------------------------------------------------
# Phase 3: User Checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_torch, reason="PyTorch not available")
class TestPhase3UserCheckpoint:
    """Phase 3: Verify baseline.json has all required checkpoint keys."""

    def test_baseline_schema(self, tmp_path):
        baseline = {
            "exp_id": "baseline",
            "config": {"lr": 0.01, "batch_size": 64},
            "metrics": {"loss": 1.5, "accuracy": 45.0},
            "status": "completed",
        }
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(baseline))
        data = json.loads(path.read_text())

        # All required keys for checkpoint
        assert "metrics" in data
        assert "loss" in data["metrics"]
        assert "accuracy" in data["metrics"]
        assert "lr" in data["config"]
        assert "batch_size" in data["config"]


# ---------------------------------------------------------------------------
# Phase 5: Experiment Loop
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not _has_torch, reason="PyTorch not available")
class TestPhase5ExperimentLoop:
    """Phase 5: Run multiple experiments and verify analysis."""

    def test_experiment_loop(self, project_dir, shared_data_dir, tmp_path):
        exp_project = tmp_path / "exp_project"
        exp_root = create_experiment_dirs(str(exp_project))
        results_dir = str(Path(exp_root) / "results")

        # Write baseline result
        baseline_output = tmp_path / "baseline_out"
        rc, stdout, stderr = run_training(
            project_dir, baseline_output, shared_data_dir,
            extra_args=["--lr", "0.01", "--seed", "42"],
        )
        assert rc == 0, f"Baseline failed: {stderr}"
        log_path = tmp_path / "baseline.log"
        log_path.write_text(stdout)
        records = parse_log(str(log_path))
        losses = extract_metric_trajectory(records, "loss")
        accs = extract_metric_trajectory(records, "accuracy")

        Path(results_dir, "baseline.json").write_text(json.dumps({
            "exp_id": "baseline",
            "config": {"lr": 0.01},
            "metrics": {"loss": losses[-1], "accuracy": accs[-1]},
            "status": "completed",
        }))

        # Experiment 1: lower lr
        exp1_setup = setup(str(exp_project), "python train.py --lr 0.001", config={"lr": 0.001})
        assert exp1_setup["exp_id"] == "exp-001"

        exp1_output = tmp_path / "exp1_out"
        rc, stdout, stderr = run_training(
            project_dir, exp1_output, shared_data_dir,
            extra_args=["--lr", "0.001", "--seed", "42"],
        )
        assert rc == 0, f"Exp1 failed: {stderr}"
        log1 = tmp_path / "exp1.log"
        log1.write_text(stdout)
        records1 = parse_log(str(log1))
        losses1 = extract_metric_trajectory(records1, "loss")
        accs1 = extract_metric_trajectory(records1, "accuracy")

        # Update exp-001 with results
        exp1_data = json.loads(Path(exp1_setup["config_path"]).read_text())
        exp1_data["metrics"] = {"loss": losses1[-1], "accuracy": accs1[-1]}
        exp1_data["status"] = "completed"
        Path(exp1_setup["config_path"]).write_text(json.dumps(exp1_data, indent=2))

        # Experiment 2: higher lr
        exp2_setup = setup(str(exp_project), "python train.py --lr 0.1", config={"lr": 0.1})
        assert exp2_setup["exp_id"] == "exp-002"

        exp2_output = tmp_path / "exp2_out"
        rc, stdout, stderr = run_training(
            project_dir, exp2_output, shared_data_dir,
            extra_args=["--lr", "0.1", "--seed", "42"],
        )
        assert rc == 0, f"Exp2 failed: {stderr}"
        log2 = tmp_path / "exp2.log"
        log2.write_text(stdout)
        records2 = parse_log(str(log2))
        losses2 = extract_metric_trajectory(records2, "loss")
        accs2 = extract_metric_trajectory(records2, "accuracy")

        exp2_data = json.loads(Path(exp2_setup["config_path"]).read_text())
        exp2_data["metrics"] = {"loss": losses2[-1], "accuracy": accs2[-1]}
        exp2_data["status"] = "completed"
        Path(exp2_setup["config_path"]).write_text(json.dumps(exp2_data, indent=2))

        # Divergence checks
        assert check_divergence(losses1)["diverged"] is False
        assert check_divergence(losses2)["diverged"] is False

        # result_analyzer: full analysis
        analysis = analyze(results_dir, "loss", baseline_id="baseline", lower_is_better=True)
        assert analysis["num_experiments"] >= 3
        assert len(analysis["ranking"]) >= 3
        assert len(analysis["deltas"]) >= 2
        # Ranking should be sorted (lowest loss first)
        ranking_vals = [r["value"] for r in analysis["ranking"]]
        assert ranking_vals == sorted(ranking_vals)

        # Correlations should exist
        assert "correlations" in analysis["correlations"]


@pytest.mark.slow
@pytest.mark.skipif(not _has_torch, reason="PyTorch not available")
class TestPhase5DivergenceDetection:
    """Phase 5: Verify divergence detection with extreme learning rate."""

    def test_divergent_training(self, project_dir, shared_data_dir, tmp_path):
        output_dir = tmp_path / "divergent_out"
        rc, stdout, stderr = run_training(
            project_dir, output_dir, shared_data_dir,
            extra_args=["--lr", "10.0", "--seed", "42", "--subset_size", "200", "--epochs", "2"],
        )

        if rc != 0:
            # Training crashed — that's acceptable for lr=10.0
            return

        # If training survived, check that divergence detection flags it
        log_path = tmp_path / "divergent.log"
        log_path.write_text(stdout)
        records = parse_log(str(log_path))
        losses = extract_metric_trajectory(records, "loss")

        if losses:
            div_result = check_divergence(losses)
            # With lr=10.0, loss should either explode, produce NaN, or be very high
            # We check: diverged OR final loss is much worse than initial
            if not div_result["diverged"]:
                # Even if not flagged as diverged, loss should be pathologically high
                assert losses[-1] > 2.0, (
                    f"lr=10.0 should cause bad training but final loss={losses[-1]}"
                )


# ---------------------------------------------------------------------------
# Phase 6: Report
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_torch, reason="PyTorch not available")
class TestPhase6Report:
    """Phase 6: Verify analysis output has correct schema for reporting."""

    def test_analysis_report_schema(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Write 3 experiment results
        for name, loss, acc, lr in [
            ("baseline", 2.0, 30.0, 0.01),
            ("exp-001", 1.5, 45.0, 0.001),
            ("exp-002", 1.8, 38.0, 0.1),
        ]:
            _write_result(results_dir, name, "completed", {"lr": lr}, {"loss": loss, "accuracy": acc})

        analysis = analyze(str(results_dir), "loss", baseline_id="baseline")
        assert analysis["num_experiments"] >= 3
        # Ranking sorted by loss (lower is better)
        assert len(analysis["ranking"]) >= 3
        ranking_vals = [r["value"] for r in analysis["ranking"]]
        assert ranking_vals == sorted(ranking_vals)
        # Deltas non-empty with correct schema
        assert len(analysis["deltas"]) >= 1
        for delta in analysis["deltas"]:
            assert "exp_id" in delta
            assert "delta" in delta
            assert "delta_pct" in delta
            assert "value" in delta

        # A3: Verify plot_results produces non-empty charts
        chart = plot_metric_comparison(str(results_dir), "loss")
        assert chart
        assert "[B]" in chart

        timeline = plot_improvement_timeline(str(results_dir), "loss")
        assert timeline

        sensitivity = plot_hp_sensitivity(str(results_dir), "loss", "lr")
        assert sensitivity


# ---------------------------------------------------------------------------
# Full Integration Test
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not _has_torch, reason="PyTorch not available")
class TestFullPipelineIntegration:
    """End-to-end: all phases sequentially."""

    def test_full_pipeline(self, project_dir, shared_data_dir, tmp_path):
        exp_project = tmp_path / "full_pipeline"
        exp_project.mkdir()

        # Phase 1: create experiment dirs
        exp_root = create_experiment_dirs(str(exp_project))
        results_dir = Path(exp_root) / "results"
        for subdir in ["logs", "reports", "scripts", "results"]:
            assert (Path(exp_root) / subdir).exists()
        assert (Path(exp_root) / "dev_notes.md").is_file()

        # A1: Pipeline state — save after dir creation and verify round-trip
        save_state(phase=2, iteration=0, running_exp_ids=[], exp_root=exp_root)
        state = load_state(exp_root)
        assert state is not None
        assert state["phase"] == 2
        assert state["iteration"] == 0

        # Phase 2: baseline training
        baseline_output = tmp_path / "baseline_output"
        rc, stdout, stderr = run_training(
            project_dir, baseline_output, shared_data_dir,
            extra_args=["--lr", "0.01", "--seed", "42"],
        )
        assert rc == 0, f"Baseline training failed: {stderr}"
        assert (baseline_output / "model.pth").exists()

        # Parse baseline logs
        baseline_log = Path(exp_root) / "logs" / "baseline.log"
        baseline_log.parent.mkdir(parents=True, exist_ok=True)
        baseline_log.write_text(stdout)
        records = parse_log(str(baseline_log))
        assert len(records) >= 2
        losses = extract_metric_trajectory(records, "loss")
        accs = extract_metric_trajectory(records, "accuracy")

        # Divergence check on baseline
        assert check_divergence(losses)["diverged"] is False

        # Save baseline results
        baseline_data = {
            "exp_id": "baseline",
            "config": {"lr": 0.01, "batch_size": 64},
            "metrics": {"loss": losses[-1], "accuracy": accs[-1]},
            "status": "completed",
        }
        (results_dir / "baseline.json").write_text(json.dumps(baseline_data))

        # A2: Schema validation on baseline
        assert validate_baseline(baseline_data)["valid"]
        assert validate_file(str(results_dir / "baseline.json"), "baseline")["valid"]

        # A1: Validate phase 5 prerequisites before experiment loop
        phase5_check = validate_phase_requirements(5, exp_root)
        assert phase5_check["valid"], f"Phase 5 prereqs failed: {phase5_check['missing']}"

        # Phase 5: two experiments
        experiments = [
            {"lr": "0.001", "config": {"lr": 0.001}},
            {"lr": "0.05", "config": {"lr": 0.05}},
        ]
        for i, exp_params in enumerate(experiments):
            exp_info = setup(
                str(exp_project),
                f"python train.py --lr {exp_params['lr']}",
                config=exp_params["config"],
            )
            exp_output = tmp_path / f"exp_{i}_output"
            rc, stdout, stderr = run_training(
                project_dir, exp_output, shared_data_dir,
                extra_args=["--lr", exp_params["lr"], "--seed", "42"],
            )
            assert rc == 0, f"Experiment {exp_info['exp_id']} failed: {stderr}"

            # Parse and save
            exp_log = Path(exp_root) / "logs" / f"{exp_info['exp_id']}.log"
            exp_log.parent.mkdir(parents=True, exist_ok=True)
            exp_log.write_text(stdout)
            exp_records = parse_log(str(exp_log))
            exp_losses = extract_metric_trajectory(exp_records, "loss")
            exp_accs = extract_metric_trajectory(exp_records, "accuracy")

            # Divergence check
            assert check_divergence(exp_losses)["diverged"] is False

            # Update experiment result file
            exp_data = json.loads(Path(exp_info["config_path"]).read_text())
            exp_data["metrics"] = {"loss": exp_losses[-1], "accuracy": exp_accs[-1]}
            exp_data["status"] = "completed"
            Path(exp_info["config_path"]).write_text(json.dumps(exp_data, indent=2))

            # A2: Schema validation on each experiment result
            assert validate_result(exp_data)["valid"]

        # A1: Save state with completed experiments, verify cleanup finds nothing stale
        save_state(phase=5, iteration=2, running_exp_ids=[], exp_root=exp_root)
        cleaned = cleanup_stale(exp_root, timeout_hours=2.0)
        assert len(cleaned) == 0

        # Phase 6: analysis
        analysis = analyze(str(results_dir), "loss", baseline_id="baseline")
        assert analysis["num_experiments"] >= 3
        assert len(analysis["ranking"]) >= 3
        assert len(analysis["deltas"]) >= 2

        # Verify directory structure matches SKILL.md spec
        assert (Path(exp_root) / "logs").exists()
        assert (Path(exp_root) / "results").exists()
        assert (Path(exp_root) / "scripts").exists()
        assert (Path(exp_root) / "reports").exists()
        assert (Path(exp_root) / "dev_notes.md").is_file()

        # Verify result files exist and validate schemas
        result_files = list(results_dir.glob("*.json"))
        assert len(result_files) >= 3  # baseline + 2 experiments
        for rf in result_files:
            schema = "baseline" if rf.stem == "baseline" else "result"
            vr = validate_file(str(rf), schema)
            assert vr["valid"], f"{rf.name} failed validation: {vr['errors']}"


# ---------------------------------------------------------------------------
# B1: Pipeline State Integration (no PyTorch needed)
# ---------------------------------------------------------------------------

class TestPipelineStateIntegration:
    """Test state persistence, resumption, phase validation, and cleanup."""

    def test_save_load_roundtrip(self, tmp_path):
        exp_root = str(tmp_path / "exp")
        save_state(phase=2, iteration=0, running_exp_ids=["exp-001"], exp_root=exp_root)
        state = load_state(exp_root)
        assert state is not None
        assert state["phase"] == 2
        assert state["iteration"] == 0
        assert state["running_experiments"] == ["exp-001"]
        assert state["status"] == "running"
        assert "timestamp" in state

    def test_load_state_missing(self, tmp_path):
        assert load_state(str(tmp_path / "nonexistent")) is None

    def test_cleanup_stale_marks_old_pipeline_interrupted(self, tmp_path):
        exp_root = str(tmp_path / "exp")
        save_state(phase=5, iteration=1, running_exp_ids=["exp-001"], exp_root=exp_root)
        # Patch the timestamp to be 3 hours ago so cleanup triggers
        state_path = tmp_path / "exp" / "pipeline-state.json"
        state = json.loads(state_path.read_text())
        from datetime import timedelta
        old_time = datetime.now(timezone.utc) - timedelta(hours=3)
        state["timestamp"] = old_time.isoformat()
        state_path.write_text(json.dumps(state))

        cleaned = cleanup_stale(exp_root, timeout_hours=2.0)
        assert any("interrupted" in c for c in cleaned)
        reloaded = load_state(exp_root)
        assert reloaded["status"] == "interrupted"

    def test_cleanup_stale_ignores_fresh_state(self, tmp_path):
        exp_root = str(tmp_path / "exp")
        save_state(phase=5, iteration=1, running_exp_ids=[], exp_root=exp_root)
        cleaned = cleanup_stale(exp_root, timeout_hours=2.0)
        assert len(cleaned) == 0
        assert load_state(exp_root)["status"] == "running"

    def test_cleanup_stale_marks_old_experiments_failed(self, tmp_path):
        exp_root = tmp_path / "exp"
        results_dir = exp_root / "results"
        results_dir.mkdir(parents=True)
        from datetime import timedelta
        old_time = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        (results_dir / "exp-001.json").write_text(json.dumps({
            "exp_id": "exp-001", "status": "running", "timestamp": old_time,
            "config": {}, "metrics": {},
        }))
        cleaned = cleanup_stale(str(exp_root), timeout_hours=2.0)
        assert any("exp-001" in c for c in cleaned)
        data = json.loads((results_dir / "exp-001.json").read_text())
        assert data["status"] == "failed"

    def test_validate_phase_sequence(self, tmp_path):
        exp_root = str(tmp_path / "exp")
        # Phase 2 (prerequisites) — always valid, no file requirements
        result = validate_phase_requirements(2, exp_root)
        assert result["valid"]

        # Phase 3 (baseline) needs results/ dir
        result = validate_phase_requirements(3, exp_root)
        assert not result["valid"]
        # Create results dir
        (tmp_path / "exp" / "results").mkdir(parents=True)
        assert validate_phase_requirements(3, exp_root)["valid"]

        # Phase 4 (checkpoint) needs baseline.json with metrics + config
        assert not validate_phase_requirements(4, exp_root)["valid"]
        baseline = {"exp_id": "baseline", "config": {"lr": 0.01}, "metrics": {"loss": 1.5}, "status": "completed"}
        (tmp_path / "exp" / "results" / "baseline.json").write_text(json.dumps(baseline))
        assert validate_phase_requirements(4, exp_root)["valid"]

        # Phase 5 (research) needs baseline.json
        assert validate_phase_requirements(5, exp_root)["valid"]

        # Phase 6 (experiment loop) needs baseline.json with metrics+config
        assert validate_phase_requirements(6, exp_root)["valid"]

    def test_invalid_phase6_without_baseline(self, tmp_path):
        exp_root = str(tmp_path / "exp")
        (tmp_path / "exp" / "results").mkdir(parents=True)
        result = validate_phase_requirements(6, exp_root)
        assert not result["valid"]
        assert any("baseline" in m for m in result["missing"])


# ---------------------------------------------------------------------------
# B2: Implement Workflow Integration (no PyTorch needed)
# ---------------------------------------------------------------------------

class TestImplementWorkflowIntegration:
    """Test research proposal parsing, conflict detection, syntax validation, and manifest writing."""

    MOCK_FINDINGS = """\
# Research Findings

## Summary
Two optimization proposals identified.

### Proposal 1: Cosine Annealing LR Schedule (Priority: High)

**Complexity:** Low
**Implementation strategy:** from_scratch

**What to change:**
- `train.py` — add cosine annealing scheduler
- `config.yaml` — add scheduler params

**Implementation steps:**
1. Import CosineAnnealingLR from torch.optim.lr_scheduler
2. Create scheduler after optimizer initialization
3. Call scheduler.step() after each epoch

### Proposal 2: Perceptual Loss Function (Priority: Medium)

**Complexity:** Medium
**Implementation strategy:** from_reference
**Reference repo:** https://github.com/example/perceptual-loss
**Reference files:** `losses/perceptual.py`, `models/vgg_features.py`

**What to change:**
- `train.py` — swap loss function
- `model.py` — add feature extractor

**Implementation steps:**
1. Clone reference repo
2. Adapt perceptual loss module
3. Integrate into training loop
"""

    def test_parse_proposals(self, tmp_path):
        findings = tmp_path / "research-findings.md"
        findings.write_text(self.MOCK_FINDINGS)
        proposals = parse_research_proposals(str(findings))
        assert len(proposals) == 2
        assert proposals[0]["name"] == "Cosine Annealing LR Schedule"
        assert proposals[0]["slug"] == "cosine-annealing-lr-schedule"
        assert proposals[0]["implementation_strategy"] == "from_scratch"
        assert proposals[1]["name"] == "Perceptual Loss Function"
        assert proposals[1]["implementation_strategy"] == "from_reference"
        assert proposals[1]["reference_repo"] == "https://github.com/example/perceptual-loss"

    def test_parse_proposals_selected(self, tmp_path):
        findings = tmp_path / "research-findings.md"
        findings.write_text(self.MOCK_FINDINGS)
        proposals = parse_research_proposals(str(findings), selected_indices=[2])
        assert len(proposals) == 1
        assert proposals[0]["index"] == 2

    def test_detect_conflicts_overlapping(self, tmp_path):
        findings = tmp_path / "research-findings.md"
        findings.write_text(self.MOCK_FINDINGS)
        proposals = parse_research_proposals(str(findings))
        # Both proposals modify train.py
        conflicts = detect_conflicts(proposals)
        assert len(conflicts) >= 1
        conflict_files = [c["file"] for c in conflicts]
        assert "train.py" in conflict_files

    def test_validate_syntax_on_fixtures(self):
        fixture_files = [
            str(RESNET_FIXTURE / "model.py"),
            str(RESNET_FIXTURE / "train.py"),
            str(RESNET_FIXTURE / "eval.py"),
        ]
        results = validate_syntax(fixture_files)
        assert len(results) == 3
        for r in results:
            assert r["passed"], f"Syntax validation failed for {r['file']}: {r['error']}"

    def test_validate_syntax_bad_file(self, tmp_path):
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(\n")
        results = validate_syntax([str(bad_file)])
        assert len(results) == 1
        assert not results[0]["passed"]
        assert results[0]["error"] is not None

    def test_write_and_validate_manifest(self, tmp_path):
        manifest_data = {
            "original_branch": "main",
            "strategy": "git_branch",
            "proposals": [
                {"name": "Cosine LR", "slug": "cosine-lr", "status": "validated",
                 "branch": "ml-opt/cosine-lr", "files_modified": ["train.py"]},
                {"name": "Perceptual Loss", "slug": "perceptual-loss", "status": "validated",
                 "branch": "ml-opt/perceptual-loss", "files_modified": ["train.py", "model.py"],
                 "implementation_strategy": "from_reference"},
            ],
        }
        manifest_path = str(tmp_path / "results" / "implementation-manifest.json")
        write_manifest(manifest_path, manifest_data)
        assert Path(manifest_path).exists()

        validation = validate_manifest(manifest_data)
        assert validation["valid"], f"Manifest invalid: {validation['errors']}"

        file_validation = validate_file(manifest_path, "manifest")
        assert file_validation["valid"], f"File validation failed: {file_validation['errors']}"


# ---------------------------------------------------------------------------
# B3: Non-Git Fallback (no PyTorch needed)
# ---------------------------------------------------------------------------

class TestNonGitFallback:
    """Test file backup strategy for non-git projects."""

    def test_is_git_repo_false(self, tmp_path):
        project = tmp_path / "non_git_project"
        project.mkdir()
        assert not is_git_repo(str(project))

    def test_is_git_repo_true(self, tmp_path):
        project = tmp_path / "git_project"
        project.mkdir()
        (project / ".git").mkdir()
        assert is_git_repo(str(project))

    def test_backup_files_creates_copies(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        (project / "train.py").write_text("print('train')")
        (project / "model.py").write_text("print('model')")
        subdir = project / "utils"
        subdir.mkdir()
        (subdir / "helpers.py").write_text("print('helpers')")

        files = [
            str(project / "train.py"),
            str(project / "model.py"),
            str(subdir / "helpers.py"),
        ]
        backup_dir = str(tmp_path / "backups")
        mapping = backup_files(files, backup_dir, project_root=str(project))

        assert len(mapping) == 3
        for original, backup in mapping.items():
            assert Path(backup).exists(), f"Backup missing: {backup}"
            assert Path(original).exists(), f"Original deleted: {original}"
            assert Path(backup).read_text() == Path(original).read_text()

        # Verify directory structure is preserved
        assert (Path(backup_dir) / "utils" / "helpers.py").exists()


# ---------------------------------------------------------------------------
# B4: Plot Integration (no PyTorch needed)
# ---------------------------------------------------------------------------

class TestPlotIntegration:
    """Test ASCII visualization from realistic mock data."""

    @pytest.fixture
    def results_dir(self, tmp_path):
        d = tmp_path / "results"
        d.mkdir()
        for name, loss, acc, lr, bs in [
            ("baseline", 2.0, 30.0, 0.01, 64),
            ("exp-001", 1.5, 45.0, 0.001, 64),
            ("exp-002", 1.8, 38.0, 0.1, 64),
            ("exp-003", 1.3, 50.0, 0.005, 128),
            ("exp-004", 1.6, 42.0, 0.01, 32),
        ]:
            _write_result(d, name, "completed", {"lr": lr, "batch_size": bs}, {"loss": loss, "accuracy": acc})
        return str(d)

    def test_metric_comparison_chart(self, results_dir):
        chart = plot_metric_comparison(results_dir, "loss")
        assert chart
        assert "[B]" in chart  # baseline marker
        assert "exp-001" in chart

    def test_improvement_timeline(self, results_dir):
        chart = plot_improvement_timeline(results_dir, "loss")
        assert chart
        assert "Best-so-far" in chart

    def test_hp_sensitivity_existing_hp(self, results_dir):
        chart = plot_hp_sensitivity(results_dir, "loss", "lr")
        assert chart
        assert "Sensitivity" in chart

    def test_hp_sensitivity_nonexistent_hp(self, results_dir):
        result = plot_hp_sensitivity(results_dir, "loss", "nonexistent_hp")
        assert "No numeric data" in result


# ---------------------------------------------------------------------------
# B5: Schema Validation Integration (no PyTorch needed)
# ---------------------------------------------------------------------------

class TestSchemaValidationIntegration:
    """Test cross-schema validation of pipeline data files."""

    def test_validate_baseline_file(self, tmp_path):
        data = {
            "exp_id": "baseline", "status": "completed",
            "config": {"lr": 0.01}, "metrics": {"loss": 1.5},
        }
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(data))
        result = validate_file(str(path), "baseline")
        assert result["valid"], f"Errors: {result['errors']}"

    def test_validate_result_file(self, tmp_path):
        data = {
            "exp_id": "exp-001", "status": "completed",
            "config": {"lr": 0.001}, "metrics": {"loss": 1.2, "accuracy": 55.0},
        }
        path = tmp_path / "exp-001.json"
        path.write_text(json.dumps(data))
        result = validate_file(str(path), "result")
        assert result["valid"], f"Errors: {result['errors']}"

    def test_validate_manifest_file(self, tmp_path):
        data = {
            "original_branch": "main",
            "strategy": "git_branch",
            "proposals": [
                {"name": "Test", "slug": "test", "status": "validated"},
            ],
        }
        path = tmp_path / "implementation-manifest.json"
        path.write_text(json.dumps(data))
        result = validate_file(str(path), "manifest")
        assert result["valid"], f"Errors: {result['errors']}"

    def test_corrupted_baseline_missing_key(self, tmp_path):
        data = {"exp_id": "baseline", "status": "completed", "config": {"lr": 0.01}}
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(data))
        result = validate_file(str(path), "baseline")
        assert not result["valid"]
        assert any("metrics" in e for e in result["errors"])

    def test_corrupted_result_bad_status(self, tmp_path):
        data = {
            "exp_id": "exp-001", "status": "INVALID",
            "config": {}, "metrics": {"loss": 1.0},
        }
        path = tmp_path / "exp-001.json"
        path.write_text(json.dumps(data))
        result = validate_file(str(path), "result")
        assert not result["valid"]
        assert any("status" in e.lower() for e in result["errors"])

    def test_validate_nonexistent_file(self, tmp_path):
        result = validate_file(str(tmp_path / "missing.json"), "result")
        assert not result["valid"]
        assert any("not found" in e.lower() for e in result["errors"])


# ---------------------------------------------------------------------------
# C1: Fixture-based Divergence Detection (no PyTorch needed)
# ---------------------------------------------------------------------------

class TestDivergenceFromFixture:
    """Divergence detection using the pre-built fixture log."""

    def test_divergent_log_fixture(self):
        log_path = str(FIXTURES / "divergent_log.txt")
        records = parse_log(log_path)
        assert len(records) >= 10
        losses = extract_metric_trajectory(records, "loss")
        assert len(losses) >= 10
        div_result = check_divergence(losses)
        assert div_result["diverged"] is True
        assert "NaN" in div_result["reason"]


# ---------------------------------------------------------------------------
# C2: Experiment ID Sequencing (no PyTorch needed)
# ---------------------------------------------------------------------------

class TestExperimentIdSequencing:
    """Verify sequential experiment ID generation and config file creation."""

    def test_sequential_ids(self, tmp_path):
        project = str(tmp_path / "project")
        expected_ids = ["exp-001", "exp-002", "exp-003", "exp-004", "exp-005"]
        actual_ids = []
        config_paths = []
        for i in range(5):
            result = setup(project, f"python train.py --lr 0.0{i+1}", config={"lr": 0.01 * (i + 1)})
            actual_ids.append(result["exp_id"])
            config_paths.append(result["config_path"])
        assert actual_ids == expected_ids

        # Verify each config file exists and has correct exp_id
        for exp_id, config_path in zip(expected_ids, config_paths):
            assert Path(config_path).exists()
            data = json.loads(Path(config_path).read_text())
            assert data["exp_id"] == exp_id

    def test_next_id_respects_existing(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "exp-003.json").write_text("{}")
        assert next_experiment_id(str(results_dir)) == "exp-004"
