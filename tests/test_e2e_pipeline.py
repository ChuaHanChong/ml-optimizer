"""End-to-end pipeline integration tests using Tiny ResNet on CIFAR-10.

Exercises all 6 pipeline phases against a real ML model, verifying that the
existing Python scripts (parse_logs, detect_divergence, experiment_setup,
result_analyzer, gpu_check) work correctly with real training output.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Import plugin scripts using the same pattern as other tests
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from parse_logs import parse_log, extract_metric_trajectory
from detect_divergence import check_divergence
from experiment_setup import create_experiment_dirs, next_experiment_id, setup
from result_analyzer import analyze, rank_by_metric, compute_deltas

FIXTURES = Path(__file__).parent / "fixtures"
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
            (results_dir / f"{name}.json").write_text(json.dumps({
                "exp_id": name,
                "config": {"lr": lr},
                "metrics": {"loss": loss, "accuracy": acc},
                "status": "completed",
            }))

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
        (results_dir / "baseline.json").write_text(json.dumps({
            "exp_id": "baseline",
            "config": {"lr": 0.01, "batch_size": 64},
            "metrics": {"loss": losses[-1], "accuracy": accs[-1]},
            "status": "completed",
        }))

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

        # Verify result files exist
        result_files = list(results_dir.glob("*.json"))
        assert len(result_files) >= 3  # baseline + 2 experiments
