"""Tests for experiment_setup.py."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from experiment_setup import (
    cleanup_stale_experiments,
    create_experiment_dirs,
    generate_train_script,
    next_experiment_id,
    setup,
    write_experiment_config,
)


def test_create_experiment_dirs(tmp_path):
    exp_root = create_experiment_dirs(str(tmp_path))
    assert Path(exp_root).exists()
    for subdir in ["logs", "reports", "scripts", "results"]:
        assert (Path(exp_root) / subdir).exists()
    dev_notes = Path(exp_root) / "dev_notes.md"
    assert dev_notes.exists()
    assert dev_notes.is_file()
    assert "# Dev Notes" in dev_notes.read_text()


def test_create_experiment_dirs_idempotent(tmp_path):
    create_experiment_dirs(str(tmp_path))
    create_experiment_dirs(str(tmp_path))  # Should not raise


def test_next_experiment_id_empty(tmp_path):
    assert next_experiment_id(str(tmp_path)) == "exp-001"


def test_next_experiment_id_existing(tmp_path):
    (tmp_path / "exp-001.json").write_text("{}")
    (tmp_path / "exp-002.json").write_text("{}")
    assert next_experiment_id(str(tmp_path)) == "exp-003"


def test_next_experiment_id_nonexistent():
    assert next_experiment_id("/nonexistent/dir") == "exp-001"


def test_write_experiment_config(tmp_path):
    path = write_experiment_config(str(tmp_path), "exp-001", {"lr": 0.001})
    data = json.loads(Path(path).read_text())
    assert data["lr"] == 0.001


def test_generate_train_script(tmp_path):
    script_path = generate_train_script(
        str(tmp_path),
        "exp-001",
        "python train.py --lr 0.001",
        gpu_id=2,
        log_file="logs/exp-001/train.log",
        env_vars={"WANDB_DISABLED": "true"},
    )
    content = Path(script_path).read_text()
    assert "CUDA_VISIBLE_DEVICES=2" in content
    assert "python train.py --lr 0.001" in content
    assert "WANDB_DISABLED=true" in content
    assert "tee logs/exp-001/train.log" in content
    # Check executable
    import stat
    assert Path(script_path).stat().st_mode & stat.S_IXUSR


def test_generate_train_script_path_with_spaces(tmp_path):
    script_path = generate_train_script(
        str(tmp_path),
        "exp-001",
        "python train.py",
        gpu_id=0,
        log_file="logs/my experiment/train.log",
    )
    content = Path(script_path).read_text()
    assert "'logs/my experiment'" in content or "'logs/my experiment/train.log'" in content


def test_setup(tmp_path):
    result = setup(str(tmp_path), "python train.py", gpu_id=0, config={"lr": 0.001})
    assert result["exp_id"] == "exp-001"
    assert Path(result["config_path"]).exists()
    assert Path(result["script_path"]).exists()

    # Second setup should increment
    result2 = setup(str(tmp_path), "python train.py", gpu_id=1)
    assert result2["exp_id"] == "exp-002"


def test_generate_train_script_has_pid(tmp_path):
    """Generated training script must record the script PID."""
    script_path = generate_train_script(
        str(tmp_path),
        "exp-001",
        "python train.py",
        gpu_id=0,
        log_file="logs/exp-001/train.log",
    )
    content = Path(script_path).read_text()
    assert "pid" in content, "Script should contain PID tracking"
    # Should write $$ (the script's own PID) to a pid file
    assert "$$" in content


def test_next_experiment_id_ignores_non_exp_json(tmp_path):
    """Non-experiment JSON files must not affect the next experiment ID."""
    (tmp_path / "exp-001.json").write_text("{}")
    (tmp_path / "experiment-summary.json").write_text("{}")
    (tmp_path / "baseline.json").write_text("{}")
    assert next_experiment_id(str(tmp_path)) == "exp-002"


def test_cleanup_stale_experiments(tmp_path):
    """Stale running experiments should be marked as failed."""
    # Create a stale "running" experiment (mtime set 3 hours ago)
    stale_file = tmp_path / "exp-001.json"
    stale_file.write_text(json.dumps({"exp_id": "exp-001", "status": "running"}))
    stale_mtime = time.time() - 3 * 3600  # 3 hours ago
    os.utime(str(stale_file), (stale_mtime, stale_mtime))

    # Create a fresh "running" experiment (mtime = now)
    fresh_file = tmp_path / "exp-002.json"
    fresh_file.write_text(json.dumps({"exp_id": "exp-002", "status": "running"}))

    # Create a stale "completed" experiment (should NOT be touched)
    done_file = tmp_path / "exp-003.json"
    done_file.write_text(json.dumps({"exp_id": "exp-003", "status": "completed"}))
    os.utime(str(done_file), (stale_mtime, stale_mtime))

    cleaned = cleanup_stale_experiments(str(tmp_path), timeout_hours=2.0)

    assert cleaned == ["exp-001"]

    data = json.loads(stale_file.read_text())
    assert data["status"] == "failed"
    assert "stale experiment" in data["notes"]

    # Fresh running experiment should be untouched
    fresh_data = json.loads(fresh_file.read_text())
    assert fresh_data["status"] == "running"

    # Completed experiment should be untouched
    done_data = json.loads(done_file.read_text())
    assert done_data["status"] == "completed"
