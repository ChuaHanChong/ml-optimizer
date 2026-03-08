"""Tests for error handling — verifies graceful handling of edge cases and failures."""

import json

from conftest import FIXTURES
from parse_logs import parse_log, parse_kv_line, extract_metric_trajectory
from detect_divergence import check_divergence, detect_nan_inf, detect_explosion, detect_plateau
from result_analyzer import load_results, analyze, compute_deltas
from experiment_setup import next_experiment_id
from gpu_check import parse_nvidia_smi, check_availability


# --- parse_logs error handling ---

def test_parse_partial_log():
    """Partial log (cut off mid-line) should parse without error."""
    records = parse_log(str(FIXTURES / "partial_log.txt"))
    assert len(records) >= 3  # At least the complete lines
    # Last partial line may or may not parse, but no exception


def test_parse_noisy_log():
    """Log with interleaved warnings should parse metrics correctly."""
    records = parse_log(str(FIXTURES / "noisy_train_log.txt"))
    losses = extract_metric_trajectory(records, "loss")
    assert len(losses) >= 8  # Should find metrics in all metric lines
    # Warnings should not produce metric entries
    assert all(isinstance(v, float) for v in losses)


def test_parse_oom_log():
    """Log that ends with OOM error should parse pre-crash metrics."""
    records = parse_log(str(FIXTURES / "oom_log.txt"))
    losses = extract_metric_trajectory(records, "loss")
    assert len(losses) >= 3  # Should have at least the 3 good steps


def test_parse_empty_file(tmp_path):
    """Empty file should return empty list without error."""
    empty_file = tmp_path / "empty.log"
    empty_file.write_text("")
    assert parse_log(str(empty_file)) == []


# --- detect_divergence error handling ---

def test_check_divergence_single_value():
    """Single value should not trigger any divergence."""
    result = check_divergence([0.5])
    assert result["diverged"] is False


def test_check_divergence_all_inf():
    """All Inf values should be detected."""
    result = check_divergence([float("inf")] * 5)
    assert result["diverged"] is True
    assert "Inf" in result["reason"]


def test_check_divergence_all_same():
    """Constant values should trigger plateau (if long enough)."""
    values = [1.0] * 30
    result = check_divergence(values, plateau_patience=20)
    assert result["diverged"] is True
    assert "plateau" in result["reason"].lower()


def test_detect_explosion_zero_avg():
    """Window with zero average should not divide by zero."""
    values = [0.0] * 15 + [1.0]
    result = detect_explosion(values, window=10, threshold=5.0)
    # avg=0 should be skipped, no crash
    assert result is None or result["diverged"] is True


def test_detect_plateau_mixed_nan_values():
    """Mix of NaN and real values should handle gracefully."""
    values = [1.0, float("nan"), 0.9, float("nan"), 0.8, float("nan")] + [0.8] * 25
    result = detect_plateau(values, patience=20)
    # Should either detect plateau or return None, but never crash


# --- result_analyzer error handling ---

def test_analyze_all_failed_experiments(tmp_path):
    """All experiments with failed/diverged status."""
    for i in range(3):
        (tmp_path / f"exp-{i:03d}.json").write_text(json.dumps({
            "exp_id": f"exp-{i:03d}",
            "status": "diverged",
            "config": {"lr": 0.001 * (i + 1)},
            "metrics": {},
        }))
    result = analyze(str(tmp_path), "loss")
    # Should not crash — ranking may be empty
    assert "num_experiments" in result


def test_load_results_corrupted_json(tmp_path):
    """Corrupted JSON files should be skipped."""
    (tmp_path / "good.json").write_text('{"metrics": {"loss": 0.5}}')
    (tmp_path / "bad.json").write_text('{"broken json')
    results = load_results(str(tmp_path))
    assert "good" in results
    assert "bad" not in results


def test_compute_deltas_zero_baseline():
    """Baseline with metric value of 0 should not crash."""
    results = {
        "baseline": {"metrics": {"loss": 0.0}},
        "exp-001": {"metrics": {"loss": 0.5}},
    }
    deltas = compute_deltas(results, "baseline", "loss")
    assert len(deltas) == 1
    # When baseline is zero, delta_pct is None (undefined percentage)
    assert deltas[0]["delta_pct"] is None


# --- experiment_setup error handling ---

def test_next_experiment_id_with_non_exp_json(tmp_path):
    """Non-experiment JSON files should not affect ID generation."""
    (tmp_path / "exp-001.json").write_text("{}")
    (tmp_path / "baseline.json").write_text("{}")
    (tmp_path / "implementation-manifest.json").write_text("{}")
    (tmp_path / "experiment-summary.json").write_text("{}")
    exp_id = next_experiment_id(str(tmp_path))
    assert exp_id == "exp-002"


# --- gpu_check error handling ---

def test_parse_nvidia_smi_completely_empty():
    """Completely empty nvidia-smi output."""
    assert parse_nvidia_smi("") == []


def test_check_availability_missing_fields():
    """GPUs with missing fields should default to unavailable."""
    gpus = [{"name": "GPU 0"}]  # No utilization or memory fields
    result = check_availability(gpus)
    assert result[0]["available"] is False  # Default utilization=100
