"""Tests for detect_divergence.py."""

import json
import math
import random

import pytest

from detect_divergence import detect_nan_inf, detect_explosion, detect_plateau, detect_gradual_drift, check_divergence, get_thresholds_for_category, MODEL_CATEGORY_DEFAULTS


def test_detect_nan():
    values = [1.0, 0.9, 0.8, float("nan"), 0.6]
    result = detect_nan_inf(values)
    assert result is not None
    assert result["diverged"] is True
    assert result["reason"] == "NaN detected"
    assert result["step"] == 3


def test_detect_inf():
    values = [1.0, 0.9, float("inf"), 0.7]
    result = detect_nan_inf(values)
    assert result is not None
    assert result["diverged"] is True
    assert result["reason"] == "Inf detected"
    assert result["step"] == 2


def test_detect_nan_inf_clean():
    values = [1.0, 0.9, 0.8, 0.7]
    assert detect_nan_inf(values) is None


def test_detect_explosion():
    # Normal values then sudden spike
    values = [1.0] * 15 + [50.0]
    result = detect_explosion(values, window=10, threshold=5.0)
    assert result is not None
    assert result["diverged"] is True
    assert "explosion" in result["reason"].lower()


def test_detect_explosion_normal():
    values = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
    result = detect_explosion(values, window=5, threshold=5.0)
    assert result is None


def test_detect_explosion_too_short():
    values = [1.0, 0.9]
    assert detect_explosion(values, window=10) is None


def test_detect_plateau():
    # Values that stop improving
    values = [1.0, 0.9, 0.8] + [0.8] * 25
    result = detect_plateau(values, patience=20, min_delta=1e-6)
    assert result is not None
    assert result["diverged"] is True
    assert "plateau" in result["reason"].lower()


def test_detect_plateau_improving():
    values = [1.0 - i * 0.01 for i in range(50)]
    result = detect_plateau(values, patience=20)
    assert result is None


def test_detect_plateau_too_short():
    values = [1.0, 0.9, 0.9]
    assert detect_plateau(values, patience=20) is None


def test_check_divergence_healthy():
    values = [1.0 - i * 0.01 for i in range(50)]
    result = check_divergence(values)
    assert result["diverged"] is False
    assert result["reason"] == "Training healthy"


def test_check_divergence_nan():
    values = [1.0, 0.9, float("nan"), 0.7]
    result = check_divergence(values)
    assert result["diverged"] is True
    assert "NaN" in result["reason"]


def test_check_divergence_empty():
    result = check_divergence([])
    assert result["diverged"] is False
    assert result["reason"] == "No data"


def test_check_divergence_explosion():
    values = [1.0] * 15 + [100.0]
    result = check_divergence(values, explosion_window=10, explosion_threshold=5.0)
    assert result["diverged"] is True
    assert "explosion" in result["reason"].lower()


def test_detect_plateau_nan_first_value():
    # NaN first, then steadily improving — should NOT trigger plateau
    values = [float("nan")] + [1.0 - i * 0.01 for i in range(30)]
    result = detect_plateau(values, patience=20)
    assert result is None


def test_detect_plateau_all_nan():
    values = [float("nan")] * 25
    result = detect_plateau(values, patience=20)
    assert result is None


@pytest.mark.parametrize("values,patience,should_diverge", [
    ([50, 55, 60, 65, 70, 75, 80, 85, 90], 5, False),
    ([50, 55, 60, 65, 70, 75, 80] + [80] * 25, 20, True),
])
def test_detect_plateau_higher_is_better(values, patience, should_diverge):
    result = detect_plateau(values, patience=patience, lower_is_better=False)
    if should_diverge:
        assert result is not None and result["diverged"] is True
        assert "plateau" in result["reason"].lower()
    else:
        assert result is None


def test_detect_explosion_higher_is_better():
    # Accuracy going from 30 to 90 — should NOT trigger explosion
    values = [30 + i * 2 for i in range(15)] + [90]
    result = detect_explosion(values, window=10, threshold=5.0, lower_is_better=False)
    assert result is None


def test_detect_explosion_higher_is_better_crash():
    # Accuracy dropping from 80 to 5 — SHOULD trigger explosion (metric crash)
    values = [80] * 15 + [5]
    result = detect_explosion(values, window=10, threshold=5.0, lower_is_better=False)
    assert result is not None
    assert result["diverged"] is True
    assert "crash" in result["reason"].lower()


def test_check_divergence_higher_is_better_healthy():
    # Improving accuracy trajectory should be healthy
    values = [50 + i for i in range(50)]
    result = check_divergence(values, lower_is_better=False)
    assert result["diverged"] is False
    assert result["reason"] == "Training healthy"


def test_detect_explosion_non_finite_current():
    """Non-finite current value is skipped (not counted as explosion)."""
    values = [1.0] * 15 + [float("inf")]
    result = detect_explosion(values, window=10, threshold=5.0)
    # inf is skipped by the isfinite check on values[i], so no explosion
    assert result is None


def test_detect_explosion_all_nan_window():
    """All-NaN window is skipped (no division by zero)."""
    values = [float("nan")] * 15 + [1.0]
    result = detect_explosion(values, window=10, threshold=5.0)
    # Window is all NaN -> window_vals is empty -> continue
    assert result is None


# --- CLI tests ---


def test_detect_gradual_drift_increasing_loss():
    """Slowly increasing loss should be caught by drift detection."""
    values = [0.5 + i * 0.005 for i in range(60)]
    result = detect_gradual_drift(values, window=50, min_slope_ratio=0.1)
    assert result is not None
    assert result["diverged"] is True
    assert "drift" in result["reason"].lower()


def test_detect_gradual_drift_healthy():
    """Decreasing loss should not trigger drift."""
    values = [1.0 - i * 0.005 for i in range(60)]
    result = detect_gradual_drift(values, window=50, min_slope_ratio=0.1)
    assert result is None


def test_detect_gradual_drift_higher_is_better():
    """Gradually decreasing accuracy should be caught."""
    values = [80 - i * 0.3 for i in range(60)]
    result = detect_gradual_drift(values, window=50, min_slope_ratio=0.1, lower_is_better=False)
    assert result is not None
    assert result["diverged"] is True
    assert "drift" in result["reason"].lower()


def test_detect_gradual_drift_noisy_stable():
    """Noisy but flat data should NOT trigger drift (R² filter)."""
    rng = random.Random(42)
    values = [0.5 + rng.gauss(0, 0.1) for _ in range(100)]
    result = detect_gradual_drift(values, window=50, min_slope_ratio=0.1)
    assert result is None


def test_detect_gradual_drift_noisy_with_real_trend():
    """Upward trend plus noise should still be detected."""
    rng = random.Random(42)
    values = [0.5 + i * 0.005 + rng.gauss(0, 0.02) for i in range(60)]
    result = detect_gradual_drift(values, window=50, min_slope_ratio=0.1)
    assert result is not None
    assert result["diverged"] is True
    assert "drift" in result["reason"].lower()


def test_detect_gradual_drift_oscillating():
    """Sine wave around stable mean should NOT trigger drift."""
    values = [0.5 + 0.1 * math.sin(i * 0.3) for i in range(100)]
    result = detect_gradual_drift(values, window=50, min_slope_ratio=0.1)
    assert result is None


def test_detect_gradual_drift_all_identical_values():
    """All-identical metric values should NOT trigger drift (ss_tot=0)."""
    values = [0.5] * 60
    result = detect_gradual_drift(values, window=50, min_slope_ratio=0.1)
    assert result is None


def test_check_divergence_all_identical_values():
    """All-identical values: no NaN, no explosion, no drift, but plateau triggers."""
    values = [0.5] * 25
    result = check_divergence(values, plateau_patience=20)
    assert result["diverged"] is True
    assert "plateau" in result["reason"].lower()


def test_detect_gradual_drift_too_short():
    """Not enough data for drift window returns None."""
    values = [0.5 + i * 0.01 for i in range(10)]
    assert detect_gradual_drift(values, window=50) is None


def test_check_divergence_gradual_drift():
    """check_divergence integrates gradual drift detection."""
    values = [0.5 + i * 0.005 for i in range(60)]
    result = check_divergence(values, gradual_drift_window=50)
    assert result["diverged"] is True
    assert "drift" in result["reason"].lower()


def test_detect_explosion_zero_average():
    """Zero-average window should NOT trigger false explosion."""
    values = [0.0] * 15 + [1.0]
    result = detect_explosion(values, window=10, threshold=5.0)
    # avg == 0 → guard triggers → no explosion
    assert result is None


def test_check_divergence_drift_over_plateau():
    """Gradual drift is detected before plateau (higher priority)."""
    # Steadily increasing loss over 60 steps with no plateau pattern
    values = [1.0 + i * 0.05 for i in range(60)]
    # The last 20 values also form a plateau-like flatness at coarse scale,
    # but the drift check runs first (lines 181-186 before 188-192)
    result = check_divergence(
        values,
        gradual_drift_window=50,
        gradual_drift_min_slope=0.01,
        gradual_drift_min_r_squared=0.05,
        plateau_patience=20,
    )
    assert result["diverged"] is True
    assert "drift" in result["reason"].lower()


def test_mixed_plateau_then_explosion():
    """Plateau followed by explosion: explosion detected first (higher priority)."""
    values = [1.0] * 30 + [50.0]
    result = check_divergence(values, explosion_window=10, explosion_threshold=5.0, plateau_patience=20)
    assert result["diverged"] is True
    assert "explosion" in result["reason"].lower()


def test_cli_basic(run_main):
    """CLI with JSON array of values."""
    r = run_main("detect_divergence.py", "[0.5, 0.4, 0.3, 0.2]")
    assert r.returncode == 0
    output = json.loads(r.stdout)
    assert output["diverged"] is False


def test_cli_higher_is_better(run_main):
    """CLI with --higher-is-better flag."""
    r = run_main("detect_divergence.py", "[50, 60, 70, 80]", "--higher-is-better")
    assert r.returncode == 0
    output = json.loads(r.stdout)
    assert output["diverged"] is False


def test_cli_invalid_json(run_main):
    """CLI with invalid JSON exits cleanly."""
    r = run_main("detect_divergence.py", "not-json")
    assert r.returncode == 1
    assert "Error" in r.stdout


def test_check_divergence_nan_before_explosion(run_main):
    """NaN is detected before explosion even when both are present."""
    # NaN at step 5, explosion at step 15 — NaN should win (higher severity)
    values = [1.0] * 5 + [float("nan")] + [1.0] * 9 + [50.0]
    result = check_divergence(values, explosion_window=10, explosion_threshold=5.0)
    assert result["diverged"] is True
    assert "NaN" in result["reason"]
    assert result["step"] == 5


def test_detect_explosion_boundary_window():
    """Exactly window+1 values is the minimum to trigger explosion."""
    window = 10
    # Exactly window+1 values: 10 normal + 1 spike
    values = [1.0] * window + [50.0]
    result = detect_explosion(values, window=window, threshold=5.0)
    assert result is not None
    assert result["diverged"] is True
    # One fewer value should NOT trigger
    values_short = [1.0] * (window - 1) + [50.0]
    result_short = detect_explosion(values_short, window=window, threshold=5.0)
    assert result_short is None


def test_detect_plateau_boundary_patience():
    """Exactly patience steps of no improvement triggers plateau."""
    patience = 20
    # 1 improving value + patience flat values = patience+1 total
    values = [1.0] + [1.0] * patience
    result = detect_plateau(values, patience=patience, min_delta=1e-6)
    assert result is not None
    assert result["diverged"] is True
    # One fewer flat value should NOT trigger
    values_short = [1.0] + [1.0] * (patience - 1)
    result_short = detect_plateau(values_short, patience=patience, min_delta=1e-6)
    assert result_short is None


def test_cli_no_args(run_main):
    """CLI with no args prints usage and exits 1."""
    r = run_main("detect_divergence.py")
    assert r.returncode == 1
    assert "Usage" in r.stdout


def test_cli_flag_only_no_json(run_main):
    """CLI with --higher-is-better but no JSON array prints usage."""
    r = run_main("detect_divergence.py", "--higher-is-better")
    assert r.returncode == 1
    assert "Usage" in r.stdout


@pytest.mark.parametrize("values,should_explode", [
    ([1e-15] * 15 + [1e-10], False),
    ([0.01] * 15 + [1.0], True),
])
def test_detect_explosion_near_zero_threshold(values, should_explode):
    """Near-zero average: false positive guard vs real explosion."""
    result = detect_explosion(values, window=10, threshold=5.0)
    if should_explode:
        assert result is not None and result["diverged"] is True
    else:
        assert result is None


# --- Short sequence guard ---


def test_check_divergence_short_sequence_healthy():
    """Short finite sequence (<5 values) returns insufficient_data, not healthy."""
    values = [1.0, 0.9, 0.8]
    result = check_divergence(values)
    assert result["diverged"] is False
    assert "insufficient" in result["reason"].lower()


def test_check_divergence_short_sequence_with_nan():
    """Short sequence with NaN still catches it (NaN check runs first)."""
    values = [1.0, float("nan"), 0.8]
    result = check_divergence(values)
    assert result["diverged"] is True
    assert "nan" in result["reason"].lower()


def test_check_divergence_short_sequence_with_inf():
    """Short sequence with Inf still catches it."""
    values = [1.0, float("inf")]
    result = check_divergence(values)
    assert result["diverged"] is True
    assert "inf" in result["reason"].lower()


def test_check_divergence_four_values_insufficient():
    """Exactly 4 finite values is below the minimum threshold."""
    values = [1.0, 0.9, 0.8, 0.7]
    result = check_divergence(values)
    assert result["diverged"] is False
    assert "insufficient" in result["reason"].lower()


def test_check_divergence_five_values_runs_checks():
    """Exactly 5 finite values should run trend-based checks."""
    values = [0.5, 0.4, 0.3, 0.2, 0.1]
    result = check_divergence(values)
    assert result["diverged"] is False
    assert "insufficient" not in result["reason"].lower()


# --- Model category configurable thresholds ---


def test_get_thresholds_rl():
    """RL thresholds have higher explosion threshold and longer patience."""
    t = get_thresholds_for_category("rl")
    assert t["explosion_threshold"] == 20.0
    assert t["plateau_patience"] == 50


def test_get_thresholds_generative():
    """Generative thresholds have moderate explosion threshold and patience."""
    t = get_thresholds_for_category("generative")
    assert t["explosion_threshold"] == 10.0
    assert t["plateau_patience"] == 40


def test_get_thresholds_supervised():
    """Supervised (None) returns empty dict — uses function defaults."""
    t = get_thresholds_for_category(None)
    assert t == {}


def test_get_thresholds_unknown_category():
    """Unknown category returns empty dict (safe fallback)."""
    t = get_thresholds_for_category("unknown_model")
    assert t == {}


def test_rl_thresholds_prevent_false_positive():
    """RL reward spikes that trigger default thresholds don't trigger RL thresholds."""
    # Reward jumps from 1.0 to 8.0 — default explosion_threshold=5.0 would trigger
    values = [1.0] * 15 + [8.0]
    # Default: should detect explosion
    result_default = check_divergence(values, explosion_threshold=5.0, lower_is_better=False)
    # Note: for higher-is-better, explosion checks for crash (value < avg/threshold)
    # So this is actually NOT an explosion for higher-is-better (reward going up is good)
    # Let's test with a reward CRASH instead
    values_crash = [8.0] * 15 + [1.0]
    # Default threshold (5.0): 1.0 < 8.0/5.0 = 1.6 → crash detected
    result_crash_default = check_divergence(values_crash, explosion_threshold=5.0, lower_is_better=False)
    assert result_crash_default["diverged"] is True
    # RL threshold (20.0): 1.0 < 8.0/20.0 = 0.4 → 1.0 > 0.4, no crash
    rl_kwargs = get_thresholds_for_category("rl")
    result_crash_rl = check_divergence(values_crash, lower_is_better=False, **rl_kwargs)
    assert result_crash_rl["diverged"] is False


def test_generative_thresholds_longer_patience():
    """Generative models have longer plateau patience (40 vs default 20)."""
    # 25 flat values triggers default patience=20 but not generative patience=40
    values = [0.5] + [0.5] * 25
    result_default = check_divergence(values, plateau_patience=20)
    assert result_default["diverged"] is True
    gen_kwargs = get_thresholds_for_category("generative")
    result_gen = check_divergence(values, **gen_kwargs)
    assert result_gen["diverged"] is False


def test_model_category_defaults_immutable():
    """get_thresholds_for_category returns a copy, not the original dict."""
    t = get_thresholds_for_category("rl")
    t["explosion_threshold"] = 999
    assert MODEL_CATEGORY_DEFAULTS["rl"]["explosion_threshold"] == 20.0


def test_cli_model_category_rl(run_main):
    """CLI --model-category rl applies RL thresholds."""
    # Reward crash from 8 to 1 — RL threshold (20.0) should NOT trigger
    r = run_main("detect_divergence.py",
                  "[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,1]",
                  "--higher-is-better", "--model-category", "rl")
    assert r.returncode == 0
    output = json.loads(r.stdout)
    assert output["diverged"] is False


def test_cli_model_category_supervised(run_main):
    """CLI --model-category supervised uses default thresholds."""
    r = run_main("detect_divergence.py", "[0.5, 0.4, 0.3, 0.2]",
                  "--model-category", "supervised")
    assert r.returncode == 0
    output = json.loads(r.stdout)
    assert output["diverged"] is False


def test_cli_explicit_threshold_overrides_category(run_main):
    """CLI explicit --explosion-threshold overrides category default."""
    # With RL category (threshold=20) this wouldn't trigger, but explicit 3.0 should
    r = run_main("detect_divergence.py",
                  "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,5]",
                  "--model-category", "rl", "--explosion-threshold", "3.0")
    assert r.returncode == 0
    output = json.loads(r.stdout)
    assert output["diverged"] is True
    assert "explosion" in output["reason"].lower()


def test_cli_plateau_patience_flag(run_main):
    """CLI --plateau-patience flag works."""
    # 12 flat values: default patience=20 won't trigger, but patience=10 will
    values = [1.0] + [1.0] * 11
    r = run_main("detect_divergence.py", json.dumps(values),
                  "--plateau-patience", "10")
    assert r.returncode == 0
    output = json.loads(r.stdout)
    assert output["diverged"] is True
    assert "plateau" in output["reason"].lower()


class TestEmptyInputEdgeCases:
    """Edge case tests for empty inputs (Task 3.5)."""

    def test_detect_nan_inf_empty(self):
        result = detect_nan_inf([])
        assert result is None  # no NaN/Inf in empty list

    def test_detect_divergence_empty(self):
        result = check_divergence([])
        assert result is not None  # should handle empty gracefully
        assert result["diverged"] is False
