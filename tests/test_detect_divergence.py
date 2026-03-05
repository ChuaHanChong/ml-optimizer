"""Tests for detect_divergence.py."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from detect_divergence import detect_nan_inf, detect_explosion, detect_plateau, check_divergence


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


def test_detect_plateau_higher_is_better():
    # Accuracy steadily improving — should NOT trigger plateau
    values = [50, 55, 60, 65, 70, 75, 80, 85, 90]
    result = detect_plateau(values, patience=5, lower_is_better=False)
    assert result is None


def test_detect_plateau_higher_is_better_stalled():
    # Accuracy stuck at 80 for 25 steps — SHOULD trigger plateau
    values = [50, 55, 60, 65, 70, 75, 80] + [80] * 25
    result = detect_plateau(values, patience=20, lower_is_better=False)
    assert result is not None
    assert result["diverged"] is True
    assert "plateau" in result["reason"].lower()


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
