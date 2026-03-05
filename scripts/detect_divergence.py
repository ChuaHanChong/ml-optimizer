#!/usr/bin/env python3
"""Detect training divergence from metric trajectories."""

import json
import math
import sys


def detect_nan_inf(values: list[float]) -> dict | None:
    """Check for NaN or Inf values."""
    for i, v in enumerate(values):
        if math.isnan(v):
            return {"diverged": True, "reason": "NaN detected", "step": i}
        if math.isinf(v):
            return {"diverged": True, "reason": "Inf detected", "step": i}
    return None


def detect_explosion(
    values: list[float], window: int = 10, threshold: float = 5.0
) -> dict | None:
    """Detect loss explosion: value > threshold * rolling average."""
    if len(values) < window + 1:
        return None
    for i in range(window, len(values)):
        window_vals = [v for v in values[i - window : i] if math.isfinite(v)]
        if not window_vals:
            continue
        avg = sum(window_vals) / len(window_vals)
        if avg == 0:
            continue
        if math.isfinite(values[i]) and values[i] > threshold * avg:
            return {
                "diverged": True,
                "reason": f"Loss explosion: {values[i]:.4f} > {threshold}x rolling avg {avg:.4f}",
                "step": i,
            }
    return None


def detect_plateau(
    values: list[float], patience: int = 20, min_delta: float = 1e-6
) -> dict | None:
    """Detect plateau: no improvement for patience consecutive steps."""
    if len(values) < patience + 1:
        return None
    best = None
    for v in values:
        if math.isfinite(v):
            best = v
            break
    if best is None:
        return None
    no_improve_count = 0
    for i in range(1, len(values)):
        if not math.isfinite(values[i]):
            continue
        if values[i] < best - min_delta:
            best = values[i]
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                return {
                    "diverged": True,
                    "reason": f"Plateau: no improvement for {patience} steps (best={best:.6f})",
                    "step": i,
                }
    return None


def check_divergence(
    values: list[float],
    explosion_window: int = 10,
    explosion_threshold: float = 5.0,
    plateau_patience: int = 20,
    plateau_min_delta: float = 1e-6,
) -> dict:
    """Run all divergence checks on a metric trajectory."""
    if not values:
        return {"diverged": False, "reason": "No data", "step": -1}

    # Check in order of severity
    result = detect_nan_inf(values)
    if result:
        return result

    result = detect_explosion(values, explosion_window, explosion_threshold)
    if result:
        return result

    result = detect_plateau(values, plateau_patience, plateau_min_delta)
    if result:
        return result

    return {"diverged": False, "reason": "Training healthy", "step": -1}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: detect_divergence.py <json-array-of-values>")
        print('Example: detect_divergence.py "[0.5, 0.4, 0.3, 100.0]"')
        sys.exit(1)
    values = json.loads(sys.argv[1])
    print(json.dumps(check_divergence(values), indent=2))
