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
    values: list[float],
    window: int = 10,
    threshold: float = 5.0,
    lower_is_better: bool = True,
) -> dict | None:
    """Detect metric explosion (or crash when lower_is_better=False)."""
    if len(values) < window + 1:
        return None
    for i in range(window, len(values)):
        window_vals = [v for v in values[i - window : i] if math.isfinite(v)]
        if not window_vals:
            continue
        avg = sum(window_vals) / len(window_vals)
        if avg == 0:
            continue
        if not math.isfinite(values[i]):
            continue
        if lower_is_better:
            if values[i] > threshold * avg:
                return {
                    "diverged": True,
                    "reason": f"Metric explosion: {values[i]:.4f} > {threshold}x rolling avg {avg:.4f}",
                    "step": i,
                }
        else:
            if values[i] < avg / threshold:
                return {
                    "diverged": True,
                    "reason": f"Metric crash: {values[i]:.4f} < rolling avg {avg:.4f} / {threshold}",
                    "step": i,
                }
    return None


def detect_plateau(
    values: list[float],
    patience: int = 20,
    min_delta: float = 1e-6,
    lower_is_better: bool = True,
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
        if lower_is_better:
            improved = values[i] < best - min_delta
        else:
            improved = values[i] > best + min_delta
        if improved:
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


def detect_gradual_drift(
    values: list[float],
    window: int = 50,
    min_slope_ratio: float = 0.1,
    lower_is_better: bool = True,
    min_r_squared: float = 0.1,
) -> dict | None:
    """Detect gradual metric drift via linear regression over a rolling window.

    Computes slope of the last *window* finite values using least-squares.
    Flags when the total drift exceeds *min_slope_ratio* times the first
    value's magnitude, in the wrong direction, AND the R² of the fit
    exceeds *min_r_squared* (to filter out noise-driven false positives).
    """
    finite = [(i, v) for i, v in enumerate(values) if math.isfinite(v)]
    if len(finite) < window:
        return None

    tail = finite[-window:]
    xs = [x for x, _ in tail]
    ys = [y for _, y in tail]

    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sx2 = sum(x * x for x in xs)

    denom = n * sx2 - sx * sx
    if denom == 0:
        return None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n

    # R² goodness-of-fit check: filter out noise-driven false positives
    y_mean = sy / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    if ss_tot > 0:
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        r_squared = 1 - ss_res / ss_tot
    else:
        r_squared = 0.0
    if r_squared < min_r_squared:
        return None

    # Total drift over the window
    total_drift = slope * (xs[-1] - xs[0])
    ref = abs(ys[0]) if ys[0] != 0 else 1.0
    if abs(total_drift) < min_slope_ratio * ref:
        return None

    drifting_wrong_way = (lower_is_better and slope > 0) or (not lower_is_better and slope < 0)
    if drifting_wrong_way:
        direction = "increasing" if slope > 0 else "decreasing"
        return {
            "diverged": True,
            "reason": f"Gradual drift: metric {direction} over {window} steps "
                      f"(slope={slope:.6f}, total_drift={total_drift:.4f})",
            "step": tail[-1][0],
        }
    return None


def check_divergence(
    values: list[float],
    explosion_window: int = 10,
    explosion_threshold: float = 5.0,
    plateau_patience: int = 20,
    plateau_min_delta: float = 1e-6,
    lower_is_better: bool = True,
    gradual_drift_window: int = 50,
    gradual_drift_min_slope: float = 0.1,
    gradual_drift_min_r_squared: float = 0.1,
) -> dict:
    """Run all divergence checks on a metric trajectory."""
    if not values:
        return {"diverged": False, "reason": "No data", "step": -1}

    # Check in order of severity
    result = detect_nan_inf(values)
    if result:
        return result

    result = detect_explosion(
        values, explosion_window, explosion_threshold, lower_is_better
    )
    if result:
        return result

    result = detect_gradual_drift(
        values, gradual_drift_window, gradual_drift_min_slope, lower_is_better,
        gradual_drift_min_r_squared,
    )
    if result:
        return result

    result = detect_plateau(
        values, plateau_patience, plateau_min_delta, lower_is_better
    )
    if result:
        return result

    return {"diverged": False, "reason": "Training healthy", "step": -1}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: detect_divergence.py <json-array-of-values> [--higher-is-better]")
        print('Example: detect_divergence.py "[0.5, 0.4, 0.3, 100.0]"')
        print('Example: detect_divergence.py "[50, 60, 70, 80]" --higher-is-better')
        sys.exit(1)
    higher_is_better = "--higher-is-better" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--higher-is-better"]
    try:
        values = json.loads(args[0])
    except json.JSONDecodeError:
        print(f"Error: invalid JSON '{args[0]}'")
        print('Usage: detect_divergence.py <json-array-of-values> [--higher-is-better]')
        sys.exit(1)
    print(json.dumps(check_divergence(values, lower_is_better=not higher_is_better), indent=2))
