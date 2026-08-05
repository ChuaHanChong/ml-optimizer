#!/usr/bin/env python3
"""Detect training divergence from metric trajectories.

Runs NaN/Inf, explosion, reward-collapse, gradual-drift, and plateau checks
over a metric trajectory (a JSON array string), plus a separate train-vs-val
overfitting check. Per-model-category threshold defaults via --model-category.

Usage:
    python3 detect_divergence.py '<json_values>'                                 # All divergence checks
    python3 detect_divergence.py --check-overfitting '<train_json>' '<val_json>' # Train-vs-val overfitting
    python3 detect_divergence.py --scan-records '<json_records>'                 # NaN/Inf scan across ALL metrics

Add --higher-is-better for reward/accuracy-like metrics (default: lower is better).
Add --model-category rl|generative|supervised for category threshold defaults.
Override thresholds: --explosion-threshold N, --plateau-patience N,
--reward-collapse-fraction F, --reward-collapse-patience N.
Result status: "diverged" (kill: NaN/Inf, explosion/crash, reward collapse) vs
"plateaued" (warning: plateau/drift — report, NEVER kill) vs "healthy".
For overfitting: --patience N, --min-gap F. Value args are JSON strings, not paths.
"""

import json
import math
import sys


def detect_nan_inf(values: list[float]) -> dict | None:
    """Check for NaN or Inf values."""
    for i, v in enumerate(values):
        if math.isnan(v):
            return {"diverged": True, "status": "diverged", "reason": "NaN detected", "step": i}
        if math.isinf(v):
            return {"diverged": True, "status": "diverged", "reason": "Inf detected", "step": i}
    return None


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def is_bounded_rate(values: list[float], lower_is_better: bool) -> bool:
    """True for bounded [0,1] higher-is-better metrics (success rate) — exempt from crash detection."""
    if lower_is_better:
        return False
    finite = [v for v in values if math.isfinite(v)]
    return bool(finite) and all(0.0 <= v <= 1.0 for v in finite)


def detect_explosion(values: list[float], window: int = 10, threshold: float = 5.0,
                     lower_is_better: bool = True) -> dict | None:
    """Detect metric explosion (or crash when lower_is_better=False).

    Shift-invariant: primary test is a robust z-score of the current value vs the
    rolling-window median over its MAD (1.4826*MAD ~ 1 sigma), so negative-valued
    metrics work. Secondary legacy ratio test runs only when window avg > 0.
    Bounded [0,1] higher-is-better metrics are exempt from crash detection.
    """
    if len(values) < window + 1:
        return None
    if is_bounded_rate(values, lower_is_better):
        return None
    for i in range(window, len(values)):
        if not math.isfinite(values[i]):
            continue
        window_vals = [v for v in values[i - window : i] if math.isfinite(v)]
        if not window_vals:
            continue
        med = _median(window_vals)
        mad = _median([abs(v - med) for v in window_vals])
        # Robust z-score vs rolling dispersion, MAD-sigma floored by window std so a
        # tight 10-sample MAD can't fabricate a spurious z on ordinary noise.
        # ponytail: std floor for small windows; drop it if the window grows large.
        if mad > 0:
            mean = sum(window_vals) / len(window_vals)
            std = (sum((v - mean) ** 2 for v in window_vals) / len(window_vals)) ** 0.5
            z = (values[i] - med) / max(1.4826 * mad, std)
            adverse = z > threshold if lower_is_better else z < -threshold
            if adverse:
                direction = "explosion" if lower_is_better else "crash"
                return {
                    "diverged": True,
                    "status": "diverged",
                    "reason": f"Metric {direction}: {values[i]:.4f} deviates "
                              f"{abs(z):.1f} MADs from rolling median {med:.4f} "
                              f"(threshold {threshold})",
                    "step": i,
                }
        # Secondary ratio test — positive-scale metrics only (avg <= 0 skipped).
        avg = sum(window_vals) / len(window_vals)
        if avg <= 0:
            continue
        if lower_is_better:
            if values[i] > threshold * avg:
                return {
                    "diverged": True,
                    "status": "diverged",
                    "reason": f"Metric explosion: {values[i]:.4f} > {threshold}x rolling avg {avg:.4f}",
                    "step": i,
                }
        else:
            if values[i] < avg / threshold:
                return {
                    "diverged": True,
                    "status": "diverged",
                    "reason": f"Metric crash: {values[i]:.4f} < rolling avg {avg:.4f} / {threshold}",
                    "step": i,
                }
    return None


def detect_reward_collapse(values: list[float], fraction: float | None = None,
                           patience: int = 10, lower_is_better: bool = True) -> dict | None:
    """RL kill rule: sustained collapse below a fraction of the rolling max.

    Only for higher-is-better metrics when a fraction is configured. Fires when
    `patience` consecutive finite values sit below fraction * rolling_max. Skipped
    while rolling max <= 0 and for bounded [0,1] success-rate metrics.
    """
    if fraction is None or lower_is_better:
        return None
    if is_bounded_rate(values, lower_is_better):
        return None
    rolling_max = None
    below_count = 0
    for i, v in enumerate(values):
        if not math.isfinite(v):
            continue
        if rolling_max is None or v > rolling_max:
            rolling_max = v
        if rolling_max <= 0:
            below_count = 0
            continue
        if v < fraction * rolling_max:
            below_count += 1
            if below_count >= patience:
                return {
                    "diverged": True,
                    "status": "diverged",
                    "reason": f"Reward collapse: {patience} consecutive values below "
                              f"{fraction} x rolling max {rolling_max:.4f}",
                    "step": i,
                }
        else:
            below_count = 0
    return None


def detect_plateau(values: list[float], patience: int = 20, min_delta: float = 1e-6,
                   lower_is_better: bool = True) -> dict | None:
    """Detect plateau: no improvement for patience steps (WARNING, not divergence).

    Result carries status="plateaued", diverged=False — reported, never kills.
    min_delta is scaled by trajectory magnitude (max(1, median |value|)) so
    large-magnitude metrics don't register microscopic noise as improvement.
    """
    if len(values) < patience + 1:
        return None
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return None
    scale = _median([abs(v) for v in finite])
    effective_min_delta = min_delta * max(1.0, scale)
    best = finite[0]
    no_improve_count = 0
    for i in range(1, len(values)):
        if not math.isfinite(values[i]):
            continue
        if lower_is_better:
            improved = values[i] < best - effective_min_delta
        else:
            improved = values[i] > best + effective_min_delta
        if improved:
            best = values[i]
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                return {
                    "diverged": False,
                    "status": "plateaued",
                    "reason": f"Plateau: no improvement for {patience} steps (best={best:.6f})",
                    "step": i,
                }
    return None


def detect_gradual_drift(values: list[float], window: int = 50, min_slope_ratio: float = 0.1,
                         lower_is_better: bool = True, min_r_squared: float = 0.1) -> dict | None:
    """Detect gradual metric drift via least-squares regression over a rolling window.

    Flags when total drift over the last *window* finite values exceeds
    *min_slope_ratio* * |first value|, in the wrong direction, AND R² of the fit
    exceeds *min_r_squared* (filters noise-driven false positives).
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
            "diverged": False,
            "status": "plateaued",
            "reason": f"Gradual drift: metric {direction} over {window} steps "
                      f"(slope={slope:.6f}, total_drift={total_drift:.4f})",
            "step": tail[-1][0],
        }
    return None


def scan_records_for_nan_inf(records: list[dict]) -> dict | None:
    """Scan ALL numeric metrics in parsed log records for NaN/Inf.

    Any non-finite value in any metric is immediate divergence (a NaN in an
    unwatched aux loss still poisons training). Returns detect_nan_inf shape plus
    the offending metric.
    """
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        for key, v in record.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if not math.isfinite(v):
                kind = "NaN" if math.isnan(v) else "Inf"
                return {
                    "diverged": True,
                    "status": "diverged",
                    "reason": f"{kind} detected in metric '{key}'",
                    "step": i,
                    "metric": key,
                }
    return None


def check_overfitting(train_values: list[float], val_values: list[float], patience: int = 5,
                      min_gap: float = 0.0, lower_is_better: bool = True) -> dict:
    """Detect overfitting: train metric improving while val worsens for `patience` steps.

    Returns a dict with `overfitting` (bool), `reason`, `step`, `severity`, trend info.
    """
    # Align to the shorter trajectory, filtering NaN/Inf from both
    pairs = []
    for i in range(min(len(train_values), len(val_values))):
        t, v = train_values[i], val_values[i]
        if math.isfinite(t) and math.isfinite(v):
            pairs.append((i, t, v))

    if len(pairs) < patience + 1:
        return {
            "overfitting": False,
            "reason": "Insufficient data for overfitting detection",
            "step": -1,
            "severity": None,
            "train_trend": None,
            "val_trend": None,
            "gap_at_detection": None,
        }

    consecutive = 0
    for j in range(1, len(pairs)):
        idx, t_curr, v_curr = pairs[j]
        _, t_prev, v_prev = pairs[j - 1]

        t_delta = t_prev - t_curr if lower_is_better else t_curr - t_prev
        v_delta = v_curr - v_prev if lower_is_better else v_prev - v_curr

        train_improving = t_delta > 0
        val_worsening = v_delta > 0 and abs(v_delta) >= min_gap

        if train_improving and val_worsening:
            consecutive += 1
            if consecutive >= patience:
                # Trends over the patience window; severity = val regression / train gain
                window_start = j - patience + 1
                t_trend = pairs[j][1] - pairs[window_start][1]
                v_trend = pairs[j][2] - pairs[window_start][2]
                gap = abs(v_trend)

                t_improvement = abs(t_trend)
                ratio = gap / t_improvement if t_improvement > 1e-12 else float("inf")
                if ratio < 2:
                    severity = "mild"
                elif ratio < 5:
                    severity = "moderate"
                else:
                    severity = "severe"

                return {
                    "overfitting": True,
                    "reason": (
                        f"Validation metric worsened for {patience} consecutive "
                        f"steps while training improved "
                        f"(train: {pairs[window_start][1]:.4f}\u2192{pairs[j][1]:.4f}, "
                        f"val: {pairs[window_start][2]:.4f}\u2192{pairs[j][2]:.4f})"
                    ),
                    "step": idx,
                    "severity": severity,
                    "train_trend": round(t_trend, 6),
                    "val_trend": round(v_trend, 6),
                    "gap_at_detection": round(gap, 6),
                }
        else:
            consecutive = 0

    return {
        "overfitting": False,
        "reason": "No sustained overfitting pattern detected",
        "step": -1,
        "severity": None,
        "train_trend": None,
        "val_trend": None,
        "gap_at_detection": None,
    }


MODEL_CATEGORY_DEFAULTS: dict[str | None, dict] = {
    "rl": {
        "explosion_threshold": 20.0,
        "plateau_patience": 50,
        "gradual_drift_min_slope": 0.3,
        "overfitting_patience": 10,
        "reward_collapse_fraction": 0.5,
        "reward_collapse_patience": 10,
    },
    "generative": {
        "explosion_threshold": 10.0,
        "plateau_patience": 40,
        "gradual_drift_min_slope": 0.2,
        "overfitting_patience": 8,
    },
    None: {},  # use function defaults for supervised learning
}


_DIVERGENCE_KEYS = {
    "explosion_threshold", "plateau_patience", "plateau_min_delta",
    "gradual_drift_min_slope", "gradual_drift_min_r_squared",
    "gradual_drift_window", "explosion_window",
    "reward_collapse_fraction", "reward_collapse_patience",
}


def get_thresholds_for_category(model_category: str | None) -> dict:
    """Divergence threshold overrides for a model category (filters out overfitting-only keys)."""
    all_defaults = MODEL_CATEGORY_DEFAULTS.get(model_category, {})
    return {k: v for k, v in all_defaults.items() if k in _DIVERGENCE_KEYS}


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
    reward_collapse_fraction: float | None = None,
    reward_collapse_patience: int = 10,
) -> dict:
    """Run all divergence checks; returns {"diverged","status","reason","step"}.

    status is "diverged" (kill: NaN/Inf, explosion/crash, reward collapse),
    "plateaued" (warning: plateau or gradual drift — NEVER kill), or "healthy".
    """
    if not values:
        return {"diverged": False, "status": "healthy", "reason": "No data", "step": -1}

    # NaN/Inf runs on any-length sequences; trend checks need >= MIN_SEQUENCE_LENGTH.
    MIN_SEQUENCE_LENGTH = 5
    finite_values = [v for v in values if math.isfinite(v)]
    if len(finite_values) < MIN_SEQUENCE_LENGTH:
        result = detect_nan_inf(values)
        if result:
            return result
        return {"diverged": False, "status": "healthy",
                "reason": "Insufficient data for trend analysis", "step": -1}

    # Kill-class checks first (most severe)
    result = detect_nan_inf(values)
    if result:
        return result

    result = detect_explosion(
        values, explosion_window, explosion_threshold, lower_is_better
    )
    if result:
        return result

    result = detect_reward_collapse(
        values, reward_collapse_fraction, reward_collapse_patience, lower_is_better
    )
    if result:
        return result

    # Warning-class checks (status="plateaued" — reported, never killed)
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

    return {"diverged": False, "status": "healthy", "reason": "Training healthy", "step": -1}


if __name__ == "__main__":
    # Overfitting check mode
    if len(sys.argv) >= 2 and sys.argv[1] == "--check-overfitting":
        if len(sys.argv) < 4:
            print("Usage: detect_divergence.py --check-overfitting '<train_json>' '<val_json>' [--higher-is-better] [--patience N] [--min-gap F] [--model-category rl|generative|supervised]")
            sys.exit(1)
        try:
            train_vals = json.loads(sys.argv[2])
            val_vals = json.loads(sys.argv[3])
        except json.JSONDecodeError as e:
            print(f"Error: invalid JSON: {e}")
            sys.exit(1)
        higher = "--higher-is-better" in sys.argv
        patience = 5
        min_gap = 0.0
        for i, arg in enumerate(sys.argv[4:], start=4):
            if arg == "--patience" and i + 1 < len(sys.argv):
                patience = int(sys.argv[i + 1])
            elif arg == "--min-gap" and i + 1 < len(sys.argv):
                min_gap = float(sys.argv[i + 1])
            elif arg == "--model-category" and i + 1 < len(sys.argv):
                cat = sys.argv[i + 1]
                cat_key = cat if cat != "supervised" else None
                all_defaults = MODEL_CATEGORY_DEFAULTS.get(cat_key, {})
                patience = all_defaults.get("overfitting_patience", patience)
        print(json.dumps(check_overfitting(train_vals, val_vals, patience=patience, min_gap=min_gap, lower_is_better=not higher), indent=2))
        sys.exit(0)

    # Multi-metric NaN/Inf scan mode
    if len(sys.argv) >= 2 and sys.argv[1] == "--scan-records":
        if len(sys.argv) < 3:
            print("Usage: detect_divergence.py --scan-records '<json_records>'")
            sys.exit(1)
        try:
            records = json.loads(sys.argv[2])
        except json.JSONDecodeError as e:
            print(f"Error: invalid JSON: {e}")
            sys.exit(1)
        result = scan_records_for_nan_inf(records)
        if result is None:
            result = {"diverged": False, "status": "healthy",
                      "reason": "No NaN/Inf in any metric", "step": -1}
        print(json.dumps(result, indent=2))
        sys.exit(0)

    if len(sys.argv) < 2:
        print("Usage: detect_divergence.py <json-array-of-values> [--higher-is-better] [--model-category rl|generative|supervised]")
        print("       [--explosion-threshold N] [--plateau-patience N] [--reward-collapse-fraction F] [--reward-collapse-patience N]")
        print("       detect_divergence.py --check-overfitting '<train_json>' '<val_json>' [flags]")
        print('Example: detect_divergence.py "[0.5, 0.4, 0.3, 100.0]"')
        print('Example: detect_divergence.py "[50, 60, 70, 80]" --higher-is-better --model-category rl')
        sys.exit(1)

    # Parse flags
    higher_is_better = "--higher-is-better" in sys.argv
    model_category = None
    extra_kwargs: dict = {}
    skip_next = False
    positional_args = []

    for i, arg in enumerate(sys.argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if arg == "--higher-is-better":
            continue
        elif arg == "--model-category" and i + 1 < len(sys.argv):
            cat = sys.argv[i + 1]
            model_category = cat if cat != "supervised" else None
            skip_next = True
        elif arg == "--explosion-threshold" and i + 1 < len(sys.argv):
            extra_kwargs["explosion_threshold"] = float(sys.argv[i + 1])
            skip_next = True
        elif arg == "--plateau-patience" and i + 1 < len(sys.argv):
            extra_kwargs["plateau_patience"] = int(sys.argv[i + 1])
            skip_next = True
        elif arg == "--reward-collapse-fraction" and i + 1 < len(sys.argv):
            extra_kwargs["reward_collapse_fraction"] = float(sys.argv[i + 1])
            skip_next = True
        elif arg == "--reward-collapse-patience" and i + 1 < len(sys.argv):
            extra_kwargs["reward_collapse_patience"] = int(sys.argv[i + 1])
            skip_next = True
        else:
            positional_args.append(arg)

    if not positional_args:
        print("Usage: detect_divergence.py <json-array-of-values> [--higher-is-better] [--model-category rl|generative|supervised]")
        print('Example: detect_divergence.py "[0.5, 0.4, 0.3, 100.0]"')
        sys.exit(1)
    try:
        values = json.loads(positional_args[0])
    except json.JSONDecodeError:
        print(f"Error: invalid JSON '{positional_args[0]}'")
        print('Usage: detect_divergence.py <json-array-of-values> [--higher-is-better]')
        sys.exit(1)

    # Apply model category defaults, then override with explicit flags
    kwargs = get_thresholds_for_category(model_category)
    kwargs.update(extra_kwargs)
    kwargs["lower_is_better"] = not higher_is_better

    print(json.dumps(check_divergence(values, **kwargs), indent=2))
