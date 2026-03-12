#!/usr/bin/env python3
"""Analyze and compare experiment results."""

import json
import math
import sys
from pathlib import Path


def load_results(results_dir: str) -> dict[str, dict]:
    """Load all experiment results from a directory."""
    path = Path(results_dir)
    results = {}
    if not path.exists():
        return results
    for f in sorted(path.glob("*.json")):
        if f.stem.lower() != "baseline" and not f.stem.startswith("exp-"):
            continue
        try:
            data = json.loads(f.read_text())
            results[f.stem] = data
        except (json.JSONDecodeError, OSError):
            continue
    return results


def rank_by_metric(results: dict[str, dict], metric: str, lower_is_better: bool = True) -> list[dict]:
    """Rank experiments by a specific metric."""
    ranked = []
    for exp_id, data in results.items():
        metrics = data.get("metrics", data)
        if metric in metrics:
            ranked.append({
                "exp_id": exp_id,
                "value": metrics[metric],
                "config": data.get("config", {}),
                "status": data.get("status"),
            })
    valid = [r for r in ranked if isinstance(r["value"], (int, float)) and math.isfinite(r["value"])]
    invalid = [r for r in ranked if not (isinstance(r["value"], (int, float)) and math.isfinite(r["value"]))]
    for r in invalid:
        r["note"] = "non-finite metric value excluded from ranking"
    valid.sort(key=lambda x: x["value"], reverse=not lower_is_better)
    return valid + invalid


def spearman_correlation(x: list, y: list) -> float:
    """Compute Spearman rank correlation coefficient between two lists.

    Uses the formula: rho = 1 - 6 * sum(d^2) / (n * (n^2 - 1))
    where d is the difference between ranks of corresponding values.
    Handles ties by assigning average ranks.
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    def _rank(values):
        """Assign ranks with average-rank tie-breaking."""
        n = len(values)
        indexed = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and values[indexed[j]] == values[indexed[j + 1]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1  # 1-based
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg_rank
            i = j + 1
        return ranks

    n = len(x)
    rx = _rank(x)
    ry = _rank(y)
    # Constant ranks have no variance — correlation is undefined; return 0.0
    if len(set(rx)) == 1 or len(set(ry)) == 1:
        return 0.0
    d_sq_sum = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1 - 6 * d_sq_sum / (n * (n ** 2 - 1))
    return rho


def compute_deltas(results: dict[str, dict], baseline_id: str, metric: str) -> list[dict]:
    """Compute metric deltas vs baseline for all experiments."""
    if baseline_id not in results:
        return []

    baseline_metrics = results[baseline_id].get("metrics", results[baseline_id])
    if metric not in baseline_metrics:
        return []

    baseline_val = baseline_metrics[metric]
    deltas = []
    for exp_id, data in results.items():
        if exp_id == baseline_id:
            continue
        metrics = data.get("metrics", data)
        if metric in metrics:
            val = metrics[metric]
            delta = val - baseline_val
            if abs(baseline_val) < 1e-8:
                delta_pct = None
            else:
                delta_pct = round(delta / abs(baseline_val) * 100, 2)
            deltas.append({
                "exp_id": exp_id,
                "value": val,
                "delta": delta,
                "delta_pct": delta_pct,
                "config": data.get("config", {}),
            })
    return deltas


def identify_correlations(results: dict[str, dict], metric: str, lower_is_better: bool = True) -> dict:
    """Identify which hyperparameters correlate with improvement."""
    # Collect (config_key, config_value, metric_value) triples
    # Filter to only completed experiments (or those without a status key for backward compat)
    entries = []
    for exp_id, data in results.items():
        status = data.get("status")
        if status is not None and status != "completed":
            continue
        metrics = data.get("metrics", data)
        config = data.get("config", {})
        if metric in metrics and config:
            entries.append({"metric_value": metrics[metric], "config": config})

    if len(entries) < 4:
        return {"correlations": [], "note": "Need at least 4 data points for meaningful correlations"}

    # Sort by metric (best first)
    entries.sort(key=lambda x: x["metric_value"], reverse=not lower_is_better)

    # For each HP, compare top half vs bottom half
    mid = len(entries) // 2
    top_half = entries[:mid] if mid > 0 else entries[:1]
    bottom_half = entries[mid:] if mid > 0 else entries[1:]

    correlations = []
    all_keys = set()
    for e in entries:
        all_keys.update(e["config"].keys())

    def _is_numeric(v):
        try:
            float(v)
            return True
        except (ValueError, TypeError):
            return False

    for key in sorted(all_keys):
        top_vals = [e["config"].get(key) for e in top_half if key in e["config"]]
        bottom_vals = [e["config"].get(key) for e in bottom_half if key in e["config"]]
        if not top_vals or not bottom_vals:
            continue

        # Filter to numeric-coercible values
        all_hp = [(e["config"][key], e["metric_value"]) for e in entries if key in e["config"]]
        numeric_pairs = []
        for hp_val, met_val in all_hp:
            try:
                numeric_pairs.append((float(hp_val), met_val))
            except (ValueError, TypeError):
                continue

        if len(numeric_pairs) >= max(2, len(all_hp) // 2):
            # Majority numeric: compute numeric correlation on the numeric subset
            hp_values = [p[0] for p in numeric_pairs]
            metric_values = [p[1] for p in numeric_pairs]
            numeric_top = [v for v in top_vals if _is_numeric(v)]
            numeric_bottom = [v for v in bottom_vals if _is_numeric(v)]
            top_avg = sum(float(v) for v in numeric_top) / len(numeric_top) if numeric_top else None
            bottom_avg = sum(float(v) for v in numeric_bottom) / len(numeric_bottom) if numeric_bottom else None
            rho = spearman_correlation(hp_values, metric_values)
            corr_entry = {
                "param": key,
                "spearman_rho": round(rho, 4),
            }
            if top_avg is not None:
                corr_entry["top_avg"] = top_avg
            if bottom_avg is not None:
                corr_entry["bottom_avg"] = bottom_avg
            if top_avg is not None and bottom_avg is not None:
                corr_entry["direction"] = "lower" if top_avg < bottom_avg else "higher"
            if len(numeric_pairs) < len(all_hp):
                corr_entry["note"] = f"{len(all_hp) - len(numeric_pairs)} non-numeric values excluded"
            correlations.append(corr_entry)
        else:
            # Categorical — report most common values
            correlations.append({
                "param": key,
                "top_common": max(set(top_vals), key=top_vals.count) if top_vals else None,
                "bottom_common": max(set(bottom_vals), key=bottom_vals.count) if bottom_vals else None,
            })

    return {"correlations": correlations}


def build_experiment_description(
    exp_id: str,
    data: dict,
    baseline_config: dict | None = None,
    max_len: int = 45,
) -> str:
    """Build a short human-readable description for a progress chart annotation.

    Combines the method name (from ``code_proposal``) with the most
    distinctive HP change vs *baseline_config*.  Falls back to exp_id
    when no richer information is available.

    Returns a string of at most *max_len* characters.
    """
    parts: list[str] = []

    # Stacked methods (multiple branches combined)
    branches = data.get("code_branches")
    if branches and isinstance(branches, list):
        names = [b.removeprefix("ml-opt/") for b in branches]
        parts.append(" + ".join(names))
    else:
        # Single method
        proposal = data.get("code_proposal") or data.get("code_branch", "")
        if proposal:
            proposal = proposal.removeprefix("ml-opt/")
            parts.append(proposal)

    # HP diff vs baseline
    config = data.get("config", {})
    if config and baseline_config:
        diffs: list[str] = []
        for key in sorted(config):
            cur = config[key]
            base = baseline_config.get(key)
            if base is not None and cur != base:
                diffs.append(f"{key}={cur}")
        if diffs:
            parts.append(", ".join(diffs[:2]))  # top 2 HP changes
    elif config and not baseline_config:
        # No baseline to diff against — show top HP value
        interesting = [(k, v) for k, v in config.items()
                       if k not in ("exp_id", "gpu_id")]
        if interesting:
            k, v = interesting[0]
            parts.append(f"{k}={v}")

    desc = " | ".join(parts) if parts else exp_id
    if len(desc) > max_len:
        desc = desc[: max_len - 3] + "..."
    return desc


def rank_methods_for_stacking(
    results: dict[str, dict],
    metric: str,
    lower_is_better: bool = True,
) -> list[dict]:
    """Rank methods by improvement over baseline for stacking.

    For each code_branch, finds the best experiment result. Excludes methods
    that didn't improve over baseline. Returns a list sorted by improvement
    magnitude (most improved first).

    Each entry contains: code_branch, code_proposal, best_metric,
    best_config, best_exp_id, improvement_pct.
    """
    baseline = results.get("baseline", {})
    baseline_metrics = baseline.get("metrics", baseline)
    if metric not in baseline_metrics:
        return []
    baseline_val = baseline_metrics[metric]

    # Group by code_branch, find best per branch
    branch_best: dict[str, dict] = {}
    for exp_id, data in results.items():
        if exp_id == "baseline":
            continue
        branch = data.get("code_branch")
        if not branch:
            continue
        status = data.get("status")
        if status is not None and status != "completed":
            continue
        exp_metrics = data.get("metrics", data)
        if metric not in exp_metrics:
            continue
        val = exp_metrics[metric]
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            continue

        if branch not in branch_best:
            branch_best[branch] = {
                "code_branch": branch,
                "code_proposal": data.get("code_proposal", branch.removeprefix("ml-opt/")),
                "best_metric": val,
                "best_config": data.get("config", {}),
                "best_exp_id": exp_id,
            }
        else:
            current = branch_best[branch]["best_metric"]
            better = val < current if lower_is_better else val > current
            if better:
                branch_best[branch]["best_metric"] = val
                branch_best[branch]["best_config"] = data.get("config", {})
                branch_best[branch]["best_exp_id"] = exp_id

    # Filter to methods that improved over baseline and compute improvement
    improved = []
    for entry in branch_best.values():
        val = entry["best_metric"]
        if lower_is_better:
            improved_over_baseline = val < baseline_val
        else:
            improved_over_baseline = val > baseline_val
        if not improved_over_baseline:
            continue
        if abs(baseline_val) < 1e-8:
            pct = None
        else:
            delta = baseline_val - val if lower_is_better else val - baseline_val
            pct = round(delta / abs(baseline_val) * 100, 2)
        entry["improvement_pct"] = pct
        improved.append(entry)

    # Sort by improvement magnitude (most improved first)
    def _sort_key(e):
        pct = e.get("improvement_pct")
        return pct if pct is not None else 0.0

    improved.sort(key=_sort_key, reverse=True)
    return improved


def group_by_method_tier(results: dict[str, dict]) -> dict[str, list[dict]]:
    """Group experiments by method_tier for three-tier analysis.

    Tiers: baseline, method_default_hp, method_tuned_hp.
    Experiments without a method_tier field are grouped as 'unknown'.
    """
    groups: dict[str, list[dict]] = {}
    for exp_id, data in results.items():
        tier = data.get("method_tier", "unknown")
        groups.setdefault(tier, []).append({"exp_id": exp_id, **data})
    return {k: v for k, v in groups.items() if v}


def analyze(results_dir: str, metric: str, baseline_id: str = "baseline", lower_is_better: bool = True) -> dict:
    """Full analysis: load, rank, compute deltas, find correlations."""
    results = load_results(results_dir)
    if not results:
        return {"error": "No results found", "results_dir": results_dir}

    result = {
        "num_experiments": len(results),
        "ranking": rank_by_metric(results, metric, lower_is_better),
        "deltas": compute_deltas(results, baseline_id, metric),
        "correlations": identify_correlations(results, metric, lower_is_better),
    }
    if baseline_id not in results:
        result["warning"] = f"Baseline '{baseline_id}' not found; deltas not computed"
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: result_analyzer.py <results_dir> <metric> [baseline_id] [lower_is_better]")
        sys.exit(1)
    results_dir = sys.argv[1]
    metric = sys.argv[2]
    baseline_id = sys.argv[3] if len(sys.argv) > 3 else "baseline"
    lower = sys.argv[4].lower() not in ("false", "0", "no") if len(sys.argv) > 4 else True
    print(json.dumps(analyze(results_dir, metric, baseline_id, lower), indent=2))
