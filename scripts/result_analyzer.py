#!/usr/bin/env python3
"""Analyze and compare experiment results."""

import json
import sys
from pathlib import Path


def load_results(results_dir: str) -> dict[str, dict]:
    """Load all experiment results from a directory."""
    path = Path(results_dir)
    results = {}
    if not path.exists():
        return results
    for f in sorted(path.glob("*.json")):
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
            })
    ranked.sort(key=lambda x: x["value"], reverse=not lower_is_better)
    return ranked


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
            pct = (delta / abs(baseline_val) * 100) if baseline_val != 0 else 0
            deltas.append({
                "exp_id": exp_id,
                "value": val,
                "delta": delta,
                "delta_pct": round(pct, 2),
                "config": data.get("config", {}),
            })
    return deltas


def identify_correlations(results: dict[str, dict], metric: str, lower_is_better: bool = True) -> dict:
    """Identify which hyperparameters correlate with improvement."""
    # Collect (config_key, config_value, metric_value) triples
    entries = []
    for exp_id, data in results.items():
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

    for key in sorted(all_keys):
        top_vals = [e["config"].get(key) for e in top_half if key in e["config"]]
        bottom_vals = [e["config"].get(key) for e in bottom_half if key in e["config"]]
        if top_vals and bottom_vals:
            # Numeric comparison
            try:
                top_avg = sum(float(v) for v in top_vals) / len(top_vals)
                bottom_avg = sum(float(v) for v in bottom_vals) / len(bottom_vals)
                correlations.append({
                    "param": key,
                    "top_avg": top_avg,
                    "bottom_avg": bottom_avg,
                    "direction": "lower" if top_avg < bottom_avg else "higher",
                })
            except (ValueError, TypeError):
                # Categorical — report most common values
                correlations.append({
                    "param": key,
                    "top_common": max(set(top_vals), key=top_vals.count) if top_vals else None,
                    "bottom_common": max(set(bottom_vals), key=bottom_vals.count) if bottom_vals else None,
                })

    return {"correlations": correlations}


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
    lower = sys.argv[4].lower() != "false" if len(sys.argv) > 4 else True
    print(json.dumps(analyze(results_dir, metric, baseline_id, lower), indent=2))
