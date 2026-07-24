#!/usr/bin/env python3
"""Analyze and compare experiment results.

Loads experiment result JSONs (baseline + round-N-<type>/exp-*.json), ranks them
by a metric, computes deltas vs baseline, and finds HP-metric correlations.

Usage:
    python3 result_analyzer.py <results_dir> <metric>                                  # Full analysis vs default baseline id
    python3 result_analyzer.py <results_dir> <metric> <baseline_id>                    # Full analysis vs a named baseline
    python3 result_analyzer.py <results_dir> <metric> <baseline_id> <lower_is_better>  # ... with explicit polarity
    python3 result_analyzer.py <results_dir> compare <exp_id_1> <exp_id_2>             # Pairwise config/metric comparison
    python3 result_analyzer.py <results_dir> compare <exp_id_1> <exp_id_2> <metric> <lower_is_better>  # ... with metric + polarity

The trailing lower_is_better arg defaults to true; pass false/0/no for accuracy-like
metrics. baseline_id defaults to "baseline"; compare's metric defaults to "loss".

Examples:
    python3 result_analyzer.py <exp_root>/results accuracy baseline false
    python3 result_analyzer.py <exp_root>/results loss
    python3 result_analyzer.py <exp_root>/results compare exp-001 exp-007 accuracy false
"""

import json
import math
import statistics
import sys
from pathlib import Path


# Statistical constants (spec-fixed, NOT tunable heuristics; per-model-category
# thresholds live in detect_divergence's MODEL_CATEGORY_DEFAULTS, not here).

# Two-sided critical |Spearman rho| at alpha=0.05 for small n (n<5 never significant).
SPEARMAN_CRITICAL_05 = {5: 1.0, 6: 0.886, 7: 0.786, 8: 0.738, 9: 0.700, 10: 0.648}

# Correlations from fewer than this many points are flagged low_n.
LOW_N = 10

# When |baseline| < this factor x batch std, report deltas vs spread (percent-of-baseline unstable).
ZERO_CENTERED_BASELINE_FACTOR = 2.0


def _avg_rank(values: list) -> list[float]:
    """Assign ranks with average-rank tie-breaking (1-based)."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j]] == values[indexed[j + 1]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def load_results(results_dir: str) -> dict[str, dict]:
    """Load all results: flat `baseline.json`/`exp-*.json` + round `round-*/exp-*.json` (exp-ids globally unique)."""
    path = Path(results_dir)
    results = {}
    if not path.exists():
        return results

    # Load baseline and any flat exp-*.json (backwards compat)
    for f in sorted(path.glob("*.json")):
        if f.stem.lower() != "baseline" and not f.stem.startswith("exp-"):
            continue
        try:
            data = json.loads(f.read_text())
            results[f.stem] = data
        except (json.JSONDecodeError, OSError):
            continue

    # Load from round directories (round-N-type/exp-*.json)
    for f in sorted(path.glob("round-*/exp-*.json")):
        if not f.stem.startswith("exp-"):
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
                # Lets consumers (dashboard, results-table) flag non-held-out-eval results.
                "eval_protocol": data.get("eval_protocol"),
            })
    valid = [r for r in ranked if isinstance(r["value"], (int, float)) and math.isfinite(r["value"])]
    invalid = [r for r in ranked if not (isinstance(r["value"], (int, float)) and math.isfinite(r["value"]))]
    for r in invalid:
        r["note"] = "non-finite metric value excluded from ranking"
    valid.sort(key=lambda x: x["value"], reverse=not lower_is_better)
    return valid + invalid


def spearman_correlation(x: list, y: list) -> float:
    """Spearman rank correlation: rho = 1 - 6*sum(d^2)/(n*(n^2-1)), average-rank ties."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    n = len(x)
    rx = _avg_rank(x)
    ry = _avg_rank(y)
    # Constant ranks have no variance — correlation is undefined; return 0.0
    if len(set(rx)) == 1 or len(set(ry)) == 1:
        return 0.0
    d_sq_sum = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d_sq_sum / (n * (n ** 2 - 1))


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
    # Batch spread — enables the zero-centered fallback (deltas vs spread when |baseline| is small).
    values = [d["value"] for d in deltas
              if isinstance(d["value"], (int, float)) and math.isfinite(d["value"])]
    batch_std = round(statistics.stdev(values), 6) if len(values) >= 2 else None
    zero_centered = (
        batch_std is not None and batch_std > 0
        and abs(baseline_val) < ZERO_CENTERED_BASELINE_FACTOR * batch_std
    )
    for d in deltas:
        d["batch_std"] = batch_std
        if zero_centered:
            d["delta_pct"] = None
            d["delta_vs_spread"] = round(d["delta"] / batch_std, 4)
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
            n_pairs = len(numeric_pairs)
            corr_entry = {
                "param": key,
                "spearman_rho": round(rho, 4),
                "n": n_pairs,
                "low_n": n_pairs < LOW_N,
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
            n_cat = len(top_vals) + len(bottom_vals)
            correlations.append({
                "param": key,
                "top_common": max(set(top_vals), key=top_vals.count) if top_vals else None,
                "bottom_common": max(set(bottom_vals), key=bottom_vals.count) if bottom_vals else None,
                "n": n_cat,
                "low_n": n_cat < LOW_N,
            })

    return {"correlations": correlations}


def build_experiment_description(
    exp_id: str,
    data: dict,
    baseline_config: dict | None = None,
    max_len: int = 45,
) -> str:
    """Short chart-annotation description: method name (`code_proposal`) + top HP diff vs baseline_config, <= max_len chars, falling back to exp_id."""
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


def aggregate_replicates(results: dict[str, dict], metric: str, lower_is_better: bool = True) -> list[dict]:
    """Group completed experiments into seed-replicate groups.

    A replicate group = same `code_branch` + identical config except `random_seed`
    (top-level field, falling back to config key). Experiments WITHOUT a random_seed
    are never pooled — each stays a singleton. Returns group dicts: {code_branch,
    code_proposal, method_tier, config (seed-stripped), exp_ids, values, n, mean,
    std (sample std, None when n==1), best_exp_id}.
    """
    groups: dict[tuple, dict] = {}
    for exp_id, data in results.items():
        if exp_id == "baseline":
            continue
        status = data.get("status")
        if status is not None and status != "completed":
            continue
        metrics = data.get("metrics", data)
        if not isinstance(metrics, dict) or metric not in metrics:
            continue
        val = metrics[metric]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            continue
        cfg = data.get("config", {})
        cfg = cfg if isinstance(cfg, dict) else {}
        seed = data.get("random_seed", cfg.get("random_seed"))
        stripped = {k: v for k, v in cfg.items() if k != "random_seed"}
        if seed is not None:
            key = (data.get("code_branch"), json.dumps(stripped, sort_keys=True, default=str))
        else:
            key = (data.get("code_branch"), f"__solo__{exp_id}")
        g = groups.setdefault(key, {
            "code_branch": data.get("code_branch"),
            "code_proposal": data.get("code_proposal"),
            "method_tier": data.get("method_tier"),
            "config": stripped,
            "exp_ids": [],
            "values": [],
        })
        g["exp_ids"].append(exp_id)
        g["values"].append(float(val))

    out = []
    for g in groups.values():
        n = len(g["values"])
        best = min(g["values"]) if lower_is_better else max(g["values"])
        g["n"] = n
        g["mean"] = sum(g["values"]) / n
        g["std"] = round(statistics.stdev(g["values"]), 6) if n >= 2 else None
        g["best_exp_id"] = g["exp_ids"][g["values"].index(best)]
        out.append(g)
    return out


def rank_methods_for_stacking(
    results: dict[str, dict],
    metric: str,
    lower_is_better: bool = True,
) -> list[dict]:
    """Rank methods by improvement over baseline for stacking (most improved first).

    Seed replicates are aggregated first — each branch ranked by its best
    replicate-group MEAN, never the luckiest single seed. Excludes non-improving
    methods. Each entry: code_branch, code_proposal, best_metric (group mean),
    best_config, best_exp_id, replicates {n, mean, std}, improvement_pct.
    """
    baseline = results.get("baseline", {})
    baseline_metrics = baseline.get("metrics", baseline)
    if metric not in baseline_metrics:
        return []
    baseline_val = baseline_metrics[metric]

    # Best replicate group per branch (group value = mean across seeds)
    branch_best: dict[str, dict] = {}
    for grp in aggregate_replicates(results, metric, lower_is_better):
        branch = grp["code_branch"]
        if not branch:
            continue
        val = grp["mean"]
        entry = {
            "code_branch": branch,
            "code_proposal": grp["code_proposal"] or branch.removeprefix("ml-opt/"),
            "best_metric": val,
            "best_config": grp["config"],
            "best_exp_id": grp["best_exp_id"],
            "replicates": {"n": grp["n"], "mean": grp["mean"], "std": grp["std"]},
        }
        if branch not in branch_best:
            branch_best[branch] = entry
        else:
            current = branch_best[branch]["best_metric"]
            better = val < current if lower_is_better else val > current
            if better:
                branch_best[branch] = entry

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


def detect_hp_interactions(
    results: dict[str, dict],
    metric: str,
    lower_is_better: bool = True,
    min_experiments: int = 5,
    min_interaction_rho: float = 0.5,
) -> dict:
    """Detect 2-way HP interaction effects using product of centered ranks.

    For each pair of numeric HPs, computes the interaction term (product
    of centered ranks) and correlates it with the metric. Reports
    interactions stronger than either individual HP correlation.
    """
    # Filter to completed experiments with metric and config
    exps = []
    for eid, data in results.items():
        if eid == "baseline":
            continue
        if data.get("status") != "completed":
            continue
        m = data.get("metrics", {})
        cfg = data.get("config", {})
        if not isinstance(m, dict) or not isinstance(cfg, dict):
            continue
        val = m.get(metric)
        if val is None:
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue
        if not math.isfinite(val):
            continue
        exps.append({"metric_val": val, "config": cfg})

    if len(exps) < min_experiments:
        return {"interactions": [], "note": f"Need at least {min_experiments} experiments for interaction detection"}

    # Identify numeric HP keys
    all_keys: set[str] = set()
    for e in exps:
        all_keys.update(e["config"].keys())

    numeric_keys = []
    for key in sorted(all_keys):
        vals = []
        for e in exps:
            v = e["config"].get(key)
            if v is not None:
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    break
        else:
            if len(set(vals)) > 1:  # skip constant HPs
                numeric_keys.append(key)

    if len(numeric_keys) < 2:
        return {"interactions": [], "note": "Need at least 2 varying numeric HPs for interaction detection"}

    interactions = []
    for a_idx in range(len(numeric_keys)):
        for b_idx in range(a_idx + 1, len(numeric_keys)):
            key_a, key_b = numeric_keys[a_idx], numeric_keys[b_idx]

            # Collect experiments that have both HPs
            subset = []
            for e in exps:
                va = e["config"].get(key_a)
                vb = e["config"].get(key_b)
                if va is not None and vb is not None:
                    try:
                        subset.append((float(va), float(vb), e["metric_val"]))
                    except (ValueError, TypeError):
                        continue

            if len(subset) < min_experiments:
                continue

            vals_a = [s[0] for s in subset]
            vals_b = [s[1] for s in subset]
            vals_metric = [s[2] for s in subset]

            # Individual correlations
            rho_a = spearman_correlation(vals_a, vals_metric)
            rho_b = spearman_correlation(vals_b, vals_metric)

            # Centered ranks
            ranks_a = _avg_rank(vals_a)
            ranks_b = _avg_rank(vals_b)
            mean_a = sum(ranks_a) / len(ranks_a)
            mean_b = sum(ranks_b) / len(ranks_b)
            centered_a = [r - mean_a for r in ranks_a]
            centered_b = [r - mean_b for r in ranks_b]

            # Interaction term = product of centered ranks
            interaction_term = [centered_a[i] * centered_b[i] for i in range(len(subset))]

            # Correlate interaction term with metric
            rho_interaction = spearman_correlation(interaction_term, vals_metric)

            # Filter: must be strong AND stronger than either individual
            if abs(rho_interaction) >= min_interaction_rho and abs(rho_interaction) > max(abs(rho_a), abs(rho_b)):
                # Generate description
                if rho_interaction < 0 and lower_is_better:
                    desc = f"Combined high {key_a} + high {key_b} correlates with better (lower) {metric}"
                elif rho_interaction > 0 and lower_is_better:
                    desc = f"Combined high {key_a} + high {key_b} correlates with worse (higher) {metric}"
                elif rho_interaction < 0 and not lower_is_better:
                    desc = f"Combined high {key_a} + high {key_b} correlates with worse (lower) {metric}"
                else:
                    desc = f"Combined high {key_a} + high {key_b} correlates with better (higher) {metric}"

                # Small-n significance gate: exact critical value where the
                # lookup table covers n; larger n falls through to the
                # min_interaction_rho strength gate above.
                n_pair = len(subset)
                crit = SPEARMAN_CRITICAL_05.get(n_pair)
                significant = crit is None or abs(rho_interaction) >= crit

                interaction_entry = {
                    "param_a": key_a,
                    "param_b": key_b,
                    "interaction_rho": round(rho_interaction, 4),
                    "individual_rho_a": round(rho_a, 4),
                    "individual_rho_b": round(rho_b, 4),
                    "description": desc,
                    "n_experiments": n_pair,
                    "significant": significant,
                }
                if not significant:
                    interaction_entry["note"] = (
                        f"not significant at this n (n={n_pair}, |rho| < {crit})"
                    )
                interactions.append(interaction_entry)

    interactions.sort(key=lambda x: abs(x["interaction_rho"]), reverse=True)
    return {"interactions": interactions, "note": None}


def compute_branch_scores(
    results: dict[str, dict],
    metric: str,
    lower_is_better: bool = True,
) -> dict[str, dict]:
    """Per-branch allocation scores (null branch keyed `__baseline__`): improvement_pct over baseline (best replicate-group mean), sample_count, replicates {n, mean, std}, composite score."""
    # Find baseline metric value
    baseline = results.get("baseline", {})
    baseline_metrics = baseline.get("metrics", {})
    baseline_val = baseline_metrics.get(metric)
    if baseline_val is None:
        return {}
    try:
        baseline_val = float(baseline_val)
    except (ValueError, TypeError):
        return {}

    # Group replicate-aggregated results by code_branch (stacked tiers
    # excluded). A branch's best is the best replicate-group MEAN; the group
    # std is that branch's measured noise floor.
    branch_groups: dict[str, list[dict]] = {}
    for grp in aggregate_replicates(results, metric, lower_is_better):
        tier = grp.get("method_tier") or ""
        if isinstance(tier, str) and tier.startswith("stacked_"):
            continue
        branch = grp["code_branch"] or "__baseline__"
        branch_groups.setdefault(branch, []).append(grp)

    scores = {}
    for branch, grps in branch_groups.items():
        if lower_is_better:
            best = min(grps, key=lambda g: g["mean"])
        else:
            best = max(grps, key=lambda g: g["mean"])

        best_val = best["mean"]
        if abs(baseline_val) > 1e-12:
            if lower_is_better:
                improvement_pct = (baseline_val - best_val) / abs(baseline_val) * 100
            else:
                improvement_pct = (best_val - baseline_val) / abs(baseline_val) * 100
        else:
            improvement_pct = 0.0

        sample_count = sum(g["n"] for g in grps)
        confidence = 1 - 1 / math.sqrt(sample_count + 1)
        score = max(improvement_pct * confidence, 0.0)

        scores[branch] = {
            "best_metric": best_val,
            "best_exp_id": best["best_exp_id"],
            "improvement_pct": round(improvement_pct, 2),
            "sample_count": sample_count,
            "replicates": {"n": best["n"], "mean": best["mean"], "std": best["std"]},
            "score": round(score, 2),
        }

    return scores


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
        "interactions": detect_hp_interactions(results, metric, lower_is_better),
        "branch_scores": compute_branch_scores(results, metric, lower_is_better),
    }
    if baseline_id not in results:
        result["warning"] = f"Baseline '{baseline_id}' not found; deltas not computed"
    return result


def compare_experiments(
    results_dir: str,
    exp_ids: list[str],
    metric: str,
    lower_is_better: bool = True,
) -> dict:
    """Pairwise comparison of experiments: config diff, metric delta, winner.

    Returns a structured dict with `config_diff`, `metrics_comparison`,
    `metadata`, and `winner` fields.
    """
    results = load_results(results_dir)
    if len(exp_ids) != 2:
        return {"error": "Exactly 2 experiment IDs required"}
    loaded = {}
    for eid in exp_ids:
        if eid not in results:
            return {"error": f"Experiment '{eid}' not found in {results_dir}"}
        loaded[eid] = results[eid]

    # Config diff
    all_keys: set[str] = set()
    for data in loaded.values():
        all_keys.update(data.get("config", {}).keys())
    config_diff = {}
    for key in sorted(all_keys):
        vals = {eid: data.get("config", {}).get(key) for eid, data in loaded.items()}
        unique = set(str(v) for v in vals.values())
        config_diff[key] = {**vals, "differs": len(unique) > 1}

    # Metrics comparison
    all_metrics: set[str] = set()
    for data in loaded.values():
        all_metrics.update(data.get("metrics", {}).keys())
    metrics_comparison = {}
    for mk in sorted(all_metrics):
        entry: dict = {}
        vals = []
        for eid, data in loaded.items():
            v = data.get("metrics", {}).get(mk)
            entry[eid] = v
            if v is not None:
                try:
                    vals.append((eid, float(v)))
                except (ValueError, TypeError):
                    pass
        if len(vals) == 2:
            delta = vals[1][1] - vals[0][1]
            base = abs(vals[0][1])
            entry["delta"] = round(delta, 6)
            entry["delta_pct"] = round(delta / base * 100, 2) if base > 1e-12 else 0.0
        metrics_comparison[mk] = entry

    # Metadata
    metadata = {}
    for eid, data in loaded.items():
        metadata[eid] = {
            "status": data.get("status"),
            "duration_seconds": data.get("duration_seconds"),
            "code_branch": data.get("code_branch"),
            "iteration": data.get("iteration"),
            "method_tier": data.get("method_tier"),
        }

    # Winner
    winner = {}
    mc = metrics_comparison.get(metric)
    if mc and "delta" in mc:
        id_a, id_b = exp_ids[0], exp_ids[1]
        va, vb = mc.get(id_a), mc.get(id_b)
        if va is not None and vb is not None:
            try:
                fa, fb = float(va), float(vb)
                if lower_is_better:
                    w = id_a if fa <= fb else id_b
                else:
                    w = id_a if fa >= fb else id_b
                imp = abs(mc["delta_pct"])
                winner = {"metric": metric, "exp_id": w, "improvement_pct": imp}
            except (ValueError, TypeError):
                pass

    return {
        "experiments": exp_ids,
        "config_diff": config_diff,
        "metrics_comparison": metrics_comparison,
        "metadata": metadata,
        "winner": winner,
    }


def format_comparison_table(comparison: dict) -> str:
    """Format a comparison dict as a human-readable ASCII table."""
    if "error" in comparison:
        return comparison["error"]
    ids = comparison["experiments"]
    lines = [f"EXPERIMENT COMPARISON: {ids[0]} vs {ids[1]}", "=" * 50]

    # Metadata
    meta = comparison.get("metadata", {})
    for field in ("status", "code_branch", "iteration", "method_tier"):
        vals = [str(meta.get(eid, {}).get(field, "—")) for eid in ids]
        lines.append(f"  {field:<20s} {vals[0]:<16s} {vals[1]:<16s}")
    lines.append("")

    # Config diff
    lines.append("Config:")
    for key, info in comparison.get("config_diff", {}).items():
        vals = [str(info.get(eid, "—")) for eid in ids]
        marker = " *" if info.get("differs") else ""
        lines.append(f"  {key:<20s} {vals[0]:<16s} {vals[1]:<16s}{marker}")
    lines.append("")

    # Metrics
    lines.append("Metrics:")
    for mk, info in comparison.get("metrics_comparison", {}).items():
        vals = []
        for eid in ids:
            v = info.get(eid)
            vals.append(f"{v:.4f}" if isinstance(v, (int, float)) else str(v))
        delta_str = ""
        if "delta_pct" in info and info["delta_pct"] is not None:
            sign = "+" if info["delta_pct"] >= 0 else ""
            delta_str = f"  {sign}{info['delta_pct']:.1f}%"
        lines.append(f"  {mk:<20s} {vals[0]:<16s} {vals[1]:<16s}{delta_str}")

    # Winner
    w = comparison.get("winner", {})
    if w:
        lines.append("")
        lines.append(f"Winner: {w.get('exp_id')} ({w.get('metric')} {w.get('improvement_pct', 0):.1f}% better)")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: result_analyzer.py <results_dir> <metric> [baseline_id] [lower_is_better]")
        print("       result_analyzer.py <results_dir> compare <exp_id_1> <exp_id_2> [metric] [lower_is_better]")
        sys.exit(1)
    results_dir = sys.argv[1]
    if sys.argv[2] == "compare":
        if len(sys.argv) < 5:
            print("Usage: result_analyzer.py <results_dir> compare <exp_id_1> <exp_id_2> [metric] [lower_is_better]")
            sys.exit(1)
        ids = [sys.argv[3], sys.argv[4]]
        metric = sys.argv[5] if len(sys.argv) > 5 else "loss"
        lower = sys.argv[6].lower() not in ("false", "0", "no") if len(sys.argv) > 6 else True
        result = compare_experiments(results_dir, ids, metric, lower)
        print(format_comparison_table(result))
        print()
        print(json.dumps(result, indent=2))
    else:
        metric = sys.argv[2]
        baseline_id = sys.argv[3] if len(sys.argv) > 3 else "baseline"
        lower = sys.argv[4].lower() not in ("false", "0", "no") if len(sys.argv) > 4 else True
        print(json.dumps(analyze(results_dir, metric, baseline_id, lower), indent=2))
