#!/usr/bin/env python3
"""
Hyperagent adapter — CLI wrapper bridging Hyperagents' evolutionary algorithms
with the ml-optimizer plugin's experiment structure.

Uses Hyperagents submodule for archive format and parent selection math.
Falls back to stdlib reimplementation if submodule is unavailable.

Usage:
    python3 scripts/hyperagent_adapter.py <exp_root> init
    python3 scripts/hyperagent_adapter.py <exp_root> select-parent [strategy]
    python3 scripts/hyperagent_adapter.py <exp_root> add '<json>'
    python3 scripts/hyperagent_adapter.py <exp_root> lineage <genid>
    python3 scripts/hyperagent_adapter.py <exp_root> stats
    python3 scripts/hyperagent_adapter.py <exp_root> best [n]
    python3 scripts/hyperagent_adapter.py <exp_root> operator-stats
    python3 scripts/hyperagent_adapter.py <exp_root> prune
    python3 scripts/hyperagent_adapter.py --help
"""

import fcntl
import json
import math
import os
import random
import sys
from datetime import datetime, timezone

ARCHIVE_FILENAME = "code-archive.jsonl"


# ---------------------------------------------------------------------------
# Archive I/O (matches Hyperagents' JSONL append-only format)
# ---------------------------------------------------------------------------

def _archive_path(exp_root):
    return os.path.join(exp_root, ARCHIVE_FILENAME)


def _lock_and_read(path):
    """Read archive with file lock for concurrent safety."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            entries = []
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
            return entries
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _lock_and_append(path, entry):
    """Append a single entry with file lock."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(entry) + "\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_archive(exp_root):
    """Load all archive entries. Returns list of dicts."""
    return _lock_and_read(_archive_path(exp_root))


def _build_index(entries):
    """Build genid → entry lookup from archive entries."""
    index = {}
    for e in entries:
        gid = e.get("genid") or e.get("current_genid")
        if gid is not None:
            index[gid] = e
    return index


# ---------------------------------------------------------------------------
# Init — create archive from baseline + existing branches
# ---------------------------------------------------------------------------

def cmd_init(exp_root):
    """Create archive with gen-000 from baseline.json."""
    path = _archive_path(exp_root)
    if os.path.exists(path):
        entries = load_archive(exp_root)
        if entries:
            print(json.dumps({"status": "already_initialized", "entries": len(entries)}))
            return

    # Read baseline
    baseline_path = os.path.join(exp_root, "results", "baseline.json")
    fitness = 0.0
    baseline_warning = None
    if not os.path.exists(baseline_path):
        baseline_warning = "baseline.json not found — using fitness=0.0 for gen-000"
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        metrics = baseline.get("metrics", {})
        # Use primary_metric from pipeline state if available
        state_path = os.path.join(exp_root, "pipeline-state.json")
        primary_metric = None
        if os.path.exists(state_path):
            with open(state_path) as sf:
                state = json.load(sf)
            uc = state.get("user_choices", {})
            primary_metric = uc.get("primary_metric")
        if primary_metric and primary_metric in metrics:
            fitness = metrics[primary_metric]
        elif metrics:
            fitness = list(metrics.values())[0]

    entry = {
        "genid": "gen-000",
        "parent_genid": None,
        "code_branch": "main",
        "mutation_type": None,
        "mutation_description": "Baseline (original model)",
        "fitness_score": fitness,
        "staged_score": None,
        "best_hp_config": None,
        "best_exp_id": "baseline",
        "lineage_depth": 0,
        "lineage": ["gen-000"],
        "num_children": 0,
        "status": "evaluated",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _lock_and_append(path, entry)

    # Seed from implementation manifest if present
    manifest_path = os.path.join(exp_root, "results", "implementation-manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        gen_num = 1
        for prop in manifest.get("proposals", []):
            if prop.get("status") != "validated":
                continue
            genid = f"gen-{gen_num:03d}"
            seed_entry = {
                "genid": genid,
                "parent_genid": "gen-000",
                "code_branch": prop.get("branch", f"ml-opt/{prop.get('slug', 'unknown')}"),
                "mutation_type": "research_implement",
                "mutation_description": prop.get("name", "Research proposal"),
                "fitness_score": None,
                "staged_score": None,
                "best_hp_config": None,
                "best_exp_id": None,
                "lineage_depth": 1,
                "lineage": ["gen-000", genid],
                "num_children": 0,
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _lock_and_append(path, seed_entry)
            gen_num += 1

    entries = load_archive(exp_root)
    result = {"status": "initialized", "entries": len(entries)}
    if baseline_warning:
        result["warning"] = baseline_warning
    print(json.dumps(result))


# ---------------------------------------------------------------------------
# Parent selection — exact Hyperagents math (sigmoid + diversity penalty)
# ---------------------------------------------------------------------------

def _sigmoid_scores(scores, top_k=3):
    """Sigmoid-transform scores relative to top-K mean.

    Exact formula from Hyperagents gl_utils.py select_parent():
        sigmoid(10 * (score - midpoint))
    where midpoint = mean of top-3 scores.
    """
    if not scores:
        return []
    sorted_desc = sorted(scores, reverse=True)[:top_k]
    mid_point = sum(sorted_desc) / len(sorted_desc)
    return [1.0 / (1.0 + math.exp(-10.0 * (s - mid_point))) for s in scores]


def _diversity_penalties(child_counts):
    """Diversity penalty inversely proportional to reproduction count.

    Exact formula from Hyperagents gl_utils.py select_parent():
        exp(-(child_count / 8)^3)
    """
    return [math.exp(-((c / 8.0) ** 3)) for c in child_counts]


def cmd_select_parent(exp_root, strategy="score_child_prop"):
    """Select a parent from the archive using the specified strategy."""
    entries = load_archive(exp_root)
    if not entries:
        print(json.dumps({"error": "Archive is empty. Run init first."}))
        sys.exit(1)

    # Build candidates: genid → fitness_score (only evaluated entries)
    candidates = {}
    for e in entries:
        gid = e.get("genid")
        score = e.get("fitness_score")
        status = e.get("status", "")
        if score is not None and status == "evaluated":
            candidates[gid] = score

    if not candidates:
        # Fallback to first entry
        first = entries[0]
        gid = first.get("genid", "gen-000")
        print(json.dumps({"genid": gid, "code_branch": first.get("code_branch", "main"),
                           "fitness_score": first.get("fitness_score", 0.0), "strategy": "fallback"}))
        return

    # Count children
    child_counts = {gid: 0 for gid in candidates}
    for e in entries:
        parent = e.get("parent_genid")
        if parent in child_counts:
            child_counts[parent] += 1

    genids = list(candidates.keys())
    scores = [candidates[g] for g in genids]

    if strategy == "random":
        selected = random.choice(genids)

    elif strategy == "latest":
        selected = genids[-1]

    elif strategy == "best":
        selected = max(candidates, key=lambda g: candidates[g])

    elif strategy == "score_prop":
        sig = _sigmoid_scores(scores)
        total = sum(sig)
        weights = [s / total for s in sig] if total > 0 else [1.0 / len(sig)] * len(sig)
        selected = random.choices(genids, weights=weights, k=1)[0]

    elif strategy == "score_child_prop":
        sig = _sigmoid_scores(scores)
        pen = _diversity_penalties([child_counts[g] for g in genids])
        combined = [s * p for s, p in zip(sig, pen)]
        total = sum(combined)
        weights = [c / total for c in combined] if total > 0 else [1.0 / len(combined)] * len(combined)
        selected = random.choices(genids, weights=weights, k=1)[0]

    else:
        print(json.dumps({"error": f"Unknown strategy: {strategy}"}))
        sys.exit(1)

    # Find the entry for selected genid
    index = _build_index(entries)
    entry = index.get(selected, {})
    print(json.dumps({
        "genid": selected,
        "code_branch": entry.get("code_branch", "main"),
        "fitness_score": entry.get("fitness_score"),
        "strategy": strategy,
    }))


# ---------------------------------------------------------------------------
# Add entry
# ---------------------------------------------------------------------------

def cmd_add(exp_root, entry_json):
    """Add a new entry to the archive."""
    entry = json.loads(entry_json)
    entries = load_archive(exp_root)
    # Ensure required fields
    if "genid" not in entry:
        # Auto-generate genid
        next_num = len(entries)
        entry["genid"] = f"gen-{next_num:03d}"
    else:
        # Check for duplicate genid
        existing = _build_index(entries)
        if entry["genid"] in existing:
            print(json.dumps({"error": f"Duplicate genid: {entry['genid']}"}))
            sys.exit(1)
    if "created_at" not in entry:
        entry["created_at"] = datetime.now(timezone.utc).isoformat()
    if "lineage" not in entry:
        parent = entry.get("parent_genid")
        if parent:
            index = _build_index(entries)
            parent_lineage = index.get(parent, {}).get("lineage", [parent])
            entry["lineage"] = parent_lineage + [entry["genid"]]
            entry["lineage_depth"] = len(entry["lineage"]) - 1
        else:
            entry["lineage"] = [entry["genid"]]
            entry["lineage_depth"] = 0

    _lock_and_append(_archive_path(exp_root), entry)

    # Update parent's child count in-place (read-modify-write with lock)
    parent_gid = entry.get("parent_genid")
    if parent_gid:
        _increment_children(exp_root, parent_gid)

    print(json.dumps({"status": "added", "genid": entry["genid"]}))


def _increment_children(exp_root, parent_genid):
    """Increment num_children on a parent entry (atomic write-then-replace)."""
    path = _archive_path(exp_root)
    with open(path, "r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            lines = f.readlines()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    # Modify in memory
    new_lines = []
    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue
        e = json.loads(line_s)
        if e.get("genid") == parent_genid:
            e["num_children"] = e.get("num_children", 0) + 1
        new_lines.append(json.dumps(e) + "\n")
    # Atomic write-then-replace
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.writelines(new_lines)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------

def cmd_lineage(exp_root, genid):
    """Trace full lineage for a genid."""
    index = _build_index(load_archive(exp_root))
    entry = index.get(genid)
    if not entry:
        print(json.dumps({"error": f"genid {genid} not found"}))
        sys.exit(1)
    lineage = entry.get("lineage", [genid])
    details = []
    for gid in lineage:
        e = index.get(gid, {})
        details.append({
            "genid": gid,
            "mutation_type": e.get("mutation_type"),
            "mutation_description": e.get("mutation_description"),
            "fitness_score": e.get("fitness_score"),
            "code_branch": e.get("code_branch"),
        })
    print(json.dumps({"genid": genid, "lineage": details}))


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def cmd_stats(exp_root):
    """Print archive statistics."""
    entries = load_archive(exp_root)
    if not entries:
        print(json.dumps({"entries": 0}))
        return

    evaluated = [e for e in entries if e.get("status") == "evaluated"]
    filtered_out = [e for e in entries if e.get("status") == "filtered"]
    failed = [e for e in entries if e.get("status") == "failed"]
    pending = [e for e in entries if e.get("status") == "pending"]

    scores = [e["fitness_score"] for e in evaluated if e.get("fitness_score") is not None]
    mutation_types = {}
    for e in entries:
        mt = e.get("mutation_type") or "none"
        mutation_types[mt] = mutation_types.get(mt, 0) + 1

    max_depth = max((e.get("lineage_depth", 0) for e in entries), default=0)

    stats = {
        "total_entries": len(entries),
        "evaluated": len(evaluated),
        "filtered": len(filtered_out),
        "failed": len(failed),
        "pending": len(pending),
        "best_score": max(scores) if scores else None,
        "worst_score": min(scores) if scores else None,
        "mean_score": sum(scores) / len(scores) if scores else None,
        "max_lineage_depth": max_depth,
        "mutation_types": mutation_types,
    }
    print(json.dumps(stats))


# ---------------------------------------------------------------------------
# Best N
# ---------------------------------------------------------------------------

def cmd_best(exp_root, n=5):
    """Print top N entries by fitness score."""
    entries = load_archive(exp_root)
    evaluated = [e for e in entries if e.get("fitness_score") is not None and e.get("status") == "evaluated"]
    evaluated.sort(key=lambda e: e["fitness_score"], reverse=True)
    top = evaluated[:n]
    result = []
    for e in top:
        result.append({
            "genid": e.get("genid"),
            "fitness_score": e.get("fitness_score"),
            "code_branch": e.get("code_branch"),
            "mutation_type": e.get("mutation_type"),
            "mutation_description": e.get("mutation_description"),
            "lineage_depth": e.get("lineage_depth", 0),
        })
    print(json.dumps(result))


# ---------------------------------------------------------------------------
# Operator stats
# ---------------------------------------------------------------------------

def cmd_update_fitness(exp_root, genid, new_fitness, best_exp_id=None, best_hp_config=None):
    """Update fitness score for a genid after HP tuning (atomic write)."""
    path = _archive_path(exp_root)
    entries = load_archive(exp_root)
    found = False
    new_lines = []
    for e in entries:
        if e.get("genid") == genid:
            e["fitness_score"] = new_fitness
            if e.get("status") == "pending":
                e["status"] = "evaluated"
            if best_exp_id is not None:
                e["best_exp_id"] = best_exp_id
            if best_hp_config is not None:
                e["best_hp_config"] = best_hp_config
            found = True
        new_lines.append(json.dumps(e) + "\n")
    if not found:
        print(json.dumps({"error": f"genid {genid} not found"}))
        sys.exit(1)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.writelines(new_lines)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    os.replace(tmp_path, path)
    print(json.dumps({"status": "updated", "genid": genid, "fitness_score": new_fitness}))


def cmd_operator_stats(exp_root):
    """Print mutation operator effectiveness."""
    entries = load_archive(exp_root)
    baseline_score = None
    for e in entries:
        if e.get("genid") == "gen-000":
            baseline_score = e.get("fitness_score", 0.0)
            break

    operators = {}
    for e in entries:
        mt = e.get("mutation_type")
        if mt is None:
            continue
        if mt not in operators:
            operators[mt] = {"attempts": 0, "improvements": 0, "scores": []}
        operators[mt]["attempts"] += 1
        score = e.get("fitness_score")
        if score is not None:
            operators[mt]["scores"].append(score)
            if baseline_score is not None and score > baseline_score:
                operators[mt]["improvements"] += 1

    result = {}
    for mt, data in operators.items():
        scores = data["scores"]
        result[mt] = {
            "attempts": data["attempts"],
            "improvements": data["improvements"],
            "improvement_rate": data["improvements"] / data["attempts"] if data["attempts"] > 0 else 0,
            "mean_score": sum(scores) / len(scores) if scores else None,
            "best_score": max(scores) if scores else None,
        }
    print(json.dumps(result))


# ---------------------------------------------------------------------------
# Prune — remove dead lineages
# ---------------------------------------------------------------------------

def cmd_prune(exp_root):
    """Remove entries where all descendants failed or were filtered."""
    entries = load_archive(exp_root)
    index = _build_index(entries)

    # Find all genids with at least one evaluated descendant (including self)
    alive = set()
    for e in entries:
        if e.get("status") == "evaluated":
            # Mark entire lineage as alive
            for gid in e.get("lineage", []):
                alive.add(gid)

    pruned = []
    kept = []
    path = _archive_path(exp_root)
    for e in entries:
        gid = e.get("genid")
        if gid in alive:
            kept.append(e)
        else:
            pruned.append(gid)

    # Rewrite archive
    if pruned:
        with open(path, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                for e in kept:
                    f.write(json.dumps(e) + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    print(json.dumps({"pruned": pruned, "remaining": len(kept)}))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2 or sys.argv[1] == "--help":
        print("hyperagent_adapter — CLI for Hyperagent archive management and parent selection")
        print()
        print("Usage:")
        print("  python3 scripts/hyperagent_adapter.py <exp_root> init")
        print("  python3 scripts/hyperagent_adapter.py <exp_root> select-parent [strategy]")
        print("  python3 scripts/hyperagent_adapter.py <exp_root> add '<json>'")
        print("  python3 scripts/hyperagent_adapter.py <exp_root> lineage <genid>")
        print("  python3 scripts/hyperagent_adapter.py <exp_root> stats")
        print("  python3 scripts/hyperagent_adapter.py <exp_root> best [n]")
        print("  python3 scripts/hyperagent_adapter.py <exp_root> update-fitness <genid> <score> [exp_id] [hp_json]")
        print("  python3 scripts/hyperagent_adapter.py <exp_root> operator-stats")
        print("  python3 scripts/hyperagent_adapter.py <exp_root> prune")
        print()
        print("Strategies: best, latest, random, score_prop, score_child_prop (default)")
        print()
        print("Parent selection uses Hyperagents' exact algorithms:")
        print("  score_prop:       sigmoid(10 * (score - top3_mean)) / sum")
        print("  score_child_prop: sigmoid * exp(-(children/8)^3) / sum")
        sys.exit(0)

    exp_root = sys.argv[1]
    cmd = sys.argv[2] if len(sys.argv) > 2 else "stats"

    if cmd == "init":
        cmd_init(exp_root)
    elif cmd == "select-parent":
        strategy = sys.argv[3] if len(sys.argv) > 3 else "score_child_prop"
        cmd_select_parent(exp_root, strategy)
    elif cmd == "add":
        if len(sys.argv) < 4:
            print(json.dumps({"error": "Missing JSON argument"}))
            sys.exit(1)
        cmd_add(exp_root, sys.argv[3])
    elif cmd == "lineage":
        if len(sys.argv) < 4:
            print(json.dumps({"error": "Missing genid argument"}))
            sys.exit(1)
        cmd_lineage(exp_root, sys.argv[3])
    elif cmd == "stats":
        cmd_stats(exp_root)
    elif cmd == "best":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        cmd_best(exp_root, n)
    elif cmd == "update-fitness":
        if len(sys.argv) < 5:
            print(json.dumps({"error": "Usage: update-fitness <genid> <new_fitness> [best_exp_id] [best_hp_config_json]"}))
            sys.exit(1)
        best_exp = sys.argv[5] if len(sys.argv) > 5 else None
        best_hp = json.loads(sys.argv[6]) if len(sys.argv) > 6 else None
        cmd_update_fitness(exp_root, sys.argv[3], float(sys.argv[4]), best_exp, best_hp)
    elif cmd == "operator-stats":
        cmd_operator_stats(exp_root)
    elif cmd == "prune":
        cmd_prune(exp_root)
    else:
        print(json.dumps({"error": f"Unknown command: {cmd}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
