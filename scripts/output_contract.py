#!/usr/bin/env python3
"""Per-agent output contracts — single source of truth.

Defines the required output files for each pipeline agent and shares them between
SubagentStart (injection) and SubagentStop (verification). Supports
glob/dir/any_of/required_if entries for mode-dependent and conditional outputs.

CLI: python3 output_contract.py inject|check <exp_root> <agent_name>
     [--round-dir X] [--exp-id X]   (check exits 2 if any contracted output is missing)
"""

import glob as glob_mod
import json
import os
import sys

# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

CONTRACTS = {
    "prerequisites-agent": [
        {
            "path": "{exp_root}/results/prerequisites.json",
            "description": "Prerequisites check report",
        },
        {
            "path": "{exp_root}/prepared-data/",
            "dir": True,
            "description": "Prepared dataset directory (required when dataset.prepared is true)",
            "required_if": {
                "file": "{exp_root}/results/prerequisites.json",
                "jsonpath": "dataset.prepared",
                "equals": True,
            },
        },
    ],
    "baseline-agent": [
        {
            "path": "{exp_root}/results/baseline.json",
            "description": "Baseline metrics + GPU profiling",
        },
        {
            "path": "{exp_root}/logs/baseline/train.log",
            "description": "Baseline training log",
        },
    ],
    "research-agent": [
        {
            "path": "{exp_root}/reports/research-findings*.md",
            "description": "Research findings (glob)",
            "glob": True,
        },
    ],
    "implement-agent": [
        {
            "path": "{exp_root}/results/implementation-manifest.json",
            "description": "Validated proposal branches",
        },
    ],
    "tuning-agent": [
        {
            "path": "{exp_root}/proposed-configs/{round_dir}/",
            "description": "Proposed HP configs directory",
            "dir": True,
        },
    ],
    "experiment-agent": [
        {
            "path": "{exp_root}/results/{round_dir}/{exp_id}.json",
            "description": "Experiment result JSON",
        },
        {
            "path": "{exp_root}/logs/{round_dir}/{exp_id}/train.log",
            "description": "Training log",
        },
        {
            "path": "{exp_root}/scripts/{round_dir}/{exp_id}/",
            "description": "Training script directory",
            "dir": True,
        },
        {
            "path": "{exp_root}/artifacts/{round_dir}/{exp_id}/",
            "description": "Checkpoint artifacts directory",
            "dir": True,
        },
    ],
    "analysis-agent": [
        {
            "any_of": [
                "{exp_root}/reports/batch-*-analysis.md",
                "{exp_root}/reports/session-review.md",
            ],
            "description": (
                "Batch analysis report (Phase 7/8 batch mode) OR session review "
                "(Phase 9 review mode — activated by scope='session')"
            ),
        },
    ],
    "report-agent": [
        {
            "path": "{exp_root}/reports/final-report.md",
            "description": "Final optimization report",
        },
        {
            "path": "{exp_root}/reports/progress_chart.png",
            "description": "Progress chart",
        },
    ],
}

# ---------------------------------------------------------------------------
# Schema examples (abbreviated) for injection text
# ---------------------------------------------------------------------------

SCHEMA_EXAMPLES = {
    "experiment-agent": """{
  "exp_id": "<exp_id>", "status": "completed",
  "config": {"lr": 0.001, ...}, "metrics": {"loss": 0.5, ...},
  "iteration": 1, "method_tier": "baseline", "duration_seconds": 120.0
}""",
    "baseline-agent": """{
  "exp_id": "baseline", "status": "completed",
  "config": {...}, "metrics": {...}, "profiling": {...}
}""",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _format_path(path_template, exp_root, **kwargs):
    """Format a path template, keeping a missing round_dir/exp_id as its {placeholder}
    (so callers see what wasn't filled and glob matching can substitute * later)."""
    try:
        return path_template.format(exp_root=exp_root, **kwargs)
    except KeyError:
        return path_template.format(exp_root=exp_root, **{
            k: "{" + k + "}" for k in ("round_dir", "exp_id")
            if k not in kwargs
        })


def _condition_satisfied(required_if, exp_root, **kwargs):
    """Evaluate a `required_if` predicate: read the referenced JSON, navigate the dotted
    jsonpath, compare to `equals`. Any read/parse/navigation failure returns False
    (can't evaluate → default to "not required" rather than blocking)."""
    ref_file = _format_path(required_if["file"], exp_root, **kwargs)
    if not os.path.isfile(ref_file):
        return False
    try:
        with open(ref_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    for key in required_if["jsonpath"].split("."):
        if not isinstance(data, dict) or key not in data:
            return False
        data = data[key]
    return data == required_if["equals"]


def get_contract(agent_name, exp_root, **kwargs):
    """Resolve an agent's contract to dicts with `path`, `description`, and optional
    `glob`/`dir`/`required_if` — or an `any_of` list (at least one must exist).
    Returns [] for unknown agents.
    """
    templates = CONTRACTS.get(agent_name)
    if not templates:
        return []

    resolved = []
    for entry in templates:
        item = {"description": entry["description"]}
        if "any_of" in entry:
            item["any_of"] = [
                _format_path(p, exp_root, **kwargs) for p in entry["any_of"]
            ]
        else:
            item["path"] = _format_path(entry["path"], exp_root, **kwargs)
            if entry.get("glob"):
                item["glob"] = True
            if entry.get("dir"):
                item["dir"] = True
        if "required_if" in entry:
            item["required_if"] = entry["required_if"]
        resolved.append(item)
    return resolved


def format_injection(agent_name, exp_root, **kwargs):
    """Format the contract as human-readable SubagentStart injection text ("" if unknown)."""
    contract = get_contract(agent_name, exp_root, **kwargs)
    if not contract:
        return ""

    lines = [f"REQUIRED OUTPUTS for {agent_name}:"]
    for idx, entry in enumerate(contract, 1):
        conditional = ""
        if "required_if" in entry:
            cond = entry["required_if"]
            conditional = (
                f" [conditional: only required if {cond['jsonpath']} == "
                f"{cond['equals']!r} in {cond['file'].split('/')[-1]}]"
            )
        if "any_of" in entry:
            lines.append(f"  {idx}. AT LEAST ONE of:{conditional}")
            for alt in entry["any_of"]:
                lines.append(f"       - {alt}")
        else:
            lines.append(f"  {idx}. {entry['path']}{conditional}")
        lines.append(f"     — {entry['description']}")
    lines.append("")

    # Append schema example if available
    example = SCHEMA_EXAMPLES.get(agent_name)
    if example:
        lines.append("JSON schema example:")
        lines.append(example)
        lines.append("")

    lines.append("The PreToolUse hook will BLOCK writes that don't match the schema or path.")
    lines.append("The SubagentStop hook will BLOCK you from finishing if any output is missing.")
    lines.append(
        f"\nBefore finishing, append a brief summary of what you did to the running log:\n"
        f"  python3 ${{CLAUDE_PLUGIN_ROOT}}/scripts/dev_notes.py {exp_root} append {agent_name} '<brief summary>'"
    )
    return "\n".join(lines)


def check_outputs(agent_name, exp_root, **kwargs):
    """Verify all contracted outputs exist -> {"complete", "missing", "found"}.

    Unknown agents have no contract (always complete). Missing round_dir/exp_id
    placeholders become `*` wildcards so glob matching can still find a match.
    """
    contract = get_contract(agent_name, exp_root, **kwargs)
    if not contract:
        return {"complete": True, "missing": [], "found": []}

    import re

    def _resolve_glob(path):
        """Replace unresolved {placeholders} with * for glob matching."""
        if "{" in path:
            return re.sub(r"\{[^}]+\}", "*", path), True
        return path, False

    def _exists(path, is_dir=False, force_glob=False):
        """Check if a path exists (file/dir, optionally via glob)."""
        resolved, had_placeholder = _resolve_glob(path)
        use_glob = force_glob or had_placeholder or "*" in resolved
        if use_glob:
            matches = glob_mod.glob(resolved)
            if is_dir:
                matches = [m for m in matches if os.path.isdir(m)]
            return bool(matches)
        if is_dir:
            return os.path.isdir(resolved)
        return os.path.isfile(resolved)

    missing = []
    found = []
    for entry in contract:
        # required_if: skip the entry entirely if the predicate is false
        if "required_if" in entry:
            if not _condition_satisfied(entry["required_if"], exp_root, **kwargs):
                continue

        if "any_of" in entry:
            alternatives = entry["any_of"]
            # any_of satisfied if at least one alternative exists
            if any(_exists(alt, force_glob=True) for alt in alternatives):
                found.append(" | ".join(alternatives))
            else:
                missing.append(" | ".join(alternatives))
            continue

        path = entry["path"]
        is_dir = entry.get("dir", False)
        force_glob = entry.get("glob", False)
        if _exists(path, is_dir=is_dir, force_glob=force_glob):
            found.append(entry["path"])
        else:
            missing.append(entry["path"])

    return {"complete": len(missing) == 0, "missing": missing, "found": found}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: output_contract.py inject|check <exp_root> <agent_name>"
            " [--round-dir X] [--exp-id X]"
        )
        sys.exit(1)

    cmd, exp_root, agent_name = sys.argv[1], sys.argv[2], sys.argv[3]
    kwargs = {}
    i = 4
    while i < len(sys.argv) - 1:
        key = sys.argv[i].lstrip("-").replace("-", "_")
        kwargs[key] = sys.argv[i + 1]
        i += 2

    if cmd == "inject":
        text = format_injection(agent_name, exp_root, **kwargs)
        if text:
            print(text)
    elif cmd == "check":
        result = check_outputs(agent_name, exp_root, **kwargs)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["complete"] else 2)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
