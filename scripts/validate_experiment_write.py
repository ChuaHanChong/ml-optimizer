#!/usr/bin/env python3
"""PreToolUse hook: validates Write/Edit operations to experiment result files.

Invoked by the Claude Code hook runner on the PreToolUse event for Write/Edit, not
as a CLI. It reads the hook payload as JSON on stdin (tool_name, tool_input with
file_path and content), resolves exp_root from the .claude/ml-optimizer.json
breadcrumb, and enforces that exp-*.json results live in round-N-<type>/ directories
(and proposed configs in proposed-configs/round-N-<type>/), that root-level files
(baseline/prerequisites/manifest/rounds-manifest) sit directly in results/, plus
schema, completeness, and goal-compliance (frozen params, OOM limits) checks. Prints
a JSON decision ({"decision": "approve"|"block", "reason": ...}) on stdout.
"""

import json
import re
import sys
from pathlib import Path

# Import schema validators from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from schema_validator import (
    validate_baseline,
    validate_manifest,
    validate_prerequisites,
    validate_result,
)

# Round directory name pattern: round-<N>-<type>
ROUND_DIR_PATTERN = re.compile(r"^round-\d+-(?:hp|evolved|research|stacked|meta)$")


def approve(reason: str | None = None) -> None:
    """Print approval response and exit."""
    resp = {"decision": "approve"}
    if reason:
        resp["reason"] = reason
    print(json.dumps(resp))
    sys.exit(0)


def block(reason: str) -> None:
    """Print block response and exit."""
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(0)


def _find_subdir_segment(file_path: str, subdir_name: str) -> tuple[list[str], int] | None:
    """If `file_path` is at `<exp_root>/<subdir_name>/...`, return
    `(path_parts, index_of_subdir)`, else `None`.

    exp_root comes from the `.claude/ml-optimizer.json` breadcrumb (the
    authoritative source, written by the orchestrator at Phase 0) — there is
    no directory-name fallback, so tests MUST create that breadcrumb.
    """
    exp_root_str = _find_exp_root(file_path)
    if exp_root_str is None:
        return None

    exp_root = Path(exp_root_str).resolve()
    parts = Path(file_path).resolve().parts
    exp_root_depth = len(exp_root.parts)
    if exp_root_depth >= len(parts):
        return None
    if parts[:exp_root_depth] != exp_root.parts:
        return None
    if parts[exp_root_depth] != subdir_name:
        return None
    return list(parts), exp_root_depth


def _find_results_segment(file_path: str) -> tuple[list[str], int] | None:
    """Find the `<exp_root>/results` segment. Returns `(parts, idx)` or None."""
    return _find_subdir_segment(file_path, "results")


def _find_proposed_configs_segment(file_path: str) -> tuple[list[str], int] | None:
    """Find the `<exp_root>/proposed-configs` segment. Returns `(parts, idx)` or None."""
    return _find_subdir_segment(file_path, "proposed-configs")


def _get_content(hook_input: dict, tool_name: str) -> str | None:
    """Extract file content from hook input, if available.

    Returns the content string for Write calls, None for Edit calls.
    """
    if tool_name == "Write":
        return hook_input.get("tool_input", {}).get("content")
    # Edit tool provides a diff, not full content -- skip content validation
    return None


def _parse_content(content: str) -> dict | None:
    """Parse JSON content string. Returns dict or None on failure."""
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _validate_schema(data: dict, schema_type: str) -> dict:
    """Run the appropriate schema validator."""
    if schema_type == "result":
        return validate_result(data)
    elif schema_type == "baseline":
        return validate_baseline(data)
    elif schema_type == "manifest":
        return validate_manifest(data)
    elif schema_type == "prerequisites":
        return validate_prerequisites(data)
    elif schema_type == "proposal":
        # Lightweight check: just verify exp_id and config exist
        errors = []
        if "exp_id" not in data:
            errors.append("Missing required field: exp_id")
        if "config" not in data:
            errors.append("Missing required field: config")
        if "random_seed" in data and (
            isinstance(data["random_seed"], bool) or not isinstance(data["random_seed"], int)
        ):
            errors.append("'random_seed' must be an integer")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": []}
    return {"valid": True, "errors": [], "warnings": []}


def _is_directly_in_results(parts: list[str], results_idx: int) -> bool:
    """Check if the file is directly inside <exp_root>/results/ (no subdirectory)."""
    # parts[results_idx] == "results", so the file is at parts[results_idx + 1]
    return len(parts) == results_idx + 2


def _is_in_round_dir(parts: list[str], results_idx: int) -> tuple[bool, str | None]:
    """Check if the file is inside a round-N-<type>/ subdirectory of results/.

    Returns (is_in_round, round_dir_name_or_None).
    """
    if len(parts) >= results_idx + 3:
        candidate = parts[results_idx + 1]
        if ROUND_DIR_PATTERN.match(candidate):
            return True, candidate
    return False, None


def _check_completeness(data: dict) -> str | None:
    """Check mandatory completeness fields based on status.

    Returns a block reason string if incomplete, None if OK.
    Placeholder writes (status: running/pending) are always allowed.
    """
    status = data.get("status", "")
    if status in ("running", "pending"):
        return None

    if status == "completed":
        missing = []
        if "iteration" not in data:
            missing.append("iteration")
        if "method_tier" not in data:
            missing.append("method_tier")
        if "duration_seconds" not in data:
            missing.append("duration_seconds")
        if "eval_protocol" not in data:
            missing.append("eval_protocol (held_out_eval|train_report|rl_final_eval)")
        tier = data.get("method_tier", "")
        if isinstance(tier, str) and tier.startswith("stacked_"):
            if "code_branches" not in data:
                missing.append("code_branches")
            if "stacking_order" not in data:
                missing.append("stacking_order")
        if missing:
            return f"Completed experiment missing mandatory fields: {', '.join(missing)}"

    if status in ("failed", "diverged"):
        if "notes" not in data:
            return f"{status.capitalize()} experiment must include 'notes' explaining what went wrong"

    return None


def _find_exp_root(file_path: str) -> str | None:
    """Find exp_root by walking up from file_path for .claude/ml-optimizer.json.

    Intentionally uncached — runs once per hook invocation (short-lived
    subprocess). Do NOT add a module-level cache; cache at the call site with an
    explicit lifetime if a future refactor calls this in a loop.
    """
    path = Path(file_path).resolve()
    for parent in path.parents:
        breadcrumb = parent / ".claude" / "ml-optimizer.json"
        if breadcrumb.is_file():
            try:
                data = json.loads(breadcrumb.read_text())
                # New multi-run format: {"active": "...", "runs": [...]}
                # Old single format:    {"exp_root": "..."}
                exp_root = data.get("active") or data.get("exp_root") or ""
                if exp_root and Path(exp_root).is_dir():
                    return exp_root
            except (json.JSONDecodeError, OSError):
                pass
    return None


def _check_goal_compliance(data: dict, file_path: str) -> str | None:
    """Check config against frozen parameters and OOM limits.

    Returns block reason or None. Gracefully degrades if goals/behaviors
    files don't exist.
    """
    config = data.get("config")
    if not isinstance(config, dict) or not config:
        return None

    status = data.get("status", "")
    if status in ("running", "pending"):
        return None

    exp_root = _find_exp_root(file_path)
    if not exp_root:
        return None

    # Check frozen parameters
    goals_path = Path(exp_root) / "optimization-goals.json"
    if goals_path.is_file():
        try:
            goals = json.loads(goals_path.read_text())
            frozen = goals.get("constraints", {}).get("frozen_parameters", [])
            for fp in frozen:
                if fp in config:
                    return f"Config modifies frozen parameter '{fp}' — this parameter cannot be changed"
        except (json.JSONDecodeError, OSError):
            pass

    # Check OOM batch size
    behaviors_path = Path(exp_root) / "learned-behaviors.json"
    if behaviors_path.is_file():
        try:
            behaviors = json.loads(behaviors_path.read_text())
            for entry in behaviors.get("resource_constraints", []):
                max_bs = entry.get("max_batch_size")
                if max_bs is not None and "batch_size" in config:
                    try:
                        bs = int(config["batch_size"])
                        if bs > int(max_bs):
                            return f"Config batch_size={bs} exceeds OOM limit {max_bs}"
                    except (ValueError, TypeError):
                        pass
        except (json.JSONDecodeError, OSError):
            pass

    return None


def _validate_root_file(
    hook_input: dict, tool_name: str, parts: list[str], results_idx: int,
    filename: str, schema_type: str | None,
) -> None:
    """Enforce a root-level results file: must sit directly in results/, plus an
    optional schema check. Calls block()/approve() (which exit). schema_type=None
    checks location only (e.g. rounds-manifest.json, written by round_manager.py)."""
    if not _is_directly_in_results(parts, results_idx):
        block(f"{filename} must be directly in <exp_root>/results/, not in a subdirectory")
    if schema_type is not None:
        content = _get_content(hook_input, tool_name)
        if content is not None:
            data = _parse_content(content)
            if data is None:
                block(f"{filename} contains invalid JSON")
            result = _validate_schema(data, schema_type)
            if not result["valid"]:
                block(f"{filename} schema validation failed: {'; '.join(result['errors'])}")
    approve()


def validate(hook_input: dict) -> None:
    """Main validation logic."""
    tool_input = hook_input.get("tool_input", {})
    file_path = tool_input.get("file_path", "")
    tool_name = hook_input.get("tool_name", "Write")

    # Check for <exp_root>/proposed-configs/ (top-level, separate from results)
    pc_segment = _find_proposed_configs_segment(file_path)
    if pc_segment is not None:
        pc_parts, pc_idx = pc_segment
        pc_filename = pc_parts[-1]
        if pc_filename.endswith(".json") and pc_filename.startswith("exp-"):
            # Must be inside a round subdirectory
            if len(pc_parts) >= pc_idx + 3:
                subdir = pc_parts[pc_idx + 1]
                if ROUND_DIR_PATTERN.match(subdir):
                    # Validate proposal content
                    content = _get_content(hook_input, tool_name)
                    if content is not None:
                        data = _parse_content(content)
                        if data is None:
                            block(f"Proposed config {pc_filename} contains invalid JSON")
                        result = _validate_schema(data, "proposal")
                        if not result["valid"]:
                            block(f"Proposed config {pc_filename} validation failed: {'; '.join(result['errors'])}")
                        # Goal compliance check
                        goal_reason = _check_goal_compliance(data, file_path)
                        if goal_reason:
                            block(f"Proposed config {pc_filename}: {goal_reason}")
                    approve()
                    return
                else:
                    block(f"Proposed config {pc_filename} is in '{subdir}/' which is not a valid round directory")
            else:
                block(f"Proposed config {pc_filename} must be inside a round subdirectory (e.g., proposed-configs/round-1-hp/)")
        approve()
        return

    # Only validate files under <exp_root>/results/
    segment = _find_results_segment(file_path)
    if segment is None:
        approve()
        return

    parts, results_idx = segment
    filename = parts[-1]

    # Non-JSON files: not our concern
    if not filename.endswith(".json"):
        approve()
        return

    # --- Determine file category and apply rules ---

    # 1. Root-level result files (must sit directly in results/). rounds-manifest
    #    is location-only; it's written by round_manager.py.
    _ROOT_SCHEMAS = {
        "baseline.json": "baseline",
        "prerequisites.json": "prerequisites",
        "implementation-manifest.json": "manifest",
        "rounds-manifest.json": None,
    }
    if filename in _ROOT_SCHEMAS:
        _validate_root_file(
            hook_input, tool_name, parts, results_idx, filename, _ROOT_SCHEMAS[filename]
        )
        return

    # 2. Experiment result files (exp-*.json)
    if filename.startswith("exp-") and filename.endswith(".json"):
        # Result file -- must be in a round directory, not directly in results/
        if _is_directly_in_results(parts, results_idx):
            block(
                f"Experiment result {filename} must be inside a round subdirectory "
                f"(e.g., results/round-1-hp/), not directly in <exp_root>/results/"
            )

        in_round, round_dir = _is_in_round_dir(parts, results_idx)
        if not in_round:
            # It's in some subdirectory but not a valid round directory
            parent_dir = parts[results_idx + 1] if len(parts) > results_idx + 1 else "unknown"
            block(
                f"Experiment result {filename} is in '{parent_dir}/' which does not match "
                f"the round directory pattern (round-N-<type> where type is "
                f"hp|evolved|research|stacked|meta)"
            )

        # Validate result content
        content = _get_content(hook_input, tool_name)
        data = None
        if content is not None:
            data = _parse_content(content)
            if data is None:
                block(f"Experiment result {filename} contains invalid JSON")
            result = _validate_schema(data, "result")
            if not result["valid"]:
                block(
                    f"Experiment result {filename} schema validation failed: "
                    f"{'; '.join(result['errors'])}"
                )

        # Completeness check (block incomplete completed experiments)
        if data is not None:
            completeness_reason = _check_completeness(data)
            if completeness_reason:
                block(f"Experiment result {filename}: {completeness_reason}")

        # Goal compliance check (frozen params, OOM limits)
        if data is not None:
            goal_reason = _check_goal_compliance(data, file_path)
            if goal_reason:
                block(f"Experiment result {filename}: {goal_reason}")

        approve()
        return

    # 3. Unrecognized JSON file under results/ -- approve with a warning
    approve(reason=f"Unrecognized file pattern '{filename}' under <exp_root>/results/ -- no validation rules applied")


def main() -> None:
    """Entry point: read hook input from stdin, validate, output decision."""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            approve()
            return
        hook_input = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        # If we can't parse the hook input, approve rather than block
        approve()
        return

    validate(hook_input)


if __name__ == "__main__":
    main()
