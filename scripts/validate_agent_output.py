#!/usr/bin/env python3
"""SubagentStop hook: verify agents produced their expected output files.

Deterministic command hook. Uses output_contract.py as single source of truth
for per-agent output requirements. Returns {"decision": "approve"} or
{"decision": "block", "reason": "..."}.

Input (stdin): JSON with agent context (cwd, optional agent metadata)
Output (stdout): JSON decision
"""

import json
import sys
from pathlib import Path

# Import contract definitions from single source of truth
sys.path.insert(0, str(Path(__file__).parent))
from output_contract import CONTRACTS, check_outputs


def _find_exp_root(cwd: str) -> str | None:
    """Find exp_root from .claude/ml-optimizer.json breadcrumb."""
    path = Path(cwd).resolve()
    for parent in [path] + list(path.parents):
        breadcrumb = parent / ".claude" / "ml-optimizer.json"
        if breadcrumb.is_file():
            try:
                data = json.loads(breadcrumb.read_text())
                exp_root = data.get("exp_root", "")
                if exp_root and Path(exp_root).is_dir():
                    return exp_root
            except (json.JSONDecodeError, OSError):
                pass
    return None


def check_agent_output(cwd: str, agent_name: str, agent_id: str = "") -> dict:
    """Check if an agent produced its expected output files.

    Uses output_contract.check_outputs() for consistency with SubagentStart injection.
    The ``agent_id`` is used to key per-agent state files (dev_notes mtime marker)
    so parallel subagent dispatches don't clobber each other.
    """
    exp_root = _find_exp_root(cwd)
    if not exp_root:
        return {"decision": "approve"}

    if agent_name not in CONTRACTS:
        return {"decision": "approve"}

    result = check_outputs(agent_name, exp_root)
    if not result["complete"]:
        return {
            "decision": "block",
            "reason": f"Agent {agent_name} did not produce expected output: {', '.join(result['missing'])}",
        }

    # Check dev_notes.md — verify THIS agent invocation wrote the last entry
    # by comparing the embedded agent_id comment.
    if agent_id:
        dev_notes = Path(exp_root) / "dev_notes.md"
        if dev_notes.is_file():
            try:
                from dev_notes import last_agent as _last_agent
                last = _last_agent(exp_root)
                if last.get("agent_id") != agent_id:
                    return {
                        "decision": "block",
                        "reason": (
                            f"Agent {agent_name} did not append to dev_notes.md "
                            f"(last entry's agent_id does not match this invocation). "
                            f"Use: python3 ${{CLAUDE_PLUGIN_ROOT}}/scripts/dev_notes.py "
                            f"<exp_root> append {agent_name} '<message>' --agent-id {agent_id}"
                        ),
                    }
            except ImportError:
                pass  # dev_notes module not available — skip check

    return {"decision": "approve"}


def _extract_agent_name(hook_input: dict) -> str:
    """Extract and normalize agent name from hook input."""
    agent_name = hook_input.get("agent_type", "")
    if not agent_name:
        agent_name = hook_input.get("subagent_type", "")
    if not agent_name:
        desc = hook_input.get("description", "")
        for known in CONTRACTS:
            if known.replace("-agent", "") in desc.lower():
                agent_name = known
                break

    # Strip plugin prefix if present
    if agent_name:
        agent_name = agent_name.replace("ml-optimizer:", "")

    return agent_name


def main() -> None:
    """Entry point: read hook input from stdin, check outputs, output decision."""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            print(json.dumps({"decision": "approve"}))
            return

        hook_input = json.loads(raw)
        cwd = hook_input.get("cwd", "")
        agent_id = hook_input.get("agent_id", "")

        agent_name = _extract_agent_name(hook_input)

        if not agent_name or not cwd:
            print(json.dumps({"decision": "approve"}))
            return

        result = check_agent_output(cwd, agent_name, agent_id=agent_id)
        print(json.dumps(result))

    except (json.JSONDecodeError, OSError):
        print(json.dumps({"decision": "approve"}))


if __name__ == "__main__":
    main()
