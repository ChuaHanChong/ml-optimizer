#!/usr/bin/env python3
"""SubagentStop hook: verify agents produced their expected output files.

Invoked by the hook runner on SubagentStop (not a CLI): reads the hook payload
JSON on stdin (agent_type/subagent_type, agent_id, cwd), resolves exp_root from
the .claude/ml-optimizer.json breadcrumb, checks the agent's output contract via
output_contract.check_outputs(), and prints an approve/block decision on stdout.
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
                exp_root = data.get("active") or data.get("exp_root", "")
                if exp_root and Path(exp_root).is_dir():
                    return exp_root
            except (json.JSONDecodeError, OSError):
                pass
    return None


def check_agent_output(cwd: str, agent_name: str, agent_id: str = "") -> dict:
    """Approve/block based on whether an agent produced its contracted output files."""
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

    if agent_name:
        agent_name = agent_name.replace("ml-optimizer:", "")  # strip plugin prefix

    return agent_name


def main() -> None:
    """Read hook input from stdin, check outputs, print the decision."""
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
