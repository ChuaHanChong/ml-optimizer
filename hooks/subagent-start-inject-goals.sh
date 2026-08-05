#!/usr/bin/env bash
# SubagentStart hook: inject goal memory + output contract for ml-optimizer agents.
# Ensures every agent sees (1) available optimization goals and (2), for agents
# with a defined output contract in output_contract.py, its required output
# paths/schemas (monitor-agent has none and is exempt).
set -euo pipefail

INPUT=$(cat)
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
[ -z "$CWD" ] && exit 0

EXP_DIR=$("${CLAUDE_PLUGIN_ROOT}/hooks/find-exp-root.sh" "$CWD")
[ -z "$EXP_DIR" ] && exit 0

SCRIPTS="${CLAUDE_PLUGIN_ROOT}/scripts"

# 1. Inject goal memory summary
if [ -f "$EXP_DIR/optimization-goals.json" ] && [ -f "$SCRIPTS/goal_memory.py" ]; then
  SUMMARY=$(python3 "$SCRIPTS/goal_memory.py" "$EXP_DIR" summary 2>/dev/null || true)
  if [ -n "$SUMMARY" ]; then
    echo "$SUMMARY"
  fi
fi

# 2. Inject output contract (tells agent exactly what files to produce)
AGENT_TYPE=$(echo "$INPUT" | jq -r '.agent_type // .subagent_type // empty')
if [ -n "$AGENT_TYPE" ] && [ -f "$SCRIPTS/output_contract.py" ]; then
  # Strip ml-optimizer: prefix
  AGENT_TYPE=$(echo "$AGENT_TYPE" | sed 's/ml-optimizer://')
  CONTRACT=$(python3 "$SCRIPTS/output_contract.py" inject "$EXP_DIR" "$AGENT_TYPE" 2>/dev/null || true)
  if [ -n "$CONTRACT" ]; then
    echo ""
    echo "$CONTRACT"
  fi
fi

# Note: the output contract (injected via output_contract.py above) tells the
# agent to call dev_notes.py append, which embeds the --agent-id as an
# <!-- agent_id: ... --> HTML comment. Nothing currently reads that comment
# back at SubagentStop — output_contract.py's CONTRACTS dict has no
# dev_notes.md entry, and validate_agent_output.py's check_agent_output()
# accepts an agent_id parameter but never uses it. The comment is written for
# future/manual verification, not automatically enforced today.

exit 0
