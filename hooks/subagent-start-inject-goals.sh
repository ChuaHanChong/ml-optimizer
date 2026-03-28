#!/usr/bin/env bash
# SubagentStart hook: inject goal memory summary when any ml-optimizer agent starts.
# This ensures every agent sees the optimization goals without relying solely on orchestrator relay.
set -euo pipefail

INPUT=$(cat)
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
[ -z "$CWD" ] && exit 0

EXP_DIR=$("${CLAUDE_PLUGIN_ROOT}/hooks/find-exp-root.sh" "$CWD")
[ -z "$EXP_DIR" ] && exit 0
[ ! -f "$EXP_DIR/optimization-goals.json" ] && exit 0

SCRIPTS="${CLAUDE_PLUGIN_ROOT}/scripts"
if [ -f "$SCRIPTS/goal_memory.py" ]; then
  SUMMARY=$(python3 "$SCRIPTS/goal_memory.py" "$EXP_DIR" summary 2>/dev/null || true)
  if [ -n "$SUMMARY" ]; then
    echo "$SUMMARY"
  fi
fi

exit 0
