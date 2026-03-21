#!/usr/bin/env bash
# Stop hook: verify final report exists if experiments were run.
# Exit 0 = allow stop, Exit 2 = block stop.

set -euo pipefail

INPUT=$(cat)
CWD=$(echo "$INPUT" | jq -r '.cwd // empty' 2>/dev/null)

if [ -z "$CWD" ]; then
  exit 0
fi

STATE_FILE="$CWD/experiments/pipeline-state.json"

# If no pipeline state, this isn't an ml-optimizer session — allow stop
if [ ! -f "$STATE_FILE" ]; then
  exit 0
fi

# If stop_hook_active flag is set, allow stop to avoid infinite loops
STOP_ACTIVE=$(jq -r '.stop_hook_active // false' "$STATE_FILE" 2>/dev/null)
if [ "$STOP_ACTIVE" = "true" ]; then
  exit 0
fi

# Check if any experiment results exist
RESULTS_DIR="$CWD/experiments/results"
EXP_COUNT=0
if [ -d "$RESULTS_DIR" ]; then
  EXP_COUNT=$(find "$RESULTS_DIR" -name 'exp-*.json' -type f 2>/dev/null | wc -l)
fi

# If no experiments were run, allow stop
if [ "$EXP_COUNT" -eq 0 ]; then
  exit 0
fi

# Experiments exist — check for final report
REPORT="$CWD/experiments/reports/final-report.md"
if [ ! -f "$REPORT" ]; then
  echo '{"decision":"block","reason":"Final report not generated. Run the report skill before stopping."}' >&2
  exit 2
fi

# All good
exit 0
