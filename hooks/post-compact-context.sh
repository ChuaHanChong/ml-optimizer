#!/usr/bin/env bash
# SessionStart (compact) hook: re-inject critical pipeline context after compaction.
# Reads pipeline-state.json and outputs a compact summary for Claude's context.

set -euo pipefail

INPUT=$(cat)
CWD=$(echo "$INPUT" | jq -r '.cwd // empty' 2>/dev/null)

if [ -z "$CWD" ]; then
  exit 0
fi

# Resolve exp_root from breadcrumb or walk-up fallback.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
EXP_DIR=$("$PLUGIN_ROOT/hooks/find-exp-root.sh" "$CWD")
if [ -z "$EXP_DIR" ]; then
  exit 0
fi

STATE_FILE="$EXP_DIR/pipeline-state.json"

if [ ! -f "$STATE_FILE" ]; then
  exit 0
fi

# Extract key fields from pipeline state
PHASE=$(jq -r '.phase // "unknown"' "$STATE_FILE" 2>/dev/null)
STATUS=$(jq -r '.status // "unknown"' "$STATE_FILE" 2>/dev/null)
ITERATION=$(jq -r '.iteration // 0' "$STATE_FILE" 2>/dev/null)
STOP_COUNT=$(jq -r '.consecutive_stop_count // 0' "$STATE_FILE" 2>/dev/null)
PRIMARY_METRIC=$(jq -r '.user_choices.primary_metric // "unknown"' "$STATE_FILE" 2>/dev/null)
LOWER_IS_BETTER=$(jq -r '.user_choices.lower_is_better // "unknown"' "$STATE_FILE" 2>/dev/null)
RUNNING=$(jq -r '.running_experiments // [] | length' "$STATE_FILE" 2>/dev/null)

# Count completed experiments
RESULTS_DIR="$EXP_DIR/results"
COMPLETED=0
if [ -d "$RESULTS_DIR" ]; then
  COMPLETED=$(find "$RESULTS_DIR" -name 'exp-*.json' -type f 2>/dev/null | wc -l)
fi

cat <<EOF
ML-OPTIMIZER PIPELINE CONTEXT (restored after compaction):
- Phase: $PHASE | Status: $STATUS | Iteration: $ITERATION
- Primary metric: $PRIMARY_METRIC (lower_is_better=$LOWER_IS_BETTER)
- Consecutive stops: $STOP_COUNT
- Experiments completed: $COMPLETED | Currently running: $RUNNING
- exp_root: $EXP_DIR
- State file: $STATE_FILE
- Read the full state file to restore detailed context before proceeding.
EOF

exit 0
