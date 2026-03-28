#!/usr/bin/env bash
# CwdChanged hook: auto-detect when user enters a project with existing experiments.
# Outputs a context message so Claude knows about the existing optimization state.
set -euo pipefail

INPUT=$(cat)
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
[ -z "$CWD" ] && exit 0

EXP_DIR=$("${CLAUDE_PLUGIN_ROOT}/hooks/find-exp-root.sh" "$CWD")
[ -z "$EXP_DIR" ] && exit 0
[ ! -f "$EXP_DIR/pipeline-state.json" ] && exit 0

python3 -c "
import json
state = json.load(open('$EXP_DIR/pipeline-state.json'))
phase = state.get('phase', '?')
iteration = state.get('iteration', 0)
metric = state.get('user_choices', {}).get('primary_metric', '?')
print(f'[ml-optimizer] Existing optimization found at $EXP_DIR: phase={phase}, iteration={iteration}, metric={metric}. Use /optimize to resume.')
" 2>/dev/null || true

exit 0
