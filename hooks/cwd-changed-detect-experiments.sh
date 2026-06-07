#!/usr/bin/env bash
# CwdChanged hook: auto-detect when user enters a project with existing experiments.
# Outputs a context message so Claude knows about the existing optimization state.
set -euo pipefail

INPUT=$(cat)
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
[ -z "$CWD" ] && exit 0

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
EXP_DIR=$("$PLUGIN_ROOT/hooks/find-exp-root.sh" "$CWD")
[ -z "$EXP_DIR" ] && exit 0
[ ! -f "$EXP_DIR/pipeline-state.json" ] && exit 0

python3 -c "
import json
state = json.load(open('$EXP_DIR/pipeline-state.json'))
phase = state.get('phase', '?')
iteration = state.get('iteration', 0)
metric = state.get('user_choices', {}).get('primary_metric', '?')
# phase == 9 means the full pipeline (Phase 0→9) completed. For a new
# optimization direction on the same project, the user should configure
# exp_root to a new subdirectory — the plugin does not run-namespace state
# files (learned-behaviors, round manifest would
# collide). For phase < 9 the existing state represents a mid-run pause
# that can be resumed.
if phase == 9:
    print(f'[ml-optimizer] Completed optimization found at $EXP_DIR (phase 9, metric={metric}). '
          f'To run a new optimization direction, point exp_root at a new, dedicated directory '
          f'(any path — the plugin has no hardcoded output location). '
          f'See $EXP_DIR/reports/final-report.md for the completed run.')
else:
    print(f'[ml-optimizer] Existing optimization found at $EXP_DIR: phase={phase}, iteration={iteration}, metric={metric}. Use /optimize to resume.')
" 2>/dev/null || true

exit 0
