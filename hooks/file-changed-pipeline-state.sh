#!/usr/bin/env bash
# FileChanged hook: validate pipeline-state.json integrity on every write
# (JSON parseability + baseline checksum). Fires regardless of who/what made
# the change — it flags corruption or a tampered baseline, but cannot tell
# an external edit from the pipeline's own writes.
set -euo pipefail

INPUT=$(cat)
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
[ -z "$CWD" ] && exit 0

EXP_DIR=$("${CLAUDE_PLUGIN_ROOT}/hooks/find-exp-root.sh" "$CWD")
[ -z "$EXP_DIR" ] && exit 0

STATE_FILE="$EXP_DIR/pipeline-state.json"
[ ! -f "$STATE_FILE" ] && exit 0

# Validate JSON is parseable
if ! python3 -c "import json; json.load(open('$STATE_FILE'))" 2>/dev/null; then
  echo "WARNING: pipeline-state.json is corrupt (invalid JSON). The pipeline may not resume correctly." >&2
  exit 0
fi

# Check baseline integrity if checksum exists
SCRIPTS="${CLAUDE_PLUGIN_ROOT}/scripts"
if [ -f "$SCRIPTS/pipeline_state.py" ]; then
  RESULT=$(python3 "$SCRIPTS/pipeline_state.py" "$EXP_DIR" verify-baseline 2>/dev/null || true)
  VALID=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('valid', True))" 2>/dev/null || echo "true")
  if [ "$VALID" = "False" ] || [ "$VALID" = "false" ]; then
    echo "CRITICAL: Baseline metrics checksum mismatch after pipeline-state.json change. Baseline may have been tampered with." >&2
  fi
fi

exit 0
