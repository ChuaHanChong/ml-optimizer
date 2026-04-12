#!/usr/bin/env bash
# PreCompact hook: log pipeline state persistence before context compaction.
# Outputs a reminder message that gets injected into Claude's context post-compaction.

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

if [ -f "$STATE_FILE" ]; then
  TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "## $TIMESTAMP — Pre-compaction checkpoint" >> "$EXP_DIR/dev_notes.md"
  echo "Pipeline state file exists at $STATE_FILE. Read it after compaction to restore context." >> "$EXP_DIR/dev_notes.md"
  echo "" >> "$EXP_DIR/dev_notes.md"
  echo "REMINDER: Pipeline state persisted at $STATE_FILE. Read it to restore phase, metric, and budget context."
fi

exit 0
