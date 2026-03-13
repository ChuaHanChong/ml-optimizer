#!/usr/bin/env bash
# PreCompact hook: log pipeline state persistence before context compaction.
# Outputs a reminder message that gets injected into Claude's context post-compaction.

set -euo pipefail

INPUT=$(cat)
CWD=$(echo "$INPUT" | jq -r '.cwd // empty' 2>/dev/null)

if [ -z "$CWD" ]; then
  exit 0
fi

STATE_FILE="$CWD/experiments/pipeline-state.json"

if [ -f "$STATE_FILE" ]; then
  TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "## $TIMESTAMP — Pre-compaction checkpoint" >> "$CWD/experiments/dev_notes.md"
  echo "Pipeline state file exists at experiments/pipeline-state.json. Read it after compaction to restore context." >> "$CWD/experiments/dev_notes.md"
  echo "" >> "$CWD/experiments/dev_notes.md"
  echo "REMINDER: Pipeline state persisted at $STATE_FILE. Read experiments/pipeline-state.json to restore phase, metric, and budget context."
fi

exit 0
