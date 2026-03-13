#!/usr/bin/env bash
# PostToolUse hook for Bash: detect critical errors (OOM, segfault, disk full)
# in command output and log them to the error tracker.
# Exit 0 always — this is advisory, never blocks.

set -euo pipefail

INPUT=$(cat)
OUTPUT=$(echo "$INPUT" | jq -r '.tool_result.stdout // empty' 2>/dev/null)
STDERR=$(echo "$INPUT" | jq -r '.tool_result.stderr // empty' 2>/dev/null)
CWD=$(echo "$INPUT" | jq -r '.cwd // empty' 2>/dev/null)

# Combine stdout and stderr for pattern matching
COMBINED="$OUTPUT$STDERR"

if [ -z "$COMBINED" ] || [ -z "$CWD" ]; then
  exit 0
fi

EXP_ROOT="$CWD/experiments"
TRACKER="$HOME/.claude/plugins/ml-optimizer/scripts/error_tracker.py"

# Only log if experiments directory exists (we're in an active optimization session)
if [ ! -d "$EXP_ROOT" ]; then
  exit 0
fi

# CUDA OOM
if echo "$COMBINED" | grep -qi 'CUDA out of memory\|RuntimeError: CUDA error\|torch.cuda.OutOfMemoryError'; then
  python3 "$TRACKER" "$EXP_ROOT" log "{\"category\":\"resource_error\",\"severity\":\"critical\",\"source\":\"hook:detect-critical-errors\",\"message\":\"CUDA out of memory detected in Bash output\"}" 2>/dev/null || true
  exit 0
fi

# OOM killer
if echo "$COMBINED" | grep -qE '^Killed$|Out of memory:|oom-kill:'; then
  python3 "$TRACKER" "$EXP_ROOT" log "{\"category\":\"resource_error\",\"severity\":\"critical\",\"source\":\"hook:detect-critical-errors\",\"message\":\"Process killed by OOM killer\"}" 2>/dev/null || true
  exit 0
fi

# Segfault
if echo "$COMBINED" | grep -qi 'Segmentation fault\|SIGSEGV'; then
  python3 "$TRACKER" "$EXP_ROOT" log "{\"category\":\"training_failure\",\"severity\":\"critical\",\"source\":\"hook:detect-critical-errors\",\"message\":\"Segmentation fault detected\"}" 2>/dev/null || true
  exit 0
fi

# Disk full
if echo "$COMBINED" | grep -qi 'No space left on device'; then
  python3 "$TRACKER" "$EXP_ROOT" log "{\"category\":\"resource_error\",\"severity\":\"critical\",\"source\":\"hook:detect-critical-errors\",\"message\":\"No space left on device\"}" 2>/dev/null || true
  exit 0
fi

exit 0
