#!/usr/bin/env bash
# File path guardrail hook for ml-optimizer autonomous mode.
# Prevents writes outside the project directory and protects critical files.
# Exit 0 = allow, Exit 2 = block.

set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null)

if [ -z "$FILE_PATH" ]; then
  exit 0
fi

# Resolve to absolute path for comparison
ABS_PATH=$(realpath -m "$FILE_PATH" 2>/dev/null || echo "$FILE_PATH")

# Allow writes to plugin directory (hooks, skills, etc.)
PLUGIN_DIR="${CLAUDE_PLUGIN_ROOT:-}"
if [ -n "$PLUGIN_DIR" ]; then
  PLUGIN_ABS=$(realpath -m "$PLUGIN_DIR" 2>/dev/null || echo "$PLUGIN_DIR")
  case "$ABS_PATH" in
    "$PLUGIN_ABS"/*)
      exit 0
      ;;
  esac
fi

# Block writes to .git/ internals
if echo "$ABS_PATH" | grep -qE '/\.git/'; then
  echo '{"decision":"block","reason":"Blocked: writing to .git/ internals can corrupt the repository"}' >&2
  exit 2
fi

# Block writes to credential/secret files
BASENAME=$(basename "$ABS_PATH")
case "$BASENAME" in
  .env|.env.*|credentials*|*secret*|*.pem|*.key|id_rsa*|id_ed25519*)
    echo "{\"decision\":\"block\",\"reason\":\"Blocked: writing to sensitive file '$BASENAME' — potential credential exposure\"}" >&2
    exit 2
    ;;
esac

# Block lock file modifications
case "$BASENAME" in
  package-lock.json|poetry.lock|yarn.lock|pnpm-lock.yaml|Pipfile.lock)
    echo "{\"decision\":\"block\",\"reason\":\"Blocked: writing to lock file '$BASENAME' — use the package manager instead\"}" >&2
    exit 2
    ;;
esac

# All checks passed
exit 0
