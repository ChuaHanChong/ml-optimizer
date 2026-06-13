#!/usr/bin/env bash
# Shared helper: resolve the active exp_root directory.
#
# Resolution: walk up parent directories from $CWD looking for a
# .claude/ml-optimizer.json breadcrumb. The breadcrumb is the authoritative
# source of truth for the user's chosen exp_root — it's written by the
# orchestrator at Phase 0 and points at any absolute path the user picked.
# The plugin does not hardcode the output directory name.
#
# Prints the exp_root path to stdout, or nothing if no breadcrumb is found.
# Hooks that call this helper must handle an empty result by exiting
# silently (no ml-optimizer session is active).
#
# There is no directory-name-based fallback. Tests and sandboxes must
# create a .claude/ml-optimizer.json breadcrumb pointing at their
# synthetic exp_root.

CWD="${1:-.}"
DEBUG="${ML_OPT_DEBUG:-}"

DIR="$CWD"
while [ "$DIR" != "/" ]; do
  BREADCRUMB="$DIR/.claude/ml-optimizer.json"
  if [ -f "$BREADCRUMB" ]; then
    # Supports two breadcrumb formats:
    #   New (multi-run): {"active": "/path/to/run-02", "runs": [...]}
    #   Old (single):    {"exp_root": "/path/to/experiments"}
    EXP=$(python3 -c "
import json
d = json.load(open('$BREADCRUMB'))
print(d.get('active') or d.get('exp_root') or '')
" 2>/dev/null || true)
    if [ -n "$EXP" ] && [ -d "$EXP" ]; then
      echo "$EXP"
      exit 0
    fi
  fi
  DIR=$(dirname "$DIR")
done

# Not found
[ "$DEBUG" = "1" ] && echo "[find-exp-root] no .claude/ml-optimizer.json breadcrumb found walking up from $CWD" >&2
exit 0
