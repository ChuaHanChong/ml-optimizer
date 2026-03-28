#!/usr/bin/env bash
# Shared helper: find the experiments root directory.
# Strategy (in order):
#   1. Read from .claude/ml-optimizer.json (breadcrumb written by orchestrator at Phase 0)
#   2. Walk up parent directories looking for experiments/pipeline-state.json
#   3. Check $CWD/experiments/ directly
# Prints the exp_root path to stdout, or nothing if not found.

CWD="${1:-.}"

# Strategy 1: breadcrumb file
DIR="$CWD"
while [ "$DIR" != "/" ]; do
  BREADCRUMB="$DIR/.claude/ml-optimizer.json"
  if [ -f "$BREADCRUMB" ]; then
    EXP=$(python3 -c "import json; print(json.load(open('$BREADCRUMB')).get('exp_root',''))" 2>/dev/null || true)
    if [ -n "$EXP" ] && [ -d "$EXP" ]; then
      echo "$EXP"
      exit 0
    fi
  fi
  DIR=$(dirname "$DIR")
done

# Strategy 2: walk up looking for experiments/pipeline-state.json
DIR="$CWD"
while [ "$DIR" != "/" ]; do
  if [ -f "$DIR/experiments/pipeline-state.json" ]; then
    echo "$DIR/experiments"
    exit 0
  fi
  DIR=$(dirname "$DIR")
done

# Not found
exit 0
