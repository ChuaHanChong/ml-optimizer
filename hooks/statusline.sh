#!/usr/bin/env bash
# ml-optimizer status line — shows phase, experiment count, best metric, GPU utilization.
# Reads pipeline-state.json and experiment results from the current working directory.
# Exits silently if no ml-optimizer session is active.

set -uo pipefail

# Check for required tools
HAS_BC=false
command -v bc &>/dev/null && HAS_BC=true

# Read workspace from stdin JSON (Claude Code provides {"cwd": "..."})
INPUT=$(cat)
CWD=$(echo "$INPUT" | jq -r '.cwd // empty' 2>/dev/null)
[ -z "$CWD" ] && exit 0

STATE_FILE="$CWD/experiments/pipeline-state.json"
[ -f "$STATE_FILE" ] || exit 0

# Parse pipeline state
PHASE=$(jq -r '.phase // "?"' "$STATE_FILE" 2>/dev/null)
ITER=$(jq -r '.iteration // "?"' "$STATE_FILE" 2>/dev/null)
METRIC=$(jq -r '.user_choices.primary_metric // "loss"' "$STATE_FILE" 2>/dev/null)
LOWER=$(jq -r '.user_choices.lower_is_better // true' "$STATE_FILE" 2>/dev/null)

# Count experiments
RESULTS_DIR="$CWD/experiments/results"
COMPLETED=0
RUNNING=0
TOTAL=0
if [ -d "$RESULTS_DIR" ]; then
    for f in "$RESULTS_DIR"/exp-*.json; do
        [ -f "$f" ] || continue
        TOTAL=$((TOTAL + 1))
        STATUS=$(jq -r '.status // ""' "$f" 2>/dev/null)
        [ "$STATUS" = "completed" ] && COMPLETED=$((COMPLETED + 1))
        [ "$STATUS" = "running" ] && RUNNING=$((RUNNING + 1))
    done
fi

# Budget info
BUDGET=""
BUDGET_MODE=$(jq -r '.user_choices.budget_mode // ""' "$STATE_FILE" 2>/dev/null)
if [ "$BUDGET_MODE" = "autonomous" ]; then
    BUDGET="auto"
else
    DIFF_MULT=$(jq -r '.user_choices.difficulty_multiplier // ""' "$STATE_FILE" 2>/dev/null)
    [ -n "$DIFF_MULT" ] && BUDGET="$DIFF_MULT"
fi

# Best metric from completed experiments
BEST=""
if [ -d "$RESULTS_DIR" ] && [ "$COMPLETED" -gt 0 ]; then
    BASELINE_FILE="$RESULTS_DIR/baseline.json"
    BASELINE_VAL=""
    [ -f "$BASELINE_FILE" ] && BASELINE_VAL=$(jq -r ".metrics.$METRIC // empty" "$BASELINE_FILE" 2>/dev/null)

    # Find best value
    BEST_VAL=""
    for f in "$RESULTS_DIR"/exp-*.json; do
        [ -f "$f" ] || continue
        S=$(jq -r '.status // ""' "$f" 2>/dev/null)
        [ "$S" = "completed" ] || continue
        V=$(jq -r ".metrics.$METRIC // empty" "$f" 2>/dev/null)
        [ -z "$V" ] && continue
        if [ -z "$BEST_VAL" ]; then
            BEST_VAL="$V"
        elif [ "$HAS_BC" = "true" ]; then
            if [ "$LOWER" = "true" ]; then
                BETTER=$(echo "$V < $BEST_VAL" | bc -l 2>/dev/null)
            else
                BETTER=$(echo "$V > $BEST_VAL" | bc -l 2>/dev/null)
            fi
            [ "$BETTER" = "1" ] && BEST_VAL="$V"
        fi
    done

    if [ -n "$BEST_VAL" ]; then
        # Format with 4 decimal places
        BEST_FMT=$(printf "%.4f" "$BEST_VAL" 2>/dev/null || echo "$BEST_VAL")
        DELTA=""
        if [ -n "$BASELINE_VAL" ] && [ "$BASELINE_VAL" != "0" ] && [ "$HAS_BC" = "true" ]; then
            if [ "$LOWER" = "true" ]; then
                PCT=$(echo "scale=1; ($BASELINE_VAL - $BEST_VAL) / $BASELINE_VAL * 100" | bc -l 2>/dev/null || echo "")
            else
                PCT=$(echo "scale=1; ($BEST_VAL - $BASELINE_VAL) / $BASELINE_VAL * 100" | bc -l 2>/dev/null || echo "")
            fi
            [ -n "$PCT" ] && DELTA=" (${PCT}%)"
        fi
        BEST="best: ${BEST_FMT}${DELTA}"
    fi
fi

# GPU status (single nvidia-smi call)
GPU_INFO=""
if command -v nvidia-smi &>/dev/null; then
    GPU_LINE=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$GPU_LINE" ]; then
        GPU_UTIL=$(echo "$GPU_LINE" | cut -d',' -f1 | tr -d ' ')
        GPU_MEM_USED=$(echo "$GPU_LINE" | cut -d',' -f2 | tr -d ' ')
        GPU_MEM_TOTAL=$(echo "$GPU_LINE" | cut -d',' -f3 | tr -d ' ')
        # Convert MiB to G
        if [ "$HAS_BC" = "true" ]; then
            GPU_MEM_USED_G=$(echo "scale=0; $GPU_MEM_USED / 1024" | bc 2>/dev/null || echo "?")
            GPU_MEM_TOTAL_G=$(echo "scale=0; $GPU_MEM_TOTAL / 1024" | bc 2>/dev/null || echo "?")
        else
            GPU_MEM_USED_G=$((GPU_MEM_USED / 1024))
            GPU_MEM_TOTAL_G=$((GPU_MEM_TOTAL / 1024))
        fi
        GPU_INFO="GPU0: ${GPU_UTIL}% ${GPU_MEM_USED_G}G/${GPU_MEM_TOTAL_G}G"
    fi
fi

# Build status line
PARTS=("[ml-opt] P${PHASE} I${ITER}")

EXP_PART="${COMPLETED}/${BUDGET:-$TOTAL} exp"
[ "$RUNNING" -gt 0 ] && EXP_PART="${EXP_PART} (${RUNNING} running)"
PARTS+=("$EXP_PART")

[ -n "$BEST" ] && PARTS+=("$BEST")
[ -n "$GPU_INFO" ] && PARTS+=("$GPU_INFO")

# Join with " | "
OUTPUT=""
for p in "${PARTS[@]}"; do
    [ -n "$OUTPUT" ] && OUTPUT="$OUTPUT | "
    OUTPUT="$OUTPUT$p"
done

echo "$OUTPUT"
