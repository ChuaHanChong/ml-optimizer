#!/usr/bin/env bash
cat <<'EOF'
{"decision":"approve","systemMessage":"A sub-agent just completed. If it produced output files (experiments/results/*.json, experiments/reports/*.md), validate them: python3 ~/.claude/plugins/ml-optimizer/scripts/schema_validator.py <file> <type>. If validation fails, fix the output before proceeding."}
EOF
