#!/usr/bin/env bash
cat <<'EOF'
{"decision":"approve","systemMessage":"A sub-agent just completed. If it produced output files (experiments/results/*.json, experiments/reports/*.md), validate them with schema_validator.py. If validation fails, fix the output before proceeding."}
EOF
