# Hardening Design: Autonomy, Tests, and Flexibility

**Date:** 2026-03-10
**Status:** Approved
**Scope:** Surgical fixes for critical/medium gaps found in comprehensive audit

## Problem Statement

A deep audit of the ml-optimizer plugin revealed that while the orchestrator was hardened for autonomous mode, individual skills contain blocking `AskUserQuestion` calls that defeat autonomous operation. Additionally, critical test coverage gaps exist (race condition bug, missing contract tests), and the most popular DL framework's log format (HuggingFace Trainer) is unsupported.

## Design

### Section 1: Autonomy Fixes (7 items)

#### 1.1 Mid-loop method proposal auto-skip
**File:** `skills/orchestrate/SKILL.md` (steps 6.1b + 6.1d)
- Step 6.1b: Add `**Autonomous mode auto-skip:** If budget_mode == "autonomous", use stored method_proposal_scope. Skip AskUserQuestion.`
- Step 6.1d: Add `**Autonomous mode auto-skip:** If budget_mode == "autonomous", accept all proposals. Skip user presentation.`

#### 1.2 Experiment timeout
**File:** `skills/orchestrate/SKILL.md` (step 4)
- Add timeout: `max_experiment_duration = baseline_training_time * 3`
- If exceeded: kill experiment, set `status: "timeout"`, log to error tracker
- Prevents hung experiments from blocking the entire batch

#### 1.3 Research failure recovery
**File:** `skills/orchestrate/SKILL.md` (Phase 5 + step 6.8)
- Fallback chain: web search fails → retry with `source: "knowledge"` → if that fails, continue HP-only
- Log each fallback step to error tracker

#### 1.4 Implement dirty-tree auto-skip
**File:** `skills/implement/SKILL.md` (Step 3)
- In autonomous mode: `git stash --include-untracked` before branch creation
- Log: "Auto-stashed working tree changes (autonomous mode)"

#### 1.5 Prerequisites autonomous defaults
**File:** `skills/prerequisites/SKILL.md` (Steps 2, 4, 4.1)
- Unknown format → use as-is + warning
- Format mismatch → skip preparation + warning
- Env manager mismatch → use detected manager
- Missing conda env → auto-create

#### 1.6 Three more orchestrate auto-skips
**File:** `skills/orchestrate/SKILL.md`
- RL polarity: Auto-infer (reward/return → higher-is-better), log decision
- Phase 3 unknown error: After 2 retries, log + exit with partial results
- All-diverge: Attempt recovery batch with halved LRs before stopping

#### 1.7 OOM feedback to hp-tune
**File:** `skills/orchestrate/SKILL.md` (Phase 6)
- Track OOM-causing batch sizes in error tracker
- Pass `max_batch_size` constraint to hp-tune on next iteration

### Section 2: Script Fixes (3 items)

#### 2.1 HuggingFace Trainer log format
**File:** `scripts/parse_logs.py`
- New parser: detect `{'key': value}` pattern (single-quote Python dicts)
- Strategy: regex detect, replace `'` → `"`, parse as JSON
- Add to format detection priority chain

#### 2.2 result_analyzer file filter
**File:** `scripts/result_analyzer.py`
- Filter `load_results()` to `exp-*.json` and `baseline.json` only
- Prevents `prerequisites.json` and `implementation-manifest.json` from inflating counts

#### 2.3 log_event race condition fix
**File:** `scripts/error_tracker.py`
- Add `fcntl.flock()` file locking around read-modify-write in `_atomic_write_json()`
- Prevents concurrent agents from losing events

### Section 3: Test Coverage (8 items)

#### 3.1 Concurrent log_event test
- 4 threads × 5 events each = 20 events, verify none lost

#### 3.2 Pipeline state full user_choices roundtrip
- All 20+ documented fields survive save/load

#### 3.3 Baseline → HP-tune contract test
- Baseline output has `profiling.gpu_memory_used_mib`, `config` fields

#### 3.4 Experiment → Monitor log format contract test
- Experiment log format parseable by `parse_logs` → valid for `detect_divergence`

#### 3.5 Empty input edge cases
- `parse_csv_lines([])`, `detect_format([])`, `detect_nan_inf([])`, `rank_by_metric({})`, `validate_event({})`

#### 3.6 HuggingFace Trainer format test
- Parse `{'loss': 0.5, 'learning_rate': 5e-5, 'epoch': 1.0}` lines

#### 3.7 Unicode log file test
- `parse_log()` with Unicode in metric names and comments

#### 3.8 Prerequisites → schema_validator integration test
- Prerequisites report passes `validate_prerequisites()`

### Section 4: Flexibility Improvements (2 items)

#### 4.1 HP-tune tabular ML iteration 1 strategy
**File:** `skills/hp-tune/SKILL.md`
- Conditional: for sklearn/XGBoost/LightGBM, explore `max_depth`/`n_estimators` first
- Learning rate is less impactful for tree-based models

#### 4.2 Research `type: hp_only` routing
**File:** `skills/orchestrate/SKILL.md`
- `hp_only` proposals skip implement, go directly to hp-tune as search space modifications

## Deferred Items

- GAN-specific `detect_divergence` model category
- Docker/container environment detection
- Multi-objective optimization support
- Keras `fit()` log parser
- CSV quoted field handling
- `parse_logs.py` tail_lines parameter
- Implement skill tabular feature engineering patterns

## Architecture Impact

No new files. Changes to:
- 4 SKILL.md files (orchestrate, implement, prerequisites, hp-tune)
- 3 Python scripts (parse_logs, result_analyzer, error_tracker)
- 1 test file (test_skill_contracts.py) + new tests in existing test files
