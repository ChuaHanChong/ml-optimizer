---
name: run-diagnostic
description: "Run end-to-end diagnostics — validates plugin structure, dispatches all 10 agents, and runs a full optimization pipeline on the test fixture via live Agent() dispatch."
allowed-tools: "Bash, Read, Write, Edit, Glob, Grep, Agent, Skill, WebSearch, WebFetch"
---

# ML Optimizer End-to-End Diagnostic

You are running a comprehensive diagnostic of the ml-optimizer plugin. This validates plugin structure via pytest, confirms all 10 agents dispatch correctly, and runs the full Phase 2→9 pipeline via live Agent() dispatch — the only way to test the multi-agent orchestration end-to-end.

## Step 1: Run structural tests (pytest)

```bash
python3 -m pytest tests/test_plugin_structure.py -v
```

Report any failures. If all pass, continue.

## Step 2: Agent dispatch smoke tests

Dispatch each of the 10 agents with a minimal smoke-test prompt. Run them in 2 batches for speed.

**Batch 1 — Procedural agents (model: sonnet):**

For each, dispatch with: "This is a smoke test. List your tools and confirm you can see your preloaded skill. Respond in 2-3 sentences."

1. `ml-optimizer:prerequisites-agent`
2. `ml-optimizer:baseline-agent`
3. `ml-optimizer:experiment-agent`
4. `ml-optimizer:monitor-agent`

**Batch 2 — Analytical agents (model: opus):**

1. `ml-optimizer:research-agent`
2. `ml-optimizer:implement-agent`
3. `ml-optimizer:tuning-agent`
4. `ml-optimizer:analysis-agent`
5. `ml-optimizer:report-agent`
6. `ml-optimizer:review-agent`

For each agent, verify:

- Agent resolves (no "not found" error)
- Agent lists its declared tools
- Agent confirms it can see its preloaded skill

Report results in a table.

## Step 3: Full pipeline via live Agent() dispatch

This is the core diagnostic — you act as the orchestrator, dispatching agents directly with pre-defined parameters. This tests the full optimization flow including all 8 autoresearch-inspired features.

**Error handling:** After each phase, verify the expected outputs exist. If a phase fails, log it as FAILED, skip to Step 3.8 (feature checklist) with partial results, and include the failure in the final report.

### 3.1: Set up test project

```bash
rm -rf /tmp/ml-opt-diagnostic
cp -r ~/.claude/plugins/ml-optimizer/tests/fixtures/tiny_resnet_cifar10/ /tmp/ml-opt-diagnostic/
cd /tmp/ml-opt-diagnostic && git init && git add . && git commit -m "initial"
mkdir -p /tmp/ml-opt-diagnostic/experiments/{results,reports,logs,scripts,artifacts}
```

Use these paths throughout the diagnostic:

- Project root: `/tmp/ml-opt-diagnostic`
- Experiment root: `/tmp/ml-opt-diagnostic/experiments`

### 3.2: Phase 2 — Prerequisites

Dispatch the prerequisites agent:

```text
Agent(
  description: "Diagnostic: check prerequisites",
  prompt: "Check prerequisites for ML project. Parameters: project_root: /tmp/ml-opt-diagnostic, framework: pytorch, training_script: train.py, config_path: config.yaml, train_data_path: embedded_in_code, val_data_path: embedded_in_code, env_manager: conda, env_name: base, exp_root: /tmp/ml-opt-diagnostic/experiments.",
  subagent_type: "ml-optimizer:prerequisites-agent"
)
```

**Verify:** Read `experiments/results/prerequisites.json`. Confirm `ready_for_baseline` is true. If not, log Phase 2 as FAILED.

### 3.3: Phase 3 — Baseline

Dispatch the baseline agent:

```text
Agent(
  description: "Diagnostic: establish baseline",
  prompt: "Establish baseline metrics. Parameters: project_root: /tmp/ml-opt-diagnostic, train_command: python train.py --epochs 2, eval_command: python eval.py, model_category: supervised, exp_root: /tmp/ml-opt-diagnostic/experiments.",
  subagent_type: "ml-optimizer:baseline-agent"
)
```

**Verify:** Read `experiments/results/baseline.json`. Confirm it has `metrics` and `config` keys.

**Store baseline checksum** (immutable baseline feature):

```bash
python3 -c "
import sys, json
sys.path.insert(0, '$HOME/.claude/plugins/ml-optimizer/scripts')
from pipeline_state import save_state, _compute_baseline_checksum
baseline = json.loads(open('/tmp/ml-opt-diagnostic/experiments/results/baseline.json').read())
checksum = _compute_baseline_checksum(baseline['metrics'])
save_state(3, 0, [], '/tmp/ml-opt-diagnostic/experiments', baseline_checksum=checksum, user_choices={
  'primary_metric': 'loss', 'lower_is_better': True, 'budget_mode': 'auto',
  'difficulty': 'easy', 'difficulty_multiplier': 8, 'fixed_time_budget': 30
})
print(f'Baseline checksum stored: {checksum[:16]}...')
"
```

**Verify baseline integrity:**

```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/pipeline_state.py /tmp/ml-opt-diagnostic/experiments verify-baseline
```

If exit code is non-zero, log Phase 3 as FAILED.

### 3.4: Phase 5 — Research (knowledge-only)

**Before dispatching:** Read `experiments/results/baseline.json` and note the actual baseline loss value. Substitute it into the prompt below.

Dispatch the research agent with `source: "knowledge"` (no internet needed):

```text
Agent(
  description: "Diagnostic: research optimization techniques",
  prompt: "Ultrathink. Research ML optimization techniques. Parameters: source: knowledge, model_type: CNN (ResNet), task: image classification (CIFAR-10), current_metrics: {loss: <ACTUAL_BASELINE_LOSS>}, problem_description: Improve classification accuracy on CIFAR-10 with a tiny ResNet, scope_level: training, exp_root: /tmp/ml-opt-diagnostic/experiments, output_path: /tmp/ml-opt-diagnostic/experiments/reports/research-findings-method-proposals.md.",
  subagent_type: "ml-optimizer:research-agent"
)
```

**Verify:**

- `experiments/reports/research-findings-method-proposals.md` exists with at least 1 proposal
- `experiments/reports/research-agenda.json` exists (research agenda feature)

### 3.5: Phase 6 — Implement

**Before dispatching:** Read `experiments/reports/research-findings-method-proposals.md` and note the proposal names/indices.

Dispatch the implement agent:

```text
Agent(
  description: "Diagnostic: implement research proposals",
  prompt: "Implement research proposals. Parameters: project_root: /tmp/ml-opt-diagnostic, findings_path: /tmp/ml-opt-diagnostic/experiments/reports/research-findings-method-proposals.md, selected_indices: [1], exp_root: /tmp/ml-opt-diagnostic/experiments.",
  subagent_type: "ml-optimizer:implement-agent"
)
```

**Verify:**

- `experiments/results/implementation-manifest.json` exists with `proposals` array
- At least one proposal has `status: "validated"`
- Git branches exist: run `git -C /tmp/ml-opt-diagnostic branch --list "ml-opt/*"`

### 3.6: Phase 7 — Experiment Loop (1 iteration)

**Before dispatching:** Read `experiments/results/implementation-manifest.json` and extract the validated branch names (e.g., `ml-opt/label-smoothing`).

#### HP-Tune

```text
Agent(
  description: "Diagnostic: propose HP configs",
  prompt: "Ultrathink. Propose HP configurations. Parameters: project_root: /tmp/ml-opt-diagnostic, num_gpus: 1, primary_metric: loss, lower_is_better: true, iteration: 1, remaining_budget: 4, fixed_time_budget: 30, code_branches: [<VALIDATED_BRANCHES>], exp_root: /tmp/ml-opt-diagnostic/experiments.",
  subagent_type: "ml-optimizer:tuning-agent"
)
```

**After hp-tune:** Read the proposed configs from `experiments/results/proposed-configs/`.

#### Experiment (for each proposed config)

```text
Agent(
  description: "Diagnostic: run experiment <EXP_ID>",
  prompt: "Run experiment. Parameters: exp_id: <EXP_ID>, config: <CONFIG_JSON>, gpu_id: 0, project_root: /tmp/ml-opt-diagnostic, train_command: python train.py --epochs 2, eval_command: python eval.py, code_branch: <BRANCH_OR_NULL>, fixed_time_budget: 30, iteration: 1, method_tier: <TIER>, proposal_source: <SOURCE_OR_NULL>, exp_root: /tmp/ml-opt-diagnostic/experiments.",
  subagent_type: "ml-optimizer:experiment-agent"
)
```

#### Monitor (concurrent with experiments)

Dispatch the monitor agent in the background for each running experiment:

```text
Agent(
  description: "Diagnostic: monitor experiment <EXP_ID>",
  prompt: "Monitor experiment for divergence. Parameters: exp_id: <EXP_ID>, log_file: /tmp/ml-opt-diagnostic/experiments/logs/<EXP_ID>/train.log, metric_to_watch: loss, lower_is_better: true, exp_root: /tmp/ml-opt-diagnostic/experiments.",
  subagent_type: "ml-optimizer:monitor-agent",
  run_in_background: true
)
```

#### Analyze

```text
Agent(
  description: "Diagnostic: analyze results",
  prompt: "Ultrathink. Analyze experiment results. Parameters: project_root: /tmp/ml-opt-diagnostic, batch_number: 1, primary_metric: loss, lower_is_better: true, remaining_budget: 3, exp_root: /tmp/ml-opt-diagnostic/experiments.",
  subagent_type: "ml-optimizer:analysis-agent"
)
```

**Verify after loop iteration:**

- `experiments/results/exp-*.json` files exist with experiment results
- `experiments/reports/batch-1-analysis.md` exists
- Research agenda updated: `python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py /tmp/ml-opt-diagnostic/experiments agenda list`
- Baseline integrity still valid: `python3 ~/.claude/plugins/ml-optimizer/scripts/pipeline_state.py /tmp/ml-opt-diagnostic/experiments verify-baseline`
- Regenerate live dashboard: `python3 ~/.claude/plugins/ml-optimizer/scripts/dashboard.py /tmp/ml-opt-diagnostic/experiments --live`

### 3.7: Phase 9 — Report + Review

#### Report agent

```text
Agent(
  description: "Diagnostic: generate final report",
  prompt: "Generate a comprehensive final report. Parameters: project_root: /tmp/ml-opt-diagnostic, primary_metric: loss, lower_is_better: true, model_description: Tiny ResNet for CIFAR-10, task_description: image classification, exp_root: /tmp/ml-opt-diagnostic/experiments.",
  subagent_type: "ml-optimizer:report-agent"
)
```

**Verify:**

- `experiments/reports/final-report.md` exists
- `experiments/reports/dashboard.html` exists and contains experiment data
- `experiments/artifacts/pipeline-overview.excalidraw` exists

#### Review agent (self-improvement)

```text
Agent(
  description: "Diagnostic: self-improvement review",
  prompt: "Ultrathink. Run self-improvement review. Parameters: project_root: /tmp/ml-opt-diagnostic, exp_root: /tmp/ml-opt-diagnostic/experiments, primary_metric: loss, lower_is_better: true, scope: session.",
  subagent_type: "ml-optimizer:review-agent"
)
```

### 3.8: Feature verification checklist

Run these checks and report pass/fail for each:

```bash
EXP=/tmp/ml-opt-diagnostic/experiments

echo "=== Feature Verification ==="

# 1. Immutable baseline
python3 ~/.claude/plugins/ml-optimizer/scripts/pipeline_state.py $EXP verify-baseline \
  && echo "✓ Immutable baseline: checksum valid" \
  || echo "✗ Immutable baseline: FAILED"

# 2. Research agenda
python3 -c "
import json
agenda = json.loads(open('$EXP/reports/research-agenda.json').read()).get('ideas', [])
print(f'✓ Research agenda: {len(agenda)} ideas') if agenda else print('✗ Research agenda: empty')
"

# 3. Dead-end catalog
python3 -c "
from pathlib import Path
p = Path('$EXP/reports/dead-ends.json')
print('✓ Dead-end catalog: exists') if p.exists() else print('— Dead-end catalog: not triggered (OK)')
"

# 4. Dashboard
python3 -c "
html = open('$EXP/reports/dashboard.html').read()
has_exps = 'exp-' in html
print('✓ Dashboard: has experiment data') if has_exps else print('✗ Dashboard: missing data')
"

# 5. Excalidraw
test -f $EXP/artifacts/pipeline-overview.excalidraw \
  && echo "✓ Excalidraw: pipeline diagram exists" \
  || echo "✗ Excalidraw: missing"

# 6. Baseline checksum in state
python3 -c "
import json
state = json.loads(open('$EXP/pipeline-state.json').read())
print('✓ Baseline checksum: stored') if 'baseline_checksum' in state else print('✗ Baseline checksum: missing')
"

# 7. Error tracking
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py $EXP summary

echo "=== Done ==="
```

### 3.9: Cleanup

```bash
rm -rf /tmp/ml-opt-diagnostic
```

## Step 4: Report

Summarize all results:

```text
ML Optimizer End-to-End Diagnostic Results
==========================================
Structural tests (pytest):  X/Y passed
Agent smoke tests:          10/10 dispatched

Full Pipeline (live Agent() dispatch):
  Phase 2 Prerequisites:    [passed/failed]
  Phase 3 Baseline:         [passed/failed]
  Phase 5 Research:         [passed/failed] — N proposals generated
  Phase 6 Implement:        [passed/failed] — N branches created
  Phase 7 Experiment Loop:  [passed/failed] — N experiments completed
    - HP-Tune:              [passed/failed]
    - Experiment:           [passed/failed]
    - Monitor:              [passed/failed]
    - Analyze:              [passed/failed]
  Phase 9 Report:           [passed/failed]
  Phase 9 Review:           [passed/failed]

Feature Verification:
  Immutable baseline:       [✓/✗]
  Research agenda:          [✓/✗] — N ideas tracked
  Dead-end catalog:         [✓/—] — triggered if branches pruned
  Dashboard (live):         [✓/✗]
  Excalidraw diagrams:      [✓/✗]
  Baseline checksum:        [✓/✗]
  Error tracking:           [✓/✗] — N events logged

Issues found: [none or list]
```
