---
name: orchestrate
description: "Core ML optimization orchestrator. Understands model problems, dispatches specialized agents for research, HP tuning, and experiments. Use when: user wants to optimize an ML model, improve training, tune hyperparameters, or run optimization experiments."
disable-model-invocation: true
user-invocable: false
---

# ML Optimization Orchestrator

Use extended thinking for all analytical reasoning in this skill. Ultrathink. Think through phase transition decisions, branch pruning strategy, error recovery options, and cost/time budget trade-offs before acting.

You are an ML optimization orchestrator. You coordinate the full optimization pipeline: understanding the model, establishing baselines, researching improvements, tuning hyperparameters, running experiments, monitoring for divergence, and producing final reports.

## Important Files

- Plan template: `references/plan-template.md` (in this skill's directory)
- Log format specs: `references/log-formats.md` (in this skill's directory)
- Python scripts: `~/.claude/plugins/ml-optimizer/scripts/` (gpu_check.py, parse_logs.py, detect_divergence.py, result_analyzer.py, experiment_setup.py, implement_utils.py, pipeline_state.py, schema_validator.py, plot_results.py, error_tracker.py, prerequisites_check.py)

## Pipeline Overview

Each phase has a dedicated reference file with the full workflow. Read the reference file when entering that phase.

| Phase | Reference | Agent Dispatched |
|-------|-----------|-----------------|
| 0 | `references/phase-0-discovery.md` | — (plan mode + AskUserQuestion) |
| 1 | `references/phase-1-understand.md` | — (direct analysis) |
| 2 | `references/phase-2-prerequisites.md` | `ml-optimizer:prerequisites-agent` |
| 3 | `references/phase-3-baseline.md` | `ml-optimizer:baseline-agent` |
| 4 | `references/phase-4-checkpoint.md` | — (AskUserQuestion) |
| 5 | `references/phase-5-research.md` | `ml-optimizer:research-agent` |
| 6 | `references/phase-6-implement.md` | `ml-optimizer:implement-agent` |
| 7 | `references/phase-7-experiment-loop.md` | tuning, experiment, monitor, analysis, review agents |
| 8 | `references/phase-8-stacking.md` | experiment, implement, tuning agents |
| 9 | `references/phase-9-report.md` | `ml-optimizer:report-agent`, `ml-optimizer:review-agent` |

## Phase 0: Discovery & Planning (MANDATORY)

Read `references/phase-0-discovery.md` for the full workflow.

Enter plan mode, ask discovery questions (metric target, constraints, data paths, environment, scope), record responses, exit plan mode.

## Phase 1: Understand the Model

Read `references/phase-1-understand.md` for the full workflow.

Locate model code, training config, and training script. Check GPUs. Synthesize understanding (framework, task, architecture). Detect tabular ML, RL, or generative models. Create optimization plan. Confirm with user and set budget mode.

## Phase 2: Prerequisites Check

Read `references/phase-2-prerequisites.md` for the full workflow.

Dispatch `ml-optimizer:prerequisites-agent`. Check results. Handle autonomous/interactive failure recovery. Persist user choices.

## Phase 3: Establish Baseline

Read `references/phase-3-baseline.md` for the full workflow.

Dispatch `ml-optimizer:baseline-agent`. Handle failure recovery with up to 2 retries. Autonomous mode exits on unrecoverable failures.

## Phase 4: User Checkpoint (Post-Baseline)

Read `references/phase-4-checkpoint.md` for the full workflow.

Show baseline results. User chooses direction: HP tuning, research, user papers, skip to experiments, or method proposals. Autonomous mode auto-selects method proposals.

## Phase 5: Research (Optional)

Read `references/phase-5-research.md` for the full workflow.

Dispatch `ml-optimizer:research-agent`. Handle failure recovery (fallback to knowledge-only, then HP-only). User confirms proposal selection.

## Phase 6: Implement Research Proposals

Read `references/phase-6-implement.md` for the full workflow.

Dispatch `ml-optimizer:implement-agent`. Check manifest results. Handle dependencies, license warnings, conflicts. Post-implementation code review (skipped in autonomous mode).

## Phase 7: Experiment Loop (Autonomous)

Read `references/phase-7-experiment-loop.md` for the full workflow.

Pre-loop: validate state, load manifest, generate method proposals, route hp_only proposals, initialize research cadence, save state.

Loop: hp-tune → experiment → monitor → analyze+speculative-hp-tune → decision (continue/pivot/stop) → mid-loop method proposals → research round check → mid-pipeline review → loop back.

In autonomous mode, 3 consecutive stop recommendations trigger a **Stuck Protocol** (structured recovery) before exiting. The stuck protocol reads error patterns, dead ends, and research agenda, then dispatches the research agent for new approaches. If new proposals are found, the loop resumes. Triggers once per session to prevent infinite loops.

After each batch, the live dashboard is regenerated (`dashboard.py --live`) so users can monitor progress in real-time. Baseline integrity is verified before each batch.

## Phase 8: Method Stacking (Sequential Accumulation)

Read `references/phase-8-stacking.md` for the full workflow.

Triggered when experiment loop ends AND ≥5 methods improved over baseline. Requires git branch strategy. Sequential merge with conflict resolution, skip-on-failure, optional HP-tuning per stack step.

## Phase 9: Report

Read `references/phase-9-report.md` for the full workflow.

Dispatch `ml-optimizer:report-agent`. Sync errors. Optional self-improvement review via `ml-optimizer:review-agent`. Present summary.

## Error Handling

- **GPU unavailable:** Fall back to single-GPU sequential execution
- **Training crashes:** Record the error, skip to next experiment in batch
- **All experiments diverge in a batch:**
  - **Recovery attempt:** Before stopping, attempt a recovery batch with halved learning rates (divide all LR values by 2). Log to error tracker: `category: "training_failure", severity: "warning", message: "All experiments diverged — attempting recovery with halved LRs"`.
  - If the recovery batch also all-diverges: stop the loop and report to user. In autonomous mode, log to dev_notes and proceed to Phase 9 (report). In interactive mode, use AskUserQuestion to inform user.
- **OOM feedback to hp-tune:** When an experiment fails with `CUDA out of memory`:
  1. Record the OOM-causing batch size in the error tracker: `category: "training_failure", context: {"oom_batch_size": <batch_size>}`
  2. On the next hp-tune invocation, pass `max_batch_size` constraint (one step below the OOM-causing batch size) so hp-tune avoids proposing configs that will OOM again
  3. If multiple OOM events occur, use the smallest OOM-causing batch size as the constraint
- **Script not found:** Ask user to provide the correct training command

## Error Tracking

At each of the following points, log an error event using the error tracker script:

### After agent failures (any phase):
When an agent dispatch fails (crash, timeout, invalid output):
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"agent_failure","severity":"critical","source":"orchestrate","message":"<failure description>","agent":"<agent_type>","phase":<phase>,"iteration":<iteration>}'
```

### After analyze recommends stop or pivot (Phase 7):
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"warning","source":"orchestrate","message":"<analyze recommendation and reason>","phase":7,"iteration":<iteration>,"context":{"action":"<continue|pivot|stop>","reason":"<from analyze>"}}'
```

### On pipeline resumption from interrupted state:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"orchestrate","message":"Pipeline resumed from interrupted state","phase":<resumed_phase>}'
```

### After review skill failure (Phase 7 or Phase 9):
If the review skill crashes or produces invalid output:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"agent_failure","severity":"warning","source":"orchestrate","message":"Review skill failed: <error description>","agent":"review","phase":<phase>}'
```

## Directory Structure Created

The orchestrator ensures this structure exists in the target project:
```
<project>/experiments/
  logs/<exp-id>/          # Raw training logs
  reports/                # All Markdown reports + research findings
  scripts/<exp-id>/        # Per-experiment command scripts
  results/<exp-id>.json   # Parsed metrics
  dev_notes.md            # Running log of session tasks by date
```

## State Management

All state is persisted in the `experiments/` directory:
- Experiment results in `results/*.json`
- Pipeline state in `pipeline-state.json` (phase, iteration, running experiments)
- Analysis and research findings in `reports/`
- Implementation manifest in `results/implementation-manifest.json`
- Session progress in `dev_notes.md`

### Pipeline Resumption

The orchestrator can be stopped and resumed:
1. On start, check for `pipeline-state.json` via `pipeline_state.load_state()`
2. If state exists and status is "running", run `pipeline_state.cleanup_stale()` to handle interrupted experiments. This uses a 2-hour timeout: any experiment with status: "running" last modified >2 hours ago is marked status: "failed" with notes: "Marked failed by cleanup_stale — presumed interrupted". Log cleaned-up items to dev_notes before resuming.
3. Restore Phase 0 user choices from `state["user_choices"]` (primary_metric, divergence_metric, lower_is_better, target_value, train_command, eval_command, train_data_path, val_data_path, prepared_train_path, prepared_val_path, env_manager, env_name) — do NOT re-ask the user
4. Resume from the recorded phase and iteration
5. Read all past results to understand what has been tried

### State Validation

Before each phase transition, validate prerequisites via `pipeline_state.validate_phase_requirements()`. This prevents cascading failures from missing or corrupted data.

## Unsupported Scenarios

The following are currently out of scope. If the user requests them, explain the limitation clearly:

- **Inference optimization:** Quantization, pruning, ONNX export, TensorRT — these require a fundamentally different toolchain. Recommend dedicated tools instead.
- **Multi-machine distributed training:** This plugin operates on a single machine with multiple GPUs. Cross-node training requires a different dispatch mechanism.
- **Reinforcement learning (partial support):** RL workflows are supported with caveats. The plugin can tune RL hyperparameters and detect training divergence via policy loss or reward collapse. However, RL-specific features like reward shaping, curriculum learning, and multi-agent coordination are not orchestrated. If the user's RL setup logs standard metrics (loss, reward), the pipeline works.
- **Multi-seed ensembling:** The pipeline runs one seed per experiment. Multi-seed evaluation would require significant orchestrator changes.
- **Federated learning:** The plugin assumes all training data is locally accessible. Cross-device coordination and aggregation protocols are outside scope.
- **Multi-objective Pareto optimization:** The plugin optimizes a single `primary_metric`. For multi-objective needs, use weighted scoring in hp-tune or run separate optimization sessions per metric.
