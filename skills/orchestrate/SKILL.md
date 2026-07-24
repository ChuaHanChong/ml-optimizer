---
name: orchestrate
description: "Core ML optimization orchestrator. Understands model problems, dispatches specialized agents for research, HP tuning, and experiments. Use when: user wants to optimize an ML model, improve training, tune hyperparameters, or run optimization experiments."
paths: "*.py, *.yaml, *.yml, *.json, *.toml, *.cfg, *.ipynb"
---

# ML Optimization Orchestrator

Think through phase-transition decisions, branch pruning, error recovery, and cost/time budget trade-offs before acting.

You are an ML optimization orchestrator coordinating the full pipeline: understanding the model, establishing baselines, researching improvements, tuning hyperparameters, running experiments, monitoring divergence, and producing final reports.

> **Path convention:** all paths written `<exp_root>/...` refer to the `exp_root` parameter from your dispatch. The plugin does not hardcode the output directory name.

## Execution Model: Direct Dispatch vs. Workflows

Two execution modes, split by phase:

- **Direct dispatch (phases 0/1, 2, 3, 4, 9)** — interactive or single-track. You drive these: plan mode and AskUserQuestion for 0/1/4, `Agent(subagent_type="ml-optimizer:<name>-agent")` for the single agent calls in 2, 3, 9. Each `Agent()` call is a fresh, self-contained spawn — no resume/registry.

- **Dynamic workflows (phases 5, 6, 7, 8)** — each launched as one `Workflow({scriptPath, args})` pointing at the bundled script under `${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/`. The script owns that phase's fan-out/loop and dispatches the existing agents internally via `agentType` (same definitions: tools, skills, model intact). You **build the args**, **launch the workflow**, then **read its structured return plus the files the agents wrote** under `<exp_root>/`. Run the user checkpoint **between** phases (after a workflow returns), never inside one — workflows take no mid-run user input.

> **These are internal pipeline steps, not user slash-commands.** The phase 5–8 scripts are bundled in this skill at `skills/orchestrate/workflows/` and launched by `scriptPath` (not saved `name`). Scripts under `.claude/workflows/` would become user `/slash-commands`; keeping them in the skill folder and invoking by `scriptPath` keeps them out of the `/command` namespace. Their `meta.name` is display-only — NOT a saved/invocable name.

**Workflow scripts and contract** (scriptPath / args in / structured return out):

| Workflow script (`scriptPath`, display `meta.name`) | args (in) | return (out) |
|---|---|---|
| `${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-5-research.js` (`phase-5-research`) | `{ exp_root, primary_metric, model_category, scope_level, source, user_papers }` | `{ findings_path, proposals:[{index,title,impact,confidence,feasibility,scope,type,implementation_strategy,files_to_modify}], agenda_initialized }` |
| `${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-6-implement.js` (`phase-6-implement`) | `{ exp_root, project_root, findings_path, selected_indices:[int], strategy:"git_branch"\|"file_backup" }` | `{ manifest_path, branches:[{slug,branch,status,validation,reviews}] }` |
| `${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-7-experiment.js` (`phase-7-experiment`) | `{ exp_root, project_root, baseline, primary_metric, divergence_metric, divergence_lower_is_better, model_category, lower_is_better, target_value, scope_level, fixed_time_budget, fixed_epoch_budget, fixed_step_budget, hp_batches_per_round, method_proposal_scope, method_proposal_iterations, seeds_per_config }` | `{ best_exp_id, best_metric, rounds_completed, exit_reason, stacking_candidates:[{branch,improvement_pct}] }` |
| `${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-8-stacking.js` (`phase-8-stacking`) | `{ exp_root, project_root, primary_metric, lower_is_better, baseline_metric, scope_level, divergence_metric, divergence_lower_is_better, model_category, fixed_time_budget, fixed_epoch_budget, fixed_step_budget, stacking_candidates }` | `{ best_stack_branch, best_stack_metric, steps:[{method,branch,kept}] }` |

**Cross-agent context = args + files.** No message bus / `SendMessage` for phases 5–8. The phase references below keep the reusable, non-dispatch content (decision tables, metric-routing rule, schemas, round lifecycle, file-output contracts, stuck-protocol/fixpoint definition) because it documents what the scripts do internally. The "build args → `Workflow({scriptPath})` → read return + files → checkpoint" pattern replaces the old `Agent()`/`SendMessage`/registry dispatch steps.

## Reference

- Plan template: `${CLAUDE_SKILL_DIR}/references/plan-template.md` (in this skill's directory)
- JSON schemas: enforced at runtime by `${CLAUDE_PLUGIN_ROOT}/scripts/schema_validator.py` — the authoritative source. Validate any file via `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/schema_validator.py <file> result|baseline|manifest|prerequisites`. Markdown templates (batch analysis, research findings) live in their owning skill's SKILL.md.
- Python scripts: `${CLAUDE_PLUGIN_ROOT}/scripts/` (gpu_check.py, scripts/parse_logs.py, scripts/detect_divergence.py, scripts/result_analyzer.py, scripts/experiment_setup.py, scripts/implement_utils.py, scripts/pipeline_state.py, scripts/schema_validator.py, scripts/plot_results.py, scripts/error_tracker.py, scripts/prerequisites_check.py, scripts/goal_memory.py)
- Workflow scripts (bundled in this skill, launched by `scriptPath` — internal pipeline steps, NOT user slash-commands): `${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/` (phase-5-research.js, phase-6-implement.js, phase-7-experiment.js, phase-8-stacking.js)

## Goal Anchoring & Behavioral Memory

The pipeline maintains two project-scoped files to prevent optimization drift:

1. **`<exp_root>/optimization-goals.json`** — goal anchor written at Phase 0: primary metric, target value, scope constraints, frozen parameters. All agents read this before acting.

2. **`<exp_root>/learned-behaviors.json`** — accumulated behavioral memory. Agents write what they learn (HP constraints, method outcomes, divergence patterns, OOM limits); later agents read it to avoid repeating mistakes.

**Key script:** `${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> <action>` — manages both files:
- `summary` — compact briefing combining goals + behaviors + dead-ends (~500 tokens, read by agents before acting)
- `validate-output <agent> <output_json>` — post-dispatch validation (orchestrator calls after hp-tune, research, analyze)
- `sync-from-errors` — pulls OOM/divergence patterns from error_tracker into behavioral memory
- `init-goals`, `read-goals`, `log-behavior`, `query-behaviors` — CRUD operations

**Validation flow:** after hp-tune, research, and analyze return, the orchestrator validates outputs against goals. Violations (frozen param changes, scope breaches, dead-end re-proposals) are auto-corrected where possible and logged as `scope_violation` entries in behavioral memory.

Each agent also has `memory: local` in its frontmatter for persistent role-specific memory at `.claude/agent-memory-local/<agent-name>/` in the target project.

## Pipeline Overview

Each phase has a dedicated reference file with the full workflow. Read it when entering that phase.

| Phase | Reference | Execution |
|-------|-----------|-----------|
| 0 | `${CLAUDE_SKILL_DIR}/references/phase-0-discovery.md` | — (plan mode + AskUserQuestion) |
| 1 | `${CLAUDE_SKILL_DIR}/references/phase-1-understand.md` | — (direct analysis) |
| 2 | `${CLAUDE_SKILL_DIR}/references/phase-2-prerequisites.md` | `Agent(ml-optimizer:prerequisites-agent)` |
| 3 | `${CLAUDE_SKILL_DIR}/references/phase-3-baseline.md` | `Agent(ml-optimizer:baseline-agent)` |
| 4 | `${CLAUDE_SKILL_DIR}/references/phase-4-checkpoint.md` | — (AskUserQuestion; pre-authorize Phase 7 autonomy) |
| 5 | `${CLAUDE_SKILL_DIR}/references/phase-5-research.md` | `Workflow({scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-5-research.js", args})` |
| 6 | `${CLAUDE_SKILL_DIR}/references/phase-6-implement.md` | `Workflow({scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-6-implement.js", args})` |
| 7 | `${CLAUDE_SKILL_DIR}/references/phase-7-experiment-loop.md` | `Workflow({scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-7-experiment.js", args})` |
| 8 | `${CLAUDE_SKILL_DIR}/references/phase-8-stacking.md` | `Workflow({scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-8-stacking.js", args})` |
| 9 | `${CLAUDE_SKILL_DIR}/references/phase-9-report.md` | `Agent(ml-optimizer:report-agent)`, `Agent(ml-optimizer:analysis-agent)` (review mode) |

## Phase 0 + 1: Discovery, Planning & Codebase Analysis (MANDATORY)

Read `${CLAUDE_SKILL_DIR}/references/phase-0-discovery.md` for the full workflow. Phase 1 details in `${CLAUDE_SKILL_DIR}/references/phase-1-understand.md`.

Enter plan mode. Ask discovery questions (metric, target, constraints, data paths, environment, scope). Record responses. Write optimization goals. **Stay in plan mode through Phase 1** — analyze codebase, create plan, estimate cost. Present the full plan. **The user can do multiple refinement rounds** — adjusting scope, constraints, or budget — before approving. Exit plan mode only when the user chooses to proceed.

## Phase 2: Prerequisites Check

Read `${CLAUDE_SKILL_DIR}/references/phase-2-prerequisites.md` for the full workflow.

Dispatch `ml-optimizer:prerequisites-agent`. Check results. Handle failure recovery. Persist user choices.

## Phase 3: Establish Baseline

Read `${CLAUDE_SKILL_DIR}/references/phase-3-baseline.md` for the full workflow.

Dispatch `ml-optimizer:baseline-agent`. Handle failure recovery with up to 2 retries.

## Phase 4: User Checkpoint (Post-Baseline)

Read `${CLAUDE_SKILL_DIR}/references/phase-4-checkpoint.md` for the full workflow.

Show baseline results. User chooses direction: HP tuning, research, user papers, skip to experiments, or method proposals. Autonomous mode auto-selects method proposals. **Pre-authorize Phase 7 autonomy here** (`method_proposal_scope`, `method_proposal_iterations`, budget) — these become Phase 7 `args` so the experiment workflow runs with no mid-run prompts.

## Phase 5: Research (Optional)

Read `${CLAUDE_SKILL_DIR}/references/phase-5-research.md` for the full workflow.

Build the args and launch the workflow:
```
Workflow({ scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-5-research.js", args: { exp_root, primary_metric, model_category, scope_level, source, user_papers } })
```
The workflow dispatches `ml-optimizer:research-agent` internally (across angles), writes `reports/research-findings.md` + inits the agenda, handles research failure recovery (knowledge-only, then HP-only) internally, and returns `{ findings_path, proposals, agenda_initialized }`. After it returns, run the user checkpoint to confirm proposal selection.

## Phase 6: Implement Research Proposals

Read `${CLAUDE_SKILL_DIR}/references/phase-6-implement.md` for the full workflow.

Build the args from the confirmed selection and launch the workflow:
```
Workflow({ scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-6-implement.js", args: { exp_root, project_root, findings_path, selected_indices, strategy } })
```
The workflow dispatches `ml-optimizer:implement-agent` (worktree-isolated) plus the reviewers (`pr-review-toolkit:code-reviewer`, `pr-review-toolkit:silent-failure-hunter`) internally, writes `results/implementation-manifest.json` + git branches, and returns `{ manifest_path, branches }`. After it returns, check the manifest: handle dependencies, license warnings, conflicts (user checkpoints as needed).

## Phase 7: Experiment Loop (Workflow)

Read `${CLAUDE_SKILL_DIR}/references/phase-7-experiment-loop.md` for the full workflow.

Build the args (from `user_choices`, baseline, and the Phase 6 manifest) and launch the workflow:
```
Workflow({ scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-7-experiment.js", args: { exp_root, project_root, baseline, primary_metric, divergence_metric, divergence_lower_is_better, model_category, lower_is_better, target_value, scope_level, fixed_time_budget, fixed_epoch_budget, fixed_step_budget, hp_batches_per_round, method_proposal_scope, method_proposal_iterations, seeds_per_config } })
```

The script owns the experiment loop — it creates rounds and dispatches tuning, experiment, and analysis agents per iteration. Divergence detection is folded into the experiment agent (it runs `detect_divergence.py` against its own log using the `divergence_metric`/`divergence_lower_is_better`/`model_category` args); a standalone `monitor-agent` dispatch is optional, not the default. After each batch the analysis agent recommends the next action (continue, pivot, or stop) and the workflow applies the decision table in phase-7-experiment-loop.md internally.

**Autonomous, no mid-run prompts:** the loop runs non-stop (Phase 7 autonomy pre-authorized at Phase 4, passed in `args`) until the target or the fixpoint is reached. When the analysis agent recommends stop, the workflow invokes the stuck protocol (research for fresh ideas), then runs the **Exit Judgment** — no hardcoded stop-count threshold. It exits at the *fixpoint*: no new in-scope proposals (`stuck_protocol_triggered=true`) AND empty research agenda AND flat best metric — the idea space is exhausted with no progress to build on. Otherwise it continues. Every exit/continue decision is logged via `pipeline_state.py log-decision`; `consecutive_stop_count` is telemetry, not a trigger.

The workflow returns `{ best_exp_id, best_metric, rounds_completed, exit_reason, stacking_candidates }`. After each batch the live dashboard is regenerated (`${CLAUDE_PLUGIN_ROOT}/scripts/dashboard.py --live`) and baseline integrity verified — all inside the workflow.

### MANDATORY: Round Lifecycle (Phase 7 & Phase 8)

The round lifecycle below runs **inside the phase-7 and phase-8 scripts** (the runtime runs `round_manager.py` and passes `round_dir` to each `agentType` dispatch). Documented here because it defines the file-output contract the workflows must honor.

Every experiment batch MUST be wrapped in a round. Enforced at runtime by the PreToolUse hook (`validate_experiment_write.py`) — `exp-*.json` written outside a `round-N-<type>/` directory is BLOCKED. Skipping this causes pipeline deadlock.

**Before dispatching tuning-agent + experiment-agents:**
```bash
round_info=$(python3 ${CLAUDE_PLUGIN_ROOT}/scripts/round_manager.py <exp_root> create-round <type> [--branch <ml-opt/slug>])
# Response: {"id": N, "dir": "round-N-<type>", "type": "<type>", "path": "..."}
# Capture the "dir" field — this is the round_dir to pass to agents.
```

Valid `<type>` values and when to use each:

| Type | When |
|---|---|
| `hp` | Default — HP tuning batches (pivot: `branch_test`/`hp_expand`/`narrow_space`/`regularization`) |
| `evolved` | ShinkaEvolve code mutation (pivot: `code_evolution`) |
| `research` | Research-implement batches (pivot: `method_proposal`) |
| `stacked` | Phase 8 method stacking |

**Pass `round_dir` to every dispatched agent** via the agent prompt (the workflow injects it):
- Tuning-agent: include `round_dir: <dir>` so it writes proposals to `proposed-configs/<round_dir>/`
- Experiment-agents: include `round_dir: <dir>` so they write results to `results/<round_dir>/`

**After experiment-agents return:**
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/round_manager.py <exp_root> check-round <round_dir>
# Response: {"complete": true/false, "total": N, "valid": N, "terminal": N, "missing_logs": [...], "invalid": [...], "non_terminal": [...]}
```

If `complete: false`:
- For each ID in `non_terminal` or missing: create a minimal failed placeholder (`{"exp_id": "...", "status": "failed", "notes": "agent did not produce result"}`) to keep the round consistent
- For each entry in `invalid`: log to error tracker and attempt repair

**After the batch is fully analyzed:**
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/round_manager.py <exp_root> close-round --summary "<one-line summary>"
```

This marks the round closed in `rounds-manifest.json`. Not strictly required for correctness but helps downstream analysis and reporting.

**Workflow action by analysis pivot_type** (applied inside the phase-7 workflow):

| Pivot Type | Workflow Action |
|---|---|
| `branch_test`, `hp_expand`, `narrow_space`, `regularization` | Adjust search space, dispatch tuning-agent |
| `code_evolution` | Dispatch tuning (evolve HPs) → implement-agent with evolve skill → experiment |
| `method_proposal`, `qualitative_change` | Dispatch research → implement → merge branches |
| `method_stacking` | Return `stacking_candidates` so the orchestrator launches the phase-8 workflow |

**State updates:** The workflow saves pipeline state after each iteration:
```
save_state(phase=7, iteration=N, running_exp_ids=[], exp_root=exp_root)
```

## Phase 8: Method Stacking (Workflow)

Read `${CLAUDE_SKILL_DIR}/references/phase-8-stacking.md` for the full workflow.

When the phase-7 workflow returns non-empty `stacking_candidates`, launch the stacking workflow:
```
Workflow({ scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-8-stacking.js", args: { exp_root, project_root, primary_metric, lower_is_better, baseline_metric, scope_level, divergence_metric, divergence_lower_is_better, model_category, fixed_time_budget, fixed_epoch_budget, fixed_step_budget, stacking_candidates } })
```
The script ranks methods by improvement magnitude (descending), holds the "current best stack" in a variable, and per step dispatches implement (merge), experiment, and analysis agents internally — resolving interference via ShinkaEvolve if needed (requires git branch strategy). It returns `{ best_stack_branch, best_stack_metric, steps }`. After it returns, the orchestrator may relaunch the phase-7 workflow on the stacked code.

```
Phase 7 (experiment workflow) ←→ Phase 8 (stacking workflow)
  phase-7 returns stacking_candidates → launch phase-8 workflow
  After stacking → relaunch phase-7 workflow on stacked code
  Loop continues until goal or user stops
```

## Phase 9: Report & Review

Read `${CLAUDE_SKILL_DIR}/references/phase-9-report.md` for the full workflow.

Two steps (both direct `Agent()` dispatch — interactive-adjacent, single-track):
1. **Report:** dispatch `Agent(ml-optimizer:report-agent)`. Sync errors. Generate dashboard. Present summary.
2. **Session review:** dispatch `Agent(ml-optimizer:analysis-agent)` (review mode) — analyzes what worked, what didn't, and how to improve.

## Error Handling

- **GPU unavailable:** fall back to single-GPU sequential execution
- **Training crashes:** record the error, skip to next experiment in batch
- **All experiments diverge in a batch:**
  - **Recovery attempt:** before stopping, attempt a recovery batch with halved learning rates (divide all LR by 2). Log: `category: "training_failure", severity: "warning", message: "All experiments diverged — attempting recovery with halved LRs"`.
  - If the recovery batch also all-diverges: stop the loop and report to user.
- **OOM feedback to hp-tune:** when an experiment fails with `CUDA out of memory`:
  1. Record the OOM-causing batch size: `category: "training_failure", context: {"oom_batch_size": <batch_size>}`
  2. On the next hp-tune invocation, pass `max_batch_size` (one step below the OOM-causing size) so hp-tune avoids configs that will OOM again
  3. If multiple OOM events, use the smallest OOM-causing batch size as the constraint
- **Script not found:** ask user for the correct training command

## Error Tracking

Log an error event using the error tracker script at each point below:

### After agent failures (any phase):
When an agent dispatch fails (crash, timeout, invalid output):
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"agent_failure","severity":"critical","source":"orchestrate","message":"<failure description>","agent":"<agent_type>","phase":<phase>,"iteration":<iteration>}'
```

### After analyze recommends stop or pivot (Phase 7):
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"warning","source":"orchestrate","message":"<analyze recommendation and reason>","phase":7,"iteration":<iteration>,"context":{"action":"<continue|pivot|stop>","reason":"<from analyze>"}}'
```

### On pipeline resumption from interrupted state:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"orchestrate","message":"Pipeline resumed from interrupted state","phase":<resumed_phase>}'
```

### After analysis review mode failure (Phase 9):
If the analysis agent's review mode crashes or produces invalid output:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"agent_failure","severity":"warning","source":"orchestrate","message":"Analysis review mode failed: <error description>","agent":"analysis","phase":<phase>}'
```

## Directory Structure Created

The orchestrator ensures this structure exists in the target project (`<exp_root>/`):
```
<exp_root>/
  artifacts/round-N-<type>/<exp-id>/  # Checkpoints, visualizations
  logs/round-N-<type>/<exp-id>/       # Training logs (train.log, eval.log)
  proposed-configs/round-N-<type>/    # HP config proposals per round
  reports/                            # Markdown reports + research findings
  results/round-N-<type>/exp-*.json   # Schema-validated experiment results
  scripts/round-N-<type>/<exp-id>/    # Training scripts (train.sh, eval.sh)
  dev_notes.md                        # Running log of session tasks
```

## State Management

All state is persisted in the `<exp_root>/` directory:
- Experiment results in `results/*.json`
- Pipeline state in `pipeline-state.json` (phase, iteration, running experiments)
- Analysis and research findings in `reports/`
- Implementation manifest in `results/implementation-manifest.json`
- Session progress in `dev_notes.md`

### Pipeline Resumption

The orchestrator can be stopped and resumed:
1. On start, check for `pipeline-state.json` via `pipeline_state.load_state()`
2. If state exists and status is "running", run `pipeline_state.cleanup_stale()` — a 2-hour timeout: any experiment with status "running" last modified >2 hours ago is marked status "failed", notes "Marked failed by cleanup_stale — presumed interrupted". Log cleaned-up items to dev_notes before resuming.
3. Restore Phase 0 user choices from `state["user_choices"]` (primary_metric, divergence_metric, lower_is_better, target_value, train_command, eval_command, train_data_path, val_data_path, prepared_train_path, prepared_val_path, env_manager, env_name) — do NOT re-ask
4. Resume from the recorded phase and iteration. For phases 5–8, relaunch the workflow (same session, optionally via `resumeFromRunId`); the file-persisted results, rounds, and manifest let it pick up where it left off.
5. Read all past results to understand what has been tried

### State Validation

Before each phase transition, validate prerequisites via `pipeline_state.validate_phase_requirements()` — prevents cascading failures from missing or corrupted data.

## Agent Dispatch for Phases 5–8 (Workflow-Driven)

Phases 5–8 launch as dynamic workflows (one `Workflow({scriptPath, args})` each, pointing at the bundled script under `${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/`). The script dispatches agents internally via `agentType: "ml-optimizer:<name>-agent"` — fresh, self-contained dispatches reusing the existing definitions (tools, skills, model intact). No persistent-agent registry and no `SendMessage` for 5–8.

The analysis agent handles both per-batch analysis (inside the phase-7/8 workflows) and session review (Phase 9, dispatched directly via `Agent()`). The `ml-optimizer:analyze` skill includes a "Session Review Mode" section activated by the `scope: "session"` dispatch parameter.

### Cross-Agent Context = args + Files

Inside a workflow, context flows two ways — no message bus:

1. **args** — the orchestrator builds the phase `args` and passes them to `Workflow({scriptPath, args})`; the script forwards them to each `agentType` dispatch via the agent prompt.
2. **files under `<exp_root>/`** — agents read what earlier agents wrote. The formerly-relayed routes are now file/args handoffs the workflow wires up:

| Route | Handoff (file / args the next agent reads) |
|-------|--------------------------------------------|
| analyze → tuning | `reports/batch-N-analysis.md` (correlations, branch scores, recommendation) + dead-ends/agenda |
| analyze → research | pivot reason + `reports/dead-ends.json` + `reports/research-agenda.json` |
| monitor → tuning | OOM batch sizes / divergence patterns logged to error tracker → `learned-behaviors.json` |
| research → implement | `reports/research-findings*.md` + selected_indices in args |
| research → tuning | `search_space` HP priors (`{param, range, scale, source}` with citations) on hp_only proposals + `reports/research-agenda.json` entries |
| experiments → analyze | `results/round-N-*/exp-*.json` (batch completion counts, best metric) |

Validate these payloads with `schema_validator.py relay <route> <json>` (the relay schemas validate the file/args payloads the workflow hands between stages).

### Workflow Resumption

Workflows resume within a session via `resumeFromRunId`. No agent-ID state to clear across sessions — each run dispatches fresh agents, and all durable state lives in `<exp_root>/` files (results, rounds-manifest, implementation-manifest, agenda, pipeline-state). On a new session, relaunch the phase's workflow; it reads those files and continues.

## Unsupported Scenarios

Out of scope. If the user requests these, explain the limitation clearly:

- **Inference optimization:** quantization, pruning, ONNX export, TensorRT — a fundamentally different toolchain; recommend dedicated tools.
- **Multi-machine distributed training:** single machine with multiple GPUs only. Cross-node training needs a different dispatch mechanism.
- **Reinforcement learning (partial support):** supported with caveats — the plugin tunes RL hyperparameters and detects divergence via policy loss or reward collapse, but RL-specific features (reward shaping, curriculum learning, multi-agent coordination) are not orchestrated. If the RL setup logs standard metrics (loss, reward), the pipeline works.
- **Multi-seed ensembling:** one seed per experiment; multi-seed evaluation would need significant orchestrator changes.
- **Federated learning:** assumes all training data is locally accessible. Cross-device coordination and aggregation protocols are out of scope.
- **Multi-objective Pareto optimization:** optimizes a single `primary_metric`. For multi-objective needs, use weighted scoring in hp-tune or run separate sessions per metric.
