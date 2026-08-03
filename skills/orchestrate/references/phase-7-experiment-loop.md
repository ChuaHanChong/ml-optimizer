# Phase 7: Experiment Loop (Workflow)

Phase 7 runs as a **dynamic workflow** (`Workflow({scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-7-experiment.js", args})`). The loop lives inside the workflow script: it creates rounds, dispatches existing agents internally via `agentType` (e.g. `agent(prompt, {agentType: "ml-optimizer:tuning-agent"})`), reads the files those agents write under `<exp_root>/`, applies the decision tree, and returns a structured summary. The loop is **autonomous with no mid-run user input** — user decisions (method-proposal scope/iterations, budget) are pre-authorized at Phase 4 and arrive in `args`.

> "Dispatch the X agent" = an `agent({agentType: "ml-optimizer:X-agent"})` call inside the workflow script. Each dispatch is a fresh, self-contained agent that reads the relevant `<exp_root>/` files (manifest, prior `batch-N-analysis.md`, agenda, dead-ends, learned-behaviors) plus the prompt args.

## Workflow Args & Return

**Args (in):**
```
{ exp_root, project_root, baseline, primary_metric, divergence_metric,
  divergence_lower_is_better, model_category, lower_is_better, target_value, scope_level,
  fixed_time_budget, fixed_epoch_budget, fixed_step_budget, hp_batches_per_round, method_proposal_scope,
  method_proposal_iterations, seeds_per_config, eval_tasks }
```
> `fixed_time_budget` (seconds) and `fixed_epoch_budget` (epochs) are the **training** budget, passed through from `user_choices`. The workflow forwards whichever is set into every experiment-agent prompt so each run is capped exactly like the baseline and stays comparable (CLAUDE.md "Training budget options"). Passed as two typed fields (not one derived `budget`) so the experiment agent knows whether to wrap `timeout` vs cap epochs. Distinct from the workflow runtime's token `budget` global, which the script uses to self-bound the loop's agent spend (`budget.remaining()`). *(A legacy scalar `budget` arg is still tolerated and treated as seconds.)* `fixed_step_budget` (integer environment timesteps) is the RL budget unit — when set it wins over the time/epoch budgets and maps to the framework's timestep flag (e.g. `--total_timesteps`).

> `model_category` comes from `user_choices` (Phase 0). The workflow uses `args.model_category || pre.model_category` (baseline.json fallback via the pre-loop) and threads it into the tuning, analysis, and experiment prompts so divergence thresholds and HP strategy match the model class.

**Return (out):**
```
{ best_exp_id, best_metric, rounds_completed, exit_reason,
  stacking_candidates: [{branch, improvement_pct}, ...] }
```

When the workflow returns non-empty `stacking_candidates`, the orchestrator launches the phase-8 stacking workflow.

## Pre-Loop: Validate Pipeline State

Before the loop, validate prerequisites:

```bash
python3 -c "
import sys; # sys.path: add the plugin's scripts/ directory
from pipeline_state import validate_phase_requirements
import json; print(json.dumps(validate_phase_requirements(6, '<exp_root>')))
"
```

**Required state:**
- `<exp_root>/results/baseline.json` must exist with `metrics` and `config` keys
- If `implementation-manifest.json` exists, it must have a `proposals` key

If validation fails, stop and report the missing prerequisites.

## Pre-Loop: Verify Baseline Integrity

Verify baseline metrics are unchanged since Phase 3:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/pipeline_state.py <exp_root> verify-baseline
```

Non-zero exit (checksum mismatch): **HALT the pipeline immediately.** Log to error tracker:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"config_error","severity":"critical","source":"orchestrate","message":"Baseline integrity check FAILED — metrics may have been modified. Pipeline halted.","phase":7}'
```
Report the error to the user. Do NOT continue — all experiment comparisons would be invalid.

Warning return (legacy pipeline without checksum): log to dev_notes and continue.

## Pre-Loop: Sync Behavioral Memory

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> sync-from-errors
```
Populates `<exp_root>/learned-behaviors.json` with OOM limits, divergence patterns, and dead-end outcomes from the error tracker. All agents read this via the `summary` command.

## Pre-Loop: Load Implementation Manifest

If `<exp_root>/results/implementation-manifest.json` exists:
1. Read the manifest
2. Collect proposals with `"status": "validated"` — skip `"validation_failed"` / `"implementation_error"`
3. Each validated proposal branch is tested with HP tuning.
   **Branch existence validation:** Before passing `code_branches` to hp-tune, verify each branch exists via `git rev-parse --verify <branch>`. Remove missing branches and log to error tracker.
4. Also test the baseline (original branch, HP-only) for comparison
5. **Non-git detection:** If manifest has `"strategy": "file_backup"`, force sequential execution (only ONE experiment at a time)

If no manifest exists, run HP-only experiments on the current code.

## Pre-Loop: Method Proposals (if `method_proposal_scope` is set)

If `method_proposal_scope` is set in `args` (pre-authorized at Phase 4), generate pre-loop method proposals — **no confirmation prompt** (acceptance delegated at Phase 4):

1. **Dispatch the research agent** via `agent({agentType: "ml-optimizer:research-agent"})` with prompt:
   "Research ML optimization techniques. Parameters: source: both, scope_level: {method_proposal_scope}, output_path: <exp_root>/reports/research-findings-method-proposals.md, model_type: {model_type}, task: {task}, current_metrics: {baseline metrics}, problem_description: {problem_description}, exp_root: {exp_root}."
   The agent reads baseline metrics from `results/baseline.json` and writes proposals to the output path.

2. **Filter proposals automatically** (no prompt): keep in-scope, non-dead-end proposals (consult `reports/dead-ends.json` and `reports/suggestion-history.json`). Validate via:
   ```bash
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> validate-output research '<proposals_json>'
   ```
   Drop scope-violating / dead-end proposals.

3. **Implement the surviving proposals** via `agent({agentType: "ml-optimizer:implement-agent"})` with prompt:
   "Implement research proposals. Parameters: findings_path: <exp_root>/reports/research-findings-method-proposals.md, selected_indices: {surviving_indices}, project_root: {project_root}."
   The agent reads the findings file and writes branches + `results/implementation-manifest.json`.

4. **Check implementation results** from `<exp_root>/results/implementation-manifest.json`:
   - Merge validated method-proposal branches into `code_branches`
   - Follow the same handling as Phase 6 (failed proposals, dependencies, license warnings) — handled inside the workflow

5. **Store method proposal state:** `method_proposal_iterations`: 1 (initial)

## Pre-Loop: Route `hp_only` Research Proposals

When processing research proposals (Phase 5 or mid-loop step 7), check each proposal's `type`:
- **`type: "hp_only"`**: Search-space modifications (e.g. "try cyclical learning rates", "increase weight decay range"), not code changes. Route directly to hp-tune as search-space adjustments — skip the implement skill entirely. Merge the suggested HP ranges into the existing `search_space` dict.
- **`type: "code_change"` or no type field**: Route through implement as normal (create branches, validate, etc.).

## Pre-Loop: Initialize Research Cadence

Initialize the research round counter:
- `batches_since_last_research = 0`
- Tracks HP tuning batches since the last research → implement cycle
- When it reaches `hp_batches_per_round`, step 8 auto-triggers a new research round

## Pre-Loop: Save Pipeline State

Save Phase 0 user choices into pipeline state so they persist across interruptions:

```bash
python3 -c "
import sys, json; # sys.path: add the plugin's scripts/ directory
from pipeline_state import save_state
save_state(6, 0, [], '<exp_root>', user_choices={
    'primary_metric': '<primary_metric>',
    'divergence_metric': '<divergence_metric>',
    'lower_is_better': <lower_is_better>,
    'divergence_lower_is_better': <divergence_lower_is_better>,
    'target_value': <target_value or None>,
    'train_command': '<train_command>',
    'eval_command': '<eval_command or None>',
    'train_data_path': '<train_data_path>',
    'val_data_path': '<val_data_path or None>',
    'prepared_train_path': '<prepared_train_path or None>',
    'prepared_val_path': '<prepared_val_path or None>',
    'env_manager': '<env_manager>',
    'env_name': '<env_name or None>',
    'model_category': '<model_category or None>',
    'user_papers': <user_papers or None>,
    'method_proposal_scope': '<method_proposal_scope or None>',
    'method_proposal_iterations': <method_proposal_iterations or 0>,
    'hp_batches_per_round': <hp_batches_per_round or 3>,
})
"
```

## Metric Routing Rule

**Critical:** Use `divergence_metric` (Phase 0 Q7, default `"loss"`) for divergence detection. Use `primary_metric` (may be "accuracy", "psnr", "f1", etc.) only for the analyze and hp-tune skills.

- Monitor skill: `metric_to_watch = <divergence_metric>`, `lower_is_better = <divergence_lower_is_better>`
- Analyze skill: `primary_metric` from Phase 0, `lower_is_better` by metric type
- HP-tune skill: uses `primary_metric` for ranking

If the monitor skill cannot find `<divergence_metric>` in the logs, it attempts auto-detection via a fallback chain (see monitor skill).

## Polarity Conflict Rule

- `primary_metric == divergence_metric` (e.g. both "loss"): no conflict, both lower-is-better.
- They differ (e.g. primary="accuracy", divergence="loss"): no conflict, independent polarity.
- `divergence_metric` higher-is-better (e.g. "reward" for RL): override monitor's `lower_is_better` to `False`. Divergence means the metric dropped sharply, not exploded.
- Store `divergence_lower_is_better` as a separate field in user_choices.

## Branch Dispatch Strategy

When the manifest has multiple code branches:

- **Iteration 1:** Test each branch with baseline HPs (one experiment per branch) to see which code changes show promise.
- **Iteration 2:** Prune branches worse than baseline. Focus experiments on surviving branches + baseline.
- **Iterations 3+:** Focus on the best branch + HP tuning. The analysis agent judges which branches are competitive with the overall best and which to drop — no fixed percentage cutoff.

## Loop Iteration:

0. **Create a new round (MANDATORY):**
   Before dispatching any agents this iteration, create a round directory. The PreToolUse hook blocks all `exp-*.json` writes not inside a `round-N-<type>/` subdirectory.

   ```bash
   round_info=$(python3 ${CLAUDE_PLUGIN_ROOT}/scripts/round_manager.py <exp_root> create-round hp)
   # Capture the "dir" field (e.g., "round-3-hp") — this is the round_dir to pass to all agents this iteration.
   ```

   **Round type by action:**
   - `create-round hp` — default, HP tuning batches (most iterations)
   - `create-round evolved` — ShinkaEvolve code mutation (pivot: `code_evolution`)
   - `create-round research` — research-implement batches (pivot: `method_proposal`)
   - `create-round stacked` — Phase 8 method stacking (see phase-8-stacking.md)

   Save `round_dir` into the iteration's local context. It is passed to every subsequent dispatch this iteration.

1. **Get HP configs:**
   Dispatch the tuning agent via `agent({agentType: "ml-optimizer:tuning-agent"})` with prompt:
   ```
   "Propose HP configurations. Parameters: project_root: {project_root}, num_gpus: {num_gpus}, search_space: {search_space}, iteration: {iteration}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, code_branches: {code_branches}, max_batch_size: {max_batch_size or omit}, warm_start_enabled: {warm_start_enabled or false}, available_checkpoints: {available_checkpoints_json or {}}, branch_scores: {branch_scores_json or {}}, round_dir: {round_dir}."
   ```
   The agent reads cross-agent context from files (no message bus): the prior round's `reports/batch-{N-1}-analysis.md` (recommendation, correlations, branch_scores), `learned-behaviors.json` (OOM `max_batch_size`, divergence patterns synced from the monitor), the research agenda, and the implementation manifest. Pass the same context as explicit args where the workflow has it in scope:
     - `code_branches`: From implementation manifest, or `[]` for HP-only.
     - `max_batch_size` *(optional)*: One step below the smallest OOM-causing batch size (from `learned-behaviors.json` / error tracker). Omit if no OOM events.
     - `branch_scores` *(optional)*: From the prior analysis (`compute_branch_scores`).
   - Reads past results and proposes the next batch of configs
   - Number of configs = `max(num_gpus, 1)`
   - **Check hp-tune recommendation:** If hp-tune output includes `"recommendation": "stop"`, log it to error tracker with `category: "pipeline_inefficiency"` and note it for analyze. Analyze makes the final continue/pivot/stop decision, but this is an early signal of search-space exhaustion.

   ### HP-Tune Failure Recovery

   If hp-tune crashes or produces invalid configs:

   1. **Validate output:** Each proposed config has required fields (`exp_id`, `config`, `gpu_id`), values within search-space bounds, and no duplicates of previously-tried configs.
   2. **If validation fails:** Retry hp-tune once with a simplified prompt: "Propose {N} configs within these ranges: {search_space}. Return valid JSON only."
   3. **If retry also fails:** Fall back to random sampling — pick `lr` uniformly from the search-space log-range, `batch_size` from the allowed set, other HPs at baseline values. The orchestrator constructs the JSON directly.
   4. **If random sampling also fails** (construction error):
      Use the baseline config as-is for all experiments in this batch (re-validates baseline, keeps loop alive). Log to error tracker: `category: "agent_failure", severity: "critical", source: "orchestrate", message: "All HP-tune fallbacks failed — using baseline config as placeholder batch"`. Log to dev_notes: "HP-tune completely failed — running baseline-config batch as placeholder." Proceed to step 2 with baseline configs.

   Log each fallback step to error tracker with `category: "agent_failure"`, `source: "orchestrate"`.

   **Goal validation (post-dispatch):** After hp-tune returns proposed configs:
   ```bash
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> validate-output hp-tune '<proposed_configs_json>'
   ```
   If `valid` is false: remove violating configs from the batch. Log each violation:
   ```bash
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> log-behavior scope_violation '{"agent":"hp-tune","violation_type":"<type>","detail":"<detail>"}'
   ```
   If ALL configs are removed, re-dispatch hp-tune with the violations as context ("Your previous proposals violated: [violations]. Please propose alternatives.").

2. **Run experiments:**
   - For each proposed config, dispatch an experiment agent via `agent({agentType: "ml-optimizer:experiment-agent"})` (runs the `ml-optimizer:experiment` skill)
   - Pass `code_branch` and `code_proposal` from the manifest (or null for HP-only)
   - **Pass `round_dir` (from Step 0)** — the experiment skill writes results to `results/<round_dir>/<exp_id>.json`. Without this, the PreToolUse hook blocks the write.
   - If multiple GPUs available, dispatch experiments in parallel (see Parallel GPU Dispatch Pattern below) — the workflow's `parallel()` fan-out, one experiment per GPU
   - Each experiment runs on a separate GPU

3. **Monitor experiments (folded into the experiment agent):**
   - Divergence detection is **not a separate concurrent dispatch** here. The experiment agent runs `${CLAUDE_PLUGIN_ROOT}/scripts/detect_divergence.py` against its own training log while training, using the divergence params passed in its dispatch. The workflow does NOT spawn a standalone `monitor-agent` per batch by default — divergence params flow to the **experiment agent** instead.
   - **If `divergence_metric` is not null**, pass the divergence params through to each experiment-agent dispatch (Step 2 / Parallel GPU Dispatch Pattern): `divergence_metric`, `divergence_lower_is_better`, `model_category`. The experiment agent feeds these to `detect_divergence.py` (e.g. `--higher-is-better` when `divergence_lower_is_better` is false, `--model-category {model_category}`) so reward-like RL metrics get correct polarity and thresholds.
   - **Optional standalone monitor:** A separate `agent({agentType: "ml-optimizer:monitor-agent"})` dispatch (concurrent with experiments) remains available for long runs where out-of-process polling is preferred, but it is **opt-in**, not the default. When used, dispatch with prompt:
     ```
     "Monitor running experiments. Parameters: log_files: {log_files}, exp_ids: {exp_ids}, project_root: {project_root}, poll_interval: 30, metric_to_watch: {divergence_metric}, lower_is_better: {divergence_lower_is_better}, model_category: {model_category}."
     ```
   - The experiment agent (or optional monitor) relays divergence/OOM findings by logging to the error tracker; the workflow syncs them into `learned-behaviors.json` (via `goal_memory.py sync-from-errors`) so the next round's tuning agent reads the OOM `max_batch_size` constraint from file.
   - Divergence status handling (experiment agent's in-process check or the optional monitor):
     - `healthy`: Training progressing normally — continue waiting
     - `diverged`: Stop the experiment automatically, record divergence reason in results
     - `completed`: Training finished naturally during monitoring — proceed to wait/analysis
     - `unmonitored`: Watched metric not found in the logs after all fallback attempts. Warn the user once (via dev_notes) that divergence monitoring is disabled for this experiment. Continue without divergence checks — rely on the experiment's hard timeout (from baseline profiling) as the safety net.
     - `failed`: The divergence check itself errored — log as `agent_failure`, continue without monitoring for remaining experiments in this batch
     - `no_output`: Log file has no parseable data yet — continue monitoring (normal for early training)
   - **If `divergence_metric` is null** (tabular ML — scikit-learn, XGBoost, LightGBM): skip divergence detection entirely. Wait for experiments to complete naturally.

   ### Early Batch Abort on Mass Divergence

   Each experiment kills itself when its own in-process divergence check fires (or the optional monitor kills it). The analysis agent sees the results (including diverged experiments) and recommends appropriate action (narrow search space, etc.).

4. **Wait for completion:**
   - All experiments in the batch must complete (or be stopped) before analysis
   - **Experiment timeout:** Each experiment has a hard timeout computed as:
     - If `baseline.json` has `profiling.estimated_timeout_seconds`: use that value directly (recorded for tabular ML and RL)
     - Else if `baseline.json` has `profiling.training_duration_seconds`: `max_experiment_duration = training_duration_seconds * 3`
     - Else if `baseline.json` has throughput profiling: `max_experiment_duration = baseline_training_time * 3`
     - Else (no profiling): fallback to 21600 seconds (6 hours)
     If an experiment exceeds the timeout:
     1. Kill the experiment process
     2. Set `status: "timeout"` in the experiment result JSON
     3. Log to error tracker: `category: "timeout", severity: "warning", source: "orchestrate", message: "Experiment <exp_id> timed out after <duration>s (limit: <max_duration>s)"`
     4. Continue with the remaining experiments in the batch
   - **Incremental dashboard updates:** As individual experiments complete, regenerate the live dashboard so users can monitor progress in real-time:
     ```bash
     python3 ${CLAUDE_PLUGIN_ROOT}/scripts/dashboard.py <exp_root> --live
     ```
     Log to dev_notes: "Experiment {exp_id} completed mid-batch: {primary_metric}={value}"
   - Save pipeline state after each batch completes

   ### Post-Batch Completeness Validation

   Before analysis, validate all experiment results from this batch with `--strict` (enforces completeness — completed experiments must have non-empty metrics, iteration, method_tier, duration_seconds):

   ```bash
   for exp_id in <batch_exp_ids>:
       python3 schema_validator.py <exp_root>/results/${exp_id}.json result --strict
   ```

   If any validation fails:
   1. Log to error tracker: `category: "config_error", severity: "warning", source: "orchestrate", message: "Experiment <exp_id> result incomplete: <errors>"`
   2. If the experiment agent is still reachable, ask it to fill missing fields
   3. Otherwise set `status: "failed"` in the result file with `notes: "Completeness validation failed: <errors>"`
   4. Continue — analyze skips failed experiments

5. **Analyze results:**
   Dispatch the analysis agent via `agent({agentType: "ml-optimizer:analysis-agent"})` with prompt:
   ```
   "Analyze batch {N} results. Parameters: project_root: {project_root}, batch_number: {batch_number}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, target_value: {target_value or null}."
   ```
   The agent reads cross-agent context from files: the batch's `results/<round_dir>/exp-*.json` (HP configs, divergence count, completion counts), `learned-behaviors.json`, the dead-end catalog, and the research agenda. It writes `reports/batch-{N}-analysis.md` (correlations, branch scores, recommendation) — the file the next round's tuning/research dispatch reads. It returns the decision (continue / pivot / stop) the workflow acts on.
     - Compares all experiments, ranks them, identifies patterns
     - Recommends: continue, pivot, or stop
   - **Live dashboard update:** After analyze completes, regenerate the dashboard:
     ```bash
     python3 ${CLAUDE_PLUGIN_ROOT}/scripts/dashboard.py <exp_root> --live
     ```
   - **Goal validation (post-dispatch):** After analyze returns:
     ```bash
     python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> validate-output analyze '<analyze_output_json>'
     ```
     If metric mismatch detected: log a critical error to the error tracker. Do NOT trust the analysis — the metric confusion could cause wrong ranking of experiments.

5b. **Mid-run goal adjustment (workflow boundary, not an in-loop prompt):**

   The phase-7 workflow takes **no mid-run user input** — it cannot pause to ask. Goal changes are handled at workflow boundaries, not inside the loop:

   1. To change a goal (e.g. "change target to 85%" or "freeze learning rate at 0.01"), the user ends the current run (or it returns at its next boundary).
   2. The **orchestrator** applies the goal update between phases via:
      ```bash
      python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> update-goals '<updates_json>'
      ```
      Example updates:
      - Target: `{"objective": {"target_value": 85.0}}`
      - Metric: `{"objective": {"primary_metric": "f1", "lower_is_better": false}}`
      - Freeze: `{"constraints": {"frozen_parameters": ["lr"]}}`
      - Scope: `{"constraints": {"scope_level": "architecture"}}`
   3. The orchestrator updates `user_choices` in pipeline state to match and logs to dev_notes: "Mid-run goal update at iteration {N}: {changes}".
   4. The orchestrator relaunches the phase-7 workflow (optionally via `resumeFromRunId`) with the updated `args`; the next hp-tune, research, and analyze dispatches read the updated goals from `optimization-goals.json` / args.

   If no goal change is requested, the workflow continues immediately — no checkpoint, no pause.

   **Important:** If `primary_metric` changes, all future experiment rankings use the new metric. Past experiments are NOT re-ranked — the change applies going forward only.

5c. **Check round completeness and close the round:**
   Verify all expected experiments produced valid results, then close the round in the manifest.

   ```bash
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/round_manager.py <exp_root> check-round <round_dir>
   # Response: {"complete": bool, "total": N, "valid": N, "terminal": N,
   #            "missing_logs": [...], "invalid": [...], "non_terminal": [...]}
   ```

   - If `complete: true`: proceed.
   - If `non_terminal` is non-empty (experiments stuck in `running`): run `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/pipeline_state.py <exp_root> cleanup` to mark stale ones failed, then re-check.
   - If `invalid` is non-empty: each entry has `exp_id` + `errors`. Log to error tracker, attempt schema repair, or mark as failed.
   - For missing experiment IDs (expected from proposed configs but no JSON file exists): create a minimal failed placeholder at `results/<round_dir>/<exp_id>.json` with `{"exp_id": "...", "status": "failed", "config": {...}, "metrics": {}, "notes": "agent did not produce result"}` to keep the round consistent.

   Close the round with a one-line summary:
   ```bash
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/round_manager.py <exp_root> close-round --summary "Iteration {N}: best {primary_metric}={value}, {completed}/{total} completed"
   ```

6. **Decision:**
   - If analyze says **continue**:
     - Loop back to step 0 (create a new round, then hp-tune → experiments)
   - If analyze says **pivot**:
     - Apply pivot adjustments → loop back to step 0 (create a new round with the appropriate `<type>`, then hp-tune → experiments)
     **Pivot dispatch by type:**
     - `"branch_test"`: Pass analyze's suggestion to hp-tune. Generate configs for untested branches with baseline HPs. No research needed.
     - `"hp_expand"`: Widen the search space around the best config (extend LR range by 2× in each direction). Pass updated `search_space` to hp-tune.
     - `"narrow_space"`: Constrain the search space to the range around the best result (analyze's `suggestion` field contains bounds). Pass narrowed `search_space` to hp-tune.
     - `"regularization"`: Add regularization HPs (weight_decay, dropout) to the search space or expand their range. Pass updated `search_space` to hp-tune. No research needed.
     - `"research"`: Route to step 7 (same as `method_proposal`).
     - `"method_proposal"`, `"qualitative_change"`: Route to step 7 (existing handling).
     - **Unknown pivot_type:** Treat as `"hp_expand"` (safest default). Log to error tracker.
   - If analyze says **stop**:
     - **Do NOT exit the loop immediately.** Exit is the orchestrator's reasoned judgment, never a fixed counter.
     - Log the stop recommendation. Increment `consecutive_stop_count` (telemetry only — not an exit trigger).
     - Invoke the **Stuck Protocol** below to search for new approaches, then run the **Exit Judgment** (see "Loop exit conditions").

     **Stuck Protocol** (invoked whenever analysis recommends stop):

       1. Read error patterns: `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> patterns`
       2. Read success metrics: `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> success <primary_metric> <lower_is_better>`
       3. Read dead-end catalog: `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> dead-end list`
       4. Read research agenda: `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> agenda list`
       5. Dispatch the research agent with all failure context via `agent({agentType: "ml-optimizer:research-agent"})` with prompt:
          ```
          "The optimization is stuck. Find new approaches that haven't been tried. Parameters: source: both, model_type: {model_type}, task: {task}, current_metrics: {best_metrics}, problem_description: {problem_description}, exp_root: {exp_root}, scope_level: {method_proposal_scope or 'architecture'}. CONTEXT: Error patterns: {patterns}. Success metrics: {success}. Dead ends (DO NOT re-propose): {dead_ends}. Research agenda: {agenda}. Focus on techniques NOT in the dead-end catalog."
          ```
          The failure context (`patterns`, `success`, `dead_ends`, `agenda`) is read from files (error tracker, dead-ends.json, research-agenda.json) and passed in the prompt.
       6. If research returns new in-scope proposals (after dead-end + suggestion-history filtering) → set `stuck_protocol_triggered=false`, reset `consecutive_stop_count=0`, route to step 7 for implementation, continue loop.
       7. If research returns no new proposals → set `stuck_protocol_triggered=true`, increment `consecutive_stop_count`, then run the **Exit Judgment** below.
   - **If analyze output is malformed or contains an unexpected action:** Treat as `agent_failure`. Log to error tracker. Retry analyze once with a simplified prompt: "Based on the experiment results, should we continue, pivot, or stop? Respond with exactly one of: continue, pivot, stop." If retry also fails, default to `continue`.
   - **Loop exit conditions:** The experiment loop is **autonomous by default** — it runs non-stop until one of the following. There is **no hardcoded stop-count threshold**; the third condition is the workflow's own evidence-based judgment.
     1. Target metric achieved (from Phase 0).
     2. User manually stops (the run ends; the orchestrator can relaunch via `resumeFromRunId`).
     3. **The workflow judges the search has reached a *fixpoint*** — no fresh direction left AND no progress to build on. An objective state, not a count: literally nothing new to try within scope and the metric has stopped moving. The workflow returns with `exit_reason` set accordingly.

     **Exit Judgment** (run after the stuck protocol whenever analyze recommended stop). Decide from evidence — never from a magic number:

       a. **Gather** the evidence the stuck protocol already collected: research output (any new in-scope proposals?), success metrics (current best + whether it improved since the last stop), the dead-end catalog, and the research agenda (any untried items left?).
       b. **CONTINUE the loop** (do not exit) if ANY of these holds — there is still something to try:
          - research returned ≥1 new in-scope proposal not in the dead-end catalog / suggestion-history → route to step 7 and implement it;
          - the research agenda still has an untried item → pivot to it;
          - the best metric improved since the previous stop → the plateau broke.
          On any of these, reset `stuck_protocol_triggered=false` and `consecutive_stop_count=0`, then continue.
       c. **RETURN from the workflow** (which advances the orchestrator to Phase 9) only at the fixpoint — when ALL of these hold:
          - the stuck protocol returned no new in-scope proposals (`stuck_protocol_triggered` is already `true`),
          - the research agenda has no untried items left, and
          - the best metric has not improved since the previous stop (flat within the noise margin).
          At this state the in-scope idea space is exhausted with no progress to build on — running the stuck protocol again with the same best metric and dead-end set would return the same nothing. This guarantees the loop terminates when (and only when) genuinely stuck. The workflow returns `{best_exp_id, best_metric, rounds_completed, exit_reason: "fixpoint", stacking_candidates}`.
       d. **Log the decision** so it is auditable:
          ```bash
          python3 ${CLAUDE_PLUGIN_ROOT}/scripts/pipeline_state.py <exp_root> log-decision '{"phase":7,"agent":"phase-7-experiment","decision_type":"loop_exit_judgment","decision":"exit"|"continue","iteration":<N>,"reasoning":"<best_metric, improved_since_last_stop, agenda_items_remaining, new_proposals_count>"}'
          ```
          Append the same reasoning to `dev_notes.md` and log to the error tracker.

     A single plateau is **not** exhaustion — breakthroughs can follow plateaus, so the bar for exit is the fixpoint in (c), not a recommendation count. `consecutive_stop_count` is telemetry only; `stuck_protocol_triggered` is the state flag that, with an unchanged best metric and an empty agenda, defines the fixpoint.

7. **Mid-loop method proposal trigger** (when analyze recommends new methods):

   If analyze returns `pivot_type: "method_proposal"` or `pivot_type: "qualitative_change"`:

   a. **Scope check (no prompt — pre-authorized at Phase 4):**
      Read `method_proposal_scope` from `args` (set at Phase 4). The old mid-loop "Scope options 1/2/3/4" prompt is gone — the workflow takes no mid-run input.
      - If `method_proposal_scope` is `null` (user opted out): treat `method_proposal`/`qualitative_change` as `hp_expand` (safest in-scope fallback) and skip steps b–e.
      - If `method_proposal_iterations` is exhausted (mid-loop research rounds already run ≥ the pre-authorized budget): skip steps b–e (no more method proposals this run); fall back to `hp_expand`.
      - Otherwise use `scope_level = method_proposal_scope` and proceed.

   b. **Generate proposals:** Dispatch the research agent via `agent({agentType: "ml-optimizer:research-agent"})` with prompt:
      ```
      "Research ML optimization techniques. Parameters: source: both, scope_level: {scope_level}, output_path: <exp_root>/reports/research-findings-method-proposals-iter{N}.md, model_type: {model_type}, task: {task}, current_metrics: {current_metrics}, problem_description: {problem_description}, exp_root: {exp_root}. CONTEXT: analyze pivot_type={pivot_type}, reason={reason}; best improvement={best_improvement}%; dead ends (DO NOT re-propose): {dead_end_catalog}."
      ```
      The analyze context (pivot reason, best improvement, dead-ends) is read from `reports/batch-{N}-analysis.md` and `reports/dead-ends.json` and passed in the prompt.

   **Goal validation (post-dispatch):** After research returns proposals:
   ```bash
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> validate-output research '<proposals_json>'
   ```
   If `valid` is false: remove scope-violating and dead-end proposals before passing to implement. Log violations to behavioral memory.

   c. **Filter proposals (no prompt):**
      The workflow auto-selects in-scope, non-dead-end proposals (acceptance delegated at Phase 4). If filtering leaves no proposals, skip implement and fall back to `hp_expand` for this iteration.

   d. **Implement proposals:** Dispatch the implement agent via `agent({agentType: "ml-optimizer:implement-agent"})` with prompt:
      ```
      "Implement research proposals. Parameters: findings_path: {findings_path}, selected_indices: {selected_indices}, project_root: {project_root}."
      ```
      Creates new `ml-opt/<slug>` branches and writes `results/implementation-manifest.json` (the file the next hp-tune dispatch reads for `code_branches`).

   e. **Merge into experiment loop:** Add the new validated branches to `code_branches`. Reset the iteration counter for these new branches only (they start at iteration 1 = `method_default_hp` tier). Existing branches keep their iteration count.

   **If `pivot_type == "code_evolution"` (evolutionary code improvement):**

   Instead of research → implement, the flow is: **tuning agent → implement agent → tuning agent → experiment agent**. The analysis agent's own conditions (|rho| < 0.3 + method improved) prevent unnecessary evolution — no artificial cooldown needed.

   a. **Tune evolve HPs:** Dispatch the tuning agent via `agent({agentType: "ml-optimizer:tuning-agent"})` with prompt:
      "Propose ShinkaEvolve evolution HPs for code_evolution. Read `learned-behaviors.json` category `evolve_hp` for prior evolution outcomes. Consider: how many generations produced the best mutation last time, whether more population diversity is needed, and the improvement trajectory. Return `evolve_recommendation: {num_generations, population_size, reasoning}`."

   b. **Execute evolve:** Dispatch the implement-agent via `agent({agentType: "ml-optimizer:implement-agent"})` with the evolve skill.
      Prompt: `Skill("ml-optimizer:evolve")` with parameters:
      - `project_root`, `parent_branch` (best method branch), `parent_metrics` (best result metrics)
      - `primary_metric`, `lower_is_better`, `scope_level`
      - `exp_root`
      - `feedback_context`: {batch_analysis from latest analyze, error_patterns, dead_ends, learned_behaviors}
      - `evolve_recommendation`: from tuning agent (step a) — `{num_generations, population_size, reasoning}`

      The evolve skill runs: `shinka-convert` → `shinka-run` (with file handoff loop) → `shinka-inspect` → commit best as `ml-opt/evolved-<slug>`.

   c. **Verify result:** The evolve skill returns `{status, branch, mutations_evaluated, best_combined_score, ...}`.
      - If `status == "validated"`: add `branch` to `code_branches`
      - If `status == "validation_failed"`: log to error tracker, continue loop without new branch
      - If `status == "shinkaevolve_unavailable"`: ShinkaEvolve couldn't be installed or crashed. Fall back to the research → implement path (the `method_proposal` steps a–e above). Log the failure.

   d. **Tune training HPs + run experiment:** Dispatch tuning agent to propose training HPs for the evolved branch, then dispatch experiment-agent. The evolved branch enters the normal HP-tune → experiment → analyze loop.

   d. **Update state:**
      - Increment `method_proposal_iterations` in user_choices
      - Save pipeline state

   e. **Continue loop:** Loop back to step 1 (hp-tune) with the expanded `code_branches` list. Reset `batches_since_last_research = 0`.

8. **Research round check** (cadence-based research trigger):

   Auto-triggers research → implement on a regular cadence, independent of analyze's pivot recommendation. Applies when `method_proposal_scope` is set.

   **Conditions (ALL must be true):**
   - `method_proposal_scope` is set (user opted into method proposals)
   - `batches_since_last_research >= hp_batches_per_round`
   - Step 7 did NOT already trigger this iteration (avoid double research)

   **If conditions met:**

   a. **Log the trigger:**
      ```bash
      python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"orchestrate","message":"Cadence-based research round triggered after <N> HP batches","phase":7,"iteration":<iteration>,"context":{"batches_since_last_research":<N>,"method_proposal_iterations":<M>}}'
      ```

   b. **Generate proposals:** Dispatch the research agent via `agent({agentType: "ml-optimizer:research-agent"})` with prompt:
      ```
      "Research ML optimization techniques. Parameters: source: both, scope_level: {method_proposal_scope}, output_path: <exp_root>/reports/research-findings-method-proposals-iter{N}.md, model_type: {model_type}, task: {task}, current_metrics: {current_metrics}, problem_description: {problem_description}, exp_root: {exp_root}. CONTEXT: last analysis summary from reports/batch-{N-1}-analysis.md; best improvement={best_improvement}%; branches active: {code_branches}; dead ends (DO NOT re-propose): {dead_end_catalog}."
      ```
      The analyze summary, best improvement, and dead-ends are read from the corresponding `<exp_root>/` files and passed in the prompt.

   c. **Check results:**
      - If research returns new proposals (not all filtered by deduplication): proceed to implement
      - If research returns **no new proposals** (all deduplicated): skip implement, double `hp_batches_per_round` (exponential backoff), log:
        ```bash
        python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"orchestrate","message":"Research round yielded no new proposals — increasing cadence to <new_value> batches","phase":7,"iteration":<iteration>}'
        ```

   d. **Implement proposals:** ALL returned proposals are implemented automatically. Dispatch the implement agent via `agent({agentType: "ml-optimizer:implement-agent"})` with prompt:
      ```
      "Implement research proposals. Parameters: findings_path: {findings_path}, selected_indices: {all_indices}, project_root: {project_root}."
      ```
      Creates new `ml-opt/<slug>` branches and updates `results/implementation-manifest.json`.

   e. **Merge into experiment loop:** Same as step 7e — add new validated branches to `code_branches`, reset iteration counter for new branches.

   f. **Update state:**
      - Increment `method_proposal_iterations`
      - Reset `batches_since_last_research = 0`
      - Save pipeline state

   **If conditions NOT met:** Increment `batches_since_last_research` and continue.

9. **Loop back:**

    **End-of-iteration sync:** Keep behavioral memory current with the latest error events:
    ```bash
    python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> sync-from-errors
    ```

    After steps 6/7/8, increment `batches_since_last_research` and return to step 1 (Get HP configs). The loop continues until the Decision step (6) forces an exit.

## Parallel GPU Dispatch Pattern

When dispatching experiments across multiple GPUs, the workflow uses its `parallel()` fan-out, one `agent({agentType: "ml-optimizer:experiment-agent"})` per experiment.

**If manifest strategy is `"file_backup"` (non-git project):** dispatch ONE experiment at a time (sequential). Wait for each to complete before starting the next. File-backup proposals share the same working directory and cannot run in parallel.

**Otherwise (git_branch strategy or HP-only):** dispatch all experiments in parallel — for each config in `proposed_configs`, call:
```
agent(
  "Run experiment {exp_id} with config: {config_json}. GPU: {gpu_id}. Project root: {project_root}. Train command: {train_command}. Eval command: {eval_command or null}. Code branch: {code_branch or null}. Code proposal: {code_proposal or null}. Proposal source: {proposal_source or null}. Method tier: {method_tier or null}. Iteration: {iteration}. Prepared train path: {prepared_train_path or null}. Prepared val path: {prepared_val_path or null}. Checkpoint source: {checkpoint_source_json or null}. Divergence metric: {divergence_metric or null}. Divergence lower is better: {divergence_lower_is_better}. Model category: {model_category or null}.",
  { agentType: "ml-optimizer:experiment-agent" }
)
```

Then wait for all experiment agents to complete before dispatching analyze. Reasoning depth is set by each agent's `effort` frontmatter (`xhigh` analytical / `medium` procedural) — the workflow adds no reasoning keyword; `agent({agentType})` inherits it.
