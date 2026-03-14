# Phase 7: Experiment Loop (Autonomous)

This loop runs autonomously without user checkpoints until complete or blocked.

## Pre-Loop: Validate Pipeline State

Before starting the experiment loop, validate all prerequisites:

```bash
python3 -c "
import sys; sys.path.insert(0, '$HOME/.claude/plugins/ml-optimizer/scripts')
from pipeline_state import validate_phase_requirements
import json; print(json.dumps(validate_phase_requirements(6, '<exp_root>')))
"
```

**Required state:**
- `experiments/results/baseline.json` must exist with `metrics` and `config` keys
- If `implementation-manifest.json` exists, it must have `proposals` key

If validation fails, stop and report the missing prerequisites to the user.

## Pre-Loop: Verify Baseline Integrity

Verify the baseline metrics haven't been modified since Phase 3:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/pipeline_state.py <exp_root> verify-baseline
```

If exit code is non-zero (baseline checksum mismatch): **HALT the pipeline immediately.** Log to error tracker:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"config_error","severity":"critical","source":"orchestrate","message":"Baseline integrity check FAILED — metrics may have been modified. Pipeline halted.","phase":7}'
```
Report the error to the user. Do NOT continue — all experiment comparisons would be invalid.

If the verification returns a warning (legacy pipeline without checksum): log to dev_notes and continue normally.

## Pre-Loop: Load Implementation Manifest

If `experiments/results/implementation-manifest.json` exists:
1. Read the manifest
2. Collect all proposals with `"status": "validated"` — skip any with `"status": "validation_failed"` or `"status": "implementation_error"`
3. Each validated proposal branch will be tested with HP tuning
   **Branch existence validation:** Before passing `code_branches` to hp-tune, verify each branch exists via `git rev-parse --verify <branch>`. Remove missing branches and log to error tracker.
4. Also test the baseline (original branch, HP-only) for comparison
5. **Non-git detection:** If manifest has `"strategy": "file_backup"`, force sequential execution (only ONE experiment at a time)

If no manifest exists, run HP-only experiments on the current code.

## Pre-Loop: Method Proposals (if user chose option 5 in Phase 4)

If `method_proposal_scope` is set in user_choices (i.e., user chose option 5 in Phase 4):

1. **Dispatch the research agent:**
   ```
   Agent(
     description: "Research method proposals",
     prompt: "Ultrathink. Research ML optimization techniques. Parameters: source: both, scope_level: {method_proposal_scope}, output_path: experiments/reports/research-findings-method-proposals.md, model_type: {model_type}, task: {task}, current_metrics: {current_metrics}, problem_description: {problem_description}, exp_root: {exp_root}.",
     subagent_type: "ml-optimizer:research-agent"
   )
   ```

2. **Present proposals to user for confirmation** (same as Phase 5 post-research checkpoint):

   **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, skip the user question. Auto-select all proposals. Log to dev_notes: `"Autonomous mode: auto-selected all N method proposals for implementation"`.

   ```
   Method proposals (from LLM knowledge + web search):
   [summary of proposals from research-findings-method-proposals.md]

   Which proposals should I pursue?
   - [1] Proposal A (complexity: low, expected: +X%)
   - [2] Proposal B (complexity: medium, expected: +Y%)
   - [3] Custom: describe your own approach
   - [4] Skip, just tune HPs on existing code
   ```

3. **If user selects proposals:** Dispatch the implement agent:
   ```
   Agent(
     description: "Implement method proposals",
     prompt: "Ultrathink. Implement research proposals. Parameters: findings_path: experiments/reports/research-findings-method-proposals.md, selected_indices: {selected_indices}, project_root: {project_root}.",
     subagent_type: "ml-optimizer:implement-agent"
   )
   ```

4. **Check implementation results** from `experiments/results/implementation-manifest.json`:
   - Merge validated method proposal branches into the `code_branches` list
   - Follow the same handling as Phase 6 (failed proposals, dependencies, license warnings)

5. **Store method proposal state:**
   - `method_proposal_iterations`: 1 (initial)

## Pre-Loop: Route `hp_only` Research Proposals

When processing research proposals (from Phase 5 or mid-loop step 7), check each proposal's `type` field:
- **`type: "hp_only"`**: These proposals recommend search space modifications (e.g., "try cyclical learning rates", "increase weight decay range") rather than code changes. Route them directly to hp-tune as search space adjustments — skip the implement skill entirely. Merge the suggested HP ranges into the existing `search_space` dict.
- **`type: "code_change"` or no type field**: Route through implement as normal (create branches, validate, etc.).

This prevents unnecessary implementation overhead for proposals that only affect HP tuning parameters.

## Pre-Loop: Initialize Research Cadence

Initialize the research round counter for autonomous mode:
- `batches_since_last_research = 0`
- This counter tracks how many HP tuning batches have run since the last research → implement cycle
- In autonomous mode, when this counter reaches `hp_batches_per_round`, step 8 auto-triggers a new research round

## Pre-Loop: Save Pipeline State

Save Phase 0 user choices into pipeline state so they persist across interruptions:

```bash
python3 -c "
import sys, json; sys.path.insert(0, '$HOME/.claude/plugins/ml-optimizer/scripts')
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
    'budget_mode': '<budget_mode>',
    'method_proposal_scope': '<method_proposal_scope or None>',
    'method_proposal_iterations': <method_proposal_iterations or 0>,
    'hp_batches_per_round': <hp_batches_per_round or 3>,
})
"
```

## Metric Routing Rule

**Critical:** Use the user's `divergence_metric` (from Phase 0 Q7, default: `"loss"`) for divergence detection. Use `primary_metric` (which may be "accuracy", "psnr", "f1", etc.) only for the analyze and hp-tune skills.

- Monitor skill: `metric_to_watch = <divergence_metric>`, `lower_is_better = <divergence_lower_is_better>`
- Analyze skill: `primary_metric` from user's Phase 0 answer, `lower_is_better` based on metric type
- HP-tune skill: uses `primary_metric` for ranking

If the monitor skill cannot find `<divergence_metric>` in the logs, it will attempt auto-detection via a fallback chain (see monitor skill for details).

## Polarity Conflict Rule

- When `primary_metric == divergence_metric` (e.g., both "loss"): no conflict, both lower-is-better.
- When they differ (e.g., primary="accuracy", divergence="loss"): no conflict, independent polarity.
- When `divergence_metric` is higher-is-better (e.g., "reward" for RL): override monitor's `lower_is_better` to `False`. Divergence means metric dropped sharply, not exploded.
- Store `divergence_lower_is_better` as a separate field in user_choices.

## Branch Dispatch Strategy

When the implementation manifest contains multiple code branches:

- **Iteration 1:** Test each branch with baseline HPs (one experiment per branch). This determines which code changes show promise.
- **Iteration 2:** Prune branches that performed worse than baseline. Focus experiments on surviving branches + baseline.
- **Iterations 3+:** Focus on the best branch + HP tuning. Only keep branches whose best metric is within 5% relative of the overall best: `abs(branch_best - overall_best) / abs(overall_best) <= 0.05`. If `overall_best` is zero, keep all branches.

## Loop Iteration:

1. **Get HP configs** (use speculative proposals from previous iteration if available):
   - **If speculative proposals are available from the previous iteration's background hp-tune:**
     1. Validate speculative proposals before use (see "Speculative Proposal Validation" below)
     2. If valid → use them as this iteration's configs. Skip hp-tune invocation entirely.
     3. If invalid → discard them and invoke hp-tune synchronously as normal.
   - **Otherwise (first iteration, or speculative proposals were discarded):**
     Dispatch the tuning agent:
     ```
     Agent(
       description: "HP tuning iteration {iteration}",
       prompt: "Ultrathink. Propose HP configurations. Parameters: project_root: {project_root}, num_gpus: {num_gpus}, search_space: {search_space}, iteration: {iteration}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, remaining_budget: {remaining_budget}, code_branches: {code_branches}, max_batch_size: {max_batch_size or omit}.",
       subagent_type: "ml-optimizer:tuning-agent"
     )
     ```
     - `remaining_budget`: `max_experiments - total_experiments_so_far`. HP-tune caps proposals at `min(max(num_gpus, 1), remaining_budget)`.
     - `code_branches`: From implementation manifest, or `[]` for HP-only.
     - `max_batch_size` *(optional)*: One step below the smallest OOM-causing batch size. Omit if no OOM events.
   - It reads past results and proposes the next batch of configs
   - Number of configs = `min(max(num_gpus, 1), remaining_budget)` (capped to prevent budget overshoot)
   - **Check hp-tune recommendation:** If hp-tune output includes `"recommendation": "stop"`, log it to error tracker with `category: "pipeline_inefficiency"` and note it for the analyze step. Analyze makes the final continue/pivot/stop decision, but hp-tune's recommendation provides an early signal of search space exhaustion.

   ### HP-Tune Failure Recovery

   If hp-tune crashes or produces invalid configs:

   1. **Validate output:** Check each proposed config has required fields (`exp_id`, `config`, `gpu_id`), values are within search space bounds, and no duplicates of previously-tried configs.
   2. **If validation fails:** Retry hp-tune once with a simplified prompt: "Propose {N} configs within these ranges: {search_space}. Return valid JSON only."
   3. **If retry also fails:** Fall back to random sampling — pick `lr` uniformly from search space log-range, `batch_size` from allowed set, other HPs at baseline values. The orchestrator constructs the JSON directly.
   4. **If random sampling also fails** (construction error):
      - **Autonomous mode:** If `budget_mode == "autonomous"`, use the baseline config as-is for all experiments in this batch (re-validates baseline, keeps loop alive). Log to error tracker: `category: "agent_failure", severity: "critical", source: "orchestrate", message: "All HP-tune fallbacks failed — using baseline config as placeholder batch (autonomous mode)"`. Log to dev_notes: "HP-tune completely failed — running baseline-config batch as placeholder." Proceed to step 2 with baseline configs.
      - **Interactive mode:** Ask user to provide configs manually via AskUserQuestion.

   Log each fallback step to error tracker with `category: "agent_failure"`, `source: "orchestrate"`.

2. **Run experiments:**
   - For each proposed config, invoke `ml-optimizer:experiment` skill
   - Pass `code_branch` and `code_proposal` from the manifest (or null for HP-only)
   - If multiple GPUs available, dispatch experiments in parallel using the Agent tool
   - Each experiment runs on a separate GPU

3. **Monitor experiments:**
   - **If `divergence_metric` is not null**, dispatch the monitor agent:
     ```
     Agent(
       description: "Monitor experiments for divergence",
       prompt: "Monitor running experiments. Parameters: log_files: {log_files}, exp_ids: {exp_ids}, project_root: {project_root}, poll_interval: 30, metric_to_watch: {divergence_metric}, lower_is_better: {divergence_lower_is_better}, model_category: {model_category}.",
       subagent_type: "ml-optimizer:monitor-agent"
     )
     ```
   - Monitor status handling:
     - `healthy`: Training is progressing normally — continue waiting
     - `diverged`: Stop the experiment automatically, record divergence reason in experiment results
     - `completed`: Training finished naturally during monitoring — proceed to wait/analysis
     - `unmonitored`: The watched metric was not found in the logs after all fallback attempts. Warn the user once (via dev_notes) that divergence monitoring is disabled for this experiment. Continue without divergence checks — rely on the experiment's hard timeout (from baseline profiling) as the safety net.
     - `failed`: Monitor itself encountered an error — log as `agent_failure`, continue without monitoring for remaining experiments in this batch
     - `no_output`: Log file has no parseable data yet — continue monitoring (normal for early training)
   - **If `divergence_metric` is null** (tabular ML — scikit-learn, XGBoost, LightGBM): skip the monitor skill entirely. Wait for experiments to complete naturally without divergence monitoring.

   ### Early Batch Abort on Mass Divergence

   When monitoring a batch of experiments in parallel, track divergence timestamps:

   - **Trigger:** If `>= 2` experiments in the same batch diverge AND all divergences occurred within 60 seconds of their respective start times:
     1. Cancel remaining running experiments in the batch (kill processes)
     2. Mark cancelled experiments as `status: "cancelled"` with `notes: "Early batch abort — mass divergence detected"`
     3. Log to error tracker: `category: "training_failure", severity: "critical", source: "orchestrate", message: "Early batch abort: <N_diverged> of <batch_size> experiments diverged within 60s of start"`
     4. Log to dev_notes: "Early batch abort at iteration <N>: <diverged_count> experiments diverged within 60s. Cancelled <cancelled_count> remaining."
     5. Proceed directly to step 4 with partial results
   - **Rationale:** Divergence within 60 seconds indicates a systematic config problem (LR too high, NaN initialization) that will affect all similar configs. Two or more confirms it's systematic, not a fluke.

4. **Wait for completion:**
   - All experiments in the batch must complete (or be stopped) before analysis
   - **Experiment timeout:** Each experiment has a hard timeout computed as:
     - If `baseline.json` has `profiling.estimated_timeout_seconds` (tabular ML): use that value directly
     - Else if `baseline.json` has throughput profiling: `max_experiment_duration = baseline_training_time * 3`
     - Else (no profiling): fallback to 21600 seconds (6 hours)
     If an experiment exceeds the timeout:
     1. Kill the experiment process
     2. Set `status: "timeout"` in the experiment result JSON
     3. Log to error tracker: `category: "timeout", severity: "warning", source: "orchestrate", message: "Experiment <exp_id> timed out after <duration>s (limit: <max_duration>s)"`
     4. Continue with the remaining experiments in the batch
   - Save pipeline state after each batch completes

5. **Analyze results + speculative hp-tune (parallel):**
   - **Start analyze synchronously:**
     ```
     Agent(
       description: "Analyze batch {N} results",
       prompt: "Ultrathink. Analyze batch {N} results. Parameters: project_root: {project_root}, batch_number: {batch_number}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, target_value: {target_value or null}, remaining_budget: {remaining_budget}.",
       subagent_type: "ml-optimizer:analysis-agent"
     )
     ```
     - It compares all experiments, ranks them, identifies patterns
     - It recommends: continue, pivot, or stop
   - **At the SAME TIME, start speculative hp-tune in background** (only if `remaining_budget > max(num_gpus, 1)`):
     ```
     Agent(
       description: "Speculative hp-tune for next batch",
       prompt: "Ultrathink. This is a SPECULATIVE proposal — the orchestrator may discard these results if analyze recommends stop or pivot. Parameters: project_root: {project_root}, num_gpus: {num_gpus}, search_space: {search_space}, iteration: {iteration + 1}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, remaining_budget: {remaining_budget - current_batch_size}, code_branches: {code_branches}.",
       subagent_type: "ml-optimizer:tuning-agent",
       run_in_background: true
     )
     ```
   - If `remaining_budget <= max(num_gpus, 1)`: skip speculative hp-tune (not enough budget for another full batch)
   - Analyze completes first (it's synchronous). Speculative hp-tune may still be running.
   - **Live dashboard update:** After analyze completes, regenerate the dashboard so users can monitor progress in real-time:
     ```bash
     python3 ~/.claude/plugins/ml-optimizer/scripts/dashboard.py <exp_root> --live
     ```
   - **Wait policy:** If analyze says "continue" and speculative hp-tune has not completed, wait up to 120 seconds. If it completes in time, validate and use. If it doesn't, discard and invoke hp-tune synchronously. Log timeout to error tracker with `category: "agent_failure", severity: "info"`.
   - If analyze says "pivot" or "stop", discard speculative hp-tune immediately — do not wait.

6. **Decision:**
   - If analyze says **continue** AND speculative proposals are available and valid:
     - Use speculative proposals → loop back to step 2 immediately (zero GPU idle time)
   - If analyze says **continue** BUT speculative proposals are invalid or unavailable:
     - Discard speculative proposals (if Agent failure, log to error tracker with `category: "agent_failure", source: "orchestrate"`). Invoke hp-tune synchronously → loop back to step 2
   - If analyze says **pivot**:
     - Discard speculative proposals. Apply pivot adjustments → invoke hp-tune synchronously → loop back to step 2
     **Pivot dispatch by type:**
     - `"branch_test"`: Pass analyze's suggestion to hp-tune. Generate configs for untested branches with baseline HPs. No research needed.
     - `"hp_expand"`: Widen the search space around the best config (extend LR range by 2× in each direction). Pass updated `search_space` to hp-tune.
     - `"narrow_space"`: Constrain the search space to the range around the best result (analyze's `suggestion` field contains bounds). Pass narrowed `search_space` to hp-tune.
     - `"regularization"`: Add regularization HPs (weight_decay, dropout) to the search space or expand their range. Pass updated `search_space` to hp-tune. No research needed.
     - `"research"`: Route to step 7 (same as `method_proposal`). Requires `remaining_budget >= 5`.
     - `"method_proposal"`, `"qualitative_change"`: Route to step 7 (existing handling). Requires `remaining_budget >= 3`.
     - **Unknown pivot_type:** Treat as `"hp_expand"` (safest default). Log to error tracker.
   - If analyze says **stop**:
     - Discard speculative proposals.
     - In `"auto"` or `"custom"` mode: exit loop
     - In `"autonomous"` mode: log the stop recommendation but continue the loop. Increment `consecutive_stop_count` in pipeline state (reset to 0 on "continue" or "pivot"). Persist via `save_state()` at the end of each iteration. On pipeline resume, read from state (default 0).
       - **On 3 consecutive stops → Stuck Protocol** (instead of immediate exit):
         If `consecutive_stop_count >= 3` AND `stuck_protocol_triggered` is false:
         1. Set `stuck_protocol_triggered = true` in pipeline state
         2. Read error patterns: `python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> patterns`
         3. Read success metrics: `python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> success <primary_metric> <lower_is_better>`
         4. Read dead-end catalog: `python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> dead-end list`
         5. Read research agenda: `python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> agenda list`
         6. Dispatch the research agent with all failure context:
            ```
            Agent(
              description: "Stuck protocol — find new approaches",
              prompt: "Ultrathink. The optimization is stuck — 3 consecutive batches showed no improvement. Find new approaches that haven't been tried. Parameters: source: both, model_type: {model_type}, task: {task}, current_metrics: {best_metrics}, problem_description: {problem_description}, exp_root: {exp_root}, scope_level: {method_proposal_scope or 'architecture'}. CONTEXT: Error patterns: {patterns}. Success metrics: {success}. Dead ends (DO NOT re-propose): {dead_ends}. Research agenda: {agenda}. Focus on techniques NOT in the dead-end catalog.",
              subagent_type: "ml-optimizer:research-agent"
            )
            ```
         7. If research returns new proposals (not all deduplicated against dead ends):
            - Route to step 7 (mid-loop method proposal trigger) for implementation
            - Reset `consecutive_stop_count` to 0
            - Log: "Stuck protocol succeeded — {N} new proposals found. Resuming loop."
            - Continue loop
         8. If research returns no new proposals: exit loop. Log: "Stuck protocol exhausted — no new proposals found."
       - **On 3 consecutive stops after stuck protocol already triggered:** Exit loop immediately (prevents infinite recovery loops).
   - **If analyze output is malformed or contains an unexpected action:** Treat as `agent_failure`. Log to error tracker. Retry analyze once with a simplified prompt: "Based on the experiment results, should we continue, pivot, or stop? Respond with exactly one of: continue, pivot, stop." If retry also fails, default to `continue` if remaining_budget > 0, or `stop` if budget exhausted.
   - **Safety limit:** Maximum experiments budget depends on `budget_mode` from Phase 1:
     - `"auto"` (default): `max(num_gpus, 1) × difficulty_multiplier` (easy=8, moderate=15, hard=25)
     - `"custom"`: user-specified `custom_budget`
     - `"autonomous"`: 999 (effectively unlimited — runs until interrupted or 3 consecutive stop recommendations)
     After budget exhausted, force exit and report. When `num_gpus=0` (CPU-only, e.g., scikit-learn), the multiplier applies to `1`.

7. **Mid-loop method proposal trigger** (when analyze recommends new methods):

   If analyze returns `pivot_type: "method_proposal"` or `pivot_type: "qualitative_change"`:

   a. **Budget gate:** If `remaining_budget < 3`, skip method proposals and recommend stop with current best result. Log:
      ```bash
      python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"orchestrate","message":"Method proposals skipped: remaining_budget (<N>) < 3","phase":7,"iteration":<iteration>}'
      ```

   b. **Scope confirmation:**
      **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, use stored `method_proposal_scope` from user_choices. Log to dev_notes: "Autonomous mode: using stored method_proposal_scope '<scope>' for mid-loop proposals". Skip AskUserQuestion.

      **Otherwise:** Ask the user which scope level to use:
      ```
      HP tuning has plateaued. I can propose new optimization methods.

      Scope options:
      1. Training strategies only (optimizers, schedulers, regularization, augmentation, loss functions)
      2. Training + architecture changes (attention, normalization, activations, block design)
      3. Full scope (training + architecture + data pipeline, distillation, ensemble)
      4. Skip — stop with current best result

      Which scope? (1/2/3/4)
      ```
      If user chooses 4 (skip), exit the loop and proceed to Phase 9 (report).

   c. **Generate proposals:** Dispatch the research agent:
      ```
      Agent(
        description: "Mid-loop research proposals",
        prompt: "Ultrathink. Research ML optimization techniques. Parameters: source: both, scope_level: {scope_level}, output_path: experiments/reports/research-findings-method-proposals-iter{N}.md, model_type: {model_type}, task: {task}, current_metrics: {current_metrics}, problem_description: {problem_description}, exp_root: {exp_root}.",
        subagent_type: "ml-optimizer:research-agent"
      )
      ```

   d. **Present proposals:**
      **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, accept all proposals automatically. Log to dev_notes: "Autonomous mode: auto-accepted all N mid-loop method proposals". Skip AskUserQuestion.

      **Otherwise:** Show the generated proposals to the user for confirmation. The user can accept all, select a subset, or reject all (which exits the loop).

   e. **Implement proposals:** Dispatch the implement agent with the confirmed method proposal findings. This creates new `ml-opt/<slug>` branches.
      ```
      Agent(
        description: "Implement mid-loop proposals",
        prompt: "Ultrathink. Implement research proposals. Parameters: findings_path: {findings_path}, selected_indices: {selected_indices}, project_root: {project_root}.",
        subagent_type: "ml-optimizer:implement-agent"
      )
      ```

   f. **Merge into experiment loop:** Add the new validated branches to `code_branches`. Reset the iteration counter for these new branches only (they start at iteration 1 = `method_default_hp` tier). Existing branches keep their iteration count.

   g. **Update state:**
      - Increment `method_proposal_iterations` in user_choices
      - Deduct expected cost from `remaining_budget`: `expected_cost = len(new_branches) * max(num_gpus, 1)` (accounts for GPU-parallel batch sizing per branch)
      - Save pipeline state

   h. **Continue loop:** Loop back to step 1 (hp-tune) with the expanded `code_branches` list. Reset `batches_since_last_research = 0`.

8. **Research round check** (autonomous mode only — cadence-based research trigger):

   This step auto-triggers research → implement on a regular cadence, independent of analyze's pivot recommendation. It only applies in autonomous mode with `method_proposal_scope` set.

   **Conditions (ALL must be true):**
   - `budget_mode == "autonomous"`
   - `method_proposal_scope` is set (user opted into method proposals)
   - `batches_since_last_research >= hp_batches_per_round`
   - Step 7 did NOT already trigger this iteration (avoid double research)

   **If conditions met:**

   a. **Log the trigger:**
      ```bash
      python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"orchestrate","message":"Autonomous research round triggered after <N> HP batches","phase":7,"iteration":<iteration>,"context":{"batches_since_last_research":<N>,"method_proposal_iterations":<M>}}'
      ```

   b. **Generate proposals:** Dispatch the research agent:
      ```
      Agent(
        description: "Autonomous research round",
        prompt: "Ultrathink. Research ML optimization techniques. Parameters: source: both, scope_level: {method_proposal_scope}, output_path: experiments/reports/research-findings-method-proposals-iter{N}.md, model_type: {model_type}, task: {task}, current_metrics: {current_metrics}, problem_description: {problem_description}, exp_root: {exp_root}.",
        subagent_type: "ml-optimizer:research-agent"
      )
      ```

   c. **Check results:**
      - If research returns new proposals (not all filtered by deduplication): proceed to implement
      - If research returns **no new proposals** (all deduplicated): skip implement, double `hp_batches_per_round` (exponential backoff), log:
        ```bash
        python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"orchestrate","message":"Research round yielded no new proposals — increasing cadence to <new_value> batches","phase":7,"iteration":<iteration>}'
        ```

   d. **Implement proposals (no user confirmation):** In autonomous mode, ALL returned proposals are implemented automatically (the user opted into autonomous operation). Dispatch the implement agent with the research findings. This creates new `ml-opt/<slug>` branches.
      ```
      Agent(
        description: "Implement autonomous research proposals",
        prompt: "Ultrathink. Implement research proposals. Parameters: findings_path: {findings_path}, selected_indices: {all_indices}, project_root: {project_root}.",
        subagent_type: "ml-optimizer:implement-agent"
      )
      ```

   e. **Merge into experiment loop:** Same as step 7f — add new validated branches to `code_branches`, reset iteration counter for new branches.

   f. **Update state:**
      - Increment `method_proposal_iterations`
      - Reset `batches_since_last_research = 0`
      - Save pipeline state

   **If conditions NOT met:** Increment `batches_since_last_research` and continue.

9. **Mid-pipeline review check** (after step 6/7/8, before looping):
   Run pattern detection:
   ```bash
   python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> patterns
   ```
   If `wasted_budget` pattern has occurrences ≥ 3, OR if the last 2 consecutive batches both had zero successful experiments:

   **If `budget_mode == "autonomous"`:**
   - Start review in background via Agent with `run_in_background: true`:
     ```
     Agent(
       description: "Async mid-pipeline review",
       prompt: "Ultrathink. Run a mid-pipeline review. Parameters: project_root: {project_root}, exp_root: {exp_root}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, scope: session.",
       subagent_type: "ml-optimizer:review-agent",
       run_in_background: true
     )
     ```
   - Continue to next loop iteration immediately (do NOT wait for review)
   - At the START of the next loop iteration (before step 1), check if the background review has completed:
     - If completed successfully: read suggestions and apply course corrections (narrow search space, prune branches, stop)
     - If completed with error or no output: log as `agent_failure` to error tracker, skip course corrections for this iteration
     - If still running: continue without waiting (suggestions will be applied in the following iteration)
   - **Design note:** Applying review suggestions one batch late is intentional — blocking the pipeline to wait for review would waste GPU time. Review suggestions are strategic (search space narrowing, branch pruning), so a one-batch delay is acceptable.
   - Log: "Mid-pipeline review started in background (autonomous mode)"

   **Otherwise (interactive/auto/custom):**
   - Dispatch review agent synchronously:
     ```
     Agent(
       description: "Mid-pipeline review",
       prompt: "Ultrathink. Run a mid-pipeline review. Parameters: project_root: {project_root}, exp_root: {exp_root}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, scope: session.",
       subagent_type: "ml-optimizer:review-agent"
     )
     ```
   - Read the review output's top suggestions
   - Apply relevant course corrections:
     - If review suggests narrowing LR range: pass constrained `search_space` to hp-tune
     - If review suggests pruning a branch: remove it from `code_branches`
     - If review suggests stopping: follow the stop recommendation
   - Log the mid-pipeline review:
     ```bash
     python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"orchestrate","message":"Mid-pipeline review triggered after consecutive failures","phase":7,"iteration":<iteration>,"context":{"trigger":"consecutive_failures"}}'
     ```

10. **Loop back:** After steps 6/7/8/9, increment `batches_since_last_research` and return to step 1 (Get HP configs). The loop continues until the Decision step (6) or budget exhaustion forces an exit.

## Speculative Proposal Validation

Before using speculative proposals from a previous iteration's background hp-tune, verify ALL of these:

1. **Branch validity:** All `code_branch` values in proposals still exist in the active branch list (none were pruned by analyze)
2. **Budget compliance:** Number of proposals ≤ `remaining_budget`
3. **No search space conflict:** If analyze recommended narrowing the search space (via pivot), check that speculative proposals fall within the new bounds
4. **No duplicates:** Speculative proposals don't duplicate experiments from the just-completed batch

If ANY check fails, discard ALL speculative proposals and invoke hp-tune synchronously with updated parameters.

## Parallel GPU Dispatch Pattern

When dispatching experiments across multiple GPUs, use the Agent tool with `subagent_type: "ml-optimizer:experiment-agent"` for each experiment.

**If manifest strategy is `"file_backup"` (non-git project):** dispatch ONE experiment at a time (sequential). Wait for each to complete before starting the next. File-backup proposals share the same working directory and cannot run in parallel.

**Otherwise (git_branch strategy or HP-only):** dispatch all experiments in parallel:

```
For each config in proposed_configs:
  Agent(
    description: "Run experiment {exp_id}",
    prompt: "Run experiment {exp_id} with config: {config_json}. GPU: {gpu_id}. Project root: {project_root}. Train command: {train_command}. Eval command: {eval_command or null}. Code branch: {code_branch or null}. Code proposal: {code_proposal or null}. Proposal source: {proposal_source or null}. Method tier: {method_tier or null}. Iteration: {iteration}. Prepared train path: {prepared_train_path or null}. Prepared val path: {prepared_val_path or null}.",
    subagent_type: "ml-optimizer:experiment-agent",
    run_in_background: true
  )
```

Then wait for all agents to complete before invoking analyze.

## Thinking Depth for Agent Dispatch

When dispatching agents via the Agent tool, include "ultrathink" in the prompt for **analytical** agents (hp-tune, research, analyze, implement) to trigger maximum reasoning depth. Do NOT include it for **procedural** agents (experiment, monitor) — these are execution-focused and don't benefit from extended thinking.

Example for analytical dispatch:
```
Agent(
  description: "Analyze batch {N} results",
  prompt: "Ultrathink. Analyze batch {N} results. Parameters: project_root: {project_root}. Primary metric: {primary_metric}. Lower is better: {lower_is_better}. Target: {target_value or null}. Remaining budget: {remaining_budget}.",
  subagent_type: "ml-optimizer:analysis-agent"
)
```
