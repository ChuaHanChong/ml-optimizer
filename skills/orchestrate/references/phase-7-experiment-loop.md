# Phase 7: Experiment Loop (Autonomous)

This loop runs autonomously without user checkpoints until complete or blocked.

## Table of Contents

- [Pre-Loop: Validate Pipeline State](#pre-loop-validate-pipeline-state) (line 28)
- [Pre-Loop: Verify Baseline Integrity](#pre-loop-verify-baseline-integrity) (line 46)
- [Pre-Loop: Sync Behavioral Memory](#pre-loop-sync-behavioral-memory) (line 61)
- [Pre-Loop: Initialize Code Archive](#pre-loop-initialize-code-archive) (line 69)
- [Pre-Loop: Load Meta-Patches](#pre-loop-load-meta-patches) (line 80)
- [Pre-Loop: Load Implementation Manifest](#pre-loop-load-implementation-manifest) (line 97)
- [Pre-Loop: Method Proposals](#pre-loop-method-proposals) (line 109)
- [Pre-Loop: Route hp_only Research Proposals](#pre-loop-route-hp_only-research-proposals) (line 183)
- [Pre-Loop: Initialize Research Cadence](#pre-loop-initialize-research-cadence) (line 191)
- [Pre-Loop: Save Pipeline State](#pre-loop-save-pipeline-state) (line 198)
- [Metric Routing Rule](#metric-routing-rule) (line 229)
- [Polarity Conflict Rule](#polarity-conflict-rule) (line 239)
- [Branch Dispatch Strategy](#branch-dispatch-strategy) (line 246)
- [Loop Iteration](#loop-iteration) (line 254) — main experiment loop (steps 1-7)
- [Parallel GPU Dispatch Pattern](#parallel-gpu-dispatch-pattern) (line 729)
- [Thinking Depth for Agent Dispatch](#thinking-depth-for-agent-dispatch) (line 749)
- [Hyperagent Driven Loop](#hyperagent-driven-loop) (line 764) — mandatory hyperagent dispatch

## Pre-Loop: Validate Pipeline State

Before starting the experiment loop, validate all prerequisites:

```bash
python3 -c "
import sys; # sys.path: add the plugin's scripts/ directory
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
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/pipeline_state.py <exp_root> verify-baseline
```

If exit code is non-zero (baseline checksum mismatch): **HALT the pipeline immediately.** Log to error tracker:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"config_error","severity":"critical","source":"orchestrate","message":"Baseline integrity check FAILED — metrics may have been modified. Pipeline halted.","phase":7}'
```
Report the error to the user. Do NOT continue — all experiment comparisons would be invalid.

If the verification returns a warning (legacy pipeline without checksum): log to dev_notes and continue normally.

## Pre-Loop: Sync Behavioral Memory

Before starting experiments, sync behavioral patterns from the error tracker:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> sync-from-errors
```
This populates `experiments/learned-behaviors.json` with OOM limits, divergence patterns, and dead-end outcomes from the error tracker. All agents will read this via the `summary` command.

## Pre-Loop: Initialize Code Archive

Initialize the evolutionary archive from baseline and existing branches:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/skills/hyperagent-init/scripts/init_archive.py --output-dir <exp_root>/hyperagent
```

This creates gen-000 from baseline and seeds any validated implementation branches from Phase 6 as gen-001, gen-002, etc. Every experiment result from this point forward updates the archive.

**Standing on the shoulders of giants:** Phase 5 (research) and Phase 6 (implement) are dedicated pre-loop phases that remain critical. They seed the archive with proven techniques from papers BEFORE the experiment loop starts. The hyperagent benefits enormously from this foundation — Phase 6 branches give it diverse starting points to build upon. User-provided papers (from Phase 0) flow through Phase 5 with highest priority. During the loop, the hyperagent can dispatch additional research-implement rounds, building FROM the best parent in the archive (not baseline).

## Pre-Loop: Load Meta-Patches (if meta-improvement has run)

If `hyperagent_state.active_meta_patches` is non-empty in pipeline state, the hyperagent has modified skill instructions for this session. Before every subsequent agent dispatch (hp-tune, analyze, research), prepend to the dispatch prompt:

```
META-PATCHES ACTIVE: The hyperagent has modified skill instructions for this session.
For the following skills, read the patched version INSTEAD of your default skill instructions:
<for each patch in active_meta_patches>
  - <skill_name>: Read experiments/meta-patches/<skill_name>-SKILL.md
    Change summary: <from meta-changelog.json patches[].change>
</for each>
```

To build this context, read `experiments/meta-patches/meta-changelog.json` and extract the `patches` array. Each entry has `skill`, `change`, `reason`, and `expected_impact`.

This enables the self-referential loop: the hyperagent's strategy improvements are applied to future agent dispatches within the same session.

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

1. **Dispatch the research agent** (resume-or-dispatch pattern):

   **IF `agent_registry["research"]` is not null** (agent exists from Phase 5):
   ```
   SendMessage(
     to: agent_registry["research"],
     message: "Research method proposals (pre-loop).
       CONTEXT FROM OTHER AGENTS:
       - BASELINE: current_metrics={current_metrics}
       Parameters: source: both, scope_level: {method_proposal_scope}, output_path: experiments/reports/research-findings-method-proposals.md, model_type: {model_type}, task: {task}, current_metrics: {current_metrics}, problem_description: {problem_description}, exp_root: {exp_root}."
   )
   ```
   → If `SendMessage` fails: fall back to the `Agent()` dispatch below, update `agent_registry["research"]`.

   **ELSE** (first dispatch — no existing agent, e.g., user skipped Phase 5):
   ```
   Agent(
     description: "Research method proposals",
     prompt: "Ultrathink. Research ML optimization techniques. Parameters: source: both, scope_level: {method_proposal_scope}, output_path: experiments/reports/research-findings-method-proposals.md, model_type: {model_type}, task: {task}, current_metrics: {current_metrics}, problem_description: {problem_description}, exp_root: {exp_root}.",
     subagent_type: "ml-optimizer:research-agent"
   )
   ```
   → Save returned `agentId` to `agent_registry["research"]`
   → Persist registry: `save_state(..., agent_registry=agent_registry)`

2. **Present proposals to user for confirmation** (same as Phase 5 post-research checkpoint):

   ```
   Method proposals (from LLM knowledge + web search):
   [summary of proposals from research-findings-method-proposals.md]

   Which proposals should I pursue?
   - [1] Proposal A (complexity: low, expected: +X%)
   - [2] Proposal B (complexity: medium, expected: +Y%)
   - [3] Custom: describe your own approach
   - [4] Skip, just tune HPs on existing code
   ```

3. **If user selects proposals:** Dispatch the implement agent (resume-or-dispatch pattern):

   **IF `agent_registry["implement"]` is not null** (agent exists from Phase 6):
   ```
   SendMessage(
     to: agent_registry["implement"],
     message: "Implement method proposals (pre-loop).
       CONTEXT FROM OTHER AGENTS:
       - RESEARCH: found proposals in experiments/reports/research-findings-method-proposals.md
       Parameters: findings_path: experiments/reports/research-findings-method-proposals.md, selected_indices: {selected_indices}, project_root: {project_root}."
   )
   ```
   → If `SendMessage` fails: fall back to the `Agent()` dispatch below, update `agent_registry["implement"]`.

   **ELSE** (first dispatch — no existing agent, e.g., user skipped Phase 6):
   ```
   Agent(
     description: "Implement method proposals",
     prompt: "Ultrathink. Implement research proposals. Parameters: findings_path: experiments/reports/research-findings-method-proposals.md, selected_indices: {selected_indices}, project_root: {project_root}.",
     subagent_type: "ml-optimizer:implement-agent"
   )
   ```
   → Save returned `agentId` to `agent_registry["implement"]`
   → Persist registry: `save_state(..., agent_registry=agent_registry)`

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

Initialize the research round counter:
- `batches_since_last_research = 0`
- This counter tracks how many HP tuning batches have run since the last research → implement cycle
- When this counter reaches `hp_batches_per_round`, step 8 auto-triggers a new research round

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
- **Iterations 3+:** Focus on the best branch + HP tuning. The analysis agent judges which branches are competitive with the overall best and which should be dropped — no fixed percentage cutoff.

## Loop Iteration:

1. **Get HP configs:**
   Dispatch the tuning agent (resume-or-dispatch pattern):

     **IF `agent_registry["tuning"]` is not null** (agent exists from a previous iteration):
     ```
     SendMessage(
       to: agent_registry["tuning"],
       message: "HP tuning iteration {iteration}.
         CONTEXT FROM OTHER AGENTS:
         - ANALYZE (batch {N-1}): {recommendation}, correlations: {correlations}, branch_scores: {scores}
         - MONITOR: max_batch_size={max_batch_size} (OOM constraint)
         Parameters: project_root: {project_root}, num_gpus: {num_gpus}, search_space: {search_space}, iteration: {iteration}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, code_branches: {code_branches}, max_batch_size: {max_batch_size or omit}, warm_start_enabled: {warm_start_enabled or false}, available_checkpoints: {available_checkpoints_json or {}}, branch_scores: {branch_scores_json or {}}."
     )
     ```
     → If `SendMessage` fails (agent no longer reachable): fall back to the `Agent()` dispatch below, update `agent_registry["tuning"]` with the new agentId.

     **ELSE** (first dispatch — no existing agent):
     ```
     Agent(
       description: "HP tuning iteration {iteration}",
       prompt: "Ultrathink. Propose HP configurations. Parameters: project_root: {project_root}, num_gpus: {num_gpus}, search_space: {search_space}, iteration: {iteration}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, code_branches: {code_branches}, max_batch_size: {max_batch_size or omit}, warm_start_enabled: {warm_start_enabled or false}, available_checkpoints: {available_checkpoints_json or {}}, branch_scores: {branch_scores_json or {}}.",
       subagent_type: "ml-optimizer:tuning-agent"
     )
     ```
     → Save returned `agentId` to `agent_registry["tuning"]`
     → Persist registry: `save_state(..., agent_registry=agent_registry)`

     - `code_branches`: From implementation manifest, or `[]` for HP-only.
     - `max_batch_size` *(optional)*: One step below the smallest OOM-causing batch size. Omit if no OOM events.
   - It reads past results and proposes the next batch of configs
   - Number of configs = `max(num_gpus, 1)`
   - **Check hp-tune recommendation:** If hp-tune output includes `"recommendation": "stop"`, log it to error tracker with `category: "pipeline_inefficiency"` and note it for the analyze step. Analyze makes the final continue/pivot/stop decision, but hp-tune's recommendation provides an early signal of search space exhaustion.

   ### HP-Tune Failure Recovery

   If hp-tune crashes or produces invalid configs:

   1. **Validate output:** Check each proposed config has required fields (`exp_id`, `config`, `gpu_id`), values are within search space bounds, and no duplicates of previously-tried configs.
   2. **If validation fails:** Retry hp-tune once with a simplified prompt: "Propose {N} configs within these ranges: {search_space}. Return valid JSON only."
   3. **If retry also fails:** Fall back to random sampling — pick `lr` uniformly from search space log-range, `batch_size` from allowed set, other HPs at baseline values. The orchestrator constructs the JSON directly.
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
   - For each proposed config, invoke `ml-optimizer:experiment` skill
   - Pass `code_branch` and `code_proposal` from the manifest (or null for HP-only)
   - If multiple GPUs available, dispatch experiments in parallel using the Agent tool
   - Each experiment runs on a separate GPU

3. **Monitor experiments:**
   - **If `divergence_metric` is not null**, dispatch the monitor agent (resume-or-dispatch pattern):

     **IF `agent_registry["monitor"]` is not null** (agent exists from a previous batch):
     ```
     SendMessage(
       to: agent_registry["monitor"],
       message: "Monitor new batch of experiments.
         CONTEXT FROM OTHER AGENTS:
         - HP-TUNE: proposed configs with LR range {lr_range}
         Parameters: log_files: {log_files}, exp_ids: {exp_ids}, project_root: {project_root}, poll_interval: 30, metric_to_watch: {divergence_metric}, lower_is_better: {divergence_lower_is_better}, model_category: {model_category}."
     )
     ```
     → If `SendMessage` fails: fall back to the `Agent()` dispatch below, update `agent_registry["monitor"]` with the new agentId.

     **ELSE** (first dispatch — no existing agent):
     ```
     Agent(
       description: "Monitor experiments for divergence",
       prompt: "Monitor running experiments. Parameters: log_files: {log_files}, exp_ids: {exp_ids}, project_root: {project_root}, poll_interval: 30, metric_to_watch: {divergence_metric}, lower_is_better: {divergence_lower_is_better}, model_category: {model_category}.",
       subagent_type: "ml-optimizer:monitor-agent"
     )
     ```
     → Save returned `agentId` to `agent_registry["monitor"]`
     → Persist registry: `save_state(..., agent_registry=agent_registry)`
   - Monitor status handling:
     - `healthy`: Training is progressing normally — continue waiting
     - `diverged`: Stop the experiment automatically, record divergence reason in experiment results
     - `completed`: Training finished naturally during monitoring — proceed to wait/analysis
     - `unmonitored`: The watched metric was not found in the logs after all fallback attempts. Warn the user once (via dev_notes) that divergence monitoring is disabled for this experiment. Continue without divergence checks — rely on the experiment's hard timeout (from baseline profiling) as the safety net.
     - `failed`: Monitor itself encountered an error — log as `agent_failure`, continue without monitoring for remaining experiments in this batch
     - `no_output`: Log file has no parseable data yet — continue monitoring (normal for early training)
   - **If `divergence_metric` is null** (tabular ML — scikit-learn, XGBoost, LightGBM): skip the monitor skill entirely. Wait for experiments to complete naturally without divergence monitoring.

   ### Early Batch Abort on Mass Divergence

   The monitor agent handles each experiment individually — killing diverging experiments as they're detected. The analysis agent will see the results (including diverged experiments) and recommend appropriate action (narrow search space, etc.).

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
   - **Incremental dashboard updates:** As individual experiments complete, regenerate the live dashboard so users can monitor progress in real-time:
     ```bash
     python3 ${CLAUDE_PLUGIN_ROOT}/scripts/dashboard.py <exp_root> --live
     ```
     Log to dev_notes: "Experiment {exp_id} completed mid-batch: {primary_metric}={value}"
   - Save pipeline state after each batch completes

   ### Post-Batch Completeness Validation

   Before proceeding to analysis, validate all experiment results from this batch with `--strict` (enforces completeness — completed experiments must have non-empty metrics, iteration, method_tier, duration_seconds):

   ```bash
   for exp_id in <batch_exp_ids>:
       python3 schema_validator.py experiments/results/${exp_id}.json result --strict
   ```

   If any validation fails:
   1. Log to error tracker: `category: "config_error", severity: "warning", source: "orchestrate", message: "Experiment <exp_id> result incomplete: <errors>"`
   2. If the experiment agent is still reachable, ask it to fill missing fields
   3. Otherwise set `status: "failed"` in the result file with `notes: "Completeness validation failed: <errors>"`
   4. Continue — analyze will skip failed experiments

5. **Analyze results:**
   Dispatch the analysis agent (resume-or-dispatch pattern):

     **IF `agent_registry["analysis"]` is not null** (agent exists from a previous batch):
     ```
     SendMessage(
       to: agent_registry["analysis"],
       message: "Analyze batch {N} results.
         CONTEXT FROM OTHER AGENTS:
         - HP-TUNE: {config_summary}
         - MONITOR: {divergence_count} diverged
         - EXPERIMENTS: {completed}/{total} completed
         Parameters: project_root: {project_root}, batch_number: {batch_number}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, target_value: {target_value or null}."
     )
     ```
     → If `SendMessage` fails: fall back to the `Agent()` dispatch below, update `agent_registry["analysis"]` with the new agentId.

     **ELSE** (first dispatch — no existing agent):
     ```
     Agent(
       description: "Analyze batch {N} results",
       prompt: "Ultrathink. Analyze batch {N} results. Parameters: project_root: {project_root}, batch_number: {batch_number}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, target_value: {target_value or null}.",
       subagent_type: "ml-optimizer:analysis-agent"
     )
     ```
     → Save returned `agentId` to `agent_registry["analysis"]`
     → Persist registry: `save_state(..., agent_registry=agent_registry)`

     - It compares all experiments, ranks them, identifies patterns
     - It recommends: continue, pivot, or stop
   - **Live dashboard update:** After analyze completes, regenerate the dashboard so users can monitor progress in real-time:
     ```bash
     python3 ${CLAUDE_PLUGIN_ROOT}/scripts/dashboard.py <exp_root> --live
     ```
   - **Goal validation (post-dispatch):** After analyze returns:
     ```bash
     python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> validate-output analyze '<analyze_output_json>'
     ```
     If metric mismatch detected: log a critical error to the error tracker. Do NOT trust the analysis — the metric confusion could cause wrong ranking of experiments.

5b. **Mid-run goal adjustment (if user provides input):**

   The user can type goal changes at any time during the session. If the user has provided input after the batch analysis is presented (e.g., "change target to 85%" or "freeze learning rate at 0.01"), process it as a goal update:

   1. Parse the user's request into a goal update
   2. Apply via:
      ```bash
      python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> update-goals '<updates_json>'
      ```
      Example updates:
      - Target: `{"objective": {"target_value": 85.0}}`
      - Metric: `{"objective": {"primary_metric": "f1", "lower_is_better": false}}`
      - Freeze: `{"constraints": {"frozen_parameters": ["lr"]}}`
      - Scope: `{"constraints": {"scope_level": "architecture"}}`
   3. Update `user_choices` in pipeline state to match
   4. Log to dev_notes: "Mid-run goal update at iteration {N}: {changes}"
   5. The next hp-tune, research, and analyze dispatches will use the updated goals

   If the user hasn't typed anything, continue immediately — no checkpoint, no pause.

   **Important:** If the user changes `primary_metric`, all future experiment rankings use the new metric. Past experiments are NOT re-ranked — the change applies going forward only.

6. **Decision:**
   - If analyze says **continue**:
     - Invoke hp-tune → loop back to step 2
   - If analyze says **pivot**:
     - Apply pivot adjustments → invoke hp-tune → loop back to step 2
     **Pivot dispatch by type:**
     - `"branch_test"`: Pass analyze's suggestion to hp-tune. Generate configs for untested branches with baseline HPs. No research needed.
     - `"hp_expand"`: Widen the search space around the best config (extend LR range by 2× in each direction). Pass updated `search_space` to hp-tune.
     - `"narrow_space"`: Constrain the search space to the range around the best result (analyze's `suggestion` field contains bounds). Pass narrowed `search_space` to hp-tune.
     - `"regularization"`: Add regularization HPs (weight_decay, dropout) to the search space or expand their range. Pass updated `search_space` to hp-tune. No research needed.
     - `"research"`: Route to step 7 (same as `method_proposal`).
     - `"method_proposal"`, `"qualitative_change"`: Route to step 7 (existing handling).
     - **Unknown pivot_type:** Treat as `"hp_expand"` (safest default). Log to error tracker.
   - If analyze says **stop**:
     - **Do NOT exit the loop.** The loop is autonomous — only the user or target achievement stops it.
     - Log the stop recommendation. The hyperagent decides what to do next based on evidence.

     **Options the hyperagent can choose from:**
     - Switch to an operator it hasn't tried recently (e.g., research-implement if only HP tuning was done, ShinkaEvolve if only LLM patches were tried)
     - Invoke the **Stuck Protocol** (below) to systematically search for new approaches
     - Try meta-improvement to change the optimization strategy itself
     - The hyperagent reads operator stats, archive trends, and error patterns to decide which option is best

     **Stuck Protocol** (available when the hyperagent judges the optimization is stuck):

     The hyperagent can invoke this when it has evidence that current approaches are exhausted — for example, multiple operators tried with no improvement, or the archive shows a clear plateau. This is a tool the hyperagent chooses to use, not an automatic trigger.

       1. Read error patterns: `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> patterns`
       2. Read success metrics: `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> success <primary_metric> <lower_is_better>`
       3. Read dead-end catalog: `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> dead-end list`
       4. Read research agenda: `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> agenda list`
       5. Dispatch the research agent with all failure context (resume-or-dispatch pattern):

          **IF `agent_registry["research"]` is not null** (agent exists from a previous research round):
          ```
          SendMessage(
            to: agent_registry["research"],
            message: "Stuck protocol — find new approaches. The optimization is stuck.
              CONTEXT FROM OTHER AGENTS:
              - Error patterns: {patterns}
              - Success metrics: {success}
              - Dead ends (DO NOT re-propose): {dead_ends}
              - Research agenda: {agenda}
              Parameters: source: both, model_type: {model_type}, task: {task}, current_metrics: {best_metrics}, problem_description: {problem_description}, exp_root: {exp_root}, scope_level: {method_proposal_scope or 'architecture'}. Focus on techniques NOT in the dead-end catalog."
          )
          ```
          → If `SendMessage` fails: fall back to the `Agent()` dispatch below, update `agent_registry["research"]` with the new agentId.

          **ELSE** (first dispatch — no existing agent):
          ```
          Agent(
            description: "Stuck protocol — find new approaches",
            prompt: "Ultrathink. The optimization is stuck. Find new approaches that haven't been tried. Parameters: source: both, model_type: {model_type}, task: {task}, current_metrics: {best_metrics}, problem_description: {problem_description}, exp_root: {exp_root}, scope_level: {method_proposal_scope or 'architecture'}. CONTEXT: Error patterns: {patterns}. Success metrics: {success}. Dead ends (DO NOT re-propose): {dead_ends}. Research agenda: {agenda}. Focus on techniques NOT in the dead-end catalog.",
            subagent_type: "ml-optimizer:research-agent"
          )
          ```
          → Save returned `agentId` to `agent_registry["research"]`
          → Persist registry: `save_state(..., agent_registry=agent_registry)`
       6. If research returns new proposals → route to step 7 for implementation, continue loop
       7. If no new proposals → try other operators (LLM patches, ShinkaEvolve, meta-improvement)
   - **If analyze output is malformed or contains an unexpected action:** Treat as `agent_failure`. Log to error tracker. Retry analyze once with a simplified prompt: "Based on the experiment results, should we continue, pivot, or stop? Respond with exactly one of: continue, pivot, stop." If retry also fails, default to `continue`.
   - **Loop exit conditions:** The experiment loop is **autonomous by default** — it runs non-stop until:
     1. Target metric achieved (from Phase 0)
     2. User manually stops
     The loop does NOT auto-stop on plateaus. Even when the analysis agent recommends "stop", the hyperagent should try different operators (research, LLM patches, ShinkaEvolve, meta-improvement) before giving up. The stuck protocol dispatches research for fresh ideas. Only the user can truly end the run — breakthroughs can come after plateaus.

7. **Mid-loop method proposal trigger** (when analyze recommends new methods):

   If analyze returns `pivot_type: "method_proposal"` or `pivot_type: "qualitative_change"`:

   a. **Scope confirmation:**
      If `method_proposal_scope` is already set in user_choices, use it. Otherwise, ask the user which scope level to use:
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

   b. **Generate proposals:** Dispatch the research agent (resume-or-dispatch pattern):

      **IF `agent_registry["research"]` is not null** (agent exists from a previous research round):
      ```
      SendMessage(
        to: agent_registry["research"],
        message: "Mid-loop research proposals.
          CONTEXT FROM OTHER AGENTS:
          - ANALYZE: pivot_type={pivot_type}, reason={reason}
          - EXPERIMENTS: best improvement={best_improvement}%
          - DEAD ENDS: {dead_end_catalog}
          Parameters: source: both, scope_level: {scope_level}, output_path: experiments/reports/research-findings-method-proposals-iter{N}.md, model_type: {model_type}, task: {task}, current_metrics: {current_metrics}, problem_description: {problem_description}, exp_root: {exp_root}."
      )
      ```
      → If `SendMessage` fails: fall back to the `Agent()` dispatch below, update `agent_registry["research"]` with the new agentId.

      **ELSE** (first dispatch — no existing agent):
      ```
      Agent(
        description: "Mid-loop research proposals",
        prompt: "Ultrathink. Research ML optimization techniques. Parameters: source: both, scope_level: {scope_level}, output_path: experiments/reports/research-findings-method-proposals-iter{N}.md, model_type: {model_type}, task: {task}, current_metrics: {current_metrics}, problem_description: {problem_description}, exp_root: {exp_root}.",
        subagent_type: "ml-optimizer:research-agent"
      )
      ```
      → Save returned `agentId` to `agent_registry["research"]`
      → Persist registry: `save_state(..., agent_registry=agent_registry)`

   **Goal validation (post-dispatch):** After research returns proposals:
   ```bash
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> validate-output research '<proposals_json>'
   ```
   If `valid` is false: remove scope-violating and dead-end proposals before passing to implement. Log violations to behavioral memory.

   c. **Present proposals:**
      Show the generated proposals to the user for confirmation. The user can accept all, select a subset, or reject all (which exits the loop).

   d. **Implement proposals:** Dispatch the implement agent with the confirmed method proposal findings (resume-or-dispatch pattern). This creates new `ml-opt/<slug>` branches.

      **IF `agent_registry["implement"]` is not null** (agent exists from a previous implementation round):
      ```
      SendMessage(
        to: agent_registry["implement"],
        message: "Implement mid-loop proposals.
          CONTEXT FROM OTHER AGENTS:
          - RESEARCH: found {N} proposals in {findings_path}
          - EXPERIMENTS: branches active: {code_branches}
          Parameters: findings_path: {findings_path}, selected_indices: {selected_indices}, project_root: {project_root}."
      )
      ```
      → If `SendMessage` fails: fall back to the `Agent()` dispatch below, update `agent_registry["implement"]` with the new agentId.

      **ELSE** (first dispatch — no existing agent):
      ```
      Agent(
        description: "Implement mid-loop proposals",
        prompt: "Ultrathink. Implement research proposals. Parameters: findings_path: {findings_path}, selected_indices: {selected_indices}, project_root: {project_root}.",
        subagent_type: "ml-optimizer:implement-agent"
      )
      ```
      → Save returned `agentId` to `agent_registry["implement"]`
      → Persist registry: `save_state(..., agent_registry=agent_registry)`

   e. **Merge into experiment loop:** Add the new validated branches to `code_branches`. Reset the iteration counter for these new branches only (they start at iteration 1 = `method_default_hp` tier). Existing branches keep their iteration count.

   **If `pivot_type == "code_evolution"` (evolutionary code improvement):**

   Instead of research → implement, the flow is: **tuning agent → implement agent → tuning agent → experiment agent**. The analysis agent's own conditions (|rho| < 0.3 + method improved) prevent unnecessary evolution — no artificial cooldown needed.

   a. **Tune evolve HPs:** Dispatch the tuning agent to propose ShinkaEvolve hyperparameters (resume-or-dispatch pattern):
      ```
      SendMessage(to: agent_registry["tuning"]) OR Agent(subagent_type="ml-optimizer:tuning-agent")
      ```
      Prompt: "Propose ShinkaEvolve evolution HPs for code_evolution. Read `learned-behaviors.json` category `evolve_hp` for prior evolution outcomes. Consider: how many generations produced the best mutation last time, whether more population diversity is needed, and the improvement trajectory. Return `evolve_recommendation: {num_generations, population_size, reasoning}`."

   b. **Execute evolve:** Dispatch the implement-agent with the evolve skill (resume-or-dispatch pattern):
      ```
      SendMessage(to: agent_registry["implement"]) OR Agent(subagent_type="ml-optimizer:implement-agent")
      ```
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
      - If `status == "shinkaevolve_unavailable"`: ShinkaEvolve couldn't be installed or crashed. Fall back to the research → implement path (step 7a-f above) instead. Log the failure.

   d. **Tune training HPs + run experiment:** Dispatch tuning agent to propose training HPs for the evolved branch, then dispatch experiment-agent. The evolved branch enters the normal HP-tune → experiment → analyze loop.

   d. **Update state:**
      - Increment `method_proposal_iterations` in user_choices
      - Save pipeline state

   e. **Continue loop:** Loop back to step 1 (hp-tune) with the expanded `code_branches` list. Reset `batches_since_last_research = 0`.

8. **Research round check** (cadence-based research trigger):

   This step auto-triggers research → implement on a regular cadence, independent of analyze's pivot recommendation. It applies when `method_proposal_scope` is set.

   **Conditions (ALL must be true):**
   - `method_proposal_scope` is set (user opted into method proposals)
   - `batches_since_last_research >= hp_batches_per_round`
   - Step 7 did NOT already trigger this iteration (avoid double research)

   **If conditions met:**

   a. **Log the trigger:**
      ```bash
      python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"orchestrate","message":"Cadence-based research round triggered after <N> HP batches","phase":7,"iteration":<iteration>,"context":{"batches_since_last_research":<N>,"method_proposal_iterations":<M>}}'
      ```

   b. **Generate proposals:** Dispatch the research agent (resume-or-dispatch pattern):

      **IF `agent_registry["research"]` is not null** (agent exists from a previous research round):
      ```
      SendMessage(
        to: agent_registry["research"],
        message: "Cadence-based research round.
          CONTEXT FROM OTHER AGENTS:
          - ANALYZE: {last_analysis_summary}
          - EXPERIMENTS: best improvement={best_improvement}%, branches active: {code_branches}
          - DEAD ENDS: {dead_end_catalog}
          Parameters: source: both, scope_level: {method_proposal_scope}, output_path: experiments/reports/research-findings-method-proposals-iter{N}.md, model_type: {model_type}, task: {task}, current_metrics: {current_metrics}, problem_description: {problem_description}, exp_root: {exp_root}."
      )
      ```
      → If `SendMessage` fails: fall back to the `Agent()` dispatch below, update `agent_registry["research"]` with the new agentId.

      **ELSE** (first dispatch — no existing agent):
      ```
      Agent(
        description: "Cadence-based research round",
        prompt: "Ultrathink. Research ML optimization techniques. Parameters: source: both, scope_level: {method_proposal_scope}, output_path: experiments/reports/research-findings-method-proposals-iter{N}.md, model_type: {model_type}, task: {task}, current_metrics: {current_metrics}, problem_description: {problem_description}, exp_root: {exp_root}.",
        subagent_type: "ml-optimizer:research-agent"
      )
      ```
      → Save returned `agentId` to `agent_registry["research"]`
      → Persist registry: `save_state(..., agent_registry=agent_registry)`

   c. **Check results:**
      - If research returns new proposals (not all filtered by deduplication): proceed to implement
      - If research returns **no new proposals** (all deduplicated): skip implement, double `hp_batches_per_round` (exponential backoff), log:
        ```bash
        python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"orchestrate","message":"Research round yielded no new proposals — increasing cadence to <new_value> batches","phase":7,"iteration":<iteration>}'
        ```

   d. **Implement proposals:** ALL returned proposals are implemented automatically. Dispatch the implement agent with the research findings (resume-or-dispatch pattern). This creates new `ml-opt/<slug>` branches.

      **IF `agent_registry["implement"]` is not null** (agent exists from a previous implementation round):
      ```
      SendMessage(
        to: agent_registry["implement"],
        message: "Implement cadence-based research proposals.
          CONTEXT FROM OTHER AGENTS:
          - RESEARCH: found {N} proposals in {findings_path}
          - EXPERIMENTS: branches active: {code_branches}
          Parameters: findings_path: {findings_path}, selected_indices: {all_indices}, project_root: {project_root}."
      )
      ```
      → If `SendMessage` fails: fall back to the `Agent()` dispatch below, update `agent_registry["implement"]` with the new agentId.

      **ELSE** (first dispatch — no existing agent):
      ```
      Agent(
        description: "Implement cadence-based research proposals",
        prompt: "Ultrathink. Implement research proposals. Parameters: findings_path: {findings_path}, selected_indices: {all_indices}, project_root: {project_root}.",
        subagent_type: "ml-optimizer:implement-agent"
      )
      ```
      → Save returned `agentId` to `agent_registry["implement"]`
      → Persist registry: `save_state(..., agent_registry=agent_registry)`

   e. **Merge into experiment loop:** Same as step 7f — add new validated branches to `code_branches`, reset iteration counter for new branches.

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

When dispatching experiments across multiple GPUs, use the Agent tool with `subagent_type: "ml-optimizer:experiment-agent"` for each experiment.

**If manifest strategy is `"file_backup"` (non-git project):** dispatch ONE experiment at a time (sequential). Wait for each to complete before starting the next. File-backup proposals share the same working directory and cannot run in parallel.

**Otherwise (git_branch strategy or HP-only):** dispatch all experiments in parallel:

```
For each config in proposed_configs:
  Agent(
    description: "Run experiment {exp_id}",
    prompt: "Run experiment {exp_id} with config: {config_json}. GPU: {gpu_id}. Project root: {project_root}. Train command: {train_command}. Eval command: {eval_command or null}. Code branch: {code_branch or null}. Code proposal: {code_proposal or null}. Proposal source: {proposal_source or null}. Method tier: {method_tier or null}. Iteration: {iteration}. Prepared train path: {prepared_train_path or null}. Prepared val path: {prepared_val_path or null}. Checkpoint source: {checkpoint_source_json or null}.",
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
  prompt: "Ultrathink. Analyze batch {N} results. Parameters: project_root: {project_root}. Primary metric: {primary_metric}. Lower is better: {lower_is_better}. Target: {target_value or null}.",
  subagent_type: "ml-optimizer:analysis-agent"
)
```

---

## Hyperagent Driven Loop

**MANDATORY: The hyperagent MUST be dispatched for Phase 7. It is the loop driver, not an optional enhancement. Do NOT implement a simplified HP-tune → experiment → analyze loop that bypasses the hyperagent. Do NOT skip the hyperagent dispatch for any reason — cost, complexity, or "simplicity". The hyperagent chooses operators (HP tuning, LLM patches, ShinkaEvolve, research-implement, meta-improvement), manages the evolutionary archive, and drives strategy. Without it, the plugin loses self-improvement, archive-based selection, and operator adaptation. If ShinkaEvolve is unavailable, the hyperagent falls back to other operators automatically — but the hyperagent itself is never optional.**

The hyperagent decides what action to take at each iteration — HP tuning, code mutation, research, or self-improvement. The analysis agent advises after each batch (continue, pivot direction, or stacking), and the hyperagent decides the specific action based on the advice + archive state + operator effectiveness. It also enables the plugin to self-improve by modifying skill instructions mid-session. When the analysis agent advises stacking, the hyperagent decides whether to proceed and transitions to Phase 8, then returns to Phase 7 on the stacked code.

### Hyperagent Dispatch (Loop Entry)

At the start of Phase 7, dispatch the hyperagent with the orchestrating hyperagent skill:

```
Agent(
  description: "Hyperagent optimization",
  prompt: "Ultrathink. Invoke Skill('ml-optimizer:hyperagent'). Run the optimization.
  Parameters: project_root: {project_root}. exp_root: {exp_root}. primary_metric: {primary_metric}. lower_is_better: {lower_is_better}. scope_level: {scope_level}. target_value: {target_value or null}.
  CONTEXT FROM OTHER AGENTS:
  - ANALYZE: {last_analyze_summary}
  - ARCHIVE STATS: {archive_stats_json}
  - DEAD ENDS: {dead_ends_summary}
  - BEHAVIORAL MEMORY: {learned_behaviors_summary}",
  subagent_type: "ml-optimizer:hyperagent-agent"
)
```

Save the hyperagent's ID to `agent_registry["hyperagent"]`. For subsequent iterations, resume via SendMessage with updated context from analyze and other agents.

The hyperagent invokes `Skill("ml-optimizer:hyperagent")` which guides its decisions within Phase 7: read context → choose operator → generate variant → staged eval → HP tune → archive. The analysis agent advises after each batch, and the orchestrator relays context between agents and tracks pipeline state.

### Per-Iteration Flow

Each iteration follows the same pattern regardless of what the hyperagent chose:

**1. Hyperagent chooses action** (from SendMessage above)

The hyperagent states its decision in natural language: which action, which parent (for code mutations), and why.

**2. Execute the action:**

| Action | Execution |
|---|---|
| `hp_tune` | Orchestrator dispatches tuning-agent (existing Step 1 flow), then experiment-agents |
| `llm_patch` | Hyperagent invokes `Skill("ml-optimizer:hyperagent-select")` then `Skill("ml-optimizer:hyperagent-generate")` with operator `llm_patch` |
| `shinka_evolve` | Hyperagent invokes `Skill("ml-optimizer:hyperagent-select")` then `Skill("ml-optimizer:hyperagent-generate")` with operator `shinka_evolve` (internally dispatches evolve skill) |
| `research_implement` | **Agent coordination:** (1) Hyperagent states it wants research-implement on a selected parent. (2) Orchestrator dispatches research-agent → produces research-findings. (3) Orchestrator dispatches implement-agent → implements FROM the selected parent branch (not baseline), creates `ml-opt/gen-<N>-<technique>`. (4) Hyperagent archives the result. The hyperagent does NOT do research/implementation itself — it coordinates through the orchestrator relay. |
| `meta_improve` | Hyperagent invokes `Skill("ml-optimizer:hyperagent-generate")` with `meta_improvement_mode: true`. Max 3 per session. |

**3. Staged eval** (for code mutations: llm_patch, shinka_evolve, research_implement):

Hyperagent invokes `Skill("ml-optimizer:hyperagent-eval")`:
- Stage 1: Quick eval (10% budget), adaptive threshold
- If PASS → full training (warm-start from staged checkpoint)
- If FAIL → archive as `status: "filtered"`, skip full training

For `hp_tune` actions: standard experiment execution (existing flow), no staged eval.

**4. Archive results:**

Hyperagent invokes `Skill("ml-optimizer:hyperagent-archive")` to update the archive with the iteration's results. Update `hyperagent_state.archive_generation` and `operator_stats`.

Log to `hyperagent_state.strategy_history`:
```json
{"iteration": N, "action": "llm_patch", "genid": "gen-007", "fitness_score": 0.87, "improved": true}
```

**5. Analyze:**

Orchestrator dispatches analysis-agent (existing Step 5 flow). The analyze skill receives archive stats and returns `continue/pivot/stop`.

**6. Decision:**

- `continue` → resume hyperagent for next iteration
- `pivot` with `code_evolution` → resume hyperagent (it will choose code mutation next)
- `pivot` with HP-focused type (`hp_expand`, `narrow_space`, etc.) → resume hyperagent with the pivot context (it will choose `hp_tune` next)
- `stop` → apply stuck protocol or exit (existing logic)

### Meta-Improvement (Self-Referential)

When the hyperagent chooses `meta_improve` (or analyze returns `pivot_type: "meta_improvement"`):

1. Hyperagent reads current skill files (hp-tune, analyze, research) + archive + operator stats
2. Generates patched skill files to `experiments/meta-patches/`
3. Writes `experiments/meta-patches/meta-changelog.json`
4. Orchestrator records patches in `hyperagent_state.active_meta_patches`
5. Subsequent agent dispatches include meta-patch context (see "Pre-Loop: Load Meta-Patches")
6. Hyperagent resumes with improved strategy

**Constraints:** Max 3 meta-improvements per session. Cannot modify orchestrator or its own skill.

### End-of-Session Promotion (Phase 9)

See `references/phase-9-report.md` Phase 9 Step 3 for the full meta-patch promotion flow.
