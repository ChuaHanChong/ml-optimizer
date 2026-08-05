# Phase 7: Experiment Loop (Workflow)

Phase 7 runs as a **dynamic workflow** (`Workflow({scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-7-experiment.js", args})`). The loop lives inside the workflow script: it creates rounds, dispatches existing agents internally via `agentType` (e.g. `agent(prompt, {agentType: "ml-optimizer:tuning-agent"})`), reads the files those agents write under `<exp_root>/`, applies the decision tree, and returns a structured summary. The loop is **autonomous with no mid-run user input** — user decisions (method-proposal scope/iterations, budget) are pre-authorized at Phase 4 and arrive in `args`.

> "Dispatch the X agent" = an `agent({agentType: "ml-optimizer:X-agent"})` call inside the workflow script. Each dispatch is a fresh, self-contained agent that reads the relevant `<exp_root>/` files (manifest, prior `batch-N-analysis.md`, agenda, dead-ends, learned-behaviors) plus the prompt args.

## Workflow Args & Return

**Args (in):**
```
{ exp_root, project_root, baseline, primary_metric, divergence_metric,
  divergence_lower_is_better, model_category, lower_is_better, target_value, scope_level,
  fixed_time_budget, fixed_epoch_budget, fixed_step_budget, hp_batches_per_round, method_proposal_scope,
  method_proposal_iterations, seeds_per_config, eval_tasks, secondary_metrics, experiments_per_gpu }
```
> `fixed_time_budget` (seconds) and `fixed_epoch_budget` (epochs) are the **training** budget, passed through from `user_choices`. The workflow forwards whichever is set into every experiment-agent prompt so each run is capped exactly like the baseline and stays comparable (CLAUDE.md "Training budget options"). Passed as two typed fields (not one derived `budget`) so the experiment agent knows whether to wrap `timeout` vs cap epochs. Distinct from the workflow runtime's token `budget` global, which the script uses to self-bound the loop's agent spend (`budget.remaining()`). *(A legacy scalar `budget` arg is still tolerated and treated as seconds.)* `fixed_step_budget` (integer environment timesteps) is the RL budget unit — when set it wins over the time/epoch budgets and maps to the framework's timestep flag (e.g. `--total_timesteps`).

> `model_category` comes from `user_choices` (Phase 0). The workflow uses `args.model_category || pre.model_category` (baseline.json fallback via the pre-loop) and threads it into the tuning, analysis, and experiment prompts so divergence thresholds and HP strategy match the model class.

**Return (out):**
```
{ best_exp_id, best_metric, rounds_completed, exit_reason,
  stacking_candidates: [{branch, improvement_pct}, ...] }
```

When the workflow returns non-empty `stacking_candidates`, the orchestrator launches the phase-8 stacking workflow.

### How `stacking_candidates` Are Collected

`stacking_candidates` accumulates in a `Map` (`stackingByBranch`) keyed by branch, via `recordStackingCandidate(branch, improvementPct)`:
- **Exclusion:** the baseline/original branch (`originalBranch`, from the pre-loop) is always excluded — an HP-only win on it is not a stackable *code* method. Non-numeric or non-positive `improvementPct` (≤ 0) is also excluded.
- **Dedup:** if a branch is recorded more than once, only the highest `improvement_pct` seen for it is kept.

It is harvested from TWO independent sources each batch:
1. The analysis-agent's own `stacking_candidates` array in its returned decision (if present) — the agent's own judgment of which branches beat baseline.
2. An independent per-result sweep: for every `completed` result with a `code_branch` and a numeric `primary_metric_value`, the workflow itself computes `improvement_pct` against the (flat) baseline metric and records it — this catches branches the analysis agent didn't explicitly flag.

After the loop exits, if any candidates remain, a post-loop pass dispatches an agent to read the dead-end catalog (`error_tracker.py dead-end list`) and drops any candidate branch whose dead-end entry is actually a verdict on that branch's OWN implemented method (e.g. the technique name matches/resembles the branch slug) — as opposed to an entry that merely happened to be tested on that branch (e.g. an unrelated HP/scheduler dead-end). This prevents stacking from re-testing a method already ruled out on better evidence, since the harvest above compares each branch's best-tuned result against the flat baseline and can look like an improvement from HP luck alone even when analysis's matched-HP comparison already found the branch doesn't help.

## Before Launch: Orchestrator Pre-Dispatch Validation

`phase-7-experiment.js` does not call `validate_phase_requirements()` itself — this check is the orchestrator's general cross-phase policy (`skills/orchestrate/SKILL.md`: "Before each phase transition, validate prerequisites via `pipeline_state.validate_phase_requirements()` — prevents cascading failures from missing or corrupted data"), run BEFORE the orchestrator even calls `Workflow({scriptPath: ".../phase-7-experiment.js", args})`.

**Required state (checked by the orchestrator, not the workflow script):**
- `<exp_root>/results/baseline.json` must exist with `metrics` and `config` keys
- `<exp_root>/optimization-goals.json` must exist

(The manifest-`proposals` check is a non-blocking warning that runs only for phases 6/8, not phase 7.)

If validation fails, the orchestrator stops and reports the missing prerequisites — the workflow is never launched.

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

## Method Proposals Before the Loop Starts

`phase-7-experiment.js`'s pre-loop dispatch does NOT generate method proposals — it only verifies baseline integrity, syncs behavioral memory, loads the implementation manifest, confirms the agenda/dead-end files exist, builds the initial search space, and detects GPU/model_category (see "Pre-Loop: Verify Baseline Integrity" through "Pre-Loop: Load Implementation Manifest" below). There is no research-agent or implement-agent call before the main loop.

Users who want reviewed, pre-loop method proposals (with a checkpoint before they're committed to the loop) run research as a separate pass first: choose `method_proposal_scope = null` for Phase 7 and run Phase 5 (research) with a post-research checkpoint (`phase-4-checkpoint.md`, the "Auto-confirm method proposals" note under "Pre-Authorize Phase 7 Autonomy" — not Option 5, which is "Skip to Experiments"). Setting `method_proposal_scope` for Phase 7 itself only pre-authorizes *mid-loop* method proposals (step 7 below and the cadence trigger, step 8) — it does not add a pre-loop round.

## Research Proposal `type` Field: Schema-Only, Not Used for Routing

`RESEARCH_PROPOSALS_SCHEMA` allows each proposal a `type` (`"hp_only"` | `"code_change"` | `null`), but the workflow does not branch on it: `runResearchImplement` / `runImplementOnly` map every returned proposal's `index` straight to the implement-agent regardless of `type` — there is no `hp_only` → `search_space`-only shortcut. An `hp_only`-typed proposal is still implemented as a code branch like any other. Treat `type` as informational only until this routing is implemented.

## Pre-Loop: Initialize Research Cadence

Initialize the research round counter:
- `batches_since_last_research = 0`
- Tracks HP tuning batches since the last research → implement cycle
- When it reaches `hp_batches_per_round`, step 8 auto-triggers a new research round

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

   **Round type:** the script hardcodes `roundType = "hp"` — every Phase 7 tuning batch creates an `hp` round, regardless of what triggered it. The `method_proposal` (research-implement) and `code_evolution` (ShinkaEvolve) paths inside Phase 7 never call `create-round` at all — they only write to `reports/` and `results/implementation-manifest.json`; any experiment run on the resulting branch still goes through a normal `hp` round. `evolved` and `research` round types are not created anywhere in Phase 7. Only Phase 8 creates `create-round stacked` (see phase-8-stacking.md).

   Save `round_dir` into the iteration's local context. It is passed to every subsequent dispatch this iteration.

1. **Get HP configs:**
   Dispatch the tuning agent via `agent({agentType: "ml-optimizer:tuning-agent"})` with prompt:
   ```
   "Propose the next batch of HP configurations. Parameters: project_root: {project_root}, exp_root: {exp_root}, num_gpus: {num_gpus}, num_configs: {num_configs}, search_space: {search_space}, iteration: {iteration}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, code_branches: {code_branches}, branch_scores: {branch_scores_json or {}}, correlations: {correlations_json or {}}, max_batch_size: {max_batch_size or omit}, warm_start_enabled: {warm_start_enabled or false}, round_dir: {round_dir}, model_category: {model_category}, seeds_per_config: {seeds_per_config or 1}."
   ```
   The agent reads cross-agent context from files (no message bus): the research agenda (`reports/research-agenda.*`, untried high-priority techniques), the dead-end catalog (`reports/dead-ends.*`, DO NOT re-propose), the most recent `reports/batch-*-analysis.md` (prior batch findings), and `learned-behaviors.json` (OOM `max_batch_size`, divergence patterns synced from the monitor). The workflow also passes explicit args it already has in scope:
     - `code_branches`: From implementation manifest, or `[]` for HP-only.
     - `branch_scores` *(optional)*: From the prior analysis (`compute_branch_scores`).
     - `correlations` *(optional)*: HP-metric correlations from the prior analysis (Spearman rank correlation).
     - `max_batch_size` *(optional)*: One step below the smallest OOM-causing batch size (from `learned-behaviors.json` / error tracker). Omitted from the prompt entirely if there's no OOM constraint.
     - `exp_root`, `model_category` (resolved as `args.model_category || pre.model_category`), `seeds_per_config` (defaults to 1 if unset): passed straight through from the workflow's own args/state.
   - Reads past results and proposes the next batch of configs
   - Number of configs (`num_configs`) = `sequential ? 1 : num_gpus * experiments_per_gpu` (file_backup projects run one experiment at a time; otherwise each of the `num_gpus` GPUs runs `experiments_per_gpu` concurrent experiments — pre-authorized at Phase 4, default 1)
   - **hp-tune's `recommendation` field:** hp-tune returns its own early `recommendation` (`"continue"` or `"stop"`) in the schema, but the workflow currently collects it and does nothing further with it — no error-tracker log, no threading into the analysis-agent prompt. Analysis still makes the final continue/pivot/stop decision independently, from the batch results.

   ### HP-Tune Failure Recovery

   The workflow has exactly ONE fallback, not a multi-tier cascade: if hp-tune returns zero configs (`configs.length === 0` — crashed, malformed schema, or genuinely nothing to propose), the workflow logs a warning and substitutes a single fallback experiment: `exp_id: "exp-fallback-<iteration>"`, `config: baseline.config` (or `{}` if the baseline has no config), `gpu_id: 0`, `code_branch: code_branches[0] || null`, `method_tier: "method_default_hp"`. This keeps the loop alive with one baseline-anchored run. There is no config-validation pass, no simplified-prompt retry, and no random-sampling fallback.

   ### hp-tune-Requested Research (`research_requested: true`)

   On any iteration, hp-tune may return `research_requested: true` when it judges no research-derived HP priors exist yet for this model class. Each occurrence triggers ONE gated research round — the same mid-loop research→implement path as step 7 — before that iteration's configs are used, subject to the same gates as step 7a's scope check:
   - Skipped (log only, proceed with baseline-anchored configs) if `method_proposal_scope` is unset or `scope_level === "training"`.
   - Skipped (log only) if the `method_proposal_iterations` budget is already exhausted.
   - Otherwise: run the research→implement cycle, merge any newly validated branches into `code_branches`, and reset `batches_since_last_research` to 0.

2. **Run experiments:**
   - For each proposed config, dispatch an experiment agent via `agent({agentType: "ml-optimizer:experiment-agent"})` (runs the `ml-optimizer:experiment` skill)
   - Pass `code_branch` and `code_proposal` from the manifest (or null for HP-only)
   - **Pass `round_dir` (from Step 0)** — the experiment skill writes results to `results/<round_dir>/<exp_id>.json`. Without this, the PreToolUse hook blocks the write.
   - If multiple GPUs available, dispatch experiments in parallel (see Parallel GPU Dispatch Pattern below) — the workflow's `parallel()` fan-out
   - GPU assignment is `gpu_id: cfg.gpu_id ?? idx % num_gpus` — the tuning-agent's own proposed `gpu_id` (a field in `HP_CONFIG_SCHEMA`) takes priority when present; `idx % num_gpus` is only the fallback when it's absent. Each experiment runs on a GPU, but when `experiments_per_gpu > 1` (pre-authorized at Phase 4, default 1), multiple experiments intentionally share a GPU; each then gets a CPU-core slice via `env_vars={"OMP_NUM_THREADS": max(1, floor(nproc / num_configs))}` (optionally pinned with `taskset`) so co-located runs don't thrash the CPU.

3. **Monitor experiments (folded into the experiment agent):**
   - Divergence detection is **not a separate concurrent dispatch** here. The experiment agent runs `${CLAUDE_PLUGIN_ROOT}/scripts/detect_divergence.py` against its own training log while training, using the divergence params passed in its dispatch. The workflow does NOT spawn a standalone `monitor-agent` per batch by default — divergence params flow to the **experiment agent** instead.
   - **If `divergence_metric` is not null**, pass the divergence params through to each experiment-agent dispatch (Step 2 / Parallel GPU Dispatch Pattern): `divergence_metric`, `divergence_lower_is_better`, `model_category`. The experiment agent feeds these to `detect_divergence.py` (e.g. `--higher-is-better` when `divergence_lower_is_better` is false, `--model-category {model_category}`) so reward-like RL metrics get correct polarity and thresholds.
   - **Standalone monitor (not currently wired in):** A separate `agent({agentType: "ml-optimizer:monitor-agent"})` dispatch (concurrent with experiments) describes a pattern for long runs where out-of-process polling is preferred, but `phase-7-experiment.js` has no flag, `user_choices` key, or code path that reaches it today — it is not invocable from within the compiled workflow. If wired in as an opt-in path, dispatch with prompt:
     ```
     "Monitor running experiments. Parameters: log_files: {log_files}, exp_ids: {exp_ids}, project_root: {project_root}, poll_interval: 30, metric_to_watch: {divergence_metric}, lower_is_better: {divergence_lower_is_better}, model_category: {model_category}."
     ```
   - The experiment agent relays divergence/OOM findings by logging to the error tracker. The analysis-agent surfaces `max_batch_size` in its returned decision; the workflow carries it forward in-memory (the `maxBatchSize` variable) into the next round's tuning-agent prompt — a per-batch JS-variable handoff, not a `learned-behaviors.json`/`sync-from-errors` file sync (that runs once, pre-loop only — see step 9 below).
   - **The experiment agent's own in-process check** (the active-by-default path, per experiment/SKILL.md Step 4) produces only two statuses: `diverged` (kill, record reason in results) or `plateaued` (never kill — recorded as a note only, training continues to its budget).
   - **The optional monitor** (not currently wired into any workflow) has a separate, wider status vocabulary — `healthy`, `diverged`, `completed`, `failed`, `no_output`, `unmonitored`, `overfitting_warning` — documented in monitor/SKILL.md; it does not apply to the active default path above.
   - **If `divergence_metric` is null** (tabular ML — scikit-learn, XGBoost, LightGBM): skip divergence detection entirely. Wait for experiments to complete naturally.

   ### Early Batch Abort on Mass Divergence

   Each experiment kills itself when its own in-process divergence check fires (or the optional monitor kills it). The analysis agent sees the results (including diverged experiments) and recommends appropriate action (narrow search space, etc.).

4. **Wait for completion:**
   - All experiments in the batch must complete (or be stopped) before analysis
   - **Experiment timeout:** Each experiment has a hard timeout computed as:
     - If `baseline.json` has `profiling.estimated_timeout_seconds`: use that value directly (recorded for tabular ML and RL)
     - Else if `baseline.json` has `profiling.training_duration_seconds`: `max_experiment_duration = training_duration_seconds * 3`
     - Else if `baseline.json` has `profiling.throughput_samples_per_sec` (iterative DL): `timeout_seconds = int(1.5 × (dataset_size × epochs) / throughput)`
     - Else (no profiling): fallback to 21600 seconds (6 hours)
     If an experiment exceeds the timeout:
     1. Kill the experiment process
     2. Set `status: "timeout"` in the experiment result JSON
     3. Log to error tracker: `category: "timeout", severity: "warning", source: "orchestrate", message: "Experiment <exp_id> timed out after <duration>s (limit: <max_duration>s)"`
     4. Continue with the remaining experiments in the batch

   ### Completeness Enforcement

   There is no separate pre-analysis validation loop here. Completeness (non-empty metrics, `iteration`, `method_tier`, `duration_seconds`, `eval_protocol` for completed experiments) is enforced automatically at write time by the PreToolUse hook (`validate_experiment_write.py`, via `schema_validator.py`'s `check_completeness()`) — an incomplete result is blocked before it's ever written, so there is nothing left to repair after the fact. The lightweight, non-fatal post-batch check that DOES run after analysis — confirming expected output files exist, not re-validating completeness — is step 5c below.

5. **Analyze results:**
   Dispatch the analysis agent via `agent({agentType: "ml-optimizer:analysis-agent"})` with prompt:
   ```
   "Analyze batch {N} results. Parameters: project_root: {project_root}, batch_number: {batch_number}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, target_value: {target_value or null}."
   ```
   The agent reads cross-agent context from files: the batch's `results/<round_dir>/exp-*.json` (HP configs, divergence count, completion counts), `learned-behaviors.json`, the dead-end catalog, and the research agenda. It writes `reports/batch-{N}-analysis.md` (correlations, branch scores, recommendation) — the file the next round's tuning/research dispatch reads. It returns the decision the workflow acts on.
     - Compares all experiments, ranks them, identifies patterns
     - Recommends one of: `continue`, `branch_test`, `hp_expand`, `narrow_space`, `regularization`, `method_proposal`, `code_evolution`, or `stop` (see the Decision step (6) below for the full tree — `"pivot"` is never an actual returned value)

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

5c. **Verify batch completeness and close the round:**
   After analysis returns and state is updated, verify the batch's outputs are present, then close the round with a summary. This is a lightweight, non-fatal check — no `check-round` call and no `non_terminal`/`invalid`/missing-placeholder repair logic.

   Verification (`verifyBatchOutputs`): for each completed `exp_id` in this batch, run:
   ```bash
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/output_contract.py check <exp_root> experiment-agent --round-dir <round_dir> --exp-id <exp_id>
   # exit code 2 means missing outputs; its JSON output carries a "missing" array
   ```
   Also confirm `reports/batch-<N>-analysis.md` exists. Collect every missing path across all exp_ids (plus the analysis file, if absent) into one flat list.
   - If anything is missing: log ONCE to the error tracker (`category: "agent_failure", severity: "warning", source: "orchestrate", message: "batch <N> missing required outputs"`) and continue — this is a non-fatal warning, not a blocking gate. There is no retry and no repair attempt.

   Close the round with a one-line summary:
   ```bash
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/round_manager.py <exp_root> close-round --summary "Iteration {N}: best {primary_metric}={value}, {completed}/{total} completed"
   ```
   Also regenerate the live dashboard: `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/dashboard.py <exp_root> --live`.

5d. **Adversarial pivot gate (before a costly pivot):**

   After the round closes (5c) and the target-reached check (see the "Loop exit conditions" bullet in step 6) finds no exit, but BEFORE the decision if/else-if chain in step 6 runs: if `decision` is `method_proposal` or `code_evolution` — each spends a full research/implement/evolve cycle — the workflow dispatches TWO independent analysis-agent "skeptics" in parallel to try to REFUTE the pivot from the run's own files, rather than trusting a single analysis pass:
   - **Lens 1:** an untried, in-scope, non-dead-end HP direction still exists that is cheaper than this pivot (checked against `reports/research-agenda.*` and the search-space coverage in the latest `reports/batch-*-analysis.md`) — if found, the pivot is premature.
   - **Lens 2:** the HP plateau is within noise rather than a real signal (re-examines the batch result spread / effect sizes in the latest `reports/batch-*-analysis.md`) — if so, keep tuning before pivoting.

   Each skeptic returns `{refuted, reason}` (default `refuted: false` absent concrete file-grounded evidence). If EITHER skeptic returns `refuted: true`, the workflow logs the reason(s) and downgrades `decision` to `"continue"` — step 6's if/else-if chain then treats the batch as a normal continue, never entering the `method_proposal`/`code_evolution` branch. `stop` decisions are NOT second-guessed here — they're already guarded by the Stuck Protocol + fixpoint judgment in step 6. This gate only runs for a costly pivot, not on every batch.

6. **Decision:**
   - If analyze says **continue**:
     - Loop back to step 0 (create a new round, then hp-tune → experiments)
   - For any other `decision` value (a "pivot"-type branch — `"pivot"` itself is never an actual returned value, see below):
     - Apply the branch's adjustments → loop back to step 0 (create a new round, then hp-tune → experiments)
     **Dispatch is a flat if/else-if chain keyed directly on the `decision` field** — there is no generic "pivot by type" routing layer. The pivot-relevant branches:
     - `decision === "branch_test"`: Pass analyze's suggestion to hp-tune. Generate configs for untested branches with baseline HPs. No research needed.
     - `decision === "hp_expand"`, `"narrow_space"`, or `"regularization"`: all three take the same branch — apply analyze's `search_space` to hp-tune (widened, narrowed, or regularization-augmented per analyze's own reasoning).
     - `decision === "method_proposal"`: Route to step 7 (mid-loop research + implement). `pivot_type === "qualitative_change"` is rerouted to `decision: "method_proposal"` upstream (unless `decision` is already `stop`/`code_evolution`/`method_proposal`) — it never appears as its own dispatch branch.
     - `decision === "code_evolution"`: Route to the ShinkaEvolve step (below).
     - **Any other/unknown value:** falls through to the default `else` branch, treated as `hp_expand` (applies analyze's `search_space` if present) — the safest default. Logged.
     - `"research"` is not a real `decision` or `pivot_type` value anywhere in the pipeline — do not route on it.
   - If analyze says **stop**:
     - **Do NOT exit the loop immediately.** Exit is the orchestrator's reasoned judgment, never a fixed counter.
     - Log the stop recommendation. Increment `consecutive_stop_count` (telemetry only — not an exit trigger).
     - Invoke the **Stuck Protocol** below to search for new approaches, then run the **Exit Judgment** (see "Loop exit conditions").

     **Stuck Protocol** (invoked whenever analysis recommends stop):

       Single dispatch, `runStuckResearch()`: the research-agent's prompt instructs it to itself read (as files under `<exp_root>`) error patterns (`error_tracker.py patterns`), success metrics, the dead-end catalog (`dead-end list` — DO NOT re-propose), and the research agenda (`agenda list`), then via `agent({agentType: "ml-optimizer:research-agent"})` with real params `source: both, scope_level: {method_proposal_scope or 'architecture'}, output_path: <exp_root>/reports/research-findings-method-proposals-iter{N}.md, exp_root, project_root, primary_metric, current best {primary_metric}, model_category`, focused on techniques NOT in the dead-end catalog. Returns `{newInScope, agendaHasUntried, selectedIndices, findingsPath}`.
       `consecutive_stop_count` always increments on a stop recommendation, before research runs — not conditioned on `newInScope`. After research completes (or is skipped), the workflow computes `continueLoop = new proposals implemented OR newInScope > 0 OR agendaHasUntried OR metric improved since last stop`. If any hold, `stuck_protocol_triggered` resets to `false` and `consecutive_stop_count` resets to `0`, and the loop continues. Only when all are false does `stuck_protocol_triggered` get set `true` and the loop exit at the fixpoint (see Exit Judgment below — the two must agree).
   - **If analyze output is malformed or missing `decision`:** defaults straight to `continue` (`phase-7-experiment.js`: `decision = analyzeRes?.decision || "continue"`) — no retry, no simplified-prompt redispatch, no dedicated error-tracker log for this case.
   - **Loop exit conditions:** The experiment loop is **autonomous by default** — it runs non-stop until one of the following. There is **no hardcoded stop-count threshold**; the third condition is the workflow's own evidence-based judgment.
     1. Target metric achieved (from Phase 0). **This check runs unconditionally after every batch** — right after the round closes (step 5c), before the adversarial pivot gate (5d) and the decision dispatch (this step) — regardless of what `decision` analyze returned for that batch, even `"continue"`. If reached, the loop breaks immediately with `exit_reason: "target_reached"`, pre-empting the rest of the decision tree for that batch.
     2. User manually stops (the run ends; the orchestrator can relaunch via `resumeFromRunId`). Before relaunching, the orchestrator should run `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/pipeline_state.py <exp_root> cleanup` — the real CLI cleanup command (`cleanup_stale()`), which marks any experiment still `"running"` past its timeout as interrupted/failed so the resumed loop doesn't wait on it. A user stop does not set `exit_reason` — the run simply ends.
     3. **The workflow judges the search has reached a *fixpoint*** — no fresh direction left AND no progress to build on. An objective state, not a count: literally nothing new to try within scope and the metric has stopped moving. The workflow returns with `exit_reason` set accordingly.
     4. **The workflow's own token-budget self-limit** (`budget.remaining() <= 0`) — distinct from the user-initiated stop in (2); checked both as the loop's `while` condition and again at the bottom of each iteration.

   **`exit_reason` takes 5 possible values:**
     - `"target_reached"` — condition 1 above; the loop breaks mid-iteration once the target is hit.
     - `"fixpoint"` — condition 3; set by the Exit Judgment below.
     - `"budget_exhausted"` — condition 4; also the **literal default/fallthrough value** the variable is initialized to before the loop starts, distinct from the training `fixed_time_budget`/`fixed_epoch_budget` (see "Workflow Args & Return" above) — this is the workflow runtime's own token-spend `budget.remaining()` self-limit.
     - `"baseline_integrity_halt"` — the pre-loop baseline-checksum check failed (`pre.halt === true`); the workflow returns before the loop ever starts, with `rounds_completed: 0` and `stacking_candidates: []`.
     - `"preloop_failed"` — the pre-loop agent dispatch itself returned falsy (agent failure); same early return as above.

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
          python3 ${CLAUDE_PLUGIN_ROOT}/scripts/pipeline_state.py <exp_root> log-decision '{"phase":7,"agent":"workflow","decision_type":"loop_exit_judgment","decision":"exit"|"continue","iteration":<N>,"reasoning":"<best_metric, improved_since_last_stop, agenda_untried (boolean), new_proposals_count>"}'
          ```
          The dispatch runs only `pipeline_state.py log-decision`; there is no separate dev_notes.md append or error-tracker log call for this specific exit-judgment record.

     A single plateau is **not** exhaustion — breakthroughs can follow plateaus, so the bar for exit is the fixpoint in (c), not a recommendation count. `consecutive_stop_count` is telemetry only; `stuck_protocol_triggered` is the state flag that, with an unchanged best metric and an empty agenda, defines the fixpoint.

7. **Mid-loop method proposal trigger** (when analyze recommends new methods):

   If `decision === "method_proposal"` (including cases where `pivot_type === "qualitative_change"` was rerouted into `decision: "method_proposal"` upstream, per step 6 — `pivot_type === "method_proposal"` itself is never checked anywhere in the workflow):

   a. **Scope check (no prompt — pre-authorized at Phase 4):**
      Read `method_proposal_scope` from `args` (set at Phase 4). The old mid-loop "Scope options 1/2/3/4" prompt is gone — the workflow takes no mid-run input.
      - If `method_proposal_scope` is `null` (user opted out): the workflow logs `"method_proposal requested but scope_level=<scope_level> disallows it — treating as continue"` and increments `batchesSinceLastResearch` only (no `search_space` change) — treated as **continue**, not `hp_expand`. Skip steps b–e.
      - If `scope_level === "training"` (even with `method_proposal_scope` set), per the scope-gated pivot rule — `"training"` scope disables research/code_evolution pivots outright: same behavior — logs the same message and increments `batchesSinceLastResearch` only, treated as **continue**, not `hp_expand`. Skip steps b–e.
      - If `method_proposal_iterations` is exhausted (mid-loop research rounds already run ≥ the pre-authorized budget): skip steps b–e (no more method proposals this run); fall back to `hp_expand`.
      - Otherwise use `scope_level = method_proposal_scope` and proceed.

   b. **Generate proposals:** Dispatch the research agent via `agent({agentType: "ml-optimizer:research-agent"})` (`runResearchImplement()`) with the real params: `source: both, scope_level: {scope_level}, output_path: <exp_root>/reports/research-findings-method-proposals-iter{N}.md, exp_root: {exp_root}, project_root: {project_root}, primary_metric: {primary_metric}, current best {primary_metric}: {bestMetric}, model_category: {model_category}`. The prompt instructs the agent to itself read `reports/dead-ends.*` and the research agenda first (DO NOT re-propose dead ends or already-tried techniques) and to initialize/update the research agenda. Returns `{findings_path, proposals, new_in_scope_count, agenda_has_untried}`.

   c. **Filter proposals (no prompt):**
      The workflow auto-selects in-scope, non-dead-end proposals (acceptance delegated at Phase 4). If `runResearchImplement()` returns zero branches, no `search_space` change is applied and no branches are merged — the same no-op as the scope-disabled case in step 7a — `batches_since_last_research` still resets to 0.

   d. **Implement proposals:** Dispatch the implement agent via `agent({agentType: "ml-optimizer:implement-agent"})` with prompt:
      ```
      "Implement research proposals. Parameters: findings_path: {findings_path}, selected_indices: {selected_indices}, project_root: {project_root}."
      ```
      Creates new `ml-opt/<slug>` branches and writes `results/implementation-manifest.json` (the file the next hp-tune dispatch reads for `code_branches`).

   e. **Merge into experiment loop:** Add the new validated branches to `code_branches`. There is no per-branch iteration counter — `iteration` is one global loop variable shared by every branch/config in a round (`phase-7-experiment.js` always interpolates the global `iteration`, never a per-branch value). A branch added mid-run enters at whatever the current global iteration is.

   **If `decision === "code_evolution"` (evolutionary code improvement — `code_evolution` is a `decision` value, never checked as `pivot_type`):**

   Instead of research → implement, the flow is: **tuning agent → implement agent → tuning agent → experiment agent**. The analysis agent's own conditions (|rho| < 0.3 + method improved) prevent unnecessary evolution — no artificial cooldown needed.

   a. **Tune evolve HPs:** Dispatch the tuning agent via `agent({agentType: "ml-optimizer:tuning-agent"})` with prompt:
      "Propose ShinkaEvolve evolution HPs for code_evolution. Read `learned-behaviors.json` category `evolve_hp` for prior evolution outcomes. Consider: how many generations produced the best mutation last time, whether more population diversity is needed, and the improvement trajectory. Return {num_generations, population_size, reasoning}." (flat — matches `EVOLVE_HP_SCHEMA`, no `evolve_recommendation` wrapper on this return; the workflow wraps it as `evolve_recommendation: {...}` itself only when forwarding into the step-b evolve dispatch)

   b. **Execute evolve:** Dispatch the implement-agent via `agent({agentType: "ml-optimizer:implement-agent"})` with the evolve skill.
      Prompt: `Skill("ml-optimizer:evolve")` with parameters:
      - `project_root`, `parent_branch` (best method branch), `parent_metrics` (best result metrics)
      - `primary_metric`, `lower_is_better`, `scope_level`
      - `exp_root`
      - `evolve_recommendation`: from tuning agent (step a) — `{num_generations, population_size}` (the workflow drops `reasoning` when building this object; the tuning agent's reasoning informs the two numbers but is not itself forwarded to the implement-agent)
      - `feedback_context`: the dispatch instructs the implement-agent to read the dead-end catalog, recurring error patterns, the most recent batch analysis, and `learned-behaviors.json` itself before evolving, and pass them as `{dead_ends, error_patterns, batch_analysis, learned_behaviors}` — matching the evolve skill's own Input Parameters contract (`skills/evolve/SKILL.md`). The agent self-serves these file reads rather than `phase-7-experiment.js` pre-fetching them, consistent with how other dispatches in this codebase read agenda/dead-end context.

      The evolve skill runs: `shinka-convert` → `shinka-run` (with file handoff loop) → `shinka-inspect` → commit best as `ml-opt/evolved-<slug>`.

   c. **Verify result:** The evolve skill returns `{status, branch, best_combined_score, notes}`.
      - If `status == "validated"`: add `branch` to `code_branches`
      - If `status == "validation_failed"`: `runCodeEvolution()` only has explicit handling for `"validated"` and `"shinkaevolve_unavailable"` — any other status (including `"validation_failed"`) silently falls through to `return null`, continuing the loop without a new branch and without an error-tracker log
      - If `status == "shinkaevolve_unavailable"`: ShinkaEvolve couldn't be installed or crashed. Fall back to the research → implement path (the `method_proposal` steps a–e above). Log the failure.

   d. **Tune training HPs + run experiment:** Dispatch tuning agent to propose training HPs for the evolved branch, then dispatch experiment-agent. The evolved branch enters the normal HP-tune → experiment → analyze loop.

   e. **Update state:** The workflow increments its in-memory `methodProposalRoundsUsed` counter (drawn from the `method_proposal_iterations` budget) — this is workflow-run-scoped only, not persisted to `user_choices` or `pipeline-state.json`. A fresh invocation (including a `resumeFromRunId` relaunch) resets it to 0, since nothing on disk carries it forward. There is no "save pipeline state" call.

   f. **Continue loop:** Loop back to step 0 (create a new round) with the expanded `code_branches` list. Unlike the `method_proposal` path (step 7e above) and the cadence trigger (step 8d), the code_evolution path INCREMENTS `batches_since_last_research` rather than resetting it to 0 — it does not count as a "research round" for cadence purposes.

8. **Research round check** (cadence-based research trigger):

   Auto-triggers research → implement on a regular cadence, independent of analyze's pivot recommendation. Applies when `method_proposal_scope` is set.

   **Conditions (ALL must be true):**
   - `methodProposalsEnabled` — `method_proposal_scope` is set AND `scope_level !== "training"`
   - `methodProposalBudgetLeft()` — the shared `method_proposal_iterations` budget (pooled across step 7, hp-tune-requested research, and this cadence trigger) is not exhausted. The Stuck Protocol's research (on `stop`) is exempt from this pool — it only checks `methodProposalsEnabled` and never calls `methodProposalBudgetLeft()` or increments the round count, so it can run on every `stop` regardless of budget.
   - `decision !== "method_proposal"` — step 7 did not already trigger this iteration
   - `decision !== "code_evolution"` — a code_evolution pivot this iteration also skips cadence research (avoid double-firing)
   - `decision !== "stop"`
   - `batches_since_last_research >= hp_batches_per_round`

   **`hp_batches_per_round` (the cadence) is fixed for the entire run** — the workflow never reassigns it, and no error-tracker log is emitted for the cadence trigger itself (unlike the `stop`/stuck-protocol path).

   **If conditions met:**

   a. **Generate proposals:** Dispatch the research agent via the same `runResearchImplement()` function as step 7b above — identical real params (`source: both, scope_level: {method_proposal_scope}, output_path: <exp_root>/reports/research-findings-method-proposals-iter{N}.md, exp_root, project_root, primary_metric, current best {primary_metric}, model_category`). Same self-serve dead-end/agenda read, same return shape.

   b. **Implement proposals:** If research returned ≥1 proposal with an integer `index`, ALL of them are implemented automatically — dispatch the implement agent via `agent({agentType: "ml-optimizer:implement-agent"})` with prompt:
      ```
      "Implement research proposals. Parameters: findings_path: {findings_path}, selected_indices: {all_indices}, project_root: {project_root}."
      ```
      Creates new `ml-opt/<slug>` branches and updates `results/implementation-manifest.json`. If research returned zero proposals, the implement dispatch is skipped entirely and no new branches are added this round — there is no exponential-backoff / cadence-doubling behavior on this path.

   c. **Merge into experiment loop:** Same as step 7e — add any newly validated branches to `code_branches`.

   d. **Update state:** Increments the in-memory `methodProposalRoundsUsed` counter (workflow-run-scoped only — not persisted to `user_choices`/`pipeline-state.json`; see the code_evolution note in step 7e) and unconditionally resets `batches_since_last_research = 0` — whether or not the research round actually returned or implemented any new proposals. There is no "save pipeline state" call.

   **If conditions NOT met:** no action here — `batches_since_last_research` was already incremented (or reset) by the Step 5/6 decision handling above; this check simply does nothing further.

9. **Loop back:**

    `sync-from-errors` runs exactly once in the whole file — inside the pre-loop dispatch, before the loop starts — not per iteration; there is no end-of-iteration behavioral-memory sync.

    After steps 6/7/8, return to step 0 (create a new round). The loop continues until the Decision step (6) forces an exit.

## Parallel GPU Dispatch Pattern

When dispatching experiments across multiple GPUs, the workflow uses its `parallel()` fan-out, one `agent({agentType: "ml-optimizer:experiment-agent"})` per experiment.

**If manifest strategy is `"file_backup"` (non-git project):** dispatch ONE experiment at a time (sequential). Wait for each to complete before starting the next. File-backup proposals share the same working directory and cannot run in parallel.

**Otherwise (git_branch strategy or HP-only):** dispatch all experiments in parallel — for each config in `proposed_configs`, call:
```
agent(
  "Run a single training experiment, then evaluate and write the result JSON. exp_id: {exp_id}. config: {config_json}. gpu_id: {gpu_id or idx % num_gpus}. project_root: {project_root}. exp_root: {exp_root}. round_dir: {round_dir}. code_branch: {code_branch or null}. code_proposal: {code_proposal or null}. proposal_source: {proposal_source or null}. method_tier: {method_tier or 'method_default_hp'}. iteration: {iteration}. checkpoint_source: {checkpoint_source_json or null}. primary_metric: {primary_metric}. model_category: {model_category or null}. eval_tasks: {eval_tasks_json}. [plus the divergence clause, training-budget clause, and — when experiments_per_gpu > 1 — the CPU-core-slicing clause, all appended as prose, not separate params]",
  { agentType: "ml-optimizer:experiment-agent", isolation: "worktree" }
)
```
Each run executes inside its own **git worktree**, checked out with `git checkout --detach` on the given `code_branch`, so parallel runs never collide on files or disturb the main tree (`isolation: "worktree"` in the dispatch opts). `train_command`, `eval_command`, `prepared_train_path`, and `prepared_val_path` are NOT passed as explicit params in the real prompt — the experiment-agent resolves training/eval commands and data paths itself from its own skill context.

Then wait for all experiment agents to complete before dispatching analyze. Reasoning depth is set by each agent's `effort` frontmatter (`xhigh` analytical / `medium` procedural) — the workflow adds no reasoning keyword; `agent({agentType})` inherits it.
