# Phase 8: Method Stacking (Workflow)

**Phase gate:** Run `pipeline_state.py <exp_root> gate 7 8` before entering. On completion: `pipeline_state.py <exp_root> log-gate 8 completed "<summary>"`.

Phase 8 runs as a **dynamic workflow** (`Workflow({scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-8-stacking.js", args})`). The orchestrator launches it when the phase-7 workflow returns non-empty `stacking_candidates`. The stacking loop lives inside the workflow script: it holds the "current best stack" in a script variable and dispatches existing agents internally via `agentType` (`agent(prompt, {agentType: "ml-optimizer:<name>-agent"})`).

> "Dispatch the X agent" = an `agent({agentType: "ml-optimizer:X-agent"})` call inside the workflow script, NOT an orchestrator `Agent()`/`SendMessage` call. No message bus, no `SendMessage`, no agent registry — cross-agent context flows via `args` + the files agents write under `<exp_root>/`.

## Workflow Args & Return

**Args (in):** `{ exp_root, project_root, primary_metric, lower_is_better, baseline_metric, scope_level, divergence_metric, divergence_lower_is_better, model_category, fixed_time_budget, fixed_epoch_budget, fixed_step_budget, stacking_candidates }`
where `stacking_candidates: [{branch, improvement_pct}, ...]` comes from the phase-7 return. `baseline_metric` (the baseline's primary-metric value) lets the workflow derive the seed method's absolute metric so the first accumulation is compared, not force-kept. `scope_level` gates the code_evolution/ShinkaEvolve repair (only runs at `"full"`, mirroring Phase 7). `fixed_time_budget`/`fixed_epoch_budget` cap each stacked run the same way the baseline was, so metrics stay comparable. `divergence_metric`/`divergence_lower_is_better`/`model_category` build the same divergenceClause Phase 7 uses, appended to every stacked experiment prompt — stacked runs get identical divergence handling to Phase 7 runs.

**Return (out):** `{ best_stack_branch, best_stack_metric, steps: [{method, branch, kept}, ...] }`

Method stacking combines different implementations (papers, LLM patches, or ShinkaEvolve) into one model. It's triggered by the analysis agent during Phase 7 (`pivot_type: "method_stacking"`) when it judges multiple improved methods could yield compound gains — not by a fixed method count. The phase-7 workflow surfaces those methods as `stacking_candidates`.

**Pre-check:** If `strategy: "file_backup"` (non-git project), the workflow skips stacking and returns immediately. Log to dev_notes.

After stacking completes, the orchestrator may relaunch the phase-7 workflow to continue optimizing on the stacked code.
- **Autonomous mode:** The workflow auto-proceeds. Log to dev_notes: "Auto-entering stacking phase with {N} improved methods."

## Workflow-Driven Stacking

The workflow ranks methods by improvement magnitude (descending — largest first) and drives the stacking loop. No archive lineage data needed — methods are stacked in order of effectiveness.

## Stacking Loop

1. **Rank methods by improvement magnitude** (descending) — the largest improvement over baseline gets stacked first.

2. **Initialize stack:** The best method's branch becomes `ml-opt/stack-1`. No experiment needed — its existing best result serves as the stack baseline.

3. **For each method** (in improvement-ranked order):

   a. **Create stack branch:**
   ```bash
   git checkout -b ml-opt/stack-<order> <stack_base_branch>
   # stack_base_branch tracks the current best stack (may be an evolved branch)
   # For order=2, this is ml-opt/stack-1 (or ml-opt/evolved-stack-1 if evolved)
   ```

   b. **Merge the next method:**
   ```bash
   git merge ml-opt/<method-slug> --no-ff --no-edit
   ```

   c. **If clean merge** → proceed to validation.

   d. **If merge conflicts** → dispatch the implement agent via `agent({agentType: "ml-optimizer:implement-agent"})` with prompt:
      ```
      "Resolve merge conflicts for stack step {order}. Merge {method_B} into {current_stack_branch}. Both methods must be preserved: [method-A description] and [method-B description]. The goal is to combine their functionality. Conflicting files: {conflicting_files} (from `git diff --name-only --diff-filter=U`). CONTEXT: methods ranked by improvement: {ranking}; recommended stacking order: {order}."
      ```
      - If implement-agent succeeds → `git add .` and `git commit -m "Resolve merge conflicts for stack-<order>"`
      - If implement-agent fails → skip this method:
        - `git merge --abort`
        - Log to error tracker: `category: "implementation_error", severity: "warning", source: "orchestrate", message: "Stacking conflict unresolved for <method-slug>"`
        - Continue to next method

   e. **Validate** (syntax, import, forward pass — same as implement skill validation).
      - If validation fails → skip: delete branch, log reason, continue.

   e2. **Create a stacking round** (MANDATORY before dispatching experiment-agent):
      ```bash
      round_info=$(python3 ${CLAUDE_PLUGIN_ROOT}/scripts/round_manager.py <exp_root> create-round stacked --branch ml-opt/stack-{order})
      # Capture "dir" field as round_dir (e.g., "round-7-stacked")
      ```
      Without this, the PreToolUse hook blocks the experiment's result write.

   f. **Run experiment** via `agent({agentType: "ml-optimizer:experiment-agent"})` with prompt:
      ```
      "Run stacking experiment. Parameters: exp_id: {exp_id}. Config: {config_json}. GPU: {gpu_id}. Project root: {project_root}. Train command: {train_command}. Eval command: {eval_command or null}. Code branch: ml-opt/stack-{order}. Method tier: stacked_default_hp. Stacking order: {order}. Stack base exp: {stack_base_exp}. Code branches: {code_branches_json}. round_dir: {round_dir}."
      ```

   g. **Evaluate result:**
      - Compare to the previous stack step's metric value.
      - **If improved:** Keep this stack step.
        - Update `stack_base_branch = ml-opt/stack-<order>`

        - **Analyze stacked result:** Dispatch the analysis agent via `agent({agentType: "ml-optimizer:analysis-agent"})` to assess whether the stacked code needs evolution or HP-tuning, with prompt:
          ```
          "Analyze stacked experiment result. Parameters: project_root: {project_root}, batch_number: stack-{order}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, exp_root: {exp_root}. CONTEXT: stacked experiment on ml-opt/stack-{order}: {stacked_metrics}; methods combined: {stacked_methods}, stacking order: {order}; best single method gain: {best_individual_gain}%. Compare the stacked gain to the best individual method gain. If the stack underperforms the best individual (methods interfering), recommend code_evolution. Otherwise recommend continue."
          ```
          The stacked metrics and best-individual gain are read from `results/round-N-stacked/exp-*.json` and the round results.

        - **If analysis recommends `pivot_type: "code_evolution"`:**

          **Step 1: Tune evolve HPs.** Dispatch the tuning agent via `agent({agentType: "ml-optimizer:tuning-agent"})` with prompt:
          "Propose ShinkaEvolve evolution HPs for stacking code_evolution. Stacked methods: {stacked_methods}. Read `learned-behaviors.json` category `evolve_hp` for prior outcomes. Return `evolve_recommendation: {num_generations, population_size, reasoning}`."

          **Step 2: Execute evolve.** Dispatch the implement agent via `agent({agentType: "ml-optimizer:implement-agent"})` with the evolve skill.
          Prompt: `Skill("ml-optimizer:evolve")` with parameters:
            - `project_root`, `parent_branch: ml-opt/stack-{order}`, `parent_metrics: {stacked metrics}`
            - `primary_metric`, `lower_is_better`, `scope_level`, `exp_root`
            - `evolve_recommendation`: from tuning agent (Step 1)
            - `feedback_context`: {batch_analysis, error_patterns, dead_ends, learned_behaviors}

          **Handle evolve result:**
          - If `status == "validated"`:
            - Run experiment on the evolved branch (`ml-opt/evolved-*`)
            - If evolved result beats the pre-evolution stack → update `stack_base_branch` to the evolved branch, add `"stack-<order>"` to `evolved_methods` in stacking state
            - If evolved result is worse → discard evolved branch, keep pre-evolution `ml-opt/stack-<order>` as stack base
          - If `status == "shinkaevolve_unavailable"` or `"validation_failed"`:
            - Log to error tracker, continue without evolution (proceed to HP-tune)

        - **HP-tune on the (potentially evolved) stack branch:** Dispatch the tuning agent via `agent({agentType: "ml-optimizer:tuning-agent"})` to propose training HPs, then run experiment.
          Prompt: "Propose HP configs for stacked method. Parameters: project_root: {project_root}, num_gpus: {num_gpus}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, code_branches: [{stack_base_branch}], iteration: 1, search_space: {narrowed_search_space}."

          Run experiments with proposed configs (via `agent({agentType: "ml-optimizer:experiment-agent"})`). If HP-tune improves, record as `method_tier: "stacked_tuned_hp"`.

        - **Re-analyze:** After evolve + HP-tune, dispatch the analysis agent again on the updated result (`agent({agentType: "ml-optimizer:analysis-agent"})`):
          - If analysis recommends `code_evolution` again → loop back to the tuning + evolve step above (with new HPs from tuning agent)
          - If analysis recommends `continue` → improvement achieved, proceed to next stack step
          - If analysis recommends `stop` → skip this method, the combination can't be fixed

      - **If analysis recommends `stop`:** Skip this method.
        - Delete `ml-opt/stack-<order>` branch
        - Log: "Method <slug> skipped in stacking (analysis determined combination unproductive)"
        - Continue to next method (next stack branch re-branches from the last successful stack)

   h. **Close the stacking round:** After all experiments and any evolve/hp-tune sub-rounds for this stack step complete:
      ```bash
      python3 ${CLAUDE_PLUGIN_ROOT}/scripts/round_manager.py <exp_root> close-round --summary "Stack step {order}: {result_summary}"
      ```
      Where `{result_summary}` is one of: `"kept ({gain}% over baseline)"`, `"skipped (interference)"`, or `"skipped (regression)"`.

4. **Save stacking state** to `pipeline-state.json` via `save_state(user_choices={"stacking": {...}})` after each stack step (for resumption):
   ```json
   {
     "user_choices": {
       "stacking": {
         "ranked_methods": ["method-b", "method-a", "method-c"],
         "current_stack_order": 3,
         "stack_base_branch": "ml-opt/stack-2",
         "stack_base_exp": "exp-stack-002",
         "skipped_methods": ["method-c"],
         "stacked_methods": ["method-b", "method-a"],
         "evolved_methods": ["stack-2"]
       }
     }
   }
   ```

5. **Final result:** The last successful `ml-opt/stack-<N>` branch is the compound best. The workflow returns `{ best_stack_branch: "ml-opt/stack-<N>", best_stack_metric, steps: [{method, branch, kept}, ...] }` to the orchestrator.
   Log to dev_notes: "Stacking complete. Final stack: [methods]. Compound gain: X% over baseline. Branch: ml-opt/stack-<N>"

## Stacking Phase Resumption

Within a session, the phase-8 workflow is resumable via `resumeFromRunId`. Across sessions / on pipeline restart, if `pipeline-state.json` contains a `stacking` key in `user_choices`, the orchestrator relaunches the phase-8 workflow, which reads the file-persisted stacking state and continues:
1. Read stacking state
2. **Validate before resuming:**
   a. `current_stack_order < len(ranked_methods)` — if not, stacking is already complete; skip to Phase 9
   b. Verify `ml-opt/stack-<current_stack_order>` branch exists (`git branch --list`). If missing, log error to error tracker and skip to Phase 9 with partial results.
   c. Verify `stack_base_exp` result file is readable. If missing, fall back to the last known good stack result from `stacked_methods`.
3. Resume from `current_stack_order + 1`
4. Continue with remaining methods in `ranked_methods` not in `stacked_methods` or `skipped_methods`
