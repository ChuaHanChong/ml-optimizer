# Phase 8: Method Stacking (Within the Autonomous Loop)

Method stacking combines different implementations (from papers, LLM patches, or ShinkaEvolve) into one model. It's triggered by the analysis agent during Phase 7 when it judges that multiple improved methods could yield compound gains — not by a fixed method count.

**Pre-check:** If `strategy: "file_backup"` (non-git project), skip stacking. Log to dev_notes.

**Trigger:** The analysis agent advises `pivot_type: "method_stacking"` when multiple methods from different papers or significant code changes have improved independently. The hyperagent decides whether to proceed, which methods to stack, and in what order — no hardcoded threshold.

After stacking completes, the hyperagent returns to Phase 7 to continue optimizing on the stacked code. The archive tracks stacked variants with lineage.
- **Autonomous mode:** Auto-proceed. Log to dev_notes: "Auto-entering stacking phase with {N} improved methods."

## Hyperagent Driven Stacking

The hyperagent helps Phase 8 by deciding which methods to stack, in what order, when to evolve for interference resolution, and when to stop. It also enables self-improvement — skill patches from Phase 7's meta-improve actions are active during stacking. The orchestrator resumes the hyperagent per stack step.

```
Dispatch hyperagent:
  SendMessage(
    to: agent_registry["hyperagent"],
    message: "Ultrathink. Phase 8: Method stacking. You have {N} methods that improved
    over baseline. Decide the stacking order based on archive lineage (methods from
    different lineages are more likely to complement; same-lineage methods may conflict).

    Methods ranked by improvement (descending — largest improvement first): {ranked_methods_json}
    Archive lineage data (for conflict detection): {lineage_data_json}

    Stack in rank order (best method first). For each step: merge, experiment, analyze.
    If methods interfere (stacked gain < best individual): dispatch ShinkaEvolve to resolve.
    If a method degrades performance: skip it.
    Use lineage data to flag potential conflicts (same-lineage methods may overlap).
    Stop when: no more methods, or stacking shows diminishing returns (you judge from evidence)."
  )
```

## Stacking Loop

1. **Rank methods by improvement magnitude** (descending) — the method with the largest improvement over baseline gets stacked first. The hyperagent uses archive lineage as a secondary signal (flag potential conflicts between same-lineage methods) but the primary ordering is always by effectiveness.

2. **Initialize stack:** The best method's branch becomes `ml-opt/stack-1`. No experiment needed — its existing best result serves as the stack baseline.

3. **For each method** (in hyperagent's chosen order):

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

   d. **If merge conflicts** → dispatch implement-agent (resume-or-dispatch pattern):

      **IF `agent_registry["implement"]` is not null** (agent exists from a previous phase):
      ```
      SendMessage(
        to: agent_registry["implement"],
        message: "Resolve merge conflicts for stack step {order}.
          CONTEXT FROM OTHER AGENTS:
          - EXPERIMENTS: methods ranked by improvement: {ranking}
          - ANALYZE: recommended stacking order: {order}
          Merge {method_B} into {current_stack_branch}. Preserve both methods.
          Conflicting files: {conflicting_files}. Both methods must be preserved: [method-A description] and [method-B description]."
      )
      ```
      → If `SendMessage` fails (agent no longer reachable): fall back to the `Agent()` dispatch below, update `agent_registry["implement"]` with the new agentId.

      **ELSE** (first dispatch — no existing agent):
      - **Prompt:** "Resolve merge conflicts in the following files. Both methods must be preserved: [method-A description] and [method-B description]. The goal is to combine their functionality."
      - **Files:** List of conflicting files from `git diff --name-only --diff-filter=U`
      → Save returned `agentId` to `agent_registry["implement"]`
      → Persist registry: `save_state(..., agent_registry=agent_registry)`

      - If implement-agent succeeds → `git add .` and `git commit -m "Resolve merge conflicts for stack-<order>"`
      - If implement-agent fails → skip this method:
        - `git merge --abort`
        - Log to error tracker: `category: "implementation_error", severity: "warning", source: "orchestrate", message: "Stacking conflict unresolved for <method-slug>"`
        - Continue to next method

   e. **Validate** (syntax, import, forward pass — same as implement skill validation).
      - If validation fails → skip: delete branch, log reason, continue.

   f. **Run experiment** by dispatching the experiment agent:
      ```
      Agent(
        description: "Run stacking experiment stack-{order}",
        prompt: "Run stacking experiment. Parameters: exp_id: {exp_id}. Config: {config_json}. GPU: {gpu_id}. Project root: {project_root}. Train command: {train_command}. Eval command: {eval_command or null}. Code branch: ml-opt/stack-{order}. Method tier: stacked_default_hp. Stacking order: {order}. Stack base exp: {stack_base_exp}. Code branches: {code_branches_json}.",
        subagent_type: "ml-optimizer:experiment-agent"
      )
      ```

   g. **Evaluate result:**
      - Compare to previous stack step's metric value.
      - **If improved:** Keep this stack step.
        - Update `stack_base_branch = ml-opt/stack-<order>`

        - **Analyze stacked result:** Dispatch the analysis agent to assess whether the stacked code needs evolution or HP-tuning (resume-or-dispatch pattern, same as Phase 7):

          **IF `agent_registry["analysis"]` is not null:**
          ```
          SendMessage(
            to: agent_registry["analysis"],
            message: "Analyze stacked experiment result.
              CONTEXT FROM OTHER AGENTS:
              - EXPERIMENTS: stacked experiment on ml-opt/stack-{order}: {stacked_metrics}
              - STACKING: methods combined: {stacked_methods}, stacking order: {order}
              - BEST INDIVIDUAL: best single method gain: {best_individual_gain}%
              Parameters: project_root: {project_root}, batch_number: stack-{order},
              primary_metric: {primary_metric}, lower_is_better: {lower_is_better},
              exp_root: {exp_root}.
              Compare the stacked gain to the best individual method gain. If the stack underperforms
              the best individual (methods interfering), recommend code_refinement. Otherwise recommend continue."
          )
          ```
          → If `SendMessage` fails: fall back to `Agent()` dispatch below, update `agent_registry["analysis"]`.

          **ELSE:**
          ```
          Agent(
            description: "Analyze stacked result stack-{order}",
            prompt: <same as above>,
            subagent_type: "ml-optimizer:analysis-agent"
          )
          ```
          → Save returned `agentId` to `agent_registry["analysis"]`
          → Persist registry: `save_state(..., agent_registry=agent_registry)`

        - **If analysis recommends `pivot_type: "code_refinement"`:**

          **Step 1: Tune evolve HPs.** Dispatch tuning agent (resume-or-dispatch):
          ```
          SendMessage(to: agent_registry["tuning"]) OR Agent(subagent_type="ml-optimizer:tuning-agent")
          ```
          Prompt: "Propose ShinkaEvolve evolution HPs for stacking code_refinement. Stacked methods: {stacked_methods}. Read `learned-behaviors.json` category `evolve_hp` for prior outcomes. Return `evolve_recommendation: {num_generations, population_size, reasoning}`."

          **Step 2: Execute evolve.** Dispatch implement-agent with evolve skill (resume-or-dispatch):
          ```
          SendMessage(to: agent_registry["implement"]) OR Agent(subagent_type="ml-optimizer:implement-agent")
          ```
          Prompt: `Skill("ml-optimizer:evolve")` with parameters:
            - `project_root`, `parent_branch: ml-opt/stack-{order}`, `parent_metrics: {stacked metrics}`
            - `primary_metric`, `lower_is_better`, `scope_level`, `exp_root`
            - `evolve_recommendation`: from tuning agent (Step 1)
            - `feedback_context`: {batch_analysis, error_patterns, dead_ends, learned_behaviors}

          **Handle evolve result:**
          - If `status == "validated"`:
            - Run experiment on the evolved branch (`ml-opt/evolved-*`)
            - If evolved result is better than pre-evolution stack → update `stack_base_branch` to evolved branch, add `"stack-<order>"` to `evolved_methods` in stacking state
            - If evolved result is worse → discard evolved branch, keep pre-evolution `ml-opt/stack-<order>` as stack base
          - If `status == "shinkaevolve_unavailable"` or `"validation_failed"`:
            - Log to error tracker, continue without evolution (proceed to HP-tune)

        - **HP-tune on (potentially evolved) stack branch:** Dispatch the tuning agent to propose training HPs, then run experiment (resume-or-dispatch pattern):

          ```
          SendMessage(to: agent_registry["tuning"]) OR Agent(subagent_type="ml-optimizer:tuning-agent")
          ```
          Prompt: "Propose HP configs for stacked method. Parameters: project_root: {project_root}, num_gpus: {num_gpus}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, code_branches: [{stack_base_branch}], iteration: 1, search_space: {narrowed_search_space}."

          Run experiments with proposed configs. If HP-tune improves, record as `method_tier: "stacked_tuned_hp"`.

        - **Re-analyze:** After evolve + HP-tune, dispatch the analysis agent again on the updated result:
          - If analysis recommends `code_refinement` again → loop back to the tuning + evolve step above (with new HPs from tuning agent)
          - If analysis recommends `continue` → improvement achieved, proceed to next stack step
          - If analysis recommends `stop` → skip this method, the combination can't be fixed

      - **If analysis recommends `stop`:** Skip this method.
        - Delete `ml-opt/stack-<order>` branch
        - Log: "Method <slug> skipped in stacking (analysis determined combination unproductive)"
        - Continue to next method (next stack branch re-branches from last successful stack)

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

5. **Final result:** The last successful `ml-opt/stack-<N>` branch is the compound best.
   Log to dev_notes: "Stacking complete. Final stack: [methods]. Compound gain: X% over baseline. Branch: ml-opt/stack-<N>"

## Stacking Phase Resumption

On pipeline restart, if `pipeline-state.json` contains a `stacking` key in `user_choices`:
1. Read stacking state
2. **Validate before resuming:**
   a. `current_stack_order < len(ranked_methods)` — if not, stacking is already complete; skip to Phase 9
   b. Verify `ml-opt/stack-<current_stack_order>` branch exists (`git branch --list`). If missing, log error to error tracker and skip to Phase 9 with partial results.
   c. Verify `stack_base_exp` result file is readable. If missing, fall back to the last known good stack result from `stacked_methods`.
3. Resume from `current_stack_order + 1`
4. Continue with remaining methods in `ranked_methods` that aren't in `stacked_methods` or `skipped_methods`
