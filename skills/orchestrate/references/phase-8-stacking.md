# Phase 8: Method Stacking (Sequential Accumulation)

**Pre-check:** If the implementation manifest uses `strategy: "file_backup"` (non-git project), skip stacking entirely. Log to dev_notes: "Stacking requires git branches — skipped for file-backup projects." Proceed to Phase 9.

**Trigger:** When the experiment loop ends (analyze recommends `stop` or budget exhausted) AND `methods_with_improvement >= 5`.

Count `methods_with_improvement` by calling `rank_methods_for_stacking()` from `result_analyzer.py`:
```bash
python3 scripts/result_analyzer.py <results_dir> <metric> [baseline_id] [lower_is_better]
```
Then count entries in the result. If fewer than 5, skip to Phase 9.

**Checkpoint:**
- **Interactive mode:** Ask user: "{N} methods showed improvement over baseline. Would you like to stack them to find compound gains? The best methods will be merged sequentially."
  - If user declines → skip to Phase 9
- **Autonomous mode:** Auto-proceed. Log to dev_notes: "Auto-entering stacking phase with {N} improved methods."

## Stacking Loop

1. **Rank methods** by improvement magnitude (descending) using `rank_methods_for_stacking()`.

2. **Initialize stack:** The best method's branch becomes `ml-opt/stack-1`. No experiment needed — its existing best result serves as the stack baseline.

3. **For each remaining method** (rank 2, 3, ... N):

   a. **Create stack branch:**
   ```bash
   git checkout -b ml-opt/stack-<order> ml-opt/stack-<order-1>
   # (For order=2, branch from ml-opt/stack-1 which is the best method's branch)
   ```

   b. **Merge the next method:**
   ```bash
   git merge ml-opt/<method-slug> --no-ff --no-edit
   ```

   c. **If clean merge** → proceed to validation.

   d. **If merge conflicts** → dispatch implement-agent:
      - **Prompt:** "Resolve merge conflicts in the following files. Both methods must be preserved: [method-A description] and [method-B description]. The goal is to combine their functionality."
      - **Files:** List of conflicting files from `git diff --name-only --diff-filter=U`
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
        - **Optional HP-tune:** If the improvement is > 1% AND remaining budget allows, dispatch the tuning agent:
          ```
          Agent(
            description: "HP-tune stacked method stack-{order}",
            prompt: "Ultrathink. Propose HP configs for stacked method. Parameters: project_root: {project_root}, num_gpus: {num_gpus}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, code_branches: [ml-opt/stack-{order}], iteration: 1, remaining_budget: {min(2, actual_remaining)}, search_space: {narrowed_search_space}.",
            subagent_type: "ml-optimizer:tuning-agent"
          )
          ```
        - If HP-tune improves further, record as `method_tier: "stacked_tuned_hp"`
      - **If worse or equal:** Skip this method.
        - Delete `ml-opt/stack-<order>` branch
        - Log: "Method <slug> skipped in stacking (metric degraded by X%)"
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
         "stacked_methods": ["method-b", "method-a"]
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
