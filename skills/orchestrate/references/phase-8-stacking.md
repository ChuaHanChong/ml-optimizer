# Phase 8: Method Stacking (Workflow)

**Phase gate:** Run `pipeline_state.py <exp_root> gate 7 8` before entering. On completion: `pipeline_state.py <exp_root> log-gate 8 completed "<summary>"`.

Phase 8 runs as a **dynamic workflow** (`Workflow({scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-8-stacking.js", args})`). The orchestrator launches it when the phase-7 workflow returns non-empty `stacking_candidates`. The stacking loop lives inside the workflow script: it holds the "current best stack" in a script variable and dispatches existing agents internally via `agentType` (`agent(prompt, {agentType: "ml-optimizer:<name>-agent"})`).

> "Dispatch the X agent" = an `agent({agentType: "ml-optimizer:X-agent"})` call inside the workflow script, NOT an orchestrator `Agent()`/`SendMessage` call. No message bus, no `SendMessage`, no agent registry — cross-agent context flows via `args` + the files agents write under `<exp_root>/`.

## Workflow Args & Return

**Args (in):** `{ exp_root, project_root, primary_metric, lower_is_better, baseline_metric, scope_level, divergence_metric, divergence_lower_is_better, model_category, fixed_time_budget, fixed_epoch_budget, fixed_step_budget, stacking_candidates }`
where `stacking_candidates: [{branch, improvement_pct}, ...]` comes from the phase-7 return. `baseline_metric` (the baseline's primary-metric value) lets the workflow derive the seed method's absolute metric so the first accumulation is compared, not force-kept. `scope_level` gates the code_evolution/ShinkaEvolve repair (only runs at `"full"`, mirroring Phase 7). `fixed_step_budget`/`fixed_time_budget`/`fixed_epoch_budget` cap each stacked run the same way the baseline was (step budget takes precedence, for RL timestep-based training), so metrics stay comparable. `divergence_metric`/`divergence_lower_is_better`/`model_category` build the same divergenceClause Phase 7 uses, appended to every stacked experiment prompt — stacked runs get identical divergence handling to Phase 7 runs.

**Return (out):** `{ best_stack_branch, best_stack_metric, steps: [{method, branch, kept}, ...] }`

Method stacking combines different implementations (papers, LLM patches, or ShinkaEvolve) into one model. The analyze skill's Pivot Decision Tree does let the analysis agent set `pivot_type: "method_stacking"` as an advisory classification — but that field is not what triggers Phase 8. The phase-7 workflow script's `decision` field (the one it actually dispatches on) has no `method_stacking` value at all, and no `pivot_type` value drives Phase 8 either. Instead, `stacking_candidates` accumulate continuously across the whole Phase 7 run: every analysis-agent dispatch may report `stacking_candidates: [{branch, improvement_pct}, ...]` in its returned schema, and after each batch the phase-7 workflow independently harvests candidates by comparing each completed experiment's branch result directly against the baseline metric. Candidates are deduped by branch (best improvement kept) as the loop progresses, regardless of whichever `decision`/`pivot_type` analysis returns that round. The orchestrator launches Phase 8 when the phase-7 workflow returns a non-empty `stacking_candidates` list at exit — not by a fixed method count and not gated by any specific decision or pivot_type on a given batch.

**Pre-check:** The workflow filters `stacking_candidates` down to entries with a valid, non-empty `branch` string (`candidates = stacking_candidates.filter(c => typeof c.branch === "string" && c.branch.length > 0)`) — this defensively covers non-git/file_backup projects, which never produce branch-bearing candidates. If the filtered list is empty, the workflow returns `{ best_stack_branch: null, best_stack_metric: null, steps: [] }` immediately and logs the reason. Log to dev_notes.

After stacking completes, the orchestrator may relaunch the phase-7 workflow to continue optimizing on the stacked code.
- **Autonomous mode:** The workflow auto-proceeds. Log to dev_notes: "Auto-entering stacking phase with {N} improved methods."

## Stacking Loop

1. **Rank methods by improvement magnitude** (descending) — the largest improvement over baseline gets stacked first (no archive lineage data needed — methods are stacked in order of effectiveness).

2. **Initialize stack:** The best method's branch becomes the *logical* `stack-1` base — `stackBaseBranch` is set to the seed method's own existing branch name; no new branch is created for it. No experiment needed — its baseline metric is derived arithmetically from `baseline_metric` and its known `improvement_pct` (see below), not read from its original result file; if `baseline_metric` isn't available, the first accumulation is force-kept instead.

3. **For each method** (in improvement-ranked order):

   a-d. **One unified merge dispatch** — not four separate steps. A single implement-agent call via `agent({agentType: "ml-optimizer:implement-agent"})`, `isolation: "worktree"`, instructs the agent to itself: create `ml-opt/stack-<order>` from `stack_base_branch` (`git checkout -b ml-opt/stack-<order> <stack_base_branch>`; for order=2, `stack_base_branch` is the seed method's own branch — no `ml-opt/stack-1` branch is ever created, `ml-opt/stack-2` is the first real stack branch cut), merge the next method (`git merge ml-opt/<method-slug> --no-ff --no-edit`), resolve conflicts itself if any occur (preserving both the existing stack's and the new method's functionality, then commit), validate (syntax/imports/forward-smoke-pass), and report one status: `"merged_clean"`, `"merged_resolved"`, `"conflict_unresolved"` (agent ran `git merge --abort` itself), or `"validation_failed"`. Returns `{status, branch, conflicting_files, notes}`.
      - If `status` is `"conflict_unresolved"` or `"validation_failed"` (or the dispatch failed) → skip this method: log via the workflow's own `log()` call (dev-notes-style, not error_tracker.py): `Stack step {order}: {method.branch} skipped ({status}).`, record `kept: false`, continue to next method. `stackOrder` is NOT reused — the merge agent already cut the branch, so gaps in the numbering are harmless.

   (Lettering here is this doc's own sequencing, not a 1:1 map to `phase-8-stacking.js`'s inline step comments — the script labels its review block "Step e"; in this doc's numbering that's e1, below.)

   e. **Validate** (syntax, import, forward pass — same as implement skill validation). This happens inside the same merge dispatch as steps a-d above, not a separate call.
      - If validation fails → skip: leave the branch in place (not deleted), log reason, continue.

   e1. **Review the merged branch** (mirror Phase 6 per-branch review): dispatch `pr-review-toolkit:code-reviewer` and `pr-review-toolkit:silent-failure-hunter` in parallel via `agent({agentType: "pr-review-toolkit:code-reviewer"})`/`agent({agentType: "pr-review-toolkit:silent-failure-hunter"})`, diffing the merged branch against both the prior stack base and the newly merged method to focus on the merge boundary (where a conflict resolution can silently drop a NaN/CUDA guard). A critical finding from either → skip this stack step: leave the branch in place (not deleted), log reason, continue.

   e2. **Create a stacking round** — not a separate workflow-level call with its own captured `round_dir`. The `round_manager.py create-round stacked --branch ml-opt/stack-{order}` instruction is inlined into the SAME experiment-agent dispatch prompt as step f below; the experiment-agent creates the round itself as a side effect (before writing results). The step-f dispatch's Return instruction never asks for `round_dir` back (it's an optional, non-required schema property that's never actually populated) — the workflow later re-locates the round directory by searching for the result exp_id (see `verifyStackStep`) rather than consuming a returned `round_dir` field.
      Without this, the PreToolUse hook blocks the experiment's result write.

   f. **Run experiment** via `agent({agentType: "ml-optimizer:experiment-agent"})` with prompt (real fields — no `round_dir`, `train_command`, or `eval_command` are passed in; the agent creates the round itself per e2 above and resolves training/eval commands from its own skill context):
      ```
      "Run stacking experiment for {stackBranch} (stack step {stackOrder}). Project root: {project_root}. exp_root: {exp_root}. Before writing results, create a stacked round: round_manager.py <exp_root> create-round stacked --branch {stackBranch}. Code branch: {stackBranch}. Method tier: stacked_default_hp. Code branches (combined): {code_branches_json}. Stacking order: {stackOrder}. Stack base exp: {stack_base_exp or 'this is the first re-run of an accumulated stack'}. Primary metric: {primary_metric}. Run inside an isolated git worktree on {stackBranch} using --detach."
      ```
      Runs with `isolation: "worktree"` in the dispatch opts, same as Phase 7's experiment dispatch.

   g. **Evaluate result:** The analysis-agent is dispatched unconditionally, for every completed stacked experiment, BEFORE the improved-vs-stop decision below — not nested inside an "if improved" branch.

        - **Analyze stacked result:** Dispatch the analysis agent via `agent({agentType: "ml-optimizer:analysis-agent"})` to assess whether the stacked code needs evolution or HP-tuning, with prompt:
          ```
          "Analyze stacked experiment result. Parameters: project_root: {project_root}, batch_number: stack-{order}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, exp_root: {exp_root}. CONTEXT: stacked experiment on ml-opt/stack-{order}: {stacked_metrics}; methods combined: {stacked_methods}, stacking order: {order}; best single method gain: {best_individual_gain}%. Compare the stacked gain to the best individual method gain. If the stack underperforms the best individual (methods interfering), recommend code_evolution. Otherwise recommend continue."
          ```
          The workflow computes the stacked metric from the just-completed experiment result and the best-individual gain from the ranked `stacking_candidates`, and passes both as literal values in the prompt above; the agent additionally reads the round's result JSON for extra context.

      - **Kept only if BOTH hold:** `recommendation !== "stop"` AND the raw metric did not regress vs. the current stack base. If both pass:
        - Update `stack_base_branch = ml-opt/stack-<order>`

        - **If analysis recommends `recommendation: "code_evolution"`:**

          **Step 1: Tune evolve HPs.** Dispatch the tuning agent via `agent({agentType: "ml-optimizer:tuning-agent"})` with prompt:
          "Propose ShinkaEvolve evolution HPs for stacking code_evolution. Stacked methods: {stacked_methods}. Read `learned-behaviors.json` category `evolve_hp` for prior outcomes. Return `evolve_recommendation: {num_generations, population_size, reasoning}`."

          **Step 2: Execute evolve.** Dispatch the implement agent via `agent({agentType: "ml-optimizer:implement-agent"})` with the evolve skill.
          Prompt: `Skill("ml-optimizer:evolve")` with parameters:
            - `project_root`, `parent_branch: ml-opt/stack-{order}`, `parent_metrics: {stacked metrics}`
            - `primary_metric`, `lower_is_better`, `scope_level`, `exp_root`
            - `evolve_recommendation`: from tuning agent (Step 1)
            - `feedback_context`: the dispatch instructs the implement-agent to read the dead-end catalog, recurring error patterns, the most recent batch analysis, and `learned-behaviors.json` itself before evolving, and pass them as `{dead_ends, error_patterns, batch_analysis, learned_behaviors}` — matching the evolve skill's own Input Parameters contract (`skills/evolve/SKILL.md`). The agent self-serves these file reads rather than the workflow script pre-fetching them, consistent with how other dispatches in this codebase read agenda/dead-end context.

          **Handle evolve result:**
          - If `status == "validated"`:
            - Run experiment on the evolved branch (`ml-opt/evolved-*`)
            - If evolved result beats the pre-evolution stack → the script promotes it in-memory (updates local `keptBranch`/`keptMetric`/`keptExp`, which become `stackBaseBranch`/`stackBaseMetric`/`stackBaseExp` for the next stack step) — local to the running workflow, not written to any persisted "stacking state"
            - If evolved result is worse → discard evolved branch, keep pre-evolution `ml-opt/stack-<order>` as stack base
          - If `status == "shinkaevolve_unavailable"` or `"validation_failed"`:
            - Log via the workflow's own `log()` call: `Stack step {order}: evolution unavailable or failed ({status}); continuing without it.` — continue without evolution (proceed to HP-tune)

        - **HP-tune on the (potentially evolved) stack branch, gated on combo gain >1%:** Dispatch the tuning agent via `agent({agentType: "ml-optimizer:tuning-agent"})` — real fields, no `num_gpus` (never referenced anywhere in phase-8-stacking.js): `exp_root: {exp_root}, project_root: {project_root}, code_branches: [{keptBranch}], primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, iteration: 1`, instructed to narrow the search space around the stacked methods' best HPs rather than re-exploring widely. Returns `configs: [{exp_id, config, reasoning}]`. Run experiments via `agent({agentType: "ml-optimizer:experiment-agent"})` — each dispatch is tagged `method_tier: "stacked_tuned_hp"` unconditionally, before the run happens; the improvement check afterward only decides whether that run's metric promotes the kept stack base for the next step.

        - **Promotion after evolve/HP-tune is a direct numeric comparison, not a second analysis call.** There is exactly one analysis-agent dispatch per stack step (the "Analyze stacked result" call above) — evolve and HP-tune promotion both use a plain `isBetter()` comparison against the pre-evolve/pre-tune kept metric (evolved branch: promoted only if it beats the pre-evolution stack metric; each HP-tune config: promoted only if it beats the current kept metric). No second analysis-agent call, and no possibility of a fresh `stop` once the step has cleared its initial stop/regression gate.

      - **Not kept** (`recommendation === "stop"` OR the metric regressed vs. the current stack base): Skip this method.
        - Leave `ml-opt/stack-<order>` in place — skipped-step branches are deliberately NOT deleted (gaps in the stack numbering are harmless; matches the workflow's own behavior)
        - Log: workflow's own `log()` call reports the branch and reason (see step h below for the exact format and the two possible reasons)
        - Continue to next method (next stack branch re-branches from the last successful stack)

   h. **Round closure is inlined, not a separate workflow-level call.** Like round creation (step e2), closing the round is embedded as free-text instruction inside each experiment-agent dispatch prompt — there is no standalone `close-round --summary` call at the workflow level. Each of the 3 dispatch prompts that can close a stacking round tells the experiment-agent to do so itself after its run:
      - Base stacked exp (step f): `"After the run completes, register and close the round via round_manager.py."`
      - Evolved exp (interference repair): `"Register and close the round when done."`
      - Each narrowed HP-tune exp (interference repair): `"Register and close the round when done."`

      Once the experiment has completed and analysis has run, the step is kept unless one of two conditions holds: analysis recommends stop, or the result regresses against the current stack base (`recommendation === "stop"` -> "analysis stop"; otherwise "regression vs current stack"), both logged by the workflow itself. (Earlier in the loop a step can also be skipped for other reasons: merge `conflict_unresolved`/`validation_failed` per step a-d, a critical review finding per e1, or the experiment not completing per step f.)

4. **Final result:** The last KEPT branch is the compound best — usually the final step's `ml-opt/stack-<N>` merge branch, but if that step's code_evolution repair was promoted (g, "Handle evolve result"), it's the evolved branch `ml-opt/evolved-stack-<N>` instead (the local `keptBranch`/`keptMetric`/`keptExp` that becomes `stackBaseBranch`/`stackBaseMetric`/`stackBaseExp`). The workflow returns `{ best_stack_branch, best_stack_metric, steps: [{method, branch, kept}, ...] }` to the orchestrator.
   Log to dev_notes: "Stacking complete. Final stack: [methods]. Compound gain: X% over baseline. Branch: ml-opt/stack-<N>"

## Stacking Phase Resumption

Within a session, the phase-8 workflow is resumable via the `Workflow` runtime's own `resumeFromRunId` — relaunch with the prior run id to continue where it stopped. There is currently no cross-session persistence of intermediate stacking progress: `pipeline-state.json` holds no `stacking` key, and the workflow's only output is its final `{ best_stack_branch, best_stack_metric, steps }` return. If a session restarts after Phase 8 was interrupted, the orchestrator re-launches `phase-8-stacking.js` as a new run (recomputing `stacking_candidates` from the phase-7 return) rather than resuming mid-stack.
