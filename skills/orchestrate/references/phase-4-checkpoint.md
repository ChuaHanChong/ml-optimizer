# Phase 4: User Checkpoint (Post-Baseline)

**Phase gate:** Run `pipeline_state.py <exp_root> gate 3 4` before entering. On completion: `pipeline_state.py <exp_root> log-gate 4 completed "<summary>"`.

**Multi-objective scenarios:** When the baseline reveals multiple improvement vectors (e.g., both metric performance and training stability need work), consider `superpowers:brainstorming` to explore trade-offs before presenting options.

Use AskUserQuestion to show baseline results and ask for direction:

```
Baseline established:
[baseline metrics summary]

GPU memory usage: [X] MiB / [Y] MiB
Training throughput: [Z] samples/sec

How would you like to proceed?
1. Full autonomous optimization (research + HP tuning + ShinkaEvolve — recommended)
2. HP tuning only (fastest, no code changes)
3. Run research first (look for methods before tuning)
4. I have research/papers to share (provide your own findings)
5. Skip to experiments (no research, no code changes — iteration 1 runs the standard baseline-HP tuning proposal)
```

## Phase 4, Option 1: Full Autonomous Optimization (default)

If the user selects option 1 (or doesn't specify): Set `scope_level = "full"`. Proceed to Phase 5 (research) with `source: "web"`. The analysis agent uses all tools: research, HP tuning, code_evolution via ShinkaEvolve. Runs fully autonomously until the goal is reached.

## Phase 4, Option 2: HP Tuning Only

If option 2: Set `scope_level = "training"`. Skip Phases 5-6 (no research, no code changes). Proceed directly to Phase 7 (experiment loop) with HP tuning on the baseline code branch only. The analysis agent will NOT recommend research or code_evolution pivots.

## Phase 4, Option 3: Research First

If option 3: Set `scope_level = "architecture"`. Proceed to Phase 5 (research) with `source: "web"`. The analysis agent can recommend research pivots but NOT code_evolution via ShinkaEvolve.

## Phase 4, Option 4: User-Provided Papers

If option 4:
1. Set `scope_level = "architecture"` (or `"full"` if the user wants evolve too)
2. Use AskUserQuestion to collect paper URLs/paths (one per line)
3. Store as `user_papers` list in pipeline state user_choices
4. Pass `user_papers` as a `phase-5-research` workflow arg (`args.user_papers`) when launching Phase 5 (`Workflow({scriptPath: ".../phase-5-research.js", args})`)
5. The research skill analyzes user papers FIRST, before web searches
6. User-provided papers get a +2 confidence bonus in proposal ranking

## Phase 4, Option 5: Skip to Experiments

If option 5: Set `scope_level = "training"`. Skip Phases 5-6 (no research, no code changes). Proceed directly to Phase 7. Note: there is currently no mechanism to pass user-specified HP configs into the Phase 7 workflow `args` — iteration 1 always runs the standard baseline-HP-per-branch tuning-agent proposal (same as Option 2), regardless of any configs the user names here.

## Pre-Authorize Phase 7 Autonomy (MANDATORY before any workflow launches)

Phases 5–8 run as dynamic workflows that take **no mid-run user input**; the Phase 7 experiment workflow runs fully autonomously to its fixpoint. So every user decision the loop would otherwise ask for must be pre-authorized **here**, at Phase 4, and persisted into `user_choices` so it can be passed as Phase 7 `args`. Do NOT defer these to mid-loop — no prompt is available once the workflow is running.

Use AskUserQuestion (or auto-select in autonomous mode) to lock in:

- **`method_proposal_scope`** — the scope ceiling for any mid-loop method proposals the workflow generates (replaces the old mid-loop "Scope options 1/2/3/4" prompt). Choose one:
   - `"training"` — training strategies only (optimizers, schedulers, regularization, augmentation, loss functions)
   - `"architecture"` — training + architecture changes (attention, normalization, activations, block design)
   - `"full"` — training + architecture + data pipeline, distillation, ensemble
   - `null` — no method proposals; the workflow runs HP-only and never researches mid-loop
   (Constrain the choice to at most the Phase-4 `scope_level`: option 2 → `null` only; option 3 → up to `"architecture"`; option 1 → up to `"full"`.)

- **`method_proposal_iterations`** — the budget/cadence for method-proposal rounds. Set `0` to disable, or a positive integer for how many mid-loop research→implement rounds the workflow may run. Together with `hp_batches_per_round`, this controls the cadence-based research trigger inside the workflow.

- **`hp_batches_per_round`** — how many HP-tuning batches run between cadence-based research rounds (default 3).

- **`budget`** — the training budget that bounds the loop: `fixed_time_budget` (seconds), `fixed_epoch_budget` (integer epochs), or `fixed_step_budget` (integer environment timesteps, for RL), or `null` for the default per-experiment timeout. Pre-confirm so the workflow never needs to ask.

- **Auto-confirm method proposals.** The old mid-loop "Present proposals to user for confirmation" step cannot run inside the workflow. By pre-authorizing scope + iterations here, the user delegates proposal acceptance to the workflow (it auto-implements in-scope, non-dead-end proposals up to `method_proposal_iterations`). To review proposals individually, choose `method_proposal_scope = null` and run research as a separate pre-loop pass (Phase 5) with the post-research checkpoint.

- **`seeds_per_config`** — how many random seeds each proposed config runs with (default 1; suggest >1 for RL — single-seed RL results are noisy, and the measured replicate spread becomes the analysis noise floor). Replicates count against the per-batch GPU slots.

- **`experiments_per_gpu`** — how many experiments may run concurrently per GPU (CPU-bound parallelism ceiling; default 1). Pre-confirm so the workflow never needs to ask.

(These are unnumbered on purpose — "Option N" elsewhere in this file always means the AskUserQuestion menu above, never an item in this list.)

Persist all of these into `user_choices` (e.g. via a `python3 -c` snippet calling `pipeline_state.save_state(phase, iteration, [], exp_root, user_choices={...})` directly — the bare `pipeline_state.py save` CLI subcommand only accepts `<phase> <iteration> [running_ids_json]`, it has no `user_choices` parameter) so they survive interruptions and are read straight into the Phase 7 workflow `args`:
`{ method_proposal_scope, method_proposal_iterations, hp_batches_per_round, fixed_time_budget, fixed_epoch_budget, fixed_step_budget, seeds_per_config, experiments_per_gpu }`.
When building the phase-7 `args`, pass `fixed_time_budget`, `fixed_epoch_budget`, and `fixed_step_budget` through as the three typed fields (do NOT collapse them into a single scalar `budget` — the experiment agent needs to know whether to wrap `timeout`, cap epochs, or cap environment timesteps). Any or all may be `null`.

A genuine user-decision point (e.g., the user explicitly wants to inspect mid-run results) is handled as a **workflow boundary**: the workflow returns to the orchestrator, the orchestrator runs the checkpoint, then relaunches the continuation via `resumeFromRunId`. It is never an in-workflow prompt.
