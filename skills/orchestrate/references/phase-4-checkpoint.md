# Phase 4: User Checkpoint (Post-Baseline)

**Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, skip the user question below. Auto-select option 5 (method proposals) with `method_proposal_scope = "architecture"` (balanced default). Log to dev_notes: `"Autonomous mode: auto-selected method proposals (scope: architecture)"`. Then proceed directly to the Pre-Loop method proposal section (within Phase 7).

Use AskUserQuestion to show baseline results and ask for direction:

```
Baseline established:
[baseline metrics summary]

GPU memory usage: [X] MiB / [Y] MiB
Training throughput: [Z] samples/sec

How would you like to proceed?
1. Focus on HP tuning (recommended for quick wins)
2. Run research first (look for architectural improvements)
3. I have research/papers to share (provide your own findings)
4. Skip to experiments with specific configs
5. Propose new optimization methods (method proposals — LLM knowledge + optional web search)
```

## Phase 4, Option 1: HP Tuning Only

If the user selects option 1: Skip Phases 5-6 (no research, no code changes). Proceed directly to Phase 7 (experiment loop) with HP tuning on the baseline code branch only.

## Phase 4, Option 2: Research First

If the user selects option 2: Proceed to Phase 5 (research) with `source: "web"`.

## Phase 4, Option 3: User-Provided Papers

If the user selects option 3:
1. Use AskUserQuestion to collect paper URLs/paths (one per line)
2. Store as `user_papers` list in pipeline state user_choices
3. When invoking `ml-optimizer:research` in Phase 5, pass `user_papers`
4. The research skill will analyze user papers FIRST before running web searches
5. User-provided papers get a +2 confidence bonus in proposal ranking

## Phase 4, Option 4: Skip to Experiments

If the user selects option 4: Use AskUserQuestion to collect their specific HP configs. Skip Phases 5-6 (no research, no code changes). Proceed to Phase 7 with the user-specified configs as the first experiment batch.

## Phase 4, Option 5: Method Proposals

If the user selects option 5:
1. Use AskUserQuestion to confirm scope level:
   ```
   What scope of changes should I propose?
   1. Training strategies only (optimizer, schedulers, regularization, augmentation) — safest
   2. Training + architecture changes (attention, normalization, activation variants) — bolder
   3. Full scope (everything including data pipeline and loss functions) — most aggressive
   ```
2. Store `method_proposal_scope` in pipeline state user_choices (values: `"training"`, `"architecture"`, `"full"`)
3. Skip Phase 5 (web research) — method proposals will be generated in Phase 7 Pre-Loop
