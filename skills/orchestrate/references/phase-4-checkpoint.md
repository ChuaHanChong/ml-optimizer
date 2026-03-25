# Phase 4: User Checkpoint (Post-Baseline)

**Multi-objective scenarios:** When the baseline reveals multiple improvement vectors (e.g., both metric performance and training stability need work), consider using `superpowers:brainstorming` to explore trade-offs before presenting options to the user.

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
5. Skip to experiments with specific configs
```

## Phase 4, Option 1: Full Autonomous Optimization (default)

If the user selects option 1 (or doesn't specify a preference): Set `scope_level = "full"`. Proceed to Phase 5 (research) with `source: "web"`. The analysis agent will use all available tools: research, HP tuning, code_refinement via ShinkaEvolve. The pipeline runs fully autonomously until the goal is reached.

## Phase 4, Option 2: HP Tuning Only

If the user selects option 2: Set `scope_level = "training"`. Skip Phases 5-6 (no research, no code changes). Proceed directly to Phase 7 (experiment loop) with HP tuning on the baseline code branch only. The analysis agent will NOT recommend research or code_refinement pivots.

## Phase 4, Option 3: Research First

If the user selects option 3: Set `scope_level = "architecture"`. Proceed to Phase 5 (research) with `source: "web"`. The analysis agent can recommend research pivots but NOT code_refinement via ShinkaEvolve.

## Phase 4, Option 4: User-Provided Papers

If the user selects option 4:
1. Set `scope_level = "architecture"` (or `"full"` if user wants evolve too)
2. Use AskUserQuestion to collect paper URLs/paths (one per line)
3. Store as `user_papers` list in pipeline state user_choices
4. When invoking `ml-optimizer:research` in Phase 5, pass `user_papers`
5. The research skill will analyze user papers FIRST before running web searches
6. User-provided papers get a +2 confidence bonus in proposal ranking

## Phase 4, Option 5: Skip to Experiments

If the user selects option 5: Use AskUserQuestion to collect their specific HP configs. Set `scope_level = "training"`. Skip Phases 5-6 (no research, no code changes). Proceed to Phase 7 with the user-specified configs as the first experiment batch.
