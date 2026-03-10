# Method Stacking Phase — Design Spec

## Context

The ml-optimizer plugin tests research-proposed methods independently on separate git branches (`ml-opt/<slug>`), with three-tier attribution tracking (baseline → method+default_HP → method+tuned_HP). However, it does not combine successful methods to capture compound gains — each branch is tested in isolation.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch), which greedily stacks improvements cumulatively, this feature adds a **stacking phase** that merges winning methods sequentially after independent validation, preserving attribution at each accumulation step.

## Design

### Pipeline Position

Stacking inserts as **Phase 6.5** between the existing experiment loop (Phase 6) and the report (Phase 7):

```
Phase 6:   Independent method testing + HP-tuning (existing)
Phase 6.5: Method stacking (new) — sequential accumulation of winners
Phase 7:   Report (existing, extended with stacking results)
```

### Trigger

The orchestrator tracks `methods_with_improvement` — a count of validated methods whose best result (either `method_default_hp` or `method_tuned_hp`) beats baseline.

**When `methods_with_improvement >= 5`** and the experiment loop ends (analyze recommends `stop`, or budget is exhausted for independent testing):

- **Interactive mode:** Ask user: "5 methods showed improvement over baseline. Would you like to stack them to find compound gains?"
- **Autonomous mode:** Auto-proceed with stacking.

The threshold of 5 is chosen because it provides enough methods for meaningful combinations without making the stacking phase itself expensive (5 stacking experiments max).

### Sequential Accumulation Algorithm

```
1. Rank all methods_with_improvement by individual improvement magnitude (descending)
2. Set stack_base = branch of best method (rank 1)
3. For each remaining method (rank 2, 3, ... N):
   a. Create branch ml-opt/stack-<order> from current stack_base
   b. git merge ml-opt/<method-slug> into ml-opt/stack-<order>
   c. If clean merge → proceed
   d. If conflicts → dispatch implement-agent to resolve
   e. If resolution fails → skip this method, log reason, continue
   f. Run validation (syntax, import, forward_pass)
   g. If validation fails → skip, log, continue
   h. Run experiment with best individual HPs from the top method in stack
   i. If improved over previous stack step:
      - Keep: update stack_base = ml-opt/stack-<order>
      - Optionally HP-tune (1-2 iterations) if LLM judges worthwhile
   j. If worse → skip this method, log, try next
4. Final stack_base = compound best result
```

### Branch Naming

- `ml-opt/stack-1` = best individual method (copied from its original branch)
- `ml-opt/stack-2` = best + second-best merged
- `ml-opt/stack-3` = stack-2 + third-best merged
- Skipped methods don't consume a stack number

### Conflict Resolution

When `git merge` produces conflicts:
1. Dispatch implement-agent with conflicting files, both method descriptions, and instruction: "Resolve conflicts preserving both methods' functionality"
2. If implement-agent succeeds → run validation before proceeding
3. If implement-agent fails → skip this method, log `stacking_conflict_unresolved` to error tracker

### HP Strategy During Stacking

- **Initial test:** Use the HP config that produced the best result for the top method in the current stack
- **If combo improves:** LLM decides whether to HP-tune (1-2 iterations max). Factors: magnitude of improvement, remaining budget, whether the added method interacts with key HPs (e.g., a new loss function might need different learning rates)
- **HP-tune scope:** Only vary HPs that the newly added method might interact with (LLM judgment), not the full search space

### New Tracking Fields

**New `method_tier` values:**
- `"stacked_default_hp"` — combination tested with best individual HPs
- `"stacked_tuned_hp"` — combination after HP-tuning

**New result JSON fields:**
```json
{
  "code_branches": ["ml-opt/perceptual-loss", "ml-opt/cosine-scheduler"],
  "stacking_order": 2,
  "stack_base_exp": "exp-012"
}
```

- `code_branches` (array): all methods in this stack. Existing `code_branch` (string) remains for single-method experiments — backward compatible.
- `stacking_order` (int): position in the accumulation chain (1 = best method alone, 2 = best + second, etc.)
- `stack_base_exp` (string): experiment ID of the previous stack step this builds on.

### Report Extension

The report skill generates a new section:

```markdown
## Method Stacking Results

| Stack | Methods Added | Metric | vs Baseline | vs Previous Stack | Status |
|-------|---------------|--------|-------------|-------------------|--------|
| 1 | perceptual-loss | 0.85 | +5.0% | — | kept |
| 2 | + cosine-scheduler | 0.82 | +8.2% | +3.2% | kept |
| 3 | + mixup | 0.87 | — | -5.7% | skipped |
| 4 | + label-smoothing | 0.80 | +10.5% | +2.4% | kept |
| 5 | + dropout-tuning | 0.79 | +11.6% | +1.2% | kept |

Final stack: perceptual-loss + cosine-scheduler + label-smoothing + dropout-tuning
Compound gain: +11.6% over baseline
Branch: ml-opt/stack-4
```

### Stacking State Persistence

The stacking phase state is saved in `pipeline-state.json` for resumption:
```json
{
  "stacking": {
    "ranked_methods": ["perceptual-loss", "cosine-scheduler", "mixup", ...],
    "current_stack_order": 3,
    "stack_base_branch": "ml-opt/stack-2",
    "stack_base_exp": "exp-015",
    "skipped_methods": ["mixup"],
    "stacked_methods": ["perceptual-loss", "cosine-scheduler"]
  }
}
```

## Files to Modify

| File | Change |
|------|--------|
| `skills/orchestrate/SKILL.md` | Add Phase 6.5 stacking phase, `methods_with_improvement` counter, stacking trigger, accumulation loop, state persistence |
| `skills/analyze/SKILL.md` | Include `methods_with_improvement` count in analyze output |
| `agents/experiment-agent.md` | Add `code_branches`, `stacking_order`, `stack_base_exp` to result schema |
| `scripts/result_analyzer.py` | Extend `group_by_method_tier()` for `stacked_*` tiers; add `rank_methods_for_stacking()` function |
| `scripts/schema_validator.py` | Validate new `stacked_*` tier values and `code_branches` array field |
| `scripts/pipeline_state.py` | Add stacking state to schema validation |
| `skills/report/SKILL.md` | Add stacking results table section |
| `skills/hp-tune/SKILL.md` | Handle stacked branch HP-tuning (use best individual HPs as starting point, narrow scope) |
| `.claude/CLAUDE.md` | Document stacking phase in architecture, design patterns, and gotchas |
| `tests/` | Add tests for `rank_methods_for_stacking()`, stacking tier validation, stacking state persistence |

## Verification

1. **Unit tests:** `rank_methods_for_stacking()` correctly ranks methods by improvement; `group_by_method_tier()` handles `stacked_*` tiers; schema validates new fields
2. **Integration test:** End-to-end stacking flow with mock results — verify branch creation, skip-on-failure, state persistence
3. **Manual test:** Run `/optimize` on a test project with method proposals, verify stacking triggers after 5 improvements
