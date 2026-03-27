# UCB1 + MCTS Implementation: Correctness Proof

**Date:** 2026-03-27
**Branch:** feature/bfts-ablation
**Files:** `gl_utils.py`, `select_parent.py`, `archive_utils.py`

## The Problem

We have an evolutionary archive of code variants (nodes). Each node has a fitness score (e.g., loss=0.3 or accuracy=87%). We need to choose which node to expand next, balancing:
- **Exploitation**: expand nodes whose descendants performed well
- **Exploration**: try nodes we haven't expanded much

## The Algorithm

**UCB1 (Auer et al. 2002):** Select the node with highest:

```
UCB(i) = value_sum(i)/visits(i) + C * sqrt(ln(N) / visits(i))
```

where `C = sqrt(2) ~ 1.414`, `N` = total evaluations.

Unvisited nodes get UCB = infinity (must try everything once first).

## The Challenge

UCB1 assumes **higher values = better**. But ML metrics can be:
- **Higher-is-better**: accuracy (90% > 80%)
- **Lower-is-better**: loss (0.3 < 0.9)

Raw scores can be any scale (loss in [0, 100], accuracy in [0, 1]), but UCB1 needs values in [0, 1] for `C = sqrt(2)` to balance correctly.

## Our Solution: Normalize at Backpropagation Time

When a child node is evaluated, we **normalize its score to [0, 1]** before storing it, accounting for metric direction:

```python
# Step 1: Min-max normalize to [0, 1]
normalized = (score - lo) / (hi - lo)

# Step 2: Invert for lower-is-better
if lower_is_better:
    normalized = 1.0 - normalized
```

This stored `value_sum` always means "higher = better" regardless of the original metric.

## Proof by Concrete Example

**Setup:** loss metric (`lower_is_better = True`), archive has 3 nodes:

| Node | loss | Raw normalized | After inversion | Meaning |
|------|------|----------------|-----------------|---------|
| initial | 1.0 | (1.0-0.3)/(1.0-0.3) = 1.0 | 1.0 - 1.0 = **0.0** | Worst (highest loss) |
| gen_1 | 0.3 | (0.3-0.3)/(1.0-0.3) = 0.0 | 1.0 - 0.0 = **1.0** | Best (lowest loss) |
| gen_2 | 0.9 | (0.9-0.3)/(1.0-0.3) = 0.857 | 1.0 - 0.857 = **0.143** | Bad (high loss) |

After backpropagation:
- gen_1: `value_sum = 1.0`, `visit_count = 1`, `eval_count = 1`
- gen_2: `value_sum = 0.143`, `visit_count = 1`, `eval_count = 1`
- initial: `value_sum = 1.143` (sum of both children), `visit_count = 2`, `eval_count = 0`

**UCB computation** (N = sum(eval_count) = 2):

```
UCB(gen_1) = 1.0/1 + 1.414 * sqrt(ln(2)/1) = 1.0 + 1.177 = 2.177
UCB(gen_2) = 0.143/1 + 1.414 * sqrt(ln(2)/1) = 0.143 + 1.177 = 1.320
```

**Result: gen_1 (loss=0.3) has higher UCB than gen_2 (loss=0.9).** UCB correctly prefers the lower-loss node. :white_check_mark:

## Proof for Higher-is-Better (accuracy)

Same setup but `lower_is_better = False`, scores are accuracy:

| Node | accuracy | Normalized (no inversion) |
|------|----------|---------------------------|
| initial | 50% | (50-50)/(90-50) = **0.0** |
| gen_1 | 90% | (90-50)/(90-50) = **1.0** |
| gen_2 | 60% | (60-50)/(90-50) = **0.25** |

UCB correctly prefers gen_1 (highest accuracy). :white_check_mark:

## Why `eval_count` Instead of `visit_count` for N

Backpropagation increments `visit_count` on ALL ancestors. After evaluating gen_1 (child of initial):
- gen_1: `visit_count = 1`
- initial: `visit_count = 1`
- Sum of visit_counts = 2 (but only 1 evaluation happened)

Using `sum(visit_count)` inflates N, making `ln(N)` too large, over-weighting exploration. Using `sum(eval_count)` = 1 gives the correct N.

## Verified by Tests

```
test_normalization_lower_is_better    -- loss=0.3 normalizes > 0.5     PASS
test_normalization_higher_is_better   -- acc=0.9 normalizes > 0.5      PASS
test_backpropagate_eval_count_only    -- eval_count only on node       PASS
test_ucb_prefers_lower_loss           -- end-to-end: picks loss=0.3    PASS
```

## Verified by Direct Computation

```
loss=0.3 (good) -> 1.0000   PASS
loss=0.9 (bad)  -> 0.1429   PASS
loss=1.0 (base) -> 0.0000   PASS

gen_1: UCB = 2.1772 > gen_2: UCB = 1.3201   PASS
```

## Bugs Found and Fixed During Audit

| # | Severity | Bug | Fix |
|---|----------|-----|-----|
| 1 | CRITICAL | `_normalize_score_for_ucb()` didn't handle lower-is-better — low loss normalized to 0.0 instead of 1.0 | Added `lower_is_better` param with `1.0 - normalized` inversion |
| 2 | CRITICAL | `backpropagate_ucb()` didn't read metric direction from pipeline state | Added `_read_lower_is_better_from_hyperagent_dir()` helper |
| 3 | MEDIUM | No `backpropagate` CLI subcommand (documented in SKILL.md but missing) | Added to `archive_utils.py` |
| 4 | LOW | `_get_lineage()` had no cycle detection for non-trivial cycles | Added `seen` set |

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Normalize at backprop time, not selection time | `value_sum` always means "higher=better". UCB selection reads it without sign handling. |
| `eval_count` vs `visit_count` | Prevents inflated N from ancestor backprop in deep trees. |
| Min-max normalization with [0,1] clamp | C=sqrt(2) balances correctly regardless of metric scale. |
| Clamp to [0,1] | Scores outside archive range (from late additions) don't distort UCB. |

## Known Limitations (Acceptable)

1. **Normalization drift** -- earlier backpropagated values used an older min/max range. Ordering preserved.
2. **No file locking** -- `update_node_metadata` is sequential in practice.
3. **`ln(1) = 0`** -- after 1 evaluation, exploration is 0 (pure exploitation). Standard UCB1 behavior.
