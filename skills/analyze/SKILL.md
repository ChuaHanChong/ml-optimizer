---
name: analyze
description: "Analyze ML experiment results after a batch completes. Ranks experiments, computes improvements over baseline, identifies HP correlations, and recommends next action (continue/pivot/stop). Use when: a batch of experiments has completed and results need analysis."
---

# Experiment Analysis

Use extended thinking for all analytical reasoning in this skill. Ultrathink. Think through HP interaction effects, trend detection across batches, and the continue/pivot/stop decision with full consideration of alternatives and edge cases.

Analyze completed experiment results to determine what worked, what didn't, and what to do next.

## Inputs Expected

From the orchestrator:
- `project_root`: Project root directory
- `batch_number`: Which batch of experiments this is (1, 2, 3, ...)
- `primary_metric`: The metric to optimize
- `lower_is_better`: Whether lower values are better (True for loss, False for PSNR/accuracy)
- `target_value`: The goal value for the primary metric (optional)

## Step 1: Load and Compare Results

Run the result analyzer:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/result_analyzer.py \
  <project_root>/experiments/results \
  <primary_metric> \
  baseline \
  <lower_is_better>
```

This returns:
- Ranking of all experiments
- Deltas vs baseline
- HP correlations

## Step 1.5: Filter Results

Before analysis, filter out non-completed experiments:
- Exclude experiments with `status: "diverged"` or `status: "failed"` from correlation analysis
- Include diverged/failed experiments in the failure analysis section (they provide boundary information)
- Note: `rank_by_metric()` includes all experiments (with a `status` field for filtering); only `identify_correlations()` auto-filters to completed experiments
- **Branch-aware grouping:** When computing HP correlations (Step 2), group results by `code_branch` field. Experiments on different code branches should be analyzed separately — identical HPs may behave differently on different code. If `code_branch` is null or missing, treat as baseline code group.

## Step 2: Deep Analysis

Beyond what the script provides, reason about:

### Performance Trends
- Is there a clear trend in the results? (e.g., lower LR consistently better)
- Are improvements accelerating or decelerating?
- How does this batch compare to previous batches?

### Failure Analysis
- Which experiments diverged or failed?
- What do failed experiments have in common?
- Are there boundary conditions being hit? (OOM, NaN at high LR)

### HP Impact Assessment
For each hyperparameter that was varied, use **relative** (percentage) thresholds to classify impact. These thresholds are relative to the baseline value, making them meaningful across different metric scales:
- **High impact:** Changing this HP caused >5% relative metric change vs baseline
- **Medium impact:** 1-5% relative metric change vs baseline
- **Low impact:** <1% relative metric change vs baseline
- **Unknown:** Not enough variation to determine

### Interaction Effects
- Did changing two HPs together produce unexpected results?
- e.g., High LR + large batch diverged, but high LR + small batch worked

## Step 3: Decide Next Action

Based on analysis, recommend ONE of:

### Continue Tuning
**When:** Clear direction for improvement exists
- Improvement trend is positive
- Unexplored regions of the search space remain
- Not yet at diminishing returns

Output:
```json
{
  "action": "continue",
  "reason": "<specific justification>",
  "direction": "<what to focus on next>",
  "suggested_changes": ["<HP1 should be lower>", "<try different scheduler>"]
}
```

### Try Different Approach (Pivot)
**When:** Current HP tuning has plateaued but goal not reached

**Pivot Decision Tree** — evaluate conditions in order:

1. **Budget check:** If `remaining_budget < 3`, do NOT pivot to research. Recommend stop with current best.
2. **Branch coverage:**
   - Untested branches exist → "Test untested code branches with baseline HPs"
   - All branches tested but some with only 1-2 configs → "Increase HP exploration on promising branches (within 5% of best)"
3. **Research status:**
   - No research done AND `remaining_budget >= 5` → "Switch to research + code changes — HP tuning alone has plateaued"
   - Research done but not all proposals implemented AND `remaining_budget >= 3` → "Implement next-priority research proposal"
4. **Failure pattern:**
   - >50% of recent experiments diverged → "Narrow search space — constrain LR to [best_lr × 0.5, best_lr × 2.0]"
   - All experiments within 1% of each other → "Try qualitatively different change (different optimizer, scheduler, data augmentation)"
   - Overfitting detected (train improving, val flat/degrading) → "Add regularization (weight_decay, dropout, data augmentation)"
5. **Default:** "Continue with hybrid exploration/exploitation batch" with diminishing-returns warning.

Output:
```json
{
  "action": "pivot",
  "reason": "<which condition from the decision tree triggered>",
  "pivot_type": "<branch_test|hp_expand|research|narrow_space|qualitative_change|regularization>",
  "suggestion": "<specific actionable next step>",
  "remaining_potential": "<estimated room for improvement>",
  "budget_remaining": "<N>"
}
```

### Stop
**When:** Goal reached OR no more improvement possible
- Target metric value achieved
- Exhaustive search completed with diminishing returns
- All reasonable approaches tried

Output:
```json
{
  "action": "stop",
  "reason": "<why we should stop>",
  "best_exp_id": "<best experiment>",
  "best_metric_value": <value>,
  "improvement_over_baseline": "<X%>"
}
```

## Step 3.5: Log Inefficiency Observations

After each analysis, log notable inefficiencies to the error tracker:

### If all experiments in batch diverged or failed:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"warning","source":"analyze","message":"All <N> experiments in batch <batch> diverged/failed — wasted budget","phase":5,"iteration":<batch_number>,"context":{"experiments_wasted":<N>}}'
```

### If recommending stop due to diminishing returns:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"analyze","message":"Diminishing returns: last <N> batches showed <X%> improvement","phase":5,"context":{"total_experiments":<N>}}'
```

### If a code branch consistently underperforms baseline:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"analyze","message":"Branch <branch> underperforms baseline across all HP configs","code_branch":"<branch>","context":{"experiments_on_branch":<N>,"best_vs_baseline":"<delta%>"}}'
```

## Step 4: Write Batch Analysis Report

Write to `experiments/reports/batch-<N>-analysis.md`:

```markdown
# Batch <N> Analysis

## Summary
- Experiments run: <count>
- Experiments completed: <count>
- Experiments diverged: <count>
- Best in batch: <exp_id> (<metric>=<value>)
- Best overall: <exp_id> (<metric>=<value>)

## Results Table
| Exp ID | Status | LR | Batch Size | Other Changes | <Metric> | vs Baseline |
|--------|--------|----|------------|---------------|----------|-------------|
| ... | ... | ... | ... | ... | ... | ... |

## HP Impact Analysis
- **Learning rate:** [impact level] — [observation]
- **Batch size:** [impact level] — [observation]
- **Weight decay:** [impact level] — [observation]
- **Other:** [observations]

## Key Findings
1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

## Recommendation
**Action:** [continue/pivot/stop]
**Reason:** [detailed justification]
**Next steps:** [if continuing, what to try]
```

## Step 5: Update Dev Notes

Append to `experiments/dev_notes.md`:
```markdown
## <date> — Batch <N> Analysis

- Best result: <exp_id> with <metric>=<value> (<X%> improvement)
- Recommendation: <action>
- Key insight: <most important finding>
```

## Output

Return to the orchestrator:
- The recommended action (continue/pivot/stop)
- Best experiment ID and metrics
- Improvement over baseline
- Key findings summary
- Path to the analysis report
