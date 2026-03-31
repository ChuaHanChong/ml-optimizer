---
name: analyze
description: "Analyze ML experiment results after a batch completes. Ranks experiments, computes improvements over baseline, identifies HP correlations, and recommends next action (continue/pivot/stop). Use when: a batch of experiments has completed and results need analysis."
user-invocable: false
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
- `scope_level`: Constraint on changes (`"training"` = HP only, `"architecture"` = HP + research, `"full"` = everything including ShinkaEvolve)

## Step 1: Load and Compare Results

> **Goal check:** Verify that the `primary_metric` and `lower_is_better` you use match the optimization goals. If they don't match, flag as a critical error.

Run the result analyzer:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/result_analyzer.py \
  <project_root>/experiments/results \
  <primary_metric> \
  baseline \
  <lower_is_better>
```

This returns:
- Ranking of all experiments
- Deltas vs baseline
- HP correlations

**Branch-aware analysis:** If experiments span multiple `code_branch` values, the global result analyzer output mixes HP correlations across branches. Before computing correlations in Step 2, filter the JSON output by `code_branch` field and analyze each group separately. Do NOT mix HP correlations across branches — identical HPs on different code branches are independent experiments. If only one branch (or all null/baseline), global analysis is sufficient.

## Step 1.1: Filter Results

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
- Check the `"interactions"` array in the result analyzer output
- If interactions are detected, note them in the batch report — they indicate which HP pairs should be explored together, not independently
- Use interaction data to inform HP tuning recommendations: "If LR × batch_size interaction is strong, suggest testing specific LR+batch combos"

## Step 2.1: Tier-Aware Analysis (if method proposals were used)

When experiments have `method_tier` fields, use them for smarter budget allocation:

### Method Effectiveness Ranking
For each code branch that has `method_default_hp` results:
1. Compare its `method_default_hp` metric to the baseline metric
2. Compute the **isolated method effect** = `(method_default_hp_metric - baseline_metric) / baseline_metric × 100%`
3. Rank branches by isolated method effect

### Branch Pruning Recommendations
Use your judgment based on the magnitude and consistency of results — no fixed percentage thresholds:
- **Prune:** If a method with default HPs performs substantially worse than baseline and the deficit is consistent across configs, recommend pruning — HP tuning is unlikely to recover it
- **Promising:** If a method clearly improves over baseline, flag as high-priority for HP tuning
- **Neutral:** If the method's effect is marginal or inconsistent, keep but deprioritize

### Inform the Decision (Step 3)
- If promising branches exist but haven't been HP-tuned yet (`method_tuned_hp` results don't exist for them), recommend **continue** with direction: "tune HPs on promising method branches"
- If all branches have been pruned (all methods hurt), recommend **pivot** or **stop**
- Include the method effectiveness ranking in the batch analysis report (Step 4)

If no experiments have `method_tier` fields, skip this step entirely.

**Fallback for missing per-branch baseline:** If a `method_default_hp` experiment exists but no per-branch baseline result is available, use the global baseline metric (from `baseline.json`) as the comparison point for computing the isolated method effect.

## Step 2.2: Statistical Confidence Assessment

Assess the statistical confidence of your findings based on available data. Use your judgment — more data means higher confidence, but don't refuse to analyze with fewer experiments. Cohen's d effect sizes (negligible/small/medium/large) and confidence labels (high/medium/low/preliminary) are guidelines, not rigid thresholds.

### Method Attribution (when method_tier data exists)
When method proposals were tested, explicitly attribute improvements:
- method_default_hp > baseline BUT method_tuned_hp ≈ method_default_hp → **method drove improvement**, HP tuning added little
- method_default_hp ≈ baseline BUT method_tuned_hp >> baseline → **requires tuned HPs to work** (interaction effect)
- Both improved → **compound effect** (method + HP tuning synergize)

Include effect sizes and confidence labels in the batch analysis report (Step 4).

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
**When:** You judge that the current approach has plateaued. Use your analysis of trends, effect sizes, confidence levels, and improvement trajectories to decide — **no hardcoded thresholds**.

**Pivot Decision Tree** — evaluate conditions in order, respecting `scope_level`. Use your judgment on what "plateaued", "declining", or "stalled" means based on the evidence:

1. **Branch coverage:**
   - Untested branches exist → `branch_test`
   - Tested branches with insufficient configs → `hp_expand`
2. **Code-level optimization** _(skip if `scope_level == "training"`)_:
   All code-level pivots emit `code_evolution`. The hyperagent decides which operator to use.
   - Trigger: ANY of these conditions → pivot_type: `"code_evolution"`:
     - HP tuning shows diminishing returns (you judge from trend analysis, not a fixed %)
     - HP correlations are weak (improvement is not explained by HP changes)
     - No research done yet and HP exploration is flattening
     - All current approaches are stalling → additionally include `"meta_improvement_recommended": true`
3. **Method stacking** _(skip if `scope_level == "training"` or non-git project)_:
   Multiple methods from different papers or significant code changes have improved independently → pivot_type: `"method_stacking"`. Combining them could yield compound gains. No fixed method count — you judge from the archive whether stacking is worth trying.
4. **Failure pattern:**
   - High divergence rate → `narrow_space`
   - Results clustering tightly (no variance) → `code_evolution` (need qualitative change)
   - Overfitting detected → `regularization`
5. **Default:** `continue` — keep exploring. The loop runs autonomously until the goal is reached or the user manually stops.

Output:
```json
{
  "action": "pivot",
  "reason": "<your evidence-based justification>",
  "pivot_type": "<branch_test|hp_expand|narrow_space|regularization|code_evolution|method_stacking>",
  "meta_improvement_recommended": false,
  "suggestion": "<specific actionable next step>",
  "remaining_potential": "<your assessment of room for improvement>"
}
```

**Role split — analysis advises, hyperagent decides:**

You (the analysis agent) evaluate evidence and advise a DIRECTION. The hyperagent reads your advice and decides the specific ACTION.

| You advise (pivot_type) | Hyperagent decides |
|---|---|
| `branch_test`, `hp_expand`, `narrow_space`, `regularization` | Delegates to tuning-agent with adjusted search space |
| `code_evolution` | Which operator: LLM patch, ShinkaEvolve, or research-implement |
| `code_evolution` + `meta_improvement_recommended` | Whether to meta-improve (modify skill instructions) |
| `method_stacking` | Whether to stack, which methods, in what order |

### Stop
**When:** Target metric achieved. This is the ONLY automatic stop condition. The loop is autonomous — it runs until the goal is reached or the user manually stops. Even if progress is slow, keep trying different approaches. Never recommend stop just because improvement is small — breakthroughs can come after plateaus.

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

## Step 3.1: Log Inefficiency Observations

After each analysis, log notable inefficiencies to the error tracker:

### If all experiments in batch diverged or failed:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"warning","source":"analyze","message":"All <N> experiments in batch <batch> diverged/failed — wasted budget","phase":7,"iteration":<batch_number>,"context":{"experiments_wasted":<N>}}'
```

### If observing diminishing returns (log for context, but do NOT recommend stop):
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"analyze","message":"Diminishing returns observed: last <N> batches showed <X%> improvement — recommend pivot to different operator","phase":7,"context":{"total_experiments":<N>}}'
```

### If a code branch consistently underperforms baseline:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"analyze","message":"Branch <branch> underperforms baseline across all HP configs","code_branch":"<branch>","context":{"experiments_on_branch":<N>,"best_vs_baseline":"<delta%>"}}'
```

## Step 3.2: Log Dead Ends

When a technique is conclusively unpromising, log it to the dead-end catalog so it's never re-proposed:

**When to log a dead end:**
- A code branch is pruned (substantially worse than baseline across all HP configs)
- All experiments in a batch diverge or fail after recovery attempt
- Analyze recommends stop and a specific method showed no improvement after tuning

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> dead-end add '{"technique":"<technique_name>","reason":"<why it failed>","branch":"<ml-opt/branch or null>","experiments_tried":<N>,"best_result":{"metric":"<primary_metric>","value":<best_value>,"baseline":<baseline_value>},"source":"analyze"}'
```

Do not log dead ends for techniques that showed mixed results (some improvement, some regression) — only for those conclusively worse.

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

## Step 4.1: Update Research Agenda

If a research agenda exists (`experiments/reports/research-agenda.json`), update it based on this batch's results:

```bash
# Check if agenda exists
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> agenda list
```

For each code branch tested in this batch, find the corresponding agenda item and update it:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> agenda update '<idea_id>' '{"status":"<tried|improved|dead-end>","priority":<new_priority>,"evidence":{"batch":<N>,"result":"<summary of results vs baseline>"},"lessons":"<what was learned>"}'
```

**Priority adjustment rules:**
- If improved over baseline: increase priority by 1-2 points, set `status: "improved"`
- If mixed results (some configs better, some worse): keep priority, set `status: "tried"`, add evidence
- If conclusively worse (substantially below baseline across all configs, in your judgment): decrease priority to 1, set `status: "dead-end"`, also log to dead-end catalog (Step 3.2)

**Add evidence-suggested ideas:** If the analysis reveals new optimization directions (e.g., LR sensitivity is very high → try cyclical LR), add them:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> agenda add '{"id":"evidence-<N>","name":"<new idea>","priority":<score>,"source":"experimental_evidence","scope":"training"}'
```

Skip this step if no agenda file exists (e.g., HP-only optimization without research phase).

## Step 4.2: Time-Budget-Normalized Analysis

If experiments have `time_budget_seconds` set (all experiments ran for the same wall-clock duration):
- **Direct comparison is valid** — all experiments had equal compute time, so metric deltas reflect true efficiency differences
- No duration normalization needed
- **Convergence check:** If many experiments show improving trends at cutoff (metric still decreasing/increasing), recommend a longer time budget in the next iteration
- Note in the batch report: "All experiments ran with {time_budget}s fixed time budget — metrics are directly comparable"

Skip this step if `time_budget_seconds` is not present in experiment results.

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

### Branch Allocation Data (for hp-tune)

When multiple code branches are being tested, include in the analysis output:
- `branch_scores`: Per-branch allocation scores from `${CLAUDE_PLUGIN_ROOT}/scripts/result_analyzer.py` (run with `branch-scores` subcommand or use `compute_branch_scores()`)
- This data is passed to hp-tune for adaptive budget allocation in the next iteration

### Stacking Readiness

Include in the analysis output:
- `methods_with_improvement`: Count of unique code_branches whose best result beats baseline.
  Compute using `rank_methods_for_stacking()` from `${CLAUDE_PLUGIN_ROOT}/scripts/result_analyzer.py`.
- `stacking_candidates`: List of method names (code_proposal values) that improved, ranked by improvement magnitude.

---

## Session Review Mode

When dispatched with `scope: "session"`, switch from batch analysis to session review mode. Instead of analyzing a single batch, review the entire optimization session to generate self-improvement recommendations. This is advisory only — present insights and recommendations, do NOT auto-apply changes.

### Review Inputs

From the orchestrator:
- `project_root`: Project root directory
- `exp_root`: Path to experiments/ directory (default: `<project_root>/experiments`)
- `primary_metric`: The metric that was optimized
- `lower_is_better`: Whether lower values are better for the primary metric
- `scope`: `"session"`

### Review Step 1: Load Error and Experiment Data

1. Run session summary and pattern detection:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> summary
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> patterns
```

2. Read supporting files:
   - `<exp_root>/reports/error-log.json` — full error event list
   - `<exp_root>/reports/batch-*-analysis.md` — batch analysis reports
   - `<exp_root>/dev_notes.md` — session narrative

3. Read experiment data:
   - `<exp_root>/results/baseline.json` — baseline metrics
   - `<exp_root>/results/exp-*.json` — all experiment results
   - `<exp_root>/results/implementation-manifest.json` — which proposals were implemented
   - `<exp_root>/reports/research-findings*.md` — what techniques were researched
   - `<exp_root>/optimization-goals.json` — what the user wanted to achieve
   - `<exp_root>/learned-behaviors.json` — what was learned during the session

### Review Step 2: Compute Success and Proposal Metrics

1. Compute success metrics:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> success <primary_metric> <lower_is_better>
```
Returns success rate, improvement rate, best improvement, duration analysis, time wasted on failures.

2. Compute proposal outcomes:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> proposals <primary_metric> <lower_is_better>
```
Returns research proposal outcomes, HP proposal stats, implementation stats.

3. Load suggestion history to detect repeats:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> suggestion-history
```

4. Query scope violations from behavioral memory:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> query-behaviors scope_violation
```

### Review Step 3: Rank Patterns and Generate Recommendations

1. Rank all detected patterns by impact score:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> rank <total_experiments>
```
Where `<total_experiments>` comes from the success metrics output. Use this ranking to order suggestions — highest score first.

2. For each pattern (in rank order), generate a specific, actionable recommendation:

```markdown
### Recommendation: [Concise Title]
- **Severity:** [Critical / Warning / Info]
- **Evidence:** [Which error events, experiments, or patterns support this]
- **Problem:** [What went wrong, with specifics from the error log]
- **Recommendation:** [What to do differently next time]
- **Expected impact:** [What would improve]
- **Confidence:** [High / Medium / Low — based on evidence strength and sample size]
```

### Review Step 4: Write session-review.md

Write the review to `<exp_root>/reports/session-review.md` containing:
- Executive summary (total experiments, success/failure/diverge counts, success rate, improvement rate, error events, patterns detected)
- What worked (top performing configurations, effective patterns, efficiency highlights)
- Proposal outcomes (research proposals table, implementation stats, HP success patterns)
- Error timeline
- Detected patterns with occurrences and affected experiments
- All improvement suggestions in the recommendation format above

After writing, log each suggestion:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log-suggestion <pattern_id> <scope>
```

### Review Rules

1. **Advisory only** — present recommendations, do NOT auto-apply changes
2. **Cite specific evidence** — every recommendation must reference concrete files (e.g., `experiments/results/exp-003.json`), exact values (e.g., "diverged at lr=0.05, NaN at step 42"), and specific `event_id` values from the error log
3. **Distinguish patterns from one-offs** — a single divergence is an event; three divergences at similar LRs is a pattern worth recommending against
4. **Note confidence based on sample size** — 2 experiments is Low confidence, 5+ is Medium, 10+ with consistent results is High
5. **Focus on strategy** — recommend HP ranges, methods to try/avoid, budget allocation, scope changes
6. **Check for repeats** — if a pattern was previously flagged (from suggestion history), note "Previously flagged" and assess whether the recommendation is still relevant
7. **Present only the top 3** most impactful recommendations to the user

### Review Error Handling

- **No error log exists:** Report "No errors tracked in this session." Still run success metrics and proposal outcomes if experiment results exist.
- **Empty error log:** Report "0 events tracked." Still analyze experiment outcomes for success patterns.
- **No experiment results:** Report "No experiments found." Only analyze error events.
- **Corrupt JSON files:** Skip the corrupt file, note it as a warning in the review.
- **Missing primary_metric or lower_is_better:** Skip success metrics and proposal outcomes, note in the review that these inputs are needed for full analysis.
