---
name: analyze
description: "Analyze ML experiment results after a batch completes. Ranks experiments, computes improvements over baseline, identifies HP correlations, and recommends next action (continue/pivot/stop). Use when: a batch of experiments has completed and results need analysis."
user-invocable: false
---

# Experiment Analysis

Think through HP interaction effects, trend detection across batches, and the continue/pivot/stop decision with full consideration of alternatives and edge cases — determine what worked, what didn't, and what to do next.

> **Path convention:** All `<exp_root>/...` paths refer to the `exp_root` dispatch parameter. The plugin does not hardcode the output directory name.

## Inputs Expected

From the orchestrator:
- `project_root`: project root directory
- `batch_number`: which batch of experiments this is (1, 2, 3, ...)
- `primary_metric`: the metric to optimize
- `lower_is_better`: whether lower values are better (True for loss, False for PSNR/accuracy)
- `target_value`: goal value for the primary metric (optional)
- `scope_level`: constraint on changes (`"training"` = HP only, `"architecture"` = HP + research, `"full"` = everything including ShinkaEvolve)
- `secondary_metrics`: optional extra metrics to track — `[{"name": ..., "lower_is_better": ..., "role": "guardrail"|"report"}]` from Phase 0 user_choices. See Step 2.3.
- `eval_tasks`: optional list of task/environment names from Phase 0 user_choices. When non-empty, each result's `metrics` carries `<primary_metric>_<task>` keys plus the `<primary_metric>` mean and `<primary_metric>_worst` aggregates computed by the experiment agent. Rank on `<primary_metric>` as always; when reporting, add a per-task breakdown column and call out any experiment whose `<primary_metric>_worst` regressed against baseline even though the mean improved — that is a policy specializing, not generalizing.

## Step 1: Load and Compare Results

> **Goal check:** verify the `primary_metric` and `lower_is_better` you use match the optimization goals. If they don't match, flag as a critical error.

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/result_analyzer.py \
  <exp_root>/results \
  <primary_metric> \
  baseline \
  <lower_is_better>
```

This returns: ranking of all experiments, deltas vs baseline, HP correlations.

**Branch-aware analysis:** if experiments span multiple `code_branch` values, the global output mixes HP correlations across branches. Before computing correlations in Step 2, filter the JSON output by `code_branch` and analyze each group separately. Do NOT mix HP correlations across branches — identical HPs on different code branches are independent experiments. If only one branch (or all null/baseline), global analysis is sufficient.

## Step 1.1: Filter Results

Before analysis, filter out non-completed experiments:
- Exclude `status: "diverged"` or `status: "failed"` from correlation analysis
- Include diverged/failed experiments in the failure analysis section (they provide boundary information)
- Note: `rank_by_metric()` includes all experiments (with a `status` field for filtering); `identify_correlations()`, `detect_hp_interactions()`, and `aggregate_replicates()` (used by `compute_branch_scores()`/`rank_methods_for_stacking()`) all auto-filter to completed experiments
- **Branch-aware grouping:** when computing HP correlations (Step 2), group by `code_branch`. Experiments on different branches are analyzed separately — identical HPs may behave differently on different code. If `code_branch` is null or missing, treat as baseline code group.
- **Warm-start segregation:** segregate results whose `checkpoint_source` is set (warm-started — a top-level result field, not nested under `config`) from from-scratch results when declaring the best experiment under a fixed budget — a warm-started run consumed its parent's training on top of its own budget, so it is NOT budget-comparable to a from-scratch run. Report the best of each group separately and label warm-started bests as not budget-fair.
- **Same-protocol comparisons only:** compare `metrics.<primary_metric>` values **only** between experiments sharing the same `eval_protocol`. A held-out-eval number and a train-report number for the same metric can differ by several points on identical code — a cross-protocol delta is invalid. If a batch mixes protocols, normalize to the canonical held-out-eval value (or refuse the delta and flag it) before ranking; never declare an improvement/dead-end/tie across protocols.

## Step 2: Deep Analysis

Beyond what the script provides, reason about:

### Performance Trends
- Is there a clear trend? (e.g., lower LR consistently better)
- Are improvements accelerating or decelerating?
- How does this batch compare to previous batches?

### Failure Analysis
- Which experiments diverged or failed?
- What do failed experiments have in common?
- Are boundary conditions being hit? (OOM, NaN at high LR)

### HP Impact Assessment
For each varied hyperparameter, use **relative** (percentage) thresholds to classify impact — relative to the baseline value, so they're meaningful across metric scales:
- **High impact:** >5% relative metric change vs baseline
- **Medium impact:** 1-5% relative metric change vs baseline
- **Low impact:** <1% relative metric change vs baseline
- **Unknown:** not enough variation to determine

### Interaction Effects
- Check the `interactions` array nested at `result.interactions.interactions` in the result analyzer output (the top-level `interactions` key holds `{interactions: [...], note: ...}`)
- If interactions are detected, note them in the batch report — they indicate which HP pairs to explore together, not independently
- Use interaction data to inform HP tuning: "If LR × batch_size interaction is strong, suggest testing specific LR+batch combos"
- **Low-n gating:** correlation entries carry `n` and `low_n`; interaction entries carry `significant` (exact small-n Spearman critical values, n=5..10). A rho from `low_n` (n < 10) single-seed runs is preliminary evidence at best — NEVER base a pivot or an hp-tune directive on a `low_n` or non-`significant` interaction alone; wait for more experiments or corroborating evidence.
- **Zero-centered metrics:** when delta entries carry `delta_vs_spread` (baseline near zero relative to the batch spread), percent-of-baseline is meaningless — reason in absolute deltas against `batch_std` instead.

## Step 2.1: Tier-Aware Analysis (if method proposals were used)

When experiments have `method_tier` fields, use them for smarter budget allocation:

### Method Effectiveness Ranking
For each code branch with `method_default_hp` results:
1. Compare its `method_default_hp` metric to the baseline metric
2. Compute the **isolated method effect** = `(method_default_hp_metric - baseline_metric) / baseline_metric × 100%`
3. Rank branches by isolated method effect

### Branch Pruning Recommendations
Use judgment based on magnitude and consistency — no fixed percentage thresholds:
- **Prune:** if a method with default HPs performs substantially worse than baseline and the deficit is consistent across configs, recommend pruning — HP tuning is unlikely to recover it
- **Promising:** if a method clearly improves over baseline, flag as high-priority for HP tuning
- **Neutral:** if the effect is marginal or inconsistent, keep but deprioritize

### Inform the Decision (Step 3)
- If promising branches exist but aren't HP-tuned yet (no `method_tuned_hp` results), recommend **continue** with direction: "tune HPs on promising method branches"
- If all branches are pruned (all methods hurt), recommend **pivot** or **stop**
- Include the method effectiveness ranking in the batch analysis report (Step 4)

If no experiments have `method_tier` fields, skip this step entirely.

**Fallback for missing per-branch baseline:** if a `method_default_hp` experiment exists but no per-branch baseline result is available, use the global baseline metric (from `baseline.json`) as the comparison point for the isolated method effect.

## Step 2.2: Statistical Confidence Assessment

Assess the statistical confidence of your findings from available data. Use judgment — more data means higher confidence, but don't refuse to analyze with fewer experiments. Cohen's d effect sizes (negligible/small/medium/large) and confidence labels (high/medium/low/preliminary) are guidelines, not rigid thresholds.

### Method Attribution (when method_tier data exists)
When method proposals were tested, explicitly attribute improvements:
- method_default_hp > baseline BUT method_tuned_hp ≈ method_default_hp → **method drove improvement**, HP tuning added little
- method_default_hp ≈ baseline BUT method_tuned_hp >> baseline → **requires tuned HPs to work** (interaction effect)
- Both improved → **compound effect** (method + HP tuning synergize)

Include effect sizes and confidence labels in the batch analysis report (Step 4).

### Seed-Replicate Noise Floor (when `random_seed` replicates exist)

When results contain seed replicates (identical config except `random_seed`), `result_analyzer.py` aggregates them to mean±std (`replicates: {n, mean, std}` on branch scores and stacking ranks). Treat the observed replicate std as the **measured noise floor**: a difference between experiments smaller than that spread is within noise — do NOT prune a branch, declare a dead end, or call a plateau on a difference inside the noise floor. Rank by replicate mean, never by the best single seed.

## Step 2.3: Secondary Metrics & Guardrails (if `secondary_metrics` provided)

When the dispatch includes `secondary_metrics`, each entry is `{name, lower_is_better, role}` with role `"guardrail"` or `"report"`:

1. **Read the extra keys** from each result's existing `metrics` dict — already parsed; do NOT re-parse logs. A missing key renders as `—` and is never treated as a regression.
2. **Report per batch:** add one column per secondary metric to the batch report's Results Table (Step 4) and note each metric's direction.
3. **Guardrail regression flag:** BEFORE declaring a best experiment or nominating stacking candidates, compare each top-ranked experiment's guardrail metrics against baseline using that metric's own `lower_is_better`. If a guardrail regressed, mark the experiment `guardrail_regressed` in the batch report and do NOT declare it best or include it in `stacking_candidates` without an explicit note stating the trade-off. `report`-role metrics never block.

Skip this step when `secondary_metrics` is absent or empty.

## Step 3: Decide Next Action

Based on analysis, recommend ONE of:

### Continue Tuning
**When:** clear direction for improvement exists
- Improvement trend is positive
- Unexplored search-space regions remain
- Not yet at diminishing returns

Return `decision: "continue"` with `batch_number` and `reason` — see the canonical Output schema at the end of this section (every decision shares one shape; unused fields are `null`/omitted). Note any specific direction (e.g. "HP1 should be lower; try a different scheduler") in `reason`.

### Try Different Approach (Pivot)
**When:** you judge the current approach has plateaued. Use your analysis of trends, effect sizes, confidence levels, and improvement trajectories — **no hardcoded thresholds**.

**Two fields, two roles.** Your structured output includes both `decision` and `pivot_type`:
- **`decision`** — the field the phase-7 workflow SCRIPT reads directly to route the next step (required by the schema, along with `batch_number`). One of: `continue | branch_test | hp_expand | narrow_space | regularization | method_proposal | code_evolution | stop`.
- **`pivot_type`** — a secondary, advisory classification field (optional in the schema — but populate it whenever `decision` is a pivot, so logs/reports and the pivot-gate skeptic check can classify it). The script special-cases exactly ONE `pivot_type` value: if `pivot_type == "qualitative_change"` AND `decision` is not already `stop`/`code_evolution`/`method_proposal`, the workflow reroutes `decision` to `"method_proposal"` for you. Every other `pivot_type` value is informational only — it does not drive dispatch by itself; `decision` does. Valid values: `branch_test | hp_expand | narrow_space | regularization | code_evolution | method_proposal | qualitative_change | method_stacking` (the last two never appear as a `decision` value — see points 3 and 4 below).

**Decision Tree** — evaluate conditions in order, respecting `scope_level`. Use judgment on what "plateaued", "declining", or "stalled" means from the evidence:

1. **Branch coverage:**
   - Untested branches exist → `decision: "branch_test"` (set `untested_branches`)
   - Tested branches with insufficient configs → `decision: "hp_expand"` (return widened `search_space`)
2. **Code-level optimization** _(skip if `scope_level == "training"`)_ — a choice between TWO distinct decisions; you must pick which applies, the workflow does not disambiguate a single shared signal into these two routes for you:
   - No research has been done yet, or HP exploration is flattening and a genuinely new method is needed → `decision: "method_proposal"` (only if `scope_level != "training"`) — the workflow dispatches research-agent then implement-agent.
   - HP tuning shows diminishing returns AND HP correlations are weak (the method's own code — not its HPs — is what's driving results, roughly |rho|<0.3) → `decision: "code_evolution"` (only if `scope_level == "full"`) — the workflow dispatches tuning-agent (evolve HPs) → implement-agent with the evolve skill (ShinkaEvolve) directly.
3. **Method stacking** _(advisory only — not a `decision` value; skip if `scope_level == "training"` or non-git project)_:
   Multiple methods from different papers or significant code changes improved independently — combining them could yield compound gains. This has NO corresponding `decision` value: the phase-7 script's dispatch field has no `method_stacking` entry at all. Instead, report every qualifying branch in `stacking_candidates` (see Stacking Readiness, Output below) on every batch where it applies — the workflow accumulates these continuously across the whole run (plus its own independent baseline-vs-branch harvest each batch) and returns the accumulated, dead-end-filtered list when the loop exits. The orchestrator launches Phase 8 automatically iff that returned list is non-empty — independent of whichever `decision`/`pivot_type` happened to fire on any one batch. You may still set `pivot_type: "method_stacking"` as a note the first batch you notice it — it has no dispatch effect. No fixed method count — judge from the archive whether stacking looks worth flagging.
4. **Failure pattern:**
   - High divergence rate → `decision: "narrow_space"`
   - Results clustering tightly (no variance — a qualitative change is needed) → `decision: "code_evolution"` (or `"method_proposal"` if no research has run yet); you may also set `pivot_type: "qualitative_change"` here as a note. (If you set `pivot_type: "qualitative_change"` but leave `decision` at something non-costly — e.g. `continue` — the workflow reroutes `decision` to `"method_proposal"` for you per the special case above; prefer setting `decision` explicitly yourself rather than relying on this fallback.)
   - Overfitting detected → `decision: "regularization"` (weight decay/dropout; for `model_category=rl`: entropy coefficient / KL penalty)
5. **Default:** `decision: "continue"` — keep exploring. The loop runs autonomously until the goal is reached or the user manually stops.

**Output** (canonical shape for Phase-7 batch-mode analysis — every Phase-7 batch returns this; `decision` and `batch_number` are schema-required, everything else is `null`/omitted when not relevant to your decision. Phase 8's stacking-assessment dispatch of this same agent uses a separate, smaller `recommendation` contract — `continue|code_evolution|stop`, `STACK_ANALYSIS_SCHEMA` in phase-8-stacking.js — with no `decision`/`batch_number` fields):
```json
{
  "decision": "<continue|branch_test|hp_expand|narrow_space|regularization|method_proposal|code_evolution|stop>",
  "pivot_type": "<branch_test|hp_expand|narrow_space|regularization|code_evolution|method_proposal|qualitative_change|method_stacking|null>",
  "batch_number": <N>,
  "best_exp_id": "<best experiment so far, or null>",
  "best_metric_value": <value, or null>,
  "improved_since_last_stop": <bool, or null>,
  "search_space": { "...": "widened/narrowed space for hp_expand/narrow_space/regularization, or null" },
  "branch_scores": { "...": "per-branch allocation scores, or null" },
  "correlations": { "...": "HP correlation data, or null" },
  "untested_branches": ["<branch>", "..."],
  "surviving_branches": ["<branch>", "..."],
  "max_batch_size": <int, or null>,
  "reason": "<your evidence-based justification>",
  "analysis_path": "<exp_root>/reports/batch-<N>-analysis.md",
  "stacking_candidates": [{ "branch": "<ml-opt/slug>", "improvement_pct": <value> }]
}
```

**Role split — you decide, the workflow script dispatches:**

You (the analysis agent) evaluate evidence and return `decision`. The phase-7 workflow script reads `decision` directly and dispatches accordingly:

| `decision` | Workflow action |
|---|---|
| `continue` | Keeps tuning the same search space |
| `branch_test` | Merges `untested_branches` into `code_branches`, dispatches tuning-agent (does not touch `search_space`) |
| `hp_expand`, `narrow_space`, `regularization` | Adjusts `search_space`, dispatches tuning-agent |
| `method_proposal` | Dispatches research-agent, then implement-agent — new methods, new branches |
| `code_evolution` | Dispatches tuning-agent (evolve HPs) → implement-agent with the evolve skill (ShinkaEvolve) → experiment |
| `stop` | Runs the stuck protocol (research-agent for fresh ideas) when `method_proposal_scope` is set AND `scope_level != "training"` (`methodProposalsEnabled`); otherwise (including when `method_proposal_scope` is null) checks only for metric improvement + fixpoint exit judgment |

`pivot_type: "method_stacking"` has no row above — it never drives a `decision` (see point 3). It just means: stacking_candidates recorded — the orchestrator enters Phase 8 automatically if any exist when the phase-7 loop exits, not because this pivot_type fired on a particular batch.

### Stop
**When:** target metric **robustly** achieved, OR approaches genuinely exhausted (no clear direction remains after exploring branches/HPs — this routes through the workflow's stuck protocol + fixpoint check rather than an immediate exit, it does not exit the loop by itself). The loop is autonomous — it runs until the goal is reached or the user manually stops. Even if progress is slow, keep trying different approaches. Never recommend stop just because improvement is small — breakthroughs can come after plateaus.

**Do NOT stop on a within-noise, single-seed target-hit.** A "target achieved" call is only valid on evidence that survives the measured noise floor (Step 2.3):
- If the best experiment's margin over target is **≥ the measured seed-noise floor**, and it is a real result on the canonical `eval_protocol` → stop is legitimate.
- If the margin is **within the noise floor**, or the best result is a single unreplicated seed, or it changed >1 variable vs the prior best → do **NOT** stop. Recommend `continue` with a cheap seed-confirmation batch (≥3 seeds on one protocol, champion config held fixed) and a stop-rule of `seed_mean ≥ target`. Gate the stop on `best_seed_mean ≥ target`, never `best_single_seed ≥ target`.
- **Exit block:** if any `untried` item in `research-agenda.json` outranks (priority) the best-executed idea, exit-to-Phase-9 is blocked — recommend `continue` and run that item first. Never leave the single highest-priority in-scope idea untried at exit.

Include in the stop/continue output: `"noise_floor"`, `"best_seed_mean"` (or null if unreplicated), and `"margin_over_target"`. These are informational/audit-trail fields for human review (e.g. in `batch-<N>-analysis.md` and the final report) — the phase-7 workflow's `ANALYSIS_DECISION_SCHEMA` has no such properties and nothing downstream currently reads them programmatically; they don't feed the workflow's fixpoint judgment.

Output:
```json
{
  "decision": "stop",
  "batch_number": <N>,
  "reason": "<why we should stop>",
  "best_exp_id": "<best experiment>",
  "best_metric_value": <value>,
  "improvement_over_baseline": "<X%>",
  "noise_floor": <value, or null>,
  "best_seed_mean": <value, or null>,
  "margin_over_target": <value, or null>
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

Do not log dead ends for techniques with mixed results (some improvement, some regression) — only conclusively worse ones.

## Step 4: Write Batch Analysis Report

Write to `<exp_root>/reports/batch-<N>-analysis.md`:

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

(When `secondary_metrics` is configured, append one column per secondary metric and mark `guardrail_regressed` rows — Step 2.3.)

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

If a research agenda exists (`<exp_root>/reports/research-agenda.json`), update it based on this batch's results:

```bash
# Check if agenda exists
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> agenda list
```

For each code branch tested in this batch, find the corresponding agenda item and update it:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> agenda update '<idea_id>' '{"status":"<tried|improved|dead-end>","priority":<new_priority>,"evidence":{"batch":<N>,"result":"<summary of results vs baseline>"},"lessons":"<what was learned>"}'
```

**Priority adjustment rules:**
- Improved over baseline: increase priority by 1-2 points, set `status: "improved"`
- Mixed results (some configs better, some worse): keep priority, set `status: "tried"`, add evidence
- Conclusively worse (substantially below baseline across all configs, in your judgment): decrease priority to 1, set `status: "dead-end"`, also log to dead-end catalog (Step 3.2)

**Add evidence-suggested ideas:** if analysis reveals new directions (e.g., LR sensitivity very high → try cyclical LR), add them:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> agenda add '{"id":"evidence-<N>","name":"<new idea>","priority":<score>,"source":"experimental_evidence","scope":"training"}'
```

Skip this step if no agenda file exists (e.g., HP-only optimization without research phase).

## Step 4.2: Time-Budget-Normalized Analysis

If experiments have `time_budget_seconds` set (all ran for the same wall-clock duration):
- **Direct comparison is valid** — equal compute time, so metric deltas reflect true efficiency differences
- No duration normalization needed
- **Convergence check:** if many experiments show improving trends at cutoff (metric still decreasing/increasing), recommend a longer time budget next iteration
- Note in the batch report: "All experiments ran with {time_budget}s fixed time budget — metrics are directly comparable"

Skip this step if `time_budget_seconds` is not present in experiment results.

## Step 5: Update Dev Notes

Append to `<exp_root>/dev_notes.md`:
```markdown
## <date> — Batch <N> Analysis

- Best result: <exp_id> with <metric>=<value> (<X%> improvement)
- Recommendation: <action>
- Key insight: <most important finding>
```

## Output

Return to the orchestrator:
- The `decision` value (continue/branch_test/hp_expand/narrow_space/regularization/method_proposal/code_evolution/stop) and, when relevant, the `pivot_type` classification
- Best experiment ID and metrics
- Improvement over baseline
- Key findings summary
- Path to the analysis report

### Branch Allocation Data (for hp-tune)

When multiple code branches are being tested, include in the analysis output:
- `branch_scores`: per-branch allocation scores — already part of the `result_analyzer.py <results_dir> <metric> ...` output you ran in Step 1 (computed internally via `compute_branch_scores()`); there is no separate `branch-scores` CLI subcommand
- Passed to hp-tune for adaptive budget allocation in the next iteration

### Stacking Readiness

Include in the analysis output:
- `methods_with_improvement`: count of unique code_branches whose best result beats baseline. Compute using `rank_methods_for_stacking()` from `${CLAUDE_PLUGIN_ROOT}/scripts/result_analyzer.py`. Informational only — not in `ANALYSIS_DECISION_SCHEMA`, nothing downstream reads it.
- `stacking_candidates`: array of `{branch, improvement_pct}` objects — one per code branch whose best result beats baseline, ranked by improvement magnitude (`branch` is the `ml-opt/<slug>` git branch, not a bare method name).

---

## Session Review Mode

When dispatched with `scope: "session"`, switch from batch analysis to session review mode. Instead of a single batch, review the entire optimization session to generate self-improvement recommendations. This is advisory only — present insights and recommendations, do NOT auto-apply changes.

### Review Inputs

From the orchestrator:
- `project_root`: project root directory
- `exp_root`: path to <exp_root>/ directory (passed explicitly by the orchestrator — no hardcoded default)
- `primary_metric`: the metric that was optimized
- `lower_is_better`: whether lower values are better for the primary metric
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
   - `<exp_root>/results/round-*/exp-*.json` — all experiment results
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
2. **Cite specific evidence** — every recommendation must reference concrete files (e.g., `<exp_root>/results/exp-003.json`), exact values (e.g., "diverged at lr=0.05, NaN at step 42"), and specific `event_id` values from the error log
3. **Distinguish patterns from one-offs** — a single divergence is an event; three divergences at similar LRs is a pattern worth recommending against
4. **Note confidence based on sample size** — 2 experiments is Low confidence, 5+ is Medium, 10+ with consistent results is High
5. **Focus on strategy** — recommend HP ranges, methods to try/avoid, budget allocation, scope changes
6. **Check for repeats** — if a pattern was previously flagged (from suggestion history), note "Previously flagged" and assess whether the recommendation is still relevant
7. **Present only the top 3** most impactful recommendations to the user

### Review Error Handling

- **No error log exists:** report "No errors tracked in this session." Still run success metrics and proposal outcomes if experiment results exist.
- **Empty error log:** report "0 events tracked." Still analyze experiment outcomes for success patterns.
- **No experiment results:** report "No experiments found." Only analyze error events.
- **Corrupt JSON files:** skip the corrupt file, note it as a warning in the review.
- **Missing primary_metric or lower_is_better:** skip success metrics and proposal outcomes, note in the review that these inputs are needed for full analysis.
