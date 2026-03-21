---
name: review
description: "Review an optimization session — analyze what worked, what failed, and what to improve in future runs. Examines error patterns, experiment outcomes, and proposal effectiveness. Advisory only — presents insights and recommendations. Use when: an optimization session has completed or encountered issues."
disable-model-invocation: true
user-invocable: false
---

# Session Review

Use extended thinking for all analytical reasoning in this skill. Ultrathink. Think through error causality chains, systemic patterns vs one-off failures, success patterns worth reinforcing, and actionable recommendations for future optimization runs.

Analyze the optimization session to identify what worked, what failed, and what to improve in future runs. Focus on experiment outcomes, HP patterns, method effectiveness, and strategy recommendations.

## Inputs Expected

From the orchestrator or direct invocation:
- `project_root`: Project root directory
- `exp_root`: Path to experiments/ directory (default: `<project_root>/experiments`)
- `primary_metric`: The metric that was optimized (e.g., "accuracy", "loss")
- `lower_is_better`: Whether lower values are better for the primary metric
- `scope`: `"session"` (default)

## Step 1: Load Error Data

> **Goal check:** Read optimization goals to assess whether the session achieved its objectives.

### Per-project data (if project_root provided):

1. Run the session summary:
```bash
python3 scripts/error_tracker.py <exp_root> summary
```

2. Run pattern detection:
```bash
python3 scripts/error_tracker.py <exp_root> patterns
```

3. Read supporting files:
   - `<exp_root>/reports/error-log.json` — full error event list
   - `<exp_root>/reports/batch-*-analysis.md` — batch analysis reports
   - `<exp_root>/dev_notes.md` — session narrative

## Step 1.1: Read Experiment Data

Read the experiment results and research findings to understand the full session context:

- `<exp_root>/results/baseline.json` — baseline metrics
- `<exp_root>/results/exp-*.json` — all experiment results
- `<exp_root>/results/implementation-manifest.json` — which proposals were implemented
- `<exp_root>/reports/research-findings*.md` — what techniques were researched
- `<exp_root>/optimization-goals.json` — what the user wanted to achieve
- `<exp_root>/learned-behaviors.json` — what was learned during the session

## Step 1.2: Compute Success Metrics

Run the success metrics analyzer to understand what *worked*, not just what failed:

```bash
python3 scripts/error_tracker.py <exp_root> success <primary_metric> <lower_is_better>
```

This returns:
- Success rate (completed / total experiments)
- Improvement rate (how many beat baseline)
- Best improvement percentage
- Duration analysis (avg time for completed vs failed experiments)
- Top performing configs and worst performing configs
- Time wasted on failures as a percentage

## Step 1.3: Compute Proposal Outcomes

Run the proposal outcome tracker to understand which decisions paid off:

```bash
python3 scripts/error_tracker.py <exp_root> proposals <primary_metric> <lower_is_better>
```

This returns:
- Research proposal outcomes: which proposals led to improvements, which didn't
- HP proposal stats: total proposed vs run vs completed vs beat baseline
- Implementation stats: validated vs failed validation vs implementation error

## Step 1.4: Load Suggestion History

Check if this pattern was previously suggested in an earlier review:

```bash
python3 scripts/error_tracker.py <exp_root> suggestion-history
```

If the output is non-empty, note which `pattern_id` values have been suggested before and their iteration counts. Use this in Step 4 to downrank repeated suggestions.

## Step 1.5: Scope Violation Analysis

Query scope violations from behavioral memory:
```bash
python3 scripts/goal_memory.py <project_root>/experiments query-behaviors scope_violation
```

Include in the review:
- Total scope violations caught by validation
- Most common violation types (frozen params, dead-end re-proposals, scope breaches)
- Whether violations decreased over the session (indicating agent learning)

## Step 2: Categorize Issues

Group all findings into three categories:

### Category A: Agent/Skill Failures
- Agent crashes, timeouts, invalid outputs
- Skill invocation failures
- Look for `category: "agent_failure"` events

### Category B: Experiment Patterns
- Divergence causes (NaN, explosion, plateau)
- OOM events and their configs
- HP combinations that consistently fail
- Proposals that never improve metrics
- Look for `category: "training_failure"`, `"divergence"`, `"implementation_error"` events

### Category C: Pipeline Inefficiencies
- Wasted budget (batches where all experiments failed)
- Redundant HP proposals
- Code branches that never beat baseline
- Suboptimal GPU utilization
- Look for `category: "pipeline_inefficiency"` events

## Step 3: Assess Severity

For each issue:
- **Critical:** Blocks the pipeline or causes data loss. Needs immediate fix.
- **Warning:** Degrades optimization quality. Should be addressed.
- **Info:** Optimization opportunity. Nice to have.

## Step 3.1: Rank and Prioritize Issues

Rank all detected patterns by impact score:

```bash
python3 scripts/error_tracker.py <exp_root> rank <total_experiments>
```

Where `<total_experiments>` is the `total_experiments` value from Step 1.2 success metrics (omit if Step 1.2 was skipped). This returns patterns sorted by score (severity weight × occurrences), with a `significance` field when total_experiments is provided. Use this ranking to order your suggestions — highest score first. In Step 6, present only the top 3 most impactful suggestions to the user.

## Step 4: Generate Recommendations

For each detected pattern (in rank order), generate a specific, actionable recommendation for future optimization runs.

### Recommendation Format

```markdown
### Recommendation: [Concise Title]
- **Severity:** [Critical / Warning / Info]
- **Evidence:** [Which error events, experiments, or patterns support this]
- **Problem:** [What went wrong, with specifics from the error log]
- **Recommendation:** [What to do differently next time]
- **Expected impact:** [What would improve]
- **Confidence:** [High / Medium / Low — based on evidence strength and sample size]
```

### Recommendation Quality Rules

1. **Be specific.** "Improve HP tuning" is useless. "Start LR search below 0.01 — all LRs above this diverged within 50 steps" is actionable.
2. **Reference evidence.** Every recommendation must cite experiment results, error patterns, or metrics.
3. **Focus on strategy.** Recommend what the user or orchestrator should do differently — HP ranges, methods to try/avoid, budget allocation, scope changes.
4. **Check for repeats.** If this pattern was previously flagged (from Step 1.4), note "Previously flagged" and consider whether the recommendation is still relevant.

### Error-Based Recommendations

| Pattern | Typical Recommendation |
|---------|----------------------|
| High LR divergence | Start LR search below the observed divergence threshold |
| OOM at batch_size | Cap batch_size at the OOM limit; consider gradient accumulation |
| All-fail batches | Narrow HP search space or try a different method |
| Redundant HP proposals | Widen search space — promising regions may be exhausted |
| Research failures | Try different search terms or knowledge-only mode |

### Success-Based Recommendations

| Signal | Typical Recommendation |
|--------|----------------------|
| LR range X-Y always beats baseline | Focus search in this range for similar models |
| Method X consistently best | Prioritize similar techniques in future research |
| Short divergence time (<30s) | Use aggressive early-abort — saves compute |
| Time wasted on failures >30% | Use narrower initial search or increase pre-flight validation |

## Step 5: Write Session Review

Write the review to `<exp_root>/reports/session-review.md`:

```markdown
# Session Review — Self-Improvement Analysis

**Date:** <date>
**Project:** <project_root>
**Scope:** session

## Executive Summary

- **Total experiments:** <N> | **Completed:** <N> | **Failed:** <N> | **Diverged:** <N>
- **Success rate:** <X%> | **Improvement rate:** <Y%>
- **Error events:** <N> (Critical: <N>, Warning: <N>, Info: <N>)
- **Patterns detected:** <list>
- **Suggestions generated:** <N>

## What Worked

### Top Performing Configurations
| Exp ID | Config Changes | Metric Value | Improvement |
|--------|---------------|--------------|-------------|
| ... | ... | ... | ... |

### Effective Patterns
- [Pattern that led to improvements — e.g., "LR in 0.0001-0.001 range consistently improved accuracy"]
- [Code change that worked — e.g., "perceptual-loss branch improved metric by 8%"]

### Efficiency Highlights
- Time wasted on failures: <X%>
- Average time to detect divergence: <N>s

## Proposal Outcomes

### Research Proposals
| Proposal | Branch | Experiments | Beat Baseline | Best Improvement |
|----------|--------|-------------|---------------|-----------------|
| ... | ... | ... | ... | ... |

### Implementation Stats
- Proposals validated: <N>/<total>
- Validation failures: <N>
- Implementation errors: <N>

### HP Success Patterns
- Configs that worked: [summary]
- Configs that failed: [summary]

## Error Timeline

| Time | Category | Severity | Message |
|------|----------|----------|---------|
| ... | ... | ... | ... |

## Detected Patterns

### [Pattern Name]
- **Occurrences:** <N>
- **Description:** <what's happening>
- **Affected experiments:** <list>

## Improvement Suggestions

[All suggestions in the format from Step 4]
```

## Step 5.1: Log Generated Suggestions

After writing the review, log each generated suggestion so future reviews can detect repeats:

```bash
python3 scripts/error_tracker.py <exp_root> log-suggestion <pattern_id> <scope>
```

Run this once per suggestion generated in Step 4.

## Step 6: Present Summary to User

Report the key findings:

```
Session Review Complete

Analyzed <N> experiments and <N> error events.

What worked:
- [Best performing config — 1 line]
- [Most effective proposal — 1 line]

What to improve:
1. [Most impactful suggestion — 1 line]
2. [Second suggestion — 1 line]
3. [Third suggestion — 1 line]

Efficiency: <X%> success rate, <Y%> time wasted on failures

Full review: <exp_root>/reports/session-review.md

These are advisory suggestions only. Review the full report and apply changes you agree with.
```

## Error Handling

- **No error log exists:** Report "No errors tracked in this session." Still run success metrics and proposal outcomes if experiment results exist.
- **Empty error log:** Report "0 events tracked." Still analyze experiment outcomes for success patterns.
- **No experiment results:** Report "No experiments found." Only analyze error events.
- **Corrupt JSON files:** Skip the corrupt file, note it as a warning in the review.
- **Missing primary_metric or lower_is_better:** Skip success metrics and proposal outcomes, note in the review that these inputs are needed for full analysis.
