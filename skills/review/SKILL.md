---
name: review
description: "Analyze error logs, experiment outcomes, and proposal effectiveness to generate improvement suggestions for the ML optimizer plugin. Can be invoked at end-of-session or mid-pipeline. Advisory only — presents suggestions, does not auto-apply changes. Use when: an optimization session has completed or encountered issues and you want to review what went wrong, what worked, and how the plugin could improve."
---

# Self-Improvement Review

Use extended thinking for all analytical reasoning in this skill. Ultrathink. Think through error causality chains, systemic patterns vs one-off failures, success patterns worth reinforcing, and the specificity of improvement suggestions before writing them.

Analyze error logs, experiment outcomes, proposal effectiveness, and cross-project memory to identify patterns and generate structured improvement suggestions for the ML optimizer plugin itself.

## Important Files

- Error tracker script: `~/.claude/plugins/ml-optimizer/scripts/error_tracker.py`
- Cross-project memory: `~/.claude/plugins/ml-optimizer/memory/cross-project-errors.json`
- Plugin skills: `~/.claude/plugins/ml-optimizer/skills/*/SKILL.md`
- Plugin agents: `~/.claude/plugins/ml-optimizer/agents/*.md`
- Plugin scripts: `~/.claude/plugins/ml-optimizer/scripts/*.py`

## Inputs Expected

From the orchestrator or direct invocation:
- `project_root`: Project root directory (optional — omit for cross-project-only review)
- `exp_root`: Path to experiments/ directory (default: `<project_root>/experiments`)
- `primary_metric`: The metric that was optimized (e.g., "accuracy", "loss")
- `lower_is_better`: Whether lower values are better for the primary metric
- `scope`: One of `"session"` (current project only), `"cross-project"` (all projects), `"both"` (default: `"both"`)

## Step 1: Load Error Data

### Per-project data (if project_root provided):

1. Run the session summary:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> summary
```

2. Run pattern detection:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> patterns
```

3. Read supporting files:
   - `<exp_root>/reports/error-log.json` — full error event list
   - `<exp_root>/reports/batch-*-analysis.md` — batch analysis reports
   - `<exp_root>/dev_notes.md` — session narrative

### Cross-project data (if scope includes cross-project):

4. Read `~/.claude/plugins/ml-optimizer/memory/cross-project-errors.json`
   - If file doesn't exist, skip cross-project analysis

5. Clean up old sessions to prevent unbounded growth:
   ```bash
   python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> cleanup ~/.claude/plugins/ml-optimizer 10
   ```

## Step 1.5: Read Target Plugin Files

Based on which error categories are present, read the relevant plugin files so suggestions are grounded in reality:

| Error Categories Present | Read These Files |
|--------------------------|------------------|
| `agent_failure` | The failing agent's definition in `agents/<agent-name>.md` |
| `divergence` or `training_failure` | `skills/hp-tune/SKILL.md`, `skills/monitor/SKILL.md` |
| `implementation_error` | `skills/implement/SKILL.md` |
| `pipeline_inefficiency` | `skills/orchestrate/SKILL.md`, `skills/analyze/SKILL.md` |
| `config_error` | `skills/experiment/SKILL.md` |
| `research_failure` | `skills/research/SKILL.md` |
| `timeout` | The timed-out agent's definition in `agents/<agent-name>.md`, `skills/orchestrate/SKILL.md` |
| `resource_error` | `skills/baseline/SKILL.md`, `skills/experiment/SKILL.md`, `skills/prerequisites/SKILL.md`, `agents/prerequisites-agent.md` |

**Why this matters:** You cannot suggest "add a batch_size cap to hp-tune Step 2" without reading what Step 2 currently says. Read the files first, then propose specific edits.

## Step 1.6: Compute Success Metrics

Run the success metrics analyzer to understand what *worked*, not just what failed:

```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> success <primary_metric> <lower_is_better>
```

This returns:
- Success rate (completed / total experiments)
- Improvement rate (how many beat baseline)
- Best improvement percentage
- Duration analysis (avg time for completed vs failed experiments)
- Top performing configs and worst performing configs
- Time wasted on failures as a percentage

## Step 1.7: Compute Proposal Outcomes

Run the proposal outcome tracker to understand which decisions paid off:

```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> proposals <primary_metric> <lower_is_better>
```

This returns:
- Research proposal outcomes: which proposals led to improvements, which didn't
- HP proposal stats: total proposed vs run vs completed vs beat baseline
- Implementation stats: validated vs failed validation vs implementation error

## Step 1.8: Load Suggestion History

Check if this pattern was previously suggested in an earlier review:

```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> suggestion-history
```

If the output is non-empty, note which `pattern_id` values have been suggested before and their iteration counts. Use this in Step 4 to downrank repeated suggestions.

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

## Step 3.5: Rank and Prioritize Issues

Rank all detected patterns by impact score:

For session-only scope:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> rank <total_experiments>
```

When scope includes cross-project (`"both"` or `"cross-project"`), pass the plugin root to enable the 1.5× cross-project boost:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> rank <total_experiments> ~/.claude/plugins/ml-optimizer
```

Where `<total_experiments>` is the `total_experiments` value from Step 1.6 success metrics (omit if Step 1.6 was skipped). This returns patterns sorted by score (severity weight × occurrences × cross-project boost), with a `significance` field when total_experiments is provided. Use this ranking to order your suggestions — highest score first. In Step 6, present only the top 3 most impactful suggestions to the user.

## Step 4: Generate Improvement Suggestions

For each detected pattern (in rank order), generate a specific, actionable suggestion. Each suggestion MUST reference a concrete plugin file and propose a specific change.

### Suggestion Format

```markdown
### Suggestion: [Concise Title]
- **Severity:** [Critical / Warning / Info]
- **Evidence:** [Which error events or patterns support this — cite event_ids or pattern_ids]
- **File:** [Exact plugin file path, e.g., skills/hp-tune/SKILL.md]
- **Current behavior:** [What the file currently says in the relevant section — quote it]
- **Problem:** [What went wrong, with specifics from the error log]
- **Proposed change:** [Specific text, logic, or threshold to add or modify]
- **Expected impact:** [What would improve if this change were applied]
- **Confidence:** [High / Medium / Low — based on evidence strength and sample size]
```

### Suggestion Quality Rules

1. **Be specific.** "Improve error handling" is useless. "Add a batch_size cap of 128 to hp-tune Step 2 when GPU memory < 12GB" is actionable.
2. **Reference evidence.** Every suggestion must cite at least one error event, pattern, or experiment result.
3. **Target the right file.** If the problem is in HP proposals, target `skills/hp-tune/SKILL.md`, not a random script.
4. **Quote current behavior.** You read the file in Step 1.5 — quote the relevant section so the user can verify.
5. **Consider side effects.** Will this change break other skills or narrow the search space too much?
6. **Cross-project suggestions get higher confidence** when the same pattern appears in 2+ projects.
7. **Check for repeats.** If this pattern was previously suggested (from Step 1.8), reduce confidence by one level (High→Medium, Medium→Low) and note "Previously suggested (iteration N)" in the Evidence field. Skip Low-confidence repeats entirely — they've been flagged enough times without action.

### Error-Based Suggestion Types

| Pattern | Target File | Typical Suggestion |
|---------|------------|-------------------|
| High LR divergence | `skills/hp-tune/SKILL.md` | Add LR ceiling based on observed divergence threshold |
| OOM at batch_size | `skills/hp-tune/SKILL.md` | Add batch_size cap based on GPU memory |
| Proposal validation failures | `skills/implement/SKILL.md` | Add pre-implementation feasibility check |
| Wasted budget (all-fail batches) | `skills/orchestrate/SKILL.md` | Add early termination if first experiment in batch diverges quickly |
| Redundant HP proposals | `skills/hp-tune/SKILL.md` | Increase minimum distance between proposals |
| Agent timeouts | `agents/<agent>.md` | Simplify agent prompt or split into sub-tasks |
| Log format unrecognized | `scripts/parse_logs.py` | Add support for the unrecognized format |
| Research failures | `skills/research/SKILL.md` | Broaden search terms or add fallback search strategies |
| Baseline failures | `skills/baseline/SKILL.md` | Add format-specific parse fallbacks |
| Timeouts | `agents/<agent>.md`, `skills/orchestrate/SKILL.md` | Simplify agent prompt or increase timeout |
| Resource errors | `skills/experiment/SKILL.md`, `skills/baseline/SKILL.md` | Add resource checks before launching experiments |
| HP interaction failures | `skills/hp-tune/SKILL.md` | Avoid LR-bucket × batch_size combos that consistently fail |
| Early phase failures | Target phase's skill | Investigate setup — failures concentrated in that phase |
| Prerequisites failures | `skills/prerequisites/SKILL.md`, `agents/prerequisites-agent.md` | Add format-specific validation or broaden environment detection |
| Dataset format mismatch | `scripts/prerequisites_check.py` | Add support for the unrecognized data format |

### Success-Based Suggestion Types

| Signal | Target File | Typical Suggestion |
|--------|------------|-------------------|
| LR range X-Y always beats baseline | `skills/hp-tune/SKILL.md` | Narrow initial search to this range for similar models |
| Code branch X consistently best | `skills/orchestrate/SKILL.md` | Allocate more budget to winning branches earlier |
| Short divergence time (<30s) | `skills/monitor/SKILL.md` | Reduce initial check interval for faster kills |
| High validation failure rate | `skills/implement/SKILL.md` | Add pre-implementation static analysis step |
| Time wasted on failures >30% | `skills/orchestrate/SKILL.md` | Add early-stop heuristic for first experiment in batch |
| Proposal X always improves | `skills/research/SKILL.md` | Prioritize similar techniques in future research |

## Step 5: Write Session Review

Write the review to `<exp_root>/reports/session-review.md`:

```markdown
# Session Review — Self-Improvement Analysis

**Date:** <date>
**Project:** <project_root>
**Scope:** <session / cross-project / both>

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

## Cross-Project Insights

[Only if scope includes cross-project]

### Recurring Patterns
- [Pattern seen across N projects with description]

### Plugin-Wide Recommendations
- [Suggestions that would improve the plugin for all projects]
```

## Step 5.5: Log Generated Suggestions

After writing the review, log each generated suggestion so future reviews can detect repeats:

```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log-suggestion <pattern_id> <scope>
```

Run this once per suggestion generated in Step 4. The `<scope>` should match the scope used for this review invocation (`session`, `cross-project`, or `both`).

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
- **No experiment results:** Report "No experiments found." Only analyze error events and cross-project memory.
- **Cross-project memory missing:** Skip cross-project analysis, note it in the review.
- **Corrupt JSON files:** Skip the corrupt file, note it as a warning in the review.
- **Missing primary_metric or lower_is_better:** Skip success metrics and proposal outcomes, note in the review that these inputs are needed for full analysis.
