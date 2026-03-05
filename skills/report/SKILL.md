---
name: report
description: "Generate a comprehensive final report for an ML optimization effort. Compiles all experiment results, creates comparison tables, highlights best configuration, and summarizes the optimization journey. Use when: optimization is complete and a final report is needed."
---

# Final Report Generator

Generate a comprehensive report summarizing the entire optimization effort.

## Reference

- Report template: `references/report-template.md` (in this skill's directory)
- Use this template as the structure for the final report.

## Inputs Expected

From the orchestrator:
- `project_root`: Project root directory
- `primary_metric`: The metric that was optimized
- `lower_is_better`: Whether lower is better for the primary metric
- `model_description`: Brief description of the model
- `task_description`: What the model does

## Step 1: Gather All Data

### Load experiment results
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/result_analyzer.py \
  <project_root>/experiments/results \
  <primary_metric> \
  baseline \
  <lower_is_better>
```

### Read all batch analyses
Use Glob to find: `experiments/reports/batch-*-analysis.md`
Read each one for key findings.

### Read dev notes
Read `experiments/dev_notes.md` for decisions, reasoning, and observations.

### Read research findings (if applicable)
Check if `experiments/reports/research-findings.md` exists.
If so, read for proposals that were tried.

### Read implementation manifest (if applicable)
Check if `experiments/results/implementation-manifest.json` exists.
If so, read for: validated proposals, branches, files modified, conflicts.

### Extract profiling data
Read `experiments/results/baseline.json` and extract the `profiling` block
for GPU memory, throughput, and max batch size.

## Step 2: Compile Results Table

Create a comprehensive comparison table with ALL experiments:

1. Load all result JSONs from `experiments/results/`
2. Sort by primary metric (best first)
3. Include: exp_id, status, key config changes, all metrics, delta vs baseline

## Step 2.5: Compile HP Sensitivity Analysis

The `result_analyzer.py` `identify_correlations()` output includes per-HP
correlation data. Format this into the "Hyperparameter Sensitivity" table:
- Only include if ≥4 experiments completed (otherwise note "insufficient data")
- Show direction (lower/higher correlates with better metric)
- For categorical params, show most common value in top vs bottom performers

## Step 3: Identify Best Configuration

1. Find the experiment with the best primary metric
2. Compare its config to baseline config parameter by parameter
3. Calculate improvement percentage for each metric
4. Note the training command used (from the experiment's script file)

## Step 4: Summarize the Journey

Read through dev notes chronologically and batch analyses to reconstruct:
1. What was the starting point?
2. What approach was taken first? (research vs. direct HP tuning)
3. What key decisions were made and why?
4. What pivots happened? (if any)
5. When did the biggest improvements come?
6. Why was optimization stopped?

## Step 5: Extract Key Findings

From the analysis reports and results, identify:
1. Which hyperparameters had the most impact?
2. What techniques worked? What didn't?
3. Any surprising results?
4. What would be worth trying next?

## Step 5.5: Generate Visualizations

Use the plot_results.py script to generate ASCII charts:

```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/plot_results.py \
  <project_root>/experiments/results <primary_metric> comparison
```

Generate:
1. Metric comparison bar chart (all experiments)
2. Improvement timeline (best-so-far over time)
3. HP sensitivity scatter for the highest-impact HP

Include the ASCII chart output in the report (in code blocks).

## Step 6: Write the Report

Write to `experiments/reports/final-report.md` using the template from `references/report-template.md`.

Fill in all sections:
- Executive summary (2-3 sentences)
- Infrastructure summary (GPU, total time)
- Baseline metrics and profiling
- Search space explored
- Research & code changes (if applicable)
- Optimization journey narrative
- Full experiments table (with code branch, duration, GPU columns)
- Best configuration details
- HP sensitivity analysis
- Visualizations (ASCII charts from plot_results.py)
- Key findings
- What worked / what didn't
- Reproduction command
- Further improvement suggestions
- Expanded appendix with artifact table

## Step 7: Write Final Dev Notes Entry

Append to `experiments/dev_notes.md`:

```markdown
## <date> — Optimization Complete

- Best experiment: <exp_id>
- <metric>: <baseline_value> -> <best_value> (<improvement%>)
- Summary: <1-2 sentences about the overall optimization effort>
- Final report: experiments/reports/final-report.md
```

## Step 8: Present to User

Provide a concise summary to the orchestrator for the user:

```
Optimization Complete!

Best: <exp_id> - <metric>=<value> (<X% improvement> over baseline)

Key changes from baseline:
- <param1>: <old> -> <new>
- <param2>: <old> -> <new>

Top findings:
1. <finding 1>
2. <finding 2>

To reproduce: <command>

Full report: experiments/reports/final-report.md
```

## Output

Return to the orchestrator:
- Path to final report
- Best experiment ID and metrics
- Improvement over baseline
- Concise summary for user display

## Quality Checklist

Before writing the report, verify:
- [ ] All experiments accounted for (none missing from the table)
- [ ] Metrics are consistent (same units, same evaluation method)
- [ ] Baseline is included for comparison
- [ ] Failed/diverged experiments are included (they provide information too)
- [ ] Reproduction command is correct and complete
- [ ] Dev notes are referenced for decision context
- [ ] HP sensitivity analysis included (or noted as insufficient data)
- [ ] Infrastructure/profiling data included
- [ ] If research proposals were used, implementation manifest summarized
- [ ] Appendix has concrete file paths, not just vague references

## Edge Cases

### Missing Baseline
If `baseline.json` does not exist or has no metrics, the report should:
- State that no baseline was established
- Report absolute metric values only (no deltas or improvement percentages)
- Recommend re-running with a proper baseline for meaningful comparison

### Single Experiment
If only one experiment was run (plus baseline), the report should:
- Skip HP sensitivity analysis (insufficient data)
- Note that results are preliminary and more experiments are recommended
- Still generate the full report structure with available data

### All Experiments Diverged
If every experiment (excluding baseline) has `status: "diverged"`, the report should:
- Highlight this prominently in the executive summary
- Analyze common factors across diverged experiments (LR too high? batch size?)
- Recommend a more conservative search space
- Still document what was learned from the failures
