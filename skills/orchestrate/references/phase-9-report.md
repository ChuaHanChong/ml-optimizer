# Phase 9: Report, Review & Promotion

**Phase gate:** If Phase 8 was skipped: `pipeline_state.py <exp_root> gate 7 9`. If Phase 8 ran: `pipeline_state.py <exp_root> gate 8 9`. On completion: `pipeline_state.py <exp_root> log-gate 9 completed "<summary>"`.

After the experiment loop exits:

**Pre-report state verification:** Before dispatching the report agent, verify critical state files exist:
- `<exp_root>/results/baseline.json` — must exist
- `<exp_root>/pipeline-state.json` — must exist
- `<exp_root>/results/exp-*.json` — at least 1 must exist
If any are missing, log to error tracker (`category: "config_error"`) and warn the user. Do NOT proceed with reporting if baseline is missing.

## Step 1: Generate Report & Present Summary

Dispatch the report agent:

```
Agent(
  description: "Generate final optimization report",
  prompt: "Generate a comprehensive final report. Parameters: project_root: {project_root}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, model_description: {model_description}, task_description: {task_description}.",
  subagent_type: "ml-optimizer:report-agent"
)
```

Generate the progress dashboard:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/dashboard.py <exp_root>
```

Present the summary to the user:

```
Optimization complete!

Best configuration: [exp_id]
[metric improvements vs baseline]

Key findings:
- [finding 1]
- [finding 2]

Full report: <exp_root>/reports/final-report.md
Dashboard: <exp_root>/reports/dashboard.html
```

## Step 2: Session Review

The analysis agent reviews the entire session to identify what worked, what didn't, and how to improve.

Dispatch the analysis agent in review mode (fresh dispatch — it reads the run's files: results, batch analyses, dead-ends, agenda):

```
Agent(
  description: "Session review",
  prompt: "Run session review. Read the run's files for context: results/round-N-*/exp-*.json, reports/batch-*-analysis.md, reports/dead-ends.json, reports/research-agenda.json. Parameters: project_root: {project_root}, exp_root: {exp_root}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, scope: session.",
  subagent_type: "ml-optimizer:analysis-agent"
)
```

