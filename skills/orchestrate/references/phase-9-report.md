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

Dispatch the analysis agent in review mode (resume-or-dispatch pattern):

**IF `agent_registry["analysis"]` is not null** (agent exists from batch analysis in Phase 7):

```
SendMessage(
  to: agent_registry["analysis"],
  message: "Ultrathink. End-of-session review. You have context from batch analyses.
    CONTEXT FROM OTHER AGENTS:
    - ANALYZE: final analysis across all batches
    - RESEARCH: all proposals attempted, {N_successful}/{N_total} improved
    - HP-TUNE: {total_iterations} iterations, best config: {best_config}
    Parameters: project_root: {project_root}, exp_root: {exp_root}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, scope: session."
)
```

→ If `SendMessage` fails: fall back to the `Agent()` dispatch below.

**ELSE** (first dispatch — no existing agent):

```
Agent(
  description: "Session review",
  prompt: "Ultrathink. Run session review. Parameters: project_root: {project_root}, exp_root: {exp_root}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, scope: session.",
  subagent_type: "ml-optimizer:analysis-agent"
)
```

→ Save returned `agentId` to `agent_registry["analysis"]`

