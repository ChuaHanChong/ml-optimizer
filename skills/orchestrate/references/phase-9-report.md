# Phase 9: Report

After the experiment loop exits:

**Pre-report state verification:** Before dispatching the report agent, verify critical state files exist:
- `experiments/results/baseline.json` — must exist
- `experiments/pipeline-state.json` — must exist
- `experiments/results/exp-*.json` — at least 1 must exist
If any are missing, log to error tracker (`category: "config_error"`) and warn the user. Do NOT proceed with reporting if baseline is missing.

1. Dispatch the report agent:
   ```
   Agent(
     description: "Generate final optimization report",
     prompt: "Generate a comprehensive final report. Parameters: project_root: {project_root}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, model_description: {model_description}, task_description: {task_description}.",
     subagent_type: "ml-optimizer:report-agent"
   )
   ```
2. It generates a comprehensive final report
3. **Self-improvement review** (resume-or-dispatch pattern):
   Ask the user:
   ```
   AskUserQuestion: "Would you like a self-improvement review? It analyzes what worked, what didn't, and suggests plugin improvements for future sessions."
   Options: ["Yes, run review", "No, skip"]
   ```
   If yes, dispatch the review agent:

   **IF `agent_registry["review"]` is not null** (agent exists from mid-pipeline reviews in Phase 7):
   ```
   SendMessage(
     to: agent_registry["review"],
     message: "End-of-session self-improvement review. You have context from any mid-pipeline reviews.
       CONTEXT FROM OTHER AGENTS:
       - ANALYZE: final analysis across all batches
       - RESEARCH: all proposals attempted, {N_successful}/{N_total} improved
       - HP-TUNE: {total_iterations} iterations, best config: {best_config}
       Parameters: project_root: {project_root}, exp_root: {exp_root}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, scope: both."
   )
   ```
   → If `SendMessage` fails (agent no longer reachable): fall back to the `Agent()` dispatch below, update `agent_registry["review"]` with the new agentId.

   **ELSE** (first dispatch — no existing agent):
   ```
   Agent(
     description: "Self-improvement review",
     prompt: "Ultrathink. Run self-improvement review. Parameters: project_root: {project_root}, exp_root: {exp_root}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, scope: both.",
     subagent_type: "ml-optimizer:review-agent"
   )
   ```
   → Save returned `agentId` to `agent_registry["review"]`
   → Persist registry: `save_state(..., agent_registry=agent_registry)`
4. Generate the progress dashboard:
   ```bash
   python3 scripts/dashboard.py <exp_root>
   ```

5. Present the summary to the user:

```
Optimization complete!

Best configuration: [exp_id]
[metric improvements vs baseline]

Key findings:
- [finding 1]
- [finding 2]

Full report: experiments/reports/final-report.md
Dashboard: experiments/reports/dashboard.html
```
