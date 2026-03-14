# Phase 9: Report

After the experiment loop exits:

1. Dispatch the report agent:
   ```
   Agent(
     description: "Generate final optimization report",
     prompt: "Generate a comprehensive final report. Parameters: project_root: {project_root}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, model_description: {model_description}, task_description: {task_description}.",
     subagent_type: "ml-optimizer:report-agent"
   )
   ```
2. It generates a comprehensive final report
3. Sync errors to cross-project memory:
   ```bash
   python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> sync ~/.claude/plugins/ml-optimizer
   ```
4. **Self-improvement review:**
   **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, auto-run review with `scope: "session"`. Skip AskUserQuestion. Log to dev_notes: "Auto-running self-improvement review (autonomous mode)." Dispatch the review agent:
   ```
   Agent(
     description: "Self-improvement review (autonomous)",
     prompt: "Ultrathink. Run self-improvement review. Parameters: project_root: {project_root}, exp_root: {exp_root}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, scope: session.",
     subagent_type: "ml-optimizer:review-agent"
   )
   ```

   **Otherwise:** Ask the user:
   ```
   AskUserQuestion: "Would you like a self-improvement review? It analyzes what worked, what didn't, and suggests plugin improvements for future sessions."
   Options: ["Yes, run review", "No, skip"]
   ```
   If yes, dispatch the review agent:
   ```
   Agent(
     description: "Self-improvement review",
     prompt: "Ultrathink. Run self-improvement review. Parameters: project_root: {project_root}, exp_root: {exp_root}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, scope: both.",
     subagent_type: "ml-optimizer:review-agent"
   )
   ```
5. Generate the progress dashboard:
   ```bash
   python3 ~/.claude/plugins/ml-optimizer/scripts/dashboard.py <exp_root>
   ```

6. Present the summary to the user:

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
