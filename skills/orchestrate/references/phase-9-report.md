# Phase 9: Report, Review & Promotion

After the experiment loop exits:

**Pre-report state verification:** Before dispatching the report agent, verify critical state files exist:
- `experiments/results/baseline.json` — must exist
- `experiments/pipeline-state.json` — must exist
- `experiments/results/exp-*.json` — at least 1 must exist
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

Full report: experiments/reports/final-report.md
Dashboard: experiments/reports/dashboard.html
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

## Step 3: Meta-Patch Promotion

If `hyperagent_state.active_meta_patches` is non-empty, the hyperagent generated strategy improvement patches during this session. Evaluate and promote them so they persist across sessions.

### 3a: Evaluate Meta-Patches

Dispatch `ml-optimizer:analysis-agent` with scope `"meta_patches"`:

```
SendMessage(
  to: agent_registry["analysis"],
  message: "Ultrathink. Evaluate meta-improvement patches.
  scope: meta_patches.
  Read experiments/meta-patches/meta-changelog.json for the list of patches.
  For each patch:
  1. Read the original skill file and the patched version
  2. Check: did experiments AFTER the patch was applied show improvement over experiments BEFORE?
  3. Assess: is the modification generalizable (not overfitted to this specific dataset/model)?
  4. Rate confidence: high (clear evidence) / medium (suggestive) / low (inconclusive)"
)
```

### 3b: Present to User

Use `AskUserQuestion` to present the evaluation results:

```
The hyperagent discovered {N} strategy improvements this session:

1. {skill}: {change} — {confidence} confidence
   Evidence: {evidence}
2. ...

Promote these to the plugin? They'll be committed to the current branch
so future optimization sessions use the improved strategies.
```

Options: "Promote all", "Select which to promote", "Skip (log for reference)"

### 3c: Promote (if approved)

For each approved patch:
1. Read the patched skill file from `experiments/meta-patches/<skill>-SKILL.md`
2. Prepend a marker header so future sessions can detect it:
   ```
   # [meta-improvement] <change description>. Session <date>.
   ```
3. Write to the plugin's skill directory: `${CLAUDE_PLUGIN_ROOT}/skills/<skill>/SKILL.md`
4. Commit directly to the plugin's current branch:
   ```bash
   cd ${CLAUDE_PLUGIN_ROOT}
   git add skills/<skill>/SKILL.md
   git commit -m "meta-improvement: <change description>"
   ```
   Immediately available in the next session without needing a merge.

### 3d: Log to Behavioral Memory

Whether promoted or not, log the meta-improvement outcome:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> log-behavior meta_improvement '{
  "patches": [{"skill": "<name>", "change": "<desc>", "promoted": true|false, "confidence": "<high|medium|low>"}],
  "session_improvement_pct": <overall_improvement>,
  "archive_generations": <total_generations>
}'
```

If `claude-mem` MCP is available, also log for cross-session recall:

```
mcp__plugin_claude-mem_mcp-search__search("hyperagent session: <summary of meta-improvements and outcomes>")
```
