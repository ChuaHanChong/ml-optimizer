# Phase 5: Research (Optional)

**Phase gate:** Run `pipeline_state.py <exp_root> gate 4 5` before entering. On completion: `pipeline_state.py <exp_root> log-gate 5 completed "<summary>"`.

Phase 5 runs as a **dynamic workflow**. If the user chose research (option 1, 3, or 4 from Phase 4), build the args and launch the workflow:

```
result = Workflow({
  scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-5-research.js",
  args: {
    exp_root,
    primary_metric,
    model_category,
    scope_level,          # from Phase 4 (architecture for option 3, etc.)
    source: "web",        # knowledge-mode degradation only applies when source is "both" (not sent by Phase 4); zero-proposal -> HP-only is an orchestrator-level decision made after this workflow returns
    user_papers           # list, or null
  }
})
```

The workflow dispatches `ml-optimizer:research-agent` internally (across two model_category-specific angles + one generic angle; user papers are folded into the first angle's dispatch), dedups, runs the deep-read + adversarial feasibility check, writes `reports/research-findings.md`, initializes the research agenda (via `error_tracker.py`), and returns:

```
{
  findings_path: "<exp_root>/reports/research-findings.md",
  proposals: [{index, title, impact, confidence, feasibility, scope, type, implementation_strategy, files_to_modify}, ...],
  agenda_initialized: true
}
```

Read `result.proposals` and the findings file to drive the post-research checkpoint. Then validate the proposals against goals:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> validate-output research '<proposals_json>'
```

## Research Failure Recovery (handled inside the workflow)

Recovery is internal to the phase-5 workflow — there is no orchestrator-side retry:

1. **Web/paper degradation is absorbed by the research agent.** The research skill runs WebSearch and alphaxiv in parallel and, for `source: "both"`, also draws on LLM training knowledge — so a web/alphaxiv failure degrades gracefully *within a single* research-agent dispatch rather than aborting (alphaxiv failure alone never blocks; WebSearch results suffice). There is no separate `source: "knowledge"` retry dispatch.
2. **No surviving proposals → honest-empty findings.** If vetting leaves zero proposals (research turned up nothing, or every candidate was a dead end), the Synthesize stage writes a minimal `research-findings.md` (Problem Statement + an empty "Proposals (Ranked by Priority)" section) and the research-agent logs a `research_failure` event (`category: "research_failure", source: "research"` — the workflow does not pin an exact severity). The workflow returns empty `proposals` (`agenda_initialized: false`), and the orchestrator continues with HP-only optimization.
3. **Each step is logged** to the error tracker for post-session review; dev_notes records "Research yielded no proposals — proceeding with HP tuning only."

## User Checkpoint (Post-Research)

After the workflow returns, use AskUserQuestion to show research findings:

```
Research findings:
[summary of proposals from research-findings.md]

Which proposals should I pursue?
- [1] Proposal A (complexity: low, expected: +X%)
- [2] Proposal B (complexity: medium, expected: +Y%)
- [3] Custom: describe your own approach
- [4] Skip research, just tune HPs
```
