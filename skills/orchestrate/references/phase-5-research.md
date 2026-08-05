# Phase 5: Research (Optional)

**Phase gate:** Run `pipeline_state.py <exp_root> gate 4 5` before entering. On completion: `pipeline_state.py <exp_root> log-gate 5 completed "<summary>"`.

Phase 5 runs as a **dynamic workflow**. If the user chose research (option 1, 3, or 4 from Phase 4), build the args and launch the workflow:

```
result = Workflow({
  scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-5-research.js",
  args: {
    exp_root,
    project_root,
    primary_metric,
    model_category,
    scope_level,          # from Phase 4 (architecture for option 3, etc.)
    source: "web",        # knowledge-mode degradation only applies when source is "both" (not sent by Phase 4); zero-proposal -> HP-only is an orchestrator-level decision made after this workflow returns
    user_papers,          # list, or null
    vault_agent,          # from Phase 0 Step 3.1 user_choices — host-project research agent, or null
    vault_paths,          # from Phase 0 Step 3.1 user_choices — starting-point note dirs (a hint, not a boundary)
    goal_summary          # `goal_memory.py <exp_root> summary` output; only needed when vault_agent is set
  }
})
```

**Optional: route the research to a host-project agent.** If the host project has its own literature agent over a curated note corpus, pass its `subagent_type` as `vault_agent` (the user picks it at Phase 0 Step 3.1). Stage routing:

| Stage | Agent | Why |
|---|---|---|
| Fan-out, Vet | `vault_agent` | Judgment stages. Neither writes a file; both return a schema validated at the tool-call layer. For a corpus-sourced candidate the deep-read is the curated note, and the novelty call is the taste that corpus encodes. |
| Synthesize (+ verify, + empty-findings) | `ml-optimizer:research-agent` | Contract stage. Writes the exact `research-findings.md` format under a `PreToolUse` hook that blocks a wrong path or schema, and emits the 1-based indices Phase 6's `selected_indices` resolve against. Index drift means implementing the wrong proposal. |

Omit `vault_agent` (or pass null) and every stage behaves exactly as before.

**When `vault_agent` is set:** pass `goal_summary` too — a host-project agent gets no SubagentStart injection, so without it the agent cannot see frozen parameters, `scope_level`, or the dead-end catalog. `vault_paths` is a starting hint, not a boundary. Never invent an agent name; an `agentType` that does not resolve fails the dispatch.

The workflow fans out across two `model_category`-specific angles plus one generic angle (user papers fold into the first angle's dispatch), dedups, runs the deep-read + adversarial feasibility check, writes `reports/research-findings.md`, initializes the research agenda (via `error_tracker.py`), and returns:

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

1. **Web/paper degradation is absorbed by the fan-out agent.** The research skill runs WebSearch and alphaxiv in parallel and, for `source: "both"`, also draws on LLM training knowledge — so a failure degrades *within a single* dispatch rather than aborting (alphaxiv alone never blocks; WebSearch suffices). There is no separate `source: "knowledge"` retry dispatch. A `vault_agent` instead falls back from its corpus to whatever web tools it carries, which may not include alphaxiv.
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
