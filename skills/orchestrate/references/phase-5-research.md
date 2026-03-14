# Phase 5: Research (Optional)

If the user chose research (option 2 or 3 from Phase 4), dispatch the research agent:
```
Agent(
  description: "Research optimization techniques",
  prompt: "Ultrathink. Research ML optimization techniques. Parameters: source: web, model_type: {model_type}, task: {task}, current_metrics: {current_metrics}, problem_description: {problem_description}, user_papers: {user_papers or null}, exp_root: {exp_root}.",
  subagent_type: "ml-optimizer:research-agent"
)
```
Wait for research findings.

## Research Failure Recovery

If the research skill fails (web search errors, timeout, or no results):

1. **First fallback:** Retry with `source: "knowledge"` (skip web search, use LLM training knowledge only). Log to error tracker: `category: "agent_failure", severity: "warning", source: "orchestrate", message: "Research web search failed — retrying with knowledge-only mode"`.
2. **Second fallback:** If knowledge-only also fails, continue with HP-only optimization (no research proposals). Log to error tracker: `category: "agent_failure", severity: "warning", source: "orchestrate", message: "Research failed entirely — continuing with HP-only optimization"`. Log to dev_notes: "Research failed — proceeding with HP tuning only."
3. **Each fallback step** is logged to the error tracker for post-session review.

## User Checkpoint (Post-Research)

**Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, skip the user question below. Auto-select all proposals. Log to dev_notes: `"Autonomous mode: auto-selected all N research proposals for implementation"`.

Use AskUserQuestion to show research findings:

```
Research findings:
[summary of proposals from research-findings.md]

Which proposals should I pursue?
- [1] Proposal A (complexity: low, expected: +X%)
- [2] Proposal B (complexity: medium, expected: +Y%)
- [3] Custom: describe your own approach
- [4] Skip research, just tune HPs
```
