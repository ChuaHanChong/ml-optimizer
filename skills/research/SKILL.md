---
name: research
description: "Research ML optimization techniques via web search and paper analysis. Extracts actionable proposals from papers with implementation details, expected impact, and complexity ratings. Use when: need to find new techniques for improving an ML model."
---

# ML Research Agent

Search for and analyze ML techniques that could improve the target model. Extract actionable, implementable proposals — not just paper summaries.

## Reference

- Paper analysis guide: `references/paper-analysis.md` (in this skill's directory)
- Read this reference FIRST to understand the extraction framework.

## Inputs Expected

From the orchestrator:
- `model_type`: Type of model (e.g., "diffusion model", "ResNet", "transformer")
- `task`: What the model does (e.g., "image restoration", "super-resolution", "classification")
- `current_metrics`: Current performance numbers
- `problem_description`: What needs improvement
- `user_papers`: Optional list of paper URLs or files provided by the user

## Step 1: Analyze User-Provided Papers (if any)

If the user provided papers or URLs:

1. For each URL, use WebFetch to retrieve the content:
   ```
   WebFetch(url: "<paper_url>")
   ```

2. For local files, use Read to read them

3. Apply the paper analysis framework from `references/paper-analysis.md`:
   - Extract core technique
   - Determine implementation details
   - Assess expected impact
   - Identify risks

## Step 1.5: Check for Existing Research (Deduplication)

Before searching, check if `experiments/reports/research-findings.md` already exists:

1. If it exists, read it and extract all previously proposed technique names
2. When generating new proposals, exclude techniques that were already proposed
3. This prevents re-proposing the same techniques on subsequent optimization runs

## Step 2: Web Search for Techniques

Construct targeted searches based on the model type and task:

### Search queries to run (adapt to the specific model/task):

1. **Architecture improvements:**
   ```
   WebSearch(query: "<task> <model_type> architecture improvement 2024 2025")
   ```

2. **Training strategies:**
   ```
   WebSearch(query: "<task> training strategy tricks <model_type>")
   ```

3. **Loss functions:**
   ```
   WebSearch(query: "<task> loss function improvement state-of-the-art")
   ```

4. **Specific improvements:**
   ```
   WebSearch(query: "<model_type> optimization techniques better performance")
   ```

5. **Recent papers:**
   ```
   WebSearch(query: "arxiv <task> <model_type> 2024 2025 improvement")
   ```

Run at least 3 searches. For each promising result, use WebFetch to get more details.

## Step 3: Analyze Found Papers

For each relevant paper or technique found:

1. Read the abstract and key results
2. Apply the extraction framework:
   - What is the technique?
   - What specifically needs to change in the code?
   - What improvement did they report?
   - How complex is the implementation?
   - What are the risks?

3. Rate feasibility for THIS specific project:
   - Is it compatible with the model architecture?
   - Does it fit the computational budget?
   - Can it be implemented without major refactoring?

## Step 4: Rank Proposals

Score each proposal on three axes:
- **Expected impact** (1-10): How much improvement is likely?
- **Feasibility** (1-10): How easy is it to implement?
- **Confidence** (1-10): How confident are we in the expected outcome?

Priority score = (impact * confidence) / (11 - min(feasibility, 10))

Note: Clamp feasibility to [1, 10] range to prevent division by zero when feasibility=11.

Sort proposals by priority score, highest first.

## Step 5: Write Research Findings

Write to `experiments/reports/research-findings.md`:

```markdown
# Research Findings

## Problem Statement
[Description of what we're trying to improve]

## Current Performance
[Baseline metrics]

## Sources Consulted
- [Paper/URL 1]: [Key takeaway]
- [Paper/URL 2]: [Key takeaway]
- ...

## Proposals (Ranked by Priority)

### Proposal 1: [Name] (Priority: X/10)
- **Type:** code_change | hp_only
- **Source:** [Paper title and URL]
- **Technique:** [Category] - [Description]
- **What to change:**
  - [Specific file and function to modify]
  - [What the change looks like]
- **Expected improvement:** [X% on metric]
- **Complexity:** Low/Medium/High
- **Risk:** [What could go wrong]
- **Implementation steps:**
  1. [Step 1]
  2. [Step 2]
  3. [Step 3]

### Proposal 2: [Name] (Priority: Y/10)
...

## Recommendations
- **Quick wins (low complexity):** [Proposals to try first]
- **High potential (medium complexity):** [Proposals for second round]
- **Ambitious (high complexity):** [Proposals if quick wins don't suffice]

## Not Recommended
- [Technique X]: [Why it's not suitable for this project]
```

## Step 6: Summary for Orchestrator

Return:
- Number of proposals found
- Top 3 proposals with brief summaries
- Recommended order of implementation
- Any dependencies between proposals
- Estimated total implementation effort

## Tips for Effective Research

1. **Be specific in searches:** "diffusion model image restoration perceptual loss" is better than "ML improvement"
2. **Check recency:** Prefer papers from 2023-2025 over older ones
3. **Look for code:** Papers with code repos are much more implementable
4. **Check benchmarks:** Make sure reported improvements are on comparable tasks/datasets
5. **Combine techniques:** Some improvements stack (e.g., better loss + better scheduler)
6. **Be honest about confidence:** If a technique seems promising but risky, say so

## Error Handling

- **WebSearch fails:** Try alternative search terms, or ask user for specific papers
- **Paper behind paywall:** Note the limitation, extract what's available from abstract
- **No relevant results:** Broaden search terms, try related tasks/model types
- **Contradictory findings:** Note both perspectives, let the user decide
