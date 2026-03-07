---
name: research
description: "Research ML optimization techniques via web search and paper analysis. Extracts actionable proposals from papers with implementation details, expected impact, and complexity ratings. Use when: need to find new techniques for improving an ML model."
---

# ML Research Agent

Use extended thinking for all analytical reasoning in this skill. Ultrathink. Critically evaluate paper claims, assess feasibility and compatibility, and reason through confidence scoring before ranking proposals.

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
- `exp_root`: Path to experiments/ directory (for error logging)

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

**Date handling:** Always use the current year dynamically. Never hardcode year strings. Use `<current_year-1> <current_year>` in search queries (e.g., if the current year is 2026, search for "2025 2026").

1. **Architecture improvements:**
   ```
   WebSearch(query: "<task> <model_type> architecture improvement <current_year-1> <current_year>")
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
   WebSearch(query: "arxiv <task> <model_type> <current_year-1> <current_year> improvement")
   ```

Run at least 3 searches. For each promising result, use WebFetch to get more details.

### Tabular ML search queries (for scikit-learn, XGBoost, LightGBM)

When the model is tree-based or ensemble, replace or supplement the DL-centric queries above with:

6. **Feature engineering:**
   ```
   WebSearch(query: "<task> feature engineering tabular data <current_year-1> <current_year>")
   ```
7. **Ensemble methods:**
   ```
   WebSearch(query: "<model_type> ensemble stacking blending tabular <task>")
   ```
8. **Tree model tuning:**
   ```
   WebSearch(query: "<model_type> hyperparameter tuning best practices <task>")
   ```
9. **Feature selection:**
   ```
   WebSearch(query: "feature selection <task> tabular data importance permutation")
   ```

The DL queries (architecture improvements, loss functions) are unlikely useful for tree-based models. Focus on data preprocessing, feature engineering, and ensemble strategies instead.

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

4. Search for reference implementations:
   - Check if the paper links to a code repository
   - Search: `WebSearch(query: "<paper_title> github implementation")`
   - If a repo is found, use WebFetch on the README to verify relevance and quality
   - Identify which source files contain the core implementation
   - Check the license (prefer permissive: MIT, Apache, BSD)
   - Decide strategy: `from_reference` if a quality repo exists (>10 stars or official, updated within 2 years, permissive license), otherwise `from_scratch`

## Step 4: Rank Proposals

Score each proposal on three axes:
- **Expected impact** (1-10): How much improvement is likely?
- **Feasibility** (1-10): How easy is it to implement?
- **Confidence** (1-10): How confident are we in the expected outcome?

Priority score = (impact * confidence) / (11 - min(feasibility, 10))

Note: Clamp feasibility to [1, 10] range to prevent division by zero when feasibility=11.

Sort proposals by priority score, highest first.

### User Paper Priority Bonus

If a proposal originated from a user-provided paper (`user_papers` input):
- Add +2 to `confidence` score (capped at 10) before computing priority
- Rationale: user identified the paper as relevant — strong signal
- Still apply feasibility and impact scoring objectively

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
- **Implementation strategy:** from_scratch | from_reference
- **Reference repo:** [GitHub URL] (only for from_reference)
- **Reference files:** `path/to/relevant.py`, `path/to/other.py` (only for from_reference)

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
2. **Check recency:** Prefer papers from the last 2-3 years over older ones
3. **Look for code:** Papers with code repos are much more implementable
4. **Check benchmarks:** Make sure reported improvements are on comparable tasks/datasets
5. **Combine techniques:** Some improvements stack (e.g., better loss + better scheduler)
6. **Be honest about confidence:** If a technique seems promising but risky, say so

## Error Handling

- **WebSearch fails:** Try alternative search terms, or ask user for specific papers
- **Paper behind paywall:** Note the limitation, extract what's available from abstract
- **No relevant results:** Broaden search terms, try related tasks/model types
- **Contradictory findings:** Note both perspectives, let the user decide

## Error Tracking

At the following points, log an error event using the error tracker:

### When WebSearch returns no useful results for a query:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"warning","source":"research","message":"No relevant results for query: <query>","phase":4,"context":{"query":"<query>","search_type":"web"}}'
```

### When all searches fail to produce any actionable proposals:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"critical","source":"research","message":"No actionable proposals found after <N> searches","phase":4,"context":{"searches_attempted":<N>}}'
```

### When a reference repo URL is unreachable or fails quality checks:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"warning","source":"research","message":"Reference repo unavailable: <url>","phase":4,"context":{"url":"<url>","proposal_name":"<name>"}}'
```

### When a paper is behind a paywall (info only):
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"info","source":"research","message":"Paper behind paywall, only abstract available: <title>","phase":4,"context":{"paper_title":"<title>"}}'
```
