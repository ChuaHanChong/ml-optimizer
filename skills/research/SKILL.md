---
name: research
description: "Research ML optimization techniques via web search, paper analysis, or LLM knowledge. Extracts actionable proposals with implementation details, expected impact, and complexity ratings. Use when: need to find new techniques for improving an ML model."
disable-model-invocation: true
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
- `source`: One of `"web"` (default), `"knowledge"`, or `"both"`. Controls how proposals are generated:
  - `"web"`: Current behavior — web search + paper analysis (Phase 5)
  - `"knowledge"`: LLM proposes methods from its own training knowledge (Phase 7 method proposals)
  - `"both"`: Web search first, then supplement with knowledge-based proposals
- `scope_level`: One of `"training"` (default), `"architecture"`, or `"full"`. Constrains what categories of changes can be proposed:
  - `"training"`: Optimizer, LR schedulers, warmup strategies, gradient clipping/accumulation, mixed precision, loss functions, weight decay, data augmentation, regularization (dropout, label smoothing), EMA
  - `"architecture"`: All of `training` + attention mechanism changes, normalization layer changes, activation function changes, block design changes, skip connection modifications
  - `"full"`: All of `architecture` + data pipeline changes, preprocessing, tokenization, feature engineering, ensemble approaches, distillation, curriculum learning, training-free methods (pruning, quantization, sparsification), test-time adaptation (TTA, test-time augmentation), inference-time search (MCTS, beam search)
- `output_path`: Where to write findings (default: `experiments/reports/research-findings.md`). When called from Phase 7, use `experiments/reports/research-findings-method-proposals.md`

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

## Step 1.1: Check for Existing Research (Deduplication)

Before searching, check for existing findings files:

1. Check `experiments/reports/research-findings.md` (Phase 5 web-based proposals)
2. Check `experiments/reports/research-findings-method-proposals*.md` (Phase 7 method proposals)
3. If any exist, read them and extract all previously proposed technique names
4. When generating new proposals, exclude techniques that were already proposed
5. This prevents re-proposing the same techniques on subsequent optimization runs

**Fuzzy matching rules:** When comparing a new technique name against previously proposed names:
- Normalize both names: lowercase, strip trailing "loss", "function", "scheduler", "strategy", "method", "technique"
- Check substring containment: if either normalized name contains the other, treat as duplicate (e.g., "perceptual loss" matches "vgg perceptual loss")
- Check common abbreviations: "lr" ↔ "learning rate", "bn" ↔ "batch normalization", "wd" ↔ "weight decay"
- If in doubt (>70% word overlap), treat as duplicate and skip

## Step 2: Web Search for Techniques

Construct targeted searches based on the model type and task:

### Search queries to run in parallel (adapt to the specific model/task):

**Run ALL applicable searches in parallel** by issuing multiple WebSearch tool calls in a single message. Do not wait for one search to complete before starting the next — they are independent queries.

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

Issue all applicable searches simultaneously in a single message. After all parallel searches return, process results from each. For each promising result, use WebFetch to get more details — WebFetch calls for different URLs can also be issued in parallel.

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

The DL queries (architecture improvements, loss functions) are unlikely useful for tree-based models. Issue only the tabular-specific searches (6-9) in parallel for these models.

### NLP/LLM search queries (for transformer-based text models)

When the model processes text (NLP, LLM, text classification, NER, machine translation):

10. **Attention/architecture:**
    ```
    WebSearch(query: "<model_type> attention mechanism improvement <task> <current_year-1> <current_year>")
    ```
11. **Fine-tuning techniques:**
    ```
    WebSearch(query: "<task> LoRA adapter PEFT efficient fine-tuning <model_type>")
    ```
12. **Tokenization/embeddings:**
    ```
    WebSearch(query: "<task> tokenization position embedding improvement transformer")
    ```

### Computer Vision search queries (for detection, segmentation, super-resolution)

When the task is object detection, segmentation, super-resolution, or pose estimation (not just classification):

13. **Task-specific architectures:**
    ```
    WebSearch(query: "<task> <model_type> architecture state-of-the-art <current_year-1> <current_year>")
    ```
14. **Data augmentation:**
    ```
    WebSearch(query: "<task> data augmentation strategy <model_type> improvement")
    ```

### Reinforcement Learning search queries

When the model category is RL (gym, gymnasium, stable-baselines3, etc.):

15. **Policy optimization:**
    ```
    WebSearch(query: "<task> policy optimization technique <model_type> <current_year-1> <current_year>")
    ```
16. **Exploration/reward:**
    ```
    WebSearch(query: "<task> exploration strategy reward shaping <model_type>")
    ```

### Time Series search queries

When the task involves forecasting, anomaly detection on temporal data, or sequence prediction:

17. **Temporal methods:**
    ```
    WebSearch(query: "<task> temporal encoding patching strategy time series <current_year-1> <current_year>")
    ```
18. **Forecasting architectures:**
    ```
    WebSearch(query: "<task> forecasting model improvement <model_type> state-of-the-art")
    ```

Issue only the domain-specific queries relevant to the detected model type/task, in parallel with the general DL or tabular queries.

## Step 2 Alternative: Knowledge-Based Proposals (when `source` is `"knowledge"`)

When `source` is `"knowledge"`, **skip Steps 1, 2, and 3 entirely** — do NOT use WebSearch or WebFetch. Instead, propose methods directly from the LLM's own training knowledge.

### Process:

1. **Analyze the model context:** Consider the model type, task, framework, current metrics, and problem description.

2. **Generate proposals within scope constraints:** Only propose techniques within the `scope_level`:

   | Scope Level | Allowed Categories |
   |---|---|
   | `training` | Optimizer changes (Adam → AdamW, LAMB, etc.), LR schedulers (cosine, one-cycle, warm restarts), warmup strategies, gradient clipping, gradient accumulation, mixed precision, loss function changes, weight decay tuning, data augmentation, regularization (dropout, label smoothing, stochastic depth), EMA |
   | `architecture` | All of `training` + attention variants (multi-head, efficient attention), normalization changes (BatchNorm → LayerNorm/GroupNorm/RMSNorm), activation functions (ReLU → SiLU/GELU/Swish), block design changes, skip/residual connection modifications, channel/dimension scaling |
   | `full` | All of `architecture` + data pipeline changes, preprocessing, tokenization changes, feature engineering, ensemble approaches, distillation, curriculum learning, different training paradigms |

3. **Apply quality standards:**
   - Each proposal must have concrete implementation steps (not vague suggestions)
   - Each proposal must specify files to modify and what to change
   - Cap confidence scores at 7/10 maximum (unless the technique is extremely well-established, e.g., label smoothing for classification)
   - All proposals are `implementation_strategy: "from_scratch"` (no reference repo)
   - All proposals must include `**Proposal source:** llm_knowledge`

4. **Apply the same ranking formula:** `(impact × confidence) / (11 - min(feasibility, 10))`

5. **Proceed to Step 4** (skip Step 3).

### When `source` is `"both"`:

Run Steps 1-3 (web search) first, then supplement with knowledge-based proposals that don't overlap with what was found. Apply deduplication between web-found and knowledge-generated proposals. Mark web-found proposals with `**Proposal source:** paper` and knowledge proposals with `**Proposal source:** llm_knowledge`.

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

Write to the path specified by `output_path` (default: `experiments/reports/research-findings.md`):

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
- **Proposal source:** paper | llm_knowledge
- **Type:** code_change | hp_only
- **Source:** [Paper title and URL, or "LLM knowledge" for knowledge-mode]
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
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"warning","source":"research","message":"No relevant results for query: <query>","phase":5,"context":{"query":"<query>","search_type":"web"}}'
```

### When all searches fail to produce any actionable proposals:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"critical","source":"research","message":"No actionable proposals found after <N> searches","phase":5,"context":{"searches_attempted":<N>}}'
```

### When a reference repo URL is unreachable or fails quality checks:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"warning","source":"research","message":"Reference repo unavailable: <url>","phase":5,"context":{"url":"<url>","proposal_name":"<name>"}}'
```

### When a paper is behind a paywall (info only):
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"info","source":"research","message":"Paper behind paywall, only abstract available: <title>","phase":5,"context":{"paper_title":"<title>"}}'
```
