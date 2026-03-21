---
name: research
description: "Research ML optimization techniques via web search, paper analysis, or LLM knowledge. Extracts actionable proposals with implementation details, expected impact, and complexity ratings. Use when: need to find new techniques for improving an ML model."
disable-model-invocation: true
user-invocable: false
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

> **Goal check:** Respect scope_level constraints and dead-end techniques from the optimization goals. Do NOT propose architecture changes when scope is "training" or re-propose dead-end techniques.

If the user provided papers or URLs:

1. For each URL, retrieve the content using the most appropriate tool:
   - **arXiv/alphaXiv URLs** (contains `arxiv.org` or `alphaxiv.org`): Use `mcp__alphaxiv__get_paper_content` for a structured summary:
     ```
     mcp__alphaxiv__get_paper_content(url: "<paper_url>")
     ```
     If the summary lacks implementation details, follow up with `fullText: true` or use `mcp__alphaxiv__answer_pdf_queries` with targeted questions.
   - **Other URLs** (blog posts, conference pages, non-arXiv PDFs): Use WebFetch:
     ```
     WebFetch(url: "<paper_url>")
     ```
   - **If alphaxiv tool fails on an arXiv URL:** Fall back to `WebFetch(url: "<paper_url>")`.

2. For local files, use Read to read them

3. Apply the paper analysis framework from `references/paper-analysis.md`:
   - Extract core technique
   - Determine implementation details
   - Assess expected impact
   - Identify risks

## Step 1.1: Check for Existing Research and Dead Ends (Deduplication)

Before searching, check for existing findings and dead ends:

1. Check `experiments/reports/research-findings.md` (Phase 5 web-based proposals)
2. Check `experiments/reports/research-findings-method-proposals*.md` (Phase 7 method proposals)
3. **Check dead-end catalog** — techniques conclusively shown to be unpromising:
   ```bash
   python3 scripts/error_tracker.py <exp_root> dead-end list
   ```
4. If any exist, read them and extract all previously proposed technique names AND dead-end technique names
5. When generating new proposals, exclude techniques that were already proposed OR are in the dead-end catalog
6. This prevents re-proposing the same techniques and avoids wasting budget on proven dead ends

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

### alphaxiv academic paper searches (run IN PARALLEL with WebSearch queries above)

**Additionally, issue the following 3 alphaxiv searches in the SAME parallel batch as the WebSearch calls above.** These provide academic paper coverage complementary to WebSearch's blog/tutorial/GitHub coverage.

**a. Semantic search** (2-3 sentence descriptive query):
```
mcp__alphaxiv__embedding_similarity_search(query: "Research on improving <task> performance for <model_type> models. Papers covering <relevant_techniques>, <training_strategies>, and optimization methods for <domain>. Include recent work on <specific_improvement_areas> and efficiency improvements.")
```

**b. Keyword search** (3-4 short terms, no quotes):
```
mcp__alphaxiv__full_text_papers_search(query: "<model_type> <task> optimization improvement")
```

**c. Agentic retrieval** (natural language question, high recall — MUST run in parallel with a and b, never alone):
```
mcp__alphaxiv__agentic_paper_retrieval(query: "What are the most effective recent techniques for improving <task> performance in <model_type> models, including <relevant_categories>?")
```

**Fallback:** If ALL alphaxiv searches fail (MCP unavailable), proceed with WebSearch results only. Log the failure:
```bash
python3 scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"warning","source":"research","message":"alphaxiv MCP unavailable — proceeding with WebSearch only","phase":5,"context":{"tool":"alphaxiv"}}'
```

**Deduplication across sources:** After all parallel searches return, merge results. Remove duplicates by matching paper titles (case-insensitive, strip leading "a/an/the"). Prefer alphaxiv results when the same paper appears in both (alphaxiv provides structured content access). Retain unique WebSearch results (blog posts, tutorials, GitHub repos that alphaxiv wouldn't find).

### alphaxiv Use Case Workflows

Apply the appropriate workflow based on the research context:

**1. Comprehensive Paper Search (default):** The parallel search pattern above. Each tool covers different blind spots — embedding search for conceptual/semantic matches, full-text search for exact keyword hits, agentic retrieval for autonomous cross-angle exploration. This is the default for all `source: "web"` or `"both"` invocations.

**2. Deep Research** — when initial results are thin (fewer than 3 actionable papers):
- After the initial parallel search, re-run `mcp__alphaxiv__embedding_similarity_search` and `mcp__alphaxiv__full_text_papers_search` with **different query angles** (broader terms, related techniques, adjacent domains, different terminology)
- Use `mcp__alphaxiv__get_paper_content` to deep-read the most promising papers found
- Use `mcp__alphaxiv__read_files_from_github_repository` to verify implementations exist and are viable
- This ensures holistic coverage even for niche or emerging topics

**3. Literature Review** — when the task needs broad coverage across multiple related techniques:
- Run the initial parallel search
- Re-run `embedding_similarity_search` and `full_text_papers_search` with varied/refined queries to fill gaps (e.g., search for each technique category separately)
- Use `mcp__alphaxiv__answer_pdf_queries` to batch-extract key information from all found papers in a single call (more efficient than reading each paper separately)
- Synthesize findings across papers to identify common patterns, contradictions, and consensus

**4. Code Analysis** — when evaluating `from_reference` proposals (see also Step 3 sub-step 4):
- Locate the paper via search tools
- Extract the GitHub URL from search results or paper content
- Use `mcp__alphaxiv__read_files_from_github_repository(githubUrl, path: "/")` to explore repo structure
- Drill into implementation directories to identify core files and assess code quality

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

1. **Get paper content:**
   - **If the paper has an arXiv URL** (from alphaxiv search results or WebSearch): Use `mcp__alphaxiv__get_paper_content` for the structured summary first. If you need deeper implementation details, use `mcp__alphaxiv__answer_pdf_queries` with targeted questions:
     ```
     mcp__alphaxiv__answer_pdf_queries(
       urls: ["https://arxiv.org/abs/XXXX.XXXXX"],
       queries: [
         "What specific model changes does this paper propose?",
         "What are the key hyperparameters and their recommended values?",
         "What improvement was reported, on which benchmark, and what was the baseline?",
         "What are the computational costs compared to the baseline?"
       ]
     )
     ```
   - **For batch analysis** of multiple papers: Pass multiple URLs to a single `mcp__alphaxiv__answer_pdf_queries` call for efficiency.
   - **If paper has no arXiv URL** (blog post, conference page): Use `WebFetch` as before.
   - **If alphaxiv tool fails:** Fall back to `WebFetch(url: "<paper_url>")`.

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
   - If a GitHub repo is found:
     a. **Explore with alphaxiv first** (structured, efficient):
        ```
        mcp__alphaxiv__read_files_from_github_repository(githubUrl: "https://github.com/org/repo", path: "/")
        ```
        This returns the full file tree AND top-level file contents (README, LICENSE, setup.py) in one call. Use it to:
        - Read the README for relevance and quality indicators
        - Check the LICENSE file directly (prefer permissive: MIT, Apache, BSD)
        - Identify which source directories contain the core implementation
     b. **Drill into relevant directories:**
        ```
        mcp__alphaxiv__read_files_from_github_repository(githubUrl: "https://github.com/org/repo", path: "src/models/")
        ```
        Read the directory containing the core implementation — fetches all files in the directory in parallel.
     c. **Fallback:** If `read_files_from_github_repository` fails, fall back to `WebFetch` on the repo's README URL.
   - Apply the repo quality gate: >10 stars or official, updated within 2 years, permissive license
   - Decide strategy: `from_reference` if quality gate passes, otherwise `from_scratch`

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
- [Paper/URL 1]: [Key takeaway] *(alphaxiv)*
- [Paper/URL 2]: [Key takeaway] *(WebSearch)*
- [Paper/URL 3]: [Key takeaway] *(user)*
- ...

Note: Mark each source with its discovery channel — `(alphaxiv)` for papers found via alphaxiv tools, `(WebSearch)` for papers found via web search, `(user)` for user-provided papers.

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

## Step 5.1: Initialize Research Agenda

After writing the research findings, initialize the research agenda from the proposals. This creates a living document that the analyze skill updates after each batch.

```bash
python3 scripts/error_tracker.py <exp_root> agenda init '<ideas_json>'
```

Where `<ideas_json>` is a JSON array of ideas derived from the proposals:
```json
[
  {"id": "proposal-1-slug", "name": "<Proposal 1 name>", "priority": <priority_score>, "source": "<paper|llm_knowledge>", "scope": "<training|architecture|full>"},
  {"id": "proposal-2-slug", "name": "<Proposal 2 name>", "priority": <priority_score>, "source": "<paper|llm_knowledge>", "scope": "<scope_level>"}
]
```

Use the proposal's priority score (from ranking) as the initial priority. The `id` should be a URL-safe slug of the proposal name (lowercase, hyphens, no spaces).

If a research agenda already exists (e.g., mid-loop research), use `agenda add` instead to append new ideas without overwriting the existing ones.

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
- **alphaxiv MCP unavailable:** Proceed with WebSearch/WebFetch only. Log to error tracker with severity "warning". This is expected when the alphaxiv MCP server is not installed or not running.
- **alphaxiv search returns no results:** Rely on WebSearch results for that query. Do NOT retry the same query with alphaxiv — move on.
- **`get_paper_content` fails on a valid arXiv URL:** Fall back to `WebFetch(url: "<arxiv_url>")`.
- **`read_files_from_github_repository` fails:** Fall back to `WebFetch` on the repo's README URL.
- **`answer_pdf_queries` times out:** Fall back to `get_paper_content(fullText: true)` and extract answers manually from the full text.

## Error Tracking

At the following points, log an error event using the error tracker:

### When WebSearch returns no useful results for a query:
```bash
python3 scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"warning","source":"research","message":"No relevant results for query: <query>","phase":5,"context":{"query":"<query>","search_type":"web"}}'
```

### When all searches fail to produce any actionable proposals:
```bash
python3 scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"critical","source":"research","message":"No actionable proposals found after <N> searches","phase":5,"context":{"searches_attempted":<N>}}'
```

### When a reference repo URL is unreachable or fails quality checks:
```bash
python3 scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"warning","source":"research","message":"Reference repo unavailable: <url>","phase":5,"context":{"url":"<url>","proposal_name":"<name>"}}'
```

### When a paper is behind a paywall (info only):
```bash
python3 scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"info","source":"research","message":"Paper behind paywall, only abstract available: <title>","phase":5,"context":{"paper_title":"<title>"}}'
```

### When alphaxiv MCP tools are unavailable (all alphaxiv searches fail):
```bash
python3 scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"warning","source":"research","message":"alphaxiv MCP tools unavailable — using WebSearch/WebFetch only","phase":5,"context":{"tool":"alphaxiv","fallback":"websearch"}}'
```
