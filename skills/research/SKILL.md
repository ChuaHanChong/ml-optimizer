---
name: research
description: "Research ML optimization techniques via web search, paper analysis, or LLM knowledge. Extracts actionable proposals with implementation details, expected impact, and complexity ratings. Use when: need to find new techniques for improving an ML model."
user-invocable: false
---

# ML Research Agent

Critically evaluate paper claims, assess feasibility/compatibility, and reason through confidence scoring before ranking proposals.

Search and analyze ML techniques that could improve the target model. Extract actionable, implementable proposals — not paper summaries.

> **Path convention:** paths written as `<exp_root>/...` refer to the `exp_root` dispatch parameter. The plugin does not hardcode the output directory name.

## Reference

- Paper analysis guide: `${CLAUDE_SKILL_DIR}/references/paper-analysis.md`. Read this FIRST for the extraction framework.

## Inputs Expected

From the orchestrator:
- `model_type`: Type of model (e.g., "diffusion model", "ResNet", "transformer")
- `task`: What the model does (e.g., "image restoration", "super-resolution", "classification")
- `current_metrics`: Current performance numbers
- `problem_description`: What needs improvement
- `user_papers`: Optional list of paper URLs or files provided by the user
- `exp_root`: Path to <exp_root>/ directory (for error logging)
- `source`: `"web"` (default), `"knowledge"`, or `"both"` — how proposals are generated:
  - `"web"`: web search + paper analysis (Phase 5)
  - `"knowledge"`: LLM proposes methods from its own training knowledge (Phase 7 method proposals)
  - `"both"`: web search first, then supplement with knowledge-based proposals
- `scope_level`: `"training"` (default), `"architecture"`, or `"full"` — constrains what categories of changes can be proposed:
  - `"training"`: Optimizer, LR schedulers, warmup strategies, gradient clipping/accumulation, mixed precision, loss functions, weight decay, data augmentation, regularization (dropout, label smoothing), EMA. **RL:** algorithm HPs, reward shaping, observation normalization, entropy/GAE schedules
  - `"architecture"`: All of `training` + attention mechanism changes, normalization layer changes, activation function changes, block design changes, skip connection modifications. **RL:** policy/value network changes
  - `"full"`: All of `architecture` + data pipeline changes, preprocessing, tokenization, feature engineering, ensemble approaches, distillation, curriculum learning, training-free methods (pruning, quantization, sparsification), test-time adaptation (TTA, test-time augmentation), inference-time search (MCTS, beam search). **RL:** env dynamics, curriculum, domain randomization
- `output_path`: where to write findings (default: `<exp_root>/reports/research-findings.md`). From Phase 7, use `<exp_root>/reports/research-findings-method-proposals-iter<N>.md`

## Step 1: Analyze User-Provided Papers (if any)

> **Goal check:** respect scope_level constraints and dead-end techniques from the optimization goals. Do NOT propose architecture changes when scope is "training", or re-propose dead-end techniques.

If the user provided papers or URLs:

1. For each URL, retrieve content with the most appropriate tool:
   - **arXiv/alphaXiv URLs** (contain `arxiv.org` or `alphaxiv.org`): `mcp__alphaxiv__get_paper_content(url: "<paper_url>")` for a structured summary. If it lacks implementation details, follow up with `fullText: true` or `mcp__alphaxiv__answer_pdf_queries` with targeted questions.
   - **Other URLs** (blog posts, conference pages, non-arXiv PDFs): `WebFetch(url: "<paper_url>")`.
   - **If alphaxiv fails on an arXiv URL:** fall back to `WebFetch(url: "<paper_url>")`.

2. For local files, use Read.

3. Apply the paper analysis framework from `${CLAUDE_SKILL_DIR}/references/paper-analysis.md`: extract core technique, determine implementation details, assess expected impact, identify risks.

## Step 1.1: Check for Existing Research and Dead Ends (Deduplication)

Before searching, check existing findings and dead ends:

1. Check ALL existing findings files in `<exp_root>/reports/`:
   - `research-findings.md` (Phase 5 web-based proposals)
   - `research-findings-method-proposals-iter*.md` (all Phase 7 research dispatches — pre-loop, mid-loop method-proposal, cadence, code_evolution fallback, and stuck-protocol rounds)
   - the current `output_path` itself (if it exists from a previous run)
2. Check the dead-end catalog (techniques conclusively unpromising):
   ```bash
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> dead-end list
   ```
3. If any exist, extract all previously proposed technique names AND dead-end names.
4. When generating new proposals, exclude techniques already proposed OR in the dead-end catalog. This avoids re-proposing the same techniques and wasting budget on proven dead ends.

**Fuzzy matching rules** when comparing a new technique name against previous names:
- Normalize both: lowercase, strip trailing "loss", "function", "scheduler", "strategy", "method", "technique"
- Substring containment: if either normalized name contains the other, treat as duplicate (e.g., "perceptual loss" matches "vgg perceptual loss")
- Common abbreviations: "lr" ↔ "learning rate", "bn" ↔ "batch normalization", "wd" ↔ "weight decay"
- If in doubt (>70% word overlap), treat as duplicate and skip

## Step 2: Web Search for Techniques

Construct targeted searches for the model type and task.

### Search queries to run in parallel (adapt to the specific model/task):

**Run ALL applicable searches in parallel** — issue multiple WebSearch calls in a single message; they are independent queries.

**Date handling:** use the current year dynamically, never hardcode year strings. Use `<current_year-1> <current_year>` in queries (e.g., current year 2026 → search "2025 2026").

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

Issue all applicable searches simultaneously in one message. After they return, process each result; for promising ones use WebFetch for details — WebFetch calls for different URLs also parallelize.

### alphaxiv academic paper searches (run IN PARALLEL with WebSearch queries above)

**Issue these 3 alphaxiv searches in the SAME parallel batch as the WebSearch calls above.** They add academic-paper coverage complementary to WebSearch's blog/tutorial/GitHub coverage.

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
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"warning","source":"research","message":"alphaxiv MCP unavailable — proceeding with WebSearch only","phase":5,"context":{"tool":"alphaxiv"}}'
```

**Deduplication across sources:** After all parallel searches return, merge results. Remove duplicates by matching paper titles (case-insensitive, strip leading "a/an/the"). Prefer alphaxiv results when the same paper appears in both (alphaxiv provides structured content access). Retain unique WebSearch results (blog posts, tutorials, GitHub repos that alphaxiv wouldn't find).

### alphaxiv Use Case Workflows

Apply the appropriate workflow based on the research context:

**1. Comprehensive Paper Search (default):** the parallel search pattern above. Each tool covers different blind spots — embedding search for conceptual/semantic matches, full-text search for exact keyword hits, agentic retrieval for autonomous cross-angle exploration. Default for all `source: "web"` or `"both"` invocations.

**2. Deep Research** — when initial results are thin (fewer than 3 actionable papers):
- Re-run `mcp__alphaxiv__embedding_similarity_search` and `mcp__alphaxiv__full_text_papers_search` with **different query angles** (broader terms, related techniques, adjacent domains, different terminology)
- `mcp__alphaxiv__get_paper_content` to deep-read the most promising papers
- `mcp__alphaxiv__read_files_from_github_repository` to verify implementations exist and are viable

**3. Literature Review** — when the task needs broad coverage across many related techniques:
- Run the initial parallel search
- Re-run `embedding_similarity_search` and `full_text_papers_search` with varied/refined queries to fill gaps (e.g., each technique category separately)
- `mcp__alphaxiv__answer_pdf_queries` to batch-extract key info from all found papers in one call (more efficient than reading each separately)
- Synthesize across papers for common patterns, contradictions, and consensus

**4. Code Analysis** — when evaluating `from_reference` proposals (see also Step 3 sub-step 4):
- Locate the paper via search tools; extract the GitHub URL from results or paper content
- `mcp__alphaxiv__read_files_from_github_repository(githubUrl, path: "/")` to explore repo structure (lightweight first pass to locate the implementation)
- **Code-graph feasibility check (GitNexus — REQUIRED for any `from_reference` proposal):** GitNexus is a HARD PREREQUISITE verified at Phase 2 — every candidate GitHub repo informing a `from_reference` proposal MUST be cloned, indexed, and understood through the code graph before you set `implementation_strategy`. Use the graph to judge whether the technique's implementation is isolated (favor `from_reference`) or entangled across the codebase (favor `from_scratch`). There is NO grep/alphaxiv-only substitute. Index through the wrapper — it runs `gitnexus analyze <repo_path> --index-only`, which keeps the cloned repo uncontaminated (does NOT inject a GitNexus section into its CLAUDE.md/AGENTS.md, does NOT install `.claude/` skills) — guarded by an `is-indexed` check (the wrapper also skips internally):
  ```bash
  python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py available || { echo "HALT: gitnexus unavailable — was guaranteed by Phase 2"; exit 1; }
  python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py is-indexed <repo_path> || \
    python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py index <repo_path>
  ```
  After indexing, query with `mcp__gitnexus__context`/`mcp__gitnexus__query`/`mcp__gitnexus__impact` to assess feasibility and set `implementation_strategy`. **HARD ERROR (not a fallback):** if `available` exits non-zero or indexing returns `success: false`, **halt** with install/repair guidance: `npm install -g gitnexus && gitnexus setup` (`gitnexus setup` auto-registers the gitnexus MCP server for Claude Code; manual MCP-registration fallback: `claude mcp add --transport stdio --scope user gitnexus gitnexus mcp`). gitnexus was guaranteed by Phase 2, so absence here is an error state — do NOT route around it with alphaxiv-only exploration. Do not commit any generated `.gitnexus/` artifact (the wrapper auto-adds it to the repo's git exclude).

  **MCP-query-failure recovery:** querying the code graph is MCP-only by design. If a `mcp__gitnexus__*` call FAILS *after* a successful index, it is a HARD ERROR — NO grep fallback, NO gitnexus-CLI query fallback. Recovery: (1) register the gitnexus MCP server — `gitnexus setup` (or manual `claude mcp add --transport stdio --scope user gitnexus gitnexus mcp`); (2) MCP tools load at session start, so restart the Claude Code session for a freshly-registered server; (3) retry the query. If it still fails, halt and surface to the user.

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

When the model category is RL (gym, gymnasium, stable-baselines3, rsl_rl, IsaacLab, skrl, brax, etc.):

15. **Policy optimization:**
    ```
    WebSearch(query: "<task> policy optimization technique <model_type> <current_year-1> <current_year>")
    ```
16. **Exploration/reward:**
    ```
    WebSearch(query: "<task> exploration strategy reward shaping <model_type>")
    ```
17. **Sim-to-real & curriculum:**
    ```
    WebSearch(query: "<task> domain randomization sim-to-real curriculum learning <model_type>")
    ```
18. **Algorithm-specific tricks:**
    ```
    WebSearch(query: "<model_type> PPO SAC implementation tricks training stability <current_year-1> <current_year>")
    ```

### Time Series search queries

When the task involves forecasting, anomaly detection on temporal data, or sequence prediction:

19. **Temporal methods:**
    ```
    WebSearch(query: "<task> temporal encoding patching strategy time series <current_year-1> <current_year>")
    ```
20. **Forecasting architectures:**
    ```
    WebSearch(query: "<task> forecasting model improvement <model_type> state-of-the-art")
    ```

### VLA / Imitation Learning search queries

When the model is a vision-language-action policy or trained from demonstrations (behavior cloning, LeRobot, robomimic, diffusion policy, ACT):

21. **Policy architectures & action chunking:**
    ```
    WebSearch(query: "<task> action chunking diffusion policy imitation learning <current_year-1> <current_year>")
    ```
22. **Demonstration data strategies:**
    ```
    WebSearch(query: "<task> demonstration augmentation co-training data mixing behavior cloning")
    ```

Issue only the domain-specific queries relevant to the detected model type/task, in parallel with the general DL or tabular queries.

## Step 2 Alternative: Knowledge-Based Proposals (when `source` is `"knowledge"`)

When `source` is `"knowledge"`, **skip Steps 1, 2, and 3 entirely** — do NOT use WebSearch or WebFetch. Propose methods directly from the LLM's own training knowledge.

### Process:

1. **Analyze the model context:** model type, task, framework, current metrics, problem description.

2. **Generate proposals using structured ideation** (diverge → converge → refine):

   **Phase A — Diverge (generate 10-15 candidate ideas):**
   Apply all 6 complementary lenses:

   | Lens | Question |
   |------|----------|
   | **Problem-First** | What specific training pathology am I seeing (slow convergence, overfitting, gradient instability)? What techniques directly target it? |
   | **Analogical Reasoning** | What works for similar tasks/architectures in adjacent domains? (e.g., NLP attention tricks for vision, vision augmentation for audio) |
   | **What Changed Recently** | What new techniques emerged in the last 1-2 years for this model family? (e.g., cosine annealing → warm restarts → one-cycle, BatchNorm → LayerNorm → RMSNorm) |
   | **Constraint Manipulation** | What assumptions does the current approach make that could be relaxed? (fixed LR → adaptive, uniform sampling → curriculum, single loss → multi-loss) |
   | **Negation/Inversion** | What if we do the opposite of the current approach? (large batch → small batch + accumulation, heavy augmentation → minimal + regularization) |
   | **Composition/Decomposition** | Can two simple techniques compound for bigger improvement? Can a complex change decompose into independent testable steps? |

   Only propose techniques within the `scope_level`:

   | Scope Level | Allowed Categories |
   |---|---|
   | `training` | Optimizer changes (Adam → AdamW, LAMB, etc.), LR schedulers (cosine, one-cycle, warm restarts), warmup strategies, gradient clipping, gradient accumulation, mixed precision, loss function changes, weight decay tuning, data augmentation, regularization (dropout, label smoothing, stochastic depth), EMA. **RL:** algorithm HPs, reward shaping, observation normalization, entropy/GAE schedules |
   | `architecture` | All of `training` + attention variants (multi-head, efficient attention), normalization changes (BatchNorm → LayerNorm/GroupNorm/RMSNorm), activation functions (ReLU → SiLU/GELU/Swish), block design changes, skip/residual connection modifications, channel/dimension scaling. **RL:** policy/value network changes |
   | `full` | All of `architecture` + data pipeline changes, preprocessing, tokenization changes, feature engineering, ensemble approaches, distillation, curriculum learning, different training paradigms. **RL:** env dynamics, curriculum, domain randomization |

   **Phase B — Converge (filter to 3-7 proposals):**
   - Eliminate ideas outside `scope_level` constraints
   - Eliminate ideas in the dead-end catalog (Step 1.1)
   - Eliminate ideas that duplicate existing proposals
   - Apply the **Two-Sentence Test**: if you can't explain the technique AND its expected benefit in two sentences, it's too vague to implement
   - Rank remaining by priority score

   **Phase C — Refine (add implementation details):**
   - For each surviving idea, specify `files_to_modify`, `implementation_steps`, `expected_improvement`
   - Cap confidence at 7/10 (knowledge-mode ceiling)

3. **Apply quality standards:**
   - Concrete implementation steps (not vague suggestions)
   - Specify files to modify and what to change
   - Cap confidence at 7/10 (unless extremely well-established, e.g., label smoothing for classification)
   - All proposals are `implementation_strategy: "from_scratch"` (no reference repo)
   - All proposals include `**Proposal source:** llm_knowledge`

4. **Apply the same ranking formula:** `(impact × confidence) / (11 - min(feasibility, 10))`

5. **Proceed to Step 4** (skip Step 3).

### When `source` is `"both"`:

Run Steps 1-3 (web search) first, then add knowledge-based proposals that don't overlap what was found. Deduplicate between web-found and knowledge-generated proposals. Mark web-found `**Proposal source:** paper`, knowledge `**Proposal source:** llm_knowledge`.

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
   - Check whether the paper links a code repository; search `WebSearch(query: "<paper_title> github implementation")`
   - If a GitHub repo is found:
     a. **Explore with alphaxiv first** (structured, efficient):
        ```
        mcp__alphaxiv__read_files_from_github_repository(githubUrl: "https://github.com/org/repo", path: "/")
        ```
        Returns the full file tree AND top-level file contents (README, LICENSE, setup.py) in one call. Use it to read the README for relevance/quality, check the LICENSE (prefer permissive: MIT, Apache, BSD), and identify which source dirs contain the core implementation.
     b. **Drill into relevant directories:**
        ```
        mcp__alphaxiv__read_files_from_github_repository(githubUrl: "https://github.com/org/repo", path: "src/models/")
        ```
        Fetches all files in the directory in parallel.
     c. **Fallback:** if `read_files_from_github_repository` fails, fall back to `WebFetch` on the repo's README URL.
   - Repo quality gate: >10 stars or official, updated within 2 years, permissive license
   - Decide strategy: `from_reference` if the gate passes, else `from_scratch`

## Step 4: Rank Proposals

Score each proposal on three axes:
- **Expected impact** (1-10): How much improvement is likely?
- **Feasibility** (1-10): How easy is it to implement?
- **Confidence** (1-10): How confident are we in the expected outcome?

Priority score = (impact * confidence) / (11 - min(feasibility, 10))

Note: Clamp feasibility to [1, 10] range to prevent division by zero when feasibility=11.

Sort proposals by priority score, highest first.

### Small Dataset Awareness

Check the dataset size from `prerequisites.json` or the training script.

- **Skip this check for online RL** (`model_category = "rl"` with `dataset.format_detected = "rl_environment"`): there is no fixed dataset — data is generated by environment interaction, so "dataset size" is not meaningful.
- **Demonstration datasets** (lerobot, rlds, robomimic, zarr formats): count **transitions** (steps), not episodes — 50 episodes of 400 steps is 20,000 samples, not 50.
- **If the dataset has fewer than 5,000 training samples** (transitions for demonstration data), adapt your search strategy: heavy augmentation and regularization often underperform on small data.
  - **Supervised models:** search for low-data techniques — transfer learning, pre-trained model fine-tuning, few-shot learning, adapters (LoRA, prefix tuning), prompt tuning, synthetic data generation, semi-supervised methods, self-training, and meta-learning approaches.
  - **Imitation learning / behavior cloning:** the supervised low-data list does not transfer — search instead for BC-appropriate low-data techniques: action chunking, observation augmentation, pretrained visual encoders, and co-training / data mixing with related demonstration datasets.

### User Paper Priority Bonus

If a proposal originated from a user-provided paper (`user_papers` input):
- Add +2 to `confidence` (capped at 10) before computing priority — the user flagged it relevant, a strong signal
- Still apply feasibility and impact scoring objectively

## Step 5: Write Research Findings

Write to the path specified by `output_path` (default: `<exp_root>/reports/research-findings.md`):

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
- **Search space** *(hp_only proposals — REQUIRED)*: `[{"param": "<hp name>", "range": [<low>, <high>], "scale": "log|linear|categorical", "source": "<paper/URL the range came from>"}]` — structured `{param, range, scale, source}` priors that hp-tune seeds directly; every entry must cite its source
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

**HP priors on `hp_only` proposals:** For a single-shot, non-fragmented research-agent invocation, every `hp_only` proposal MUST carry the **Search space** field — a JSON array of `{param, range, scale, source}` entries with citations, as PRIMARY search-space priors for the hp-tune skill. Note: the Phase 5 fan-out/vet/synthesize pipeline's schemas (CANDIDATE_SCHEMA, the Vet stage) do not currently ask for or preserve a `search_space` field, so this requirement does not apply end-to-end on that path today. Validate the array shape with `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/schema_validator.py relay research_to_tuning '{"search_space": [...]}'`.

## Step 5.1: Initialize Research Agenda

After writing findings, initialize the research agenda from the proposals — a living document the analyze skill updates after each batch.

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> agenda init '<ideas_json>'
```

Where `<ideas_json>` is a JSON array of ideas derived from the proposals:
```json
[
  {"id": "proposal-1-slug", "name": "<Proposal 1 name>", "priority": <priority_score>, "source": "<paper|llm_knowledge>", "scope": "<training|architecture|full>", "search_space": [{"param": "<hp>", "range": [<low>, <high>], "scale": "<log|linear|categorical>", "source": "<citation>"}]},
  {"id": "proposal-2-slug", "name": "<Proposal 2 name>", "priority": <priority_score>, "source": "<paper|llm_knowledge>", "scope": "<scope_level>"}
]
```

Use the proposal's priority score as the initial priority. `id` is a URL-safe slug of the proposal name (lowercase, hyphens, no spaces).

For `hp_only` proposals, copy the proposal's **Search space** array into the agenda entry's `search_space` key (same `{param, range, scale, source}` shape) — hp-tune reads the agenda to seed research-derived priors.

If a research agenda already exists (e.g., mid-loop research), use `agenda add` to append without overwriting existing ideas.

**CRITICAL — Output verification:** after writing, verify the file exists at `output_path` (Read tool). If missing or empty, re-write it — the orchestrator depends on this file existing at the exact `output_path`.

## Step 6: Structured Return

Return the structured schema fields specified in your dispatch prompt (e.g. `findings_path`, `agenda_initialized`, and the ranked `proposals[]` array) — each Phase 5/7 workflow dispatch site defines its own exact required return schema; there is no separate freeform summary to produce.

## Tips for Effective Research

1. **Be specific in searches:** "diffusion model image restoration perceptual loss" beats "ML improvement"
2. **Check recency:** prefer papers from the last 2-3 years
3. **Look for code:** papers with code repos are far more implementable
4. **Check benchmarks:** ensure reported improvements are on comparable tasks/datasets
5. **Combine techniques:** some improvements stack (e.g., better loss + better scheduler)
6. **Be honest about confidence:** if a technique is promising but risky, say so

## Error Handling

- **WebSearch fails:** try alternative search terms, or note the gap in research-findings.md for the user to see at the post-research checkpoint (research-agent has no AskUserQuestion tool and cannot ask mid-dispatch)
- **Paper behind paywall:** note the limitation, extract what's available from the abstract
- **No relevant results:** broaden search terms, try related tasks/model types
- **Contradictory findings:** note both perspectives, let the user decide
- **alphaxiv MCP unavailable:** proceed with WebSearch/WebFetch only; log to error tracker, severity "warning". Expected when the alphaxiv MCP server is not installed/running.
- **alphaxiv search returns no results:** rely on WebSearch for that query. Do NOT retry the same query with alphaxiv — move on.
- **`get_paper_content` fails on a valid arXiv URL:** fall back to `WebFetch(url: "<arxiv_url>")`.
- **`read_files_from_github_repository` fails:** fall back to `WebFetch` on the repo's README URL.
- **`answer_pdf_queries` times out:** fall back to `get_paper_content(fullText: true)` and extract answers manually.

## Error Tracking

Log an error event via the error tracker at these points:

### When WebSearch returns no useful results for a query:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"warning","source":"research","message":"No relevant results for query: <query>","phase":5,"context":{"query":"<query>","search_type":"web"}}'
```

### When all searches fail to produce any actionable proposals:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"critical","source":"research","message":"No actionable proposals found after <N> searches","phase":5,"context":{"searches_attempted":<N>}}'
```

### When a reference repo URL is unreachable or fails quality checks:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"warning","source":"research","message":"Reference repo unavailable: <url>","phase":5,"context":{"url":"<url>","proposal_name":"<name>"}}'
```

### When a paper is behind a paywall (info only):
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"info","source":"research","message":"Paper behind paywall, only abstract available: <title>","phase":5,"context":{"paper_title":"<title>"}}'
```

### When alphaxiv MCP tools are unavailable (all alphaxiv searches fail):
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"research_failure","severity":"warning","source":"research","message":"alphaxiv MCP tools unavailable — using WebSearch/WebFetch only","phase":5,"context":{"tool":"alphaxiv","fallback":"websearch"}}'
```

## Domain Randomization and Curriculum Proposals

**Domain-randomization priors.** When a paper reports randomization ranges (friction, mass, damping, lighting, sensor noise, latency), convert each into a `search_space` entry pair rather than a single range entry:

```json
[
  {"param": "friction_center", "range": [0.6, 1.4], "scale": "linear", "source": "<paper title / URL>"},
  {"param": "friction_width", "range": [0.0, 0.8], "scale": "linear", "source": "<paper title / URL>"}
]
```

The effective randomization range is `center ± width/2`. Two scalars keep the entries proposable by hp-tune and make an inverted range impossible. `width` starting at `0.0` lets the search discover that a parameter should not be randomized at all. Only emit these when `scope_level` is `"full"` — at `"training"` or `"architecture"` scope they are a scope violation.

**Curriculum.** A curriculum is a schedule over training, not a config value, so it is a `type: "code_change"` proposal like any other — never an `hp_only` one. Propose it as the code that ramps a randomization width (or task difficulty) over training, naming the ramp's own parameters so they become tunable afterwards:

- `files_to_modify`: the training loop and the environment-construction site
- the implementation should expose the ramp's shape and endpoints as config values (e.g. `friction_width_final`, `curriculum_warmup_frac`)

Do NOT propose a curriculum as a bare hyperparameter. A schedule flag that the training code does not read produces a run labelled "curriculum" that ran no curriculum — a result that cannot be trusted and is worse than no result. Once the ramp is implemented on its branch, its parameters are ordinary HPs the tuning agent handles with no further special casing.
