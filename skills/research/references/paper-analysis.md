# Paper Analysis Guide

## How to Extract Implementable Insights from ML Papers

When analyzing a paper, focus on what can be **directly implemented**, not the theoretical contribution.

## Extraction Framework

For each paper, extract:

### 1. Core Technique
- **What is it?** one-sentence description
- **What problem does it solve?** (e.g., training instability, slow convergence, poor generalization)
- **Category:** Architecture / Loss function / Training strategy / Data augmentation / Regularization / Other

### 2. Implementation Details
- **Code changes required:** which files/functions need modification?
- **Dependencies:** any new libraries?
- **Complexity estimate:**
  - **Low:** a few lines (e.g., swap loss function, add a layer)
  - **Medium:** modify a module (e.g., new attention mechanism, custom scheduler)
  - **High:** significant refactoring (e.g., new training paradigm, different architecture)
- **Reference implementation:**
  - URL (GitHub/GitLab link from paper or search)
  - Quality indicators (stars, recency, official vs community)
  - Relevant files (which contain the core implementation)
  - Framework (PyTorch, TensorFlow, JAX, etc.)
  - **Repo exploration:** `mcp__alphaxiv__read_files_from_github_repository(githubUrl, path="/")` for structured exploration — returns file tree + top-level files in one call. Drill into implementation dirs with `path: "src/"` or `path: "models/"`. Falls back to WebFetch on the README URL if alphaxiv is unavailable.
- **Implementation strategy recommendation:**
  - `from_reference`: quality repo exists, code extractable, compatible or translatable framework
  - `from_scratch`: no repo, incompatible framework needing full rewrite, or repo too entangled to extract

### 3. Expected Impact
- **What improvement does the paper report?** (quantitative if available)
- **On what benchmark/dataset?** (how comparable to our task?)
- **Conditions for improvement:** (e.g., "works best with large batch sizes", "requires pre-training")
- **Realistic expectation:** papers report best-case; expect 30-70% of reported gains

### 4. Risks and Requirements
- **Could it make things worse?** (e.g., training instability, more memory)
- **Computational cost:** more/less expensive than the current approach?
- **Compatibility:** does it work with our model architecture and framework?

## Red Flags in Papers

Be skeptical when:
- Results only on toy datasets
- No ablation study
- Improvement within standard deviation
- Method needs extensive HP tuning to work
- No code AND ambiguous method description
- Reference repo uses an incompatible framework with deep infrastructure entanglement
- Reference repo has no license (legal risk for adaptation)
- Reference repo is >3 years old with deprecated dependencies
- Preprint with no peer review AND no reference implementation
- alphaxiv summary shows conflicting claims between abstract and results

## Search Strategy

### For architecture improvements:
- Search: "[task] [model_type] architecture improvement <current_year-1> <current_year>"
- Look for: new attention mechanisms, better upsampling, efficient blocks

### For training improvements:
- Search: "[task] training strategy" or "[model_type] training tricks"
- Look for: better schedulers, curriculum learning, progressive training

### For loss function improvements:
- Search: "[task] loss function" or "perceptual loss [task]"
- Look for: new loss formulations, loss combinations, adaptive weighting

### For data improvements:
- Search: "[task] data augmentation" or "[domain] augmentation strategy"
- Look for: domain-specific augmentations, mixing strategies

### For NLP improvements:
- Search: "[task] language model training improvement <year>"
- Look for: efficient attention, better tokenization, knowledge distillation, prompt tuning

### For audio/speech improvements:
- Search: "[task] speech model training <year>"
- Look for: acoustic feature improvements, CTC alternatives, streaming architectures

### For graph learning improvements:
- Search: "graph neural network [task] improvement <year>"
- Look for: message passing alternatives, over-smoothing solutions, graph transformers

## Paper Content Extraction with alphaxiv

For papers found via any source, use alphaxiv's content tools for efficient extraction:

### For individual paper analysis:
`mcp__alphaxiv__get_paper_content(url)` gives a structured summary (~2000 tokens) — faster and more LLM-friendly than raw WebFetch. Use `fullText: true` only when the summary lacks implementation details.

### For targeted extraction across multiple papers:
`mcp__alphaxiv__answer_pdf_queries` asks specific questions about multiple papers at once:
```
mcp__alphaxiv__answer_pdf_queries(
  urls: ["<paper_1_url>", "<paper_2_url>", "<paper_3_url>"],
  queries: [
    "What is the core technique and how does it differ from standard approaches?",
    "What specific code/architecture changes are needed to implement this?",
    "What hyperparameters does this introduce and what are the recommended ranges?",
    "What improvement was reported, on which benchmark, and what was the baseline?",
    "What are the computational overhead and memory requirements?"
  ]
)
```
Especially useful when comparing candidate techniques — one call extracts the same info from all papers.

### For reference repo exploration:
`mcp__alphaxiv__read_files_from_github_repository` explores paper codebases:
1. Start with `path: "/"` for the repo structure and top-level files (README, LICENSE, setup.py)
2. Drill into the implementation directory (e.g., `path: "models/"` or `path: "src/"`)
3. Read specific implementation files for core technique code

This replaces cloning repos locally for initial assessment. Reserve cloning (via `${CLAUDE_PLUGIN_ROOT}/scripts/implement_utils.py clone`) for the implement phase when actual code adaptation happens.

### Fallback:
If alphaxiv tools are unavailable, use `WebFetch(url)` for paper content and `WebFetch` on GitHub README URLs for repo assessment.

## Previously Tried Techniques

Before proposing, check if `<exp_root>/reports/research-findings.md` already exists. If so:
1. Read all previously proposed technique names
2. Do NOT re-propose already-tried techniques
3. Note in the output: "Excluded N previously-proposed techniques"

This prevents re-implementing techniques from prior optimization runs.

## Output Format

Rank proposals by priority score: `(impact * confidence) / (11 - min(feasibility, 10))`

Note: clamp feasibility to [1, 10] to prevent division by zero. Higher feasibility = easier to implement = higher priority.

```markdown
### Proposal: [Name]
- **Type:** code_change | hp_only
- **Source:** [Paper title, URL]
- **Technique:** [Category] - [Brief description]
- **Implementation:**
  - Files to modify: [list]
  - Changes: [description]
  - New dependencies: [if any]
- **Expected improvement:** [X% on metric, based on paper results on comparable task]
- **Complexity:** Low/Medium/High
- **Risk:** [What could go wrong]
- **Priority score:** [1-10]
- **Implementation strategy:** from_scratch | from_reference
- **Reference repo:** [URL] (only for from_reference)
- **Reference files:** `path/to/file.py` (only for from_reference)
```

### Proposal Type Classification

- **`code_change`**: modifies model architecture, loss functions, data pipeline, or training loop code. Goes through the implement skill for branch creation.
- **`hp_only`**: achievable purely through hyperparameter or config changes. Examples: "use cosine annealing" (scheduler config), "increase weight decay" (optimizer param), "add warmup" (scheduler config). Bypasses implement, goes directly to hp-tune.
