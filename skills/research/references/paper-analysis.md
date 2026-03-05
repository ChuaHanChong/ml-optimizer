# Paper Analysis Guide

## How to Extract Implementable Insights from ML Papers

When analyzing a paper, focus on what can be **directly implemented**, not just the theoretical contribution.

## Extraction Framework

For each paper, extract:

### 1. Core Technique
- **What is it?** One-sentence description
- **What problem does it solve?** (e.g., training instability, slow convergence, poor generalization)
- **Category:** Architecture / Loss function / Training strategy / Data augmentation / Regularization / Other

### 2. Implementation Details
- **Code changes required:** Which files/functions need modification?
- **Dependencies:** Any new libraries needed?
- **Complexity estimate:**
  - **Low:** Change a few lines (e.g., swap loss function, add a layer)
  - **Medium:** Modify a module (e.g., new attention mechanism, custom scheduler)
  - **High:** Significant refactoring (e.g., new training paradigm, different architecture)

### 3. Expected Impact
- **What improvement does the paper report?** (quantitative if available)
- **On what benchmark/dataset?** (how comparable is it to our task?)
- **Conditions for improvement:** (e.g., "works best with large batch sizes", "requires pre-training")
- **Realistic expectation:** Papers report best-case; expect 30-70% of reported gains

### 4. Risks and Requirements
- **Could it make things worse?** (e.g., adds training instability, increases memory usage)
- **Computational cost:** More/less expensive than current approach?
- **Compatibility:** Does it work with our model architecture and framework?

## Red Flags in Papers

Be skeptical when:
- Results only shown on toy datasets
- No ablation study
- Improvement is within standard deviation
- Method requires extensive HP tuning to work
- No code available and method description is ambiguous

## Search Strategy

### For architecture improvements:
- Search: "[task] [model_type] architecture improvement 2024 2025"
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

## Previously Tried Techniques

Before proposing, check if `experiments/reports/research-findings.md` already exists. If so:
1. Read all previously proposed technique names
2. Do NOT re-propose techniques that were already tried
3. Note in the output: "Excluded N previously-proposed techniques"

This prevents wasting effort on re-implementing techniques from prior optimization runs.

## Output Format

Rank proposals by: (expected impact * feasibility) / complexity

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
```

### Proposal Type Classification

- **`code_change`**: Requires modifying model architecture, loss functions, data pipeline, or training loop code. These go through the implement skill for branch creation.
- **`hp_only`**: Can be achieved purely through hyperparameter or config changes. Examples: "use cosine annealing" (scheduler config), "increase weight decay" (optimizer param), "add warmup" (scheduler config). These bypass implement and go directly to hp-tune.
