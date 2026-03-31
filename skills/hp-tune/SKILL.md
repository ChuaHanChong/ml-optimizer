---
name: hp-tune
description: "LLM-driven hyperparameter tuning for ML models. Reads past experiment results, reasons about the search space, and proposes the next batch of configurations. No Optuna/Bayesian — Claude reasons directly about what to try next. Use when: need to propose new HP configurations based on past results."
user-invocable: false
---

# Hyperparameter Tuning (LLM-Driven)

Use extended thinking for all analytical reasoning in this skill. Ultrathink. Think through trade-offs, HP interaction effects, exploration vs exploitation balance, and diminishing returns before proposing configurations.

You are acting as an intelligent hyperparameter tuning agent. Instead of using grid search, random search, or Bayesian optimization, you reason directly about past results to propose the next batch of configurations.

## Reference

- Tuning strategy guide: `${CLAUDE_SKILL_DIR}/references/tuning-strategy.md` (in this skill's directory)
- Read this reference FIRST before proposing any configs.

## Inputs Expected

From the orchestrator:
- `project_root`: Project root directory
- `num_gpus`: Number of GPUs available (determines batch size)
- `search_space`: Defined HP ranges from the optimization plan
- `iteration`: Which tuning iteration this is (1, 2, 3, ...)
- `primary_metric`: The metric to optimize (e.g., "loss", "accuracy", "psnr")
- `lower_is_better`: Whether lower metric values are better (True for loss, False for accuracy)
- `code_branches`: List of validated code branches from the implementation manifest (e.g., `["ml-opt/perceptual-loss"]`), or `[]` for HP-only. In iteration 1, generate one config per branch (with baseline HPs) plus one for the original code, instead of spanning the search space.
- `warm_start_enabled`: Whether checkpoint warm-starting is enabled (boolean, default false). When true and iteration >= 2, propose warm-starting from the best completed experiment on the same branch.
- `available_checkpoints`: Dict mapping exp_id to checkpoint info (optional). Only provided when `warm_start_enabled` is true. Example: `{"exp-005": {"checkpoint_path": "experiments/artifacts/exp-005/best.pt", "code_branch": "ml-opt/method-a"}}`.
- `branch_scores`: Per-branch allocation scores from analyze (optional). Dict mapping branch name to `{"improvement_pct": X, "sample_count": N, "score": Y}`.

## Step 1: Load Past Results

> **Goal check:** Respect frozen parameters, OOM limits, and dead-end constraints from the optimization goals. Never propose configs that violate these.

Read all experiment results:
```bash
python3 -c "
import json, sys
# sys.path: add the plugin's scripts/ directory
from result_analyzer import load_results, rank_by_metric
results = load_results('<project_root>/experiments/results')
print(json.dumps({k: v for k, v in results.items()}, indent=2))
"
```

Also load the baseline:
- Read `experiments/results/baseline.json` for the starting point
- Note the baseline metrics and config

## Step 2: Analyze What's Been Tried

Use the result analyzer:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/result_analyzer.py \
  <project_root>/experiments/results \
  <primary_metric> \
  baseline \
  <lower_is_better>
```

**Branch-aware analysis:** Group past results by `code_branch` field before analysis. Experiments on different code branches should be analyzed separately — HP sensitivities may differ between branches. For example, `lr=0.001` on a perceptual-loss branch may behave very differently from `lr=0.001` on baseline code.

**Check dead-end catalog:** Read techniques that were conclusively shown to be unpromising:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> dead-end list
```
Avoid proposing HP configs for branches/methods listed as dead ends. Focus HP exploration on branches that still show potential.

**Check research agenda:** Read the living research agenda for context on which untried techniques are high-priority:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> agenda list
```
If high-priority untried ideas exist, consider whether HP exploration should focus on branches related to those ideas (to maximize their potential) or on the current best branch.

**Training-budget-aware proposals:**

If `fixed_time_budget` is set (all experiments train for the same wall-clock duration):
- Estimate steps fitting in budget: `estimated_steps = fixed_time_budget × baseline_throughput_steps_per_sec`
- Propose LR schedules appropriate for that step count (e.g., 1-cycle over estimated_steps)
- Avoid proposing epoch counts — the time budget controls duration
- For short budgets (< 120s), avoid slow-convergence schedules (cosine annealing over 100+ epochs)

If `fixed_epoch_budget` is set (all experiments train for the same number of epochs):
- Do NOT vary epoch count in proposals — all experiments use the fixed epoch count
- Focus HP variation on LR, batch size, weight decay, and other non-epoch parameters
- Propose LR schedules appropriate for the fixed epoch count

From this analysis, understand:
- **Best result so far:** Which config AND branch gave the best metric value?
- **Worst result:** What should be avoided?
- **Diverged experiments:** What caused them? (too high LR, too large batch, etc.)
- **Trends:** Is there a clear direction (e.g., lower LR consistently better)? Do trends differ by branch?
- **Untried regions:** What parts of the search space haven't been explored?
- **Branch performance:** Which code branches are consistently better/worse?

## Step 3: Reason About Next Configs

This is the core of LLM-driven HP tuning. Think through the following:

### Iteration 1 (Exploration)
If this is the first tuning iteration (only baseline exists):

**If `code_branches` is non-empty:** Generate one config per branch using baseline HPs, plus one config for the original code (no branch). This tests each code change in isolation before HP tuning. Assign each config a `code_branch` and `code_proposal` field. Cap total configs at `len(code_branches) + 1`.

**If `code_branches` is empty (HP-only):**

**Tabular ML iteration 1 adjustment:** If the detected framework is scikit-learn, XGBoost, or LightGBM (tree-based models):
- **Iteration 1 priority**: Explore `max_depth` and `n_estimators` first (these have the highest impact for tree-based models)
- **Iteration 2+**: Then tune `learning_rate`/`eta`, `min_child_weight`, `subsample`, `colsample_bytree`
- **Rationale**: Learning rate is less impactful for tree-based models compared to tree structure parameters

For neural network frameworks (PyTorch, TensorFlow, JAX): keep the existing strategy (learning rate first).

**Default strategy (neural networks):**
- Propose configs that span the search space
- Focus on learning rate first (highest impact)
- One config per order of magnitude of LR
- Keep other HPs at baseline values

### Iteration 2+ (Exploitation + Exploration)
Based on past results:

1. **Identify the best region:** Where did the best results come from?
2. **Zoom in:** Propose configs close to the best, with small variations
3. **Check for interactions:** If LR was tuned, now vary batch size or weight decay
4. **Explore edges:** If best result was at the boundary, extend the search
5. **Avoid repeats:** Never propose a config identical to one already tried

### Reasoning Template

For each proposed config, provide this reasoning:

```
Config <N>: {lr: X, batch_size: Y, ...}
Reasoning:
- Based on: [which past result informed this choice]
- Change from best: [what's different and why]
- Expected outcome: [what we hope to learn]
- Risk: [what could go wrong]
```

### Warm-Start Proposals (Iteration 2+, when `warm_start_enabled`)

When warm-starting is enabled and `available_checkpoints` is non-empty:
1. For each branch being tuned, find the best checkpoint on that same branch
2. Propose warm-started configs with lower LR (0.3-0.5x) and fewer epochs (0.3-0.5x)
3. Mix: at most 2/3 warm-started, at least 1/3 from-scratch (maintains exploration)
4. Only warm-start from same `code_branch` — cross-branch is unsafe
5. Never warm-start from diverged/failed experiments
6. Set `checkpoint_source` in the proposed config: `{"exp_id": "<source>", "checkpoint_path": "<path>"}`

### Adaptive Branch Budget Allocation (Iteration 2+, multiple branches)

When `branch_scores` is provided and `code_branches` has 2+ entries:
1. Allocate experiment slots proportionally to branch scores: `slots = round(total * score / sum_scores)`
2. Every surviving branch gets minimum 1 slot
3. High-score branches get more exploitation configs; low-score get more exploration
4. If all branches are within 1% of each other, fall back to equal allocation
5. Log allocation breakdown to dev_notes

**Formula:** `score = max(improvement_pct × confidence, 0.0)` where `confidence = 1 - 1/√(sample_count + 1)`. Branches worse than baseline get score 0 and receive no allocation.

### Interaction-Aware Proposals

When the analyze output includes HP interactions:
- If a strong interaction is detected (e.g., LR × batch_size), propose configs that explore the interacting pair TOGETHER, not independently
- The interaction rho sign indicates which combinations to prefer

### Categorical Hyperparameters

When `search_space` includes non-numeric choices (e.g., `optimizer: ["adam", "sgd", "adamw"]`, `scheduler: ["cosine", "step"]`):

- **Iteration 1:** Include each categorical option at least once across proposals (combined with reasonable numeric defaults)
- **Iteration 2+:** Focus on the best-performing categorical values. Cross them with numeric tuning — e.g., if "adam" outperformed "sgd", try "adam" with varied learning rates
- **Interaction effects:** Categorical choices often change the optimal numeric range (e.g., SGD needs higher LR than Adam). When switching optimizer, also broaden the LR range
- **Grouping:** Treat categorical choices as separate "branches" in analysis — don't interpolate between them

## Step 4: Validate Proposals

Before finalizing, check each proposed config:

1. **Batch size cap:** Total proposals must not exceed `max(num_gpus, 1)`.
2. **GPU memory:** Will the batch size fit? (Check against baseline profiling)
3. **Not a duplicate:** Has this exact config been tried before?
4. **Within search space:** All values within defined ranges
5. **Sensible combinations:** LR and batch size follow linear scaling rule

## Step 4.1: Log Tuning Issues

### If proposals duplicate previously tried configs (caught in step 4.3):
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"hp-tune","message":"Regenerated <N> proposals due to duplication with past configs","phase":7,"iteration":<iteration>}'
```

## Step 5: Write Proposed Configs

Create a directory for proposed configs:
```bash
mkdir -p <project_root>/experiments/results/proposed-configs
```

For each proposed config, write a JSON file:
```json
{
  "exp_id": "<next_exp_id>",
  "config": {
    "lr": <value>,
    "batch_size": <value>,
    "weight_decay": <value>,
    "scheduler": "<type>",
    "epochs": <value>,
    ...
  },
  "code_branch": "<branch name or null for baseline>",
  "code_proposal": "<proposal name or null>",
  "proposal_source": "<paper|llm_knowledge|null>",
  "method_tier": "<baseline|method_default_hp|method_tuned_hp>",
  "gpu_id": <assigned_gpu>,
  "reasoning": "<why this config was chosen>",
  "iteration": <tuning_iteration>
}
```

**Field rules:**
- Baseline config (no code change): `"code_branch": null, "code_proposal": null`
- Branch config: `"code_branch": "ml-opt/<slug>"` (from manifest), `"code_proposal": "<slug>"` (matching the manifest entry's `slug` field)
- Iteration 2+: inherit from the branch being tested, or null for baseline code

**`method_tier` rules:**
- `"baseline"`: No code branch, running baseline HPs on original code
- `"method_default_hp"`: Has a code branch, iteration 1 (testing the code change with baseline/default HPs)
- `"method_tuned_hp"`: Has a code branch, iteration 2+ (tuning HPs on the code branch)
- `"stacked_default_hp"`: Stacked branch, first run (testing combined code with best individual HPs)
- `"stacked_tuned_hp"`: Stacked branch, after HP tuning (tuning HPs on the stacked code)

**`proposal_source` rules:**
- If config has `code_branch`: inherit `proposal_source` from the implementation manifest's matching proposal entry
- If `code_branch` is null: set `proposal_source` to `null`
- Iterations 2+: carry forward from the branch's original `proposal_source`
- `"paper"`: Proposal originated from web research (Phase 5)
- `"llm_knowledge"`: Proposal originated from LLM knowledge (Phase 7 method proposals)
- `null`: For baseline experiments (no code change)

Use `${CLAUDE_PLUGIN_ROOT}/scripts/experiment_setup.py` to generate proper experiment IDs:
```bash
python3 -c "
import sys
# sys.path: add the plugin's scripts/ directory
from experiment_setup import next_experiment_id
print(next_experiment_id('<project_root>/experiments/results'))
"
```

## Step 6: Document Tuning Decision

Append to `experiments/dev_notes.md`:
```markdown
## <date> — HP Tuning Iteration <N>

- Best so far: [exp_id] with [metric]=[value]
- [N] experiments completed, [M] diverged
- Strategy: [Exploration/exploitation/hybrid] because [reasoning]
- Proposed: [exp-X (lr=...), exp-Y (lr=...)]
```

## Output

Return to the orchestrator:
- List of proposed configs (exp_id, config, gpu_id)
- Reasoning summary
- Any concerns or notes (e.g., "approaching diminishing returns")
- Recommendation: `"continue"` or `"stop"` (see "When to Recommend Stopping" below)

## When to Recommend Stopping

Recommend stopping the tuning loop if:
1. Last 3+ experiments showed <1% improvement over the best
2. The search space has been thoroughly explored (no promising untried regions)
3. The goal metric has been achieved
4. All reasonable LR values have been tried and the best is clear

**Note:** The "<1% improvement" threshold is **relative** to the baseline value (i.e., `delta / baseline * 100`). For metrics with very small absolute values (e.g., loss=0.001), even a tiny absolute change may be a large relative improvement. Always use percentage change, not absolute delta, for stopping decisions.

Include a `"recommendation": "continue"|"stop"` field in your output.

### HP-Tuning for Stacked Methods

When invoked during the stacking phase (identifiable by `method_tier: "stacked_default_hp"` in recent results):

1. **Starting point:** Use the HP config from the best individual method in the stack (passed as `baseline_config`).
2. **Narrow scope:** Only vary HPs that the newly added method likely interacts with. For example:
   - New loss function → vary `learning_rate`, `weight_decay`
   - New augmentation → vary `batch_size`, `learning_rate`
   - New scheduler → vary `learning_rate`, `warmup_steps`
3. **Budget:** Cap at 2 iterations maximum during stacking.
4. **Proposals:** Generate `max(num_gpus, 1)` configs, all targeting the stack branch.
