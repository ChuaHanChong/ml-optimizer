---
name: hp-tune
description: "LLM-driven hyperparameter tuning for ML models. Reads past experiment results, reasons about the search space, and proposes the next batch of configurations. No Optuna/Bayesian — Claude reasons directly about what to try next. Use when: need to propose new HP configurations based on past results."
---

# Hyperparameter Tuning (LLM-Driven)

You are acting as an intelligent hyperparameter tuning agent. Instead of using grid search, random search, or Bayesian optimization, you reason directly about past results to propose the next batch of configurations.

## Reference

- Tuning strategy guide: `references/tuning-strategy.md` (in this skill's directory)
- Read this reference FIRST before proposing any configs.

## Inputs Expected

From the orchestrator:
- `project_root`: Project root directory
- `num_gpus`: Number of GPUs available (determines batch size)
- `search_space`: Defined HP ranges from the optimization plan
- `iteration`: Which tuning iteration this is (1, 2, 3, ...)

## Step 1: Load Past Results

Read all experiment results:
```bash
python3 -c "
import json, sys
sys.path.insert(0, '$HOME/.claude/plugins/ml-optimizer/scripts')
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
python3 ~/.claude/plugins/ml-optimizer/scripts/result_analyzer.py \
  <project_root>/experiments/results \
  <primary_metric> \
  baseline \
  <lower_is_better>
```

**Branch-aware analysis:** Group past results by `code_branch` field before analysis. Experiments on different code branches should be analyzed separately — HP sensitivities may differ between branches. For example, `lr=0.001` on a perceptual-loss branch may behave very differently from `lr=0.001` on baseline code.

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

## Step 4: Validate Proposals

Before finalizing, check each proposed config:

1. **GPU memory:** Will the batch size fit? (Check against baseline profiling)
2. **Not a duplicate:** Has this exact config been tried before?
3. **Within search space:** All values within defined ranges
4. **Sensible combinations:** LR and batch size follow linear scaling rule

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
  "gpu_id": <assigned_gpu>,
  "reasoning": "<why this config was chosen>",
  "iteration": <tuning_iteration>
}
```

Use `experiment_setup.py` to generate proper experiment IDs:
```bash
python3 -c "
import sys
sys.path.insert(0, '$HOME/.claude/plugins/ml-optimizer/scripts')
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

## When to Recommend Stopping

Recommend stopping the tuning loop if:
1. Last 3+ experiments showed <1% improvement over the best
2. The search space has been thoroughly explored (no promising untried regions)
3. The goal metric has been achieved
4. All reasonable LR values have been tried and the best is clear

Include a `"recommendation": "continue"|"stop"` field in your output.
