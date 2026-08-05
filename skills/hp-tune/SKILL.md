---
name: hp-tune
description: "LLM-driven hyperparameter tuning for ML models. Reads past experiment results, reasons about the search space, and proposes the next batch of configurations. No Optuna/Bayesian — Claude reasons directly about what to try next. Use when: need to propose new HP configurations based on past results."
user-invocable: false
---

# Hyperparameter Tuning (LLM-Driven)

Reason directly about past results to propose the next config batch — no grid/random/Bayesian search. Weigh trade-offs, HP interactions, exploration vs exploitation, and diminishing returns first.

> **Path convention:** Paths `<exp_root>/...` refer to the `exp_root` dispatch parameter. Output directory name is not hardcoded.

## Reference

- Tuning strategy guide: `${CLAUDE_SKILL_DIR}/references/tuning-strategy.md`. Read it FIRST, before proposing any configs.

## Inputs Expected

From the orchestrator:
- `project_root`: Project root directory
- `num_gpus`: GPUs available (determines batch size)
- `num_configs`: Total proposal batch size the orchestrator computed for this round (e.g. `num_gpus * experiments_per_gpu`, or `1` for sequential file-backup projects). **Cap all proposals at this value** when provided — do not re-derive the cap from `num_gpus` or `code_branches` count.
- `search_space`: HP ranges from the optimization plan
- `iteration`: Tuning iteration number (1, 2, 3, ...)
- `primary_metric`: Metric to optimize (e.g. "loss", "accuracy", "psnr")
- `lower_is_better`: Whether lower values are better (True for loss, False for accuracy)
- `model_category`: `"rl"` | `"generative"` | `"supervised"` (optional). Drives the search-space prior order and the RL warm-start rules below.
- `seeds_per_config`: Seed replicates per config (int, default 1; >1 suggested for RL). When >1, propose each config `seeds_per_config` times with distinct `random_seed` values; replicates count against the batch cap.
- `code_branches`: Validated code branches from the implementation manifest (e.g. `["ml-opt/perceptual-loss"]`), or `[]` for HP-only. In iteration 1, generate one config per branch (baseline HPs) plus one for the original code, instead of spanning the search space.
- `warm_start_enabled`: Whether checkpoint warm-starting is enabled (boolean, default false). When true and iteration >= 2, propose warm-starting from the best completed experiment on the same branch.
- `branch_scores`: Per-branch allocation scores from analyze (optional). Dict mapping branch name to `{"improvement_pct": X, "sample_count": N, "score": Y}`.
- `round_dir`: Current round directory (e.g. `"round-3-hp"`). **Required.** Passed by the orchestrator after `round_manager.py create-round`. Proposed config JSONs MUST be written inside `proposed-configs/<round_dir>/`. If missing, fetch via `round_manager.py current-round`.
## Step 1: Load Past Results

> **Goal check:** Respect frozen parameters, OOM limits, and dead-end constraints from the optimization goals. Never propose configs that violate these.

Read all experiment results:
```bash
python3 -c "
import json, sys
# sys.path: add the plugin's scripts/ directory
from result_analyzer import load_results, rank_by_metric
results = load_results('<exp_root>/results')
print(json.dumps({k: v for k, v in results.items()}, indent=2))
"
```

Also load the baseline:
- Read `<exp_root>/results/baseline.json` for the starting point
- Note the baseline metrics and config

## Step 2: Analyze What's Been Tried

Use the result analyzer:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/result_analyzer.py \
  <exp_root>/results \
  <primary_metric> \
  baseline \
  <lower_is_better>
```

**Branch-aware analysis:** Group past results by `code_branch` before analysis. Experiments on different branches are analyzed separately — HP sensitivities may differ. E.g. `lr=0.001` on a perceptual-loss branch may behave very differently from `lr=0.001` on baseline code.

**Check dead-end catalog:** Read techniques conclusively shown unpromising:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> dead-end list
```
Avoid proposing HP configs for branches/methods listed as dead ends. Focus exploration on branches that still show potential.

**Check research agenda:** Read the living agenda for which untried techniques are high-priority:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> agenda list
```
If high-priority untried ideas exist, consider whether HP exploration should focus on branches related to those ideas or on the current best branch.

**Search-space prior order (where HP ranges come from):**

1. **Research-derived priors (PRIMARY):** `search_space` entries (`{param, range, scale, source}`) on `hp_only` proposals in `<exp_root>/reports/research-findings*.md` and on research-agenda entries (`agenda list` above). They carry citations — always prefer them over built-in guidance.
2. **Baseline-captured HPs:** ranges seeded around the HP values in `<exp_root>/results/baseline.json` `config`. For `model_category=rl`, seed around the captured gamma, clip_range, ent_coef, n_steps (and learning rate).
3. **`tuning-strategy.md` reference sections:** cold-start fallback ONLY — use when neither research priors nor baseline HPs cover a parameter.

**Research-round request:** When `model_category` is not `"supervised"` (i.e. `rl` or `generative`) AND no research-derived priors exist yet (no `search_space` entries in findings or agenda), do NOT fall back to supervised defaults. Propose only conservative baseline-anchored configs (prior order 2) for this batch, and include `"research_requested": true` plus a note naming the parameters that need cited priors — the workflow routes a research round before broad HP exploration when `scope_level` permits research and the `method_proposal_iterations` budget is not exhausted; otherwise it logs the request and proceeds with baseline-anchored configs only.

**Training-budget-aware proposals:**

If `fixed_time_budget` is set (all experiments train the same wall-clock duration):
- **Overshoot the epoch count deliberately — don't calibrate precisely to the budget.** No reliable epochs/sec estimate exists (`baseline.json`'s `profiling.throughput_samples_per_sec` is samples/sec, not epochs/sec — converting needs dataset size, which varies by config). Overshooting is harmless with periodic checkpointing (timeout just cuts training off wherever it reaches); undershooting wastes budget. When a scheduler needs an epoch count (e.g. cosine annealing's `T_max`), propose one noticeably higher than any plausible fit — same pattern the baseline skill uses (`--epochs 200` as a deliberately oversized ceiling — see skills/baseline/SKILL.md Step 2.2).
- Propose LR schedules for the time budget, not a specific epoch target — a schedule keyed to the overshot ceiling still works since it just gets truncated.
- For short budgets (< 120s), avoid slow-convergence schedules as the primary lever — an oversized epoch flag itself is fine.

If `fixed_epoch_budget` is set (all experiments train the same number of epochs):
- Do NOT vary epoch count — all experiments use the fixed epoch count
- Focus HP variation on LR, batch size, weight decay, and other non-epoch parameters
- Propose LR schedules appropriate for the fixed epoch count

From this analysis, understand:
- **Best result so far:** Which config AND branch gave the best metric?
- **Worst result:** What to avoid?
- **Diverged experiments:** What caused them? (LR too high, batch too large, etc.)
- **Trends:** Clear direction (e.g. lower LR consistently better)? Do trends differ by branch?
- **Untried regions:** What search-space parts are unexplored?
- **Branch performance:** Which branches are consistently better/worse?

## Step 3: Reason About Next Configs

Think through the following.

### Iteration 1 (Exploration)
First tuning iteration (only baseline exists):

**If `code_branches` is non-empty:** Generate one config per branch using baseline HPs, plus one for the original code (no branch). Tests each code change in isolation before HP tuning. Assign each config a `code_branch` and `code_proposal` field. Cap total proposals at the passed `num_configs` value.

**If `code_branches` is empty (HP-only):**

**Tabular ML iteration 1 adjustment:** If the framework is scikit-learn, XGBoost, or LightGBM (tree-based):
- **Iteration 1 priority:** Explore `max_depth` and `n_estimators` first (highest impact for tree-based models)
- **Iteration 2+:** Then tune `learning_rate`/`eta`, `min_child_weight`, `subsample`, `colsample_bytree`
- **Rationale:** Learning rate is less impactful than tree structure for tree-based models

For neural network frameworks (PyTorch, TensorFlow, JAX): keep the existing strategy (learning rate first).

**Default strategy (neural networks):**
- Propose configs spanning the search space
- Focus on learning rate first (highest impact)
- One config per order of magnitude of LR
- Keep other HPs at baseline values

### Iteration 2+ (Exploitation + Exploration)
Based on past results:

1. **Identify the best region:** Where did the best results come from?
2. **Zoom in:** Propose configs close to the best, with small variations
3. **Check interactions:** If LR was tuned, now vary batch size or weight decay
4. **Explore edges:** If the best result was at the boundary, extend the search
5. **Avoid repeats:** Never propose a config identical to one already tried

### Reasoning Template

For each proposed config, provide:

```
Config <N>: {lr: X, batch_size: Y, ...}
Reasoning:
- Based on: [which past result informed this choice]
- Change from best: [what's different and why]
- Expected outcome: [what we hope to learn]
- Risk: [what could go wrong]
```

### Warm-Start Proposals (Iteration 2+, when `warm_start_enabled`)

When `warm_start_enabled` is true and iteration >= 2, discover available checkpoints
yourself — no `available_checkpoints` input is provided. Scan the results already loaded
in Step 1 (`results/round-*/exp-*.json`) for `status: "completed"` experiments on the
branch being tuned, and check each for an `artifacts_dir` (checkpoint files under it) or a
`checkpoint_source` field. If at least one qualifying checkpoint exists for a branch:
1. For each branch being tuned, find the best checkpoint on that same branch
2. Propose warm-started configs with lower LR (0.3-0.5x) and fewer epochs (0.3-0.5x)
3. Mix: at most 2/3 warm-started, at least 1/3 from-scratch (maintains exploration)
4. Only warm-start from the same `code_branch` — cross-branch is unsafe
5. Never warm-start from diverged/failed experiments
6. Set `checkpoint_source` in the config: `{"exp_id": "<source>", "checkpoint_path": "<path>"}`
7. **RL warm-start requirement (`model_category=rl`):** warm-start only when the checkpoint bundles the optimizer state AND the observation/reward normalizer statistics (plus the replay buffer for off-policy algorithms like SAC/TD3). A bare policy-weights checkpoint silently resets normalization and exploration state — if the full bundle is unavailable, warm-starting is FORBIDDEN: propose from-scratch configs instead.
8. **RL budget phrasing:** phrase warm-start budget reductions in environment timesteps (e.g. 0.3-0.5x `total_timesteps`), never epochs.

### Adaptive Branch Budget Allocation (Iteration 2+, multiple branches)

When `branch_scores` is provided and `code_branches` has 2+ entries:
1. Allocate slots proportionally to branch scores: `slots = round(total * score / sum_scores)`
2. Every surviving branch gets minimum 1 slot
3. High-score branches get more exploitation configs; low-score get more exploration
4. If all branches are within 1% of each other, fall back to equal allocation
5. Log allocation breakdown to dev_notes

**Formula:** `score = max(improvement_pct × confidence, 0.0)` where `confidence = 1 - 1/√(sample_count + 1)`. Branches worse than baseline get score 0 and no allocation.

### Interaction-Aware Proposals

When the analyze output includes HP interactions:
- If a strong interaction is detected (e.g. LR × batch_size), propose configs exploring the interacting pair TOGETHER, not independently
- The interaction rho sign indicates which combinations to prefer

### Categorical Hyperparameters

When `search_space` includes non-numeric choices (e.g. `optimizer: ["adam", "sgd", "adamw"]`, `scheduler: ["cosine", "step"]`):

- **Iteration 1:** Include each categorical option at least once across proposals (with reasonable numeric defaults)
- **Iteration 2+:** Focus on the best-performing categorical values. Cross with numeric tuning — e.g. if "adam" beat "sgd", try "adam" with varied learning rates
- **Interaction effects:** Categorical choices often change the optimal numeric range (e.g. SGD needs higher LR than Adam). When switching optimizer, also broaden the LR range
- **Grouping:** Treat categorical choices as separate "branches" in analysis — don't interpolate between them

## Step 4: Validate Proposals

Before finalizing, check each config:

1. **Batch size cap:** Total proposals must not exceed the passed `num_configs` value (fall back to `max(num_gpus, 1)` only if `num_configs` was not provided for this dispatch).
2. **GPU memory:** Will the batch size fit? (Check against baseline profiling)
3. **Not a duplicate:** Has this exact config been tried? **Seed-replicate exemption:** configs identical except for `random_seed` are intentional replicates (when `seeds_per_config` > 1), NOT duplicates — do not regenerate or drop them.
4. **Within search space:** All values within defined ranges
5. **Sensible combinations:** LR and batch size follow the linear scaling rule

## Step 4.1: Log Tuning Issues

### If proposals duplicate previously tried configs (dup check, Step 4):
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"hp-tune","message":"Regenerated <N> proposals due to duplication with past configs","phase":7,"iteration":<iteration>}'
```

## Step 5: Write Proposed Configs

First, get the current round directory from your dispatch prompt (`round_dir`). If not provided, fetch it:
```bash
round_dir=$(python3 ${CLAUDE_PLUGIN_ROOT}/scripts/round_manager.py <exp_root> current-round | python3 -c "import json,sys; print(json.load(sys.stdin)['dir'])")
```

Create the proposed-configs directory **inside the round subdirectory**:
```bash
mkdir -p <exp_root>/proposed-configs/<round_dir>
```

**IMPORTANT:** The PreToolUse hook (`validate_experiment_write.py`) blocks any `exp-*.json` write to `proposed-configs/` outside a `round-N-<type>/` subdirectory. Always write proposals to `proposed-configs/<round_dir>/<exp_id>.json`.

For each config, write a JSON file at `<exp_root>/proposed-configs/<round_dir>/<exp_id>.json`:
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
  "random_seed": <int — only on seed replicates>,
  "reasoning": "<why this config was chosen>",
  "iteration": <tuning_iteration>
}
```

**Field rules:**
- Baseline config (no code change): `"code_branch": null, "code_proposal": null`
- Branch config: `"code_branch": "ml-opt/<slug>"` (from manifest), `"code_proposal": "<slug>"` (matching the manifest entry's `slug` field)
- Iteration 2+: inherit from the branch being tested, or null for baseline code
- `random_seed`: set on seed replicates (`seeds_per_config` > 1) — each replicate shares the config but carries a distinct `random_seed`, applied via the framework's `--seed` flag. PYTHONHASHSEED is never recorded as the experiment seed.

**`method_tier` rules:**
- `"baseline"`: No code branch, running baseline HPs on original code
- `"method_default_hp"`: Has a code branch, iteration 1 (testing the code change with baseline/default HPs)
- `"method_tuned_hp"`: Has a code branch, iteration 2+ (tuning HPs on the code branch)
- `"stacked_default_hp"`: Stacked branch, first run (testing combined code with best individual HPs)
- `"stacked_tuned_hp"`: Stacked branch, after HP tuning (tuning HPs on the stacked code)

**`proposal_source` rules:**
- If the config has `code_branch`: inherit `proposal_source` from the implementation manifest's matching proposal entry
- If `code_branch` is null: set `proposal_source` to `null`
- Iterations 2+: carry forward from the branch's original `proposal_source`
- `"paper"`: Proposal originated from web research (Phase 5)
- `"llm_knowledge"`: Proposal originated from LLM knowledge (Phase 7 method proposals)
- `null`: For baseline experiments (no code change)

Use `round_manager.py next-id` for globally unique experiment IDs (scans all round directories, not just the flat `results/`):
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/round_manager.py <exp_root> next-id
# Returns: {"exp_id": "exp-NNN"}
```

Exp-ids are globally unique across rounds — `exp-001` only ever exists in one round.

## Step 6: Document Tuning Decision

Append to `<exp_root>/dev_notes.md`:
```markdown
## <date> — HP Tuning Iteration <N>

- Best so far: [exp_id] with [metric]=[value]
- [N] experiments completed, [M] diverged
- Strategy: [Exploration/exploitation/hybrid] because [reasoning]
- Proposed: [exp-X (lr=...), exp-Y (lr=...)]
```

## Output

Return to the orchestrator:
- Proposed configs (exp_id, config, gpu_id)
- Reasoning summary
- Concerns or notes (e.g. "approaching diminishing returns")
- Recommendation: `"continue"` or `"stop"` (see below)

## When to Recommend Stopping

Recommend stopping the tuning loop if:
1. Last 3+ experiments showed <1% improvement over the best
2. The search space is thoroughly explored (no promising untried regions)
3. The goal metric is achieved
4. All reasonable LR values are tried and the best is clear

**Note:** The "<1% improvement" threshold is **relative** to the baseline value (`delta / baseline * 100`). For metrics with very small absolute values (e.g. loss=0.001), a tiny absolute change may be a large relative improvement. Always use percentage change, not absolute delta.

Include a `"recommendation": "continue"|"stop"` field in your output.

### HP-Tuning for Stacked Methods

When invoked during the stacking phase (identifiable by `method_tier: "stacked_default_hp"` in recent results):

1. **Starting point:** Narrow around the best HPs of the stacked methods — passed as prose guidance in the dispatch prompt, not a structured `baseline_config` field (no such field exists; read the stacked methods' HPs from context).
2. **Narrow scope:** Only vary HPs the newly added method likely interacts with. E.g.:
   - New loss function → vary `learning_rate`, `weight_decay`
   - New augmentation → vary `batch_size`, `learning_rate`
   - New scheduler → vary `learning_rate`, `warmup_steps`
3. **Budget:** Cap at 1 iteration during stacking (phase-8-stacking.js dispatches tuning-agent at most once per stack step — one narrowed round, no loop).
4. **Proposals:** phase-8-stacking.js passes no `num_configs`/`num_gpus` for this dispatch (only `exp_root, project_root, code_branches, primary_metric, lower_is_better, iteration: 1`) — propose a small narrowed batch (a handful of configs) sized to "1 small round", all targeting the stack branch.

## Domain-Randomization Parameters

Randomization ranges are proposed as a **center/width scalar pair**, never as an explicit `[low, high]` list: `friction_center` + `friction_width` (effective range `center ± width/2`). Two scalars are what you already know how to propose, and the pair cannot produce an inverted range regardless of the values you pick — there is no cross-parameter constraint mechanism that would catch `low > high`.

- `width = 0` means no randomization for that parameter. A width search space therefore spans "off" to "wide" continuously, which is what makes the amount of randomization tunable at all.
- Propose the pair only when a research-derived `search_space` entry supplies it with a cited `source`. Do not invent randomization ranges from a built-in table — see the research-derived priors section.
- **Scope gate:** these parameters change environment dynamics, so they require `scope_level` `"architecture"` or `"full"`. At `"training"` scope, proposing one is a goal violation (`goal_memory.py::check_goal_compliance`, enforced live by the PreToolUse hook per Write) — that one config's write is blocked; other configs in the same batch are unaffected. At `"training"` scope, leave DR parameters out entirely.
- Never propose a `<name>_width` alone. A width without its center is not a randomization parameter and will not be recognized as one.
