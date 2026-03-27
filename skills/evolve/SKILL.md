---
name: evolve
description: Orchestrate evolutionary code refinement on a method branch via the full ShinkaEvolve pipeline (convert → run → inspect).
disable-model-invocation: true
user-invocable: false
---

# Evolve Skill

Use extended thinking for all reasoning. Ultrathink.

## Overview

This skill orchestrates **evolutionary code refinement** on the best method branch. It runs the full ShinkaEvolve pipeline internally — converting code, running evolution, and extracting the best mutation — then commits the result as a new branch.

ShinkaEvolve provides population management, island model, novelty detection, and selection pressure. The implement-agent acts as the LLM backend via file-based handoff (`SHINKA_PROVIDER=claude_code`).

## Prerequisites

ShinkaEvolve is available via git submodule at `${CLAUDE_PLUGIN_ROOT}/skills/evolve/ShinkaEvolve/`. If not initialized, run:
```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/setup_evolve.sh
```

If the submodule is missing or imports fail, report `status: "shinkaevolve_unavailable"` and return immediately. The orchestrator will fall back to the research → implement path.

## Input Parameters

- `project_root`: Path to the project
- `parent_branch`: Git branch with the current best implementation
- `parent_metrics`: Dict of parent's metrics (e.g., `{loss: 0.35, accuracy: 0.88}`)
- `primary_metric`: Which metric to optimize
- `lower_is_better`: Metric direction
- `scope_level`: Constraint on changes (`"training"`, `"architecture"`, `"full"`)
- `exp_root`: Path to experiments directory
- `feedback_context`: Structured feedback:
  - `batch_analysis`: Summary of what worked/failed
  - `error_patterns`: From error-log.json
  - `dead_ends`: Techniques to avoid
  - `learned_behaviors`: Accumulated patterns
- `evolve_recommendation`: Dict from tuning agent with `{num_generations, population_size, reasoning}` — the tuning agent decides these values based on prior evolution outcomes

## Step 0: Resolve Evolve HPs

The tuning agent sets evolve HPs via `evolve_recommendation`. Resolve in priority order:

1. Use `evolve_recommendation` from dispatch parameters (set by tuning agent).
2. If not provided, read `<exp_root>/learned-behaviors.json` for entries with `category: "evolve_hp"`. Use the most recent recommendation.
3. If neither exists, use defaults: `num_generations=10`, `population_size=2`.

After evolution completes, log the outcome to learned-behaviors for future runs:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> log-behavior evolve_hp \
  '{"num_generations": <actual>, "population_size": <actual>, "mutations_evaluated": <N>, "best_improvement_pct": <X>, "insight": "<what happened>"}'
```

## Step 1: Convert Best Branch to ShinkaEvolve Task

Checkout the parent branch first:
```bash
git checkout <parent_branch>
```

Then invoke via `Skill("ml-optimizer:shinka-convert")`.

**Input:** The best branch's training code (now checked out).

**Task:** Create a ShinkaEvolve task directory at `<exp_root>/artifacts/shinka-task/`:
- `initial.py` — Snapshot of training code with `EVOLVE-BLOCK` markers around mutable regions (optimizer, scheduler, loss, augmentation — scoped by `scope_level`)
- `evaluate.py` — Fitness evaluator that runs training and reports metrics in ShinkaEvolve's contract:
  ```json
  {"combined_score": <float>, "public": {<metrics>}, "private": {}, "text_feedback": "<summary>"}
  ```
  **Important:** The evaluator must construct the model the same way `initial.py` does — import model construction from the candidate program, not hardcode parameters. If the evolved code changes model architecture (e.g., `base_channels`, layer count, hidden dimensions), `evaluate.py` must pick up those changes. Use dynamic imports or read architecture params from the candidate code.
- `shinka.yaml` — Evolution config

**Scope enforcement:** Only wrap code sections allowed by `scope_level` in EVOLVE-BLOCK markers:
- `"training"`: optimizer, scheduler, loss function, augmentation, regularization
- `"architecture"`: + model layers, attention mechanisms, normalization
- `"full"`: any code

**combined_score formula:** For `lower_is_better` metrics (like loss), negate: `combined_score = -loss`. For higher-is-better (accuracy), use directly: `combined_score = accuracy`.

## Step 2: Run Evolution

Invoke via `Skill("ml-optimizer:shinka-run")`.

**Overrides for evolve context:**
- Set env vars `SHINKA_PROVIDER=claude_code` and `SHINKA_HANDOFF_DIR=<exp_root>` before launching — this enables file-based LLM handoff (Step 3)
- Run in background (`&`) and capture PID — needed for the handoff polling loop
- Autonomous execution — this is the "explicitly autonomous" exception in shinka-run's batch control policy, no user confirmation between batches
- Use `--task-dir <exp_root>/artifacts/shinka-task`, `--results_dir <exp_root>/artifacts/shinka-results`
- Use `--num_generations` and `--max-proposal-jobs` from Step 0's resolved evolve HPs

## Step 3: Fulfill Mutation Requests (File Handoff Loop)

While `shinka-run` is running in the background, poll for LLM requests:

```
While shinka-run process is alive (check: kill -0 $SHINKA_PID 2>/dev/null):
  For each <exp_root>/evolve/pending/*.json:
    1. Read {system_msg, user_msg, model_name} from the request
    2. Generate the code mutation:
       - Read the system_msg (contains EVOLVE-BLOCK code + instructions)
       - Read the user_msg (contains parent metrics + mutation request)
       - Generate a SEARCH/REPLACE patch that improves the code
       - Respect EVOLVE-BLOCK boundaries
       - Check dead_ends — do NOT use dead-end techniques
       - Stay within scope_level constraints
    3. Write response to <exp_root>/evolve/completed/<same_filename>.json:
       {"content": "<SEARCH/REPLACE patch>"}
  Sleep 2s
```

**Important:** You ARE the LLM that ShinkaEvolve is calling. Read the prompt carefully and generate a high-quality code mutation. Understand the code, identify weaknesses, propose targeted improvements.

## Step 4: Inspect Results

Invoke via `Skill("ml-optimizer:shinka-inspect")`.

After `shinka-run` completes:
1. Load top programs from `<exp_root>/artifacts/shinka-results/`
2. Rank by `combined_score`
3. Select the best program that passes correctness checks

## Step 5: Apply Winner and Commit

1. **Read the best evolved program** from shinka-inspect output
2. **Derive slug** from parent branch name (e.g., `ml-opt/label-smoothing` → slug `label-smoothing`)
3. **Create branch:** `git checkout -b ml-opt/evolved-<slug> <parent_branch>`
   If the branch already exists, append a suffix: `ml-opt/evolved-<slug>-2`, `-3`, etc.
4. **Apply the winning code changes** to the project files (replace the relevant sections with the evolved code)
5. **Update eval.py if architecture changed:** If the evolved code modified model architecture parameters (e.g., `base_channels`, layer count, hidden dimensions), update the project's `eval.py` to match. The eval script must construct the model the same way the evolved training code does.
6. **Mark changes:** `# [ml-opt] evolved: <description>`
7. **Validate:**
   ```bash
   python3 -c "import py_compile; py_compile.compile('<modified_file>', doraise=True)"
   ```
8. **Commit:**
   ```bash
   git add -A && git commit -m "evolve: <description of best mutation>"
   ```
9. **Return to original branch:** `git checkout <original_branch>`

## Output Format

```json
{
  "status": "validated|validation_failed|shinkaevolve_unavailable",
  "branch": "ml-opt/evolved-<slug>",
  "description": "<what the best mutation changed and why>",
  "mutations_evaluated": 12,
  "best_combined_score": 0.85,
  "generations_completed": 10,
  "files_modified": ["train.py"],
  "reasoning": "<why this mutation was chosen based on feedback>"
}
```

If ShinkaEvolve is unavailable or crashes: `status: "shinkaevolve_unavailable"`, no branch created. The orchestrator falls back to research → implement.

## Important Rules

- **Respect scope_level.** Training scope = only optimizer, scheduler, loss, augmentation, regularization.
- **Check dead ends.** If a technique is in `feedback_context.dead_ends`, do NOT use it.
- **Preserve provenance.** All code must have `# [ml-opt] evolved: <description>` comments.
- **Return to original branch.** Never leave the repo on the evolved branch.
- **Use analysis-recommended HPs.** Always use `evolve_recommendation` from Step 0 — the analysis agent sizes the run based on budget and prior outcomes.
- **Report failures cleanly.** If shinka-run crashes or produces no valid programs, return `status: "shinkaevolve_unavailable"`. Do not attempt ad-hoc mutations — let the orchestrator decide the fallback.
- **File handoff cleanup.** After evolution completes, remove `<exp_root>/evolve/pending/` and `<exp_root>/evolve/completed/` contents.
