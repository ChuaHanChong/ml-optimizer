# Phase 0: Discovery & Planning (MANDATORY)

**You MUST enter plan mode before doing any analysis or code exploration.**

1. **Enter plan mode:**
   - Use `EnterPlanMode` immediately when this skill is invoked
   - Do NOT skip this phase — even if the user provided a model path or description

1.1. **Check for a completed prior run:**
   If the breadcrumbed `<exp_root>` has a `pipeline-state.json` with
   `phase == 9`, ask the user via `AskUserQuestion`:

     > *This `<exp_root>` already has a completed run. I'll create a new
     > directory for your new direction — the previous run stays untouched.
     > Sound good? (Or say "continue here" to build on the existing state,
     > or "resume" to re-enter the completed run.)*

   - `phase < 9` → normal resume, proceed without asking.
   - No `pipeline-state.json` → fresh workspace, proceed normally.
   - User explicitly said "continue" or "resume" → skip this check.

2. **Ask discovery questions:**
   Use `AskUserQuestion` to gather the following (combine into a single, organized prompt):

   ```
   Before I start optimizing, I need to understand your goals and constraints:

   1. **Optimization target:** What metric do you want to improve? (e.g., accuracy, loss, F1, BLEU, latency)
   2. **Current performance:** What is the current value of that metric? (if known)
   3. **Target performance:** What value are you aiming for? (or "as good as possible")
   4. **Constraints:**
      - Maximum training time per experiment?
      - GPU memory limit? (or should I auto-detect?)
      - Any parameters you do NOT want changed?
   5. **Prior attempts:** Have you already tried any optimizations? What worked/didn't?
   6. **Scope preference** _(default: full autonomous optimization)_:
      - HP tuning only (fastest, no code changes)
      - HP tuning + research (web search for methods, no evolutionary code refinement)
      - Full autonomous optimization (HP + research + ShinkaEvolve — **default if not specified**)
   7. **Divergence metric name** _(skip for scikit-learn, XGBoost, or LightGBM — these train in a single fit() call with no iterative loss stream)_: What metric should be monitored for training divergence? (default: "loss". Common alternatives: "train_loss", "val_loss", "objective", "nll_loss", "perplexity" for LLMs). For RL tasks: if a policy/value loss is logged, use it. If only reward is logged, set divergence_metric to the reward metric name and note that the monitor skill will use reward-based heuristics (higher-is-better divergence detection).
   7a. **Divergence polarity** _(auto-inferred, confirm if ambiguous)_:
       Based on the metric name from Q7, infer the polarity:
       - Metrics containing "loss", "error", "nll", "objective", "perplexity" → `divergence_lower_is_better = True`
       - Metrics containing "reward", "accuracy", "psnr", "ssim", "f1", "auc", "return" → `divergence_lower_is_better = False`
       - If the metric name doesn't match either list, ask: "Is a lower value of [metric] better (like loss) or is a higher value better (like reward)?"
       Store as `divergence_lower_is_better` in user_choices.
   8. **Optimization type:** Are you optimizing training performance or inference performance? (This plugin focuses on **training** optimization — inference optimization like quantization, pruning, or ONNX conversion is out of scope.)
   9. **Training budget per experiment** (optional):
      - **Fixed time**: all experiments train for the same wall-clock duration (e.g., "60 seconds each") — makes metrics directly comparable by time
      - **Fixed epochs**: all experiments train for the same number of epochs (e.g., "10 epochs each") — deterministic and reproducible
      - **Default** (leave blank): experiments run until their configured epochs complete, with a safety timeout
      Store as `fixed_time_budget` (seconds) or `fixed_epoch_budget` (integer) in user_choices.
   10. **Anything else** I should know about this model or training setup?
   11. **Dataset location:** Where are your training and validation datasets?
       - Directory path(s), or "embedded in code" if the training script downloads/generates data
   12. **Environment:** Which environment manager do you use?
       - conda (environment name?) / uv / pip / venv / poetry / other
   ```

3. **Record user responses:**

   - Store the user's answers — they will guide every subsequent phase
   - If the user is unsure about some answers, note those as areas to investigate in Phase 1

3.1. **Write experiment root breadcrumb and optimization goals:**
   First, write a breadcrumb so hooks can find the `<exp_root>` directory (even if it's on a different mount).
   The breadcrumb supports multiple runs — each new run appends to `runs[]` and sets `active`:
   ```bash
   mkdir -p .claude
   python3 -c "
import json
from pathlib import Path
bc = Path('.claude/ml-optimizer.json')
data = json.loads(bc.read_text()) if bc.is_file() else {}
runs = data.get('runs', [])
exp = '<exp_root>'
if exp not in runs:
    runs.append(exp)
json.dump({'active': exp, 'runs': runs}, bc.open('w'), indent=2)
"
   ```
   Then create the goal anchor file:
   ```bash
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> init-goals '<goals_json>'
   ```
   Where `<goals_json>` is constructed from the user's answers:
   ```json
   {
     "objective": {
       "primary_metric": "<from Q1>",
       "lower_is_better": "<inferred from metric name>",
       "target_value": "<from Q3, or null>",
       "problem_description": "<synthesized from user context>"
     },
     "constraints": {
       "scope_level": "<from Q6: 'training' if HP-only, 'architecture' if HP+research, 'full' if full autonomous or not specified (default)>",
       "model_category": "<from Q8 or auto-detected>",
       "frozen_parameters": ["<from Q4: parameters user does NOT want changed>"],
       "fixed_time_budget": "<from Q9: seconds, or null>",
       "fixed_epoch_budget": "<from Q9: integer, or null>"
     },
     "divergence": {
       "metric": "<from Q7>",
       "lower_is_better": "<from Q7a>"
     }
   }
   ```
   This file persists in `<exp_root>/optimization-goals.json` and is read by all agents before acting.

3.2. **Brainstorm optimization strategy:**
   Use `Skill("superpowers:brainstorming")` to explore the optimization space with the user. This helps surface non-obvious approaches, trade-offs, and priorities before committing to a plan. The brainstorming skill structures the conversation to explore:
   - What are the most likely bottlenecks? (data, model capacity, training recipe, regularization)
   - What trade-offs matter? (speed vs accuracy, simplicity vs performance)
   - What's the risk appetite? (conservative HP tuning vs aggressive code mutations)
   - Are there domain-specific techniques worth prioritizing?

   The brainstorming output informs the scope, search space, and strategy for the optimization plan.

3.3. **Present understanding and invite refinement:**
   Summarize what you understood back to the user:
   ```
   Here's what I understood:
   - Metric: {primary_metric} ({direction}), target: {target_value}
   - Scope: {scope_level} ({explanation})
   - Budget: {budget_description}
   - Constraints: {frozen_params, time limits, etc.}
   - Dataset: {location}
   - Environment: {env_manager} ({env_name})
   - Strategy insights from brainstorming: {key_insights}

   Would you like to adjust anything before I analyze the codebase?
   ```
   Use `AskUserQuestion` for this. If the user wants changes:
   - Update the relevant user_choices
   - Re-write `optimization-goals.json` via `goal_memory.py init-goals` with updated values
   - Re-run brainstorming if the user wants to explore a different direction
   - Re-present the summary
   - Repeat until the user is satisfied

4. **Initialize Hyperagent state:**

   The plugin operates as a self-referential hyperagent by default. Initialize state:
   ```python
   from pipeline_state import init_hyperagent_state
   save_state(..., hyperagent_state=init_hyperagent_state())
   ```
   Run `bash ${CLAUDE_PLUGIN_ROOT}/scripts/setup_hyperagent.sh` to verify the Hyperagents submodule and symlinks are ready.

   **Cross-session learning:** Check if prior meta-improvement patches exist in the plugin and load them:
   ```bash
   # Scan for promoted patches in the plugin
   grep -rl "# \[meta-improvement\]" ${CLAUDE_PLUGIN_ROOT}/skills/*/SKILL.md 2>/dev/null
   ```
   - For each skill file containing `# [meta-improvement]` markers: add the skill name to `hyperagent_state.active_meta_patches` so Phase 7's pre-loop meta-patch loading will include them in agent dispatch context.
   - If `claude-mem` MCP is available, query for prior hyperagent sessions: `mcp__plugin_claude-mem_mcp-search__search("hyperagent session meta-improvement")`
   - Inform the user: "Found {N} strategy improvements from prior sessions. These are active for this session."
   - Update hyperagent_state:
     ```python
     ha = init_hyperagent_state()
     ha["active_meta_patches"] = ["hp-tune-SKILL.md", ...]  # from grep results
     save_state(..., hyperagent_state=ha)
     ```

5. **Analyze codebase (still in plan mode):**

   **Do NOT exit plan mode yet.** Stay in plan mode and run Phase 1 steps 1-7 (read-only analysis):
   - Locate model code, training config, training script
   - Check GPU availability
   - Synthesize model understanding
   - Create optimization plan from `references/plan-template.md`
   - Estimate cost/time budget

   See `references/phase-1-understand.md` for the full workflow.

6. **Present full optimization plan to user:**

   Use `AskUserQuestion` to present the plan and offer choices:
   ```
   Here's my optimization plan based on your goals and the codebase analysis:

   **Model:** {model_type} ({framework})
   **Task:** {task_description}
   **HP search space:** {summary}
   **Estimated experiments:** {N} (across {gpu_count} GPUs)
   **Estimated GPU-hours:** {X}
   **Scope:** {scope_level}

   How many HP tuning batches between research rounds? (default: 3)

   Would you like to:
   1. Proceed with this plan
   2. Adjust scope, constraints, or budget (returns to discovery)
   3. Re-brainstorm strategy (explore different optimization directions)
   4. Ask questions about the approach
   ```

   - **Option 1 (proceed):** Continue to Step 7.
   - **Option 2 (adjust):** Go back to Step 3.7 — user refines answers, goals are re-written, codebase is re-analyzed if needed, plan is re-generated and re-presented.
   - **Option 3 (re-brainstorm):** Re-run `Skill("superpowers:brainstorming")` with the user's new direction, then re-generate the plan and re-present.
   - **Option 4 (questions):** Answer the user's questions, then re-present the same options.

   Store `hp_batches_per_round` (default: 3) in user_choices.

   **The user can loop through options 2, 3, and 4 as many times as they want.** Only option 1 advances the pipeline.

7. **Exit plan mode:**
   - Use `ExitPlanMode` only after the user chooses to proceed (option 1)
   - Proceed to Phase 2 (prerequisites)
