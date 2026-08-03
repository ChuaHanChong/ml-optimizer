# Phase 0: Discovery & Planning (MANDATORY)

**You MUST enter plan mode before any analysis or code exploration.**

1. **Enter plan mode:**
   - Use `EnterPlanMode` immediately when this skill is invoked.
   - Do NOT skip this phase — even if the user gave a model path or description.

1.1. **Check for a completed prior run:**
   If the breadcrumbed `<exp_root>` has a `pipeline-state.json` with `phase == 9`, ask via `AskUserQuestion`:

     > *This `<exp_root>` already has a completed run. I'll create a new directory for your new direction — the previous run stays untouched. Sound good? (Or say "continue here" to build on the existing state, or "resume" to re-enter the completed run.)*

   - `phase < 9` → normal resume, proceed without asking.
   - No `pipeline-state.json` → fresh workspace, proceed normally.
   - User said "continue" or "resume" → skip this check.

2. **Ask discovery questions:**
   Use `AskUserQuestion` to gather the following (combine into one organized prompt):

   ```
   Before optimizing, I need your goals and constraints:

   1. **Optimization target:** Which metric to improve? (e.g., accuracy, loss, F1, BLEU, latency)
   1a. **Secondary metrics** (optional): Additional metrics to track alongside the primary. For each collect: name, whether lower is better, and role — "guardrail" (top-ranked results must not regress it) or "report" (informational only). Store as `secondary_metrics` (list of `{name, lower_is_better, role}`) in user_choices; empty if none.
   1b. **Evaluation tasks** (optional): If the model is evaluated on more than one task or environment, collect the task names. Store as `eval_tasks` (list of strings) in user_choices; empty list if single-task.

   **State plainly to the user:** the plugin does not run tasks itself — it parses whatever `eval_command` prints. For multi-task to work, `eval_command` must emit one key per task named `<primary_metric>_<task>` (e.g. `success_rate_pick=0.83 success_rate_place=0.61`). The plugin then computes the mean under `<primary_metric>` and the worst task under `<primary_metric>_worst`. If `eval_command` cannot do this, leave `eval_tasks` empty — a wrong task list produces aggregates over the wrong denominator, silently making runs incomparable.

   Task names may not be `worst`, `std`, or `mean` — those suffixes are reserved for aggregates.
   If the eval also reports a per-task standard deviation, order it `<primary_metric>_std_<task>` (e.g. `success_rate_std_pick`) — never `<primary_metric>_<task>_std`, which would be misread as an undeclared task named `"<task>_std"`.

   **Suggest a guardrail:** if `eval_tasks` is non-empty, offer to add `{"name": "<primary_metric>_worst", "lower_is_better": <same as primary>, "role": "guardrail"}` to `secondary_metrics`. This flags a config that lifts the mean while tanking one task — the failure mode multi-task evaluation exists to catch.
   2. **Current performance:** Current value of that metric? (if known)
   3. **Target performance:** Target value? (or "as good as possible")
   4. **Constraints:**
      - Max training time per experiment?
      - GPU memory limit? (or auto-detect?)
      - Parameters you do NOT want changed?
   5. **Prior attempts:** Optimizations already tried? What worked/didn't?
   6. **Scope preference** _(default: full autonomous optimization)_:
      - HP tuning only (fastest, no code changes)
      - HP tuning + research (web search for methods, no evolutionary code refinement)
      - Full autonomous optimization (HP + research + ShinkaEvolve — **default if not specified**)
   7. **Divergence metric name** _(skip for scikit-learn, XGBoost, LightGBM — single fit() call, no iterative loss stream)_: Which metric to monitor for training divergence? (default: "loss"; alternatives: "train_loss", "val_loss", "objective", "nll_loss", "perplexity" for LLMs). For RL: use policy/value loss if logged; if only reward is logged, set divergence_metric to the reward metric name — the monitor skill uses reward-based heuristics (higher-is-better divergence detection).
   7a. **Divergence polarity** _(auto-inferred, confirm if ambiguous)_:
       Infer polarity from the Q7 metric name:
       - Contains "loss", "error", "nll", "objective", "perplexity" → `divergence_lower_is_better = True`
       - Contains "reward", "accuracy", "psnr", "ssim", "f1", "auc", "return" → `divergence_lower_is_better = False`
       - No match → ask: "Is a lower value of [metric] better (like loss) or higher better (like reward)?"
       Store as `divergence_lower_is_better` in user_choices.
   8. **Optimization type:** Training or inference performance? (This plugin focuses on **training** — inference optimization like quantization, pruning, ONNX conversion is out of scope.)
   9. **Training budget per experiment** (optional):
      - **Fixed time**: all experiments train the same wall-clock duration (e.g., "60 seconds each") — metrics directly comparable by time
      - **Fixed epochs**: all experiments train the same number of epochs (e.g., "10 epochs each") — deterministic and reproducible
      - **Fixed environment steps (RL)**: all experiments train the same number of environment timesteps (e.g., "500000 steps each") — the natural RL budget unit. If `model_category` is `rl`, steer the user here: epochs are usually meaningless for online RL
      - **Default** (blank): run until configured epochs complete, with a safety timeout
      Store as `fixed_time_budget` (seconds), `fixed_epoch_budget` (integer), or `fixed_step_budget` (integer environment timesteps) in user_choices.
   9a. **Seeds per config** (optional; suggested for RL, where single-seed results are noisy): How many random seeds per configuration? (default: 1). Values > 1 measure run-to-run variance — the measured spread becomes the noise floor for analysis decisions. Store as `seeds_per_config` (integer) in user_choices.
   10. **Anything else** about this model or training setup?
   11. **Dataset location:** Where are your training and validation datasets?
       - Directory path(s), or "embedded in code" if the training script downloads/generates data
       - Or: "no dataset — training interacts with a simulator/RL environment" (online RL). Set `train_data_path` and `val_data_path` to `null` in user_choices; prerequisites skips dataset validation and records format `rl_environment`.
   12. **Environment:** Which environment manager?
       - conda (environment name?) / uv / pip / venv / poetry / other
   ```

3. **Record user responses:**
   - Store the answers — they guide every subsequent phase.
   - If the user is unsure on some, note those as Phase 1 investigation areas.

3.1. **Write experiment root breadcrumb and optimization goals:**
   First write a breadcrumb so hooks can find `<exp_root>` (even on a different mount). The breadcrumb supports multiple runs — each new run appends to `runs[]` and sets `active`:
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
   `<goals_json>` is built from the user's answers:
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
       "fixed_epoch_budget": "<from Q9: integer, or null>",
       "fixed_step_budget": "<from Q9: integer environment timesteps, or null>"
     },
     "divergence": {
       "metric": "<from Q7>",
       "lower_is_better": "<from Q7a>"
     }
   }
   ```
   Persists in `<exp_root>/optimization-goals.json`, read by all agents before acting.

3.2. **Brainstorm optimization strategy:**
   Use `Skill("superpowers:brainstorming")` to explore the optimization space with the user — surfacing non-obvious approaches, trade-offs, and priorities before committing to a plan. It explores:
   - Most likely bottlenecks? (data, model capacity, training recipe, regularization)
   - Which trade-offs matter? (speed vs accuracy, simplicity vs performance)
   - Risk appetite? (conservative HP tuning vs aggressive code mutations)
   - Domain-specific techniques worth prioritizing?

   The output informs the plan's scope, search space, and strategy.

3.3. **Present understanding and invite refinement:**
   Summarize back to the user:
   ```
   Here's what I understood:
   - Metric: {primary_metric} ({direction}), target: {target_value}
   - Scope: {scope_level} ({explanation})
   - Model category: {model_category} (auto-detected in Phase 1 — please confirm; drives RL/generative-specific monitoring, budgets, and evaluation)
   - Budget: {budget_description}
   - Constraints: {frozen_params, time limits, etc.}
   - Dataset: {location}
   - Environment: {env_manager} ({env_name})
   - Strategy insights from brainstorming: {key_insights}

   Adjust anything before I analyze the codebase?
   ```
   Use `AskUserQuestion`. If the user wants changes:
   - Update the relevant user_choices.
   - Re-write `optimization-goals.json` via `goal_memory.py init-goals` with updated values.
   - Re-run brainstorming if exploring a different direction.
   - Re-present the summary.
   - Repeat until satisfied.

4. **Analyze codebase (still in plan mode):**

   **Do NOT exit plan mode yet.** Run Phase 1 steps 1-7 (read-only analysis):
   - Locate model code, training config, training script
   - Check GPU availability
   - Synthesize model understanding
   - Create optimization plan from `references/plan-template.md`
   - Estimate cost/time budget

   See `references/phase-1-understand.md` for the full workflow.

6. **Present full optimization plan to user:**

   Use `AskUserQuestion` to present the plan and offer choices:
   ```
   Here's my optimization plan from your goals and the codebase analysis:

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
   - **Option 3 (re-brainstorm):** Re-run `Skill("superpowers:brainstorming")` with the user's new direction, then re-generate and re-present the plan.
   - **Option 4 (questions):** Answer the user's questions, then re-present the same options.

   Store `hp_batches_per_round` (default: 3) in user_choices.

   **The user can loop through options 2, 3, and 4 as many times as they want.** Only option 1 advances the pipeline.

7. **Exit plan mode:**
   - Use `ExitPlanMode` only after the user chooses to proceed (option 1).
   - Proceed to Phase 2 (prerequisites).
