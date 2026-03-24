# Phase 0: Discovery & Planning (MANDATORY)

**You MUST enter plan mode before doing any analysis or code exploration.**

1. **Enter plan mode:**
   - Use `EnterPlanMode` immediately when this skill is invoked
   - Do NOT skip this phase — even if the user provided a model path or description

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
   6. **Scope preference:**
      - HP tuning only (fastest, no code changes)
      - HP tuning + architecture research (slower, potentially bigger gains)
      - Let me decide based on analysis
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
   - **Complex optimization scenarios:** If the user's problem has multiple potential improvement
     vectors (e.g., training instability AND poor generalization AND slow convergence) or they're
     unsure about approach, consider using `superpowers:brainstorming` to systematically explore
     optimization strategies before committing to a plan.

3.5. **Write optimization goals:**
   After recording user responses, create the goal anchor file:
   ```bash
   python3 scripts/goal_memory.py <exp_root> init-goals '<goals_json>'
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
       "scope_level": "<from Q6: 'training' if HP-only, 'architecture' if HP+research, 'full' if let-me-decide>",
       "budget_mode": "<from budget selection>",
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
   This file persists in `experiments/optimization-goals.json` and is read by all agents before acting.

4. **Exit plan mode:**
   - Use `ExitPlanMode` once you have enough information to proceed
   - Summarize your understanding back to the user before moving on
