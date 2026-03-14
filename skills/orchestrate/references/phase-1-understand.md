# Phase 1: Understand the Model

0. **Cross-session memory lookup** (optional but recommended):
   Before analyzing the codebase, use `claude-mem:mem-search` to search for past optimization sessions involving similar model types, tasks, or frameworks. This may surface:
   - HP ranges that worked well for similar models
   - Optimization techniques that succeeded or failed for this type of task
   - Common pitfalls encountered in previous sessions
   Use any relevant findings to inform the optimization plan (Phase 1, step 6).

1. **Locate model code:**
   - Use Glob to find Python files: `**/*.py`
   - Look for model definitions:
     - PyTorch: `nn.Module`, `torch.nn.Module`
     - Lightning: `LightningModule`, `pl.LightningModule`
     - TF/Keras: `tf.keras.Model`, `keras.Model`, `tf.Module`
     - JAX/Flax: `flax.linen.Module`, `nn.Module` (Flax)
     - HuggingFace: `PreTrainedModel`, `Trainer`
     - scikit-learn: `from sklearn`, `BaseEstimator`, `Pipeline`
     - XGBoost: `import xgboost`, `xgb.XGBClassifier`, `xgb.XGBRegressor`
     - LightGBM: `import lightgbm`, `lgb.LGBMClassifier`, `lgb.LGBMRegressor`
   - Look for training scripts (files with `train` in the name, `main.py`, `run.py`, etc.)

2. **Locate training config:**
   - Use Glob to find: `**/*.yaml`, `**/*.yml`, `**/*.json`
   - Look for config files with training parameters (lr, batch_size, epochs, etc.)

3. **Read key files:**
   - Read the model definition file(s)
   - Read the training config
   - Read the training script to understand the training loop

4. **Check GPU availability:**
   ```bash
   python3 ~/.claude/plugins/ml-optimizer/scripts/gpu_check.py
   ```

5. **Synthesize understanding:**
   - Model type and architecture
   - Task (classification, restoration, generation, etc.)
   - Current training setup (optimizer, scheduler, loss function)
   - Dataset information
   - Known metrics and current performance (if available)
   - **Tabular ML detection:** If the framework is scikit-learn, XGBoost, or LightGBM:
     - GPU check is optional (XGBoost/LightGBM may use GPU; scikit-learn does not)
     - Divergence monitoring is typically unnecessary (training is fast, no iterative loss to watch)
     - The experiment budget should use the CPU fallback: `max(num_gpus, 1) × 5`
     - Set `divergence_metric` to `null` (do not ask Q7) and skip the monitor skill during Phase 7. Divergence detection is only meaningful for frameworks with iterative training loops.
   - **RL detection:** If the codebase imports `gym`, `gymnasium`, `stable-baselines3`, `ray.rllib`, `tianshou`, or `cleanrl`:
     - Set `model_category = "rl"` in user_choices
     - The primary_metric is likely "reward" or "episode_return" — confirm with user
     - Divergence metric: use policy/value loss if logged; otherwise use reward with `divergence_lower_is_better = False`
     - Baseline eval: use average reward over N episodes (see baseline skill RL section)
     - Training is episodic — throughput is measured in steps/sec or episodes/hour
     - **Polarity validation:** After setting `divergence_metric` and `divergence_lower_is_better`, check for inconsistency: if the metric name contains "reward", "return", or "score" but `divergence_lower_is_better` is True, or if the metric name contains "loss", "error", "nll" but `divergence_lower_is_better` is False:
       - **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, auto-infer the correct polarity from the metric name (reward/return/score → `False`, loss/error/nll → `True`). Log to dev_notes: "Auto-inferred divergence polarity for '<metric_name>': lower_is_better=<value> (autonomous mode)". Skip AskUserQuestion.
       - **Otherwise:** Warn the user: "For reward-like metrics, divergence means the metric drops (lower_is_better=False). You set lower_is_better=True — confirm this is correct?" Use AskUserQuestion. This prevents silently killing experiments during improvement.
   - **Generative model detection:** If the codebase contains GAN discriminator/generator pairs, diffusion schedulers (`DDPMScheduler`, `noise_scheduler`), or VAE encoder/decoder with KL loss:
     - Set `model_category = "generative"` with sub-type `"gan"`, `"diffusion"`, or `"vae"`
     - For GANs: primary_metric is often FID or IS — confirm with user. Divergence metric: generator_loss or discriminator_loss
     - For diffusion: primary_metric is often FID or LPIPS. Divergence metric: denoising loss
     - For VAE: primary_metric is reconstruction quality. Watch for KL collapse (kl_term → 0)

6. **Create optimization plan:**
   - Read `references/plan-template.md` for the template structure
   - Fill in all sections based on your analysis AND the user's answers from Phase 0
   - Use the user's stated metric, target, and constraints — do not override them
   - Define the HP search space (informed by the user's scope preference)

7. **Estimate cost/time budget:**
   - Use the baseline profiling data (training time per experiment) and GPU count
   - Estimate: `total_experiments = num_branches × iterations × num_gpus`
   - Show the user: estimated total GPU-hours and wall-clock time
   - If the estimate exceeds the user's max training time constraint from Phase 0, warn and adjust

8. **Confirm plan with user:**
   **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, skip the confirmation below. Log the plan summary to `experiments/dev_notes.md` with a note: "Plan auto-approved (autonomous mode)." Proceed directly to Phase 2.

   **Otherwise:** Use AskUserQuestion to confirm the plan aligns with their Phase 0 answers:
   ```
   Based on your goals and my analysis, here is the optimization plan:
   [plan summary]

   Key decisions:
   - Primary metric: [metric from Phase 0]
   - Target: [target from Phase 0]
   - Search space: [summary]
   - Estimated experiments: [N]
   - Estimated total GPU-hours: [X]
   - Scope: [HP-only / HP + research, per Phase 0]

   Experiment budget mode:
   1. Auto (default): I'll judge the experiment budget based on task difficulty — [N] experiments estimated
   2. Autonomous (run until you interrupt — no experiment limit, continuous research every N batches)
   3. Custom: specify your own experiment limit

   If autonomous: How many HP tuning batches between research rounds? (default: 3)

   Does this match your expectations? Any adjustments?
   ```

   Store the budget mode as `budget_mode` in user_choices (`"auto"`, `"autonomous"`, or `"custom"`). Default: `"auto"`.

   **Note:** Autonomous mode skips ALL user checkpoints after Phase 0 discovery (Phase 4 direction, Phase 5 proposal selection, Phase 6 dependency/license approval, Pre-Loop method proposal selection). All decisions are auto-resolved with logging to dev_notes and error tracker for post-session review.

   If autonomous, also store `hp_batches_per_round` (default: 3). This controls how often the orchestrator auto-triggers a full research → implement cycle. Set to a higher value (e.g., 5-10) for slower research cadence.

   If custom, store `custom_budget` (user-specified number).

   **Adaptive budget calculation (auto mode):**

   In `"auto"` mode, the orchestrator judges task difficulty based on Phase 1 analysis and sets a `difficulty_multiplier`:

   | Difficulty | Multiplier | Criteria |
   |-----------|-----------|----------|
   | `easy`    | `× 8`    | Tabular ML (sklearn/XGBoost/LightGBM), ≤3 tunable HPs, HP-only (no code changes) |
   | `moderate`| `× 15`   | Standard supervised learning (CNN/MLP), moderate HP space (4-8 HPs), 1-2 research proposals |
   | `hard`    | `× 25`   | Complex architecture (transformers, GANs), large HP space (8+ HPs), 3+ proposals, RL, or generative tasks |

   Formula: `max_experiments = max(num_gpus, 1) × difficulty_multiplier`

   Store `difficulty` and `difficulty_multiplier` in user_choices. The user can override the budget at the Phase 4 checkpoint if they disagree with the assessment.

   **Budget by mode:**
   - `"auto"`: `max_experiments = max(num_gpus, 1) × difficulty_multiplier` (8, 15, or 25 based on assessed difficulty)
   - `"autonomous"`: `max_experiments = 999` (effectively unlimited — loop runs until user interrupts or context window exhaustion). In autonomous mode, the analyze skill's `"stop"` recommendation is logged but NOT enforced — the loop continues. The only hard stops are: user interruption, context window limit, or all reasonable approaches exhausted (analyze recommends stop 3 consecutive times). Research → implement cycles auto-trigger every `hp_batches_per_round` batches (see step 8).
   - `"custom"`: `max_experiments = custom_budget` (user-specified value)
