# Phase 1: Understand the Model

**This phase runs within plan mode (entered in Phase 0). Do NOT exit plan mode here — Phase 0's planning loop (Step 6) handles plan presentation, user refinement, and ExitPlanMode.**

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
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gpu_check.py
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
       - Auto-infer the correct polarity from the metric name (reward/return/score → `False`, loss/error/nll → `True`). Log to dev_notes: "Auto-inferred divergence polarity for '<metric_name>': lower_is_better=<value>". If the auto-inference is ambiguous, use AskUserQuestion to confirm with the user.
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

8. **Return to Phase 0 planning loop:**
   After Steps 1-7, return to Phase 0 Step 6 which presents the full optimization plan to the user and handles the multi-round refinement loop. The user can adjust scope, constraints, or budget and the plan will be re-generated. Plan mode is exited only when the user approves.
