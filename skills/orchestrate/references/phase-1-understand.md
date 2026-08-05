# Phase 1: Understand the Model

**Runs within plan mode (entered in Phase 0). Do NOT exit plan mode here — Phase 0's planning loop (Step 6 presents the plan and handles refinement; Step 7 calls ExitPlanMode) handles plan presentation, user refinement, and ExitPlanMode.**

0. **Cross-session memory lookup** (optional but recommended):
   Before analyzing the codebase, use `claude-mem:mem-search` for past sessions with similar model types, tasks, or frameworks. May surface:
   - HP ranges that worked for similar models
   - Techniques that succeeded or failed for this task type
   - Common pitfalls from previous sessions
   Feed relevant findings into the optimization plan (step 6).

1. **Locate model code:**
   - Use Glob for Python files: `**/*.py`
   - Look for model definitions:
     - PyTorch: `nn.Module`, `torch.nn.Module`
     - Lightning: `LightningModule`, `pl.LightningModule`
     - TF/Keras: `tf.keras.Model`, `keras.Model`, `tf.Module`
     - JAX/Flax: `flax.linen.Module`, `nn.Module` (Flax)
     - HuggingFace: `PreTrainedModel`, `Trainer`
     - scikit-learn: `from sklearn`, `BaseEstimator`, `Pipeline`
     - XGBoost: `import xgboost`, `xgb.XGBClassifier`, `xgb.XGBRegressor`
     - LightGBM: `import lightgbm`, `lgb.LGBMClassifier`, `lgb.LGBMRegressor`
   - Look for training scripts (`train` in the name, `main.py`, `run.py`, etc.)

2. **Locate training config:**
   - Use Glob for: `**/*.yaml`, `**/*.yml`, `**/*.json`
   - Look for config files with training parameters (lr, batch_size, epochs, etc.)

3. **Read key files:**
   - Model definition file(s)
   - Training config
   - Training script (to understand the training loop)

4. **Check GPU availability:**
   ```bash
   # Local GPUs
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gpu_check.py
   # Remote GPUs (user_choices.remote): pass thresholds + host, else you profile the wrong machine
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gpu_check.py 30 80 <remote.host>
   ```

5. **Synthesize understanding:**
   - Model type and architecture
   - Task (classification, restoration, generation, etc.)
   - Current training setup (optimizer, scheduler, loss function)
   - Dataset information
   - Known metrics and current performance (if available)
   - **Tabular ML detection:** If the framework is scikit-learn, XGBoost, or LightGBM:
     - GPU check is optional (XGBoost/LightGBM may use GPU; scikit-learn does not)
     - Divergence monitoring is typically unnecessary (fast training, no iterative loss to watch)
     - Experiment budget follows the standard `num_configs` rule from Phase 7 (no tabular-specific multiplier)
     - Set `divergence_metric` to `null` (overriding the Q7 answer) and skip the monitor skill in Phase 7. Divergence detection is only meaningful for iterative training loops.
   - **RL detection:** If the codebase imports `gym`, `gymnasium`, `stable-baselines3`, `ray.rllib`, `tianshou`, `cleanrl`, `mujoco`, `dm_control`, `habitat`, `isaaclab`, `omni.isaac.lab`, `rsl_rl`, `skrl`, `brax`, `sample_factory`, or `robosuite`:
     - Set `model_category = "rl"` in user_choices
     - primary_metric is likely "reward" or "episode_return" — confirm with user
     - Divergence metric: policy/value loss if logged; otherwise reward with `divergence_lower_is_better = False`
     - Baseline eval: average reward over N episodes (see baseline skill RL section)
     - Training is episodic — throughput in steps/sec or episodes/hour
     - **Polarity validation:** After setting `divergence_metric` and `divergence_lower_is_better`, check for inconsistency: metric name contains "reward"/"return"/"score" but `divergence_lower_is_better` is True, or contains "loss"/"error"/"nll" but `divergence_lower_is_better` is False:
       - Auto-infer the correct polarity from the name (reward/return/score → `False`, loss/error/nll → `True`). Log to dev_notes: "Auto-inferred divergence polarity for '<metric_name>': lower_is_better=<value>". If ambiguous, use AskUserQuestion to confirm.
   - **Generative model detection:** If the codebase has GAN discriminator/generator pairs, diffusion schedulers (`DDPMScheduler`, `noise_scheduler`), or VAE encoder/decoder with KL loss:
     - Set `model_category = "generative"` with sub-type `"gan"`, `"diffusion"`, or `"vae"`
     - GANs: primary_metric is often FID or IS — confirm with user. Divergence metric: generator_loss or discriminator_loss
     - Diffusion: primary_metric is often FID or LPIPS. Divergence metric: denoising loss
     - VAE: primary_metric is reconstruction quality. Watch for KL collapse (kl_term → 0)

6. **Create optimization plan:**
   - Read `references/plan-template.md` for the template structure
   - Fill all sections from your analysis AND the user's Phase 0 answers
   - Use the user's stated metric, target, and constraints — do not override them
   - Define the HP search space (informed by the user's scope preference)

7. **Estimate cost/time budget:**
   - Use baseline profiling data (training time per experiment) and GPU count
   - Estimate: `total_experiments ≈ iterations × num_configs`, where `num_configs = num_gpus × experiments_per_gpu` (default 1 GPU-slot each; or 1 total for sequential file_backup projects). Code branches compete for these per-round slots (iteration 1 proposes one config per branch, capped at `num_configs`) rather than multiplying the round size.
   - Show the user estimated total GPU-hours and wall-clock time
   - If it exceeds the user's max training time from Phase 0, warn and adjust

8. **Return to Phase 0 planning loop:**
   After Steps 1-7, return to Phase 0 Step 6, which presents the full optimization plan and handles the multi-round refinement loop. The user can adjust scope, constraints, or budget and the plan is re-generated. Plan mode exits only when the user approves.
