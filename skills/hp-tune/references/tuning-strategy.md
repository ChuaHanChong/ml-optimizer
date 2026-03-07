# Hyperparameter Tuning Strategy Guide

## Tuning Order (Priority)

Tune hyperparameters in this order, as earlier ones have the largest impact:

### 1. Learning Rate (Highest Impact)
- **Start:** Find the order of magnitude first (1e-2, 1e-3, 1e-4, 1e-5)
- **Refine:** Once the right order is found, search within that range (e.g., 1e-4 to 5e-4)
- **Common ranges by model type:**
  - Transformers: 1e-5 to 1e-3
  - CNNs: 1e-4 to 1e-2
  - Diffusion models: 1e-5 to 1e-4
  - Fine-tuning: 1e-6 to 1e-4
- **If training diverges:** LR is probably too high
- **If loss plateaus early:** LR might be too low

### 2. Batch Size (Constrained by GPU Memory)
- **Rule of thumb:** Larger batch = smoother gradients = can use higher LR
- **Linear scaling rule:** When doubling batch size, multiply LR by ~1.5-2x
- **Memory constraint:** Max batch size ≈ (GPU memory - model memory) / (per-sample memory)
- **Sweet spot:** Usually 16-64 for vision, 8-32 for NLP, 4-16 for high-resolution

### 3. Learning Rate Schedule
- **Cosine annealing:** Good default for most tasks
- **Warmup + cosine:** Best for transformers and large models
- **Step decay:** Simple, works well for CNNs
- **Warmup steps:** Usually 1-5% of total training steps
- **Minimum LR:** Usually 1e-6 or 0.01x initial LR

### 4. Weight Decay
- **Default range:** 0 to 0.1
- **Common values:** 0.01 (AdamW default), 0.05 (vision transformers)
- **Higher values:** Better generalization but may slow convergence
- **Skip for:** Batch norm parameters, bias terms

### 5. Architecture-Specific Parameters
- **Dropout:** 0.0 to 0.5 (start with 0.1)
- **Number of layers/channels:** Usually fixed, but can tune if computational budget allows
- **Attention heads:** Must divide hidden dim evenly

### 6. Data Augmentation (Lower Priority for HP Tuning)
- Usually better to get the right LR first, then tune augmentation
- Exception: if baseline is clearly overfitting
- **Vision:** Random crop, flip, color jitter, CutMix, MixUp
- **NLP:** Back-translation, synonym replacement, random deletion, token masking
- **Audio:** SpecAugment, time stretching, pitch shifting, noise injection
- **Tabular/Graph:** Feature dropout, edge dropout, node feature masking

## Reasoning Framework

When proposing new configs, reason about:

1. **What worked:** Which HPs led to the best results so far?
2. **What failed:** Which combinations caused divergence or poor results?
3. **Interpolation:** Try values between the best and second-best
4. **Extrapolation:** If the best was at the edge of the search space, extend it
5. **Interaction effects:** LR and batch size interact (linear scaling rule)
6. **Diminishing returns:** If last 3 experiments improved by <1%, consider stopping

### Branch-Aware Tuning

When experiments run on different code branches (from the implement skill), results must be analyzed per-branch:

- **Group by `code_branch`:** Before analyzing trends, partition results by their `code_branch` field. Experiments on `ml-opt/perceptual-loss` and experiments on baseline code are independent groups.
- **Analyze separately:** HP sensitivities may differ between branches. `lr=0.001` might be optimal on one branch but diverge on another due to different gradient magnitudes from code changes.
- **Don't cross-compare HPs:** A config that works well on branch A is not evidence that it will work on branch B. Only compare experiments within the same branch.
- **Branch performance ranking:** After sufficient experiments per branch, compare the *best result from each branch* to identify which code changes are most promising. Focus future HP tuning budget on the best-performing branches.

## Batch Sizing Strategy

When proposing N experiments (one per GPU):

- **Exploration batch (first 1-2 batches):** Wide spread across search space
  - e.g., LR in {1e-2, 1e-3, 1e-4, 1e-5} — cover order of magnitude
- **Exploitation batch (later batches):** Narrow focus around best results
  - e.g., if 1e-3 was best: try {5e-4, 8e-4, 1.2e-3, 2e-3}
- **Hybrid batch:** Mix exploration and exploitation
  - 2/3 configs near best result, 1/3 exploring new regions

## Multi-Objective Optimization

When optimizing for multiple metrics simultaneously (e.g., accuracy AND latency):

1. **Weighted scoring:** Combine metrics into a single score: `score = w1 * metric1_normalized + w2 * metric2_normalized`. Ask the user for relative weights.
2. **Pareto frontier:** Identify experiments where no other experiment is better on ALL metrics simultaneously. Present the Pareto-optimal set to the user.
3. **Constraint-based:** Optimize primary metric subject to secondary metric constraint (e.g., "maximize accuracy where latency < 100ms").
4. **Sequential:** First optimize the primary metric, then fine-tune the secondary without regressing.

When `secondary_metric` is provided, include both metrics in the ranking and note trade-offs.

## Multi-Loss Training

When the model uses multiple loss terms (e.g., reconstruction + perceptual + adversarial):

1. **Identify the dominant loss:** Which loss term contributes most to the total gradient? Start by tuning its weight.
2. **Loss weight tuning order:** Keep one loss fixed (usually the primary task loss) and tune weights of auxiliary losses.
3. **Relative scaling:** Auxiliary loss weights should typically be 0.001x–0.1x the primary loss magnitude.
4. **Diagnostic:** If one loss decreases while another increases, the weights are imbalanced. Monitor component losses alongside the combined total.

## Effective Hyperparameters

Some "code changes" are effectively HP-only changes:
- **Mixed precision training:** `torch.cuda.amp.autocast()` — no architecture change, just a training mode flag. Doubles effective batch size with same memory.
- **Gradient accumulation:** Simulates larger batch sizes without more memory. Effective_batch = batch_size × accumulation_steps.
- **Gradient clipping:** `torch.nn.utils.clip_grad_norm_()` — prevents explosion without changing architecture.

Consider these alongside traditional HPs when tuning.

## Per-Model-Type Guidance

### Vision Transformers (ViT, DeiT, Swin)
- Batch size: 256-4096 (use gradient accumulation)
- Weight decay: 0.05-0.3 (higher than CNNs)
- Warmup: 5-10% of training

### Large Language Models (Fine-tuning)
- LoRA rank: 8-64 (higher = more capacity but more memory)
- LR: 1e-5 to 3e-4 (very sensitive)
- Batch size: as large as possible with gradient accumulation

### GANs
- Generator LR: 1e-4 to 2e-4
- Discriminator LR: usually same or 2-4x generator
- DO NOT use weight decay on generator
- Betas: (0.0, 0.9) for Adam (not the default 0.9, 0.999)

### NLP / Language Models
- Batch size: largest that fits (gradient accumulation for effective batch 256+)
- LR: 1e-5 to 5e-5 (fine-tuning), 1e-4 to 1e-3 (pre-training)
- Warmup: 6–10% of total steps
- Weight decay: 0.01–0.1
- Key HP: sequence length (affects memory quadratically for attention models)

### Audio / Speech Models
- Batch size: often measured in seconds of audio, not samples
- LR: 1e-4 to 3e-4 typical
- Spectrogram parameters (n_fft, hop_length) are effectively HPs
- Data augmentation: SpecAugment (time/frequency masking)

### Graph Neural Networks
- Number of layers: 2–4 (over-smoothing with too many)
- Hidden dimension: 32–256
- Dropout: 0.3–0.6 (higher than vision/NLP due to small datasets)
- LR: 1e-3 to 1e-2

### Diffusion Models (DDPM, DDIM, Stable Diffusion)
- LR: 1e-5 to 3e-4 (1e-4 from scratch, 1e-5 for fine-tuning)
- Batch size: as large as fits — diffusion benefits from large batches
- EMA decay: 0.9999 (critical — do not skip)
- Noise schedule: cosine usually better than linear for fewer timesteps
- Warmup: 5000-10000 steps
- **Failure modes:** blurry outputs (LR too high or no EMA), mode collapse (batch too small), very slow convergence is NORMAL
- **Interactions:** T (timesteps) × LR, batch size × EMA decay

### Variational Autoencoders (VAE)
- LR: 1e-4 to 1e-3
- KL weight (beta): Start 0.0001, anneal to 1.0 (beta-VAE annealing)
- Latent dimension: 32-512 depending on data complexity
- **Failure modes:** posterior collapse (KL dominates too early → use annealing), blurry reconstructions (beta too high)
- **Interactions:** beta × latent_dim (higher dim tolerates higher beta), LR × beta (high LR + high beta → collapse)

### Tree-Based & Ensemble Models (XGBoost, LightGBM, RandomForest, GradientBoosting)

These models have fundamentally different hyperparameters. Batch size, dropout, and attention heads do not apply.

**Tuning order (priority):**
1. **n_estimators / num_boost_round** (100–10000): Use early stopping. Start with 500.
2. **max_depth** (3–12): Complexity control. XGBoost default 6.
3. **learning_rate / eta** (0.01–0.3): Boosting step size. 0.1 for exploration, 0.01–0.03 for final.
4. **min_samples_leaf / min_child_weight** (1–100): Leaf regularization. Start 5–20.
5. **subsample / bagging_fraction** (0.5–1.0): Row sampling. Default 0.8.
6. **colsample_bytree / feature_fraction** (0.3–1.0): Column sampling. Default 0.8.
7. **num_leaves** (LightGBM only, 15–255): ~2^max_depth.
8. **reg_alpha / reg_lambda** (0–10): L1/L2 regularization. Start at 0.

**Key differences from neural network tuning:**
- No batch size, no LR schedule, no dropout
- Early stopping replaces epoch counting: `early_stopping_rounds=50`
- Training is fast → run more experiments within budget
- Feature engineering often matters more than HP tuning

**Interaction effects:**
- `learning_rate` × `n_estimators`: lower LR needs more trees
- `max_depth` × `num_leaves`: don't set both high in LightGBM
- `subsample` × `colsample_bytree`: compound to reduce data per tree

**scikit-learn RandomForest/GradientBoosting:**
- `n_estimators` (100–2000), `max_depth` (5–30 or None), `min_samples_split` (2–20), `min_samples_leaf` (1–10)
- `max_features` ("sqrt", "log2", 0.3–1.0): Column sampling per split
- No learning_rate for RandomForest; GradientBoosting uses 0.01–0.3

## Anti-Patterns to Avoid

- Don't change all HPs at once — change 1-2 per experiment for interpretability
- Don't ignore failed experiments — they provide valuable information about boundaries
- Don't repeat identical configs (check past results first)
- Don't use extremely large LR just because loss is high — check if the metric is appropriate
- Don't tune past diminishing returns — know when to stop

## Constraint Handling

### Hard Constraints
Parameters that cannot be violated (will cause failure):
- **GPU memory:** Batch size × per-sample memory must fit. Proposals exceeding this should be rejected before dispatch.
- **Training time:** If the user set a max training time per experiment, estimate duration from baseline profiling and reject configs that would exceed it.

### Soft Constraints
Parameters the user prefers but can be relaxed if justified:
- **Frozen parameters:** User may say "don't change the optimizer". Respect unless analysis strongly suggests otherwise — then ask.
- **Search space bounds:** If the best result is at the boundary of the defined search space, propose extending the range and explain why.

### Constraint Propagation
When one constraint changes, propagate effects:
- Increasing batch size → may need to reduce model size or enable gradient checkpointing
- Enabling mixed precision → doubles effective batch size capacity → may want to increase batch size
- Changing optimizer → may need to re-tune learning rate from scratch
