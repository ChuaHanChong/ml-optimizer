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

## Reasoning Framework

When proposing new configs, reason about:

1. **What worked:** Which HPs led to the best results so far?
2. **What failed:** Which combinations caused divergence or poor results?
3. **Interpolation:** Try values between the best and second-best
4. **Extrapolation:** If the best was at the edge of the search space, extend it
5. **Interaction effects:** LR and batch size interact (linear scaling rule)
6. **Diminishing returns:** If last 3 experiments improved by <1%, consider stopping

## Batch Sizing Strategy

When proposing N experiments (one per GPU):

- **Exploration batch (first 1-2 batches):** Wide spread across search space
  - e.g., LR in {1e-2, 1e-3, 1e-4, 1e-5} — cover order of magnitude
- **Exploitation batch (later batches):** Narrow focus around best results
  - e.g., if 1e-3 was best: try {5e-4, 8e-4, 1.2e-3, 2e-3}
- **Hybrid batch:** Mix exploration and exploitation
  - 2/3 configs near best result, 1/3 exploring new regions

## Anti-Patterns to Avoid

- Don't change all HPs at once — change 1-2 per experiment for interpretability
- Don't ignore failed experiments — they provide valuable information about boundaries
- Don't repeat identical configs (check past results first)
- Don't use extremely large LR just because loss is high — check if the metric is appropriate
- Don't tune past diminishing returns — know when to stop
