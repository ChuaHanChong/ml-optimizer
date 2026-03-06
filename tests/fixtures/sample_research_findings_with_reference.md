# Research Findings

## Problem Statement
Improve image restoration quality for a diffusion-based model.

## Current Performance
- PSNR: 28.5 dB
- SSIM: 0.82
- Training time: 4 hours on 2x A100

## Sources Consulted
- [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://arxiv.org/abs/2111.09881): Multi-Dconv head transposed attention
- [NAFNet: Nonlinear Activation Free Network](https://arxiv.org/abs/2204.04676): Simple baseline with LayerNorm and channel attention

## Proposals (Ranked by Priority)

### Proposal 1: Multi-Dconv Head Transposed Attention (Priority: 9/10)
- **Type:** code_change
- **Source:** Restormer paper, https://arxiv.org/abs/2111.09881
- **Technique:** Architecture - Replace standard self-attention with multi-Dconv head transposed attention (MDTA) for efficient high-resolution feature processing
- **What to change:**
  - `models/attention.py`: Add MDTA module
  - `models/restoration_net.py`: Replace attention blocks with MDTA
- **Expected improvement:** +1.5 dB PSNR
- **Complexity:** Medium
- **Risk:** Requires careful kernel size selection for depthwise convolutions
- **Implementation steps:**
  1. Implement MDTA module with depthwise convolution-based Q/K/V projections
  2. Replace existing multi-head attention in transformer blocks
  3. Add kernel_size parameter to model config
- **Implementation strategy:** from_reference
- **Reference repo:** https://github.com/swz30/Restormer
- **Reference files:** `basicsr/models/archs/restormer_arch.py`, `basicsr/models/archs/local_arch.py`

### Proposal 2: SimpleGate Activation (Priority: 7/10)
- **Type:** code_change
- **Source:** NAFNet paper, https://arxiv.org/abs/2204.04676
- **Technique:** Architecture - Replace GELU activation with SimpleGate (channel split + element-wise product)
- **What to change:**
  - `models/blocks.py`: Add SimpleGate module and replace GELU in feed-forward blocks
- **Expected improvement:** +0.3 dB PSNR with lower compute cost
- **Complexity:** Low
- **Risk:** Minimal — simple drop-in replacement
- **Implementation steps:**
  1. Implement SimpleGate: split channels in half, multiply the two halves
  2. Replace GELU in feed-forward network blocks
  3. Adjust channel dimensions (SimpleGate halves output channels)
- **Implementation strategy:** from_scratch

### Proposal 3: Cosine Annealing with Warm Restarts (Priority: 6/10)
- **Type:** hp_only
- **Source:** SGDR: Stochastic Gradient Descent with Warm Restarts
- **Technique:** Training - Use cosine annealing with periodic warm restarts
- **What to change:**
  - `configs/train.yaml`: Change scheduler to CosineAnnealingWarmRestarts
- **Expected improvement:** +0.2 dB PSNR
- **Complexity:** Low
- **Risk:** May need to tune T_0 and T_mult parameters
- **Implementation steps:**
  1. Change scheduler config to CosineAnnealingWarmRestarts
  2. Set T_0 and T_mult in config
  3. Verify scheduler stepping is per-epoch
- **Implementation strategy:** from_scratch

## Recommendations
- **Quick wins (low complexity):** Proposal 2 (SimpleGate), Proposal 3 (Cosine Annealing)
- **High potential (medium complexity):** Proposal 1 (MDTA)
- **Ambitious (high complexity):** None

## Not Recommended
- Full Restormer architecture replacement: Too invasive for the current model
