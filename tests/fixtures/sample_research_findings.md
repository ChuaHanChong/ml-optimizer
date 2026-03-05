# Research Findings

## Problem Statement
Improve classification accuracy and reduce validation loss for a CNN-based model.

## Current Performance
- Accuracy: 82.5%
- Val Loss: 0.71
- Training time: 2 hours on 1x A100

## Sources Consulted
- [Perceptual Losses for Real-Time Style Transfer](https://arxiv.org/abs/1603.08155): VGG-based perceptual loss
- [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257): Swin attention blocks
- [CutMix: Regularization Strategy](https://arxiv.org/abs/1905.04899): Region-based augmentation

## Proposals (Ranked by Priority)

### Proposal 1: Perceptual Loss Function (Priority: 8/10)
- **Source:** Perceptual Losses for Real-Time Style Transfer
- **Technique:** Loss - Add VGG-based perceptual loss alongside existing cross-entropy loss
- **What to change:**
  - `models/classifier.py`: Add perceptual loss computation
  - `configs/train.yaml`: Add perceptual loss weight parameter
- **Expected improvement:** +2% accuracy
- **Complexity:** Low
- **Risk:** Slight increase in training memory; VGG features must be on same device
- **Implementation steps:**
  1. Add VGG feature extractor module (frozen, no grad)
  2. Compute perceptual loss between predicted and target features
  3. Combine with existing cross-entropy loss using configurable weight

### Proposal 2: Swin Transformer Attention Blocks (Priority: 6/10)
- **Source:** SwinIR paper
- **Technique:** Architecture - Replace mid-block attention with Swin Transformer blocks
- **What to change:**
  - `models/attention.py`: Add SwinTransformerBlock class
  - `models/classifier.py`: Integrate Swin blocks into backbone
- **Expected improvement:** +1.5% accuracy
- **Complexity:** High
- **Risk:** Significant architectural change; may require tuning window size
- **Implementation steps:**
  1. Implement SwinTransformerBlock with window attention and shifted windows
  2. Replace self-attention in mid-block with Swin attention
  3. Add window_size parameter to model config
  4. Verify output shapes match existing architecture

### Proposal 3: CutMix Data Augmentation (Priority: 7/10)
- **Source:** CutMix paper
- **Technique:** Augmentation - Apply CutMix to training image pairs
- **What to change:**
  - `data/dataset.py`: Add CutMix transform to training pipeline
- **Expected improvement:** +1% accuracy
- **Complexity:** Low
- **Risk:** Must only apply during training, not validation
- **Implementation steps:**
  1. Implement CutMix function that blends rectangular regions between image pairs
  2. Add CutMix to training dataloader with configurable probability
  3. Guard augmentation with training-only flag

## Recommendations
- **Quick wins (low complexity):** Proposal 1 (Perceptual Loss), Proposal 3 (CutMix)
- **High potential (medium complexity):** None
- **Ambitious (high complexity):** Proposal 2 (Swin Attention)

## Not Recommended
- MixUp augmentation: Not as effective as CutMix for this architecture
