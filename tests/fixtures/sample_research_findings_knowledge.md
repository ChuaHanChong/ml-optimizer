# Research Findings

## Problem Statement
Improve classification accuracy for a ResNet-based image classifier.

## Current Performance
- Accuracy: 85.2%
- Val Loss: 0.52
- Training time: 1 hour on 1x RTX 4090

## Sources Consulted
- LLM knowledge: well-established training practices for CNNs

## Proposals (Ranked by Priority)

### Proposal 1: Cosine Annealing with Warm Restarts (Priority: 7/10)
- **Proposal source:** llm_knowledge
- **Type:** code_change
- **Source:** Common best practice for CNN training (SGDR: Stochastic Gradient Descent with Warm Restarts)
- **Technique:** Training - Replace step LR schedule with cosine annealing warm restarts
- **What to change:**
  - `train.py`: Replace StepLR with CosineAnnealingWarmRestarts scheduler
- **Expected improvement:** +0.5-1% accuracy
- **Complexity:** Low
- **Risk:** May need to tune T_0 and T_mult parameters
- **Implementation steps:**
  1. Import CosineAnnealingWarmRestarts from torch.optim.lr_scheduler
  2. Replace existing StepLR scheduler with CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
  3. Update training loop to call scheduler.step() after each epoch
- **Implementation strategy:** from_scratch

### Proposal 2: Label Smoothing (Priority: 6/10)
- **Proposal source:** llm_knowledge
- **Type:** code_change
- **Source:** Well-known regularization technique (Rethinking the Inception Architecture)
- **Technique:** Regularization - Add label smoothing to cross-entropy loss
- **What to change:**
  - `train.py`: Add label_smoothing parameter to CrossEntropyLoss
- **Expected improvement:** +0.3-0.5% accuracy
- **Complexity:** Low
- **Risk:** Too much smoothing can hurt performance on clean labels
- **Implementation steps:**
  1. Set label_smoothing=0.1 in nn.CrossEntropyLoss
  2. Optionally make smoothing factor configurable
- **Implementation strategy:** from_scratch

### Proposal 3: Mixup Training (Priority: 5/10)
- **Proposal source:** llm_knowledge
- **Type:** code_change
- **Source:** Standard data augmentation technique (mixup: Beyond Empirical Risk Minimization)
- **Technique:** Augmentation - Apply mixup to training batches
- **What to change:**
  - `train.py`: Add mixup data augmentation in training loop
  - `utils/augment.py`: Create mixup utility function
- **Expected improvement:** +0.5-1% accuracy
- **Complexity:** Medium
- **Risk:** Must handle mixed labels correctly; only apply during training
- **Implementation steps:**
  1. Implement mixup function: interpolate between pairs of images and labels
  2. Apply mixup with probability 0.5 during training
  3. Use mixed labels for loss computation
- **Implementation strategy:** from_scratch

## Recommendations
- **Quick wins (low complexity):** Proposal 1 (Cosine Annealing), Proposal 2 (Label Smoothing)
- **Medium effort:** Proposal 3 (Mixup)
