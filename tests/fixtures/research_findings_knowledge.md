# Research Findings — Method Proposals (Knowledge Mode)

## Summary

Three optimization proposals identified using LLM training knowledge, scoped to training-level changes.

### Proposal 1: Cosine Annealing with Warm Restarts (Priority: High)

**Complexity:** Low
**Implementation strategy:** from_scratch
**Proposal source:** llm_knowledge
**Confidence:** 7
**Impact:** 8
**Feasibility:** 9
**Scope:** training

**What to change:**
- `train.py` — replace step LR with cosine annealing warm restarts

**Implementation steps:**
1. Import CosineAnnealingWarmRestarts from torch.optim.lr_scheduler
2. Replace current scheduler with CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
3. Call scheduler.step() after each batch instead of each epoch

### Proposal 2: Label Smoothing Cross-Entropy (Priority: Medium)

**Complexity:** Low
**Implementation strategy:** from_scratch
**Proposal source:** llm_knowledge
**Confidence:** 6
**Impact:** 5
**Feasibility:** 9
**Scope:** training

**What to change:**
- `train.py` — use label smoothing in loss function

**Implementation steps:**
1. Replace nn.CrossEntropyLoss() with nn.CrossEntropyLoss(label_smoothing=0.1)
2. No other changes needed

### Proposal 3: Mixup Data Augmentation (Priority: Medium)

**Complexity:** Medium
**Implementation strategy:** from_scratch
**Proposal source:** llm_knowledge
**Confidence:** 7
**Impact:** 7
**Feasibility:** 7
**Scope:** architecture

**What to change:**
- `train.py` — add mixup augmentation to training loop
- `utils/mixup.py` — new mixup helper module

**Implementation steps:**
1. Create mixup helper function that interpolates pairs of inputs and labels
2. Apply mixup with alpha=0.2 during training
3. Use original (non-mixed) data for validation
