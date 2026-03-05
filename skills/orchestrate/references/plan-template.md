# Optimization Plan Template

## Goal
- **Primary metric:** [e.g., accuracy, loss, F1]
- **Target:** [e.g., improve accuracy from 85% to 90%]
- **Constraints:** [e.g., must fit in 24GB GPU, training < 48h]

## Model Overview
- **Type:** [e.g., CNN, ResNet, transformer]
- **Task:** [e.g., classification, detection, generation]
- **Framework:** [e.g., PyTorch, Lightning, custom pipeline]
- **Current performance:** [baseline metrics]

## Infrastructure
- **GPUs available:** [count and type]
- **GPU memory:** [per-GPU]
- **Estimated training time:** [per experiment]

## Approach
### Phase 1: Hyperparameter Tuning
- Learning rate range: [min, max]
- Batch size options: [constrained by GPU memory]
- Scheduler: [options]
- Regularization: [dropout, weight decay ranges]

### Phase 2: Architecture Changes (if applicable)
- [Change 1 from research]
- [Change 2 from research]

### Phase 3: Training Strategy
- [Data augmentation changes]
- [Loss function modifications]
- [Multi-stage training]

## Search Space
| Parameter | Range | Priority |
|-----------|-------|----------|
| lr | [1e-5, 1e-3] | High |
| batch_size | [4, 8, 16, 32] | Medium |
| weight_decay | [0, 1e-4, 1e-3] | Medium |
| scheduler | [cosine, step, linear] | Low |

## Success Criteria
- [ ] Beat baseline by >X% on primary metric
- [ ] Training stable (no divergence)
- [ ] Results reproducible
