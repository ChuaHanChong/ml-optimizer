# Log and Data Formats

> **Note:** Metric names are task-dependent (accuracy, f1, psnr, bleu, etc.). The examples below use generic names.

## Experiment Result JSON (`experiments/results/<exp-id>.json`)
```json
{
  "exp_id": "exp-001",
  "status": "completed|failed|diverged|timeout|running|pending",
  "config": {
    "lr": 0.001,
    "batch_size": 16,
    "weight_decay": 0.01,
    "scheduler": "cosine",
    "epochs": 100,
    "custom_params": {}
  },
  "metrics": {
    "loss": 0.6789,
    "accuracy": 82.5,
    "val_loss": 0.7123
  },
  "gpu_id": 0,
  "duration_seconds": 3600,
  "log_file": "experiments/logs/exp-001/train.log",
  "script_file": "experiments/scripts/exp-001.sh",
  "code_branch": "ml-opt/perceptual-loss|null",
  "code_proposal": "Perceptual Loss Function|null",
  "method_tier": "baseline|method_default_hp|method_tuned_hp|null",
  "proposal_source": "paper|llm_knowledge|null",
  "iteration": 1,
  "notes": "Optional notes about this experiment"
}
```

## Baseline Result JSON (`experiments/results/baseline.json`)
```json
{
  "exp_id": "baseline",
  "status": "completed",
  "config": {
    "lr": 0.001,
    "batch_size": 32,
    "weight_decay": 0.01,
    "scheduler": "cosine",
    "epochs": 100
  },
  "metrics": {
    "loss": 1.0,
    "accuracy": 75.0
  },
  "profiling": {
    "gpu_memory_used_mib": 4200,
    "gpu_memory_total_mib": 24576,
    "throughput_samples_per_sec": 120.5,
    "estimated_max_batch_size": 128
  },
  "eval_command": "python eval.py --checkpoint best.pt",
  "train_command": "python train.py --config config.yaml",
  "notes": "Baseline evaluation - current model state"
}
```

## Dev Notes (`experiments/dev_notes.md`)

An append-only journal file. Each entry is dated:

```markdown
# Dev Notes

Session task log.

## 2025-01-15 — Baseline

- **Goal:** Establish baseline metrics
- loss: 1.0, accuracy: 75.0%
- **Next:** HP tuning

## 2025-01-15 — HP Tuning Iteration 1

- Best so far: exp-001 with loss=0.8
- Proposed: exp-002 (lr=0.0001), exp-003 (lr=0.01)
```

## Research Findings (`experiments/reports/research-findings.md`)
```markdown
# Research Findings

## Problem Statement
[Description of what we're trying to improve]

## Current Performance
[Baseline metrics]

## Sources Consulted
- [Paper/URL 1]: [Key takeaway]

## Proposals (Ranked by Priority)

### Proposal 1: [Name] (Priority: X/10)
- **Type:** code_change | hp_only
- **Source:** [Paper title and URL]
- **Technique:** [Category] - [Description]
- **What to change:**
  - [Specific file and function to modify]
  - [What the change looks like]
- **Expected improvement:** [X% on metric]
- **Complexity:** Low/Medium/High
- **Risk:** [What could go wrong]
- **Implementation steps:**
  1. [Step 1]
  2. [Step 2]
- **Implementation strategy:** from_scratch | from_reference
- **Reference repo:** [GitHub URL] (only for from_reference)
- **Reference files:** `path/to/relevant.py` (only for from_reference)

### Proposal 2: [Name] (Priority: Y/10)
...
```

Proposal types:
- `"code_change"`: Requires modifying model/training code (routed to implement skill)
- `"hp_only"`: Can be achieved via hyperparameter/config changes only (routed directly to hp-tune)

Priority score: `(impact * confidence) / (11 - min(feasibility, 10))` where impact, confidence, and feasibility are each scored 1-10.

## Pipeline State (`experiments/pipeline-state.json`)
```json
{
  "phase": 5,
  "iteration": 2,
  "running_experiments": ["exp-003", "exp-004"],
  "timestamp": "2025-01-15T14:30:00Z",
  "status": "running|interrupted|completed",
  "user_choices": {
    "primary_metric": "accuracy",
    "divergence_metric": "loss",
    "lower_is_better": false,
    "target_value": 0.95,
    "train_command": "python train.py --config config.yaml",
    "eval_command": "python eval.py --checkpoint best.pt"
  }
}
```

The `user_choices` field persists Phase 0 decisions so they survive orchestrator interruptions without re-asking the user.

## Batch Analysis (`experiments/reports/batch-<N>-analysis.md`)
```markdown
# Batch N Analysis

## Experiments in this batch
| Exp ID | Key Changes | Primary Metric | vs Baseline |
|--------|-------------|----------------|-------------|
| exp-001 | lr=0.001 | 30.2 | +0.5 |

## Key Findings
- [What worked]
- [What didn't]

## HP Impact Analysis
- [Which parameters had the most effect]

## Recommendation
- [Continue/pivot/stop] because [reason]
```
