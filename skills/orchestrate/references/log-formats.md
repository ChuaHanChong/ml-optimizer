# Log and Data Formats

> **Note:** Metric names are task-dependent (accuracy, f1, psnr, bleu, etc.). The examples below use generic names.

## Experiment Result JSON (`experiments/results/<exp-id>.json`)
```json
{
  "exp_id": "exp-001",
  "status": "completed|failed|diverged|running",
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
  "notes": "Optional notes about this experiment"
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

## Problem
[Description of the problem being solved]

## Sources Consulted
- [Paper/URL 1]: [Key takeaway]

## Proposals (ranked by expected impact)

### Proposal 1: [Name]
- **Technique:** [What to change]
- **Expected improvement:** [Estimate]
- **Complexity:** Low/Medium/High
- **Implementation:** [Brief description of code changes needed]

### Proposal 2: [Name]
...
```

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
