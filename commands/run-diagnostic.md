---
name: run-diagnostic
description: "Run end-to-end diagnostics — validates plugin structure, dispatches all 10 agents, and runs a mini optimization pipeline on the test fixture."
---

# ML Optimizer End-to-End Diagnostic

You are running an end-to-end diagnostic of the ml-optimizer plugin. This validates plugin structure, all 10 agents dispatch correctly, and the full pipeline works end-to-end.

## Step 1: Run structural tests (pytest)

Run the plugin structure validation tests:

```bash
/data/hanchong/miniconda3/bin/python -m pytest tests/test_plugin_structure.py -v
```

Report any failures. If all pass, continue.

## Step 2: Agent dispatch smoke tests

Dispatch each of the 10 agents with a minimal smoke-test prompt. Run them in 2 batches of 5 for speed.

**Batch 1 — Procedural agents (model: sonnet):**

For each of these agents, dispatch with: "This is a smoke test. List your tools and confirm you can see your preloaded skill. Respond in 2-3 sentences."

1. `ml-optimizer:prerequisites-agent`
2. `ml-optimizer:baseline-agent`
3. `ml-optimizer:experiment-agent`
4. `ml-optimizer:monitor-agent`

**Batch 2 — Analytical agents (model: opus):**

1. `ml-optimizer:research-agent`
2. `ml-optimizer:implement-agent`
3. `ml-optimizer:tuning-agent`
4. `ml-optimizer:analysis-agent`
5. `ml-optimizer:report-agent`
6. `ml-optimizer:review-agent`

For each agent, verify:

- Agent resolves (no "not found" error)
- Agent lists its declared tools
- Agent confirms it can see its preloaded skill

Report results in a table.

## Step 3: Mini pipeline run (optional)

If all agents pass, ask the user:
"All 10 agents dispatched successfully. Run a mini optimization pipeline on the test fixture? (This takes ~5-10 min and tests the full Phase 2-7 flow.)"

If yes:

1. Copy `tests/fixtures/tiny_resnet_cifar10/` to `/tmp/tiny_resnet_cifar10_smoke_test/`
2. Init a git repo there
3. Run the optimization pipeline with:
   - primary_metric: loss
   - budget: 2 experiments (minimal)
   - scope: HP-only (no research/implement — faster)
   - env: conda base
4. Verify `experiments/` directory is created with baseline.json, exp-*.json, and final-report.md

## Step 4: Report

Summarize results:

```text
ML Optimizer End-to-End Diagnostic Results
================================
Structural tests: X/Y passed
Agent dispatch:   10/10 passed (or list failures)
Pipeline run:     [passed/skipped/failed]

Issues found: [none or list]
```
