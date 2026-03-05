# ml-optimizer

A Claude Code plugin that acts as an autonomous agent orchestrator for ML model optimization.

## Overview

The ml-optimizer plugin understands your ML model, establishes baselines, researches improvements, tunes hyperparameters, runs experiments (in parallel across GPUs), monitors for training divergence, and produces structured reports.

**Key design decisions:**
- LLM-driven hyperparameter tuning (Claude reasons about results — no Optuna/grid search)
- Research via web search + user-provided papers
- Log file polling for divergence monitoring
- Structured `experiments/` directory in your project
- User checkpoints after baseline and research; experiment loop is autonomous

## Prerequisites

### Required

- **Python 3.8+**
- **Claude Code** — the plugin runs inside Claude Code sessions
- **git** — used for branch isolation when implementing research proposals

### ML Training Dependencies

Your ML project will need its own training stack. The bundled test fixtures use:

```bash
pip install torch torchvision   # PyTorch (used by example model)
pip install pyyaml              # YAML config parsing (has fallback if missing)
```

The plugin's orchestration scripts (`scripts/`) use **only the Python standard library**, so they work regardless of your ML framework.

### Optional

| Dependency | What it enables |
|------------|-----------------|
| NVIDIA GPU + drivers | GPU profiling via `nvidia-smi`, parallel experiments |
| `pytest` | Running the plugin's test suite (`pip install pytest`) |
| Web search access | Research skill fetches recent papers and techniques |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | Set automatically by experiment scripts to assign GPUs |

## Quick Start

```
/optimize my model for better accuracy
```

Or invoke directly:
```
Use the ml-optimizer:orchestrate skill to optimize my model
```

## Skills

| Skill | Description |
|-------|-------------|
| `orchestrate` | Core coordinator — runs the full optimization pipeline |
| `baseline` | Establishes baseline metrics and profiles GPU usage |
| `research` | Searches for ML techniques via web and paper analysis |
| `implement` | Applies research proposals as code changes with git isolation and validation |
| `hp-tune` | Proposes HP configs based on past results (LLM-driven) |
| `experiment` | Runs a single training experiment on a specific GPU |
| `monitor` | Watches training logs for divergence (NaN, explosion, plateau) |
| `analyze` | Post-batch analysis — ranks results, recommends next action |
| `report` | Generates comprehensive final optimization report |

## Workflow

```
1. Understand model (read code + config)
2. Establish baseline
   -> User checkpoint: review baseline, choose direction
3. Optional research
   -> User checkpoint: review findings, pick proposals
3.5. Implement proposals (if code changes needed)
   -> Creates git branches, applies changes, validates
4. Experiment loop (autonomous, branch-aware):
   a. hp-tune proposes configs
   b. experiment runs training (parallel across GPUs)
   c. monitor watches for divergence
   d. analyze decides: continue/pivot/stop
5. Generate final report
```

## Project Directory Structure

The plugin creates this structure in your project:

```
<project>/experiments/
  logs/<exp-id>/          # Raw training logs
  reports/                # All Markdown reports (analysis, research findings, final report)
  scripts/<exp-id>.sh     # Bash scripts used
  results/<exp-id>.json   # Parsed metrics
  dev_notes.md            # Running log of session tasks by date
```

## Python Utilities

Located in `scripts/`:

| Script | Purpose |
|--------|---------|
| `gpu_check.py` | Parse nvidia-smi, identify free GPUs |
| `parse_logs.py` | Parse training logs (kv, JSON, CSV formats) |
| `detect_divergence.py` | Detect NaN, loss explosion, plateaus |
| `result_analyzer.py` | Compare experiments, rank by metric, find HP correlations |
| `experiment_setup.py` | Create experiment directories, generate IDs and scripts |
| `implement_utils.py` | Proposal parsing, branch management, syntax validation, manifest writing |

## Running Tests

```bash
cd ~/.claude/plugins/ml-optimizer
python -m pytest tests/ -v
```

## Agent Definitions

The plugin defines three subagent types in `agents/`:

- **research-agent** — Paper search and technique extraction
- **tuning-agent** — HP reasoning and config proposal
- **implement-agent** — Code change application and validation
- **experiment-agent** — Single experiment execution on a GPU
