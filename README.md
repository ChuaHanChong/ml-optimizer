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

- **Python 3.10+**
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
| `prerequisites` | Validates dataset format, prepares data, installs dependencies |
| `baseline` | Establishes baseline metrics and profiles GPU usage |
| `research` | Searches for ML techniques via web and paper analysis |
| `implement` | Applies research proposals as code changes with git isolation and validation |
| `hp-tune` | Proposes HP configs based on past results (LLM-driven) |
| `experiment` | Runs a single training experiment on a specific GPU |
| `monitor` | Watches training logs for divergence (NaN, explosion, plateau) |
| `analyze` | Post-batch analysis — ranks results, recommends next action |
| `report` | Generates comprehensive final optimization report |
| `review` | Self-improvement analysis and mid-pipeline course correction |

## Workflow

```
0. Discovery (plan mode, user Q&A — data paths, env manager)
1. Understand model (read code + config, check GPUs)
2. Prerequisites (validate dataset, prepare data, install deps)
3. Establish baseline
   -> User checkpoint: review baseline, choose direction
4. Optional research
   -> User checkpoint: review findings, pick proposals
4.5. Implement proposals (if code changes needed)
   -> Creates git branches, applies changes, validates
5. Experiment loop (autonomous, branch-aware):
   a. hp-tune proposes configs
   b. experiment runs training (parallel across GPUs)
   c. monitor watches for divergence
   d. analyze decides: continue/pivot/stop
   e. review auto-triggers after 3+ consecutive all-fail batches
6. Generate final report
   -> Optional review for self-improvement analysis
```

## Project Directory Structure

The plugin creates this structure in your project:

```
<project>/experiments/
  logs/<exp-id>/                        # Raw training logs
  reports/                              # Markdown reports (analysis, research, final)
  reports/error-log.json                # Structured error event log
  reports/suggestion-history.json       # Suggestion feedback loop
  reports/session-review.md             # Self-improvement review
  scripts/<exp-id>.sh                   # Generated training scripts
  results/prerequisites.json            # Prerequisites check report
  results/baseline.json                 # Baseline metrics and GPU profiling
  results/<exp-id>.json                 # Per-experiment results
  results/proposed-configs/             # HP config proposals from hp-tune
  results/implementation-manifest.json  # Validated proposal branches
  prepared-data/                        # Prepared dataset (if preprocessing needed)
  pipeline-state.json                   # Resumable pipeline state
  dev_notes.md                          # Running session log
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
| `pipeline_state.py` | Pipeline state save/load/validate and stale experiment cleanup |
| `schema_validator.py` | Validate experiment result, baseline, and manifest JSON schemas |
| `plot_results.py` | ASCII bar/line charts for metric comparison and HP sensitivity |
| `prerequisites_check.py` | Dataset format detection, environment validation, GPU-aware install commands |
| `error_tracker.py` | Error tracking, pattern detection, success metrics, suggestion ranking |

## Running Tests

```bash
cd ~/.claude/plugins/ml-optimizer
python -m pytest tests/ -v
```

## Agent Definitions

The plugin defines five subagent types in `agents/`:

- **research-agent** — Paper search and technique extraction
- **tuning-agent** — HP reasoning and config proposal
- **implement-agent** — Code change application and validation
- **experiment-agent** — Single experiment execution on a GPU
- **prerequisites-agent** — Dataset validation, environment setup, dependency installation
