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

Only `orchestrate` is directly invocable. All other skills have `disable-model-invocation: true` and are called internally via agents dispatched by the orchestrate skill.

| Skill | Description | User-facing |
|-------|-------------|-------------|
| `orchestrate` | Core coordinator — runs the full optimization pipeline | Yes |
| `prerequisites` | Validates dataset format, prepares data, installs dependencies | Internal |
| `baseline` | Establishes baseline metrics and profiles GPU usage | Internal |
| `research` | Searches for ML techniques via web and paper analysis | Internal |
| `implement` | Applies research proposals as code changes with git isolation and validation | Internal |
| `hp-tune` | Proposes HP configs based on past results (LLM-driven) | Internal |
| `experiment` | Runs a single training experiment on a specific GPU | Internal |
| `monitor` | Watches training logs for divergence (NaN, explosion, plateau) | Internal |
| `analyze` | Post-batch analysis — ranks results, recommends next action | Internal |
| `report` | Generates comprehensive final optimization report | Internal |
| `review` | Self-improvement analysis and mid-pipeline course correction | Internal |

## Workflow

```
0. Discovery (plan mode, user Q&A — data paths, env manager)
1. Understand model (read code + config, check GPUs)
2. Prerequisites (validate dataset, prepare data, install deps)
3. Establish baseline
4. User checkpoint: review baseline, choose direction
5. Research (web search + LLM knowledge for techniques)
6. Implement proposals (creates git branches, applies + validates code changes)
7. Experiment loop (autonomous, branch-aware):
   a. hp-tune proposes configs (or uses speculative proposals from prior batch)
   b. experiment runs training (parallel across GPUs)
   c. monitor watches for divergence (concurrent with experiments)
   d. analyze + speculative hp-tune decides: continue / pivot / stop
   e. method proposal trigger (if analyze recommends pivot)
   f. autonomous research cadence (periodic in autonomous mode)
   g. mid-pipeline review (auto-triggers on repeated failures)
8. Method stacking (if 5+ methods improved over baseline):
   -> Sequentially merges best methods, skip-on-failure, optional HP-tune per step
9. Generate final report
   -> Optional self-improvement review
```

## Project Directory Structure

The plugin creates this structure in your project:

```
<project>/experiments/
  logs/<exp-id>/                        # Raw training logs
  scripts/<exp-id>/                     # Per-experiment command scripts
  artifacts/<exp-id>/                   # Per-experiment artifacts (checkpoints, plots)
  results/prerequisites.json            # Prerequisites check report
  results/baseline.json                 # Baseline metrics and GPU profiling
  results/<exp-id>.json                 # Per-experiment results
  results/proposed-configs/             # HP config proposals from hp-tune
  results/implementation-manifest.json  # Validated proposal branches
  reports/                              # Markdown reports (analysis, research, final)
  reports/error-log.json                # Structured error event log
  reports/suggestion-history.json       # Suggestion feedback loop
  reports/session-review.md             # Self-improvement review
  prepared-data/                        # Prepared dataset (if preprocessing needed)
  pipeline-state.json                   # Resumable pipeline state
  dev_notes.md                          # Running session log
```

## Python Utilities

All scripts in `scripts/` use only the standard library and work as both importable modules and CLI tools:

| Script | CLI Usage |
|--------|-----------|
| `gpu_check.py` | `python3 scripts/gpu_check.py` |
| `parse_logs.py` | `python3 scripts/parse_logs.py <logfile>` — parses kv/JSON/CSV/XGBoost/HuggingFace Trainer logs |
| `detect_divergence.py` | `python3 scripts/detect_divergence.py '<json_values>' [--higher-is-better] [--model-category rl\|generative\|supervised]` |
| `result_analyzer.py` | `python3 scripts/result_analyzer.py <results_dir> <metric> [baseline_id] [lower_is_better]` |
| `experiment_setup.py` | `python3 scripts/experiment_setup.py <project_root> <train_command> [gpu_id] [config_json]` |
| `implement_utils.py` | `python3 scripts/implement_utils.py <findings.md> '<indices_json>'` — also: `clone <url> <dest>`, `analyze <path>` |
| `pipeline_state.py` | `python3 scripts/pipeline_state.py <exp_root> validate\|save\|load\|cleanup` |
| `schema_validator.py` | `python3 scripts/schema_validator.py <filepath> result\|baseline\|manifest\|prerequisites` |
| `plot_results.py` | `python3 scripts/plot_results.py <results_dir> <metric> comparison\|timeline\|sensitivity <hp>\|progress [--higher-is-better]` |
| `prerequisites_check.py` | `python3 scripts/prerequisites_check.py scan-imports\|check-packages\|detect-env\|detect-format\|detect-format-project\|validate-data\|bulk-install-cmd\|gpu-install-cmd` |
| `error_tracker.py` | `python3 scripts/error_tracker.py <exp_root> log\|show\|patterns\|summary\|sync\|success\|proposals\|rank\|cleanup\|log-suggestion\|suggestion-history` |

## Running Tests

```bash
cd ~/.claude/plugins/ml-optimizer
python -m pytest tests/ -v                          # all tests
python -m pytest tests/test_parse_logs.py -v         # single file
python -m pytest tests/ -m "not slow" -v             # skip real training tests
python -m pytest tests/test_e2e_pipeline.py -m slow  # real training E2E only
```

No build step. No linter. Python 3.10+ required. All scripts use only the standard library.

## Agent Definitions

Ten subagent types in `agents/`. The orchestrate skill dispatches agents directly via `Agent(subagent_type="ml-optimizer:<name>-agent")`.

| Agent | Tools | Model | Preloaded Skill |
|-------|-------|-------|-----------------|
| `research-agent` | WebSearch, WebFetch, Read, Write, Bash, Glob, Grep, Skill | inherited (ultrathink) | `ml-optimizer:research` |
| `implement-agent` | Bash, Read, Write, Edit, Glob, Grep, Skill, WebSearch, WebFetch | inherited (ultrathink) | `ml-optimizer:implement` |
| `tuning-agent` | Read, Write, Bash, Glob, Grep, Skill, WebSearch, WebFetch | inherited (ultrathink) | `ml-optimizer:hp-tune` |
| `analysis-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | inherited (ultrathink) | `ml-optimizer:analyze` |
| `report-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | inherited | `ml-optimizer:report` |
| `review-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | inherited (ultrathink) | `ml-optimizer:review` |
| `baseline-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | sonnet | `ml-optimizer:baseline` |
| `monitor-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | sonnet | `ml-optimizer:monitor` |
| `experiment-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | sonnet | `ml-optimizer:experiment` |
| `prerequisites-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | sonnet | `ml-optimizer:prerequisites` |

Analytical agents use "ultrathink" prompting and `model: opus`. Procedural agents use Sonnet for lower cost/latency.

## Hooks (Autonomous Mode Safeguards)

Lifecycle hooks in `hooks/` protect against common failure modes during autonomous operation:

| Hook | Event | Purpose |
|------|-------|---------|
| `bash-safety.sh` | PreToolUse (Bash) | Blocks `rm -rf /`, `git push --force`, `curl\|bash`, `chmod 777`, etc. |
| `file-guardrail.sh` | PreToolUse (Write/Edit) | Blocks writes to `.git/`, `.env`, credentials, lock files |
| `detect-critical-errors.sh` | PostToolUse (Bash) | Detects CUDA OOM, segfault, disk full; logs to error tracker |
| `pre-compact.sh` | PreCompact | Saves pipeline state snapshot to dev_notes before context compaction |
| `post-compact-context.sh` | SessionStart | Re-injects phase/metric/budget context after compaction |
| `subagent-stop-hook.sh` | SubagentStop | Reminds agent to validate outputs before stopping |
| Stop prompt | Stop | Verifies final report exists before session ends |

Exit code `2` = block action. Exit code `0` = allow. Configured in `hooks/hooks.json`.

## Key Design Patterns

- **Non-git fallback**: If the project isn't a git repo, file backups replace branch isolation. Experiments run sequentially.
- **Budget modes**: `auto` (difficulty-based: easy x8, moderate x15, hard x25 per GPU), `autonomous` (unlimited — stops after 3 consecutive stop recommendations), or custom integer.
- **Metric routing**: Monitor/divergence always uses loss. Analyze/hp-tune use the user's `primary_metric`.
- **Speculative hp-tune**: In Phase 7, hp-tune runs in background alongside analyze. If analyze says "continue", proposals are used immediately — eliminating GPU idle time.
- **OOM feedback loop**: When experiments OOM, batch size is recorded. Next hp-tune call receives `max_batch_size` to avoid re-proposing failing configs.
- **All-diverge recovery**: If all experiments in a batch diverge, a recovery batch with halved learning rates runs before stopping.
- **Research cadence**: In autonomous mode, research triggers every N batches. If no new proposals found, cadence doubles (exponential backoff).
- **Pipeline resumption**: `pipeline-state.json` persists phase, user choices, and stop count. On restart, stale experiments are cleaned up and phase gates prevent cascading failures.
- **Three-tier result tracking**: Experiments carry `method_tier` (baseline / method_default_hp / method_tuned_hp) and `proposal_source` (paper / llm_knowledge) for attribution analysis.
- **Method stacking**: After independent method testing, top methods are sequentially merged. Clean merges proceed; conflicts are LLM-resolved. Degrading combinations are skipped.

## Gotchas

- `detect_divergence.py` CLI takes a **JSON string**, not a file path: `'[0.5, 0.4, 100.0]'`
- `implement_utils.py` has **three CLI modes**: default (parse proposals), `clone <url> <dest>`, and `analyze <path>`
- **Metric routing is split**: monitor uses loss, analyze uses primary_metric. Mixing these up causes silent wrong behavior.
- **Branch experiments are independent**: results on `ml-opt/branch-a` don't predict what works on `ml-opt/branch-b`.
- **Tabular ML frameworks** (sklearn, XGBoost, LightGBM) skip divergence monitoring entirely.
- **Multiple research findings files**: `research-findings.md` (Phase 5), `research-findings-method-proposals.md` (pre-loop), `research-findings-method-proposals-iter<N>.md` (mid-loop). Deduplication checks all of them.

## License

MIT License. See [LICENSE](LICENSE) for details.
