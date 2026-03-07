# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A Claude Code plugin that orchestrates autonomous ML model optimization. It dispatches specialized agents for research, hyperparameter tuning, experiment execution, and result analysis. The plugin uses LLM-driven HP tuning (Claude reasons about results directly — no Optuna/grid search).

## Usage

In a Claude Code session, type:
```
/optimize <model-path-or-description>
```
This invokes `commands/optimize.md`, which delegates to the `ml-optimizer:orchestrate` skill.

## Running Tests

```bash
python -m pytest tests/ -v            # all tests
python -m pytest tests/test_parse_logs.py -v   # single file
python -m pytest tests/test_parse_logs.py::test_name -v  # single test
```

No build step. No linter configured. Python 3.10+ required. The `scripts/` directory uses only the Python standard library.

## Architecture

### Plugin Structure

```
.claude-plugin/plugin.json  — Plugin metadata (name, version)
commands/optimize.md        — /optimize slash command (entry point)
skills/                     — Skill definitions (SKILL.md files)
agents/                     — 5 subagent definitions
scripts/                    — Python utilities (stdlib only)
memory/                     — Placeholder for future cross-project patterns
tests/                      — pytest test suite
```

### Skill Pipeline (Orchestrator Flow)

The `orchestrate` skill coordinates an 8-phase pipeline. Each skill is invoked via `ml-optimizer:<skill-name>`:

```
Phase 0: Discovery (plan mode, user Q&A — includes data paths and env manager)
Phase 1: Understand model (read code, check GPUs)
Phase 2: prerequisites → Validate dataset format, prepare data, install dependencies
Phase 3: baseline → Establish baseline metrics
Phase 4: User checkpoint
Phase 5: research → Find techniques via web/papers
Phase 5.5: implement → Apply proposals as git branches
Phase 6: Experiment loop (autonomous):
         hp-tune → propose configs
         experiment → run training (parallel across GPUs)
         monitor → watch for divergence
         analyze → decide continue/pivot/stop
         review → Mid-pipeline review (auto-triggered after 3+ consecutive all-fail batches)
Phase 7: report → Final optimization report
         review → Self-improvement analysis (optional, end-of-session)
```

### Metric Routing Rule

Always monitor `"loss"` for divergence detection (monitor skill). Use the user's `primary_metric` (accuracy, PSNR, F1, etc.) for analyze and hp-tune skills.

### Branch-Aware Experiments

The implement skill creates `ml-opt/<slug>` branches per research proposal. The experiment loop tests each branch with HP variations. The tuning agent groups results by `code_branch` — identical HPs on different branches are treated as independent experiments.

### Agent Definitions (`agents/`)

Five subagent types with specified tool access:
- **research-agent**: WebSearch, WebFetch, Read, Write, Bash, Glob, Grep
- **tuning-agent**: Read, Write, Bash, Glob, Grep
- **implement-agent**: Bash, Read, Write, Edit, Glob, Grep, WebFetch
- **experiment-agent**: Bash, Read, Write, Glob, Grep
- **prerequisites-agent**: Bash, Read, Write, Glob, Grep

Analytical agents (hp-tune, research, analyze, implement) use "ultrathink" prompting. Procedural agents (experiment, monitor, prerequisites) do not.

### Python Scripts (`scripts/`)

All scripts work as both importable modules and CLI tools:

| Script | CLI Usage |
|--------|-----------|
| `gpu_check.py` | `python3 scripts/gpu_check.py` — parse nvidia-smi |
| `parse_logs.py` | `python3 scripts/parse_logs.py <logfile>` — parse kv/JSON/CSV logs |
| `detect_divergence.py` | `python3 scripts/detect_divergence.py '<json_values>'` — detect NaN/explosion/plateau |
| `result_analyzer.py` | `python3 scripts/result_analyzer.py <results_dir> <metric> [baseline_id] [lower_is_better]` |
| `experiment_setup.py` | Generates experiment IDs and directory structure |
| `implement_utils.py` | `python3 scripts/implement_utils.py <findings.md> '<indices_json>'` — parse proposals; also `clone <url> <dest>` and `analyze <path>` subcommands |
| `pipeline_state.py` | `python3 scripts/pipeline_state.py <exp_root> validate|save|load|cleanup` |
| `schema_validator.py` | Validates JSON result files against expected schemas |
| `plot_results.py` | Generates result visualizations |
| `prerequisites_check.py` | `python3 scripts/prerequisites_check.py scan-imports\|check-packages\|detect-env\|detect-format\|detect-format-project\|validate-data\|bulk-install-cmd\|gpu-install-cmd` — dataset, environment, and GPU-aware install validation |
| `error_tracker.py` | `python3 scripts/error_tracker.py <exp_root> log\|show\|patterns\|summary\|sync\|success\|proposals\|rank\|cleanup\|log-suggestion\|suggestion-history` — error tracking, pattern detection, success metrics, proposal outcomes, suggestion ranking, suggestion history |

### State & Output (in target project)

The plugin creates `experiments/` in the user's project:
```
experiments/
  results/prerequisites.json         — Prerequisites check report
  results/baseline.json              — Baseline metrics
  results/exp-*.json                 — Per-experiment results
  results/implementation-manifest.json — Validated proposal branches
  results/proposed-configs/          — HP config proposals
  prepared-data/                     — Prepared dataset (if preprocessing needed)
  pipeline-state.json                — Resumable pipeline state
  logs/<exp-id>/train.log            — Raw training logs
  reports/                           — Analysis reports, research findings
  reports/error-log.json             — Structured error event log
  reports/suggestion-history.json    — Suggestion feedback loop (tracks what was suggested)
  reports/session-review.md          — Self-improvement review (from review skill)
  scripts/<exp-id>.sh                — Generated training scripts
  dev_notes.md                       — Running session log
```

### Pipeline Resumption

The orchestrator can be stopped and resumed. On restart it reads `pipeline-state.json` and uses `cleanup_stale()` to handle interrupted experiments (marks them as failed after a timeout). Phase validation via `validate_phase_requirements()` prevents cascading failures. Pipeline state persists Phase 0 user choices (`primary_metric`, `divergence_metric`, `divergence_lower_is_better`, `lower_is_better`, `target_value`, `train_command`, `eval_command`, `train_data_path`, `val_data_path`, `prepared_train_path`, `prepared_val_path`, `env_manager`, `env_name`, `model_category`) via `save_state(user_choices={...})` so they survive interruptions without re-asking the user.

## Key Design Patterns

- **Non-git fallback**: If the target project isn't a git repo, the implement skill uses file backups instead of branches. This forces sequential (not parallel) experiment execution.
- **Experiment budget**: Default max experiments = `max(num_gpus, 1) * 5`. When `num_gpus=0` (CPU-only, e.g., scikit-learn/XGBoost), the budget is `1 * 5 = 5` experiments. The orchestrator passes `remaining_budget` to hp-tune, which caps proposals at `min(max(num_gpus, 1), remaining_budget)` to prevent overshoot. The analyze skill recommends stop when diminishing returns detected.
- **Proposal priority scoring**: `(impact * confidence) / (11 - min(feasibility, 10))` — feasibility clamped to [1,10] to prevent division by zero.
- **Spearman correlation**: `result_analyzer.py` uses rank correlation with average-rank tie-breaking to identify HP-metric relationships (no scipy dependency).
- **Dual implementation strategy**: Research proposals include an `implementation_strategy` field (`from_scratch` or `from_reference`). The implement agent dispatches accordingly — either implementing from paper descriptions (Section 8) or cloning and adapting reference repos (Section 9). Strategy is decided by the research agent based on repo availability and quality.

## Test Fixtures

`tests/fixtures/` contains a minimal PyTorch project (`tiny_resnet_cifar10/`), sample training logs (normal, divergent, OOM, tqdm, noisy, python-logging, partial), sample research findings (with and without reference repos), sample result/config files, dataset loader scripts (CSV, ImageFolder, HuggingFace), and a sample error log (`sample_error_log.json`). Used by the pytest suite.

## Gotchas

- **`detect_divergence.py` CLI takes a JSON string, not a file path**: `python3 scripts/detect_divergence.py '[0.5, 0.4, 100.0]'` — the quotes are required.
- **`implement_utils.py` has three CLI modes**: default (parse proposals), `clone <url> <dest>`, and `analyze <path>`. Each has different argument patterns.
- **Metric routing is split**: Monitor/divergence always uses loss (lower-is-better). Analyze/hp-tune use the user's `primary_metric`. Mixing these up causes silent wrong behavior.
- **Branch experiments are independent**: Results on `ml-opt/branch-a` tell you nothing about what HPs will work on `ml-opt/branch-b`. The tuning agent must group by `code_branch` before analyzing trends.
- **Mid-pipeline review auto-triggers**: After 3+ consecutive all-fail batches in Phase 6, the orchestrator automatically invokes the review skill with `scope: "session"` to suggest course corrections. It can also be invoked manually at end of session.
- **Tabular ML frameworks skip divergence monitoring**: When the detected framework is scikit-learn, XGBoost, or LightGBM, the orchestrator sets `divergence_metric` to `null` and skips the monitor skill. The baseline skill skips GPU profiling and throughput estimation for these frameworks.
