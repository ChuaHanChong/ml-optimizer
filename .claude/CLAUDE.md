# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A Claude Code plugin that orchestrates autonomous ML model optimization. It dispatches specialized agents for research, hyperparameter tuning, experiment execution, and result analysis. The plugin uses LLM-driven HP tuning (Claude reasons about results directly — no Optuna/grid search).

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
skills/                     — 9 skill definitions (SKILL.md files)
agents/                     — 4 subagent definitions
scripts/                    — Python utilities (stdlib only)
tests/                      — pytest test suite
```

### Skill Pipeline (Orchestrator Flow)

The `orchestrate` skill coordinates a 6-phase pipeline. Each skill is invoked via `ml-optimizer:<skill-name>`:

```
Phase 0: Discovery (plan mode, user Q&A)
Phase 1: Understand model (read code, check GPUs)
Phase 2: baseline → Establish baseline metrics
Phase 3: User checkpoint
Phase 4: research → Find techniques via web/papers
Phase 4.5: implement → Apply proposals as git branches
Phase 5: Experiment loop (autonomous):
         hp-tune → propose configs
         experiment → run training (parallel across GPUs)
         monitor → watch for divergence
         analyze → decide continue/pivot/stop
Phase 6: report → Final optimization report
```

### Metric Routing Rule

Always monitor `"loss"` for divergence detection (monitor skill). Use the user's `primary_metric` (accuracy, PSNR, F1, etc.) for analyze and hp-tune skills.

### Branch-Aware Experiments

The implement skill creates `ml-opt/<slug>` branches per research proposal. The experiment loop tests each branch with HP variations. The tuning agent groups results by `code_branch` — identical HPs on different branches are treated as independent experiments.

### Agent Definitions (`agents/`)

Four subagent types with specified tool access:
- **research-agent**: WebSearch, WebFetch, Read, Write, Bash, Glob, Grep
- **tuning-agent**: Read, Write, Bash, Glob, Grep
- **implement-agent**: Bash, Read, Write, Edit, Glob, Grep, WebFetch
- **experiment-agent**: Bash, Read, Write, Glob, Grep

Analytical agents (hp-tune, research, analyze, implement) use "ultrathink" prompting. Procedural agents (experiment, monitor) do not.

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

### State & Output (in target project)

The plugin creates `experiments/` in the user's project:
```
experiments/
  results/baseline.json              — Baseline metrics
  results/exp-*.json                 — Per-experiment results
  results/implementation-manifest.json — Validated proposal branches
  results/proposed-configs/          — HP config proposals
  pipeline-state.json                — Resumable pipeline state
  logs/<exp-id>/train.log            — Raw training logs
  reports/                           — Analysis reports, research findings
  scripts/<exp-id>.sh                — Generated training scripts
  dev_notes.md                       — Running session log
```

### Pipeline Resumption

The orchestrator can be stopped and resumed. On restart it reads `pipeline-state.json` and uses `cleanup_stale()` to handle interrupted experiments (marks them as failed after a timeout). Phase validation via `validate_phase_requirements()` prevents cascading failures. Pipeline state persists Phase 0 user choices (`primary_metric`, `divergence_metric`, `lower_is_better`, `target_value`, `train_command`, `eval_command`) via `save_state(user_choices={...})` so they survive interruptions without re-asking the user.

## Key Design Patterns

- **Non-git fallback**: If the target project isn't a git repo, the implement skill uses file backups instead of branches. This forces sequential (not parallel) experiment execution.
- **Experiment budget**: Default max experiments = `num_gpus * 5`. The orchestrator passes `remaining_budget` to hp-tune, which caps proposals at `min(num_gpus, remaining_budget)` to prevent overshoot. The analyze skill recommends stop when diminishing returns detected.
- **Proposal priority scoring**: `(impact * confidence) / (11 - min(feasibility, 10))` — feasibility clamped to [1,10] to prevent division by zero.
- **Spearman correlation**: `result_analyzer.py` uses rank correlation with average-rank tie-breaking to identify HP-metric relationships (no scipy dependency).
- **Dual implementation strategy**: Research proposals include an `implementation_strategy` field (`from_scratch` or `from_reference`). The implement agent dispatches accordingly — either implementing from paper descriptions (Section 8) or cloning and adapting reference repos (Section 9). Strategy is decided by the research agent based on repo availability and quality.

## Test Fixtures

`tests/fixtures/` contains a minimal PyTorch project (`tiny_resnet_cifar10/`), sample training logs (normal, divergent, OOM, tqdm, noisy), sample research findings (with and without reference repos), and sample result/config files. Used by the 359-test pytest suite.

## Gotchas

- **`detect_divergence.py` CLI takes a JSON string, not a file path**: `python3 scripts/detect_divergence.py '[0.5, 0.4, 100.0]'` — the quotes are required.
- **`implement_utils.py` has three CLI modes**: default (parse proposals), `clone <url> <dest>`, and `analyze <path>`. Each has different argument patterns.
- **Metric routing is split**: Monitor/divergence always uses loss (lower-is-better). Analyze/hp-tune use the user's `primary_metric`. Mixing these up causes silent wrong behavior.
- **Branch experiments are independent**: Results on `ml-opt/branch-a` tell you nothing about what HPs will work on `ml-opt/branch-b`. The tuning agent must group by `code_branch` before analyzing trends.
