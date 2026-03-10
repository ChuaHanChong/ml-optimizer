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
Phase 5.1: implement → Apply proposals as git branches
Phase 6: Experiment loop (autonomous, pipelined):
         hp-tune → propose configs (or use speculative proposals from previous iteration)
         experiment → run training (parallel across GPUs)
         monitor → watch for divergence (concurrent with experiments)
         analyze + speculative hp-tune → decide continue/pivot/stop + prepare next batch in parallel
         [method_proposal] → mid-loop research + implement
         [research_round] → autonomous cadence-based research
         review → Mid-pipeline review (async in autonomous mode, sync in interactive)
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
- **tuning-agent**: Read, Write, Bash, Glob, Grep, WebSearch, WebFetch
- **implement-agent**: Bash, Read, Write, Edit, Glob, Grep, WebSearch, WebFetch
- **experiment-agent**: Bash, Read, Write, Glob, Grep, WebSearch, WebFetch
- **prerequisites-agent**: Bash, Read, Write, Glob, Grep, WebSearch, WebFetch

Analytical agents (hp-tune, research, analyze, implement) use "ultrathink" prompting. Procedural agents (experiment, monitor, prerequisites) do not.

Agents are dispatched as `general-purpose` subagents and can invoke external skills:
- **research-agent**: Uses `context7` for framework API docs, `claude-mem:mem-search` for cross-session learning
- **implement-agent**: Uses `context7` for API docs, `feature-dev:code-explorer` for codebase analysis, `superpowers:systematic-debugging` for error recovery
- **orchestrator**: Uses `claude-mem:mem-search` in Phase 1 for cross-session recall, `feature-dev:code-reviewer` in Phase 5.1 for post-implementation quality review

### Python Scripts (`scripts/`)

All scripts work as both importable modules and CLI tools:

| Script | CLI Usage |
|--------|-----------|
| `gpu_check.py` | `python3 scripts/gpu_check.py` — parse nvidia-smi |
| `parse_logs.py` | `python3 scripts/parse_logs.py <logfile>` — parse kv/JSON/CSV/XGBoost/HuggingFace Trainer logs |
| `detect_divergence.py` | `python3 scripts/detect_divergence.py '<json_values>' [--higher-is-better] [--model-category rl\|generative\|supervised] [--explosion-threshold N] [--plateau-patience N]` — detect NaN/explosion/plateau with configurable thresholds |
| `result_analyzer.py` | `python3 scripts/result_analyzer.py <results_dir> <metric> [baseline_id] [lower_is_better]` |
| `experiment_setup.py` | Generates experiment IDs and directory structure |
| `implement_utils.py` | `python3 scripts/implement_utils.py <findings.md> '<indices_json>'` — parse proposals; also `clone <url> <dest>` and `analyze <path>` subcommands |
| `pipeline_state.py` | `python3 scripts/pipeline_state.py <exp_root> validate|save|load|cleanup` |
| `schema_validator.py` | `python3 scripts/schema_validator.py <filepath> result\|baseline\|manifest\|prerequisites` — validates JSON against expected schemas |
| `plot_results.py` | `python3 scripts/plot_results.py <results_dir> <metric> comparison\|timeline\|sensitivity <hp>\|progress [--higher-is-better]` — ASCII charts + matplotlib progress chart |
| `prerequisites_check.py` | `python3 scripts/prerequisites_check.py scan-imports\|check-packages\|detect-env\|detect-format\|detect-format-project\|validate-data\|bulk-install-cmd\|gpu-install-cmd` — dataset, environment, and GPU-aware install validation |
| `error_tracker.py` | `python3 scripts/error_tracker.py <exp_root> log\|show\|patterns\|summary\|sync\|success\|proposals\|rank\|cleanup\|log-suggestion\|suggestion-history` — error tracking, pattern detection, success metrics, proposal outcomes, suggestion ranking, suggestion history |

### State & Output (in target project)

The plugin creates `experiments/` in the user's project:
```
experiments/
  results/prerequisites.json         — Prerequisites check report
  results/baseline.json              — Baseline metrics
  results/exp-*.json                 — Per-experiment results (schema-validated; includes iteration, method_tier, proposal_source)
  results/implementation-manifest.json — Validated proposal branches
  results/proposed-configs/          — HP config proposals
  prepared-data/                     — Prepared dataset (if preprocessing needed)
  pipeline-state.json                — Resumable pipeline state
  logs/<exp-id>/train.log            — Raw training logs
  reports/                           — Analysis reports, research findings (web + method proposals)
  reports/error-log.json             — Structured error event log
  reports/suggestion-history.json    — Suggestion feedback loop (tracks what was suggested)
  reports/session-review.md          — Self-improvement review (from review skill)
  scripts/<exp-id>.sh                — Generated training scripts
  dev_notes.md                       — Running session log
```

### Method Proposals (LLM Knowledge + Web Search)

The research skill supports `source: "both"` mode where the LLM proposes optimization methods using its training knowledge supplemented by web search. Proposals are scoped by `scope_level`: `"training"` (safest), `"architecture"`, or `"full"`. This is triggered:
- **Pre-loop** (Phase 4, option 5): User chooses to generate method proposals before the experiment loop
- **Mid-loop** (Phase 6, step 6.5): When analyze recommends `pivot_type: "method_proposal"` or `"qualitative_change"`

Both triggers require user confirmation of scope and proposals. Knowledge-based proposals have confidence capped at 7/10.

### Three-Tier Result Tracking

Experiments carry two tracking fields:
- **`method_tier`**: `"baseline"` | `"method_default_hp"` | `"method_tuned_hp"` — which tier of the comparison
- **`proposal_source`**: `"paper"` | `"llm_knowledge"` | `null` — origin of the method

This enables three-tier attribution: baseline metrics → method with default HPs (isolated method effect) → method with tuned HPs (combined effect). The report skill generates a three-tier comparison table when these fields are present.

### Pipeline Resumption

The orchestrator can be stopped and resumed. On restart it reads `pipeline-state.json` and uses `cleanup_stale()` to handle interrupted experiments (marks them as failed after a timeout). Phase validation via `validate_phase_requirements()` prevents cascading failures. Pipeline state persists Phase 0 user choices (`primary_metric`, `divergence_metric`, `divergence_lower_is_better`, `lower_is_better`, `target_value`, `train_command`, `eval_command`, `train_data_path`, `val_data_path`, `prepared_train_path`, `prepared_val_path`, `env_manager`, `env_name`, `model_category`, `user_papers`, `budget_mode`, `difficulty`, `difficulty_multiplier`, `method_proposal_scope`, `method_proposal_iterations`, `hp_batches_per_round`) via `save_state(user_choices={...})` so they survive interruptions without re-asking the user. A separate `user-choices-backup.json` provides redundant recovery if the main state file corrupts.

## Key Design Patterns

- **Non-git fallback**: If the target project isn't a git repo, the implement skill uses file backups instead of branches. Each proposal is validated against a clean baseline backup (restore-before-apply pattern) to prevent cross-proposal code leakage. This forces sequential (not parallel) experiment execution.
- **Experiment budget**: Two modes: `"auto"` (default: agent judges difficulty — easy=×8, moderate=×15, hard=×25 multiplied by `max(num_gpus, 1)`) and `"autonomous"` (unlimited — runs until interrupted or 3 consecutive stop recommendations). Users can also specify a custom budget. The orchestrator passes `remaining_budget` to both hp-tune and analyze. HP-tune caps proposals at `min(max(num_gpus, 1), remaining_budget)` to prevent overshoot. The analyze skill uses `remaining_budget` in its pivot decision tree to gate research pivots (requires `>= 3` for method proposals, `>= 5` for full research). In autonomous mode, stop recommendations are logged but not enforced until 3 consecutive. **Continuous research**: In autonomous mode with `method_proposal_scope` set, the orchestrator auto-triggers a research → implement cycle every `hp_batches_per_round` batches (default 3) without user confirmation. If research yields no new proposals, the cadence doubles (exponential backoff).
- **Proposal priority scoring**: `(impact * confidence) / (11 - min(feasibility, 10))` — feasibility clamped to [1,10] to prevent division by zero.
- **Spearman correlation**: `result_analyzer.py` uses rank correlation with average-rank tie-breaking to identify HP-metric relationships (no scipy dependency).
- **Dual implementation strategy**: Research proposals include an `implementation_strategy` field (`from_scratch` or `from_reference`). The implement agent dispatches accordingly — either implementing from paper descriptions (Section 8) or cloning and adapting reference repos (Section 9). Strategy is decided by the research agent based on repo availability and quality.
- **Research skill modes**: The research skill accepts `source` (`"web"` | `"knowledge"` | `"both"`), `scope_level` (`"training"` | `"architecture"` | `"full"`), and `output_path` parameters. Knowledge mode skips web search and uses LLM training knowledge only.
- **Autonomous mode auto-skip**: When `budget_mode == "autonomous"`, all user checkpoints after Phase 0 are auto-resolved (Phase 1 plan → auto-approve, Phase 2 partial prereqs → proceed, Phase 4 direction → method proposals, Phase 5 proposal selection → all proposals, Phase 5.1 dependencies → auto-install, license warnings → auto-accept, Phase 6 mid-loop scope/proposals → use stored scope + accept all, RL polarity → auto-infer, Phase 7 self-review → auto-run). Only unrecoverable errors (Phase 2 failed) still block. Phase 3 unknown errors exit with partial results in autonomous mode. Decisions are logged to dev_notes and error tracker for post-session review. The implement skill auto-stashes dirty working trees, and the prerequisites skill auto-resolves format/env mismatches.
- **Speculative hp-tune**: In Phase 6, the orchestrator starts a background hp-tune call in parallel with analyze. If analyze says "continue" and proposals pass validation (no pruned branches, within budget, no duplicates), the proposals are used immediately — eliminating 30-60s of GPU idle time per batch. If analyze says stop/pivot, speculative proposals are discarded.
- **Parallel research**: All WebSearch calls in the research skill are issued simultaneously in a single tool-call message. WebFetch follow-ups for different URLs are also parallelized. Domain-specific query sets (NLP, CV, RL, time-series) are issued alongside generic queries.
- **Parallel implementation**: When using git branch strategy with multiple proposals, each proposal is implemented in a separate git worktree via parallel Agent dispatches. File-backup strategy remains sequential.
- **Async mid-pipeline review**: In autonomous mode, mid-pipeline review runs in the background while the next experiment batch starts. Suggestions are applied one batch late — acceptable trade-off vs blocking the pipeline.
- **Configurable divergence thresholds**: `detect_divergence.py` supports per-model-category threshold overrides via `MODEL_CATEGORY_DEFAULTS` dict and `--model-category` CLI flag. RL models use `explosion_threshold=20.0` (prevents false positives on reward spikes), generative models use `plateau_patience=40` (accommodates slow convergence). Individual thresholds can also be overridden via `--explosion-threshold` and `--plateau-patience` CLI flags.
- **Experiment timeout**: Each experiment has a hard timeout of `baseline_training_time * 3` (fallback: 6 hours). Timed-out experiments are killed and marked `status: "timeout"`.
- **Research failure recovery**: If web search fails, the orchestrator retries with `source: "knowledge"` (LLM-only). If that also fails, it continues with HP-only optimization. Each fallback is logged.
- **OOM feedback loop**: When experiments OOM, the batch size is recorded in the error tracker. On the next hp-tune invocation, `max_batch_size` is passed to prevent re-proposing configs that will OOM.
- **All-diverge recovery**: If all experiments in a batch diverge, a recovery batch with halved learning rates is attempted before stopping.
- **HP-only research routing**: Research proposals with `type: "hp_only"` skip the implement skill and are routed directly to hp-tune as search space modifications.
- **Tabular ML HP strategy**: For tree-based models (sklearn/XGBoost/LightGBM), iteration 1 explores `max_depth`/`n_estimators` first instead of learning rate.
- **Concurrent-safe error logging**: `error_tracker.py` uses `fcntl.flock()` file locking around the read-modify-write in `log_event()` to prevent concurrent agents from losing events.
- **Result file filtering**: `result_analyzer.py` only loads `exp-*.json` and `baseline.json` files, preventing non-experiment files from inflating counts.
- **HuggingFace Trainer log format**: `parse_logs.py` detects and parses HuggingFace Trainer's single-quote Python dict format (`{'loss': 0.5, 'epoch': 1.0}`).
- **Baseline eval auto-fallback**: In autonomous mode, if no eval command is found, baseline uses training output metrics instead of blocking on user input.
- **Pre-flight file validation**: The implement skill validates all `files_to_modify` exist before creating branches or starting implementation. Missing-file proposals are marked `preflight_failed`.
- **Early batch abort**: If >= 2 experiments diverge within 60 seconds of start, remaining experiments in the batch are cancelled to save compute.
- **Tabular ML adaptive timeout**: For non-iterative frameworks, experiment timeout is computed from `fit_duration * (max_iters / profiling_iters) * 2` instead of a generic 4-hour fallback.

## Test Fixtures

`tests/fixtures/` contains a minimal PyTorch project (`tiny_resnet_cifar10/`), sample training logs (normal, divergent, OOM, tqdm, noisy, python-logging, partial, XGBoost session, LightGBM session), sample research findings (with and without reference repos, including knowledge-mode proposals), sample result/config files, dataset loader scripts (CSV, ImageFolder, HuggingFace), and a sample error log (`sample_error_log.json`). Used by the pytest suite.

## Gotchas

- **`detect_divergence.py` CLI takes a JSON string, not a file path**: `python3 scripts/detect_divergence.py '[0.5, 0.4, 100.0]'` — the quotes are required. Pass `--higher-is-better` for reward-like metrics. Pass `--model-category rl` for RL-appropriate thresholds.
- **`implement_utils.py` has three CLI modes**: default (parse proposals), `clone <url> <dest>`, and `analyze <path>`. Each has different argument patterns.
- **Metric routing is split**: Monitor/divergence always uses loss (lower-is-better). Analyze/hp-tune use the user's `primary_metric`. Mixing these up causes silent wrong behavior.
- **Branch experiments are independent**: Results on `ml-opt/branch-a` tell you nothing about what HPs will work on `ml-opt/branch-b`. The tuning agent must group by `code_branch` before analyzing trends.
- **Mid-pipeline review auto-triggers**: In Phase 6, the orchestrator checks error patterns after each batch. If `wasted_budget` pattern reaches ≥ 3 occurrences OR the last 2 consecutive batches had zero successful experiments, it invokes the review skill with `scope: "session"` to suggest course corrections. It can also be invoked manually at end of session.
- **Tabular ML frameworks skip divergence monitoring**: When the detected framework is scikit-learn, XGBoost, or LightGBM, the orchestrator sets `divergence_metric` to `null` and skips the monitor skill. The baseline skill skips GPU profiling and throughput estimation for these frameworks.
- **Research findings files can be multiple**: `research-findings.md` (Phase 5 web search), `research-findings-method-proposals.md` (Phase 6 pre-loop), `research-findings-method-proposals-iter<N>.md` (Phase 6 mid-loop triggers). The research skill's deduplication checks all of these to avoid re-proposing tried techniques.
