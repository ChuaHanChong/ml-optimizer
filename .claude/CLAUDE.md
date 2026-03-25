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

No build step. No linter configured. Python 3.10+ required. The `scripts/` directory uses only the Python standard library (except `scripts/plot_results.py` which requires matplotlib for chart generation).

## MCP Server Dependencies (Recommended)

| MCP Server | Purpose | Used by | Required? |
|------------|---------|---------|-----------|
| **alphaxiv** (`api.alphaxiv.org/mcp/v1`) | Academic paper search (2.5M+ arXiv papers), paper content extraction, PDF Q&A, GitHub repo exploration | research-agent (6 tools), implement-agent (2 tools) | No — falls back to WebSearch/WebFetch |
| **claude-mem** | Cross-session memory — recalls past optimization sessions, avoids re-proposing failed techniques | research-agent, orchestrator (Phase 1) | No — works without but loses cross-session learning |
| **context7** | Framework API documentation lookup (PyTorch, TensorFlow, etc.) | research-agent, implement-agent | No — falls back to WebSearch |

The plugin works without any MCP servers but benefits significantly from alphaxiv (better paper discovery and analysis) and claude-mem (learning across optimization sessions).

## Architecture

### Plugin Structure

```
.claude-plugin/plugin.json  — Plugin metadata (name, version)
commands/optimize.md        — /optimize slash command (entry point)
skills/                     — Skill definitions (SKILL.md files)
skills/evolve/ShinkaEvolve/ — Git submodule (SakanaAI/ShinkaEvolve) for evolutionary code optimization
agents/                     — 9 subagent definitions
scripts/                    — Python utilities (stdlib only)
tests/                      — pytest test suite
```

### Skill Pipeline (Orchestrator Flow)

The `orchestrate` skill coordinates a 10-phase pipeline. Each phase dispatches a named agent via `Agent(subagent_type="ml-optimizer:<name>-agent")`. Persistent agents (research, implement, tuning, analysis, monitor) are resumed via `SendMessage(to: agentId)` for subsequent dispatches; ephemeral agents (prerequisites, baseline, experiment, report) get fresh spawns:

```
Phase 0: Discovery (plan mode, user Q&A — includes data paths and env manager)
Phase 1: Understand model (read code, check GPUs)
Phase 2: prerequisites → Validate dataset format, prepare data, install dependencies
Phase 3: baseline → Establish baseline metrics
Phase 4: User checkpoint
Phase 5: research → Find techniques via web/papers
Phase 6: implement → Apply proposals as git branches
Phase 7: Experiment loop (autonomous, pipelined):
         hp-tune → propose configs
         experiment → run training (parallel across GPUs)
         monitor → watch for divergence (concurrent with experiments)
         analyze → decide continue/pivot/stop
         [method_proposal] → mid-loop research + implement
         [code_refinement] → tuning (evolve HPs) → evolve → tuning (training HPs) → experiment (scope_level=full only)
         [research_round] → cadence-based research (when method proposals enabled)
Phase 8: Method stacking (if 5+ methods improved):
         Sequential accumulation — merge best methods one by one
         LLM conflict resolution, skip-on-failure
         Per step: analyze → tuning (evolve HPs) → evolve → tuning (training HPs) → experiment
         Analysis agent loops until improvement or recommends stop
Phase 9: report → Final optimization report
         review → Self-improvement analysis (optional, end-of-session)
```

### Metric Routing Rule

Always monitor `"loss"` for divergence detection (monitor skill). Use the user's `primary_metric` (accuracy, PSNR, F1, etc.) for analyze and hp-tune skills.

### Branch-Aware Experiments

The implement skill creates `ml-opt/<slug>` branches per research proposal. The experiment loop tests each branch with HP variations. The tuning agent groups results by `code_branch` — identical HPs on different branches are treated as independent experiments.

### Agent Definitions (`agents/`)

Nine subagent types, each with a preloaded skill and specified tool access. Skills are instruction documents only — they have `disable-model-invocation: true` and `user-invocable: false`, and are not directly invocable. The one exception is `orchestrate`, which is the user-facing entry point (no `disable-model-invocation`, no `user-invocable: false`) so the `/optimize` command can load it via the Skill tool. All 9 agents have `memory: local` for persistent role-specific memory at `.claude/agent-memory-local/<agent-name>/` in the target project.

**Persistent agents** — dispatched once via `Agent(subagent_type=...)`, resumed via `SendMessage(to: agentId)` for subsequent tasks. The orchestrator tracks agent IDs in `pipeline-state.json` under `agent_registry` and falls back to fresh dispatch if resumption fails.

**Ephemeral agents** — fresh `Agent()` spawn each time (single-use or parallel tasks).

**Procedural agents** (`model: sonnet` — lower cost/latency, no ultrathink):
- **baseline-agent** *(ephemeral)*: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:baseline]`
- **monitor-agent** *(persistent)*: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:monitor]`
- **experiment-agent** *(ephemeral)*: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:experiment]`
- **prerequisites-agent** *(ephemeral)*: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:prerequisites]`

**Analytical agents** (`model: opus`, ultrathink prompting):
- **research-agent** *(persistent)*: WebSearch, WebFetch, Read, Write, Bash, Glob, Grep, Skill, alphaxiv MCP tools (6) — skills: `[ml-optimizer:research, claude-mem:mem-search]`
- **tuning-agent** *(persistent)*: Read, Write, Bash, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:hp-tune]`
- **implement-agent** *(persistent)*: Bash, Read, Write, Edit, Glob, Grep, Skill, WebSearch, WebFetch, alphaxiv MCP tools (2: repo reader, PDF Q&A) — skills: `[ml-optimizer:implement, ml-optimizer:evolve, ml-optimizer:shinka-setup, ml-optimizer:shinka-convert, ml-optimizer:shinka-run, ml-optimizer:shinka-inspect, superpowers:systematic-debugging, feature-dev:code-explorer, feature-dev:code-reviewer]`
- **analysis-agent** *(persistent)*: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:analyze]` (includes session review mode)
- **report-agent** *(ephemeral)*: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:report]`

For parallel execution, use `run_in_background: true`. External skills are also available:
- **research-agent**: Uses `context7` for framework API docs, `claude-mem:mem-search` for cross-session learning, `alphaxiv` MCP for academic paper search/analysis (6 tools: embedding search, full-text search, agentic retrieval, paper content, PDF Q&A, GitHub repo reader)
- **implement-agent**: Uses `context7` for API docs, `feature-dev:code-explorer` for codebase analysis, `feature-dev:code-reviewer` for post-implementation quality review (advisory, Step 4e.5), `superpowers:systematic-debugging` for error recovery, `alphaxiv` MCP for reference repo exploration (`read_files_from_github_repository`) and paper clarification (`answer_pdf_queries`)
- **orchestrator**: Uses `claude-mem:mem-search` in Phase 1 for cross-session recall, `superpowers:brainstorming` in Phase 0/4 for complex multi-objective optimization scenarios

### Python Scripts (`scripts/`)

All scripts work as both importable modules and CLI tools:

| Script | CLI Usage |
|--------|-----------|
| `scripts/gpu_check.py` | `python3 scripts/gpu_check.py` — parse nvidia-smi |
| `scripts/parse_logs.py` | `python3 scripts/parse_logs.py <logfile>` — parse kv/JSON/CSV/XGBoost/HuggingFace Trainer logs |
| `scripts/detect_divergence.py` | `python3 scripts/detect_divergence.py '<json_values>' [--higher-is-better] [--model-category rl\|generative\|supervised] [--explosion-threshold N] [--plateau-patience N]` — detect NaN/explosion/plateau with configurable thresholds. Also: `python3 scripts/detect_divergence.py --check-overfitting '<train_json>' '<val_json>' [--higher-is-better] [--patience N] [--min-gap F] [--model-category rl\|generative\|supervised]` — detect overfitting |
| `scripts/result_analyzer.py` | `python3 scripts/result_analyzer.py <results_dir> <metric> [baseline_id] [lower_is_better]` — full analysis. Also: `python3 scripts/result_analyzer.py <results_dir> compare <exp_id_1> <exp_id_2> [metric] [lower_is_better]` — pairwise comparison |
| `scripts/experiment_setup.py` | Generates experiment IDs and directory structure |
| `scripts/implement_utils.py` | `python3 scripts/implement_utils.py <findings.md> '<indices_json>'` — parse proposals; also `clone <url> <dest>`, `analyze <path>`, and `diff <project_root> <branch>` subcommands |
| `scripts/pipeline_state.py` | `python3 scripts/pipeline_state.py <exp_root> validate|save|load|cleanup|verify-baseline` |
| `scripts/schema_validator.py` | `python3 scripts/schema_validator.py <filepath> result\|baseline\|manifest\|prerequisites [--strict]` — validates JSON against expected schemas. `--strict` enforces completeness (completed experiments must have non-empty metrics, iteration, method_tier, duration_seconds) |
| `scripts/plot_results.py` | `python3 scripts/plot_results.py <results_dir> <metric> comparison\|timeline\|sensitivity <hp>\|progress [--higher-is-better]` — ASCII charts + matplotlib progress chart |
| `scripts/prerequisites_check.py` | `python3 scripts/prerequisites_check.py scan-imports\|check-packages\|detect-env\|detect-format\|detect-format-project\|validate-data\|bulk-install-cmd\|gpu-install-cmd` — dataset, environment, and GPU-aware install validation |
| `scripts/dashboard.py` | `python3 scripts/dashboard.py <exp_root> [--live] [--table] [--serve --port 8080]` — generate self-contained HTML dashboard with progress timeline, results table, HP sensitivity, research agenda, error summary, method explanations. `--live` enables 30s auto-refresh. `--table` generates `results-table.md` (Markdown results summary). |
| `scripts/excalidraw_gen.py` | `python3 scripts/excalidraw_gen.py <exp_root> pipeline\|comparison\|hp-landscape\|architecture <args>` — generate Excalidraw JSON diagrams (pipeline overview, experiment comparison, HP landscape, architecture changes) |
| `scripts/error_tracker.py` | `python3 scripts/error_tracker.py <exp_root> log\|show\|patterns\|summary\|success\|proposals\|rank\|log-suggestion\|suggestion-history\|dead-end <add\|list\|check>\|agenda <init\|update\|list\|add>` — error tracking, pattern detection, success metrics, proposal outcomes, suggestion ranking, suggestion history, dead-end catalog, research agenda |
| `scripts/goal_memory.py` | `python3 scripts/goal_memory.py <exp_root> init-goals\|read-goals\|log-behavior <category> <json>\|query-behaviors [category]\|validate-output <agent> <json>\|summary\|sync-from-errors` — goal anchoring, behavioral memory, agent output validation, compact briefings |
| `scripts/setup_evolve.sh` | `bash scripts/setup_evolve.sh` — initialize ShinkaEvolve submodule and create skill symlinks for auto-discovery |

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
  optimization-goals.json            — Goal anchor (written at Phase 0, read by all agents)
  learned-behaviors.json             — Accumulated behavioral memory (HP constraints, method outcomes, divergence patterns)
  pipeline-state.json                — Resumable pipeline state
  logs/<exp-id>/train.log            — Raw training logs
  reports/                           — Analysis reports, research findings (web + method proposals)
  reports/error-log.json             — Structured error event log
  reports/suggestion-history.json    — Suggestion feedback loop (tracks what was suggested)
  reports/dead-ends.json             — Dead-end catalog (techniques conclusively shown to be unpromising)
  reports/dead-ends.md               — Human-readable dead-end companion
  reports/research-agenda.json       — Living research agenda (reprioritized after each batch)
  reports/research-agenda.md         — Human-readable research agenda companion
  reports/dashboard.html              — Self-contained HTML progress dashboard
  reports/session-review.md          — Self-improvement review (from review skill)
  scripts/<exp-id>/                  — Per-experiment command scripts (train, eval, etc.)
  artifacts/                         — Model checkpoints, intermediate files, images, plots
  artifacts/<exp-id>/                — Per-experiment artifacts (checkpoints, visualizations)
  artifacts/*.excalidraw             — Excalidraw diagrams (pipeline, comparison, HP landscape, architecture)
  dev_notes.md                       — Running session log
```

### Method Proposals (LLM Knowledge + Web Search)

The research skill supports `source: "both"` mode where the LLM proposes optimization methods using its training knowledge supplemented by web search. Proposals are scoped by `scope_level`: `"training"` (safest), `"architecture"`, or `"full"`. This is triggered:
- **Pre-loop** (Phase 4, option 5): User chooses to generate method proposals before the experiment loop
- **Mid-loop** (Phase 7, step 7): When analyze recommends `pivot_type: "method_proposal"` or `"qualitative_change"`

Both triggers require user confirmation of scope and proposals. Knowledge-based proposals have confidence capped at 7/10.

### Three-Tier Result Tracking

Experiments carry two tracking fields:
- **`method_tier`**: `"baseline"` | `"method_default_hp"` | `"method_tuned_hp"` — which tier of the comparison
- **`proposal_source`**: `"paper"` | `"llm_knowledge"` | `null` — origin of the method

This enables three-tier attribution: baseline metrics → method with default HPs (isolated method effect) → method with tuned HPs (combined effect). The report skill generates a three-tier comparison table when these fields are present.

Additionally, stacking experiments use:
- **`stacked_default_hp`**: Combined methods tested with best individual HPs
- **`stacked_tuned_hp`**: Combined methods after HP-tuning

Stacking experiments also carry `code_branches` (array of combined branches), `stacking_order`, and `stack_base_exp`.

### Pipeline Resumption

The orchestrator can be stopped and resumed. On restart it reads `pipeline-state.json` and uses `cleanup_stale()` to handle interrupted experiments (marks them as failed after a timeout). Phase validation via `validate_phase_requirements()` prevents cascading failures. Pipeline state persists Phase 0 user choices (`primary_metric`, `divergence_metric`, `divergence_lower_is_better`, `lower_is_better`, `target_value`, `train_command`, `eval_command`, `train_data_path`, `val_data_path`, `prepared_train_path`, `prepared_val_path`, `env_manager`, `env_name`, `model_category`, `user_papers`, `method_proposal_scope`, `method_proposal_iterations`, `hp_batches_per_round`, `fixed_time_budget`, `fixed_epoch_budget`) via `save_state(user_choices={...})` so they survive interruptions without re-asking the user. The experiment loop also persists `consecutive_stop_count` (for the 3-consecutive-stop exit rule), `stuck_protocol_triggered` (prevents infinite recovery loops), `baseline_checksum` (SHA-256 of baseline metrics for integrity verification), and `agent_registry` (persistent agent IDs for SendMessage resumption) at the root level of pipeline state. On new session start, `agent_registry` is cleared since subagent transcripts are session-scoped — all agents start fresh. A separate `user-choices-backup.json` provides redundant recovery if the main state file corrupts.

## Key Design Patterns

- **Resumable subagents**: 5 persistent agents (research, implement, tuning, analysis, monitor) are dispatched once via `Agent()` and resumed via `SendMessage(to: agentId)` for subsequent tasks. This preserves accumulated context (search results, HP trends, codebase knowledge) across the pipeline. 4 ephemeral agents (prerequisites, baseline, experiment, report) get fresh spawns. The orchestrator tracks agent IDs in `pipeline-state.json` under `agent_registry` and falls back to fresh dispatch if resumption fails. Agent IDs are session-scoped — cleared on new session start since subagent transcripts don't survive across sessions.
- **Inter-agent communication (orchestrator relay)**: When resuming a persistent agent, the orchestrator includes a `CONTEXT FROM OTHER AGENTS:` section with relevant findings from other agents. This enables indirect communication: analyze findings reach hp-tune, monitor OOM info reaches hp-tune, research proposals reach implement, etc. The orchestrator acts as a message bus. Key relay routes: analyze→tuning (correlations, branch scores), analyze→research (pivot reasons, dead-ends), monitor→tuning (OOM constraints), research→implement (proposals), experiments→analyze (batch results).
- **Non-git fallback**: If the target project isn't a git repo, the implement skill uses file backups instead of branches. Each proposal is validated against a clean baseline backup (restore-before-apply pattern) to prevent cross-proposal code leakage. This forces sequential (not parallel) experiment execution.
- **Loop exit conditions**: The experiment loop runs until: (1) target metric achieved, (2) analysis agent recommends "stop" 3 consecutive times (triggers stuck protocol), or (3) user manually stops. HP-tune proposes `max(num_gpus, 1)` configs per batch. **Continuous research**: When `method_proposal_scope` is set, the orchestrator auto-triggers a research → implement cycle every `hp_batches_per_round` batches (default 3). If research yields no new proposals, the cadence doubles (exponential backoff).
- **Proposal priority scoring**: `(impact * confidence) / (11 - min(feasibility, 10))` — feasibility clamped to [1,10] to prevent division by zero.
- **Spearman correlation**: `scripts/result_analyzer.py` uses rank correlation with average-rank tie-breaking to identify HP-metric relationships (no scipy dependency).
- **Dual implementation strategy**: Research proposals include an `implementation_strategy` field (`from_scratch` or `from_reference`). The implement agent dispatches accordingly — either implementing from paper descriptions (Section 8) or cloning and adapting reference repos (Section 9). Strategy is decided by the research agent based on repo availability and quality.
- **Research skill modes**: The research skill accepts `source` (`"web"` | `"knowledge"` | `"both"`), `scope_level` (`"training"` | `"architecture"` | `"full"`), and `output_path` parameters. Knowledge mode skips web search and uses LLM training knowledge only.
- **Scope-gated pivots**: The analysis agent's pivot decision tree respects `scope_level`: `"training"` (HP-only) disables research and code_refinement pivots — only HP adjustments are available. `"architecture"` enables research but not ShinkaEvolve. `"full"` enables everything including code_refinement via ShinkaEvolve. This ensures the pipeline matches what the user asked for.
- **Auto-resolution behaviors**: The pipeline auto-resolves many situations: Phase 2 partial prereqs → proceed with warnings, RL polarity → auto-infer from metric name, dirty working trees → auto-stash, environment mismatches → use detected manager, missing conda envs → auto-create, no eval command → fall back to training output metrics. Only unrecoverable errors (Phase 2 failed, baseline failed) block the pipeline. Decisions are logged to dev_notes and error tracker for post-session review.
- **Parallel research**: All WebSearch calls in the research skill are issued simultaneously in a single tool-call message, alongside 3 alphaxiv search calls (embedding similarity, full-text keyword, agentic retrieval). WebFetch follow-ups for different URLs are also parallelized. Domain-specific query sets (NLP, CV, RL, time-series) are issued alongside generic queries. If alphaxiv MCP is unavailable, WebSearch provides full coverage as fallback.
- **Parallel implementation**: When using git branch strategy with multiple proposals, each proposal is implemented in a separate git worktree via parallel Agent dispatches. File-backup strategy remains sequential.
- **Configurable divergence thresholds**: `scripts/detect_divergence.py` supports per-model-category threshold overrides via `MODEL_CATEGORY_DEFAULTS` dict and `--model-category` CLI flag. RL models use `explosion_threshold=20.0, plateau_patience=50` (prevents false positives on reward spikes). Generative models use `explosion_threshold=10.0, plateau_patience=40` (accommodates slow convergence). Individual thresholds can also be overridden via `--explosion-threshold` and `--plateau-patience` CLI flags.
- **Experiment timeout**: Each experiment has a hard timeout of `baseline_training_time * 3` (fallback: 6 hours). Timed-out experiments are killed and marked `status: "timeout"`.
- **Research failure recovery**: If web search fails (both WebSearch and alphaxiv), the orchestrator retries with `source: "knowledge"` (LLM-only). If that also fails, it continues with HP-only optimization. Each fallback is logged. Within a search, alphaxiv failure alone does not trigger the knowledge fallback — WebSearch results are sufficient to proceed.
- **alphaxiv MCP integration**: The research agent uses all 6 alphaxiv MCP tools for academic paper discovery (3 search tools run in parallel), paper content extraction (`get_paper_content`, `answer_pdf_queries`), and reference repo exploration (`read_files_from_github_repository`). The implement agent uses 2 alphaxiv tools (`read_files_from_github_repository` for pre-clone repo assessment, `answer_pdf_queries` for clarifying ambiguous implementation steps from source papers). All alphaxiv searches run in parallel with WebSearch. alphaxiv is optional — if the MCP server is unavailable, all workflows fall back to WebSearch/WebFetch transparently.
- **OOM feedback loop**: When experiments OOM, the batch size is recorded in the error tracker. On the next hp-tune invocation, `max_batch_size` is passed to prevent re-proposing configs that will OOM.
- **All-diverge recovery**: If all experiments in a batch diverge, a recovery batch with halved learning rates is attempted before stopping.
- **HP-only research routing**: Research proposals with `type: "hp_only"` skip the implement skill and are routed directly to hp-tune as search space modifications.
- **Tabular ML HP strategy**: For tree-based models (sklearn/XGBoost/LightGBM), iteration 1 explores `max_depth`/`n_estimators` first instead of learning rate.
- **Training budget options**: Phase 0 offers two budget modes: `fixed_time_budget` (seconds) for wall-clock-normalized comparison, or `fixed_epoch_budget` (integer) for deterministic reproducibility. When set, both baseline AND experiments use the same budget. The baseline skill (Step 2.2) wraps training with `timeout` when `fixed_time_budget` is set, ensuring fair baseline-vs-experiment comparisons. Framework-native time limits (Lightning `--max_time`, HuggingFace `timeout` in TrainingArguments) are preferred when available. Results include `time_budget_seconds` for downstream analysis. HP-tune adjusts proposals for the budget (shorter convergence schedules, appropriate LR scaling). Makes experiment metrics directly comparable without duration normalization.
- **Small dataset awareness**: The research skill checks dataset size. For datasets under 5K samples, it shifts search toward low-data techniques (transfer learning, fine-tuning, few-shot learning, adapters, prompt tuning, synthetic data, semi-supervised methods) instead of heavy augmentation and regularization which underperform on small data.
- **Evolutionary code refinement**: When HP tuning shows diminishing returns, the analyze skill can recommend `pivot_type: "code_refinement"`. The orchestrator dispatches the implement-agent with the evolve skill (`Skill("ml-optimizer:evolve")`), which orchestrates the full ShinkaEvolve pipeline internally: `shinka-convert` (create task from best branch) → `shinka-run` (run evolution with file-based handoff, `SHINKA_PROVIDER=claude_code`) → `shinka-inspect` (extract best mutation) → commit as `ml-opt/evolved-<slug>`. Evolve HPs (`num_generations`, `population_size`) are tuning-agent-driven: the orchestrator dispatches the tuning agent to propose evolve HPs before dispatching the implement agent, based on prior evolution outcomes stored in `learned-behaviors.json` under category `evolve_hp`. Defaults: 10 generations, population 2. If ShinkaEvolve is unavailable, the evolve skill reports `shinkaevolve_unavailable` and the orchestrator falls back to the research → implement path. Setup: run `bash scripts/setup_evolve.sh` to init the submodule and create symlinks (`skills/shinka-*` → `skills/evolve/ShinkaEvolve/skills/shinka-*`). The symlinks are required for Claude Code's skill auto-discovery.
- **Auto-repair loop**: When training or evaluation commands fail during baseline establishment or experiment execution, the agent captures stderr, diagnoses the error, applies a fix (install package, adjust path, reduce batch size), and retries up to 3 times. OOM errors are not retried (deterministic). SyntaxErrors are not retried (code bugs). Identical errors on consecutive attempts skip further retries (loop detection). Each retry is logged to the error tracker. This is intra-agent retry, separate from the orchestrator's Phase 3 retry logic.
- **Goal anchoring & behavioral memory**: `scripts/goal_memory.py` maintains two project-scoped files: `optimization-goals.json` (goal anchor, written once at Phase 0) and `learned-behaviors.json` (accumulated behavioral memory). The orchestrator calls `validate-output` after hp-tune, research, and analyze dispatches to catch drift (frozen param changes, scope breaches, dead-end re-proposals, metric mismatches). Each agent also reads a compact `summary` combining goals + constraints + dead-ends before acting. All 9 agents have `memory: local` for persistent role-specific memory at `.claude/agent-memory-local/<agent-name>/` in the target project.
- **Immutable baseline**: After baseline is established (Phase 3), a SHA-256 checksum of the baseline metrics dict is stored in `pipeline-state.json`. Before each experiment batch (Phase 7) and on pipeline resumption, the checksum is verified against `baseline.json`. If the metrics have changed (accidental modification, file corruption, or tampering), the pipeline halts with a critical error. Prevents invalid experiment comparisons during long optimization runs.
- **Stuck protocol**: When 3 consecutive stop recommendations occur, the orchestrator triggers a structured recovery instead of exiting. It reads error patterns, success metrics, dead ends, and the research agenda, then dispatches the research agent with full failure context. If new proposals are found, the loop resumes with `consecutive_stop_count` reset. Triggers once per session (`stuck_protocol_triggered` in pipeline state) to prevent infinite loops. If the protocol finds no new approaches, the loop exits normally.
- **Research agenda as living document**: `scripts/error_tracker.py` maintains `research-agenda.json` — a prioritized list of optimization ideas that evolves over the session. The research skill initializes it from proposals (Phase 5). The analyze skill updates it after each batch: marking ideas as tried/improved/dead-end, adjusting priorities based on evidence, and adding new ideas suggested by experimental results. The hp-tune skill reads it to understand which untried techniques are high-priority. The report skill includes a summary in the final report.
- **Overfitting detection**: `scripts/detect_divergence.py` provides `check_overfitting(train_values, val_values)` that compares train vs val metric trajectories. Detects when val metric worsens for N consecutive steps while train improves. Severity classified as mild/moderate/severe. Default patience=5 steps; model-category overrides: RL patience=10, generative patience=8. CLI: `--check-overfitting '<train_json>' '<val_json>'`.
- **HP interaction detection**: `scripts/result_analyzer.py` provides `detect_hp_interactions()` that computes 2-way interaction terms using product of centered ranks. Reports interactions stronger than either individual HP correlation. Integrated into `analyze()` output.
- **Adaptive branch budget allocation**: `scripts/result_analyzer.py` provides `compute_branch_scores()` that scores branches by `improvement_pct * confidence_factor`. HP-tune allocates experiment slots proportionally to scores in iteration 2+.
- **Checkpoint warm-starting**: Experiments can optionally warm-start from a previous checkpoint. `scripts/experiment_setup.py` supports `checkpoint_path` parameter. HP-tune proposes warm-started configs (lower LR, fewer epochs, same-branch only) when enabled.
- **Dead-end catalog**: `scripts/error_tracker.py` maintains `dead-ends.json` tracking techniques conclusively shown to be unpromising. The research and hp-tune skills consult this catalog before proposing new techniques, preventing wasted budget on proven dead ends. Fuzzy matching (case-insensitive, substring containment, hyphen/underscore normalization) prevents near-duplicate re-proposals. The analyze skill logs dead ends when branches are pruned or all experiments fail.
- **Concurrent-safe error logging**: `scripts/error_tracker.py` uses `fcntl.flock()` file locking around the read-modify-write in `log_event()` to prevent concurrent agents from losing events.
- **Result file filtering**: `scripts/result_analyzer.py` only loads `exp-*.json` and `baseline.json` files, preventing non-experiment files from inflating counts.
- **alphaxiv query format differences**: `embedding_similarity_search` expects 2-3 descriptive sentences, `full_text_papers_search` expects 3-4 short keywords (no quotes), and `agentic_paper_retrieval` expects a natural language question. Using the wrong format degrades result quality. All 3 must be called in parallel per alphaxiv documentation.
- **HuggingFace Trainer log format**: `scripts/parse_logs.py` detects and parses HuggingFace Trainer's single-quote Python dict format (`{'loss': 0.5, 'epoch': 1.0}`).
- **Baseline eval auto-fallback**: If no eval command is found, baseline uses training output metrics instead of blocking on user input.
- **Pre-flight file validation**: The implement skill validates all `files_to_modify` exist before creating branches or starting implementation. Missing-file proposals are marked `preflight_failed`.
- **Tabular ML adaptive timeout**: For non-iterative frameworks, experiment timeout is computed from `fit_duration * (max_iters / profiling_iters) * 2` instead of a generic 4-hour fallback.
- **Method stacking (Phase 8)**: After independent method testing identifies ≥5 methods that improve over baseline, the orchestrator sequentially merges them in descending order of improvement. Each stack step creates `ml-opt/stack-<N>` by merging the next method into the current best stack. Clean merges proceed directly; conflicts are resolved by the implement-agent. If a combination degrades performance, that method is skipped. After each successful stack step, the analysis agent assesses the stacked result — if the stacked gain is less than the best individual method's gain (indicating method interference), it recommends `code_refinement` and the evolve skill (`Skill("ml-optimizer:evolve")`) optimizes code-level interactions via ShinkaEvolve (5 generations, population 2). This mirrors Phase 7's pattern where the analysis agent is always the decision-maker for evolution. If the evolved code improves over the pre-evolution stack, it becomes the new stack base; otherwise it's discarded. Optional HP-tuning (1-2 iterations, narrowed scope) follows on the (potentially evolved) code when the combo shows >1% improvement. Stacking state persists in `pipeline-state.json` (including `evolved_methods` tracking) for resumption. Requires git branch strategy — skipped for `file_backup` projects.
- **Structured ideation for knowledge mode**: The research skill's knowledge-based proposal generation (Phase 7 method proposals) uses a structured diverge-converge-refine process with 6 ideation lenses (Problem-First, Analogical Reasoning, What Changed Recently, Constraint Manipulation, Negation/Inversion, Composition/Decomposition). Generates 10-15 candidates, filters via scope/dead-end/two-sentence-test, refines survivors with implementation details.
- **Statistical confidence assessment**: The analyze skill (Step 2.2) computes effect sizes (Cohen's d) for HP impact when ≥5 experiments exist, and labels findings by confidence level (high/medium/low). Method attribution distinguishes whether improvements came from the code change, HP tuning, or their compound effect.
- **Reproducibility metadata**: The experiment skill (Step 1.3) captures random seeds, pip freeze snapshots, git SHA, and framework versions. Stored under `"reproducibility"` key in result JSONs. Enables exact reproduction of best experiments.
- **Report threats to validity**: The report template includes a "Threats to Validity" section covering single-seed risk, limited search space, dataset specificity, budget constraints, and noise margins.
- **Citation verification**: The report skill (Step 5.3) cross-references technique claims against experiment data and spot-checks source URL accessibility before writing the final report.

## Test Fixtures

`tests/fixtures/` contains a minimal PyTorch project (`tiny_resnet_cifar10/`), sample training logs (normal, divergent, OOM, tqdm, noisy, python-logging, partial, XGBoost session, LightGBM session), sample research findings (with and without reference repos, including knowledge-mode proposals), sample result/config files, dataset loader scripts (CSV, ImageFolder, HuggingFace), and a sample error log (`sample_error_log.json`). Used by the pytest suite.

## Gotchas

- **`scripts/goal_memory.py validate-output` returns exit code 2 for violations**: Exit code 0 = valid, 1 = script error, 2 = validation violations found. The orchestrator should check the exit code and parse the JSON output for the `violations` array.
- **`scripts/goal_memory.py` imports from `scripts/error_tracker.py`**: For dead-end checks, it lazily imports `is_dead_end` and `get_dead_ends`. Both scripts must be in the same `scripts/` directory.
- **`scripts/detect_divergence.py` CLI takes a JSON string, not a file path**: `python3 scripts/detect_divergence.py '[0.5, 0.4, 100.0]'` — the quotes are required. Pass `--higher-is-better` for reward-like metrics. Pass `--model-category rl` for RL-appropriate thresholds.
- **`scripts/implement_utils.py` has three CLI modes**: default (parse proposals), `clone <url> <dest>`, and `analyze <path>`. Each has different argument patterns.
- **Metric routing is split**: Monitor/divergence always uses loss (lower-is-better). Analyze/hp-tune use the user's `primary_metric`. Mixing these up causes silent wrong behavior.
- **Branch experiments are independent**: Results on `ml-opt/branch-a` tell you nothing about what HPs will work on `ml-opt/branch-b`. The tuning agent must group by `code_branch` before analyzing trends.
- **`agent_registry` is session-scoped**: Agent IDs in `pipeline-state.json` under `agent_registry` are only valid within the same Claude conversation session. On pipeline resumption in a new session, the registry must be cleared (`agent_registry = {}`) because subagent transcripts don't survive across sessions. The orchestrator clears it automatically on load. Don't rely on agent_registry for cross-session state — use `memory: local` and shared files for that.
- **Tabular ML frameworks skip divergence monitoring**: When the detected framework is scikit-learn, XGBoost, or LightGBM, the orchestrator sets `divergence_metric` to `null` and skips the monitor skill. The baseline skill skips GPU profiling and throughput estimation for these frameworks.
- **Research findings files can be multiple**: `research-findings.md` (Phase 5 web search), `research-findings-method-proposals.md` (Phase 7 pre-loop), `research-findings-method-proposals-iter<N>.md` (Phase 7 mid-loop triggers). The research skill's deduplication checks all of these to avoid re-proposing tried techniques.
