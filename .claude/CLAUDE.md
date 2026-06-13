# CLAUDE.md

Architecture details and operational rules for agents. For installation, setup, dashboard, tests, and general usage — see README.md.

Entry point: `/optimize <model-path>` → `commands/optimize.md` → `ml-optimizer:orchestrate` skill.

## MCP Server Dependencies

These are installed separately — not bundled with the plugin. **gitnexus is required**; alphaxiv, context7, and claude-mem are recommended.

| MCP Server | Purpose | Used by | Required? |
|------------|---------|---------|-----------|
| **alphaxiv** | arXiv paper search, paper content extraction, PDF Q&A, GitHub repo exploration | research-agent (6 tools), implement-agent (2 tools) | No — falls back to WebSearch/WebFetch |
| **gitnexus** | Code knowledge graph — index a repo and query its structure, call-graph, and blast-radius (`mcp__gitnexus__context`, `mcp__gitnexus__query`, `mcp__gitnexus__impact`) | implement-agent (3 tools), research-agent (3 tools) | **Yes** — required; the pipeline indexes every code repo and agents must query it to understand code |
| **context7** | Framework API documentation lookup (PyTorch, TensorFlow, etc.) | research-agent, implement-agent | No — falls back to WebSearch |
| **claude-mem** | Cross-session memory — recalls past optimization sessions, avoids re-proposing failed techniques | research-agent, orchestrator | No — works without but loses cross-session learning |

Install: `claude mcp add --transport http --scope user alphaxiv https://api.alphaxiv.org/mcp/v1`. For gitnexus (required): `npm install -g gitnexus && gitnexus setup` — `gitnexus setup` auto-registers the gitnexus MCP server for Claude Code (manual fallback for MCP registration only: `claude mcp add --transport stdio --scope user gitnexus gitnexus mcp`). For context7 and claude-mem, install from the Claude Code marketplace.

**gitnexus is a hard prerequisite** — like git, it is required, not optional. Phase 2 verifies it is installed and blocks the pipeline if absent (see "GitNexus code-graph understanding" below). The plugin still functions without the *optional* servers but benefits significantly from alphaxiv (better paper discovery and analysis) and claude-mem (learning across optimization sessions).

## Architecture

### Plugin Structure

```
.claude-plugin/plugin.json      — Plugin metadata (name, version)
commands/optimize.md             — /optimize slash command (entry point)
skills/                          — Skill definitions (SKILL.md files)
skills/evolve/ShinkaEvolve/      — Git submodule (SakanaAI/ShinkaEvolve) for evolutionary code mutation
skills/shinka-*/                 — Symlinks → evolve/ShinkaEvolve/skills/shinka-* (created by setup_evolve.sh)
agents/                          — 10 agent definitions (9 subagents + orchestrator-agent main-thread)
scripts/                         — Python utilities (stdlib only)
tests/                           — pytest test suite
```

### Skill Pipeline (Orchestrator Flow)

The `orchestrate` skill coordinates a 10-phase pipeline. Each phase dispatches a named agent via `Agent(subagent_type="ml-optimizer:<name>-agent")`. Persistent agents (research, implement, tuning, analysis, monitor) are resumed via `SendMessage(to: agentId)` for subsequent dispatches; ephemeral agents (prerequisites, baseline, experiment, report) get fresh spawns. The orchestrator drives Phase 7 and Phase 8 directly, dispatching tuning/experiment/monitor/analysis agents per iteration:

```
Phase 0+1: Discovery & Planning (plan mode throughout — multi-round refinement until user approves)
         Discovery Q&A → write goals → analyze codebase → present plan → user refines → repeat
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
         [code_evolution] → tuning (evolve HPs) → evolve → tuning (training HPs) → experiment (scope_level=full only)
         [research_round] → cadence-based research (when method proposals enabled)
Phase 8: Method stacking (orchestrator driven, when analysis advises):
         Sequential accumulation — merge best methods one by one
         LLM conflict resolution, skip-on-failure
         Per step: analyze → tuning (evolve HPs) → evolve → tuning (training HPs) → experiment
         Analysis agent loops until improvement or recommends stop
Phase 9: report → Final optimization report
         review → Session review (what worked, what didn't, how to improve)
```

### Dispatch Chain & Output Map

```
/optimize → orchestrator-agent (main thread, settings.json)
  │
  ├─ Phase 2: Agent(prerequisites-agent)          [ephemeral]
  │   → results/prerequisites.json, prepared-data/ (if data prep needed)
  │
  ├─ Phase 3: Agent(baseline-agent)               [ephemeral]
  │   → results/baseline.json, logs/baseline/train.log
  │
  ├─ Phase 5: Agent(research-agent)               [persistent → SendMessage]
  │   → reports/research-findings.md
  │
  ├─ Phase 6: Agent(implement-agent)              [persistent → SendMessage]
  │   → results/implementation-manifest.json, git branches ml-opt/<slug>
  │
  ├─ Phase 7: Orchestrator drives experiment loop  [dispatches per iteration]
  │   │  Orchestrator DECIDES action based on analysis, DISPATCHES workers:
  │   │
  │   ├─ hp_tune (default): Orchestrator dispatches:
  │   │   ├─ tuning-agent (SendMessage)              → proposed-configs/round-N-<type>/exp-*.json (top-level dir, not under results/)
  │   │   ├─ experiment-agents (parallel, ephemeral) → results/round-N-<type>/exp-*.json
  │   │   │                                            logs/round-N-<type>/<exp-id>/train.log
  │   │   │                                            scripts/round-N-<type>/<exp-id>/train.sh
  │   │   │                                            artifacts/round-N-<type>/<exp-id>/
  │   │   ├─ monitor-agent (SendMessage, concurrent) → (no file output, relay only)
  │   │   └─ analysis-agent (SendMessage)            → reports/batch-<N>-analysis.md (per-batch, required)
  │   │                                                reports/dead-ends.json, dead-ends.md
  │   │                                                reports/research-agenda.json, research-agenda.md
  │   │                                                reports/suggestion-history.json
  │   │                                                relay to orchestrator
  │   │
  │   ├─ code_evolution: Orchestrator dispatches:
  │   │   ├─ tuning-agent (evolve HPs)           → evolve_recommendation
  │   │   └─ implement-agent (evolve skill)       → git branch ml-opt/evolved-<slug>
  │   │
  │   └─ method_proposal: Orchestrator dispatches:
  │       ├─ research-agent (SendMessage)         → reports/research-findings-method-proposals*.md
  │       └─ implement-agent (SendMessage)        → results/implementation-manifest.json + branches
  │
  ├─ Phase 8: Orchestrator drives stacking loop
  │   └─ Per stack step: implement(merge) → experiment → analysis → [evolve] → [hp-tune]
  │       → results/round-N-stacked/exp-*.json, git branches ml-opt/stack-<N>
  │
  └─ Phase 9: Agent(report-agent) [ephemeral] + analysis-agent (review mode)
      → reports/final-report.md, reports/progress_chart.png
      → reports/session-review.md, reports/dashboard.html, results-table.md

Cross-cutting outputs (managed by scripts or multiple agents):
  round_manager.py       → results/rounds-manifest.json (round lifecycle)
  error_tracker.py       → reports/error-log.json (error tracking)
  pipeline_state.py      → pipeline-state.json (phase, iteration, agent_registry)
  goal_memory.py         → optimization-goals.json, learned-behaviors.json (goal anchoring)
  excalidraw_gen.py      → artifacts/*.excalidraw (on-demand diagrams)
  Multiple agents        → dev_notes.md (running session log, appended by many agents)
```

#### Directory Structure

The plugin creates an `<exp_root>/` directory — the user chooses the name and location at Phase 0, there is no hardcoded default. The layout inside `<exp_root>/` is:

```
<exp_root>/
├── artifacts/
│   ├── round-N-<type>/                   — Per-round artifact grouping
│   │   └── <exp-id>/                     — Checkpoints, visualizations
│   └── *.excalidraw                      — Excalidraw diagrams (on-demand)
├── logs/
│   ├── baseline/train.log                — Baseline training log
│   └── round-N-<type>/                   — Per-round log grouping
│       └── <exp-id>/train.log            — Training log (eval.log if separate eval)
├── prepared-data/                         — Prepared dataset (if preprocessing needed)
├── proposed-configs/
│   └── round-N-<type>/                   — HP config proposals (per round)
│       └── exp-*.json                    — Proposed HP configurations
├── reports/
│   ├── dashboard.html                    — Self-contained HTML progress dashboard
│   ├── dead-ends.json                    — Dead-end catalog
│   ├── dead-ends.md                      — Human-readable companion
│   ├── error-log.json                    — Structured error event log
│   ├── final-report.md                   — Final optimization report
│   ├── progress_chart.png                — Matplotlib progress chart
│   ├── research-agenda.json              — Living research agenda
│   ├── research-agenda.md                — Human-readable companion
│   ├── research-findings.md              — Web search research findings
│   ├── research-findings-method-proposals.md — LLM knowledge-mode proposals
│   ├── session-review.md                 — Session review
│   └── suggestion-history.json           — Suggestion feedback loop
├── results/
│   ├── baseline.json                     — Baseline metrics (immutable after Phase 3)
│   ├── prerequisites.json                — Prerequisites check report
│   ├── implementation-manifest.json      — Validated proposal branches
│   ├── rounds-manifest.json              — Round index (source of truth)
│   └── round-N-<type>/                   — Types: hp, evolved, research, stacked, meta
│       └── exp-*.json                    — Schema-validated experiment results
├── scripts/
│   ├── baseline/train.sh                 — Baseline training script
│   └── round-N-<type>/                   — Per-round script grouping
│       └── <exp-id>/train.sh             — Training script (eval.sh if separate eval)
├── dev_notes.md                           — Running session log
├── learned-behaviors.json                 — Accumulated behavioral memory
├── optimization-goals.json                — Goal anchor (Phase 0, read by all agents)
├── pipeline-state.json                    — Resumable pipeline state
└── results-table.md                       — Auto-generated Markdown results summary
```

### Metric Routing Rule

Always monitor `"loss"` for divergence detection (monitor skill). Use the user's `primary_metric` (accuracy, PSNR, F1, etc.) for analyze and hp-tune skills.

### Branch-Aware Experiments

The implement skill creates `ml-opt/<slug>` branches per research proposal. The experiment loop tests each branch with HP variations. The tuning agent groups results by `code_branch` — identical HPs on different branches are treated as independent experiments.

### Agent Definitions (`agents/`)

Ten agent types total — nine subagent types plus one main-thread agent — each with a preloaded skill and specified tool access. `orchestrate` is the user-facing entry point (invoked via `/optimize`). Other skills are preloaded into agents via the `skills:` array in their agent definitions. The `orchestrator-agent` is the main-thread agent activated by `settings.json` — it loads the orchestrate skill and auto-starts Phase 0 via `initialPrompt`. All agents have `memory: local` for persistent role-specific memory at `.claude/agent-memory-local/<agent-name>/` in the target project.

**Persistent agents** — dispatched once via `Agent(subagent_type=...)`, resumed via `SendMessage(to: agentId)` for subsequent tasks. The orchestrator tracks agent IDs in `pipeline-state.json` under `agent_registry` and falls back to fresh dispatch if resumption fails.

**Ephemeral agents** — fresh `Agent()` spawn each time (single-use or parallel tasks).

**Procedural agents** (`model: sonnet[1m]` — lower cost/latency, no ultrathink):
- **baseline-agent** *(ephemeral)*: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:baseline]`
- **monitor-agent** *(persistent)*: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:monitor]`
- **experiment-agent** *(ephemeral)*: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:experiment]`
- **prerequisites-agent** *(ephemeral)*: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:prerequisites]`

**Analytical agents** (`model: opus[1m]`, ultrathink prompting):
- **research-agent** *(persistent)*: WebSearch, WebFetch, Read, Write, Bash, Glob, Grep, Skill, alphaxiv MCP tools (6), gitnexus MCP tools (3: `context`, `query`, `impact`) — skills: `[ml-optimizer:research, claude-mem:mem-search, superpowers:verification-before-completion]`
- **tuning-agent** *(persistent)*: Read, Write, Bash, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:hp-tune, claude-mem:mem-search, superpowers:verification-before-completion]`
- **implement-agent** *(persistent)*: Bash, Read, Write, Edit, LSP, Glob, Grep, Skill, WebSearch, WebFetch, alphaxiv MCP tools (2: repo reader, PDF Q&A), gitnexus MCP tools (3: `context`, `query`, `impact`) — skills: `[ml-optimizer:implement, ml-optimizer:evolve, ml-optimizer:shinka-setup, ml-optimizer:shinka-convert, ml-optimizer:shinka-run, ml-optimizer:shinka-inspect, superpowers:systematic-debugging, superpowers:verification-before-completion, karpathy-skills:karpathy-guidelines]` — runs the implement skill to apply the selected research proposals as git branches `ml-opt/<slug>`, **sequentially, inside a git worktree outside `<exp_root>/`** (implement skill Step 3.1/4), with progressive validation incl. `LSP` (pyright). The orchestrator dispatches it once in Phase 6 (step 1). Post-implementation review (`feature-dev:code-reviewer`, `pr-review-toolkit:silent-failure-hunter`, `pr-review-toolkit:pr-test-analyzer`) is orchestrator-driven in Phase 6 (steps 6–7)
- **analysis-agent** *(persistent)*: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:analyze, claude-mem:mem-search, superpowers:verification-before-completion]` (includes session review mode)
- **report-agent** *(ephemeral)*: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:report, superpowers:verification-before-completion]`

**Main-thread agent** (activated by `settings.json`):
- **orchestrator-agent**: Agent, Read, Write, Edit, Bash, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:orchestrate, superpowers:verification-before-completion]` — main thread when plugin is enabled, auto-starts Phase 0 via `initialPrompt: "/ml-optimizer:orchestrate"`

For parallel execution, use `run_in_background: true`. External skills are also available:
- **research-agent**: Uses `context7` for framework API docs, `claude-mem:mem-search` for cross-session learning, `alphaxiv` MCP for academic paper search/analysis (6 tools: embedding search, full-text search, agentic retrieval, paper content, PDF Q&A, GitHub repo reader), and `gitnexus` MCP (`context`/`query`/`impact`) — **required** — to index candidate reference repos (immediately after clone) and query their structure/call-graph for a feasibility read before recommending a `from_reference` strategy. Querying gitnexus is mandatory, not best-effort
- **implement-agent**: Uses `context7` for API docs, `superpowers:systematic-debugging` for error recovery, `superpowers:verification-before-completion` to confirm changes actually work before finishing, `karpathy-skills:karpathy-guidelines` to keep code changes surgical and avoid overcomplication, `alphaxiv` MCP for reference repo exploration (`read_files_from_github_repository`) and paper clarification (`answer_pdf_queries`), and `gitnexus` MCP (`context`/`query`/`impact`) — **required** — to index a cloned reference repo (`from_reference`, immediately after clone) and the target repo, then query the call-graph and blast-radius (`impact`) before editing. Querying gitnexus before adapting/editing code is mandatory, not best-effort. During validation it runs `LSP` (pyright) diagnostics on modified `.py` files to catch undefined names, type mismatches, and unresolved imports statically before any GPU time. The orchestrator dispatches **one implement-agent** for all selected proposals (Phase 6 step 1); it implements them sequentially in a git worktree (Step 3.1). **Post-implementation review is orchestrator-owned** (Phase 6 steps 6–7): `feature-dev:code-reviewer` + `pr-review-toolkit:silent-failure-hunter` per validated branch (step 6, the latter catches swallowed NaN losses, failed CUDA/optimizer ops, `except: pass` around training/eval) and `pr-review-toolkit:pr-test-analyzer` on the unit test (step 7, advisory).
- **orchestrator**: Uses `claude-mem:mem-search` in Phase 1 for cross-session recall, `superpowers:brainstorming` in Phase 0/4 for complex multi-objective optimization scenarios

### Python Scripts (`scripts/`)

All scripts work as both importable modules and CLI tools:

| Script | CLI Usage |
|--------|-----------|
| `${CLAUDE_PLUGIN_ROOT}/scripts/gpu_check.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gpu_check.py` — parse nvidia-smi |
| `${CLAUDE_PLUGIN_ROOT}/scripts/parse_logs.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/parse_logs.py <logfile>` — parse kv/JSON/CSV/XGBoost/HuggingFace Trainer logs |
| `${CLAUDE_PLUGIN_ROOT}/scripts/detect_divergence.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/detect_divergence.py '<json_values>' [--higher-is-better] [--model-category rl\|generative\|supervised] [--explosion-threshold N] [--plateau-patience N]` — detect NaN/explosion/plateau with configurable thresholds. Also: `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/detect_divergence.py --check-overfitting '<train_json>' '<val_json>' [--higher-is-better] [--patience N] [--min-gap F] [--model-category rl\|generative\|supervised]` — detect overfitting |
| `${CLAUDE_PLUGIN_ROOT}/scripts/result_analyzer.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/result_analyzer.py <results_dir> <metric> [baseline_id] [lower_is_better]` — full analysis. Also: `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/result_analyzer.py <results_dir> compare <exp_id_1> <exp_id_2> [metric] [lower_is_better]` — pairwise comparison |
| `${CLAUDE_PLUGIN_ROOT}/scripts/experiment_setup.py` | Generates experiment IDs and directory structure |
| `${CLAUDE_PLUGIN_ROOT}/scripts/implement_utils.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/implement_utils.py <findings.md> '<indices_json>'` — parse proposals; also `clone <url> <dest>`, `analyze <path>`, and `diff <project_root> <branch>` subcommands |
| `${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py available\|mcp-registered\|require\|index <path> [--force]\|is-indexed <path>` — required GitNexus code-graph helper. `available` checks the `gitnexus` CLI is on PATH; `mcp-registered` prints `{"registered": true\|false\|null}` (true = MCP server registered, false = not registered, null = cannot determine because the `claude` CLI is absent; always exits 0); `require` reports availability plus `"mcp_registered": true\|false\|null` and exits nonzero only when the CLI itself is absent (its install guidance is `["npm install -g gitnexus", "gitnexus setup"]`); `index` runs `gitnexus analyze <path> --index-only` (non-invasive — never injects a GitNexus section into the indexed repo's CLAUDE.md/AGENTS.md and never installs `.claude/` skills there; writes `<path>/.gitnexus`, auto-adds `.gitnexus/` to the repo git exclude, skips re-indexing an already-indexed path unless `--force`, never raises — failures reported via the returned dict); `is-indexed` checks for an existing index. The wrapper never raises, but the skills treat `available()==False` / index `success: false` as a **hard error** (halt with install/repair guidance), not a fallback |
| `${CLAUDE_PLUGIN_ROOT}/scripts/pipeline_state.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/pipeline_state.py <exp_root> validate\|save\|load\|cleanup\|verify-baseline\|gate\|log-gate\|log-decision\|replay-check\|decisions` — phase gates, decision logging |
| `${CLAUDE_PLUGIN_ROOT}/scripts/schema_validator.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/schema_validator.py <filepath> result\|baseline\|manifest\|prerequisites [--strict]` — validates JSON schemas. `--strict` enforces completeness. Also: `relay <route> <json>` for inter-agent relay validation (5 routes) |
| `${CLAUDE_PLUGIN_ROOT}/scripts/plot_results.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/plot_results.py <results_dir> <metric> comparison\|timeline\|sensitivity <hp>\|progress [--higher-is-better]` — ASCII charts + matplotlib progress chart |
| `${CLAUDE_PLUGIN_ROOT}/scripts/prerequisites_check.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/prerequisites_check.py scan-imports\|check-packages\|detect-env\|detect-format\|detect-format-project\|validate-data\|bulk-install-cmd\|gpu-install-cmd` — dataset, environment, and GPU-aware install validation |
| `${CLAUDE_PLUGIN_ROOT}/scripts/dashboard.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/dashboard.py <exp_root> [--live] [--table] [--serve --port 8080]` — generate self-contained HTML dashboard with progress timeline, results table, HP sensitivity, research agenda, error summary, method explanations. `--live` enables 30s auto-refresh. `--table` generates `results-table.md` (Markdown results summary). |
| `${CLAUDE_PLUGIN_ROOT}/scripts/excalidraw_gen.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/excalidraw_gen.py <exp_root> pipeline\|comparison\|hp-landscape\|architecture <args>` — generate Excalidraw JSON diagrams (pipeline overview, experiment comparison, HP landscape, architecture changes) |
| `${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log\|show\|patterns\|summary\|success\|proposals\|rank\|log-suggestion\|suggestion-history\|dead-end <add\|list\|check>\|agenda <init\|update\|list\|add>` — error tracking, pattern detection, success metrics, proposal outcomes, suggestion ranking, suggestion history, dead-end catalog, research agenda |
| `${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> init-goals\|read-goals\|log-behavior <category> <json>\|query-behaviors [category]\|validate-output <agent> <json>\|summary\|sync-from-errors` — goal anchoring, behavioral memory, agent output validation, compact briefings |
| `${CLAUDE_PLUGIN_ROOT}/scripts/setup_evolve.sh` | `bash scripts/setup_evolve.sh` — initialize ShinkaEvolve submodule and create skill symlinks for auto-discovery |
| `${CLAUDE_PLUGIN_ROOT}/scripts/round_manager.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/round_manager.py <exp_root> create-round\|current-round\|register-experiment\|close-round\|next-id\|check-baseline\|check-prerequisites\|check-manifest\|check-round\|check-proposals` — round lifecycle + completeness checking |
| `${CLAUDE_PLUGIN_ROOT}/scripts/validate_experiment_write.py` | PreToolUse hook — validates Write/Edit operations to experiment result files against schema + correct round directory |



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

The orchestrator can be stopped and resumed. On restart it reads `pipeline-state.json` and uses `cleanup_stale()` to handle interrupted experiments (marks them as failed after a timeout). Phase validation via `validate_phase_requirements()` prevents cascading failures. Pipeline state persists Phase 0 user choices (`primary_metric`, `divergence_metric`, `divergence_lower_is_better`, `lower_is_better`, `target_value`, `train_command`, `eval_command`, `train_data_path`, `val_data_path`, `prepared_train_path`, `prepared_val_path`, `env_manager`, `env_name`, `model_category`, `user_papers`, `method_proposal_scope`, `method_proposal_iterations`, `hp_batches_per_round`, `fixed_time_budget`, `fixed_epoch_budget`) via `save_state(user_choices={...})` so they survive interruptions without re-asking the user. The experiment loop also persists `consecutive_stop_count` (telemetry only — not an exit trigger), `stuck_protocol_triggered` (the loop-exit state flag: set `true` when the stuck protocol returns no new in-scope proposals; combined with an empty research agenda and a flat best metric it defines the *fixpoint* at which the orchestrator exits to Phase 9, and reset `false` whenever fresh proposals or metric improvement appear), `baseline_checksum` (SHA-256 of baseline metrics for integrity verification), and `agent_registry` (persistent agent IDs for SendMessage resumption) at the root level of pipeline state. On new session start, `agent_registry` is cleared since subagent transcripts are session-scoped — all agents start fresh. A separate `user-choices-backup.json` provides redundant recovery if the main state file corrupts.

### Multi-Run Pattern (one `<exp_root>` = one optimization run)

The plugin does **not** run-namespace state files inside `<exp_root>/`. Each `<exp_root>` is one self-contained optimization run — its behavioral memory, round manifest, and reports are all scoped to that directory. For a new optimization direction on the same project, point `exp_root` at a new directory at Phase 0. The plugin has no hardcoded output location; `exp_root` can be any absolute path.

```
# Each run gets its own <exp_root>:
<exp_root> = ~/my-project/runs/01-label-smoothing/   # first run
<exp_root> = ~/my-project/runs/02-augmentation/      # second run, fresh state
<exp_root> = ~/my-project/runs/03-architecture/      # third run
```

**Breadcrumb (`.claude/ml-optimizer.json`):** The orchestrator writes a breadcrumb at the project root so hooks can find the active `<exp_root>`. The breadcrumb tracks all runs — each Phase 0 appends to `runs` and sets `active`:

```json
{"active": "/home/user/runs/02-augmentation", "runs": ["/home/user/runs/01-label-smoothing", "/home/user/runs/02-augmentation"]}
```

Hooks and the write-validator read `active` to resolve the current `<exp_root>`. The old single-run format (`{"exp_root": "..."}`) is still supported for backward compatibility.

Phase 0 (`skills/orchestrate/references/phase-0-discovery.md` Step 1.1) detects a prior completed run (`phase == 9`) and prompts the user to start a new directory or continue. The `cwd-changed-detect-experiments.sh` hook emits a "completed optimization found" message on phase 9 entry. Cross-run comparison is not automated — diff the `results-table.md` files manually.

## Key Design Patterns

- **Resumable subagents**: 5 persistent agents (research, implement, tuning, analysis, monitor) are dispatched once via `Agent()` and resumed via `SendMessage(to: agentId)` for subsequent tasks. This preserves accumulated context (search results, HP trends, codebase knowledge) across the pipeline. 4 ephemeral agents (prerequisites, baseline, experiment, report) get fresh spawns. The orchestrator tracks agent IDs in `pipeline-state.json` under `agent_registry` and falls back to fresh dispatch if resumption fails. Agent IDs are session-scoped — cleared on new session start since subagent transcripts don't survive across sessions.
- **Inter-agent communication (orchestrator relay)**: When resuming a persistent agent, the orchestrator includes a `CONTEXT FROM OTHER AGENTS:` section with relevant findings from other agents. This enables indirect communication: analyze findings reach hp-tune, monitor OOM info reaches hp-tune, research proposals reach implement, etc. The orchestrator acts as a message bus. Key relay routes: analyze→tuning (correlations, branch scores), analyze→research (pivot reasons, dead-ends), monitor→tuning (OOM constraints), research→implement (proposals), experiments→analyze (batch results).
- **Non-git fallback**: If the target project isn't a git repo, the implement skill uses file backups instead of branches. Each proposal is validated against a clean baseline backup (restore-before-apply pattern) to prevent cross-proposal code leakage. This forces sequential (not parallel) experiment execution.
- **Loop exit conditions**: The experiment loop is autonomous — runs until: (1) target metric achieved, (2) user manually stops, or (3) the orchestrator judges approaches are exhausted. When analysis advises stop, the orchestrator invokes the stuck protocol (research for fresh ideas), then makes an evidence-based decision: continue if research yields fresh non-dead-end proposals, exit to Phase 9 if the search is genuinely out of directions (no new proposals, flat metric, empty agenda). There is no hardcoded stop-count threshold — `consecutive_stop_count` is one input to the judgment.
- **Proposal priority scoring**: `(impact * confidence) / (11 - min(feasibility, 10))` — feasibility clamped to [1,10] to prevent division by zero.
- **Spearman correlation**: `${CLAUDE_PLUGIN_ROOT}/scripts/result_analyzer.py` uses rank correlation with average-rank tie-breaking to identify HP-metric relationships (no scipy dependency).
- **Dual implementation strategy**: Research proposals include an `implementation_strategy` field (`from_scratch` or `from_reference`). The implement agent dispatches accordingly — either implementing from paper descriptions (Section 8) or cloning and adapting reference repos (Section 9). Strategy is decided by the research agent based on repo availability and quality.
- **Research skill modes**: The research skill accepts `source` (`"web"` | `"knowledge"` | `"both"`), `scope_level` (`"training"` | `"architecture"` | `"full"`), and `output_path` parameters. Knowledge mode skips web search and uses LLM training knowledge only.
- **Scope-gated pivots**: The analysis agent's pivot decision tree respects `scope_level`: `"training"` (HP-only) disables research and code_evolution pivots — only HP adjustments are available. `"architecture"` enables research but not ShinkaEvolve. `"full"` enables everything including code_evolution via ShinkaEvolve. This ensures the pipeline matches what the user asked for.
- **Auto-resolution behaviors**: The pipeline auto-resolves many situations: Phase 2 partial prereqs → proceed with warnings, RL polarity → auto-infer from metric name, dirty working trees → no action needed (implementation runs in an isolated git worktree, main tree untouched), environment mismatches → use detected manager, missing conda envs → auto-create, no eval command → fall back to training output metrics. Only unrecoverable errors (Phase 2 failed, baseline failed) block the pipeline. Decisions are logged to dev_notes and error tracker for post-session review.
- **Parallel research**: All WebSearch calls in the research skill are issued simultaneously in a single tool-call message, alongside 3 alphaxiv search calls (embedding similarity, full-text keyword, agentic retrieval). WebFetch follow-ups for different URLs are also parallelized. Domain-specific query sets (NLP, CV, RL, time-series) are issued alongside generic queries. If alphaxiv MCP is unavailable, WebSearch provides full coverage as fallback.
- **Sequential implementation in a worktree**: Phase 6 dispatches a **single implement-agent** that implements all selected proposals sequentially, one `ml-opt/<slug>` branch each, **inside a git worktree outside `<exp_root>/`** (implement skill Step 3.1) so the project's main working tree is never disturbed; the worktree is removed at the end and the branches persist. Branches are created with `git checkout -b ml-opt/<slug> <original_branch>` (branching off the base commit, which is valid even though `<original_branch>` is checked out in the main tree). Not parallelized — implementation is reasoning/editing, not GPU-bound (unlike Phase 7 experiments). Consistent with the other single-dispatch phases (prerequisites, baseline, research).
- **Configurable divergence thresholds**: `${CLAUDE_PLUGIN_ROOT}/scripts/detect_divergence.py` supports per-model-category threshold overrides via `MODEL_CATEGORY_DEFAULTS` dict and `--model-category` CLI flag. RL models use `explosion_threshold=20.0, plateau_patience=50` (prevents false positives on reward spikes). Generative models use `explosion_threshold=10.0, plateau_patience=40` (accommodates slow convergence). Individual thresholds can also be overridden via `--explosion-threshold` and `--plateau-patience` CLI flags.
- **Experiment timeout**: Each experiment has a hard timeout of `baseline_training_time * 3` (fallback: 6 hours). Timed-out experiments are killed and marked `status: "timeout"`.
- **Research failure recovery**: If web search fails (both WebSearch and alphaxiv), the orchestrator retries with `source: "knowledge"` (LLM-only). If that also fails, it continues with HP-only optimization. Each fallback is logged. Within a search, alphaxiv failure alone does not trigger the knowledge fallback — WebSearch results are sufficient to proceed.
- **alphaxiv MCP integration**: The research agent uses all 6 alphaxiv MCP tools for academic paper discovery (3 search tools run in parallel), paper content extraction (`get_paper_content`, `answer_pdf_queries`), and reference repo exploration (`read_files_from_github_repository`). The implement agent uses 2 alphaxiv tools (`read_files_from_github_repository` for pre-clone repo assessment, `answer_pdf_queries` for clarifying ambiguous implementation steps from source papers). All alphaxiv searches run in parallel with WebSearch. alphaxiv is optional — if the MCP server is unavailable, all workflows fall back to WebSearch/WebFetch transparently.
- **GitNexus code-graph understanding (REQUIRED)**: GitNexus (required MCP + CLI, mirrors the alphaxiv integration pattern) indexes a repo into a queryable code knowledge graph. It is a **hard prerequisite** — on par with git — and there is **no grep/analyze fallback for code understanding**. **Querying the graph is MCP-only by design**: agents query exclusively via `mcp__gitnexus__context`/`mcp__gitnexus__query`/`mcp__gitnexus__impact` — there is **no gitnexus-CLI query fallback** (agents never run `gitnexus query`/`context`/`impact` from Bash). If the MCP server is not registered or fails, code understanding fails — recovery is `gitnexus setup` then restart the session (MCP tools load at session start, so a freshly-registered server needs a session restart to become available). **Phase 2 (prerequisites) verifies it is installed** (`python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py available`); if the CLI is unavailable, Phase 2 BLOCKS as an unrecoverable prerequisite failure (on par with "Phase 2 failed blocks the pipeline") with the install instructions (`npm install -g gitnexus && gitnexus setup`; manual MCP-registration fallback `claude mcp add --transport stdio --scope user gitnexus gitnexus mcp`). Phase 2 is **best-effort about MCP registration**: if the CLI is installed but the MCP server is not registered (`gitnexus_utils.py mcp-registered` / `require` reports `mcp_registered: false`), it WARNS rather than hard-blocking — only a missing CLI hard-blocks. **Every code repo is indexed**: the TARGET project is indexed once at Phase 2 (`python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py index <project_root>`, graph at `<project_root>/.gitnexus`); EVERY reference repo is indexed immediately after clone (`python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py index <ref_repo>`). Indexing is **non-invasive** — the wrapper runs `gitnexus analyze <path> --index-only` (never a bare `gitnexus analyze`), so it does NOT inject a GitNexus section into the indexed repo's CLAUDE.md/AGENTS.md and does NOT install `.claude/` skills there; the indexed repo (or worktree) is never contaminated. The implement agent indexes the cloned reference repo (`from_reference` strategy) and the target repo, then **must** query structure/call-graph with `mcp__gitnexus__context`/`mcp__gitnexus__query` and blast-radius with `mcp__gitnexus__impact` before editing — so changes are surgical and side-effects are understood. The research agent indexes candidate reference repos and **must** query them for a feasibility read before recommending `from_reference`. Querying gitnexus is mandatory, not best-effort. `${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py` provides `available()`/`index()`/`is_indexed()` helpers (CLI: `available`, `mcp-registered`, `require`, `index <path> [--force]`, `is-indexed <path>`); `index()` runs `gitnexus analyze <path> --index-only` and **never raises** — any failure is reported via the returned dict. The wrapper never raises, but the skills treat `available()==False` / index `success: false` as a **hard error** (halt with install/repair guidance), NOT a silent fallback. `implement_utils.py analyze` remains only for its narrow framework-detection role — it is NOT a gitnexus fallback. `.gitnexus/` index artifacts are auto-excluded by the wrapper (it adds `.gitnexus/` to the repo's git exclude on a successful index) but must still never be committed (never `git add`-ed).
- **OOM feedback loop**: When experiments OOM, the batch size is recorded in the error tracker. On the next hp-tune invocation, `max_batch_size` is passed to prevent re-proposing configs that will OOM.
- **All-diverge recovery**: If all experiments in a batch diverge, a recovery batch with halved learning rates is attempted before stopping.
- **HP-only research routing**: Research proposals with `type: "hp_only"` skip the implement skill and are routed directly to hp-tune as search space modifications.
- **Tabular ML HP strategy**: For tree-based models (sklearn/XGBoost/LightGBM), iteration 1 explores `max_depth`/`n_estimators` first instead of learning rate.
- **Training budget options**: Phase 0 offers two budget modes: `fixed_time_budget` (seconds) for wall-clock-normalized comparison, or `fixed_epoch_budget` (integer) for deterministic reproducibility. When set, both baseline AND experiments use the same budget. The baseline skill (Step 2.2) wraps training with `timeout` when `fixed_time_budget` is set, ensuring fair baseline-vs-experiment comparisons. Framework-native time limits (Lightning `--max_time`, HuggingFace `timeout` in TrainingArguments) are preferred when available. Results include `time_budget_seconds` for downstream analysis. HP-tune adjusts proposals for the budget (shorter convergence schedules, appropriate LR scaling). Makes experiment metrics directly comparable without duration normalization.
- **Small dataset awareness**: The research skill checks dataset size. For datasets under 5K samples, it shifts search toward low-data techniques (transfer learning, fine-tuning, few-shot learning, adapters, prompt tuning, synthetic data, semi-supervised methods) instead of heavy augmentation and regularization which underperform on small data.
- **Evolutionary code refinement**: When HP tuning shows diminishing returns, the analyze skill can recommend `pivot_type: "code_evolution"`. The orchestrator dispatches the implement-agent with the evolve skill (`Skill("ml-optimizer:evolve")`), which orchestrates the full ShinkaEvolve pipeline internally: `shinka-convert` (create task from best branch) → `shinka-run` (run evolution with file-based handoff, `SHINKA_PROVIDER=claude_code`) → `shinka-inspect` (extract best mutation) → commit as `ml-opt/evolved-<slug>`. Evolve HPs (`num_generations`, `population_size`) are tuning-agent-driven: the orchestrator dispatches the tuning agent to propose evolve HPs before dispatching the implement agent, based on prior evolution outcomes stored in `learned-behaviors.json` under category `evolve_hp`. Defaults: 10 generations, population 2. If ShinkaEvolve is unavailable, the evolve skill reports `shinkaevolve_unavailable` and the orchestrator falls back to the research → implement path. Setup: run `bash scripts/setup_evolve.sh` to init the submodule and create symlinks (`skills/shinka-*` → `skills/evolve/ShinkaEvolve/skills/shinka-*`). The symlinks are required for Claude Code's skill auto-discovery.
- **Auto-repair loop**: When training or evaluation commands fail during baseline establishment or experiment execution, the agent captures stderr, diagnoses the error, applies a fix (install package, adjust path, reduce batch size), and retries up to 3 times. OOM errors are not retried (deterministic). SyntaxErrors are not retried (code bugs). Identical errors on consecutive attempts skip further retries (loop detection). Each retry is logged to the error tracker. This is intra-agent retry, separate from the orchestrator's Phase 3 retry logic.
- **Goal anchoring & behavioral memory**: `${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py` maintains two project-scoped files: `optimization-goals.json` (goal anchor, written once at Phase 0) and `learned-behaviors.json` (accumulated behavioral memory). The orchestrator calls `validate-output` after hp-tune, research, and analyze dispatches to catch drift (frozen param changes, scope breaches, dead-end re-proposals, metric mismatches). Each agent also reads a compact `summary` combining goals + constraints + dead-ends before acting. All 10 agents have `memory: local` for persistent role-specific memory at `.claude/agent-memory-local/<agent-name>/` in the target project.
- **Immutable baseline**: After baseline is established (Phase 3), a SHA-256 checksum of the baseline metrics dict is stored in `pipeline-state.json`. Before each experiment batch (Phase 7) and on pipeline resumption, the checksum is verified against `baseline.json`. If the metrics have changed (accidental modification, file corruption, or tampering), the pipeline halts with a critical error. Prevents invalid experiment comparisons during long optimization runs.
- **Stuck protocol**: When analysis advises stop, the orchestrator reads error patterns, dead ends, and the research agenda, then dispatches research for fresh ideas. It then runs the **Exit Judgment**: if research returns new in-scope proposals (or the agenda has untried items, or the metric improved), reset `stuck_protocol_triggered`/`consecutive_stop_count` and continue; otherwise set `stuck_protocol_triggered=true`. Exit to Phase 9 only at the *fixpoint* — no new in-scope proposals AND empty agenda AND flat best metric (the in-scope idea space is exhausted with no progress to build on). There is no hardcoded stop-count threshold; the decision is logged via `pipeline_state.py log-decision`.
- **Research agenda as living document**: `${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py` maintains `research-agenda.json` — a prioritized list of optimization ideas that evolves over the session. The research skill initializes it from proposals (Phase 5). The analyze skill updates it after each batch: marking ideas as tried/improved/dead-end, adjusting priorities based on evidence, and adding new ideas suggested by experimental results. The hp-tune skill reads it to understand which untried techniques are high-priority. The report skill includes a summary in the final report.
- **Overfitting detection**: `${CLAUDE_PLUGIN_ROOT}/scripts/detect_divergence.py` provides `check_overfitting(train_values, val_values)` that compares train vs val metric trajectories. Detects when val metric worsens for N consecutive steps while train improves. Severity classified as mild/moderate/severe. Default patience=5 steps; model-category overrides: RL patience=10, generative patience=8. CLI: `--check-overfitting '<train_json>' '<val_json>'`.
- **HP interaction detection**: `${CLAUDE_PLUGIN_ROOT}/scripts/result_analyzer.py` provides `detect_hp_interactions()` that computes 2-way interaction terms using product of centered ranks. Reports interactions stronger than either individual HP correlation. Integrated into `analyze()` output.
- **Adaptive branch budget allocation**: `${CLAUDE_PLUGIN_ROOT}/scripts/result_analyzer.py` provides `compute_branch_scores()` that scores branches by `improvement_pct * confidence_factor`. HP-tune allocates experiment slots proportionally to scores in iteration 2+.
- **Checkpoint warm-starting**: Experiments can optionally warm-start from a previous checkpoint. `${CLAUDE_PLUGIN_ROOT}/scripts/experiment_setup.py` supports `checkpoint_path` parameter. HP-tune proposes warm-started configs (lower LR, fewer epochs, same-branch only) when enabled.
- **Dead-end catalog**: `${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py` maintains `dead-ends.json` tracking techniques conclusively shown to be unpromising. The research and hp-tune skills consult this catalog before proposing new techniques, preventing wasted budget on proven dead ends. Fuzzy matching (case-insensitive, substring containment, hyphen/underscore normalization) prevents near-duplicate re-proposals. The analyze skill logs dead ends when branches are pruned or all experiments fail.
- **Concurrent-safe error logging**: `${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py` uses `fcntl.flock()` file locking around the read-modify-write in `log_event()` to prevent concurrent agents from losing events.
- **Result file filtering**: `${CLAUDE_PLUGIN_ROOT}/scripts/result_analyzer.py` only loads `exp-*.json` and `baseline.json` files, preventing non-experiment files from inflating counts.
- **alphaxiv query format differences**: `embedding_similarity_search` expects 2-3 descriptive sentences, `full_text_papers_search` expects 3-4 short keywords (no quotes), and `agentic_paper_retrieval` expects a natural language question. Using the wrong format degrades result quality. All 3 must be called in parallel per alphaxiv documentation.
- **HuggingFace Trainer log format**: `${CLAUDE_PLUGIN_ROOT}/scripts/parse_logs.py` detects and parses HuggingFace Trainer's single-quote Python dict format (`{'loss': 0.5, 'epoch': 1.0}`).
- **Baseline eval auto-fallback**: If no eval command is found, baseline uses training output metrics instead of blocking on user input.
- **Pre-flight file validation**: The implement skill validates all `files_to_modify` exist before creating branches or starting implementation. Missing-file proposals are marked `preflight_failed`.
- **Tabular ML adaptive timeout**: For non-iterative frameworks, experiment timeout is computed from `fit_duration * (max_iters / profiling_iters) * 2` instead of a generic 4-hour fallback.
- **Method stacking (Phase 8)**: After independent method testing identifies ≥5 methods that improve over baseline, the orchestrator sequentially merges them in descending order of improvement. Each stack step creates `ml-opt/stack-<N>` by merging the next method into the current best stack. Clean merges proceed directly; conflicts are resolved by the implement-agent. If a combination degrades performance, that method is skipped. After each successful stack step, the analysis agent assesses the stacked result — if the stacked gain is less than the best individual method's gain (indicating method interference), it recommends `code_evolution` and the evolve skill (`Skill("ml-optimizer:evolve")`) optimizes code-level interactions via ShinkaEvolve (5 generations, population 2). This mirrors Phase 7's pattern where the analysis agent is always the decision-maker for evolution. If the evolved code improves over the pre-evolution stack, it becomes the new stack base; otherwise it's discarded. Optional HP-tuning (1-2 iterations, narrowed scope) follows on the (potentially evolved) code when the combo shows >1% improvement. Stacking state persists in `pipeline-state.json` (including `evolved_methods` tracking) for resumption. Requires git branch strategy — skipped for `file_backup` projects.
- **Structured ideation for knowledge mode**: The research skill's knowledge-based proposal generation (Phase 7 method proposals) uses a structured diverge-converge-refine process with 6 ideation lenses (Problem-First, Analogical Reasoning, What Changed Recently, Constraint Manipulation, Negation/Inversion, Composition/Decomposition). Generates 10-15 candidates, filters via scope/dead-end/two-sentence-test, refines survivors with implementation details.
- **Statistical confidence assessment**: The analyze skill (Step 2.2) computes effect sizes (Cohen's d) for HP impact when ≥5 experiments exist, and labels findings by confidence level (high/medium/low). Method attribution distinguishes whether improvements came from the code change, HP tuning, or their compound effect.
- **Reproducibility metadata**: The experiment skill (Step 1.3) captures random seeds, pip freeze snapshots, git SHA, and framework versions. Stored under `"reproducibility"` key in result JSONs. Enables exact reproduction of best experiments.
- **Report threats to validity**: The report template includes a "Threats to Validity" section covering single-seed risk, limited search space, dataset specificity, budget constraints, and noise margins.
- **Citation verification**: The report skill (Step 5.3) cross-references technique claims against experiment data and spot-checks source URL accessibility before writing the final report.

## Gotchas

- **`${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py validate-output` returns exit code 2 for violations**: Exit code 0 = valid, 1 = script error, 2 = validation violations found. The orchestrator should check the exit code and parse the JSON output for the `violations` array.
- **`${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py` imports from `${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py`**: For dead-end checks, it lazily imports `is_dead_end` and `get_dead_ends`. Both scripts must be in the same `scripts/` directory.
- **`${CLAUDE_PLUGIN_ROOT}/scripts/detect_divergence.py` CLI takes a JSON string, not a file path**: `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/detect_divergence.py '[0.5, 0.4, 100.0]'` — the quotes are required. Pass `--higher-is-better` for reward-like metrics. Pass `--model-category rl` for RL-appropriate thresholds.
- **`${CLAUDE_PLUGIN_ROOT}/scripts/implement_utils.py` has four CLI modes**: default (parse proposals), `clone <url> <dest>`, `analyze <path>`, and `diff <project_root> <branch>`. Each has different argument patterns. Note: `analyze` is for framework detection only — it is NOT a gitnexus fallback (gitnexus is required; see below).
- **Don't commit `.gitnexus/` index artifacts**: indexing (via `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py index <path>`, which runs `gitnexus analyze <path> --index-only`) writes a `.gitnexus/` directory into the indexed repo (target project and every cloned reference repo). The wrapper now auto-excludes `.gitnexus/` (adds it to the repo's git exclude on a successful index), but these are local index artifacts — still never `git add` or commit them.
- **Indexing is non-invasive (`--index-only`)**: index commands go through the wrapper (`python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py index <path>`), which runs `gitnexus analyze <path> --index-only` — NOT a bare `gitnexus analyze`. The `--index-only` flag keeps the index pure: it does NOT inject a GitNexus section into the indexed repo's CLAUDE.md/AGENTS.md and does NOT install `.claude/` skills there, so the indexed repo (or worktree) is never contaminated. Don't write prose implying a bare `gitnexus analyze <path>` is run or that indexing rewrites those files.
- **Querying is MCP-only by design**: agents query the code graph exclusively via the MCP tools `mcp__gitnexus__context`/`mcp__gitnexus__query`/`mcp__gitnexus__impact`. There is **no gitnexus-CLI query fallback** — never tell an agent to run `gitnexus query`/`context`/`impact` from Bash. If the MCP server is not registered or fails, code understanding fails — recovery is `gitnexus setup` then **restart the session** (MCP tools load at session start, so a freshly-registered server only becomes available after a restart). Phase 2 is best-effort about MCP registration: if the CLI is installed but the MCP server is not registered (`gitnexus_utils.py mcp-registered`/`require` reports `mcp_registered: false`), it WARNS rather than hard-blocking — only a missing CLI hard-blocks.
- **GitNexus is required, not optional**: There is no grep/analyze fallback for code understanding. Phase 2 verifies `gitnexus_utils.py available` and BLOCKS the pipeline as an unrecoverable prerequisite failure if the CLI is absent. The wrapper `gitnexus_utils.py` never raises, but the skills treat `available()==False` / index `success: false` as a hard error (halt with install/repair guidance: `npm install -g gitnexus && gitnexus setup`; manual MCP-registration fallback `claude mcp add --transport stdio --scope user gitnexus gitnexus mcp`).
- **`gitnexus setup` has global install side effects**: beyond auto-registering the gitnexus MCP server for Claude Code, `gitnexus setup` also installs gitnexus's own global skills (7) and PreToolUse/PostToolUse hooks into `~/.claude/`, affecting all Claude Code projects. Mention this so users aren't surprised.
- **Metric routing is split**: Monitor/divergence always uses loss (lower-is-better). Analyze/hp-tune use the user's `primary_metric`. Mixing these up causes silent wrong behavior.
- **Branch experiments are independent**: Results on `ml-opt/branch-a` tell you nothing about what HPs will work on `ml-opt/branch-b`. The tuning agent must group by `code_branch` before analyzing trends.
- **`agent_registry` is session-scoped**: Agent IDs in `pipeline-state.json` under `agent_registry` are only valid within the same Claude conversation session. On pipeline resumption in a new session, the registry must be cleared (`agent_registry = {}`) because subagent transcripts don't survive across sessions. The orchestrator clears it automatically on load. Don't rely on agent_registry for cross-session state — use `memory: local` and shared files for that.
- **Experiment results MUST live in round directories**: `exp-*.json` files must be written to `results/round-N-<type>/exp-*.json`, NEVER directly to `results/exp-*.json`. The PreToolUse hook (`validate_experiment_write.py`) blocks any Write/Edit that violates this pattern. Before dispatching experiment-agents, the orchestrator must call `round_manager.py create-round <type>` and pass the current round directory to the agent. Valid round types: `hp`, `evolved`, `research`, `stacked`.
- **Proposed configs are top-level with round structure**: HP tuning proposals go in `proposed-configs/round-N-<type>/exp-*.json` (top-level, NOT under `results/`). The PreToolUse hook validates this. The round directory must exist (created by `round_manager.py create-round`) before the tuning agent writes proposals.
- **Exp-ids are globally unique across rounds**: `exp-001` only exists in one round. `round_manager.py next-id` scans all round directories to find the next available ID. Don't try to reuse exp-ids across rounds — results from earlier rounds are preserved and still counted.
- **`round_manager.py` uses `fcntl.flock` for manifest writes**: `rounds-manifest.json` is protected by a file lock for concurrent safety. Manual edits to the manifest may be lost if agents are running. Always use `round_manager.py` CLI commands (`create-round`, `register-experiment`, `close-round`) to update it.
- **Tabular ML frameworks skip divergence monitoring**: When the detected framework is scikit-learn, XGBoost, or LightGBM, the orchestrator sets `divergence_metric` to `null` and skips the monitor skill. The baseline skill skips GPU profiling and throughput estimation for these frameworks.
- **Research findings files can be multiple**: `research-findings.md` (Phase 5 web search), `research-findings-method-proposals.md` (Phase 7 pre-loop), `research-findings-method-proposals-iter<N>.md` (Phase 7 mid-loop triggers). The research skill's deduplication checks all of these to avoid re-proposing tried techniques.
- **ShinkaEvolve must use the local submodule, not PyPI**: The PyPI package `shinka-evolve` lacks the `file_handoff_provider` module required for `SHINKA_PROVIDER=claude_code`. Always use the local submodule — either install editable (`pip install -e skills/evolve/ShinkaEvolve/`) via `setup_evolve.sh`, or prepend `PYTHONPATH=${CLAUDE_PLUGIN_ROOT}/skills/evolve/ShinkaEvolve:$PYTHONPATH` before `shinka_run`.
- **ShinkaEvolve subprocess Python resolution**: `shinka/launch/scheduler.py` falls back to `sys.executable` (the Python running ShinkaEvolve) when no conda env or activate script is configured. This usually resolves correctly. If evaluation subprocesses fail with "python not found", set `python_executable` in `LocalJobConfig` or ensure `python` is on PATH.
- **ShinkaEvolve file handoff timeout is configurable via `SHINKA_HANDOFF_TIMEOUT`**: Default is 600s (10 minutes). Set before launching `shinka_run` with `SHINKA_PROVIDER=claude_code`. The agent must write a `<id>.inprogress` marker when it picks up a pending request — this resets `shinka_run`'s timeout from the acknowledgment point. Without the marker, the timeout runs from request creation. `shinka_run` also writes `<id>.heartbeat` files every 5s so the agent can verify liveness.
- **3-checkpoint output enforcement**: Every agent output is enforced at 3 points: (1) **SubagentStart** injects the output contract — exact paths, schemas, and examples — so agents know what to produce before they start (`scripts/output_contract.py` via `hooks/subagent-start-inject-goals.sh`). (2) **PreToolUse** hook blocks invalid JSON writes: wrong path, bad schema, missing completeness fields (completed experiments need `iteration`, `method_tier`, `duration_seconds`; stacked need `code_branches`, `stacking_order`; failed/diverged need `notes`), frozen parameter violations, OOM batch size violations. Placeholder writes (`running`/`pending`) are exempt. (3) **SubagentStop** hook blocks agents from finishing if any required output file is missing (result JSON, training log, script dir, artifacts dir, etc.). Contracts are defined once in `scripts/output_contract.py` and shared by both SubagentStart (injection) and SubagentStop (verification). Contracts support two advanced fields: (a) `any_of` for mode-dependent outputs — e.g., analysis-agent must produce EITHER `reports/batch-<N>-analysis.md` (batch mode) OR `reports/session-review.md` (Phase 9 review mode), and at least one is required; (b) `required_if` for conditional outputs driven by the contents of another output — e.g., prerequisites-agent must produce `prepared-data/` ONLY when `dataset.prepared == true` in `prerequisites.json`. The condition is evaluated at SubagentStop by reading the referenced file and navigating a dotted jsonpath; missing or malformed reference files skip the conditional gracefully (they're caught by the unconditional entry instead). Agents not in the contract (monitor) are auto-approved.
