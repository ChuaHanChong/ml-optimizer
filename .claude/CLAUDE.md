# CLAUDE.md

Architecture details and operational rules for agents. For installation, setup, dashboard, tests, and general usage — see README.md.

Entry point: `/optimize <model-path>` → `commands/optimize.md` → `ml-optimizer:orchestrate` skill.

## MCP Server Dependencies (Recommended)

These are installed separately — not bundled with the plugin.

| MCP Server | Purpose | Used by | Required? |
|------------|---------|---------|-----------|
| **alphaxiv** | arXiv paper search, paper content extraction, PDF Q&A, GitHub repo exploration | research-agent (6 tools), implement-agent (2 tools) | No — falls back to WebSearch/WebFetch |
| **context7** | Framework API documentation lookup (PyTorch, TensorFlow, etc.) | research-agent, implement-agent | No — falls back to WebSearch |
| **claude-mem** | Cross-session memory — recalls past optimization sessions, avoids re-proposing failed techniques | research-agent, orchestrator | No — works without but loses cross-session learning |

Install: `claude mcp add --transport http --scope user alphaxiv https://api.alphaxiv.org/mcp/v1`. For context7 and claude-mem, install from the Claude Code marketplace.

The plugin works without any MCP servers but benefits significantly from alphaxiv (better paper discovery and analysis) and claude-mem (learning across optimization sessions).

## Architecture

### Plugin Structure

```
.claude-plugin/plugin.json      — Plugin metadata (name, version)
commands/optimize.md             — /optimize slash command (entry point)
skills/                          — Skill definitions (SKILL.md files)
skills/evolve/ShinkaEvolve/      — Git submodule (SakanaAI/ShinkaEvolve) for evolutionary code mutation
skills/hyperagent/Hyperagents/   — Git submodule (facebookresearch/Hyperagents) for evolutionary code search
skills/hyperagent-*/             — Symlinks → hyperagent/Hyperagents/skills/hyperagent-* (created by setup_hyperagent.sh)
skills/shinka-*/                 — Symlinks → evolve/ShinkaEvolve/skills/shinka-* (created by setup_evolve.sh)
agents/                          — 11 agent definitions (10 subagents + orchestrator-agent main-thread)
scripts/                         — Python utilities (stdlib only)
tests/                           — pytest test suite
```

### Skill Pipeline (Orchestrator Flow)

The `orchestrate` skill coordinates a 10-phase pipeline. Each phase dispatches a named agent via `Agent(subagent_type="ml-optimizer:<name>-agent")`. Persistent agents (research, implement, tuning, analysis, monitor) are resumed via `SendMessage(to: agentId)` for subsequent dispatches; ephemeral agents (prerequisites, baseline, experiment, report) get fresh spawns:

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
Phase 8: Method stacking (hyperagent driven, when analysis advises):
         Sequential accumulation — merge best methods one by one
         LLM conflict resolution, skip-on-failure
         Per step: analyze → tuning (evolve HPs) → evolve → tuning (training HPs) → experiment
         Analysis agent loops until improvement or recommends stop
Phase 9: report → Final optimization report
         review → Session review (what worked, what didn't, how to improve)
         promotion → Meta-patch promotion (if hyperagent generated skill patches)
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
  ├─ Phase 7: Agent(hyperagent-agent)             [persistent → SendMessage]
  │   │  Hyperagent DECIDES action, orchestrator DISPATCHES workers:
  │   │
  │   ├─ hp_tune: Hyperagent returns → Orchestrator dispatches:
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
  │   ├─ research_implement: Hyperagent returns → Orchestrator dispatches:
  │   │   ├─ research-agent (SendMessage)         → reports/research-findings-method-proposals*.md
  │   │   └─ implement-agent (SendMessage)        → results/implementation-manifest.json + branches
  │   │
  │   ├─ llm_patch / shinka_evolve: Hyperagent executes directly
  │   │   → Skill(hyperagent-generate) → git branch ml-opt/gen-N-<slug>
  │   │   → then orchestrator dispatches tuning + experiments on new branch
  │   │
  │   └─ meta_improve: Hyperagent executes directly
  │       → Skill(hyperagent-generate) → meta-patches/<skill>-SKILL.md
  │       → meta-patches/meta-changelog.json
  │
  ├─ Phase 8: Orchestrator resumes hyperagent → drives stacking loop
  │   └─ Per stack step: implement(merge) → experiment → analysis → [evolve] → [hp-tune]
  │       → results/round-N-stacked/exp-*.json, git branches ml-opt/stack-<N>
  │
  └─ Phase 9: Agent(report-agent) [ephemeral] + analysis-agent (review mode)
      → reports/final-report.md, reports/progress_chart.png
      → reports/session-review.md, reports/dashboard.html, results-table.md

Cross-cutting outputs (managed by scripts or multiple agents):
  round_manager.py       → results/rounds-manifest.json (round lifecycle)
  hyperagent-archive     → hyperagent/archive.jsonl, hyperagent/gen_X/ (evolutionary lineage)
  error_tracker.py       → reports/error-log.json (error tracking)
  pipeline_state.py      → pipeline-state.json (phase, iteration, agent_registry)
  goal_memory.py         → optimization-goals.json, learned-behaviors.json (goal anchoring)
  excalidraw_gen.py      → artifacts/*.excalidraw (on-demand diagrams)
  Multiple agents        → dev_notes.md (running session log, appended by many agents)
```

#### Directory Structure

The plugin creates an `experiments/` directory (location configured at Phase 0, can be anywhere):

```
<exp_root>/
├── artifacts/
│   ├── round-N-<type>/                   — Per-round artifact grouping
│   │   └── <exp-id>/                     — Checkpoints, visualizations
│   └── *.excalidraw                      — Excalidraw diagrams (on-demand)
├── hyperagent/
│   ├── archive.jsonl                     — Evolutionary archive (lineage + fitness)
│   └── gen_X/                            — Per-generation metadata + eval reports
├── logs/
│   ├── baseline/train.log                — Baseline training log
│   └── round-N-<type>/                   — Per-round log grouping
│       └── <exp-id>/train.log            — Training log (eval.log if separate eval)
├── meta-patches/
│   ├── <skill>-SKILL.md                  — Session-scoped skill modifications
│   └── meta-changelog.json               — Changelog of meta-improvements
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

Eleven agent types total — ten subagent types plus one main-thread agent — each with a preloaded skill and specified tool access. `orchestrate` is the user-facing entry point (invoked via `/optimize`). Other skills are preloaded into agents via the `skills:` array in their agent definitions. The `orchestrator-agent` is the main-thread agent activated by `settings.json` — it loads the orchestrate skill and auto-starts Phase 0 via `initialPrompt`. All agents have `memory: local` for persistent role-specific memory at `.claude/agent-memory-local/<agent-name>/` in the target project.

**Persistent agents** — dispatched once via `Agent(subagent_type=...)`, resumed via `SendMessage(to: agentId)` for subsequent tasks. The orchestrator tracks agent IDs in `pipeline-state.json` under `agent_registry` and falls back to fresh dispatch if resumption fails.

**Ephemeral agents** — fresh `Agent()` spawn each time (single-use or parallel tasks).

**Procedural agents** (`model: sonnet` — lower cost/latency, no ultrathink):
- **baseline-agent** *(ephemeral)*: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:baseline]`
- **monitor-agent** *(persistent)*: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:monitor]`
- **experiment-agent** *(ephemeral)*: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:experiment]`
- **prerequisites-agent** *(ephemeral)*: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:prerequisites]`

**Analytical agents** (`model: opus`, ultrathink prompting):
- **research-agent** *(persistent)*: WebSearch, WebFetch, Read, Write, Bash, Glob, Grep, Skill, alphaxiv MCP tools (6) — skills: `[ml-optimizer:research, claude-mem:mem-search]`
- **tuning-agent** *(persistent)*: Read, Write, Bash, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:hp-tune, claude-mem:mem-search]`
- **implement-agent** *(persistent)*: Bash, Read, Write, Edit, Glob, Grep, Skill, WebSearch, WebFetch, alphaxiv MCP tools (2: repo reader, PDF Q&A) — skills: `[ml-optimizer:implement, ml-optimizer:evolve, ml-optimizer:shinka-setup, ml-optimizer:shinka-convert, ml-optimizer:shinka-run, ml-optimizer:shinka-inspect, superpowers:systematic-debugging, feature-dev:code-explorer, feature-dev:code-reviewer]`
- **analysis-agent** *(persistent)*: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:analyze, claude-mem:mem-search]` (includes session review mode)
- **hyperagent-agent** *(persistent)*: Read, Write, Edit, Bash, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:hyperagent, ml-optimizer:hyperagent-generate, ml-optimizer:hyperagent-select, ml-optimizer:hyperagent-eval, ml-optimizer:hyperagent-archive, ml-optimizer:hyperagent-init, ml-optimizer:hyperagent-inspect, ml-optimizer:evolve, ml-optimizer:shinka-*, claude-mem:mem-search, superpowers:systematic-debugging, feature-dev:code-explorer]` — enables self-improvement, drives Phase 7 experiments and Phase 8 stacking in a loop
- **report-agent** *(ephemeral)*: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:report]`

**Main-thread agent** (activated by `settings.json`):
- **orchestrator-agent**: Agent, Read, Write, Edit, Bash, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:orchestrate]` — main thread when plugin is enabled, auto-starts Phase 0 via `initialPrompt: "/ml-optimizer:orchestrate"`

For parallel execution, use `run_in_background: true`. External skills are also available:
- **research-agent**: Uses `context7` for framework API docs, `claude-mem:mem-search` for cross-session learning, `alphaxiv` MCP for academic paper search/analysis (6 tools: embedding search, full-text search, agentic retrieval, paper content, PDF Q&A, GitHub repo reader)
- **implement-agent**: Uses `context7` for API docs, `feature-dev:code-explorer` for codebase analysis, `feature-dev:code-reviewer` for post-implementation quality review (advisory, Step 4e.5), `superpowers:systematic-debugging` for error recovery, `alphaxiv` MCP for reference repo exploration (`read_files_from_github_repository`) and paper clarification (`answer_pdf_queries`)
- **orchestrator**: Uses `claude-mem:mem-search` in Phase 1 for cross-session recall, `superpowers:brainstorming` in Phase 0/4 for complex multi-objective optimization scenarios

### Hyperagent Architecture (Self-Referential Evolutionary Optimization)

The plugin IS a **self-referential, self-improving hyperagent** — always on by default, not an optional mode. Powered by Facebook Research's Hyperagents (DGM framework, arXiv 2603.19461). The hyperagent enables self-improvement and drives Phase 7 (experiments) and Phase 8 (method stacking) in a loop, deciding at each iteration what action to take:

- **HP tuning** — delegates to tuning-agent (cheapest, best for early exploration)
- **LLM patch** — hyperagent directly modifies code (structural/architectural changes)
- **ShinkaEvolve** — dispatches evolve skill for fine-grained code mutation
- **Research-implement** — dispatches research + implement agents for paper-informed changes
- **Meta-improvement** — modifies the plugin's own skill instructions (session-scoped, max 3 per session, end-of-session promotion gate)

The hyperagent learns which operators are effective and adapts its strategy. The archive tracks all code variants with lineage. The analyze skill advises, the hyperagent decides.

**Submodule:** `skills/hyperagent/Hyperagents/` (CC BY-NC-SA 4.0 license). Skills at `skills/hyperagent/skills/hyperagent-*/`, symlinked to `skills/hyperagent-*/` for auto-discovery. Setup: `bash scripts/setup_hyperagent.sh` (inits submodule + creates symlinks, same pattern as `setup_evolve.sh`).

**6 Hyperagent skills:**
- `hyperagent-init` — Create archive from baseline + existing branches
- `hyperagent-inspect` — Inspect archive state, lineage, operator stats, and generation history
- `hyperagent-select` — Parent selection (6 strategies: best, latest, random, score_prop, score_child_prop, ucb). Uses Hyperagents' exact math: `sigmoid(10(s - μ)) × exp(-(children/8)³)`. UCB1 (Auer et al. 2002) adds balanced explore/exploit via `value/visits + C×sqrt(ln(N)/visits)` with MCTS-style backpropagation.
- `hyperagent-generate` — Hyperagent generates code variant (replaces Hyperagents' litellm hyperagent with Claude Code Opus agent). Can dispatch ShinkaEvolve as mutation operator.
- `hyperagent-eval` — Two-stage evaluation: cheap staged eval (10% budget) → adaptive threshold → full training if passes. Warm-starts from staged checkpoint.
- `hyperagent-archive` — Update archive with results, track lineage and operator effectiveness

**Archive:** `experiments/hyperagent/archive.jsonl` — Hyperagents-native JSONL format with `gen_X/` directories for metadata and eval reports. Managed by `gl_utils.py` directly.

**ShinkaEvolve + Hyperagent collaboration:** ShinkaEvolve is one mutation operator within the experiment loop. When the hyperagent needs fine-grained code tuning (numerical constants, local optimizations), it dispatches ShinkaEvolve via `Skill("ml-optimizer:evolve")`. When it needs structural/architectural changes, it generates LLM patches directly.

**Pipeline state:** `hyperagent_state` in `pipeline-state.json` tracks: `enabled` (always true), `archive_generation`, `strategy_history` (log of hyperagent decisions), `meta_improvement_count`, `active_meta_patches`, `operator_stats`.

**Simplified analyze pivots:** All code-level pivots (research, method proposals, code refinement) emit `code_evolution`. HP-focused pivots (`branch_test`, `hp_expand`, `narrow_space`, `regularization`) remain separate. When all approaches stall, analyze includes `meta_improvement_recommended: true` so the hyperagent considers self-improvement.

**Meta-improvement patches:** Session-scoped skill modifications at `experiments/meta-patches/`. End-of-session: analysis-agent evaluates patches, presents validated ones to user for promotion to the plugin branch.

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
| `${CLAUDE_PLUGIN_ROOT}/scripts/pipeline_state.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/pipeline_state.py <exp_root> validate\|save\|load\|cleanup\|verify-baseline\|gate\|log-gate\|log-decision\|replay-check\|decisions\|meta-patch` — phase gates, decision logging, meta-patch lifecycle |
| `${CLAUDE_PLUGIN_ROOT}/scripts/schema_validator.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/schema_validator.py <filepath> result\|baseline\|manifest\|prerequisites [--strict]` — validates JSON schemas. `--strict` enforces completeness. Also: `relay <route> <json>` for inter-agent relay validation (7 routes) |
| `${CLAUDE_PLUGIN_ROOT}/scripts/plot_results.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/plot_results.py <results_dir> <metric> comparison\|timeline\|sensitivity <hp>\|progress [--higher-is-better]` — ASCII charts + matplotlib progress chart |
| `${CLAUDE_PLUGIN_ROOT}/scripts/prerequisites_check.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/prerequisites_check.py scan-imports\|check-packages\|detect-env\|detect-format\|detect-format-project\|validate-data\|bulk-install-cmd\|gpu-install-cmd` — dataset, environment, and GPU-aware install validation |
| `${CLAUDE_PLUGIN_ROOT}/scripts/dashboard.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/dashboard.py <exp_root> [--live] [--table] [--serve --port 8080]` — generate self-contained HTML dashboard with progress timeline, results table, HP sensitivity, research agenda, error summary, method explanations. `--live` enables 30s auto-refresh. `--table` generates `results-table.md` (Markdown results summary). |
| `${CLAUDE_PLUGIN_ROOT}/scripts/excalidraw_gen.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/excalidraw_gen.py <exp_root> pipeline\|comparison\|hp-landscape\|architecture <args>` — generate Excalidraw JSON diagrams (pipeline overview, experiment comparison, HP landscape, architecture changes) |
| `${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log\|show\|patterns\|summary\|success\|proposals\|rank\|log-suggestion\|suggestion-history\|dead-end <add\|list\|check>\|agenda <init\|update\|list\|add>` — error tracking, pattern detection, success metrics, proposal outcomes, suggestion ranking, suggestion history, dead-end catalog, research agenda |
| `${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> init-goals\|read-goals\|log-behavior <category> <json>\|query-behaviors [category]\|validate-output <agent> <json>\|summary\|sync-from-errors` — goal anchoring, behavioral memory, agent output validation, compact briefings |
| `${CLAUDE_PLUGIN_ROOT}/scripts/setup_evolve.sh` | `bash scripts/setup_evolve.sh` — initialize ShinkaEvolve submodule and create skill symlinks for auto-discovery |
| `${CLAUDE_PLUGIN_ROOT}/scripts/setup_hyperagent.sh` | `bash scripts/setup_hyperagent.sh` — initialize Hyperagents submodule and verify environment |
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

The orchestrator can be stopped and resumed. On restart it reads `pipeline-state.json` and uses `cleanup_stale()` to handle interrupted experiments (marks them as failed after a timeout). Phase validation via `validate_phase_requirements()` prevents cascading failures. Pipeline state persists Phase 0 user choices (`primary_metric`, `divergence_metric`, `divergence_lower_is_better`, `lower_is_better`, `target_value`, `train_command`, `eval_command`, `train_data_path`, `val_data_path`, `prepared_train_path`, `prepared_val_path`, `env_manager`, `env_name`, `model_category`, `user_papers`, `method_proposal_scope`, `method_proposal_iterations`, `hp_batches_per_round`, `fixed_time_budget`, `fixed_epoch_budget`) via `save_state(user_choices={...})` so they survive interruptions without re-asking the user. The experiment loop also persists `consecutive_stop_count` (for the 3-consecutive-stop exit rule), `stuck_protocol_triggered` (prevents infinite recovery loops), `baseline_checksum` (SHA-256 of baseline metrics for integrity verification), and `agent_registry` (persistent agent IDs for SendMessage resumption) at the root level of pipeline state. On new session start, `agent_registry` is cleared since subagent transcripts are session-scoped — all agents start fresh. A separate `user-choices-backup.json` provides redundant recovery if the main state file corrupts.

## Key Design Patterns

- **Resumable subagents**: 6 persistent agents (research, implement, tuning, analysis, monitor, hyperagent) are dispatched once via `Agent()` and resumed via `SendMessage(to: agentId)` for subsequent tasks. This preserves accumulated context (search results, HP trends, codebase knowledge) across the pipeline. 4 ephemeral agents (prerequisites, baseline, experiment, report) get fresh spawns. The hyperagent joins the persistent set — it accumulates understanding of which mutation operators are effective across generations. The orchestrator tracks agent IDs in `pipeline-state.json` under `agent_registry` and falls back to fresh dispatch if resumption fails. Agent IDs are session-scoped — cleared on new session start since subagent transcripts don't survive across sessions.
- **Inter-agent communication (orchestrator relay)**: When resuming a persistent agent, the orchestrator includes a `CONTEXT FROM OTHER AGENTS:` section with relevant findings from other agents. This enables indirect communication: analyze findings reach hp-tune, monitor OOM info reaches hp-tune, research proposals reach implement, etc. The orchestrator acts as a message bus. Key relay routes: analyze→tuning (correlations, branch scores), analyze→research (pivot reasons, dead-ends), analyze→hyperagent (pivot_type, stacking recommendation, meta_improvement_recommended), monitor→tuning (OOM constraints), research→implement (proposals), experiments→analyze (batch results), hyperagent→tuning (action, evolve HPs, target branch).
- **Non-git fallback**: If the target project isn't a git repo, the implement skill uses file backups instead of branches. Each proposal is validated against a clean baseline backup (restore-before-apply pattern) to prevent cross-proposal code leakage. This forces sequential (not parallel) experiment execution.
- **Loop exit conditions**: The experiment loop is autonomous — runs until: (1) target metric achieved, or (2) user manually stops. When analysis advises stop, the hyperagent tries other operators before giving up. Phase 7 ↔ Phase 8 loop continuously.
- **Proposal priority scoring**: `(impact * confidence) / (11 - min(feasibility, 10))` — feasibility clamped to [1,10] to prevent division by zero.
- **Spearman correlation**: `${CLAUDE_PLUGIN_ROOT}/scripts/result_analyzer.py` uses rank correlation with average-rank tie-breaking to identify HP-metric relationships (no scipy dependency).
- **Dual implementation strategy**: Research proposals include an `implementation_strategy` field (`from_scratch` or `from_reference`). The implement agent dispatches accordingly — either implementing from paper descriptions (Section 8) or cloning and adapting reference repos (Section 9). Strategy is decided by the research agent based on repo availability and quality.
- **Research skill modes**: The research skill accepts `source` (`"web"` | `"knowledge"` | `"both"`), `scope_level` (`"training"` | `"architecture"` | `"full"`), and `output_path` parameters. Knowledge mode skips web search and uses LLM training knowledge only.
- **Scope-gated pivots**: The analysis agent's pivot decision tree respects `scope_level`: `"training"` (HP-only) disables research and code_evolution pivots — only HP adjustments are available. `"architecture"` enables research but not ShinkaEvolve. `"full"` enables everything including code_evolution via ShinkaEvolve. This ensures the pipeline matches what the user asked for.
- **Auto-resolution behaviors**: The pipeline auto-resolves many situations: Phase 2 partial prereqs → proceed with warnings, RL polarity → auto-infer from metric name, dirty working trees → auto-stash, environment mismatches → use detected manager, missing conda envs → auto-create, no eval command → fall back to training output metrics. Only unrecoverable errors (Phase 2 failed, baseline failed) block the pipeline. Decisions are logged to dev_notes and error tracker for post-session review.
- **Parallel research**: All WebSearch calls in the research skill are issued simultaneously in a single tool-call message, alongside 3 alphaxiv search calls (embedding similarity, full-text keyword, agentic retrieval). WebFetch follow-ups for different URLs are also parallelized. Domain-specific query sets (NLP, CV, RL, time-series) are issued alongside generic queries. If alphaxiv MCP is unavailable, WebSearch provides full coverage as fallback.
- **Parallel implementation**: When using git branch strategy with multiple proposals, each proposal is implemented in a separate git worktree via parallel Agent dispatches. File-backup strategy remains sequential.
- **Configurable divergence thresholds**: `${CLAUDE_PLUGIN_ROOT}/scripts/detect_divergence.py` supports per-model-category threshold overrides via `MODEL_CATEGORY_DEFAULTS` dict and `--model-category` CLI flag. RL models use `explosion_threshold=20.0, plateau_patience=50` (prevents false positives on reward spikes). Generative models use `explosion_threshold=10.0, plateau_patience=40` (accommodates slow convergence). Individual thresholds can also be overridden via `--explosion-threshold` and `--plateau-patience` CLI flags.
- **Experiment timeout**: Each experiment has a hard timeout of `baseline_training_time * 3` (fallback: 6 hours). Timed-out experiments are killed and marked `status: "timeout"`.
- **Research failure recovery**: If web search fails (both WebSearch and alphaxiv), the orchestrator retries with `source: "knowledge"` (LLM-only). If that also fails, it continues with HP-only optimization. Each fallback is logged. Within a search, alphaxiv failure alone does not trigger the knowledge fallback — WebSearch results are sufficient to proceed.
- **alphaxiv MCP integration**: The research agent uses all 6 alphaxiv MCP tools for academic paper discovery (3 search tools run in parallel), paper content extraction (`get_paper_content`, `answer_pdf_queries`), and reference repo exploration (`read_files_from_github_repository`). The implement agent uses 2 alphaxiv tools (`read_files_from_github_repository` for pre-clone repo assessment, `answer_pdf_queries` for clarifying ambiguous implementation steps from source papers). All alphaxiv searches run in parallel with WebSearch. alphaxiv is optional — if the MCP server is unavailable, all workflows fall back to WebSearch/WebFetch transparently.
- **OOM feedback loop**: When experiments OOM, the batch size is recorded in the error tracker. On the next hp-tune invocation, `max_batch_size` is passed to prevent re-proposing configs that will OOM.
- **All-diverge recovery**: If all experiments in a batch diverge, a recovery batch with halved learning rates is attempted before stopping.
- **HP-only research routing**: Research proposals with `type: "hp_only"` skip the implement skill and are routed directly to hp-tune as search space modifications.
- **Tabular ML HP strategy**: For tree-based models (sklearn/XGBoost/LightGBM), iteration 1 explores `max_depth`/`n_estimators` first instead of learning rate.
- **Training budget options**: Phase 0 offers two budget modes: `fixed_time_budget` (seconds) for wall-clock-normalized comparison, or `fixed_epoch_budget` (integer) for deterministic reproducibility. When set, both baseline AND experiments use the same budget. The baseline skill (Step 2.2) wraps training with `timeout` when `fixed_time_budget` is set, ensuring fair baseline-vs-experiment comparisons. Framework-native time limits (Lightning `--max_time`, HuggingFace `timeout` in TrainingArguments) are preferred when available. Results include `time_budget_seconds` for downstream analysis. HP-tune adjusts proposals for the budget (shorter convergence schedules, appropriate LR scaling). Makes experiment metrics directly comparable without duration normalization.
- **Small dataset awareness**: The research skill checks dataset size. For datasets under 5K samples, it shifts search toward low-data techniques (transfer learning, fine-tuning, few-shot learning, adapters, prompt tuning, synthetic data, semi-supervised methods) instead of heavy augmentation and regularization which underperform on small data.
- **Evolutionary code refinement**: When HP tuning shows diminishing returns, the analyze skill can recommend `pivot_type: "code_evolution"`. The orchestrator dispatches the implement-agent with the evolve skill (`Skill("ml-optimizer:evolve")`), which orchestrates the full ShinkaEvolve pipeline internally: `shinka-convert` (create task from best branch) → `shinka-run` (run evolution with file-based handoff, `SHINKA_PROVIDER=claude_code`) → `shinka-inspect` (extract best mutation) → commit as `ml-opt/evolved-<slug>`. Evolve HPs (`num_generations`, `population_size`) are tuning-agent-driven: the orchestrator dispatches the tuning agent to propose evolve HPs before dispatching the implement agent, based on prior evolution outcomes stored in `learned-behaviors.json` under category `evolve_hp`. Defaults: 10 generations, population 2. If ShinkaEvolve is unavailable, the evolve skill reports `shinkaevolve_unavailable` and the orchestrator falls back to the research → implement path. Setup: run `bash scripts/setup_evolve.sh` to init the submodule and create symlinks (`skills/shinka-*` → `skills/evolve/ShinkaEvolve/skills/shinka-*`). The symlinks are required for Claude Code's skill auto-discovery.
- **Auto-repair loop**: When training or evaluation commands fail during baseline establishment or experiment execution, the agent captures stderr, diagnoses the error, applies a fix (install package, adjust path, reduce batch size), and retries up to 3 times. OOM errors are not retried (deterministic). SyntaxErrors are not retried (code bugs). Identical errors on consecutive attempts skip further retries (loop detection). Each retry is logged to the error tracker. This is intra-agent retry, separate from the orchestrator's Phase 3 retry logic.
- **Goal anchoring & behavioral memory**: `${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py` maintains two project-scoped files: `optimization-goals.json` (goal anchor, written once at Phase 0) and `learned-behaviors.json` (accumulated behavioral memory). The orchestrator calls `validate-output` after hp-tune, research, and analyze dispatches to catch drift (frozen param changes, scope breaches, dead-end re-proposals, metric mismatches). Each agent also reads a compact `summary` combining goals + constraints + dead-ends before acting. All 11 agents have `memory: local` for persistent role-specific memory at `.claude/agent-memory-local/<agent-name>/` in the target project.
- **Immutable baseline**: After baseline is established (Phase 3), a SHA-256 checksum of the baseline metrics dict is stored in `pipeline-state.json`. Before each experiment batch (Phase 7) and on pipeline resumption, the checksum is verified against `baseline.json`. If the metrics have changed (accidental modification, file corruption, or tampering), the pipeline halts with a critical error. Prevents invalid experiment comparisons during long optimization runs.
- **Stuck protocol**: When analysis advises stop, the hyperagent tries other operators (research, LLM patches, ShinkaEvolve, meta-improvement) before giving up. It reads error patterns, dead ends, and the research agenda to inform the choice. The loop is autonomous — only the user or target achievement stops it.
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
- **`${CLAUDE_PLUGIN_ROOT}/scripts/implement_utils.py` has four CLI modes**: default (parse proposals), `clone <url> <dest>`, `analyze <path>`, and `diff <project_root> <branch>`. Each has different argument patterns.
- **Metric routing is split**: Monitor/divergence always uses loss (lower-is-better). Analyze/hp-tune use the user's `primary_metric`. Mixing these up causes silent wrong behavior.
- **Branch experiments are independent**: Results on `ml-opt/branch-a` tell you nothing about what HPs will work on `ml-opt/branch-b`. The tuning agent must group by `code_branch` before analyzing trends.
- **`agent_registry` is session-scoped**: Agent IDs in `pipeline-state.json` under `agent_registry` are only valid within the same Claude conversation session. On pipeline resumption in a new session, the registry must be cleared (`agent_registry = {}`) because subagent transcripts don't survive across sessions. The orchestrator clears it automatically on load. Don't rely on agent_registry for cross-session state — use `memory: local` and shared files for that.
- **Experiment results MUST live in round directories**: `exp-*.json` files must be written to `results/round-N-<type>/exp-*.json`, NEVER directly to `results/exp-*.json`. The PreToolUse hook (`validate_experiment_write.py`) blocks any Write/Edit that violates this pattern. Before dispatching experiment-agents, the orchestrator must call `round_manager.py create-round <type>` and pass the current round directory to the agent. Valid round types: `hp`, `evolved`, `research`, `stacked`, `meta`.
- **Proposed configs are top-level with round structure**: HP tuning proposals go in `proposed-configs/round-N-<type>/exp-*.json` (top-level, NOT under `results/`). The PreToolUse hook validates this. The round directory must exist (created by `round_manager.py create-round`) before the tuning agent writes proposals.
- **Exp-ids are globally unique across rounds**: `exp-001` only exists in one round. `round_manager.py next-id` scans all round directories to find the next available ID. Don't try to reuse exp-ids across rounds — results from earlier rounds are preserved and still counted.
- **`round_manager.py` uses `fcntl.flock` for manifest writes**: `rounds-manifest.json` is protected by a file lock for concurrent safety. Manual edits to the manifest may be lost if agents are running. Always use `round_manager.py` CLI commands (`create-round`, `register-experiment`, `close-round`) to update it.
- **Tabular ML frameworks skip divergence monitoring**: When the detected framework is scikit-learn, XGBoost, or LightGBM, the orchestrator sets `divergence_metric` to `null` and skips the monitor skill. The baseline skill skips GPU profiling and throughput estimation for these frameworks.
- **Research findings files can be multiple**: `research-findings.md` (Phase 5 web search), `research-findings-method-proposals.md` (Phase 7 pre-loop), `research-findings-method-proposals-iter<N>.md` (Phase 7 mid-loop triggers). The research skill's deduplication checks all of these to avoid re-proposing tried techniques.
- **Hyperagent mode is always on — NEVER skip it**: `hyperagent_state.enabled` defaults to `true`. The hyperagent MUST be dispatched in Phase 7 — it is the loop driver, not optional. Do NOT fall back to a simpler HP-tune → experiment → analyze loop that bypasses the hyperagent. The hyperagent drives Phase 7 ↔ Phase 8 in a loop from the start. It naturally starts with HP tuning (cheapest) and escalates to code mutations (LLM patches, ShinkaEvolve), research-implement, stacking, and self-improvement as needed. If ShinkaEvolve is unavailable, the hyperagent falls back to other operators — but the hyperagent itself is never optional.
- **Hyperagent archive is separate from experiment results**: `hyperagent/archive.jsonl` tracks code variants with lineage (Hyperagents-native format). `exp-*.json` tracks experiment results. They link via the `genid` field. Don't confuse them.
- **Meta-patches are session-scoped**: Files in `experiments/meta-patches/` only affect the current session. They're instructions overlaid on top of default skills. Promotion to the plugin repo requires user approval at Phase 9.
- **Hyperagents submodule is CC BY-NC-SA 4.0**: NonCommercial + ShareAlike license. The adapter script reimplements the core algorithms in stdlib Python, so the submodule is a reference, not a runtime dependency.
- **ShinkaEvolve branch naming in Hyperagent mode**: When ShinkaEvolve is used as a mutation operator within the Hyperagent loop, the evolve skill creates `ml-opt/evolved-<slug>` branches. The hyperagent must rename these to `ml-opt/gen-<N>-evolved-<slug>` for archive consistency.
- **ShinkaEvolve must use the local submodule, not PyPI**: The PyPI package `shinka-evolve` lacks the `file_handoff_provider` module required for `SHINKA_PROVIDER=claude_code`. Always use the local submodule — either install editable (`pip install -e skills/evolve/ShinkaEvolve/`) via `setup_evolve.sh`, or prepend `PYTHONPATH=${CLAUDE_PLUGIN_ROOT}/skills/evolve/ShinkaEvolve:$PYTHONPATH` before `shinka_run`.
- **ShinkaEvolve subprocess Python resolution**: `shinka/launch/scheduler.py` falls back to `sys.executable` (the Python running ShinkaEvolve) when no conda env or activate script is configured. This usually resolves correctly. If evaluation subprocesses fail with "python not found", set `python_executable` in `LocalJobConfig` or ensure `python` is on PATH.
- **ShinkaEvolve file handoff timeout is configurable via `SHINKA_HANDOFF_TIMEOUT`**: Default is 600s (10 minutes). Set before launching `shinka_run` with `SHINKA_PROVIDER=claude_code`. The agent must write a `<id>.inprogress` marker when it picks up a pending request — this resets `shinka_run`'s timeout from the acknowledgment point. Without the marker, the timeout runs from request creation. `shinka_run` also writes `<id>.heartbeat` files every 5s so the agent can verify liveness.
- **3-checkpoint output enforcement**: Every agent output is enforced at 3 points: (1) **SubagentStart** injects the output contract — exact paths, schemas, and examples — so agents know what to produce before they start (`scripts/output_contract.py` via `hooks/subagent-start-inject-goals.sh`). (2) **PreToolUse** hook blocks invalid JSON writes: wrong path, bad schema, missing completeness fields (completed experiments need `iteration`, `method_tier`, `duration_seconds`; stacked need `code_branches`, `stacking_order`; failed/diverged need `notes`), frozen parameter violations, OOM batch size violations. Placeholder writes (`running`/`pending`) are exempt. (3) **SubagentStop** hook blocks agents from finishing if any required output file is missing (result JSON, training log, script dir, artifacts dir, etc.). Contracts are defined once in `scripts/output_contract.py` and shared by both SubagentStart (injection) and SubagentStop (verification). Contracts support two advanced fields: (a) `any_of` for mode-dependent outputs — e.g., analysis-agent must produce EITHER `reports/batch-<N>-analysis.md` (batch mode) OR `reports/session-review.md` (Phase 9 review mode), and at least one is required; (b) `required_if` for conditional outputs driven by the contents of another output — e.g., prerequisites-agent must produce `prepared-data/` ONLY when `dataset.prepared == true` in `prerequisites.json`. The condition is evaluated at SubagentStop by reading the referenced file and navigating a dotted jsonpath; missing or malformed reference files skip the conditional gracefully (they're caught by the unconditional entry instead). Agents not in the contract (monitor, hyperagent) are auto-approved.
