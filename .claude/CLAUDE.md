# CLAUDE.md

Architecture details and operational rules for agents. For installation, setup, dashboard, tests, and general usage — see README.md.

Entry point: `/optimize <model-path>` → `commands/optimize.md` → `ml-optimizer:orchestrate` skill.

## MCP Server Dependencies

Installed separately, not bundled. **gitnexus is required**; alphaxiv, context7, claude-mem are recommended.

| MCP Server | Purpose | Used by | Required? |
|------------|---------|---------|-----------|
| **alphaxiv** | arXiv paper search, paper content extraction, PDF Q&A, GitHub repo exploration | research-agent (6 tools), implement-agent (2 tools) | No — falls back to WebSearch/WebFetch |
| **gitnexus** | Code knowledge graph — index a repo and query its structure, call-graph, and blast-radius (`mcp__gitnexus__context`, `mcp__gitnexus__query`, `mcp__gitnexus__impact`) | implement-agent (3 tools), research-agent (3 tools) | **Yes** — required; the pipeline indexes every code repo and agents must query it to understand code |
| **context7** | Framework API documentation lookup (PyTorch, TensorFlow, etc.) | research-agent, implement-agent | No — falls back to WebSearch |
| **claude-mem** | Cross-session memory — recalls past optimization sessions, avoids re-proposing failed techniques | research-agent, orchestrator | No — works without but loses cross-session learning |

Install: `claude mcp add --transport http --scope user alphaxiv https://api.alphaxiv.org/mcp/v1`. For gitnexus (required): `npm install -g gitnexus && gitnexus setup` — `gitnexus setup` auto-registers the gitnexus MCP server for Claude Code (manual MCP-registration fallback: `claude mcp add --transport stdio --scope user gitnexus gitnexus mcp`). Install context7 and claude-mem from the Claude Code marketplace.

**gitnexus is a hard prerequisite** — like git, required not optional. Phase 2 verifies install and blocks the pipeline if absent (see "GitNexus code-graph understanding" below). The plugin functions without the *optional* servers but benefits from alphaxiv (paper discovery/analysis) and claude-mem (cross-session learning).

## Required: Dynamic Workflows (Phases 5–8)

Phases 5–8 run as **dynamic workflows** (the `Workflow` tool), not turn-by-turn `Agent` dispatch. The orchestrator stays a live agent for the interactive/trivial phases (0/1 planning, 2 prerequisites, 3 baseline, 4 checkpoint, 9 report) and launches **one workflow per phase** for 5–8 via `Workflow({scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-N-<name>.js", args})`, reading each structured return and running the user checkpoint **between** phases. The four scripts are **bundled inside the orchestrate skill** at `skills/orchestrate/workflows/phase-{5,6,7,8}-*.js` — **internal pipeline steps, not user-facing /commands**. (Placing them under `.claude/workflows/` would expose them as user `/slash-commands`; launching by `scriptPath` keeps them out of the command namespace.) Each script keeps its `meta.name` (display only) and args/return contract unchanged.

Dynamic workflows are a **hard requirement** for phases 5–8 — **no `Agent` fallback path**. The workflow runtime's own `agent({agentType})` call is the dispatch mechanism, so the old `SendMessage` / persistent-agent / `agent_registry` / relay machinery is removed, not kept as a second path. Without dynamic workflows enabled on your plan, phases 5–8 cannot run (see README "Prerequisites").

## Architecture

### Plugin Structure

```
.claude-plugin/plugin.json      — Plugin metadata (name, version)
commands/optimize.md             — /optimize slash command (entry point)
skills/                          — Skill definitions (SKILL.md files)
skills/orchestrate/workflows/    — Phase 5–8 dynamic workflow scripts (phase-{5,6,7,8}-*.js), internal pipeline steps launched via Workflow({scriptPath, args}) — NOT user-facing /commands
skills/evolve/ShinkaEvolve/      — Git submodule (SakanaAI/ShinkaEvolve) for evolutionary code mutation
skills/shinka-*/                 — Symlinks → evolve/ShinkaEvolve/skills/shinka-* (created by setup_evolve.sh)
agents/                          — 10 agent definitions (9 subagents + orchestrator-agent main-thread)
scripts/                         — Python utilities (stdlib only)
tests/                           — pytest test suite
```

### Skill Pipeline (Orchestrator Flow)

The `orchestrate` skill coordinates a 10-phase pipeline. Phases 0/1, 2, 3, 4, 9 dispatch a named agent via `Agent(subagent_type="ml-optimizer:<name>-agent")` (interactive/trivial). Phases 5–8 run as **dynamic workflows** — one `Workflow({scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-N-<name>.js", args})` per phase; the script holds that phase's fan-out/loop and reuses the existing agent definitions via `agent(prompt, {agentType: "ml-optimizer:<name>-agent"})` (tools/skills/model intact). All spawns inside a workflow are fresh `agentType` dispatches — no `SendMessage`, no persistent-agent resumption, no `agent_registry`. Cross-agent context flows via `args` + files agents write under `<exp_root>/` (results, manifests, agenda, batch analyses) + each workflow's structured `schema:` return. The Phase 7 loop (experiments ↔ analyze decisions) lives inside `phase-7-experiment.js`; Phase 8 stacking inside `phase-8-stacking.js`:

```
Phase 0+1: Discovery & Planning (plan mode throughout — multi-round refinement until user approves)
         Discovery Q&A → write goals → analyze codebase → present plan → user refines → repeat
Phase 2: prerequisites → Validate dataset format, prepare data, install dependencies
Phase 3: baseline → Establish baseline metrics
Phase 4: User checkpoint (also pre-authorizes Phase 7 autonomy: method_proposal scope/iterations + budget)
Phase 5: Workflow(phase-5-research) → Find techniques via web/papers
Phase 6: Workflow(phase-6-implement) → Apply proposals as git branches
Phase 7: Workflow(phase-7-experiment) — Experiment loop (autonomous, pipelined):
         hp-tune → propose configs
         experiment → run training (parallel across GPUs)
         monitor → divergence polling folded into each experiment agent (standalone monitor-agent opt-in)
         analyze → decide continue/pivot/stop
         [method_proposal] → mid-loop research + implement
         [code_evolution] → tuning (evolve HPs) → evolve → tuning (training HPs) → experiment (scope_level=full only)
         [pivot gate] → before a costly method_proposal/code_evolution pivot, independent skeptics may refute it → downgrade to continue
         [research_round] → cadence-based research (when method proposals enabled)
Phase 8: Workflow(phase-8-stacking) — Method stacking (when analysis advises):
         Sequential accumulation — merge best methods one by one
         LLM conflict resolution + per-merge review (code-reviewer + silent-failure-hunter), skip on failure/critical
         Per step: analyze → tuning (evolve HPs) → evolve → tuning (training HPs) → experiment
         Analysis agent loops until improvement or recommends stop
Phase 9: Agent(report) → Final optimization report
         Agent(analysis, review mode) → Session review (what worked, what didn't, how to improve)
```

### Dispatch Chain & Output Map

```
/optimize → orchestrator-agent (main thread, settings.json)
  │  Phases 0/1, 2, 3, 4, 9 → direct Agent() (interactive/trivial).
  │  Phases 5, 6, 7, 8       → Workflow({scriptPath, args}) (dynamic workflows, bundled in skills/orchestrate/workflows/).
  │
  ├─ Phase 2: Agent(prerequisites-agent)
  │   → results/prerequisites.json, prepared-data/ (if data prep needed)
  │
  ├─ Phase 3: Agent(baseline-agent)
  │   → results/baseline.json, logs/baseline/train.log
  │
  ├─ Phase 5: Workflow(phase-5-research)
  │   │  Reuses research-agent via agentType (parallel angles → dedup → vet → synthesize).
  │   → reports/research-findings.md (+ research agenda init)
  │   return: { findings_path, proposals[], agenda_initialized }
  │
  ├─ Phase 6: Workflow(phase-6-implement)
  │   │  pipeline(): implement-agent (worktree, parallel) → reviewers in parallel.
  │   │  Reuses implement-agent + feature-dev:code-reviewer + pr-review-toolkit:silent-failure-hunter.
  │   → results/implementation-manifest.json, git branches ml-opt/<slug>
  │   return: { manifest_path, branches[] }
  │
  ├─ Phase 7: Workflow(phase-7-experiment)  [autonomous loop inside the script]
  │   │  The while-loop holds the round/decision state; each iteration dispatches
  │   │  fresh agents via agentType. Cross-agent context flows via files + args
  │   │  (no SendMessage, no relay). Decision tree: continue / branch_test /
  │   │  hp_expand / narrow / regularization / stop; stuck protocol = a research
  │   │  agent() call + fixpoint check.
  │   │
  │   ├─ hp_tune (default):
  │   │   ├─ tuning-agent (agentType)               → proposed-configs/round-N-<type>/exp-*.json (top-level dir, not under results/)
  │   │   ├─ experiment-agents (parallel, agentType) → results/round-N-<type>/exp-*.json
  │   │   │                                            logs/round-N-<type>/<exp-id>/train.log
  │   │   │                                            scripts/round-N-<type>/<exp-id>/train.sh
  │   │   │                                            artifacts/round-N-<type>/<exp-id>/
  │   │   ├─ monitor-agent (agentType, opt-in)       → (no file output; in-run divergence polling folded into experiment-agent)
  │   │   └─ analysis-agent (agentType)              → reports/batch-<N>-analysis.md (per-batch, required)
  │   │                                                reports/dead-ends.json, dead-ends.md
  │   │                                                reports/research-agenda.json, research-agenda.md
  │   │                                                reports/suggestion-history.json
  │   │                                                (decision returned to the workflow via schema)
  │   │
  │   ├─ code_evolution:
  │   │   ├─ tuning-agent (evolve HPs)           → evolve_recommendation
  │   │   └─ implement-agent (evolve skill)       → git branch ml-opt/evolved-<slug>
  │   │
  │   └─ method_proposal:
  │       ├─ research-agent (agentType)          → reports/research-findings-method-proposals*.md
  │       └─ implement-agent (agentType)         → results/implementation-manifest.json + branches
  │   return: { best_exp_id, best_metric, rounds_completed, exit_reason, stacking_candidates[] }
  │
  ├─ Phase 8: Workflow(phase-8-stacking)  [sequential accumulation loop inside the script]
  │   └─ Per stack step: implement(merge) → review → experiment → analysis → [evolve] → [hp-tune]
  │       → results/round-N-stacked/exp-*.json, git branches ml-opt/stack-<N>
  │   return: { best_stack_branch, best_stack_metric, steps[] }
  │
  └─ Phase 9: Agent(report-agent) + Agent(analysis-agent, review mode)
      → reports/final-report.md, reports/progress_chart.png
      → reports/session-review.md, reports/dashboard.html, results-table.md

Cross-cutting outputs (managed by scripts or multiple agents):
  round_manager.py       → results/rounds-manifest.json (round lifecycle)
  error_tracker.py       → reports/error-log.json (error tracking)
  pipeline_state.py      → pipeline-state.json (phase, iteration, user_choices, stacking state)
  goal_memory.py         → optimization-goals.json, learned-behaviors.json (goal anchoring)
  excalidraw_gen.py      → artifacts/*.excalidraw (on-demand diagrams)
  Multiple agents        → dev_notes.md (running session log, appended by many agents)
```

#### Directory Structure

The plugin creates an `<exp_root>/` directory — user chooses name and location at Phase 0, no hardcoded default. Layout inside `<exp_root>/`:

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

Monitor the user's `divergence_metric` for divergence detection (monitor skill) — default `"loss"` (lower-is-better); reward-like metrics allowed with `divergence_lower_is_better=false`. Use `primary_metric` (accuracy, PSNR, F1, etc.) for analyze and hp-tune.

### Branch-Aware Experiments

The implement skill creates `ml-opt/<slug>` branches per proposal; the experiment loop tests each with HP variations. The tuning agent groups results by `code_branch` — identical HPs on different branches are independent experiments.

### Agent Definitions (`agents/`)

Ten agent types — nine subagents plus one main-thread agent — each with a preloaded skill and specified tool access. `orchestrate` is the user-facing entry point (via `/optimize`); other skills preload into agents via the `skills:` array in their definitions. The `orchestrator-agent`, activated by `settings.json`, loads the orchestrate skill and auto-starts Phase 0 via `initialPrompt`. All agents have `memory: local` for persistent role-specific memory at `.claude/agent-memory-local/<agent-name>/` in the target project.

All nine worker agents dispatch as **fresh spawns** — `Agent()` for phases 2/3/9, the workflow runtime's `agent({agentType: "ml-optimizer:<name>-agent"})` for 5–8. No persistent-agent / `SendMessage` / `agent_registry` resumption. Cross-agent context flows via `args`, files under `<exp_root>/`, and each workflow's structured return. Role knowledge persists via `memory: local` (see below).

**Procedural agents** (`model: sonnet[1m]`, `effort: medium` — lower cost/latency):
- **baseline-agent**: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:baseline]`
- **monitor-agent**: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:monitor]`
- **experiment-agent**: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:experiment]`
- **prerequisites-agent**: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:prerequisites]`

**Analytical agents** (`model: opus[1m]`, `effort: xhigh`):
- **research-agent**: WebSearch, WebFetch, Read, Write, Bash, Glob, Grep, Skill, alphaxiv MCP tools (6), gitnexus MCP tools (3: `context`, `query`, `impact`) — skills: `[ml-optimizer:research, claude-mem:mem-search, superpowers:verification-before-completion]`
- **tuning-agent**: Read, Write, Bash, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:hp-tune, claude-mem:mem-search, superpowers:verification-before-completion]`
- **implement-agent**: Bash, Read, Write, Edit, LSP, Glob, Grep, Skill, WebSearch, WebFetch, alphaxiv MCP tools (2: repo reader, PDF Q&A), gitnexus MCP tools (3: `context`, `query`, `impact`) — skills: `[ml-optimizer:implement, ml-optimizer:evolve, ml-optimizer:shinka-setup, ml-optimizer:shinka-convert, ml-optimizer:shinka-run, ml-optimizer:shinka-inspect, superpowers:systematic-debugging, superpowers:verification-before-completion, karpathy-skills:karpathy-guidelines]` — runs the implement skill to apply the selected research proposals as git branches `ml-opt/<slug>`, **inside a git worktree outside `<exp_root>/`** (implement skill Step 3.1/4), with progressive validation incl. `LSP` (pyright). Dispatched inside the Phase 6 workflow; the workflow runtime (not a subagent) can fan implement-agents out in parallel, one worktree per branch. Must query gitnexus (`context`/`query`/`impact`) before editing. Post-implementation review (`feature-dev:code-reviewer`, `pr-review-toolkit:silent-failure-hunter`, `pr-review-toolkit:pr-test-analyzer`) runs as parallel stages inside the same workflow
- **analysis-agent**: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:analyze, claude-mem:mem-search, superpowers:verification-before-completion]` (includes session review mode)
- **report-agent**: Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:report, superpowers:verification-before-completion]`

**Main-thread agent** (activated by `settings.json`):
- **orchestrator-agent**: Agent, Read, Write, Edit, Bash, Glob, Grep, Skill, WebSearch, WebFetch — skills: `[ml-optimizer:orchestrate, superpowers:verification-before-completion]` — main thread when plugin is enabled, auto-starts Phase 0 via `initialPrompt: "/ml-optimizer:orchestrate"`

For parallel execution, use `run_in_background: true`. External skills/tools also available:
- **research-agent**: `context7` for framework API docs, `claude-mem:mem-search` for cross-session learning, `alphaxiv` MCP for paper search/analysis (6 tools: embedding search, full-text search, agentic retrieval, paper content, PDF Q&A, GitHub repo reader), and `gitnexus` MCP (`context`/`query`/`impact`) — **required** — to index candidate reference repos (immediately after clone) and query structure/call-graph for a feasibility read before recommending `from_reference`. Querying gitnexus is mandatory, not best-effort
- **implement-agent**: `context7` for API docs, `superpowers:systematic-debugging` for error recovery, `superpowers:verification-before-completion` to confirm changes work before finishing, `karpathy-skills:karpathy-guidelines` to keep changes surgical, `alphaxiv` MCP for reference repo exploration (`read_files_from_github_repository`) and paper clarification (`answer_pdf_queries`), and `gitnexus` MCP (`context`/`query`/`impact`) — **required** — to index a cloned reference repo (`from_reference`, immediately after clone) and the target repo, then query call-graph and blast-radius (`impact`) before editing. Querying gitnexus before adapting/editing is mandatory, not best-effort. During validation it runs `LSP` (pyright) diagnostics on modified `.py` files to catch undefined names, type mismatches, and unresolved imports statically before any GPU time. Inside the Phase 6 workflow it implements each proposal in its own git worktree (Step 3.1); the workflow can fan these out in parallel. **Post-implementation review runs as parallel stages of the same Phase 6 workflow**: `feature-dev:code-reviewer` + `pr-review-toolkit:silent-failure-hunter` per validated branch (the latter catches swallowed NaN losses, failed CUDA/optimizer ops, `except: pass` around training/eval) and `pr-review-toolkit:pr-test-analyzer` on the unit test (advisory).
- **orchestrator**: `claude-mem:mem-search` in Phase 1 for cross-session recall, `superpowers:brainstorming` in Phase 0/4 for complex multi-objective scenarios

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
| `${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py available\|mcp-registered\|require\|index <path> [--force]\|is-indexed <path>` — required GitNexus code-graph helper. `available` checks the `gitnexus` CLI is on PATH; `mcp-registered` prints `{"registered": true\|false\|null}` (true = MCP server registered, false = not, null = cannot determine because the `claude` CLI is absent; always exits 0); `require` reports availability plus `"mcp_registered": true\|false\|null` and exits nonzero only when the CLI is absent (install guidance `["npm install -g gitnexus", "gitnexus setup"]`); `index` runs `gitnexus analyze <path> --index-only` (non-invasive — never injects a GitNexus section into the indexed repo's CLAUDE.md/AGENTS.md, never installs `.claude/` skills; writes `<path>/.gitnexus`, auto-adds `.gitnexus/` to the repo git exclude, skips an already-indexed path unless `--force`, never raises — failures via the returned dict); `is-indexed` checks for an existing index. The wrapper never raises, but skills treat `available()==False` / index `success: false` as a **hard error** (halt with install/repair guidance), not a fallback |
| `${CLAUDE_PLUGIN_ROOT}/scripts/pipeline_state.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/pipeline_state.py <exp_root> validate\|save\|load\|cleanup\|verify-baseline\|gate\|log-gate\|log-decision\|replay-check\|decisions` — phase gates, decision logging |
| `${CLAUDE_PLUGIN_ROOT}/scripts/schema_validator.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/schema_validator.py <filepath> result\|baseline\|manifest\|prerequisites [--strict]` — validates JSON schemas. `--strict` enforces completeness. Also: `relay <route> <json>` validates the file/args payloads handed off between workflow stages (5 routes; schemas unchanged from the former inter-agent relay) |
| `${CLAUDE_PLUGIN_ROOT}/scripts/plot_results.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/plot_results.py <results_dir> <metric> comparison\|timeline\|sensitivity <hp>\|progress [--higher-is-better]` — ASCII charts + matplotlib progress chart |
| `${CLAUDE_PLUGIN_ROOT}/scripts/prerequisites_check.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/prerequisites_check.py scan-imports\|check-packages\|detect-env\|detect-format\|detect-format-project\|validate-data\|bulk-install-cmd\|gpu-install-cmd` — dataset, environment, and GPU-aware install validation |
| `${CLAUDE_PLUGIN_ROOT}/scripts/dashboard.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/dashboard.py <exp_root> [--live] [--table] [--serve --port 8080]` — self-contained HTML dashboard (progress timeline, results table, HP sensitivity, research agenda, error summary, method explanations). `--live` = 30s auto-refresh. `--table` generates `results-table.md`. |
| `${CLAUDE_PLUGIN_ROOT}/scripts/excalidraw_gen.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/excalidraw_gen.py <exp_root> pipeline\|comparison\|hp-landscape\|architecture <args>` — generate Excalidraw JSON diagrams (pipeline overview, experiment comparison, HP landscape, architecture changes) |
| `${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log\|show\|patterns\|summary\|success\|proposals\|rank\|log-suggestion\|suggestion-history\|dead-end <add\|list\|check>\|agenda <init\|update\|list\|add>` — error tracking, pattern detection, success metrics, proposal outcomes, suggestion ranking, suggestion history, dead-end catalog, research agenda |
| `${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> init-goals\|read-goals\|log-behavior <category> <json>\|query-behaviors [category]\|validate-output <agent> <json>\|summary\|sync-from-errors` — goal anchoring, behavioral memory, agent output validation, compact briefings |
| `${CLAUDE_PLUGIN_ROOT}/scripts/setup_evolve.sh` | `bash scripts/setup_evolve.sh` — initialize ShinkaEvolve submodule and create skill symlinks for auto-discovery |
| `${CLAUDE_PLUGIN_ROOT}/scripts/round_manager.py` | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/round_manager.py <exp_root> create-round\|current-round\|register-experiment\|close-round\|next-id\|check-baseline\|check-prerequisites\|check-manifest\|check-round\|check-proposals` — round lifecycle + completeness checking |
| `${CLAUDE_PLUGIN_ROOT}/scripts/validate_experiment_write.py` | PreToolUse hook — validates Write/Edit to experiment result files against schema + correct round directory |

### Method Proposals (LLM Knowledge + Web Search)

The research skill supports `source: "both"` mode: the LLM proposes methods from training knowledge supplemented by web search. Proposals scoped by `scope_level`: `"training"` (safest), `"architecture"`, or `"full"`. Triggered:
- **Pre-loop** (Phase 4, option 5): user generates method proposals before the experiment loop
- **Mid-loop** (inside the Phase 7 workflow): when analyze recommends `pivot_type: "method_proposal"` or `"qualitative_change"`, the workflow runs research + implement `agent()` calls (or a nested workflow)

Because the Phase 7 workflow takes **no mid-run user input**, the scope (`method_proposal_scope`) and iteration budget (`method_proposal_iterations`) are **pre-authorized at Phase 4** and read from `user_choices` — the mid-loop trigger does not re-prompt. Knowledge-based proposals have confidence capped at 7/10.

### Three-Tier Result Tracking

Experiments carry two tracking fields:
- **`method_tier`**: `"baseline"` | `"method_default_hp"` | `"method_tuned_hp"` — which tier of the comparison
- **`proposal_source`**: `"paper"` | `"llm_knowledge"` | `null` — origin of the method

Enables three-tier attribution: baseline metrics → method with default HPs (isolated effect) → method with tuned HPs (combined effect). The report skill generates a three-tier comparison table when these fields are present.

Additionally, stacking experiments use:
- **`stacked_default_hp`**: Combined methods tested with best individual HPs
- **`stacked_tuned_hp`**: Combined methods after HP-tuning

Stacking experiments also carry `code_branches` (array of combined branches), `stacking_order`, and `stack_base_exp`.

### Pipeline Resumption

The orchestrator can be stopped and resumed. On restart it reads `pipeline-state.json` and uses `cleanup_stale()` to mark interrupted experiments failed after a timeout. `validate_phase_requirements()` prevents cascading failures. Pipeline state persists Phase 0 user choices (`primary_metric`, `divergence_metric`, `divergence_lower_is_better`, `lower_is_better`, `target_value`, `train_command`, `eval_command`, `train_data_path`, `val_data_path`, `prepared_train_path`, `prepared_val_path`, `env_manager`, `env_name`, `model_category`, `user_papers`, `method_proposal_scope`, `method_proposal_iterations`, `hp_batches_per_round`, `fixed_time_budget`, `fixed_epoch_budget`, `fixed_step_budget`) via `save_state(user_choices={...})` so they survive interruptions without re-asking. The experiment loop also persists, at pipeline-state root level: `consecutive_stop_count` (telemetry only — not an exit trigger), `stuck_protocol_triggered` (the loop-exit flag: set `true` when the stuck protocol returns no new in-scope proposals; with an empty research agenda and a flat best metric it defines the *fixpoint* at which the orchestrator exits to Phase 9; reset `false` whenever fresh proposals or metric improvement appear), and `baseline_checksum` (SHA-256 of baseline metrics for integrity). A separate `user-choices-backup.json` gives redundant recovery if the main state file corrupts.

Phase 5–8 workflows resume **within a session** via the `Workflow` runtime's `resumeFromRunId` — relaunch with the prior run id to continue where it stopped. With the file-persisted results/rounds/manifest under `<exp_root>/`, an interrupted phase resumes without re-running completed work. Because phases 5–8 use fresh `agentType` dispatches (no `agent_registry`), there is no session-scoped agent state to clear — the orchestrator just relaunches the workflow. Phase 7 runs **without mid-run user input**: its autonomy parameters (`method_proposal_scope`, `method_proposal_iterations`, budget) are pre-authorized at Phase 4 and read from `user_choices`. A genuine user-decision point returns as a workflow boundary, resumed via `resumeFromRunId`.

### Multi-Run Pattern (one `<exp_root>` = one optimization run)

The plugin does **not** run-namespace state files inside `<exp_root>/`. Each `<exp_root>` is one self-contained run — its behavioral memory, round manifest, and reports are scoped to that directory. For a new optimization direction on the same project, point `exp_root` at a new directory at Phase 0. No hardcoded output location; `exp_root` can be any absolute path.

```
# Each run gets its own <exp_root>:
<exp_root> = ~/my-project/runs/01-label-smoothing/   # first run
<exp_root> = ~/my-project/runs/02-augmentation/      # second run, fresh state
<exp_root> = ~/my-project/runs/03-architecture/      # third run
```

**Breadcrumb (`.claude/ml-optimizer.json`):** The orchestrator writes a breadcrumb at the project root so hooks find the active `<exp_root>`. It tracks all runs — each Phase 0 appends to `runs` and sets `active`:

```json
{"active": "/home/user/runs/02-augmentation", "runs": ["/home/user/runs/01-label-smoothing", "/home/user/runs/02-augmentation"]}
```

Hooks and the write-validator read `active` to resolve the current `<exp_root>`. The old single-run format (`{"exp_root": "..."}`) is still supported for backward compatibility.

Phase 0 (`skills/orchestrate/references/phase-0-discovery.md` Step 1.1) detects a prior completed run (`phase == 9`) and prompts to start a new directory or continue. The `cwd-changed-detect-experiments.sh` hook emits a "completed optimization found" message on phase 9 entry. Cross-run comparison is not automated — diff the `results-table.md` files manually.

## Key Design Patterns

- **Workflow-driven phases 5–8**: Phases 5–8 run as dynamic workflows bundled in the orchestrate skill (`skills/orchestrate/workflows/phase-{5,6,7,8}-*.js`), launched via `Workflow({scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-N-<name>.js", args})` — one per phase. They are **internal pipeline steps, not user-facing /commands** — launching by `scriptPath` (not saved name) keeps them out of the `/command` namespace (`.claude/workflows/` would expose them as user `/slash-commands`, which we avoid). The script holds that phase's fan-out/loop; the Phase 7 experiment loop and Phase 8 stacking loop live entirely inside their scripts. **Required** — no `Agent`/`SendMessage` fallback path; the runtime's `agent()` call is the dispatch mechanism. The orchestrator reads each structured `schema:` return and runs the user checkpoint **between** phases ("run each stage as its own workflow"). Phases 0/1, 2, 3, 4, 9 stay direct `Agent()` dispatch (interactive/trivial).
- **agentType reuse**: Inside a workflow, agents dispatch via `agent(prompt, {agentType: "ml-optimizer:<name>-agent"})`, reusing the existing definitions (tools, skills, model, effort intact) — scripts never reimplement an agent's behavior. Workflow agents reach session MCP tools (alphaxiv, context7, claude-mem) via ToolSearch. Reviews reuse `feature-dev:code-reviewer` and `pr-review-toolkit:silent-failure-hunter` the same way.
- **File/args handoff (no relay)**: No `SendMessage`, no `agent_registry`, no "CONTEXT FROM OTHER AGENTS" relay. Cross-agent context inside a workflow flows via (a) the workflow's `args`, (b) files agents write under `<exp_root>/` (results, manifests, research agenda, `batch-N-analysis.md`), and (c) each agent's structured return relayed by the script to the next stage. E.g. analysis writes `batch-N-analysis.md` and the next round's tuning agent reads it; the implement workflow reads the proposals the research workflow returns. `schema_validator.py relay <route> <json>` validates these file/args payloads (schemas unchanged from the former relay routes).
- **Non-git fallback**: If the target project isn't a git repo, the implement skill uses file backups instead of branches. Each proposal is validated against a clean baseline backup (restore-before-apply) to prevent cross-proposal leakage. This forces sequential (not parallel) experiment execution.
- **Loop exit conditions**: The experiment loop is autonomous — runs until: (1) target metric achieved, (2) user stops, or (3) the orchestrator judges approaches exhausted. When analysis advises stop, the orchestrator invokes the stuck protocol (research for fresh ideas), then decides on evidence: continue if research yields fresh non-dead-end proposals, exit to Phase 9 if genuinely out of directions (no new proposals, flat metric, empty agenda). No hardcoded stop-count threshold — `consecutive_stop_count` is one input to the judgment.
- **Proposal priority scoring**: `(impact * confidence) / (11 - min(feasibility, 10))` — feasibility clamped to [1,10] to prevent division by zero.
- **Spearman correlation**: `${CLAUDE_PLUGIN_ROOT}/scripts/result_analyzer.py` uses rank correlation with average-rank tie-breaking to identify HP-metric relationships (no scipy dependency).
- **Dual implementation strategy**: Proposals include an `implementation_strategy` field (`from_scratch` or `from_reference`). The implement agent dispatches accordingly — implement from paper descriptions (Section 8) or clone and adapt reference repos (Section 9). The research agent decides strategy from repo availability and quality.
- **Research skill modes**: The research skill accepts `source` (`"web"` | `"knowledge"` | `"both"`), `scope_level` (`"training"` | `"architecture"` | `"full"`), and `output_path`. Knowledge mode skips web search, uses LLM training knowledge only.
- **Scope-gated pivots**: The analysis agent's pivot tree respects `scope_level`: `"training"` (HP-only) disables research and code_evolution — only HP adjustments; `"architecture"` enables research but not ShinkaEvolve; `"full"` enables everything including code_evolution via ShinkaEvolve. Keeps the pipeline matching what the user asked for.
- **Auto-resolution behaviors**: The pipeline auto-resolves many situations: Phase 2 partial prereqs → proceed with warnings; RL polarity → infer from metric name; dirty working trees → no action (implementation runs in an isolated git worktree, main tree untouched); environment mismatches → use detected manager; missing conda envs → auto-create; no eval command → fall back to training output metrics. Only unrecoverable errors (Phase 2 failed, baseline failed) block the pipeline. Decisions logged to dev_notes and error tracker for post-session review.
- **Parallel research**: All research-skill WebSearch calls issue simultaneously in one tool-call message, alongside 3 alphaxiv search calls (embedding similarity, full-text keyword, agentic retrieval). WebFetch follow-ups for different URLs are also parallelized. Domain-specific query sets (NLP, CV, RL, time-series) issue alongside generic queries. If alphaxiv MCP is unavailable, WebSearch is full-coverage fallback.
- **Worktree-isolated implementation (workflow-parallel)**: The Phase 6 workflow (`phase-6-implement.js`) dispatches implement-agents via `agentType`, **one git worktree per `ml-opt/<slug>` branch, outside `<exp_root>/`** (implement skill Step 3.1) so the main working tree is never disturbed; each worktree is removed at the end, branches persist. Branches use `git checkout -b ml-opt/<slug> <original_branch>` (branching off the base commit, valid even though `<original_branch>` is checked out in the main tree). Because the runtime — not a subagent — dispatches, implement-agents fan out **in parallel** (per-branch worktrees prevent cross-proposal leakage); the reviewers (`feature-dev:code-reviewer`, `pr-review-toolkit:silent-failure-hunter`) run as parallel stages in the same workflow.
- **Configurable divergence thresholds**: `${CLAUDE_PLUGIN_ROOT}/scripts/detect_divergence.py` supports per-model-category overrides via `MODEL_CATEGORY_DEFAULTS` dict and `--model-category` flag. RL uses `explosion_threshold=20.0, plateau_patience=50` (avoids false positives on reward spikes); generative uses `explosion_threshold=10.0, plateau_patience=40` (accommodates slow convergence). Individual thresholds also override via `--explosion-threshold` and `--plateau-patience`.
- **Experiment timeout**: Hard timeout of `baseline_training_time * 3` (fallback: 6 hours). Timed-out experiments are killed and marked `status: "timeout"`.
- **Research failure recovery**: If web search fails (both WebSearch and alphaxiv), the orchestrator retries with `source: "knowledge"` (LLM-only); if that also fails, it continues HP-only. Each fallback is logged. Within a search, alphaxiv failure alone does not trigger the knowledge fallback — WebSearch results suffice.
- **alphaxiv MCP integration**: The research agent uses all 6 alphaxiv tools for paper discovery (3 search tools in parallel), content extraction (`get_paper_content`, `answer_pdf_queries`), and reference repo exploration (`read_files_from_github_repository`). The implement agent uses 2 (`read_files_from_github_repository` for pre-clone repo assessment, `answer_pdf_queries` to clarify ambiguous implementation steps from source papers). All alphaxiv searches run in parallel with WebSearch. Optional — if the MCP server is unavailable, all workflows fall back to WebSearch/WebFetch transparently.
- **GitNexus code-graph understanding (REQUIRED)**: GitNexus (required MCP + CLI, mirrors the alphaxiv pattern) indexes a repo into a queryable code knowledge graph. A **hard prerequisite** — on par with git — with **no grep/analyze fallback for code understanding**. **Querying is MCP-only by design**: agents query exclusively via `mcp__gitnexus__context`/`mcp__gitnexus__query`/`mcp__gitnexus__impact` — **no gitnexus-CLI query fallback** (agents never run `gitnexus query`/`context`/`impact` from Bash). If the MCP server is not registered or fails, code understanding fails — recovery is `gitnexus setup` then restart the session (MCP tools load at session start, so a freshly-registered server needs a restart to appear). **Phase 2 verifies install** (`python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py available`); if the CLI is absent, Phase 2 BLOCKS as an unrecoverable prerequisite failure (on par with "Phase 2 failed blocks the pipeline") with install instructions (`npm install -g gitnexus && gitnexus setup`; manual MCP-registration fallback `claude mcp add --transport stdio --scope user gitnexus gitnexus mcp`). Phase 2 is **best-effort about MCP registration**: if the CLI is installed but the MCP server isn't registered (`gitnexus_utils.py mcp-registered`/`require` reports `mcp_registered: false`), it WARNS rather than hard-blocks — only a missing CLI hard-blocks. **Every code repo is indexed**: the TARGET project once at Phase 2 (`python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py index <project_root>`, graph at `<project_root>/.gitnexus`); EVERY reference repo immediately after clone (`python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py index <ref_repo>`). Indexing is **non-invasive** — the wrapper runs `gitnexus analyze <path> --index-only` (never a bare `gitnexus analyze`), so it does NOT inject a GitNexus section into the indexed repo's CLAUDE.md/AGENTS.md and does NOT install `.claude/` skills there; the indexed repo (or worktree) is never contaminated. The implement agent indexes the cloned reference repo (`from_reference`) and the target repo, then **must** query structure/call-graph with `mcp__gitnexus__context`/`mcp__gitnexus__query` and blast-radius with `mcp__gitnexus__impact` before editing — so changes are surgical and side-effects understood. The research agent indexes candidate reference repos and **must** query them for a feasibility read before recommending `from_reference`. Querying gitnexus is mandatory, not best-effort. `${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py` provides `available()`/`index()`/`is_indexed()` (CLI: `available`, `mcp-registered`, `require`, `index <path> [--force]`, `is-indexed <path>`); `index()` runs `gitnexus analyze <path> --index-only` and **never raises** — failures reported via the returned dict. The wrapper never raises, but skills treat `available()==False` / index `success: false` as a **hard error** (halt with install/repair guidance), NOT a silent fallback. `implement_utils.py analyze` remains only for its narrow framework-detection role — NOT a gitnexus fallback. `.gitnexus/` index artifacts are auto-excluded by the wrapper (it adds `.gitnexus/` to the repo's git exclude on a successful index) but must still never be committed (never `git add`-ed).
- **OOM feedback loop**: On OOM the batch size is recorded in the error tracker; the next hp-tune invocation is passed `max_batch_size` to avoid re-proposing configs that will OOM.
- **All-diverge recovery**: If all experiments in a batch diverge, a recovery batch with halved learning rates is attempted before stopping.
- **HP-only research routing**: Proposals with `type: "hp_only"` skip the implement skill and route directly to hp-tune as search-space modifications.
- **Tabular ML HP strategy**: For tree-based models (sklearn/XGBoost/LightGBM), iteration 1 explores `max_depth`/`n_estimators` first, not learning rate.
- **Training budget options**: Phase 0 offers three budget modes: `fixed_time_budget` (seconds) for wall-clock-normalized comparison, `fixed_epoch_budget` (integer) for deterministic reproducibility, or `fixed_step_budget` (integer environment timesteps) for RL — mapped to the framework's timestep flag (e.g., `--total_timesteps`). When set, both baseline AND experiments use the same budget. The baseline skill (Step 2.2) wraps training with `timeout` when `fixed_time_budget` is set, for fair comparisons. Framework-native time limits (Lightning `--max_time`, HuggingFace `timeout` in TrainingArguments) are preferred when available. Results include `time_budget_seconds`. HP-tune adjusts proposals for the budget (shorter schedules, appropriate LR scaling). Makes metrics directly comparable without duration normalization.
- **Small dataset awareness**: The research skill checks dataset size. Under 5K samples it shifts toward low-data techniques (transfer learning, fine-tuning, few-shot, adapters, prompt tuning, synthetic data, semi-supervised) instead of heavy augmentation and regularization, which underperform on small data.
- **Evolutionary code refinement**: When HP tuning shows diminishing returns, the analyze skill can recommend `pivot_type: "code_evolution"`. The orchestrator dispatches the implement-agent with the evolve skill (`Skill("ml-optimizer:evolve")`), which runs the full ShinkaEvolve pipeline: `shinka-convert` (task from best branch) → `shinka-run` (file-based handoff, `SHINKA_PROVIDER=claude_code`) → `shinka-inspect` (extract best mutation) → commit as `ml-opt/evolved-<slug>`. Evolve HPs (`num_generations`, `population_size`) are tuning-agent-driven: the orchestrator dispatches the tuning agent to propose them first, from prior outcomes in `learned-behaviors.json` under category `evolve_hp`. Defaults: 10 generations, population 2. If ShinkaEvolve is unavailable, the evolve skill reports `shinkaevolve_unavailable` and the orchestrator falls back to research → implement. Setup: `bash scripts/setup_evolve.sh` inits the submodule and creates symlinks (`skills/shinka-*` → `skills/evolve/ShinkaEvolve/skills/shinka-*`), required for Claude Code skill auto-discovery.
- **Auto-repair loop**: When training/eval commands fail during baseline or experiments, the agent captures stderr, diagnoses, fixes (install package, adjust path, reduce batch size), and retries up to 3 times. OOM (deterministic) and SyntaxErrors (code bugs) are not retried. Identical errors on consecutive attempts skip further retries (loop detection). Each retry is logged. Intra-agent retry, separate from the orchestrator's Phase 3 retry logic.
- **Goal anchoring & behavioral memory**: `${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py` maintains two project-scoped files: `optimization-goals.json` (goal anchor, written once at Phase 0) and `learned-behaviors.json` (accumulated behavioral memory). The orchestrator calls `validate-output` after hp-tune, research, and analyze to catch drift (frozen param changes, scope breaches, dead-end re-proposals, metric mismatches). Each agent reads a compact `summary` (goals + constraints + dead-ends) before acting. All 10 agents have `memory: local` for persistent role-specific memory at `.claude/agent-memory-local/<agent-name>/`.
- **Immutable baseline**: After baseline (Phase 3), a SHA-256 checksum of the baseline metrics dict is stored in `pipeline-state.json`. Before each experiment batch (Phase 7) and on resumption, it is verified against `baseline.json`. If metrics changed (modification, corruption, or tampering), the pipeline halts with a critical error. Prevents invalid comparisons during long runs.
- **Stuck protocol**: When analysis advises stop, the orchestrator reads error patterns, dead ends, and the research agenda, then dispatches research for fresh ideas, then runs the **Exit Judgment**: if research returns new in-scope proposals (or the agenda has untried items, or the metric improved), reset `stuck_protocol_triggered`/`consecutive_stop_count` and continue; else set `stuck_protocol_triggered=true`. Exit to Phase 9 only at the *fixpoint* — no new in-scope proposals AND empty agenda AND flat best metric (in-scope idea space exhausted with no progress to build on). No hardcoded stop-count threshold; the decision is logged via `pipeline_state.py log-decision`.
- **Research agenda as living document**: `${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py` maintains `research-agenda.json` — a prioritized list of ideas that evolves over the session. The research skill initializes it from proposals (Phase 5). The analyze skill updates it after each batch: marking ideas tried/improved/dead-end, adjusting priorities on evidence, and adding new ideas from results. The hp-tune skill reads it for high-priority untried techniques. The report skill summarizes it in the final report.
- **Overfitting detection**: `${CLAUDE_PLUGIN_ROOT}/scripts/detect_divergence.py` provides `check_overfitting(train_values, val_values)` comparing train vs val trajectories — detects when val worsens N consecutive steps while train improves. Severity mild/moderate/severe. Default patience=5; model-category overrides: RL patience=10, generative patience=8. CLI: `--check-overfitting '<train_json>' '<val_json>'`.
- **HP interaction detection**: `${CLAUDE_PLUGIN_ROOT}/scripts/result_analyzer.py` provides `detect_hp_interactions()` computing 2-way interaction terms via product of centered ranks. Reports interactions stronger than either individual HP correlation. Integrated into `analyze()` output.
- **Adaptive branch budget allocation**: `${CLAUDE_PLUGIN_ROOT}/scripts/result_analyzer.py` provides `compute_branch_scores()` scoring branches by `improvement_pct * confidence_factor`. HP-tune allocates experiment slots proportionally in iteration 2+.
- **Checkpoint warm-starting**: Experiments can warm-start from a previous checkpoint. `${CLAUDE_PLUGIN_ROOT}/scripts/experiment_setup.py` supports `checkpoint_path`. HP-tune proposes warm-started configs (lower LR, fewer epochs, same-branch only) when enabled — saving ~50-80% compute in later iterations.
- **Dead-end catalog**: `${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py` maintains `dead-ends.json` tracking techniques conclusively unpromising. The research and hp-tune skills consult it before proposing, preventing wasted budget. Fuzzy matching (case-insensitive, substring containment, hyphen/underscore normalization) prevents near-duplicate re-proposals. The analyze skill logs dead ends when branches are pruned or all experiments fail.
- **Concurrent-safe error logging**: `${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py` uses `fcntl.flock()` around the read-modify-write in `log_event()` so concurrent agents don't lose events.
- **Result file filtering**: `${CLAUDE_PLUGIN_ROOT}/scripts/result_analyzer.py` only loads `exp-*.json` and `baseline.json`, so non-experiment files don't inflate counts.
- **alphaxiv query format differences**: `embedding_similarity_search` expects 2-3 descriptive sentences, `full_text_papers_search` expects 3-4 short keywords (no quotes), `agentic_paper_retrieval` expects a natural-language question. Wrong format degrades quality. All 3 called in parallel per alphaxiv docs.
- **HuggingFace Trainer log format**: `${CLAUDE_PLUGIN_ROOT}/scripts/parse_logs.py` detects and parses HuggingFace Trainer's single-quote Python dict format (`{'loss': 0.5, 'epoch': 1.0}`).
- **Baseline eval auto-fallback**: If no eval command is found, baseline uses training output metrics instead of blocking on user input.
- **Pre-flight file validation**: The implement skill validates all `files_to_modify` exist before creating branches or starting. Missing-file proposals are marked `preflight_failed`.
- **Tabular ML adaptive timeout**: For non-iterative frameworks, experiment timeout is `fit_duration * (max_iters / profiling_iters) * 2` instead of a generic 4-hour fallback.
- **Method stacking (Phase 8)**: After independent testing finds ≥5 methods that improve over baseline, the orchestrator sequentially merges them in descending improvement order. Each step creates `ml-opt/stack-<N>` by merging the next method into the current best stack. Clean merges proceed directly; conflicts are resolved by the implement-agent. Each merged branch is reviewed (`feature-dev:code-reviewer` + `pr-review-toolkit:silent-failure-hunter`, mirroring Phase 6) before the stacked experiment — a critical finding (e.g., a conflict resolution that dropped a NaN/CUDA guard) skips that method, since a merge-introduced silent failure would corrupt the metric. If a combination degrades, that method is skipped. After each successful step, the analysis agent assesses the result — if the stacked gain is less than the best individual method's gain (method interference), it recommends `code_evolution` and the evolve skill (`Skill("ml-optimizer:evolve")`) optimizes code-level interactions via ShinkaEvolve (5 generations, population 2), mirroring Phase 7 where analysis is always the decision-maker for evolution. If the evolved code improves over the pre-evolution stack, it becomes the new base; else discarded. Optional HP-tuning (1-2 iterations, narrowed scope) follows on the (potentially evolved) code when the combo shows >1% improvement. Stacking state persists in `pipeline-state.json` (including `evolved_methods`) for resumption. Requires git branch strategy — skipped for `file_backup` projects.
- **Structured ideation for knowledge mode**: The research skill's knowledge-based proposal generation (Phase 7 method proposals) uses a diverge-converge-refine process with 6 ideation lenses (Problem-First, Analogical Reasoning, What Changed Recently, Constraint Manipulation, Negation/Inversion, Composition/Decomposition). Generates 10-15 candidates, filters via scope/dead-end/two-sentence-test, refines survivors with implementation details.
- **Statistical confidence assessment**: The analyze skill (Step 2.2) computes effect sizes (Cohen's d) for HP impact when ≥5 experiments exist, and labels findings by confidence (high/medium/low). Method attribution distinguishes whether improvements came from the code change, HP tuning, or their compound effect.
- **Reproducibility metadata**: The experiment skill (Step 1.3) captures random seeds, pip freeze snapshots, git SHA, and framework versions under the `"reproducibility"` key in result JSONs. Enables exact reproduction of best experiments.
- **Report threats to validity**: The report template includes a "Threats to Validity" section covering single-seed risk, limited search space, dataset specificity, budget constraints, and noise margins.
- **Citation verification**: The report skill (Step 5.3) cross-references technique claims against experiment data and spot-checks source URL accessibility before writing the final report.

## Gotchas

- **`${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py validate-output` returns exit code 2 for violations**: 0 = valid, 1 = script error, 2 = violations found. The orchestrator checks the exit code and parses the JSON output's `violations` array.
- **`${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py` imports from `${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py`**: for dead-end checks it lazily imports `is_dead_end` and `get_dead_ends`. Both scripts must be in the same `scripts/` directory.
- **`${CLAUDE_PLUGIN_ROOT}/scripts/detect_divergence.py` CLI takes a JSON string, not a file path**: `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/detect_divergence.py '[0.5, 0.4, 100.0]'` — quotes required. Pass `--higher-is-better` for reward-like metrics, `--model-category rl` for RL thresholds.
- **`${CLAUDE_PLUGIN_ROOT}/scripts/implement_utils.py` has four CLI modes**: default (parse proposals), `clone <url> <dest>`, `analyze <path>`, `diff <project_root> <branch>` — different argument patterns each. `analyze` is framework detection only — NOT a gitnexus fallback (gitnexus is required; see below).
- **Don't commit `.gitnexus/` index artifacts**: indexing (`python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py index <path>`, which runs `gitnexus analyze <path> --index-only`) writes a `.gitnexus/` directory into the indexed repo (target project and every cloned reference repo). The wrapper auto-excludes `.gitnexus/` (adds it to the repo's git exclude on a successful index), but these are local artifacts — never `git add` or commit them.
- **Indexing is non-invasive (`--index-only`)**: index commands go through the wrapper (`python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py index <path>`), which runs `gitnexus analyze <path> --index-only` — NOT a bare `gitnexus analyze`. The `--index-only` flag keeps the index pure: no GitNexus section injected into the indexed repo's CLAUDE.md/AGENTS.md and no `.claude/` skills installed, so the indexed repo (or worktree) is never contaminated. Don't write prose implying a bare `gitnexus analyze <path>` is run or that indexing rewrites those files.
- **Querying is MCP-only by design**: agents query the code graph exclusively via `mcp__gitnexus__context`/`mcp__gitnexus__query`/`mcp__gitnexus__impact`. **No gitnexus-CLI query fallback** — never tell an agent to run `gitnexus query`/`context`/`impact` from Bash. If the MCP server is not registered or fails, code understanding fails — recovery is `gitnexus setup` then **restart the session** (MCP tools load at session start, so a freshly-registered server only appears after a restart). Phase 2 is best-effort about MCP registration: if the CLI is installed but the MCP server isn't (`gitnexus_utils.py mcp-registered`/`require` reports `mcp_registered: false`), it WARNS rather than hard-blocks — only a missing CLI hard-blocks.
- **GitNexus is required, not optional**: no grep/analyze fallback for code understanding. Phase 2 verifies `gitnexus_utils.py available` and BLOCKS the pipeline as an unrecoverable prerequisite failure if the CLI is absent. The wrapper never raises, but skills treat `available()==False` / index `success: false` as a hard error (halt with install/repair guidance: `npm install -g gitnexus && gitnexus setup`; manual MCP-registration fallback `claude mcp add --transport stdio --scope user gitnexus gitnexus mcp`).
- **`gitnexus setup` has global install side effects**: beyond auto-registering the gitnexus MCP server for Claude Code, it also installs gitnexus's own global skills (7) and PreToolUse/PostToolUse hooks into `~/.claude/`, affecting all Claude Code projects. Mention this so users aren't surprised.
- **Metric routing is split**: Monitor/divergence uses `divergence_metric` (default `"loss"`, lower-is-better; reward-like metrics allowed with `divergence_lower_is_better=false`). Analyze/hp-tune use `primary_metric`. Mixing these up causes silent wrong behavior.
- **Branch experiments are independent**: results on `ml-opt/branch-a` tell you nothing about HPs for `ml-opt/branch-b`. The tuning agent must group by `code_branch` before analyzing trends.
- **Workflow scripts are plain JS, not TS**: `skills/orchestrate/workflows/phase-{5,6,7,8}-*.js` (bundled in the orchestrate skill, launched via `Workflow({scriptPath, args})` — not user `/commands`) are plain JavaScript, no build step. Each begins with `export const meta = {...}` (`meta.name` display-only) and uses runtime-provided `args`, `agent()`, `parallel()`, `pipeline()`, `log()`, and `schema:` returns.
- **No nondeterministic calls in workflows**: scripts must not call `Date.now()`, `Math.random()`, or similar — workflows are replay/resume-safe and nondeterminism breaks that. Use exp-ids and round directories from `round_manager.py` for unique names, and timestamps from the agents/scripts (not the workflow body).
- **Workflows take no mid-run user input**: a dynamic workflow cannot prompt mid-run, so **Phase 7 is pre-authorized at Phase 4** — `method_proposal_scope`, `method_proposal_iterations`, and the budget are confirmed there and read from `user_choices`. A genuine user-decision point returns as a workflow boundary (relaunch via `resumeFromRunId`).
- **Workflow concurrency caps**: the `Workflow` runtime caps fan-out at **16 concurrent** `agent()`/stage calls and **1000 total** per run. The Phase 6 implement fan-out and Phase 7 experiment parallelism must respect these — chunk larger batches.
- **Within-session resumption via `resumeFromRunId`**: phases 5–8 resume **within the same session** by relaunching with the prior `resumeFromRunId`; the file-persisted results/rounds/manifest under `<exp_root>/` carry completed work across the boundary. No cross-session agent state — workers are fresh `agentType` spawns each time.
- **Experiment results MUST live in round directories**: `exp-*.json` must be written to `results/round-N-<type>/exp-*.json`, NEVER directly to `results/exp-*.json`. The PreToolUse hook (`validate_experiment_write.py`) blocks any violating Write/Edit. Before dispatching experiment-agents, call `round_manager.py create-round <type>` and pass the round directory to the agent. Valid round types: `hp`, `evolved`, `research`, `stacked`.
- **Proposed configs are top-level with round structure**: HP proposals go in `proposed-configs/round-N-<type>/exp-*.json` (top-level, NOT under `results/`). The PreToolUse hook validates this. The round directory must exist (via `round_manager.py create-round`) before the tuning agent writes proposals.
- **Exp-ids are globally unique across rounds**: `exp-001` exists in only one round. `round_manager.py next-id` scans all round directories for the next available ID. Don't reuse exp-ids across rounds — earlier results are preserved and still counted.
- **`round_manager.py` uses `fcntl.flock` for manifest writes**: `rounds-manifest.json` is file-locked for concurrent safety. Manual edits may be lost if agents are running. Always use `round_manager.py` CLI (`create-round`, `register-experiment`, `close-round`).
- **Tabular ML frameworks skip divergence monitoring**: for scikit-learn, XGBoost, or LightGBM the orchestrator sets `divergence_metric` to `null` and skips the monitor skill. The baseline skill skips GPU profiling and throughput estimation for these.
- **Research findings files can be multiple**: `research-findings.md` (Phase 5 web search), `research-findings-method-proposals.md` (Phase 7 pre-loop), `research-findings-method-proposals-iter<N>.md` (Phase 7 mid-loop). The research skill's deduplication checks all of these to avoid re-proposing tried techniques.
- **ShinkaEvolve must use the local submodule, not PyPI**: the PyPI `shinka-evolve` lacks the `file_handoff_provider` module required for `SHINKA_PROVIDER=claude_code`. Always use the local submodule — install editable (`pip install -e skills/evolve/ShinkaEvolve/`) via `setup_evolve.sh`, or prepend `PYTHONPATH=${CLAUDE_PLUGIN_ROOT}/skills/evolve/ShinkaEvolve:$PYTHONPATH` before `shinka_run`.
- **ShinkaEvolve subprocess Python resolution**: `shinka/launch/scheduler.py` falls back to `sys.executable` (the Python running ShinkaEvolve) when no conda env or activate script is configured. Usually resolves correctly. If evaluation subprocesses fail with "python not found", set `python_executable` in `LocalJobConfig` or ensure `python` is on PATH.
- **ShinkaEvolve file handoff timeout via `SHINKA_HANDOFF_TIMEOUT`**: default 600s (10 min). Set before launching `shinka_run` with `SHINKA_PROVIDER=claude_code`. The agent must write a `<id>.inprogress` marker when it picks up a pending request — this resets `shinka_run`'s timeout from the acknowledgment point; without it the timeout runs from request creation. `shinka_run` also writes `<id>.heartbeat` files every 5s for liveness.
- **3-checkpoint output enforcement**: every agent output is enforced at 3 points: (1) **SubagentStart** injects the output contract — exact paths, schemas, examples — so agents know what to produce before starting (`scripts/output_contract.py` via `hooks/subagent-start-inject-goals.sh`). (2) **PreToolUse** blocks invalid JSON writes: wrong path, bad schema, missing completeness fields (completed need `iteration`, `method_tier`, `duration_seconds`; stacked need `code_branches`, `stacking_order`; failed/diverged need `notes`), frozen-parameter violations, OOM batch-size violations. Placeholder writes (`running`/`pending`) are exempt. (3) **SubagentStop** blocks finishing if any required output file is missing (result JSON, training log, script dir, artifacts dir, etc.). Contracts are defined once in `scripts/output_contract.py`, shared by SubagentStart (injection) and SubagentStop (verification). Two advanced fields: (a) `any_of` for mode-dependent outputs — e.g., analysis-agent must produce EITHER `reports/batch-<N>-analysis.md` (batch mode) OR `reports/session-review.md` (Phase 9 review mode), at least one required; (b) `required_if` for conditional outputs driven by another output's contents — e.g., prerequisites-agent produces `prepared-data/` ONLY when `dataset.prepared == true` in `prerequisites.json`. The condition is evaluated at SubagentStop by reading the referenced file and navigating a dotted jsonpath; missing/malformed reference files skip the conditional gracefully (caught by the unconditional entry instead). Agents not in the contract (monitor) are auto-approved.
