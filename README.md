# ml-optimizer

An autonomous ML model optimization plugin. Powered by Claude Code.

> Autonomous by default — runs until the goal is reached or you stop it. No hardcoded thresholds. No fixed pipeline. The orchestrator dispatches specialized agents, the analysis agent advises direction, and evidence drives decisions.

## Why This Approach?

Traditional ML optimization tools follow a fixed pipeline — same strategy every time. The ml-optimizer differs:

1. **Optimizes your ML model** — research-informed techniques, LLM-driven hyperparameter tuning, ShinkaEvolve evolutionary code mutations, and parallel GPU experiments

2. **Adapts its strategy on evidence** — the analysis agent evaluates results after each batch and recommends the next action (continue, pivot, or stop); the Phase 7 workflow acts on these directly, and the orchestrator reads the structured result between phases.

3. **Gets better each session** — claude-mem recalls insights from prior sessions; behavioral memory tracks what works and what doesn't.

**Core principle: standing on the shoulders of giants.** The best ML optimization builds on proven research. The plugin prioritizes finding and implementing techniques from papers before inventing from scratch. User-provided papers get highest priority. Research-implement is a first-class strategy, not a fallback.

## Overview

The ml-optimizer understands your ML model, establishes baselines, researches improvements, tunes hyperparameters, evolves code, runs experiments (parallel across GPUs), monitors for divergence, and improves each session.

**Key design decisions:**
- **Workflow-driven experiment loop**: Phases 5–8 run as dynamic workflows. The Phase 7 script holds the iteration loop and dispatches tuning, experiment, and analysis agents via `agentType` (a standalone monitor-agent dispatch exists but isn't wired into the loop — divergence polling is folded into each experiment agent instead). Analysis advises, the workflow routes the next action; the orchestrator reads the structured return between phases.
- LLM-driven hyperparameter tuning (Claude reasons about results — no Optuna/grid search)
- Research via web search + alphaxiv academic paper search + user-provided papers
- ShinkaEvolve for fine-grained evolutionary code mutations
- Structured output directory in your project (location chosen at Phase 0 — no hardcoded default)

### Key Features

Inspired by [SakanaAI ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) and [karpathy/autoresearch](https://github.com/karpathy/autoresearch):

| Feature | What it does |
|---------|-------------|
| **ShinkaEvolve** | Fine-grained evolutionary code mutations via LLM-generated SEARCH/REPLACE diff patches confined to `EVOLVE-BLOCK` regions. Dispatched by the Phase 7/Phase 8 workflow through the implement-agent |
| **Research-Implement** | Paper-informed optimization — finds techniques via web/papers, implements as git branches, tests with HP tuning |
| **Stuck Protocol** | When analysis advises stop, the Phase 7 workflow dispatches research for fresh ideas, then judges continue-or-exit from evidence (fresh proposals vs. exhausted agenda) — no fixed stop-count threshold |
| **Dead-End Catalog** | Tracks techniques conclusively unpromising. Research and hp-tune agents consult it before proposing, preventing wasted budget |
| **Research Agenda** | Living document initialized from proposals, reprioritized after each batch on experimental evidence |
| **Progress Dashboard** | Self-contained HTML dashboard with auto-refresh (`--live`), SVG timeline, sortable results, HP sensitivity, method explanations |
| **Immutable Baseline** | SHA-256 checksum of baseline metrics verified once before the experiment loop begins — halts if metrics are modified |
| **Goal Anchoring** | `optimization-goals.json` written at Phase 0; all agents read it before acting. Post-dispatch validation catches frozen param changes, scope breaches, dead-end re-proposals |
| **Cross-Session Learning** | claude-mem recalls insights from prior sessions; behavioral memory tracks what works across runs |
| **Behavioral Memory** | `learned-behaviors.json` accumulates HP constraints, method outcomes, divergence patterns. All agents have `memory: local` for persistent role-specific learning |
| **Workflow-Driven Phases 5–8** | Research, implement, experiment, and stacking run as dynamic workflows bundled in the orchestrate skill (`skills/orchestrate/workflows/phase-{5,6,7,8}-*.js`). The orchestrator launches one `Workflow({scriptPath, args})` per phase; each script holds that phase's fan-out/loop and reuses the existing agents via `agentType`. Internal pipeline steps launched by `scriptPath`, not user-facing `/commands` |
| **File/Args Handoff** | Cross-agent context inside a workflow flows through `args` + files agents write under `<exp_root>/` (results, manifests, research agenda, `batch-N-analysis.md`) + each workflow's structured return — no `SendMessage`, no `agent_registry`, no relay |
| **GitNexus Code Graph** | **Required** MCP + CLI that indexes a repo into a queryable code knowledge graph. The pipeline indexes every code repo (target + every cloned reference repo); implement/research agents must query structure, call-graph, and blast-radius before editing. No grep/analyze fallback — Phase 2 verifies it and blocks if absent |

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Claude Code** — the plugin runs inside Claude Code sessions
- **Dynamic workflows enabled** — **required** for phases 5–8. Phases 5 (research), 6 (implement), 7 (experiment loop), 8 (stacking) run as dynamic workflows (the `Workflow` tool) with **no `Agent` fallback path**. A research-preview / paid-plan feature; per-phase launch approval applies (choose "don't ask again" to run unattended). If unavailable on your plan, phases 5–8 cannot run.
- **git** — branch isolation when implementing research proposals
- **GitNexus** — **required** for code understanding. The pipeline indexes every code repo (target project + every cloned reference repo) into a queryable code knowledge graph, and implement/research agents must query it (MCP-only — via `mcp__gitnexus__context`/`query`/`impact`; no CLI-query fallback) before adapting or editing code. **No grep/analyze fallback**. Phase 2 verifies the CLI is installed and blocks if absent. Install:
  ```bash
  npm install -g gitnexus && gitnexus setup
  ```
  `gitnexus setup` auto-registers the gitnexus MCP server for Claude Code (manual MCP-registration fallback: `claude mcp add --transport stdio --scope user gitnexus gitnexus mcp`). It also installs gitnexus's own global skills (7) and PreToolUse/PostToolUse hooks into `~/.claude/`, affecting all Claude Code projects. A freshly-registered MCP server becomes available only after a session restart. Indexing is non-invasive (`gitnexus analyze --index-only`) — it doesn't modify the indexed repo's CLAUDE.md/AGENTS.md or install `.claude/` skills there. The `.gitnexus/` index artifacts are auto-excluded from git, but don't commit them.
- **Your ML project** — the plugin brings its own orchestration (stdlib only, aside from matplotlib used by plot_results.py); your training code brings its own stack (PyTorch, TensorFlow, scikit-learn, XGBoost, LightGBM, etc.)

#### MCP Servers (gitnexus required; alphaxiv/context7/claude-mem recommended)

**gitnexus is required** (see [Prerequisites](#prerequisites)); the rest are recommended — the plugin works without them but benefits from their presence. Install separately — **not** bundled with the plugin.

| MCP Server | What it enables | Used by | Required? |
|------------|-----------------|---------|-----------|
| **alphaxiv** | arXiv paper search, paper content extraction, PDF Q&A, GitHub repo exploration | research-agent (6 tools), implement-agent (2 tools) | No — falls back to WebSearch/WebFetch |
| **gitnexus** | Code knowledge graph — index a repo and query its structure, call-graph, and blast-radius (`context`/`query`/`impact`) | implement-agent (3 tools), research-agent (3 tools) | **Yes** — required; the pipeline indexes every code repo and agents must query it to understand code |
| **context7** | Framework API documentation lookup (PyTorch, TensorFlow, etc.) | research-agent, implement-agent | No — falls back to WebSearch |
| **claude-mem** | Cross-session memory — recalls past optimization sessions, avoids re-proposing failed techniques | research-agent, orchestrator | No — works without but loses cross-session learning |

**Install alphaxiv:**
```bash
claude mcp add --transport http --scope user alphaxiv https://api.alphaxiv.org/mcp/v1
```

**Install gitnexus (required):**
```bash
npm install -g gitnexus && gitnexus setup
```
See [Prerequisites](#prerequisites) for what `gitnexus setup` installs (`~/.claude/` global skills + hooks, session-restart note) and why GitNexus is required (no grep/analyze fallback; MCP-only querying; non-invasive `--index-only`).

**Install context7:** from the Claude Code marketplace via `/plugin` → Discover → search "context7".

**Install claude-mem:** install the claude-mem plugin from the marketplace; it provides the MCP server automatically.

### Installation

First, add the marketplace:

```bash
claude plugin marketplace add ChuaHanChong/ml-optimizer
```

Then install the plugin:

```bash
# Recommended: local (project-scoped, not checked into version control)
claude plugin install ml-optimizer --scope local

# Alternative: project (project-scoped, checked into version control)
claude plugin install ml-optimizer --scope project
```

After installation, run the setup scripts and reload — from inside an active Claude Code session, so `${CLAUDE_PLUGIN_ROOT}` resolves (the bare `scripts/` path only works if CWD is already the plugin's installed directory):

```bash
# Initialize submodule (ShinkaEvolve)
bash ${CLAUDE_PLUGIN_ROOT}/scripts/setup_evolve.sh
```

Run `/reload-plugin` or restart Claude Code. The `/optimize` command and all 10 agents become available automatically.

> **Why local/project?** Agent memory (`memory: local`) stores learnings in `.claude/agent-memory-local/` within the project. Local or project installation keeps plugin code, agent memory, and experiment data together — scoped to your ML project, not polluting other workspaces.

#### Development / local testing

Clone the repo into your project's `.claude/plugins/` directory:

```bash
mkdir -p .claude/plugins
git clone https://github.com/ChuaHanChong/ml-optimizer.git .claude/plugins/ml-optimizer
```

Then load the plugin directly:

```bash
claude --plugin-dir .claude/plugins/ml-optimizer --dangerously-skip-permissions
```

### Quick Start

In a Claude Code session, run the slash command:

```
/optimize my model at ./src/train.py
```

The plugin will:

1. **Discovery** — ask about your metric, target, constraints, dataset, environment
2. **Analysis** — read your code, detect the framework, present an optimization plan for approval
3. **Baseline** — establish current metrics with unmodified code
4. **Optimization loop** — research techniques → implement as git branches → run experiments → analyze → repeat
5. **Report** — final report at `<exp_root>/reports/final-report.md` and live dashboard at `<exp_root>/reports/dashboard.html` (auto-refreshed during optimization)

Or invoke the skill directly without the slash command:

```
Use the ml-optimizer:orchestrate skill to improve my training loop
```

## Components

### Skills

Only `orchestrate` is user-facing (via `/optimize`). Other skills preload into agents via the `skills:` array in their definitions and are read automatically on dispatch.

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
| `evolve` | Orchestrates evolutionary code refinement via full ShinkaEvolve pipeline (convert → run → inspect) | Internal |
| `shinka-setup` | Scaffold a new ShinkaEvolve task (`evaluate.py` + `initial.<ext>`) from a task description | Internal |
| `shinka-convert` | Convert ML project into ShinkaEvolve task format with EVOLVE-BLOCK markers | Internal |
| `shinka-run` | Run ShinkaEvolve evolution with file-based LLM handoff (`SHINKA_PROVIDER=claude_code`) | Internal |
| `shinka-inspect` | Extract/rank top-performing mutations from ShinkaEvolve results into a Markdown bundle for iteration planning | Internal |

### Agent Definitions

Ten agent types in `agents/`. The plugin ships `settings.json` with `"agent": "ml-optimizer:orchestrator-agent"` — when enabled, the orchestrator agent becomes the main thread and auto-starts Phase 0.

| Agent | Tools | Model | Effort | Preloaded Skill |
|-------|-------|-------|--------|-----------------|
| **`orchestrator-agent`** | Agent, Workflow, Read, Write, Edit, Bash, Glob, Grep, Skill, WebSearch, WebFetch | **opus[1m]** | xhigh | `ml-optimizer:orchestrate` + verification |
| `research-agent` | WebSearch, WebFetch, Read, Write, Bash, Glob, Grep, Skill + alphaxiv MCP (6) + gitnexus MCP (3) | opus[1m] | xhigh | `ml-optimizer:research` + mem-search + verification |
| `implement-agent` | Bash, Read, Write, Edit, LSP, Glob, Grep, Skill, WebSearch, WebFetch + alphaxiv MCP (2) + gitnexus MCP (3) | opus[1m] | xhigh | `ml-optimizer:implement` + evolve + shinka-* + debugging + verification + karpathy-guidelines |
| `tuning-agent` | Read, Write, Bash, Glob, Grep, Skill, WebSearch, WebFetch | opus[1m] | xhigh | `ml-optimizer:hp-tune` + mem-search + verification |
| `analysis-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | opus[1m] | xhigh | `ml-optimizer:analyze` + mem-search + verification |
| `report-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | opus[1m] | xhigh | `ml-optimizer:report` + verification |
| `baseline-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | sonnet[1m] | medium | `ml-optimizer:baseline` |
| `monitor-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | sonnet[1m] | medium | `ml-optimizer:monitor` |
| `experiment-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | sonnet[1m] | medium | `ml-optimizer:experiment` |
| `prerequisites-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | sonnet[1m] | medium | `ml-optimizer:prerequisites` |

The **orchestrator-agent** is the main-thread agent (activated by `settings.json`); the rest are specialized workers — see [`.claude/CLAUDE.md`](.claude/CLAUDE.md) for dispatch (`Agent()` for phases 0/1/2/3/4/9, `agentType` inside the phase-5–8 workflows). Analytical agents use `effort: xhigh` + `model: opus[1m]`; procedural agents use `effort: medium` + `model: sonnet[1m]`. The `[1m]` suffix requests the 1M-token context window (Opus auto-upgrades to 1M on Max/Team/Enterprise; Sonnet 1M may consume usage credits). `xhigh` is supported on Opus 4.8/4.7 and falls back to the highest supported level on older Opus.

### Python Utilities

All scripts in `scripts/` use only the standard library, except `plot_results.py` which requires `matplotlib` (imported at module level) for its progress-chart (PNG) output, and work as both importable modules and CLI tools. [`.claude/CLAUDE.md`](.claude/CLAUDE.md) is the source of truth for this table — keep this copy in sync with it.

| Script | CLI Usage |
|--------|-----------|
| `scripts/gpu_check.py` | `python3 scripts/gpu_check.py` |
| `scripts/parse_logs.py` | `python3 scripts/parse_logs.py <logfile>` — auto-detects and parses 8 log formats: JSON, Python `logging`, tqdm, XGBoost/LightGBM, SB3/rsl_rl, HuggingFace Trainer, CSV, kv |
| `scripts/detect_divergence.py` | `python3 scripts/detect_divergence.py '<json_values>' [--higher-is-better] [--model-category rl\|generative\|supervised] [--reward-collapse-fraction F] [--reward-collapse-patience N]` — also: `--check-overfitting '<train_json>' '<val_json>' [--patience N] [--min-gap F]` |
| `scripts/result_analyzer.py` | `python3 scripts/result_analyzer.py <results_dir> <metric> [baseline_id] [lower_is_better]` — also: `compare <exp_id_1> <exp_id_2> [metric]` |
| `scripts/experiment_setup.py` | `python3 scripts/experiment_setup.py <exp_root> <train_command> [gpu_id] [config_json] [round_dir] [env_vars_json]` |
| `scripts/implement_utils.py` | `python3 scripts/implement_utils.py <findings.md> '<indices_json>'` — also: `clone <url> <dest>`, `analyze <path>`, `diff <project_root> <branch>` |
| `scripts/gitnexus_utils.py` | `python3 scripts/gitnexus_utils.py available\|mcp-registered\|require\|index <path> [--force]\|is-indexed <path>` — required GitNexus code-graph helper. `available` checks the CLI is on PATH; `mcp-registered` reports whether the gitnexus MCP server is registered (`{"registered": true\|false\|null}`, always exits 0); `require` reports availability plus `mcp_registered` and exits nonzero only when the CLI is absent; `index` runs `gitnexus analyze <path> --index-only` (non-invasive — no edits to the indexed repo's CLAUDE.md/AGENTS.md, no `.claude/` skill install; auto-adds `.gitnexus/` to git exclude; skips already-indexed paths unless `--force`); `is-indexed` checks for an existing index. Never raises — failures via the returned dict. Skills treat `available()==False` / index `success: false` as a hard error (halt with install/repair guidance), not a fallback |
| `scripts/pipeline_state.py` | `python3 scripts/pipeline_state.py <exp_root> validate\|save\|load\|cleanup\|verify-baseline\|gate\|log-gate\|log-decision\|replay-check\|decisions` |
| `scripts/schema_validator.py` | `python3 scripts/schema_validator.py <filepath> result\|baseline\|manifest\|prerequisites\|hp_proposal\|rounds_manifest [--strict]` — also: `relay <route> <json>` validates the file/args payloads handed off between workflow stages (6 routes). `--strict` enforces completeness |
| `scripts/plot_results.py` | `python3 scripts/plot_results.py <results_dir> <metric> comparison\|timeline\|sensitivity <hp>\|progress [--higher-is-better]` |
| `scripts/prerequisites_check.py` | `python3 scripts/prerequisites_check.py scan-imports\|check-packages\|detect-env\|detect-format\|detect-format-project\|validate-data\|bulk-install-cmd\|gpu-install-cmd` |
| `scripts/error_tracker.py` | `python3 scripts/error_tracker.py <exp_root> log\|show\|patterns\|summary\|success\|proposals\|rank\|log-suggestion\|suggestion-history\|dead-end <add\|list\|check>\|agenda <init\|update\|list\|add>` |
| `scripts/dashboard.py` | `python3 scripts/dashboard.py <exp_root> [--live] [--table] [--serve --port 8080]` — HTML dashboard + Markdown results table |
| `scripts/excalidraw_gen.py` | `python3 scripts/excalidraw_gen.py <exp_root> pipeline\|comparison\|hp-landscape\|architecture <args>` — Excalidraw JSON diagrams |
| `scripts/goal_memory.py` | `python3 scripts/goal_memory.py <exp_root> init-goals\|update-goals\|read-goals\|log-behavior\|query-behaviors\|validate-output\|summary\|sync-from-errors` — goal anchoring, behavioral memory, agent output validation |
| `scripts/setup_evolve.sh` | `bash scripts/setup_evolve.sh` — initialize ShinkaEvolve submodule and create skill symlinks for auto-discovery |
| `scripts/round_manager.py` | `python3 scripts/round_manager.py <exp_root> create-round\|current-round\|register-experiment\|close-round\|next-id\|check-baseline\|check-prerequisites\|check-manifest\|check-round\|check-proposals` — round lifecycle + completeness checking |
| `scripts/output_contract.py` | `python3 scripts/output_contract.py inject\|check <exp_root> <agent_name> [--round-dir X] [--exp-id X]` — single source of truth for per-agent output contracts. `inject` renders human-readable contract text for the SubagentStart hook; `check` verifies that all required outputs exist for the SubagentStop hook. Supports `glob`, `dir`, `any_of` (mode-dependent outputs), and `required_if` (conditional outputs driven by another JSON file) |
| `scripts/validate_experiment_write.py` | PreToolUse hook — blocks Write/Edit operations to experiment result JSONs when they fail schema validation, are written to the wrong directory (non-round subdirs under `results/`), miss completeness fields for `status: completed` (`iteration`, `method_tier`, `duration_seconds`, `eval_protocol`) or stacked entries (`code_branches`, `stacking_order`), violate frozen-parameter constraints from `optimization-goals.json`, or exceed the OOM batch-size cap recorded in `learned-behaviors.json`. Placeholder writes with `status: running\|pending` are exempt. Called by the harness; no direct CLI |
| `scripts/validate_agent_output.py` | SubagentStop hook — reads the stdin hook context, looks up the agent's contract via `output_contract.py`, and blocks the agent from finishing if any required output file is missing. Called by the harness; no direct CLI |

### Hooks (Safety Guardrails)

Lifecycle hooks in `hooks/` protect against common failure modes during autonomous operation.

**Output-structure enforcement uses a 3-checkpoint model** on every agent dispatch:

1. **SubagentStart** (`subagent-start-inject-goals.sh`) — injects the agent's output contract (exact paths, schemas, examples) into its prompt before work begins, so it can't claim it didn't know what to produce.
2. **PreToolUse Write/Edit** (`validate_experiment_write.py`) — blocks invalid writes to experiment result JSONs while the agent works, so malformed outputs never land on disk.
3. **SubagentStop** (`validate_agent_output.py`) — verifies every required output exists after work finishes, blocking completion if anything is missing.

All three layers read the same per-agent contract from `scripts/output_contract.py`, the single source of truth. The contract supports regular paths, `glob` patterns, `dir` entries, `any_of` (mode-dependent outputs — e.g., analysis-agent produces either `batch-<N>-analysis.md` or `session-review.md`), and `required_if` (conditional outputs driven by another JSON file — e.g., prerequisites-agent produces `prepared-data/` only when `dataset.prepared == true` in `prerequisites.json`). JSON schemas are enforced at runtime by `scripts/schema_validator.py`, used by layer 2.

| Hook | Event | Type | Purpose |
|------|-------|------|---------|
| `bash-safety.sh` | PreToolUse (Bash) | command | Blocks `rm -rf /`, `git push --force`, `curl\|bash`, `chmod 777`, etc. |
| `file-guardrail.sh` | PreToolUse (Write/Edit) | command | Blocks writes to `.git/`, `.env`, credentials, lock files |
| `validate_experiment_write.py` | PreToolUse (Write/Edit) | command | Layer 2 of the output-structure enforcement: blocks Write/Edit to experiment result JSONs that fail schema validation, land outside a round subdirectory, miss mandatory completeness fields (`iteration`/`method_tier`/`duration_seconds`/`eval_protocol` for `completed`; `code_branches`/`stacking_order` for stacked; `notes` for failed/diverged), violate frozen parameters from `optimization-goals.json`, or exceed OOM batch-size caps from `learned-behaviors.json`. Placeholder writes (`status: running\|pending`) are exempt |
| `detect-critical-errors.sh` | PostToolUse + PostToolUseFailure (Bash) | command | Detects CUDA OOM, segfault, disk full; logs to error tracker |
| `pre-compact.sh` | PreCompact | command | Saves pipeline state snapshot to dev_notes before context compaction |
| `post-compact-context.sh` | SessionStart (compact) | command | Re-injects phase/status/iteration/metric/experiment-count context after compaction |
| `validate_agent_output.py` | SubagentStop | command | Layer 3 of the output-structure enforcement: looks up the agent's contract in `output_contract.py` and blocks the agent from finishing if any required output file is missing |
| `subagent-start-inject-goals.sh` | SubagentStart | command | Layer 1 of the output-structure enforcement: injects the optimization-goals summary from `goal_memory.py` AND the per-agent output contract from `output_contract.py inject` into the agent's prompt — so every agent sees its exact required output paths, schema, and `dev_notes.md` logging instructions before it starts work |
| `file-changed-pipeline-state.sh` | FileChanged (pipeline-state.json) | command | Detects external corruption of pipeline state |
| `cwd-changed-detect-experiments.sh` | CwdChanged | command | Auto-detects existing `<exp_root>` and offers to resume (mid-run) or start a new run (if phase 9) |
| `stop-check.sh` | Stop | command | Verifies a final report exists before the session ends when `pipeline-state.json` indicates experiments were run |

Exit code `2` = block action, `0` = allow. Configured in `hooks/hooks.json`.

### Evolutionary Submodule

The plugin integrates ShinkaEvolve as a git submodule. Setup is handled during [Installation](#installation).

#### ShinkaEvolve (SakanaAI)

[ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) provides fine-grained evolutionary code mutations, dispatched by the Phase 7/Phase 8 workflow via the implement-agent's evolve skill (`Skill("ml-optimizer:evolve")`) when HP tuning shows diminishing returns. Full pipeline: `shinka-convert` → `shinka-run` (file-based LLM handoff, `SHINKA_PROVIDER=claude_code`, `SHINKA_HANDOFF_TIMEOUT=600`) → `shinka-inspect` → commit. The agent writes `.inprogress` markers when picking up mutation requests, and `shinka_run` writes `.heartbeat` files for liveness — so the handoff survives even when mutation generation takes 30-60 seconds.

Exposes 4 Claude Code skills: `shinka-setup`, `shinka-convert`, `shinka-run`, `shinka-inspect`.

## How It Works

All nine worker agents are **fresh spawns** — `Agent()` for the interactive/trivial phases (0/1, 2, 3, 4, 9) and the workflow runtime's `agent({agentType})` for phases 5–8. No persistent-agent / `SendMessage` / `agent_registry` resumption; cross-agent context flows via files + args, role knowledge persists via `memory: local`.

The full architecture — phase-by-phase pipeline, dispatch chain & per-phase output map, `<exp_root>/` layout, every design pattern and gotcha, and the multi-run breadcrumb — lives in **[`.claude/CLAUDE.md`](.claude/CLAUDE.md)**, the architecture source of truth. A visual ASCII flow is in [`docs/workflow-diagram.txt`](docs/workflow-diagram.txt).

## Development

### Test Dependencies

The plugin's own scripts use only the Python standard library, except `plot_results.py` (matplotlib). Running the test suite requires:

```bash
pip install pytest              # test runner
pip install torch torchvision   # used by bundled test fixtures (tests/fixtures/tiny_resnet_cifar10/)
pip install pyyaml              # YAML config parsing in fixtures
pip install matplotlib          # required to import/run scripts/plot_results.py and its tests
```

### Running Tests

```bash
cd <plugin-directory>
python3 -m pytest tests/ -v                           # all tests (~1400, ~8 minutes)
python3 -m pytest tests/test_parsing.py -v            # single file
python3 -m pytest tests/ -m "not slow" -v             # skip real training tests (fast)
python3 -m pytest tests/test_e2e_pipeline.py -m slow  # real end-to-end training only
```

The `slow` marker identifies tests that run real training (see `@pytest.mark.slow` in `tests/test_e2e_pipeline.py`). Most tests are unit/integration with no GPU requirement. No build step, no linter. Python 3.10+ required.

## License

MIT License. See [LICENSE](LICENSE) for details.
