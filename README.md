# ml-optimizer

An autonomous ML model optimization plugin. Powered by Claude Code.

> Autonomous by default — runs until the goal is reached or you stop it. No hardcoded thresholds. No fixed pipeline. The orchestrator dispatches specialized agents, the analysis agent advises direction, and evidence drives decisions.

## Why This Approach?

Traditional ML optimization tools follow a fixed pipeline — same strategy every time regardless of the task. The ml-optimizer takes a different approach:

1. **Optimizes your ML model** — using research-informed techniques, LLM-driven hyperparameter tuning, ShinkaEvolve evolutionary code mutations, and parallel GPU experiments

2. **Adapts its strategy based on evidence** — the analysis agent evaluates results after each batch and recommends the next action (continue, pivot approach, or stop). The orchestrator acts on these recommendations directly.

3. **Gets better with each session** — claude-mem recalls insights from prior sessions. Behavioral memory tracks what works and what doesn't.

**Core principle: Standing on the shoulders of giants.** The best ML optimization comes from building on proven research. The plugin prioritizes finding and implementing techniques from papers before inventing from scratch. User-provided papers get highest priority. Research-implement is a first-class strategy, not a fallback.

## Overview

The ml-optimizer understands your ML model, establishes baselines, researches improvements, tunes hyperparameters, evolves code, runs experiments (in parallel across GPUs), monitors for training divergence, and gets better at optimizing with each session.

**Key design decisions:**
- **Orchestrator-driven experiment loop**: The orchestrator dispatches tuning, experiment, monitor, and analysis agents per iteration. Analysis advises, orchestrator routes the next action.
- LLM-driven hyperparameter tuning (Claude reasons about results — no Optuna/grid search)
- Research via web search + alphaxiv academic paper search + user-provided papers
- ShinkaEvolve for fine-grained evolutionary code mutations
- Structured output directory in your project (location chosen at Phase 0 — the plugin has no hardcoded default)

### Key Features

Inspired by [SakanaAI ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) and [karpathy/autoresearch](https://github.com/karpathy/autoresearch):

| Feature | What it does |
|---------|-------------|
| **Orchestrator-Driven Loop** | The orchestrator drives Phase 7 (experiments) ↔ Phase 8 (stacking), dispatching agents per iteration. Analysis advises direction, orchestrator routes action |
| **ShinkaEvolve** | Fine-grained evolutionary code mutations via AST-level changes. Dispatched by the orchestrator through the implement-agent |
| **Research-Implement** | Paper-informed optimization — finds techniques via web/papers, implements as git branches, tests with HP tuning |
| **Stuck Protocol** | When analysis advises stop, orchestrator dispatches research for fresh ideas, then judges whether to continue or exit from the evidence (fresh proposals vs. exhausted agenda) — no fixed stop-count threshold |
| **Dead-End Catalog** | Tracks techniques conclusively shown to be unpromising. Research and hp-tune agents consult it before proposing, preventing wasted budget |
| **Research Agenda** | Living document initialized from proposals, reprioritized after each batch based on experimental evidence |
| **Progress Dashboard** | Self-contained HTML dashboard with auto-refresh (`--live`), SVG timeline, sortable results, HP sensitivity, method explanations |
| **Immutable Baseline** | SHA-256 checksum of baseline metrics verified before each batch — halts if metrics are modified |
| **Goal Anchoring** | `optimization-goals.json` written at Phase 0; all agents read it before acting. Post-dispatch validation catches frozen param changes, scope breaches, dead-end re-proposals |
| **Cross-Session Learning** | claude-mem recalls insights from prior sessions. Behavioral memory tracks what works across runs |
| **Behavioral Memory** | `learned-behaviors.json` accumulates HP constraints, method outcomes, divergence patterns. All agents have `memory: local` for persistent role-specific learning |
| **Resumable Subagents** | 5 persistent agents (research, implement, tuning, analysis, monitor) resumed via `SendMessage` — preserving accumulated context across the pipeline |
| **Inter-Agent Relay** | Orchestrator relays findings between agents via `CONTEXT FROM OTHER AGENTS:` sections — analyze findings reach hp-tune, monitor OOM info reaches hp-tune, research proposals reach implement |

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Claude Code** — the plugin runs inside Claude Code sessions
- **git** — used for branch isolation when implementing research proposals
- **Your ML project** — the plugin brings its own orchestration (stdlib only); your training code brings its own stack (PyTorch, TensorFlow, scikit-learn, XGBoost, LightGBM, etc.)

#### MCP Servers (Recommended)

These MCP servers enhance the plugin's capabilities. The plugin works without them but benefits significantly from their presence. Install them separately — they are **not** bundled with the plugin.

| MCP Server | What it enables | Used by |
|------------|-----------------|---------|
| **alphaxiv** | arXiv paper search, paper content extraction, PDF Q&A, GitHub repo exploration | research-agent (6 tools), implement-agent (2 tools) |
| **context7** | Framework API documentation lookup (PyTorch, TensorFlow, etc.) | research-agent, implement-agent |
| **claude-mem** | Cross-session memory — recalls past optimization sessions, avoids re-proposing failed techniques | research-agent, orchestrator |

**Install alphaxiv:**
```bash
claude mcp add --transport http --scope user alphaxiv https://api.alphaxiv.org/mcp/v1
```

**Install context7:** Install from the official Claude Code marketplace via `/plugin` → Discover → search for "context7".

**Install claude-mem:** Install the claude-mem plugin from the marketplace, which provides the MCP server automatically.

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

After installation, run the setup scripts and reload:

```bash
# Initialize submodule (ShinkaEvolve)
bash scripts/setup_evolve.sh
```

Run `/reload-plugin` or restart Claude Code. The `/optimize` command and all 10 agents will be available automatically.

> **Why local/project?** Agent memory (`memory: local`) stores learnings in `.claude/agent-memory-local/` within the project. Local or project-based installation keeps plugin code, agent memory, and experiment data together — scoped to your ML project, not polluting other workspaces.

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
2. **Analysis** — read your code, detect the framework, present an optimization plan for your approval
3. **Baseline** — establish current metrics with the unmodified code
4. **Optimization loop** — research techniques → implement as git branches → run experiments → analyze → repeat
5. **Report** — final report at `<exp_root>/reports/final-report.md` and live progress dashboard at `<exp_root>/reports/dashboard.html` (auto-refreshed during optimization)

Or invoke the skill directly without the slash command:

```
Use the ml-optimizer:orchestrate skill to improve my training loop
```

## Components

### Skills

Only `orchestrate` is user-facing (invoked via `/optimize`). Other skills are preloaded into agents via the `skills:` array in their agent definitions and read automatically on dispatch.

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
| `shinka-setup` | Initialize ShinkaEvolve environment and dependencies | Internal |
| `shinka-convert` | Convert ML project into ShinkaEvolve task format with EVOLVE-BLOCK markers | Internal |
| `shinka-run` | Run ShinkaEvolve evolution with file-based LLM handoff (`SHINKA_PROVIDER=claude_code`) | Internal |
| `shinka-inspect` | Extract best mutation from ShinkaEvolve results and commit to branch | Internal |

### Agent Definitions

Ten agent types in `agents/`. The plugin ships `settings.json` with `"agent": "ml-optimizer:orchestrator-agent"` — when the plugin is enabled, the orchestrator agent becomes the main thread and auto-starts Phase 0.

| Agent | Tools | Model | Effort | Preloaded Skill |
|-------|-------|-------|--------|-----------------|
| **`orchestrator-agent`** | Agent, Read, Write, Edit, Bash, Glob, Grep, Skill, WebSearch, WebFetch | **opus[1m]** | xhigh | `ml-optimizer:orchestrate` + verification |
| `research-agent` | WebSearch, WebFetch, Read, Write, Bash, Glob, Grep, Skill + alphaxiv MCP (6) | opus[1m] | xhigh | `ml-optimizer:research` + mem-search + verification |
| `implement-agent` | Bash, Read, Write, Edit, LSP, Glob, Grep, Skill, WebSearch, WebFetch + alphaxiv MCP (2) | opus[1m] | xhigh | `ml-optimizer:implement` + evolve + shinka-* + debugging + verification + karpathy-guidelines |
| `tuning-agent` | Read, Write, Bash, Glob, Grep, Skill, WebSearch, WebFetch | opus[1m] | xhigh | `ml-optimizer:hp-tune` + mem-search + verification |
| `analysis-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | opus[1m] | xhigh | `ml-optimizer:analyze` + mem-search + verification |
| `report-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | opus[1m] | xhigh | `ml-optimizer:report` + verification |
| `baseline-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | sonnet[1m] | medium | `ml-optimizer:baseline` |
| `monitor-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | sonnet[1m] | medium | `ml-optimizer:monitor` |
| `experiment-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | sonnet[1m] | medium | `ml-optimizer:experiment` |
| `prerequisites-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | sonnet[1m] | medium | `ml-optimizer:prerequisites` |

The **orchestrator-agent** is the main-thread agent (activated by `settings.json`). It drives the experiment loop directly (Phase 7 and Phase 8). Other agents are specialized workers coordinated through the orchestrator relay. Analytical agents use `effort: xhigh` and `model: opus[1m]`. Procedural agents use `effort: medium` and `model: sonnet[1m]`. The `[1m]` suffix requests the 1M-token context window (Opus auto-upgrades to 1M on Max/Team/Enterprise; Sonnet 1M may consume usage credits). `xhigh` is supported on Opus 4.8/4.7 and falls back to the highest supported level on older Opus.

### Python Utilities

All scripts in `scripts/` use only the standard library and work as both importable modules and CLI tools:

| Script | CLI Usage |
|--------|-----------|
| `scripts/gpu_check.py` | `python3 scripts/gpu_check.py` |
| `scripts/parse_logs.py` | `python3 scripts/parse_logs.py <logfile>` — parses kv/JSON/CSV/XGBoost/HuggingFace Trainer logs |
| `scripts/detect_divergence.py` | `python3 scripts/detect_divergence.py '<json_values>' [--higher-is-better] [--model-category rl\|generative\|supervised]` — also: `--check-overfitting '<train_json>' '<val_json>' [--patience N] [--min-gap F]` |
| `scripts/result_analyzer.py` | `python3 scripts/result_analyzer.py <results_dir> <metric> [baseline_id] [lower_is_better]` — also: `compare <exp_id_1> <exp_id_2> [metric]` |
| `scripts/experiment_setup.py` | `python3 scripts/experiment_setup.py <project_root> <train_command> [gpu_id] [config_json]` |
| `scripts/implement_utils.py` | `python3 scripts/implement_utils.py <findings.md> '<indices_json>'` — also: `clone <url> <dest>`, `analyze <path>`, `diff <project_root> <branch>` |
| `scripts/pipeline_state.py` | `python3 scripts/pipeline_state.py <exp_root> validate\|save\|load\|cleanup\|verify-baseline\|gate\|log-gate\|log-decision\|replay-check\|decisions` |
| `scripts/schema_validator.py` | `python3 scripts/schema_validator.py <filepath> result\|baseline\|manifest\|prerequisites [--strict]` — also: `relay <route> <json>` for inter-agent relay validation. `--strict` enforces completeness |
| `scripts/plot_results.py` | `python3 scripts/plot_results.py <results_dir> <metric> comparison\|timeline\|sensitivity <hp>\|progress [--higher-is-better]` |
| `scripts/prerequisites_check.py` | `python3 scripts/prerequisites_check.py scan-imports\|check-packages\|detect-env\|detect-format\|detect-format-project\|validate-data\|bulk-install-cmd\|gpu-install-cmd` |
| `scripts/error_tracker.py` | `python3 scripts/error_tracker.py <exp_root> log\|show\|patterns\|summary\|sync\|success\|proposals\|rank\|cleanup\|log-suggestion\|suggestion-history\|dead-end <add\|list\|check>\|agenda <init\|update\|list\|add>` |
| `scripts/dashboard.py` | `python3 scripts/dashboard.py <exp_root> [--live] [--table] [--serve --port 8080]` — HTML dashboard + Markdown results table |
| `scripts/excalidraw_gen.py` | `python3 scripts/excalidraw_gen.py <exp_root> pipeline\|comparison\|hp-landscape\|architecture <args>` — Excalidraw JSON diagrams |
| `scripts/goal_memory.py` | `python3 scripts/goal_memory.py <exp_root> init-goals\|read-goals\|log-behavior\|query-behaviors\|validate-output\|summary\|sync-from-errors` — goal anchoring, behavioral memory, agent output validation |
| `scripts/setup_evolve.sh` | `bash scripts/setup_evolve.sh` — initialize ShinkaEvolve submodule and create skill symlinks for auto-discovery |
| `scripts/round_manager.py` | `python3 scripts/round_manager.py <exp_root> create-round\|current-round\|register-experiment\|close-round\|next-id\|check-baseline\|check-prerequisites\|check-manifest\|check-round\|check-proposals` — round lifecycle + completeness checking |
| `scripts/output_contract.py` | `python3 scripts/output_contract.py inject\|check <exp_root> <agent_name> [--round-dir X] [--exp-id X]` — single source of truth for per-agent output contracts. `inject` renders human-readable contract text for the SubagentStart hook; `check` verifies that all required outputs exist for the SubagentStop hook. Supports `glob`, `dir`, `any_of` (mode-dependent outputs), and `required_if` (conditional outputs driven by another JSON file) |
| `scripts/validate_experiment_write.py` | PreToolUse hook — blocks Write/Edit operations to experiment result JSONs when they fail schema validation, are written to the wrong directory (non-round subdirs under `results/`), miss completeness fields for `status: completed` (`iteration`, `method_tier`, `duration_seconds`) or stacked entries (`code_branches`, `stacking_order`), violate frozen-parameter constraints from `optimization-goals.json`, or exceed the OOM batch-size cap recorded in `learned-behaviors.json`. Placeholder writes with `status: running\|pending` are exempt. Called by the harness; no direct CLI |
| `scripts/validate_agent_output.py` | SubagentStop hook — reads the stdin hook context, looks up the agent's contract via `output_contract.py`, and blocks the agent from finishing if any required output is missing. Also checks that the last entry in `dev_notes.md` has a matching `agent_id` (stateless correlation). Called by the harness; no direct CLI |

### Hooks (Safety Guardrails)

Lifecycle hooks in `hooks/` protect against common failure modes during autonomous operation.

**Output-structure enforcement uses a 3-checkpoint model** that runs on every agent dispatch:

1. **SubagentStart** (`subagent-start-inject-goals.sh`) — injects the agent's output contract (exact paths, schemas, examples) into its prompt before work begins, so the agent cannot claim it didn't know what to produce.
2. **PreToolUse Write/Edit** (`validate_experiment_write.py`) — blocks invalid writes to experiment result JSONs while the agent is working, so malformed outputs never land on disk.
3. **SubagentStop** (`validate_agent_output.py`) — verifies every required output exists after work finishes, blocking the agent from completing if anything is missing.

All three layers read the same per-agent contract from `scripts/output_contract.py`, which is the single source of truth. The contract supports regular paths, `glob` patterns, `dir` entries, `any_of` (mode-dependent outputs — e.g., analysis-agent produces either `batch-<N>-analysis.md` or `session-review.md`), and `required_if` (conditional outputs driven by another JSON file — e.g., prerequisites-agent produces `prepared-data/` only when `dataset.prepared == true` in `prerequisites.json`). JSON schemas are enforced at runtime by `scripts/schema_validator.py`, used by layer 2.

| Hook | Event | Type | Purpose |
|------|-------|------|---------|
| `bash-safety.sh` | PreToolUse (Bash) | command | Blocks `rm -rf /`, `git push --force`, `curl\|bash`, `chmod 777`, etc. |
| `file-guardrail.sh` | PreToolUse (Write/Edit) | command | Blocks writes to `.git/`, `.env`, credentials, lock files |
| `validate_experiment_write.py` | PreToolUse (Write/Edit) | command | Layer 2 of the output-structure enforcement: blocks Write/Edit to experiment result JSONs that fail schema validation, land outside a round subdirectory, miss mandatory completeness fields (`iteration`/`method_tier`/`duration_seconds` for `completed`; `code_branches`/`stacking_order` for stacked; `notes` for failed/diverged), violate frozen parameters from `optimization-goals.json`, or exceed OOM batch-size caps from `learned-behaviors.json`. Placeholder writes (`status: running\|pending`) are exempt |
| `detect-critical-errors.sh` | PostToolUse + PostToolUseFailure (Bash) | command | Detects CUDA OOM, segfault, disk full; logs to error tracker |
| `pre-compact.sh` | PreCompact | command | Saves pipeline state snapshot to dev_notes before context compaction |
| `post-compact-context.sh` | SessionStart (compact) | command | Re-injects phase/metric/budget context after compaction |
| `validate_agent_output.py` | SubagentStop | command | Layer 3 of the output-structure enforcement: looks up the agent's contract in `output_contract.py` and blocks the agent from finishing if any required output file is missing. Also enforces `dev_notes.md` updates by comparing the last entry's embedded `agent_id` against the current invocation |
| `subagent-start-inject-goals.sh` | SubagentStart | command | Layer 1 of the output-structure enforcement: injects the optimization-goals summary from `goal_memory.py` AND the per-agent output contract from `output_contract.py inject` into the agent's prompt — so every agent sees its exact required output paths, schema, and `dev_notes.md` logging instructions before it starts work |
| `file-changed-pipeline-state.sh` | FileChanged (pipeline-state.json) | command | Detects external corruption of pipeline state |
| `cwd-changed-detect-experiments.sh` | CwdChanged | command | Auto-detects existing `<exp_root>` and offers to resume (mid-run) or start a new run (if phase 9) |
| `stop-check.sh` | Stop | command | Verifies a final report exists before the session ends when `pipeline-state.json` indicates experiments were run |

Exit code `2` = block action. Exit code `0` = allow. Configured in `hooks/hooks.json`.



### Evolutionary Submodule

The plugin integrates ShinkaEvolve as a git submodule. Setup is handled during [Installation](#installation).

#### ShinkaEvolve (SakanaAI)

[ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) provides fine-grained evolutionary code mutations. Dispatched by the orchestrator via the implement-agent when HP tuning shows diminishing returns.

The orchestrator dispatches ShinkaEvolve via the implement-agent's evolve skill (`Skill("ml-optimizer:evolve")`). The full pipeline: `shinka-convert` → `shinka-run` (file-based LLM handoff, `SHINKA_PROVIDER=claude_code`, `SHINKA_HANDOFF_TIMEOUT=600`) → `shinka-inspect` → commit. The agent writes `.inprogress` markers when picking up mutation requests, and `shinka_run` writes `.heartbeat` files for liveness — ensuring the handoff survives even when mutation generation takes 30-60 seconds.

Exposes 4 Claude Code skills: `shinka-setup`, `shinka-convert`, `shinka-run`, `shinka-inspect`.

## How It Works

### Workflow

5 agents are **persistent** (resumed via `SendMessage` with accumulated context), 4 are **ephemeral** (fresh spawn). The orchestrator drives Phase 7 and Phase 8 directly and relays findings between agents.

```
0+1. Discovery & Planning (plan mode — multi-round refinement until user approves)
     Q&A → write goals → analyze codebase → present plan → user refines → repeat
2. Prerequisites (validate dataset, prepare data, install deps)          [ephemeral]
3. Establish baseline                                                    [ephemeral]
4. User checkpoint: review baseline, choose direction
5. Research (alphaxiv + web search + LLM knowledge for techniques)       [persistent]
6. Implement proposals (creates git branches, applies + validates)       [persistent]
7. Orchestrator-Driven Experiment Loop (autonomous):
   -> Each iteration:
      hp-tune → experiments (parallel) → monitor → analyze → decide
   -> Pivot types: branch_test, hp_expand, narrow_space, regularization,
      code_evolution (ShinkaEvolve), method_proposal, method_stacking
   -> Stuck protocol: research for fresh ideas after stop recommendation
   -> Exit: target reached, user stops, or orchestrator judges approaches exhausted
8. Method stacking (orchestrator driven):
   -> Analysis advises stacking, orchestrator ranks by improvement
   -> Sequentially merges best methods, skip-on-failure
   -> Analysis agent loops evolve + HP-tune until improvement or stop
9. Report & Review:
   -> Generate final report                                              [ephemeral]
   -> Session review (what worked, what didn't, how to improve)          [persistent]
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
  │   │   ├─ tuning-agent (SendMessage)              → proposed-configs/round-N-<type>/exp-*.json
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
│   └── round-N-<type>/                   — Types: hp, evolved, research, stacked
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

#### Multi-Run Support

Each `<exp_root>` is one optimization run. For a new direction on the same project, point `<exp_root>` at a new directory at Phase 0 — state files are not run-namespaced. The `.claude/ml-optimizer.json` breadcrumb tracks all runs:

```json
{"active": "/home/user/runs/02-augmentation", "runs": ["/home/user/runs/01-label-smoothing", "/home/user/runs/02-augmentation"]}
```

Hooks read `active` to resolve the current run. Old single-run format (`{"exp_root": "..."}`) is still supported. Phase 0 detects completed runs (`phase == 9`) and prompts the user to start a new directory.

## Reference

### Key Design Patterns

- **Non-git fallback**: If the project isn't a git repo, file backups replace branch isolation. Experiments run sequentially.
- **Metric routing**: Monitor/divergence always uses loss. Analyze/hp-tune use the user's `primary_metric`.
- **OOM feedback loop**: When experiments OOM, batch size is recorded. Next hp-tune call receives `max_batch_size` to avoid re-proposing failing configs.
- **All-diverge recovery**: If all experiments in a batch diverge, a recovery batch with halved learning rates runs before stopping.
- **Research cadence**: When method proposals are enabled, research triggers every N batches. If no new proposals found, cadence doubles (exponential backoff).
- **Pipeline resumption**: `pipeline-state.json` persists phase, user choices, and stop count. On restart, stale experiments are cleaned up and phase gates prevent cascading failures.
- **Loop exit conditions**: The experiment loop is autonomous — runs until: (1) target metric achieved, (2) user manually stops, or (3) the orchestrator judges approaches are exhausted (stuck protocol returns no fresh, non-dead-end proposals, the metric is flat, and the research agenda is empty). There is no hardcoded stop-count threshold — exit is the orchestrator's evidence-based decision. `consecutive_stop_count` is persisted as one input to that judgment.
- **Three-tier result tracking**: Experiments carry `method_tier` (baseline / method_default_hp / method_tuned_hp) and `proposal_source` (paper / llm_knowledge) for attribution analysis.
- **Method stacking**: After independent method testing, top methods are sequentially merged. Clean merges proceed; conflicts are LLM-resolved. Degrading combinations are skipped. After each successful stack step, the analysis agent assesses whether methods are interfering — if stacked gain < best individual, the evolve skill optimizes code interactions via ShinkaEvolve.
- **Goal anchoring & behavioral memory**: `scripts/goal_memory.py` maintains `optimization-goals.json` (goal anchor) and `learned-behaviors.json` (accumulated learnings). The orchestrator validates agent outputs post-dispatch. All agents have `memory: local` for persistent role-specific memory at `.claude/agent-memory-local/<agent-name>/` in the target project.
- **Overfitting detection**: Monitor compares train vs val metrics to detect overfitting (val worsens while train improves). Reports severity and triggers regularization prioritization.
- **HP interaction detection**: `detect_hp_interactions()` identifies 2-way HP interaction effects (e.g., "high LR only works with small batch size"). Integrated into analysis output.
- **Adaptive branch budget**: HP-tune allocates more experiments to promising branches and fewer to struggling ones. Scores by improvement × confidence factor.
- **Checkpoint warm-starting**: Experiments can resume from prior checkpoints (lower LR, fewer epochs). Saves 50-80% compute in later iterations.
- **ShinkaEvolve code evolution**: When HP tuning shows diminishing returns, the orchestrator dispatches the implement-agent with the evolve skill for fine-grained code mutations. The full pipeline: `shinka-convert` → `shinka-run` (file-based LLM handoff, `SHINKA_PROVIDER=claude_code`) → `shinka-inspect` → commit. Evolve HPs are tuning-agent-driven. The handoff uses a configurable timeout (`SHINKA_HANDOFF_TIMEOUT`, default 600s) with `.inprogress` marker acknowledgment.
- **Small dataset awareness**: Research agent shifts search toward low-data techniques (transfer learning, few-shot learning, adapters, prompt tuning, semi-supervised methods) when dataset has fewer than 5K samples.
- **Structured ideation**: Knowledge-mode research proposals use a diverge-converge-refine process with 6 ideation lenses (Problem-First, Analogical Reasoning, What Changed Recently, Constraint Manipulation, Negation/Inversion, Composition/Decomposition) plus a Two-Sentence Test filter.
- **Statistical confidence assessment**: Analysis computes Cohen's d effect sizes for HP impact and labels findings by evidence strength (high/medium/low). Method attribution distinguishes code-change vs HP-tuning vs compound effects.
- **Reproducibility metadata**: Each experiment captures random seeds, pip freeze snapshots, git SHA, and framework versions under a `"reproducibility"` key in result JSONs.
- **Report quality gates**: Final reports include "Threats to Validity" section and citation verification (Step 5.3) that cross-references claims against experiment data and checks source URL accessibility.

### Gotchas

- `scripts/detect_divergence.py` CLI takes a **JSON string**, not a file path: `'[0.5, 0.4, 100.0]'`
- `scripts/implement_utils.py` has **three CLI modes**: default (parse proposals), `clone <url> <dest>`, and `analyze <path>`
- **Metric routing is split**: monitor uses loss, analyze uses primary_metric. Mixing these up causes silent wrong behavior.
- **Branch experiments are independent**: results on `ml-opt/branch-a` don't predict what works on `ml-opt/branch-b`.
- **Tabular ML frameworks** (sklearn, XGBoost, LightGBM) skip divergence monitoring entirely.
- **Multiple research findings files**: `research-findings.md` (Phase 5), `research-findings-method-proposals.md` (pre-loop), `research-findings-method-proposals-iter<N>.md` (mid-loop). Deduplication checks all of them.
- **`scripts/goal_memory.py validate-output` returns exit code 2** for violations (0=valid, 1=script error, 2=violations). Imports `scripts/error_tracker.py` lazily for dead-end checks — both must be in `scripts/`.
- **ShinkaEvolve must use the local submodule, not PyPI**: The PyPI package `shinka-evolve` lacks `file_handoff_provider`. Use the local submodule via `setup_evolve.sh` or `PYTHONPATH`.
- **ShinkaEvolve uses bare `python`** for subprocess evaluation. On systems where only `python3` exists, use a conda env or create a symlink.
- **ShinkaEvolve file handoff timeout**: Default 600s, configurable via `SHINKA_HANDOFF_TIMEOUT` env var. The agent must write `<id>.inprogress` when picking up a pending request to extend the deadline. `shinka_run` writes `<id>.heartbeat` every 5s for liveness detection.
- **Phase gate protocol**: Each phase reference file includes a gate check (`pipeline_state.py gate <from> <to>`) and completion log (`log-gate`). Prevents illegal phase transitions.
- **Inter-agent relay contracts**: `schema_validator.py relay <route> <json>` validates relay messages. 5 routes defined. Persistent agents emit `RELAY_ACK`.
- **Decision logging**: `pipeline_state.py log-decision` records LLM decisions with SHA-256 input hashing for divergence detection.

## Development

### Test Dependencies

The plugin's own scripts use only the Python standard library. Running the test suite requires:

```bash
pip install pytest              # test runner
pip install torch torchvision   # used by bundled test fixtures (tests/fixtures/tiny_resnet_cifar10/)
pip install pyyaml              # YAML config parsing in fixtures
```

### Running Tests

```bash
cd <plugin-directory>
python3 -m pytest tests/ -v                           # all tests (~1000, ~8 minutes)
python3 -m pytest tests/test_parsing.py -v            # single file
python3 -m pytest tests/ -m "not slow" -v             # skip real training tests (fast)
python3 -m pytest tests/test_e2e_pipeline.py -m slow  # real end-to-end training only
```

The `slow` marker identifies tests that run real training (see `@pytest.mark.slow` in `tests/test_e2e_pipeline.py`). Most tests are unit/integration tests with no GPU requirement. No build step. No linter. Python 3.10+ required.

## License

MIT License. See [LICENSE](LICENSE) for details.
