# ml-optimizer

A **self-referential and self-improving hyperagent** for autonomous ML model optimization. Powered by Claude Code.

> **The plugin IS the hyperagent.** It doesn't just optimize your ML model — it optimizes how it optimizes. The hyperagent controls the entire loop, learns which strategies work, and evolves its own approach with each session. No hardcoded thresholds. No fixed pipeline. Autonomous by default — runs until the goal is reached or you stop it.

## Why a Hyperagent?

Traditional ML optimization tools follow a fixed pipeline — same strategy every time regardless of the task. The ml-optimizer takes a different approach:

1. **Optimizes your ML model** (the task) — using an evolutionary archive with lineage tracking, parent selection, staged evaluation, and three mutation operators (LLM patches, ShinkaEvolve, research-implement)

2. **Optimizes its own optimization strategy** (the meta-task) — the hyperagent modifies the plugin's own skill instructions based on what worked and what didn't. The agents make evidence-based decisions, not rule-based.

3. **Gets better with each session** — promoted meta-patches persist in the plugin, claude-mem recalls insights from prior sessions. Future optimization sessions start smarter.

**Core principle: Standing on the shoulders of giants.** The best ML optimization comes from building on proven research. The plugin prioritizes finding and implementing techniques from papers before inventing from scratch. User-provided papers get highest priority. Research-implement is a first-class strategy, not a fallback.

## Overview

The ml-optimizer understands your ML model, establishes baselines, researches improvements, tunes hyperparameters, evolves code, runs experiments (in parallel across GPUs), monitors for training divergence, and gets better at optimizing with each session.

**Key design decisions:**
- **Hyperagent architecture**: Enables self-improvement and drives Phase 7 (experiments) ↔ Phase 8 (stacking) in a loop, choosing between HP tuning, LLM code patches, ShinkaEvolve, research-implement, stacking, and meta-improvement at each iteration
- **Self-referential improvement**: The hyperagent can modify the plugin's own skill instructions to evolve the optimization strategy (session-scoped with end-of-session promotion)
- **Archive-based evolutionary search**: Population archive with lineage tracking and parent selection (Hyperagents' DGM framework, sigmoid + diversity penalty, UCB1 tree search with MCTS backpropagation)
- **Staged evaluation**: Two-stage eval (10% budget cheap filter → full training if promising) saves 50-80% compute
- LLM-driven hyperparameter tuning (Claude reasons about results — no Optuna/grid search)
- Research via web search + alphaxiv academic paper search + user-provided papers
- ShinkaEvolve for fine-grained evolutionary code mutations
- Structured `experiments/` directory in your project

### Hyperagent Capabilities

Inspired by [Facebook Research Hyperagents](https://github.com/facebookresearch/Hyperagents) (DGM framework), [SakanaAI ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) and [karpathy/autoresearch](https://github.com/karpathy/autoresearch):

| Feature | What it does |
|---------|-------------|
| **Hyperagent** | Opus agent that enables self-improvement and helps Phase 7 ↔ Phase 8 in a loop. Analysis advises direction, hyperagent decides specific action |
| **Evolutionary Code Archive** | JSONL archive tracking code variants with parent-child lineage, fitness scores, and mutation operators. 6 parent selection strategies including UCB1 (Auer et al. 2002) with MCTS-style backpropagation for balanced explore/exploit |
| **Staged Evaluation** | Two-stage eval: 10% budget cheap filter with adaptive threshold → full training only if promising. Warm-starts from staged checkpoint |
| **Three Mutation Operators** | LLM patches (structural), ShinkaEvolve (fine-grained AST), research-implement (paper-informed). Hyperagent learns which operator is most effective |
| **Self-Referential Improvement** | Hyperagent modifies the plugin's own skill instructions (hp-tune, analyze, research). Session-scoped with end-of-session promotion gate |
| **Cross-Session Learning** | Promoted meta-patches persist in the plugin. claude-mem recalls prior sessions |
| **Stuck Protocol** | When analysis advises stop, the hyperagent tries other operators first. If stuck, reads error patterns and dispatches research for new approaches |
| **Dead-End Catalog** | Tracks techniques conclusively shown to be unpromising. Research and hp-tune agents consult it before proposing, preventing wasted budget |
| **Research Agenda** | Living document initialized from proposals, reprioritized after each batch based on experimental evidence |
| **Progress Dashboard** | Self-contained HTML dashboard with auto-refresh (`--live`), SVG timeline, sortable results, HP sensitivity, method explanations |
| **Immutable Baseline** | SHA-256 checksum of baseline metrics verified before each batch — halts if metrics are modified |
| **Goal Anchoring** | `optimization-goals.json` written at Phase 0; all agents read it before acting. Post-dispatch validation catches frozen param changes, scope breaches, dead-end re-proposals |
| **Behavioral Memory** | `learned-behaviors.json` accumulates HP constraints, method outcomes, divergence patterns. All agents have `memory: local` for persistent role-specific learning |
| **Resumable Subagents** | 6 persistent agents (research, implement, tuning, analysis, monitor, meta) resumed via `SendMessage` — preserving accumulated context across the pipeline |
| **Inter-Agent Relay** | Orchestrator relays findings between agents via `CONTEXT FROM OTHER AGENTS:` sections — analyze findings reach hp-tune, monitor OOM info reaches hp-tune, research proposals reach implement |

## Installation

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
# Initialize submodules (ShinkaEvolve + Hyperagents)
bash scripts/setup_evolve.sh
bash scripts/setup_hyperagent.sh
```

Run `/reload-plugin` or restart Claude Code. The `/optimize` command and all 11 agents will be available automatically.

> **Why local/project?** Agent memory (`memory: local`) stores learnings in `.claude/agent-memory-local/` within the project. Local or project-based installation keeps plugin code, agent memory, and experiment data together — scoped to your ML project, not polluting other workspaces.

### Development / local testing

Load the plugin directly from source without installing:

```bash
claude --plugin-dir <path-to-ml-optimizer> --dangerously-skip-permissions
```

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

### MCP Servers (Recommended)

These MCP servers enhance the plugin's capabilities. The plugin works without them but benefits significantly from their presence:

| MCP Server | What it enables | Used by |
|------------|-----------------|---------|
| **alphaxiv** (`api.alphaxiv.org/mcp/v1`) | Academic paper search (2.5M+ arXiv papers), paper content extraction, PDF Q&A, GitHub repo exploration | research-agent (6 tools), implement-agent (2 tools) |
| **claude-mem** | Cross-session memory — recalls past optimization sessions, avoids re-proposing failed techniques | research-agent, orchestrator |

To connect alphaxiv: Use `/mcp` in Claude Code and add the alphaxiv SSE endpoint with OAuth authentication.
To connect claude-mem: Install the claude-mem plugin which provides the MCP server automatically.

### Optional

| Dependency | What it enables |
|------------|-----------------|
| NVIDIA GPU + drivers | GPU profiling via `nvidia-smi`, parallel experiments |
| `pytest` | Running the plugin's test suite (`pip install pytest`) |
| Web search access | Research skill fetches recent papers and techniques (fallback when alphaxiv unavailable) |

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
| `evolve` | Orchestrates evolutionary code refinement via full ShinkaEvolve pipeline (convert → run → inspect) | Internal |
| `hyperagent` | Orchestrates the self-referential optimization — Phase 7 experiments, Phase 8 stacking, meta-improvement | Internal |
| `hyperagent-inspect` | Extract best archive entries as Markdown context bundle | Internal |
| `hyperagent-init` | Create evolutionary archive from baseline + existing branches | Internal |
| `hyperagent-select` | Select parent from archive (6 strategies: sigmoid + diversity, UCB1 tree search) | Internal |
| `hyperagent-generate` | Core mutation — LLM patch / ShinkaEvolve / research-implement + self-referential improvement | Internal |
| `hyperagent-eval` | Two-stage evaluation: staged filter → full training with warm-start | Internal |
| `hyperagent-archive` | Update archive with results, track lineage and operator effectiveness | Internal |

## Workflow

6 agents are **persistent** (resumed via `SendMessage` with accumulated context), 4 are **ephemeral** (fresh spawn). The hyperagent enables self-improvement and drives Phase 7 ↔ Phase 8 in a loop. The orchestrator relays findings between agents.

```
0+1. Discovery & Planning (plan mode — multi-round refinement until user approves)
     Q&A → write goals → analyze codebase → present plan → user refines → repeat
2. Prerequisites (validate dataset, prepare data, install deps)          [ephemeral]
3. Establish baseline                                                    [ephemeral]
4. User checkpoint: review baseline, choose direction
5. Research (alphaxiv + web search + LLM knowledge for techniques)       [persistent]
6. Implement proposals (creates git branches, applies + validates)       [persistent]
7. Hyperagent Driven Experiment Loop (autonomous):
   -> Initialize code archive (baseline + implementation branches)
   -> Each iteration, the hyperagent decides:
      a. HP tuning (delegates to tuning-agent)                           [persistent]
      b. LLM code patch (hyperagent modifies code directly)              [persistent]
      c. ShinkaEvolve mutation (delegates to evolve skill)
      d. Research-implement (delegates to research + implement agents)
      e. Meta-improvement (modifies skill files, max 3/session)
   -> Staged eval → full experiment → archive → analyze → next iteration
   -> When analysis advises stop, hyperagent tries other operators
8. Method stacking (Phase 7 ↔ Phase 8 loop):
   -> Analysis advises stacking, hyperagent decides
   -> Sequentially merges best methods, skip-on-failure
   -> Analysis agent loops evolve + HP-tune until improvement or stop
9. Report, Review & Promotion:
   -> Generate final report                                              [ephemeral]
   -> Session review (what worked, what didn't, how to improve)          [persistent]
   -> Meta-patch promotion (if hyperagent generated skill patches)
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
  optimization-goals.json               # Goal anchor (Phase 0, read by all agents)
  learned-behaviors.json                # Accumulated behavioral memory (HP constraints, outcomes)
  reports/                              # Markdown reports (analysis, research, final)
  reports/error-log.json                # Structured error event log
  reports/suggestion-history.json       # Suggestion feedback loop
  reports/session-review.md             # Session review
  results-table.md                      # Auto-generated Markdown results summary
  prepared-data/                        # Prepared dataset (if preprocessing needed)
  code-archive.jsonl                    # Hyperagent evolutionary archive (lineage + fitness)
  meta-patches/                         # Session-scoped meta-improvement skill patches
  pipeline-state.json                   # Resumable pipeline state + agent_registry + hyperagent_state
  dev_notes.md                          # Running session log
```

## Python Utilities

All scripts in `scripts/` use only the standard library and work as both importable modules and CLI tools:

| Script | CLI Usage |
|--------|-----------|
| `scripts/gpu_check.py` | `python3 scripts/gpu_check.py` |
| `scripts/parse_logs.py` | `python3 scripts/parse_logs.py <logfile>` — parses kv/JSON/CSV/XGBoost/HuggingFace Trainer logs |
| `scripts/detect_divergence.py` | `python3 scripts/detect_divergence.py '<json_values>' [--higher-is-better] [--model-category rl\|generative\|supervised]` — also: `--check-overfitting '<train_json>' '<val_json>' [--patience N] [--min-gap F]` |
| `scripts/result_analyzer.py` | `python3 scripts/result_analyzer.py <results_dir> <metric> [baseline_id] [lower_is_better]` — also: `compare <exp_id_1> <exp_id_2> [metric]` |
| `scripts/experiment_setup.py` | `python3 scripts/experiment_setup.py <project_root> <train_command> [gpu_id] [config_json]` |
| `scripts/implement_utils.py` | `python3 scripts/implement_utils.py <findings.md> '<indices_json>'` — also: `clone <url> <dest>`, `analyze <path>` |
| `scripts/pipeline_state.py` | `python3 scripts/pipeline_state.py <exp_root> validate\|save\|load\|cleanup` |
| `scripts/schema_validator.py` | `python3 scripts/schema_validator.py <filepath> result\|baseline\|manifest\|prerequisites [--strict]` — `--strict` enforces completeness |
| `scripts/plot_results.py` | `python3 scripts/plot_results.py <results_dir> <metric> comparison\|timeline\|sensitivity <hp>\|progress [--higher-is-better]` |
| `scripts/prerequisites_check.py` | `python3 scripts/prerequisites_check.py scan-imports\|check-packages\|detect-env\|detect-format\|detect-format-project\|validate-data\|bulk-install-cmd\|gpu-install-cmd` |
| `scripts/error_tracker.py` | `python3 scripts/error_tracker.py <exp_root> log\|show\|patterns\|summary\|sync\|success\|proposals\|rank\|cleanup\|log-suggestion\|suggestion-history\|dead-end <add\|list\|check>\|agenda <init\|update\|list\|add>` |
| `scripts/dashboard.py` | `python3 scripts/dashboard.py <exp_root> [--live] [--table] [--serve --port 8080]` — HTML dashboard + Markdown results table |
| `scripts/excalidraw_gen.py` | `python3 scripts/excalidraw_gen.py <exp_root> pipeline\|comparison\|hp-landscape\|architecture <args>` — Excalidraw JSON diagrams |
| `scripts/goal_memory.py` | `python3 scripts/goal_memory.py <exp_root> init-goals\|read-goals\|log-behavior\|query-behaviors\|validate-output\|summary\|sync-from-errors` — goal anchoring, behavioral memory, agent output validation |
| `skills/hyperagent/Hyperagents/skills/*/scripts/*.py` | Per-skill helper scripts (init\_archive.py, select\_parent.py, run\_eval.py, archive\_utils.py, inspect\_best.py). Import `gl_utils.py` directly from the Hyperagents submodule. |
| `scripts/setup_hyperagent.sh` | `bash scripts/setup_hyperagent.sh` — initialize Hyperagents submodule and create skill symlinks for auto-discovery |

## Running Tests

```bash
cd <plugin-directory>
python -m pytest tests/ -v                           # all tests
python -m pytest tests/test_parse_logs.py -v         # single file
python -m pytest tests/ -m "not slow" -v             # skip real training tests
python -m pytest tests/test_e2e_pipeline.py -m slow  # real training E2E only
```

No build step. No linter. Python 3.10+ required. All scripts use only the standard library.

## Agent Definitions

Eleven agent types in `agents/`. The plugin ships `settings.json` with `"agent": "ml-optimizer:orchestrator-agent"` — when the plugin is enabled, the orchestrator agent becomes the main thread and auto-starts Phase 0.

| Agent | Tools | Model | Effort | Preloaded Skill |
|-------|-------|-------|--------|-----------------|
| **`orchestrator`** | Agent, Read, Write, Edit, Bash, Glob, Grep, Skill, WebSearch, WebFetch | **opus** | high | `ml-optimizer:orchestrate` (main thread) |
| **`hyperagent`** | Read, Write, Edit, Bash, Glob, Grep, Skill, WebSearch, WebFetch | **opus** | high | `ml-optimizer:hyperagent` + all hyperagent-* + evolve + shinka-* + claude-mem |
| `research-agent` | WebSearch, WebFetch, Read, Write, Bash, Glob, Grep, Skill + alphaxiv MCP (6) | opus | high | `ml-optimizer:research` |
| `implement-agent` | Bash, Read, Write, Edit, Glob, Grep, Skill, WebSearch, WebFetch + alphaxiv MCP (2) | opus | high | `ml-optimizer:implement` + evolve + shinka-* |
| `tuning-agent` | Read, Write, Bash, Glob, Grep, Skill, WebSearch, WebFetch | opus | high | `ml-optimizer:hp-tune` |
| `analysis-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | opus | high | `ml-optimizer:analyze` |
| `report-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | opus | high | `ml-optimizer:report` |
| `baseline-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | sonnet | medium | `ml-optimizer:baseline` |
| `monitor-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | sonnet | medium | `ml-optimizer:monitor` |
| `experiment-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | sonnet | medium | `ml-optimizer:experiment` |
| `prerequisites-agent` | Bash, Read, Write, Glob, Grep, Skill, WebSearch, WebFetch | sonnet | medium | `ml-optimizer:prerequisites` |

The **orchestrator** is the main-thread agent (activated by `settings.json`). The **hyperagent** is the Phase 7/8 loop driver. Other agents are specialized workers coordinated through the orchestrator relay. Analytical agents use `effort: high` and `model: opus`. Procedural agents use `effort: medium` and `model: sonnet`.

## Hooks (Safety Guardrails)

Lifecycle hooks in `hooks/` protect against common failure modes during autonomous operation:

| Hook | Event | Type | Purpose |
|------|-------|------|---------|
| `bash-safety.sh` | PreToolUse (Bash) | command | Blocks `rm -rf /`, `git push --force`, `curl\|bash`, `chmod 777`, etc. |
| `file-guardrail.sh` | PreToolUse (Write/Edit) | command | Blocks writes to `.git/`, `.env`, credentials, lock files |
| `detect-critical-errors.sh` | PostToolUse + PostToolUseFailure (Bash) | command | Detects CUDA OOM, segfault, disk full; logs to error tracker |
| `pre-compact.sh` | PreCompact | command | Saves pipeline state snapshot to dev_notes before context compaction |
| `post-compact-context.sh` | SessionStart (compact) | command | Re-injects phase/metric/budget context after compaction |
| SubagentStop | SubagentStop | prompt | LLM-based check: validates agent outputs and memory updates before stopping |
| `subagent-start-inject-goals.sh` | SubagentStart | command | Injects goal memory summary when any agent starts |
| `file-changed-pipeline-state.sh` | FileChanged (pipeline-state.json) | command | Detects external corruption of pipeline state |
| `cwd-changed-detect-experiments.sh` | CwdChanged | command | Auto-detects existing `experiments/` and offers to resume |
| Stop prompt | Stop | command | Verifies final report exists before session ends |
| SessionStart prompt | SessionStart | prompt | Routes ML optimization requests to orchestrate skill |
| UserPromptSubmit prompt | UserPromptSubmit | prompt | Routes ML optimization requests to orchestrate skill |

Exit code `2` = block action. Exit code `0` = allow. Configured in `hooks/hooks.json`.

## Key Design Patterns

- **Non-git fallback**: If the project isn't a git repo, file backups replace branch isolation. Experiments run sequentially.
- **Metric routing**: Monitor/divergence always uses loss. Analyze/hp-tune use the user's `primary_metric`.
- **OOM feedback loop**: When experiments OOM, batch size is recorded. Next hp-tune call receives `max_batch_size` to avoid re-proposing failing configs.
- **All-diverge recovery**: If all experiments in a batch diverge, a recovery batch with halved learning rates runs before stopping.
- **Research cadence**: When method proposals are enabled, research triggers every N batches. If no new proposals found, cadence doubles (exponential backoff).
- **Pipeline resumption**: `pipeline-state.json` persists phase, user choices, and stop count. On restart, stale experiments are cleaned up and phase gates prevent cascading failures.
- **Loop exit conditions**: The experiment loop is autonomous — runs until: (1) target metric achieved, or (2) user manually stops. When analysis advises stop, the hyperagent tries other operators first.
- **Three-tier result tracking**: Experiments carry `method_tier` (baseline / method_default_hp / method_tuned_hp) and `proposal_source` (paper / llm_knowledge) for attribution analysis.
- **Method stacking**: After independent method testing, top methods are sequentially merged. Clean merges proceed; conflicts are LLM-resolved. Degrading combinations are skipped. After each successful stack step, the analysis agent assesses whether methods are interfering — if stacked gain < best individual, the evolve skill optimizes code interactions via ShinkaEvolve.
- **Goal anchoring & behavioral memory**: `scripts/goal_memory.py` maintains `optimization-goals.json` (goal anchor) and `learned-behaviors.json` (accumulated learnings). The orchestrator validates agent outputs post-dispatch. All 9 agents have `memory: local` for persistent role-specific memory at `.claude/agent-memory-local/<agent-name>/` in the target project.
- **Overfitting detection**: Monitor compares train vs val metrics to detect overfitting (val worsens while train improves). Reports severity and triggers regularization prioritization.
- **HP interaction detection**: `detect_hp_interactions()` identifies 2-way HP interaction effects (e.g., "high LR only works with small batch size"). Integrated into analysis output.
- **Adaptive branch budget**: HP-tune allocates more experiments to promising branches and fewer to struggling ones. Scores by improvement × confidence factor.
- **Checkpoint warm-starting**: Experiments can resume from prior checkpoints (lower LR, fewer epochs). Saves 50-80% compute in later iterations.
- **Hyperagent-driven optimization**: The hyperagent drives Phase 7 ↔ Phase 8 in a loop and enables self-improvement. It maintains a code archive (`code-archive.jsonl`) with lineage tracking and selects parents using Hyperagents' exact algorithms: `sigmoid(10(s - μ)) × exp(-(children/8)³)`. Three mutation operators: LLM patches (structural), ShinkaEvolve (fine-grained), research-implement (paper-informed). The hyperagent learns which operator is effective and adapts.
- **Staged evaluation**: Every code mutation gets a cheap pre-filter (10% budget, adaptive threshold) before full training. Warm-starts from staged checkpoint. Saves 50-80% compute by filtering unpromising variants early.
- **Self-referential meta-improvement**: The hyperagent can modify the plugin's own skill instructions (hp-tune, analyze, research). Session-scoped patches in `experiments/meta-patches/`. Max 3 per session. End-of-session promotion gate: analysis-agent evaluates, user approves, committed to plugin branch.
- **ShinkaEvolve as mutation operator**: ShinkaEvolve is one tool within the Hyperagent loop. The hyperagent dispatches it for fine-grained code mutations (numerical constants, local optimizations) via `Skill("ml-optimizer:evolve")`. The full pipeline: `shinka-convert` → `shinka-run` (file-based LLM handoff) → `shinka-inspect` → commit. Evolve HPs are tuning-agent-driven.
- **Small dataset awareness**: Research agent shifts search toward low-data techniques (transfer learning, few-shot learning, adapters, prompt tuning, semi-supervised methods) when dataset has fewer than 5K samples.
- **Structured ideation**: Knowledge-mode research proposals use a diverge-converge-refine process with 6 ideation lenses (Problem-First, Analogical Reasoning, What Changed Recently, Constraint Manipulation, Negation/Inversion, Composition/Decomposition) plus a Two-Sentence Test filter.
- **Statistical confidence assessment**: Analysis computes Cohen's d effect sizes for HP impact and labels findings by evidence strength (high/medium/low). Method attribution distinguishes code-change vs HP-tuning vs compound effects.
- **Reproducibility metadata**: Each experiment captures random seeds, pip freeze snapshots, git SHA, and framework versions under a `"reproducibility"` key in result JSONs.
- **Report quality gates**: Final reports include "Threats to Validity" section and citation verification (Step 5.3) that cross-references claims against experiment data and checks source URL accessibility.

## Gotchas

- `scripts/detect_divergence.py` CLI takes a **JSON string**, not a file path: `'[0.5, 0.4, 100.0]'`
- `scripts/implement_utils.py` has **three CLI modes**: default (parse proposals), `clone <url> <dest>`, and `analyze <path>`
- **Metric routing is split**: monitor uses loss, analyze uses primary_metric. Mixing these up causes silent wrong behavior.
- **Branch experiments are independent**: results on `ml-opt/branch-a` don't predict what works on `ml-opt/branch-b`.
- **Tabular ML frameworks** (sklearn, XGBoost, LightGBM) skip divergence monitoring entirely.
- **Multiple research findings files**: `research-findings.md` (Phase 5), `research-findings-method-proposals.md` (pre-loop), `research-findings-method-proposals-iter<N>.md` (mid-loop). Deduplication checks all of them.
- **`scripts/goal_memory.py validate-output` returns exit code 2** for violations (0=valid, 1=script error, 2=violations). Imports `scripts/error_tracker.py` lazily for dead-end checks — both must be in `scripts/`.

## Evolutionary Submodules

The plugin integrates two evolutionary frameworks as git submodules:

### Hyperagents (Facebook Research)

[Hyperagents](https://github.com/facebookresearch/Hyperagents) provides the archive management and parent selection algorithms for the evolutionary code search loop. The hyperagent uses these to maintain a population of code variants with lineage tracking. Parent selection supports 6 strategies including UCB1 (Auer et al. 2002) with MCTS-style backpropagation — scores are min-max normalized to [0,1] with metric direction awareness (lower-is-better inversion), and `eval_count` tracks true evaluations separately from `visit_count` (inflated by ancestor backprop) to ensure correct explore/exploit balance.

```bash
bash scripts/setup_hyperagent.sh
```

```
skills/hyperagent/
  Hyperagents/                              # Git submodule (ChuaHanChong/HyperAgents)
    utils/gl_utils.py                       # Archive + parent selection algorithms
    skills/hyperagent-*/                    # Claude Code skills (in the submodule)
  hyperagent-init → hyperagent/Hyperagents/skills/hyperagent-init       # Symlinks
  hyperagent-select → hyperagent/Hyperagents/skills/hyperagent-select
  hyperagent-generate → hyperagent/Hyperagents/skills/hyperagent-generate
  hyperagent-eval → hyperagent/Hyperagents/skills/hyperagent-eval
  hyperagent-archive → hyperagent/Hyperagents/skills/hyperagent-archive
  hyperagent-inspect → hyperagent/Hyperagents/skills/hyperagent-inspect
```

### ShinkaEvolve (SakanaAI)

[ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) provides fine-grained evolutionary code mutations. Used as one mutation operator within the Hyperagent loop.

```bash
bash scripts/setup_evolve.sh
```

```
skills/evolve/
  ShinkaEvolve/                             # Git submodule (ChuaHanChong/ShinkaEvolve)
    skills/shinka-*/                        # ShinkaEvolve's Claude Code skills
  shinka-setup → evolve/ShinkaEvolve/skills/shinka-setup     # Symlinks
  shinka-convert → evolve/ShinkaEvolve/skills/shinka-convert
  shinka-run → evolve/ShinkaEvolve/skills/shinka-run
  shinka-inspect → evolve/ShinkaEvolve/skills/shinka-inspect
```

The hyperagent dispatches ShinkaEvolve via `Skill("ml-optimizer:evolve")` for fine-grained mutations (numerical constants, local optimizations). The full pipeline: `shinka-convert` → `shinka-run` (file-based LLM handoff, `SHINKA_PROVIDER=claude_code`) → `shinka-inspect` → commit.

## Progress Dashboard

The plugin generates a self-contained HTML dashboard to monitor optimization progress. No external dependencies required.

### Generate a static dashboard

```bash
python3 scripts/dashboard.py <project>/experiments
# Output: experiments/reports/dashboard.html
```

Open `experiments/reports/dashboard.html` in any browser. It shows experiment results, HP sensitivity, timeline, error summary, and method explanations.

### Live auto-refresh (during active optimization)

```bash
python3 scripts/dashboard.py <project>/experiments --live
```

Adds a 30-second auto-refresh to the HTML — the browser reloads itself to pick up new results. The orchestrator runs this automatically after each experiment batch in Phase 7.

### Serve via HTTP (remote machines)

```bash
python3 scripts/dashboard.py <project>/experiments --serve --port 8080
```

Starts a local HTTP server at `http://localhost:8080/dashboard.html`. Useful when SSH'd into a remote machine — port-forward and view in your local browser. Can be combined with `--live` for auto-refreshing served dashboard.

### Markdown results table

```bash
python3 scripts/dashboard.py <project>/experiments --table
```

Generates `experiments/results-table.md` — a git-trackable Markdown summary with ranked results, improvement percentages, and HP correlations. Can be combined with other flags:

```bash
python3 scripts/dashboard.py <project>/experiments --live --table  # Both HTML + Markdown
```

## License

MIT License. See [LICENSE](LICENSE) for details.
