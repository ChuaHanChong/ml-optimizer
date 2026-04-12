---
name: hyperagent
description: Orchestrate the self-referential optimization — Phase 7 experiments (HP tuning, code mutations, research-implement, staged eval), Phase 8 method stacking, and self-improving meta-patches. Archive-based lineage tracking with parent selection.
user-invocable: false
---

# Hyperagent Skill

Use extended thinking for all reasoning. Ultrathink.

> **Path convention:** All paths written as `<exp_root>/...` refer to the `exp_root` parameter from your dispatch. The plugin does not hardcode the output directory name.

## Overview

This skill enables the plugin to **self-improve** while optimizing. The hyperagent helps Phase 7 (experiments) and Phase 8 (method stacking) in a loop, choosing the best strategy at each iteration. It also maintains an evolutionary archive with lineage tracking and parent selection.

**Core principle: Standing on the shoulders of giants.** Research-implement is a first-class strategy. The best ML improvements come from proven techniques in papers — prioritize them before inventing from scratch. User-provided papers get highest priority.

**Autonomous by default.** The loop runs non-stop until the target is reached or the user manually stops. No hardcoded thresholds — the agents make evidence-based decisions. Analysis advises direction, the hyperagent decides specific action.

Powered by Facebook Research's Hyperagents (DGM framework) for archive management and parent selection algorithms, and SakanaAI's ShinkaEvolve for fine-grained code mutations.

## Prerequisites

The Hyperagents submodule must be available. If not initialized:
```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/setup_hyperagent.sh
```

## Input Parameters

- `project_root`: Path to the user's project
- `exp_root`: Path to the output directory (any name — set at Phase 0)
- `primary_metric`: Metric being optimized
- `lower_is_better`: Metric direction
- `scope_level`: Constraint on changes (`"training"`, `"architecture"`, `"full"`)
- `target_value`: Target metric value (or null for "as good as possible")

## Step 0: Initialize Archive (first iteration only)

If the archive doesn't exist yet:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/skills/hyperagent-init/scripts/init_archive.py --output-dir <exp_root>/hyperagent
```
This creates gen-000 from baseline and seeds validated branches from Phase 6.

## The Loop

Phase 7 ↔ Phase 8 in a loop. Each iteration:

### Step 1: Read Context

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/goal_memory.py <exp_root> summary
python3 ${CLAUDE_PLUGIN_ROOT}/skills/hyperagent-archive/scripts/archive_utils.py stats --output-dir <exp_root>/hyperagent
python3 ${CLAUDE_PLUGIN_ROOT}/skills/hyperagent-archive/scripts/archive_utils.py operator-stats --output-dir <exp_root>/hyperagent
python3 ${CLAUDE_PLUGIN_ROOT}/skills/hyperagent-archive/scripts/archive_utils.py best --output-dir <exp_root>/hyperagent -n 5
```

Also read the latest analysis output (from the orchestrator's context relay).

### Step 2: Decide Action

Based on the context + analysis advice, decide ONE action. State your decision clearly in your response so the orchestrator knows what to do next.

**Decision guidance (analysis advises, you decide):**

| Operator | When | Who executes | Generate skill operator |
|---|---|---|---|
| `hp_tune` | Default start, after code mutations, analysis says continue/hp_expand/narrow_space | Orchestrator dispatches tuning-agent + experiment-agents | N/A (orchestrator) |
| `research_implement` | User papers available, no research yet, fresh ideas needed | Orchestrator dispatches research-agent + implement-agent FROM selected parent | N/A (orchestrator) |
| `llm_patch` | HP plateaued, research explored, structural changes needed | You execute via `Skill("ml-optimizer:hyperagent-generate")` | `llm_patch` |
| `shinka_evolve` | Fine-grained tuning needed on a good variant | You execute via `Skill("ml-optimizer:hyperagent-generate")` which invokes `Skill("ml-optimizer:evolve")` | `external_tool` |
| `meta_improve` | All approaches stalling, strategy needs change (max 3/session) | You execute via `Skill("ml-optimizer:hyperagent-generate")` with `meta_improvement_mode: true` | `meta_improve` |
| `method_stacking` | Analysis advises stacking, multiple methods improved | Orchestrator runs Phase 8 stacking flow | N/A (orchestrator) |

**Key distinction:** For `hp_tune`, `research_implement`, and `method_stacking`, you return the decision and the orchestrator dispatches the worker agents. For `llm_patch`, `shinka_evolve`, and `meta_improve`, you execute directly using your skills. When invoking `Skill("ml-optimizer:hyperagent-generate")`, pass the **Generate skill operator** column value as `mutation_operator`.

**Decision logging:** After deciding your action, log it before executing. Use `archive_generation` from the archive stats (Step 1) as the `iteration` value:
```
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/pipeline_state.py <exp_root> log-decision '{"phase": 7, "iteration": <archive_generation>, "agent": "hyperagent", "decision_type": "operator_selection", "decision": "<chosen_operator>", "reasoning": "<why>", "context_summary": "archive_gen=<N>, analysis_advice=<advice>"}'
```

### Step 3: Execute (for actions you handle directly)

For `llm_patch` or `shinka_evolve`:
1. `Skill("ml-optimizer:hyperagent-select")` — pick parent from archive
2. `Skill("ml-optimizer:hyperagent-generate")` — create code variant
3. `Skill("ml-optimizer:hyperagent-eval")` — staged eval (cheap filter → full training if passes)

For `meta_improve`:
- **Before generating**, validate the counter: `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/pipeline_state.py <exp_root> meta-patch validate '{"skill": "<target>", "change": "<desc>", "reason": "<why>", "expected_impact": "<impact>"}'`. If validation fails (max 3 per session exceeded or forbidden skill), skip meta_improve for this iteration and select a different operator (e.g., `llm_patch` or `research_implement`).
- `Skill("ml-optimizer:hyperagent-generate")` with `meta_improvement_mode: true`
- Writes patched skill files to `<exp_root>/meta-patches/`
- **After generating**, log it: `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/pipeline_state.py <exp_root> meta-patch log '{"skill": "<skill>", "change": "<desc>", "reason": "<why>", "expected_impact": "<impact>"}'`

After execution, report what you did: which action, the genid, branch name, fitness score, and whether HP tuning is needed on the new code.

### Step 4: Archive

`Skill("ml-optimizer:hyperagent-archive")` — update the archive with results, track lineage and operator effectiveness.

After HP tuning improves a variant:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/skills/hyperagent-archive/scripts/archive_utils.py update-fitness --output-dir <exp_root>/hyperagent <genid> <new_score> [--exp-id <best_exp_id>]
```

### Step 5: Report Meta-Patches (if meta-improve was used)

After a `meta_improve` action, report to the orchestrator that patched skill files were generated at `<exp_root>/meta-patches/`. The orchestrator will then include these patches in all subsequent agent dispatches:

```
META-PATCHES ACTIVE: Read the patched version at <exp_root>/meta-patches/<skill>-SKILL.md
INSTEAD of your default skill. Changes: <from meta-changelog.json>
```

This ensures the self-referential improvement takes effect immediately — not just in the next session. The orchestrator reads `<exp_root>/meta-patches/meta-changelog.json` to know which skills were patched.

### Step 6: Analyze

The orchestrator dispatches the analysis-agent (with meta-patches if active). The analysis advises: continue, pivot direction, or method_stacking. The orchestrator resumes you with the analysis output. Loop back to Step 1.

## Sub-Skills (invoked internally)

Do not invoke these directly — this skill orchestrates them:

- `Skill("ml-optimizer:hyperagent-init")` — Create archive from baseline + existing branches
- `Skill("ml-optimizer:hyperagent-select")` — Select parent (5 strategies: sigmoid + diversity)
- `Skill("ml-optimizer:hyperagent-generate")` — Core mutation (3 operators + meta-improvement)
- `Skill("ml-optimizer:hyperagent-eval")` — Two-stage evaluation (staged → full with warm-start)
- `Skill("ml-optimizer:hyperagent-archive")` — Update archive with results
- `Skill("ml-optimizer:hyperagent-inspect")` — Extract top variants as Markdown context bundle

## ShinkaEvolve Integration

ShinkaEvolve is one mutation operator. When the hyperagent chooses `shinka_evolve`:

1. `Skill("ml-optimizer:hyperagent-select")` picks the parent
2. `Skill("ml-optimizer:hyperagent-generate")` invokes `Skill("ml-optimizer:evolve")` internally
3. The evolve skill runs the full ShinkaEvolve pipeline: `shinka-convert` → `shinka-run` → `shinka-inspect`
   **CRITICAL:** Set `SHINKA_PROVIDER=claude_code` and `SHINKA_HANDOFF_DIR=<exp_root>` before launching shinka-run. This enables file-based LLM handoff where YOU act as the LLM backend by polling `<exp_root>/evolve/pending/` and writing responses to `<exp_root>/evolve/completed/`. Without these env vars, shinka-run will try external API keys and fail.
4. The result branch is renamed to `ml-opt/gen-<N>-evolved-<slug>` for archive consistency

## Output

The loop produces:
- `<exp_root>/code-archive.jsonl` — evolutionary archive with lineage
- `<exp_root>/results/exp-*.json` — experiment results (standard format)
- `<exp_root>/meta-patches/` — session-scoped skill modifications (if meta-improve was used)
- `<exp_root>/meta-patches/meta-changelog.json` — changelog of meta-improvements

## Important Rules

- **Autonomous.** Never auto-stop on plateaus. Try different operators. Only the user or target achievement stops the loop.
- **Standing on the shoulders of giants.** Try research-implement early. User papers get highest priority.
- **After every code mutation, HP-tune.** Report that the new code needs HP tuning so the orchestrator dispatches tuning-agent.
- **Update fitness after HP tuning.** Call `update-fitness` so parent selection reflects tuned results.
- **Respect scope_level.** Training = HP only. Architecture = + model changes. Full = anything including ShinkaEvolve.
- **Check dead ends.** Never re-propose techniques in the dead-end catalog.
- **Max 3 meta-improvements per session.** Safety cap on self-modification.
- **Return decisions, don't dispatch agents.** For `hp_tune`, `research_implement`, and `method_stacking`, return the decision. The orchestrator dispatches worker agents.
