---
name: evolve
description: Generate a single targeted code mutation from a parent branch based on fitness feedback. Each invocation produces ONE mutation in an isolated git worktree.
disable-model-invocation: true
user-invocable: false
---

# Evolve Skill

Use extended thinking for all reasoning. Ultrathink. Analyze parent code deeply, understand what worked and failed, then generate ONE targeted mutation that addresses a specific weakness.

## Overview

This skill generates a **single code mutation** from a parent implementation, guided by fitness feedback. It operates in one of two modes:

1. **ShinkaEvolve mode (primary):** ShinkaEvolve's `shinka-run` drives the evolution loop. When ShinkaEvolve needs an LLM to generate a mutation, it writes a prompt to a file. The orchestrator picks it up and dispatches this skill to fulfill the request. The response flows back via the file handoff. In this mode, the mutation prompt (system_msg + user_msg) comes from ShinkaEvolve's prompt sampler — respect its EVOLVE-BLOCK markers and SEARCH/REPLACE patch format.

2. **Direct dispatch mode (fallback):** The orchestrator creates git worktrees and dispatches this skill directly, without ShinkaEvolve. Used when ShinkaEvolve is not available or for simple single-generation evolution.

**Architecture:** ShinkaEvolve's 4 built-in Claude Code skills (`shinka-setup`, `shinka-convert`, `shinka-run`, `shinka-inspect`) handle the evolution lifecycle. This skill handles the actual code mutation generation within that lifecycle. ShinkaEvolve manages population, islands, selection, and novelty detection — this skill just generates ONE mutation per invocation.

## Input Parameters

- `project_root`: Path to the isolated git worktree for this mutation
- `parent_branch`: Git branch with the current best implementation
- `parent_code_diff`: The diff of parent vs main (provided by orchestrator to avoid git operations in worktree)
- `parent_metrics`: Dict of parent's metrics
- `mutation_type`: One of `"targeted_refinement"`, `"variant_exploration"`, `"cross_pollination"`
- `mutation_id`: Unique ID for this mutation (e.g., `"mut-1-3"` = generation 1, mutation 3)
- `generation`: Current generation number
- `feedback_context`: Structured feedback from previous generations:
  - `batch_analysis`: Summary of what worked/failed
  - `error_patterns`: From error-log.json
  - `dead_ends`: Techniques to avoid
  - `learned_behaviors`: Accumulated patterns
  - `previous_mutations`: What was tried in prior generations and their results
- `primary_metric`: Which metric to optimize
- `lower_is_better`: Metric direction
- `scope_level`: Constraint on changes (`"training"`, `"architecture"`, `"full"`)
- `exp_root`: Path to experiments directory (in the main project, not the worktree)

## Step 1: Understand the Parent

1. **Read parent code diff** from `parent_code_diff` parameter (pre-computed by orchestrator)
2. **Read parent metrics** from `parent_metrics` parameter
3. **Read feedback context** from `feedback_context` parameter
4. **Check goal constraints:** Read `<exp_root>/optimization-goals.json` for scope limits and frozen parameters

## Step 2: Design ONE Mutation

Based on the `mutation_type` assigned by the orchestrator:

### If `targeted_refinement`:
- Identify the weakest aspect from feedback (e.g., "loss plateaued", "overfitting detected", "slow convergence")
- Design a focused code change that addresses this specific weakness
- Change ONE thing only — keep everything else identical to parent

### If `variant_exploration`:
- Identify an alternative approach to the parent's strategy
- Replace a component (optimizer, scheduler, augmentation, loss function) with an alternative
- Maintain the parent's overall structure but swap the target component

### If `cross_pollination`:
- Requires `previous_mutations` in feedback_context to contain successful elements from other branches
- Combine TWO successful elements from different mutations/branches
- Only attempt if orchestrator provided cross-pollination candidates

## Step 3: Apply the Mutation

**In ShinkaEvolve mode:** The prompt from ShinkaEvolve specifies editable regions via `EVOLVE-BLOCK` markers:
```python
# EVOLVE-BLOCK-START
<editable code here>
# EVOLVE-BLOCK-END
```
Only modify code within these markers. Use ShinkaEvolve's SEARCH/REPLACE patch format in your response:
```
<<<<<<< SEARCH
<original code to find>
========
<replacement code>
>>>>>>> REPLACE
```

**In direct dispatch mode:** The orchestrator has set up a git worktree on `parent_branch`. Apply changes directly using Edit tool.

In both modes:

1. **Read the target files** (in worktree or as provided by ShinkaEvolve)
2. **Apply code changes** — respecting EVOLVE-BLOCK boundaries if present
3. **Mark all changes** with `# [ml-opt] evolve-v<generation>-<mutation_id>: <description>`
4. **Stay within scope_level constraints**

## Step 4: Validate

1. **Syntax check:**
   ```bash
   python3 -c "import py_compile; py_compile.compile('<modified_file>', doraise=True)"
   ```

2. **Import check:**
   ```bash
   python3 -c "import <module>"
   ```

3. **If validation fails:** Attempt ONE fix. If still fails, report `status: "validation_failed"`.

## Step 5: Commit and Report

**In ShinkaEvolve mode:** Return the mutation as text (SEARCH/REPLACE patches). ShinkaEvolve handles committing and evaluation. No git operations needed.

**In direct dispatch mode:** Commit in the worktree:
```bash
git add -A && git commit -m "evolve v<generation> mut-<id>: <description>"
```

**In both modes, return structured result:**
   ```json
   {
     "mutation_id": "<mutation_id>",
     "status": "validated|validation_failed",
     "branch": "<worktree branch name>",
     "description": "<what was changed and why>",
     "mutation_type": "<type>",
     "files_modified": ["<path>"],
     "reasoning": "<why this mutation was chosen based on feedback>"
   }
   ```

## Important Rules

- **ONE mutation per invocation.** You generate exactly one code variant. The orchestrator handles population management.
- **Work only in your worktree.** Do not touch the main project directory.
- **Respect scope_level.** Training scope = only optimizer, scheduler, loss, augmentation, regularization.
- **Check dead ends.** If a technique is in `feedback_context.dead_ends`, do NOT use it.
- **Keep it focused.** Change ONE thing. Compound mutations are hard to attribute.
- **Preserve provenance.** All code must have `# [ml-opt] evolve-v<N>-<id>` comments.
