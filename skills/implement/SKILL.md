---
name: implement
description: "Apply research proposals as code changes. Creates isolated git branches per proposal, implements modifications (architecture, loss, augmentation), validates with progressive checks, and produces a manifest for the experiment loop. Use when: research proposals need to be turned into actual code changes, or when implementing specific ML improvements before experiments."
user-invocable: false
---

# Implement Research Proposals

Apply research proposals as actual code changes with git branch isolation, progressive validation, and a structured manifest for the experiment loop.

Use extended thinking for all analytical reasoning in this skill. Ultrathink. Think through implementation approaches, potential side effects, validation strategies, and backwards compatibility before making code changes.

> **Path convention:** All paths written as `<exp_root>/...` refer to the `exp_root` parameter from your dispatch. The plugin does not hardcode the output directory name.

## Reference

- Implementation patterns: `${CLAUDE_SKILL_DIR}/references/implementation-patterns.md` (in this skill's directory)
- Validation checklist: `${CLAUDE_SKILL_DIR}/references/validation-checklist.md` (in this skill's directory)
- Python helpers: `${CLAUDE_PLUGIN_ROOT}/scripts/implement_utils.py` (`analyze` is for framework detection only — NOT a gitnexus fallback)
- GitNexus code-graph wrapper (REQUIRED): `${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py` (`available`/`index`/`is-indexed`). GitNexus is a HARD PREREQUISITE verified at Phase 2 — there is NO grep/`analyze` fallback for code understanding. The wrapper never raises, but this skill treats `available()==False` or an indexing failure as a HARD ERROR (halt with install/repair guidance), not a silent fallback. `index <path>` runs `gitnexus analyze <path> --index-only` — indexing is NON-INVASIVE: it does NOT inject a GitNexus section into the indexed repo's CLAUDE.md/AGENTS.md and does NOT install `.claude/` skills, and it auto-adds `.gitnexus/` to the repo's git exclude (you still must never `git add` it). Querying the resulting graph is MCP-only (`mcp__gitnexus__context`/`query`/`impact`) — there is no gitnexus-CLI query path in this plugin.

## Inputs Expected

From the orchestrator or direct invocation:
- `findings_path`: Path to `<exp_root>/reports/research-findings.md`
- `selected_indices`: List of proposal indices to implement (1-based)
- `project_root`: Project root directory

## Step 1: Load Proposals

> **Goal check:** Verify each proposal's changes are within the `scope_level` from the optimization goals. Training scope = no model architecture changes. Architecture scope = model files allowed. Full scope = all changes allowed.

Parse the research findings file for selected proposals:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/implement_utils.py \
  <findings_path> '<selected_indices_json>'
```

This returns structured proposals with names, slugs, files to modify, and implementation steps.

If no findings file exists, ask the user to run the `ml-optimizer:research` skill first.

## Step 1.1: Classify Proposals by Strategy

Group proposals by their `implementation_strategy` field:

- **`from_reference` proposals:** Will require cloning a reference repo. If multiple proposals share the same `reference_repo` URL, clone it once and reuse.
- **`from_scratch` proposals:** Will be implemented from paper descriptions and implementation steps only.

Note: Proposals without an `implementation_strategy` field default to `from_scratch` (backward compatibility).

## Step 2: Detect Conflicts

Check for proposals that modify the same files:

From the CLI output, inspect the `conflicts` array. If conflicts exist:

1. **Inform the user** which proposals conflict and on which files
2. **Recommend** implementing conflicting proposals on separate branches (which is the default)
3. **Warn** that merging conflicting branches later may require manual conflict resolution

Proceed with implementation — git branch isolation handles conflicts naturally.

## Step 3: Determine Isolation Strategy

Check if the project is a git repository:

```bash
cd <project_root> && git rev-parse --is-inside-work-tree 2>/dev/null
```

**If git repo (preferred):**
- `strategy = "git_branch"`
- Record `original_branch` via `git -C <project_root> rev-parse --abbrev-ref HEAD`
- Implementation runs in a dedicated **git worktree** (Step 3.1) created from `<original_branch>`'s **commit**, so the project's main working tree is never touched. **No need to stash** — uncommitted changes in the main tree stay there and cannot contaminate the proposal branches (the worktree is a clean checkout of the commit).
- Each proposal gets branch `ml-opt/<slug>`

**If not a git repo (fallback):**
- `strategy = "file_backup"`
- Back up files to `<exp_root>/backups/<slug>/` before each modification
- Apply changes sequentially, validating after each

## Step 3.1: Set Up the Implementation Worktree (git strategy)

Do the implementation inside a git **worktree** so the project's main working tree is never disturbed. Put it **outside** `<exp_root>/` (a worktree nested under `<exp_root>/` can be wiped by a cleanup race):
```bash
PROJECT_HASH=$(echo "<project_root>" | sha1sum | cut -c1-8)
WORKTREE_ROOT="/tmp/ml-opt-impl-worktrees-${PROJECT_HASH}"; mkdir -p "$WORKTREE_ROOT"
WORKTREE_PATH="$WORKTREE_ROOT/impl"
# --detach: <original_branch> is checked out in the main tree, so attach detached at its commit.
git -C <project_root> worktree add --detach "$WORKTREE_PATH" <original_branch>
cd "$WORKTREE_PATH"
```

Implement all selected proposals here **sequentially**, one `ml-opt/<slug>` branch each (Step 4 loop). After the last proposal, remove the worktree — the `ml-opt/<slug>` branches (with commits) persist in the repo for the experiment loop:
```bash
cd - >/dev/null
git -C <project_root> worktree list --porcelain | grep -q "worktree $WORKTREE_PATH$" \
  && git -C <project_root> worktree remove "$WORKTREE_PATH" || git -C <project_root> worktree prune
```

**Backup strategy (non-git project):** skip the worktree — back up and edit files in `<project_root>` directly (Step 4).

## Step 3.2: Pre-Flight File Existence Validation

Before starting any implementation (parallel or sequential), validate that all target files exist:

**File-backup strategy note:** For `strategy == "file_backup"`, pre-flight validation is critical because there is no branch isolation. If implementation fails mid-way, the working directory may be corrupted. Verify the baseline backup (`<exp_root>/backups/_baseline/`) is intact before proceeding with other proposals.

For each proposal:

1. **Check every path in `files_to_modify`:**
   - Use Glob or `test -f` to verify each file exists under `<project_root>`

2. **If any file is missing:**
   a. **Search for similar files** using Glob: `**/<filename>` and `**/<filename_stem>*<extension>`
      - If matches found: log candidates to dev_notes
   b. **Classify viability:**
      - If ALL `files_to_modify` are missing: mark proposal as `status: "preflight_failed"`, set `notes: "All target files missing"`. Skip this proposal entirely. Log to error tracker: `category: "implementation_error", severity: "warning", source: "implement", message: "Pre-flight failed: all files missing for proposal <name>"`
      - If SOME missing but others exist: log warning to dev_notes. Check if proposal's implementation steps mention creating these files (expected-missing). If so, proceed. If not, log the gap but still attempt implementation.

3. **Remove preflight-failed proposals** from the active list before implementing (Step 4). Include them in the manifest with `status: "preflight_failed"`.

## Step 3.3: Use the GitNexus Code Graph for the Target Project (REQUIRED)

The target `<project_root>` was already indexed at Phase 2 (graph at `<project_root>/.gitnexus`) — GitNexus is a HARD PREREQUISITE guaranteed by Phase 2. Before modifying any code you MUST use the gitnexus MCP tools to understand the code and scope edits precisely — confirm what depends on the code each proposal will change before touching it. There is NO grep/`analyze` substitute for this code understanding.

First confirm the graph is available (it should be, from Phase 2). If it is somehow missing, re-index the **main `<project_root>`** (read-only structural analysis — it does not modify source), not the throwaway worktree:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py available || { echo "HALT: gitnexus unavailable — was guaranteed by Phase 2"; exit 1; }
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py is-indexed <project_root> || \
  python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py index <project_root>
```

Before editing each proposal's `files_to_modify` you MUST use `mcp__gitnexus__impact` to assess the blast radius of the planned change and `mcp__gitnexus__context` (and `mcp__gitnexus__query` as needed) to understand the surrounding code, so edits stay minimal and don't break callers.

**HARD ERROR (not a fallback):** if `gitnexus_utils.py available` exits non-zero, or re-indexing returns `success: false`, **halt** and report it as a prerequisite/repair error — gitnexus was guaranteed installed and the target indexed at Phase 2, so its absence here is an error state, not a condition to silently route around. Surface the install/repair guidance: `npm install -g gitnexus && gitnexus setup` (`gitnexus setup` auto-registers the gitnexus MCP server for Claude Code; manual MCP-registration fallback: `claude mcp add --transport stdio --scope user gitnexus gitnexus mcp`), then re-index via `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py index <project_root>`. Do NOT fall back to `Read`/`Grep`/`Glob`/`analyze` for code understanding. (`implement_utils.py analyze` remains available only for framework detection, never as a code-graph substitute.)

**MCP-query-failure recovery:** Querying the code graph is MCP-only by design. If a `mcp__gitnexus__*` tool call FAILS *after* a successful index, it is a HARD ERROR — there is NO grep fallback and NO gitnexus-CLI query fallback. Recovery: (1) ensure the gitnexus MCP server is registered — run `gitnexus setup` (or the manual `claude mcp add --transport stdio --scope user gitnexus gitnexus mcp`); (2) MCP tools load at session start, so restart the Claude Code session for a freshly-registered server to become available; (3) retry the query. If it still fails, halt and surface this to the user.

**Do not commit `.gitnexus/`:** indexing writes `<project_root>/.gitnexus/` (the wrapper auto-adds it to the repo's git exclude on success). It is a generated artifact — do not `git add` it, and ensure it is never included in any `ml-opt/<slug>` branch commit (Step 4g commits only the proposal's `<modified_files>`, so this is satisfied by committing explicit paths, not `git add .`).

## Step 4: Implement Each Proposal (Sequential)

For each selected proposal, in order.

> **Working directory:** for the **git strategy** you are inside the Step 3.1 worktree (`$WORKTREE_PATH`) — that is where the modified code lives, so run all edits, validation, tests, and commits from here. Where a command below shows `<project_root>`, use `$WORKTREE_PATH` (the worktree) under the git strategy; under the **file-backup strategy** there is no worktree, so it means the real `<project_root>`.

### 4a. Set up isolation

**Git strategy** (inside the Step 3.1 worktree): create the proposal's branch directly off `<original_branch>` — do **not** `git checkout <original_branch>` first (it's checked out in the main tree; branching from its commit-ish is allowed and avoids the conflict):
```bash
git checkout -b ml-opt/<slug> <original_branch>
```

**Backup strategy:**
```bash
python3 -c "from implement_utils import backup_files; ..."
```

### 4b. Check implementation strategy

Check the proposal's `implementation_strategy` field and follow the appropriate path:

#### Path A: `from_reference` (Code Adaptation)

Follow `${CLAUDE_SKILL_DIR}/references/implementation-patterns.md` Section 9.

0. **Pre-clone exploration (alphaxiv):**

   Before cloning, use alphaxiv to explore the repo remotely:
   ```
   mcp__alphaxiv__read_files_from_github_repository(githubUrl: "<reference_repo_url>", path: "/")
   ```
   This returns the file tree + top-level files (README, LICENSE) in one call. Use it to:
   - **Verify `reference_files`** from the proposal actually exist. If they don't, explore directories to find the correct paths.
   - **Check LICENSE** directly. If restrictive or missing, set `license_warning` immediately.
   - **Assess scope**: If the relevant code is in 2-3 files with no complex internal dependencies, read them directly via alphaxiv and skip local cloning entirely.

   If the proposal's implementation steps are ambiguous and a source paper URL is available, clarify with:
   ```
   mcp__alphaxiv__answer_pdf_queries(
     urls: ["<source_paper_url>"],
     queries: ["What is the exact implementation of the proposed technique?", "What hyperparameters are introduced and what are the defaults?"]
   )
   ```

   **Fallback:** If alphaxiv is unavailable, proceed directly to step 1 (local cloning).

1. **Clone reference repo:**
   ```bash
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/implement_utils.py clone <reference_repo_url> <exp_root>/reference-repos/<slug>
   ```
   If multiple proposals share the same repo, clone once and reuse.

2. **Detect framework (narrow use of `analyze`):**
   ```bash
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/implement_utils.py analyze <exp_root>/reference-repos/<slug>
   ```
   This is for framework detection only — it is NOT a substitute for the code-graph understanding below.

3. **Index the reference repo with GitNexus and understand it via the code graph (REQUIRED):** EVERY reference repo MUST be indexed immediately after clone. Index it through the wrapper (it runs `gitnexus analyze <path> --index-only`, which keeps the cloned reference repo uncontaminated — it does NOT inject a GitNexus section into the repo's CLAUDE.md/AGENTS.md and does NOT install `.claude/` skills) and use the gitnexus MCP tools to locate the core implementation and its internal dependencies — you MUST understand the reference repo through the code graph before adapting any code (no `analyze`/`Read`/`Grep` substitute for this understanding):
   ```bash
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py available || { echo "HALT: gitnexus unavailable — was guaranteed by Phase 2"; exit 1; }
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py is-indexed <exp_root>/reference-repos/<slug> || \
     python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py index <exp_root>/reference-repos/<slug>
   ```
   After indexing, use `mcp__gitnexus__context`, `mcp__gitnexus__query`, and `mcp__gitnexus__impact` to understand the repo's structure and dependency edges before extracting/adapting code.

   **HARD ERROR (not a fallback):** if `available` exits non-zero or indexing returns `success: false`, **halt** with install/repair guidance: `npm install -g gitnexus && gitnexus setup` (`gitnexus setup` auto-registers the gitnexus MCP server for Claude Code; manual MCP-registration fallback: `claude mcp add --transport stdio --scope user gitnexus gitnexus mcp`), then re-index via `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py index <exp_root>/reference-repos/<slug>`. gitnexus was guaranteed by Phase 2, so its absence here is an error state — do NOT route around it with `analyze`/`Read`/`Grep` for code understanding.

   **MCP-query-failure recovery:** Querying the code graph is MCP-only by design. If a `mcp__gitnexus__*` tool call FAILS *after* a successful index, it is a HARD ERROR — there is NO grep fallback and NO gitnexus-CLI query fallback. Recovery: (1) ensure the gitnexus MCP server is registered — run `gitnexus setup` (or the manual `claude mcp add --transport stdio --scope user gitnexus gitnexus mcp`); (2) MCP tools load at session start, so restart the Claude Code session for a freshly-registered server to become available; (3) retry the query. If it still fails, halt and surface this to the user.

   **Do not commit the generated `<exp_root>/reference-repos/<slug>/.gitnexus/` artifact** (the wrapper auto-adds it to the repo's git exclude, and the repo is cleaned up in step 8 anyway).

4. **Read reference code:** Read the files listed in the proposal's `reference_files` (guided by the gitnexus code graph from step 3). Identify the core implementation, internal dependencies, and external packages.

5. **Read target files:** Read every file listed in the proposal's `files_to_modify` to understand the existing code structure.

6. **Adapt and apply:**
   - Extract only the relevant functions/classes from the reference
   - Adapt imports, framework calls, and tensor conventions to the target project
   - Apply changes using Edit, keeping modifications minimal
   - Add provenance comments: `# [ml-opt] Adapted from <url>, file: <original_path>`
   - Add license comment: `# [ml-opt] License: <license_type>`

7. **Check license:** Read the LICENSE file in the cloned repo. If missing or restrictive, note `license_warning` for the manifest.

8. **Cleanup:** Remove the cloned repo:
   ```bash
   python3 -c "
   import sys; # sys.path: add the plugin's scripts/ directory
   from implement_utils import cleanup_reference_repo
   cleanup_reference_repo('<exp_root>/reference-repos/<slug>')
   "
   ```

#### Path B: `from_scratch` (Paper-Based)

Follow `${CLAUDE_SKILL_DIR}/references/implementation-patterns.md` Sections 1-8.

1. **Read implementation patterns:** Find the matching category for this proposal:
   - Loss function changes → Section 1
   - Architecture changes → Section 2
   - Data augmentation → Section 3
   - Training strategy → Section 4
   - Regularization → Section 5
   - Paper-based implementation → Section 8
   Follow the "what to read first" and "minimal change pattern" guidance.

2. **If steps are ambiguous:** If the proposal's implementation steps are vague and a paper URL is available in the Source field, use WebFetch to re-read the paper for clarification before proceeding.

3. **Read target files:** Before modifying, **read every file** listed in the proposal's `files_to_modify`. Understand the current code structure, where changes should be inserted, and what surrounding code depends on.

4. **Apply changes:** Follow the proposal's implementation steps exactly:
   - Use Edit to apply each change
   - Keep changes minimal — only what the proposal specifies
   - Add a comment marking the change: `# [ml-opt] <proposal_name>`
   - If the proposal requires a new file (e.g., a new module), use Write

**Important rules (both paths):**
- Do NOT improvise changes beyond what the proposal specifies
- Do NOT refactor surrounding code
- Do NOT change configs unless the proposal explicitly requires it
- If a step is unclear, stop and report it rather than guessing

### 4e. Validate

Read `${CLAUDE_SKILL_DIR}/references/validation-checklist.md` and run checks progressively:

**Mandatory (always run):**

1. **Syntax check:**
   ```bash
   python3 -c "
   import sys; # sys.path: add the plugin's scripts/ directory
   from implement_utils import validate_syntax
   import json; print(json.dumps(validate_syntax([<file_list>]), indent=2))
   "
   ```

2. **Import check:**
   ```bash
   python3 -c "
   import sys; # sys.path: add the plugin's scripts/ directory
   from implement_utils import validate_imports
   import json; print(json.dumps(validate_imports('<module_path>', '<project_root>')))
   "
   ```

3. **LSP static check (Pyright):** For each modified `.py` file, use the `LSP` tool to get diagnostics. This catches undefined names, type mismatches, wrong call signatures, and unresolved imports **statically** — before any GPU time. Treat **errors** as blocking (fix and re-check); **warnings** are advisory (log to dev_notes). If the `LSP` tool is unavailable (pyright not installed), skip and note it — the syntax/import checks above still apply.

**Recommended (run if project supports it):**

4. Model instantiation check — attempt if the project has a model factory function (e.g., `get_model()`)
5. Forward pass shape check — attempt if model instantiation succeeds

See `${CLAUDE_SKILL_DIR}/references/validation-checklist.md` for commands. Attempt Level 3 validation when the project structure supports it (e.g., has a clear model factory or config-based instantiation).

### 4e.5. Code Quality Gate

Your quality gate is the syntax/import/**LSP** checks above (step 4e) — record the result in the manifest's `validation` block.

### 4f. Write Unit Tests

After validation passes (at least Level 1-2), write a focused unit test for the implemented proposal.

**Test file location:** `<exp_root>/tests/test_<slug>.py`

**What to test (choose based on proposal type):**

| Proposal Type | Test Focus |
|---------------|-----------|
| Loss function | Finite scalar output, correct shape, edge cases (zero input, batch=1) |
| Augmentation | Output shape == input shape, batch operation, value range |
| Architecture module | Forward pass shape, parameter count, gradient flow |
| Scheduler/optimizer | Instantiation, step without error, LR changes as expected |
| Regularization | Active in train mode, inactive in eval (if applicable) |
| Data pipeline | Output shape/dtype correct, batch iteration works |

**Test template:**

```python
"""Unit tests for ml-opt proposal: <proposal_name>."""
import pytest

# Framework-specific imports based on detected framework
# PyTorch: import torch
# TF/Keras: import tensorflow as tf
# sklearn: import numpy as np


class Test<ProposalClassName>:
    """Tests for <proposal_name> implementation."""

    def test_output_shape(self):
        """Verify output shape matches expected dimensions."""
        ...

    def test_no_nan(self):
        """Verify no NaN/Inf in output."""
        ...

    def test_edge_case(self):
        """Verify behavior with edge-case inputs."""
        ...
```

**Constraints:**
- Max 50 lines per test file
- No external test data — generate inline with `torch.randn`, `np.random`, etc.
- Each test function must complete in <5 seconds
- Import only the specific module/function being tested, not the full model

**Run tests** (from the worktree, so the test imports the *modified* code):
```bash
python3 -m pytest <exp_root>/tests/test_<slug>.py -v --timeout=30 2>&1 | head -50
```

**Record results in manifest:**
- Add `"unit_tests"` field to the proposal's `validation` block: `"pass"`, `"fail"`, or `"skipped"`
- Add `"test_file"` field to the proposal: path to the test file

**If tests fail:** Log a warning but do NOT mark the proposal as `validation_failed`. Test failures are informational — they suggest the implementation may have issues but don't block experimentation.

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"implementation_error","severity":"info","source":"implement","message":"Unit tests failed for proposal <name>: <failure_summary>","context":{"proposal_name":"<name>","test_file":"<exp_root>/tests/test_<slug>.py"}}'
```

**If test writing is not feasible** (e.g., the proposal only modifies config files, or the changed module requires complex setup that can't be isolated): set `validation.unit_tests: "skipped"` and `test_file: null`. Log reason in the proposal's `notes` field.

### 4g. Commit (git strategy only)

```bash
git add <modified_files>
git commit -m "ml-opt: implement <proposal_name>"
```

Record the commit SHA for the manifest.

### 4g.1. Extract diff summary and write explanation

After committing, extract a structured diff summary for the dashboard:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/implement_utils.py diff <project_root> <branch>
```

Store the result as `diff_summary` in the manifest proposal entry.

Also write an `explanation` field — a 1-2 sentence plain-language description of what the code change does and why it should improve the primary metric. For example: "Replaces CrossEntropyLoss with FocalLoss to better handle class imbalance, which should improve accuracy on minority classes." This explanation is shown in the live dashboard to help users understand each method without reading code.

### 4h. Reset for the next proposal

**Git strategy:** nothing to reset — the next proposal's 4a branches fresh from `<original_branch>` (`git checkout -b ml-opt/<next-slug> <original_branch>`).

**Backup strategy:** restore the baseline backup before the next proposal.

## Step 5: Write Implementation Manifest

Write `<exp_root>/results/implementation-manifest.json`:

```json
{
  "original_branch": "<branch name>",
  "strategy": "git_branch|file_backup",
  "proposals": [
    {
      "name": "Perceptual Loss Function",
      "slug": "perceptual-loss-function",
      "branch": "ml-opt/perceptual-loss-function",
      "status": "validated|validation_failed|implementation_error",
      "files_modified": ["path/to/file1.py", "path/to/file2.py"],
      "files_created": ["path/to/new_module.py"],
      "complexity": "Low",
      "implementation_strategy": "from_scratch|from_reference",
      "reference_repo": "https://github.com/...",
      "reference_files_used": ["path/in/repo.py"],
      "adaptation_notes": "Translated from TF to PyTorch",
      "license_warning": null,
      "validation": {
        "syntax": "pass|fail",
        "import": "pass|fail",
        "model_instantiate": "pass|fail|skipped",
        "forward_pass": "pass|fail|skipped",
        "unit_tests": "pass|fail|skipped"
      },
      "test_file": "<exp_root>/tests/test_<slug>.py|null",
      "explanation": "Plain-language description of what changed and why it should improve the metric",
      "diff_summary": {"files_changed": 2, "lines_added": 45, "lines_removed": 10, "changed_functions": ["train_step", "compute_loss"]},
      "commit_sha": "abc123...",
      "notes": "Any observations or warnings"
    }
  ],
  "conflicts": [
    {
      "file": "path/to/shared_file.py",
      "proposal_indices": [1, 2]
    }
  ],
  "new_dependencies": []
}
```

Use the helper:
```python
from implement_utils import write_manifest
write_manifest("<exp_root>/results/implementation-manifest.json", manifest_data)
```

## Step 5.1: Validate Manifest

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/schema_validator.py \
  <exp_root>/results/implementation-manifest.json manifest
```

If validation fails, fix the manifest and re-validate before proceeding.

## Step 6: Write Dev Notes

Write a summary to `<exp_root>/reports/implementation-summary.md`:

```markdown
# Implementation Summary

## Date: <date>
## Proposals Implemented: N/M

### <Proposal Name>
- Branch: ml-opt/<slug>
- Status: validated / validation_failed
- Files modified: <list>
- Notes: <any observations>

### <Next Proposal>
...

## Conflicts Detected
- <file>: touched by proposals [X, Y]

## New Dependencies
- <package>: required by <proposal> (NOT installed — needs user confirmation)
```

## Step 7: Report Back

Return to the orchestrator or user:

```
Implementation complete:
- Implemented: X/Y proposals
- Validated: Z/Y proposals
- Conflicts: N files shared between proposals

Manifest: <exp_root>/results/implementation-manifest.json

Validated branches ready for experiments:
- ml-opt/<slug-1> (Proposal 1: <name>)
- ml-opt/<slug-2> (Proposal 3: <name>)

[If any failed:]
Failed validation:
- ml-opt/<slug-3> (Proposal 2: <name>) — syntax error in <file>

[If new dependencies:]
New dependencies needed (install before experiments):
- <package>: pip install <package>
```

## Error Handling

- **File not found (during implementation):** If a file listed in the proposal doesn't exist, report it and skip that file. Mark the proposal as `implementation_error`. Log to error tracker:
  ```bash
  python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"implementation_error","severity":"warning","source":"implement","message":"File not found: <file_path> for proposal <name>","context":{"proposal_name":"<name>","proposal_slug":"<slug>"}}'
  ```
- **Syntax validation fails:** Keep the branch as-is (for debugging). Mark as `validation_failed`. The experiment skill will skip it. Log to error tracker:
  ```bash
  python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"implementation_error","severity":"warning","source":"implement","message":"Syntax validation failed for proposal <name>","context":{"proposal_name":"<name>","proposal_slug":"<slug>","files_modified":["<files>"]}}'
  ```
- **Import validation fails:** Check if a new dependency is needed. Flag it in `new_dependencies`. Mark as `validation_failed`. Log to error tracker:
  ```bash
  python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"implementation_error","severity":"warning","source":"implement","message":"Import validation failed for proposal <name>","context":{"proposal_name":"<name>","proposal_slug":"<slug>","implementation_strategy":"<strategy>"}}'
  ```
- **Git conflicts on branch creation:** If `ml-opt/<slug>` already exists, use a while loop to find an available name: `ml-opt/<slug>-2`, `ml-opt/<slug>-3`, etc.
- **Not a git repo and no backup possible:** Report to user, do not proceed with modifications.

## Non-Git Fallback Details

When using `file_backup` strategy:
1. **Before the first proposal:** Create a baseline backup of ALL files that ANY proposal will modify → `<exp_root>/backups/_baseline/`. This is the clean reference state.
2. **Before each proposal:** Restore ALL target files from `<exp_root>/backups/_baseline/` first (ensures a clean slate). Then apply this proposal's changes.
3. Validate
4. If validation fails: restore from baseline backup
5. If validation passes: backup the modified state to `<exp_root>/backups/<slug>/`, then restore from baseline backup (return to clean state before next proposal)
6. The manifest records backup paths instead of branch names

**Critical:** The restore-before-apply pattern (step 2) prevents proposal A's changes from leaking into proposal B's code. Each proposal is validated and backed up independently against the original code.

**Limitation:** With file backup, proposals cannot be tested in parallel. The experiment skill must restore each proposal's backup before running its experiments, then restore baseline before the next proposal's experiments.
