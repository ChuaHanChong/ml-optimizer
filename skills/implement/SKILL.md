---
name: implement
description: "Apply research proposals as code changes. Creates isolated git branches per proposal, implements modifications (architecture, loss, augmentation), validates with progressive checks, and produces a manifest for the experiment loop. Use when: research proposals need to be turned into actual code changes, or when implementing specific ML improvements before experiments."
---

# Implement Research Proposals

Apply research proposals as actual code changes with git branch isolation, progressive validation, and a structured manifest for the experiment loop.

Use extended thinking for all analytical reasoning in this skill. Ultrathink. Think through implementation approaches, potential side effects, validation strategies, and backwards compatibility before making code changes.

## Important Files

- Implementation patterns: `references/implementation-patterns.md` (in this skill's directory)
- Validation checklist: `references/validation-checklist.md` (in this skill's directory)
- Python helpers: `~/.claude/plugins/ml-optimizer/scripts/implement_utils.py`

## Inputs Expected

From the orchestrator or direct invocation:
- `findings_path`: Path to `experiments/reports/research-findings.md`
- `selected_indices`: List of proposal indices to implement (1-based)
- `project_root`: Project root directory

## Step 1: Load Proposals

Parse the research findings file for selected proposals:

```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/implement_utils.py \
  <findings_path> '<selected_indices_json>'
```

This returns structured proposals with names, slugs, files to modify, and implementation steps.

If no findings file exists, ask the user to run the `ml-optimizer:research` skill first.

## Step 1.5: Classify Proposals by Strategy

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
- Record `original_branch` via `git rev-parse --abbrev-ref HEAD`
- **Check for uncommitted changes:**
  ```bash
  git status --porcelain
  ```
  If output is non-empty (dirty working tree), use AskUserQuestion:
  ```
  Your working tree has uncommitted changes. These changes would be carried into
  all proposal branches, which could contaminate the baseline comparison.

  Please either:
  1. Commit your changes: git commit -am "WIP"
  2. Stash your changes: git stash

  Then re-run the implement skill.
  ```
  Do NOT proceed with branch creation on a dirty working tree.
- Each proposal gets branch `ml-opt/<slug>`

**If not a git repo (fallback):**
- `strategy = "file_backup"`
- Back up files to `experiments/backups/<slug>/` before each modification
- Apply changes sequentially, validating after each

## Step 4: Implement Each Proposal

For each selected proposal, in order:

### 4a. Set up isolation

**Git strategy:**
```bash
git checkout <original_branch>
git checkout -b ml-opt/<slug>
```

**Backup strategy:**
```bash
python3 -c "from implement_utils import backup_files; ..."
```

### 4b. Check implementation strategy

Check the proposal's `implementation_strategy` field and follow the appropriate path:

#### Path A: `from_reference` (Code Adaptation)

Follow `references/implementation-patterns.md` Section 9.

1. **Clone reference repo:**
   ```bash
   python3 ~/.claude/plugins/ml-optimizer/scripts/implement_utils.py clone <reference_repo_url> experiments/reference-repos/<slug>
   ```
   If multiple proposals share the same repo, clone once and reuse.

2. **Analyze repo structure:**
   ```bash
   python3 ~/.claude/plugins/ml-optimizer/scripts/implement_utils.py analyze experiments/reference-repos/<slug>
   ```

3. **Read reference code:** Read the files listed in the proposal's `reference_files`. Identify the core implementation, internal dependencies, and external packages.

4. **Read target files:** Read every file listed in the proposal's `files_to_modify` to understand the existing code structure.

5. **Adapt and apply:**
   - Extract only the relevant functions/classes from the reference
   - Adapt imports, framework calls, and tensor conventions to the target project
   - Apply changes using Edit, keeping modifications minimal
   - Add provenance comments: `# [ml-opt] Adapted from <url>, file: <original_path>`
   - Add license comment: `# [ml-opt] License: <license_type>`

6. **Check license:** Read the LICENSE file in the cloned repo. If missing or restrictive, note `license_warning` for the manifest.

7. **Cleanup:** Remove the cloned repo:
   ```bash
   python3 -c "
   import sys; sys.path.insert(0, '$HOME/.claude/plugins/ml-optimizer/scripts')
   from implement_utils import cleanup_reference_repo
   cleanup_reference_repo('experiments/reference-repos/<slug>')
   "
   ```

#### Path B: `from_scratch` (Paper-Based)

Follow `references/implementation-patterns.md` Sections 1-8.

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

Read `references/validation-checklist.md` and run checks progressively:

**Mandatory (always run):**

1. **Syntax check:**
   ```bash
   python3 -c "
   import sys; sys.path.insert(0, '$HOME/.claude/plugins/ml-optimizer/scripts')
   from implement_utils import validate_syntax
   import json; print(json.dumps(validate_syntax([<file_list>]), indent=2))
   "
   ```

2. **Import check:**
   ```bash
   python3 -c "
   import sys; sys.path.insert(0, '$HOME/.claude/plugins/ml-optimizer/scripts')
   from implement_utils import validate_imports
   import json; print(json.dumps(validate_imports('<module_path>', '<project_root>')))
   "
   ```

**Recommended (run if project supports it):**

3. Model instantiation check — attempt if the project has a model factory function (e.g., `get_model()`)
4. Forward pass shape check — attempt if model instantiation succeeds

See `references/validation-checklist.md` for commands. Attempt Level 3 validation when the project structure supports it (e.g., has a clear model factory or config-based instantiation).

### 4f. Commit (git strategy only)

```bash
git add <modified_files>
git commit -m "ml-opt: implement <proposal_name>"
```

Record the commit SHA for the manifest.

### 4g. Return to original branch

```bash
git checkout <original_branch>
```

Then proceed to the next proposal.

## Step 5: Write Implementation Manifest

Write `experiments/results/implementation-manifest.json`:

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
        "forward_pass": "pass|fail|skipped"
      },
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
write_manifest("experiments/results/implementation-manifest.json", manifest_data)
```

## Step 6: Write Dev Notes

Write a summary to `experiments/reports/implementation-summary.md`:

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

Manifest: experiments/results/implementation-manifest.json

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

- **File not found:** If a file listed in the proposal doesn't exist, report it and skip that file. Mark the proposal as `implementation_error`.
- **Syntax validation fails:** Keep the branch as-is (for debugging). Mark as `validation_failed`. The experiment skill will skip it.
- **Import validation fails:** Check if a new dependency is needed. Flag it in `new_dependencies`. Mark as `validation_failed`.
- **Git conflicts on branch creation:** If `ml-opt/<slug>` already exists, use a while loop to find an available name: `ml-opt/<slug>-2`, `ml-opt/<slug>-3`, etc.
- **Not a git repo and no backup possible:** Report to user, do not proceed with modifications.

## Non-Git Fallback Details

When using `file_backup` strategy:
1. Before each proposal: backup all target files to `experiments/backups/<slug>/`
2. Apply changes to the original files
3. Validate
4. If validation fails: restore from backup
5. If validation passes: leave changes in place, but note that only ONE proposal can be active at a time
6. The manifest records backup paths instead of branch names

**Limitation:** With file backup, proposals cannot be tested in parallel. The experiment skill must restore backups between runs.
