# Method Stacking Phase Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Phase 6.5 stacking phase that sequentially merges winning methods to capture compound gains, with skip-on-failure and per-step attribution.

**Architecture:** After independent method testing (Phase 6), rank methods by improvement over baseline. Create `ml-opt/stack-N` branches by sequentially merging the best methods. Test each combo — keep if improved, skip if worse. LLM-driven conflict resolution for merge conflicts.

**Tech Stack:** Python 3.10+ stdlib only (scripts), markdown skill definitions, pytest

---

## Chunk 1: Schema & Utility Foundation

### Task 1: Extend schema_validator with stacked tier values

**Files:**
- Modify: `scripts/schema_validator.py:24`
- Test: `tests/test_schema_validator.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_schema_validator.py`, add after the existing `test_validate_result_invalid_method_tier` test:

```python
def test_validate_result_stacked_method_tiers():
    """Stacked method tier values pass validation."""
    for tier in ["stacked_default_hp", "stacked_tuned_hp"]:
        data = {
            "exp_id": "exp-stack-001", "status": "completed",
            "config": {"lr": 0.001}, "metrics": {"loss": 0.5},
            "method_tier": tier,
            "code_branches": ["ml-opt/perceptual-loss", "ml-opt/cosine-scheduler"],
            "stacking_order": 2,
            "stack_base_exp": "exp-012",
        }
        result = validate_result(data)
        assert result["valid"] is True, f"method_tier '{tier}' should be valid"


def test_validate_result_code_branches_must_be_list():
    """code_branches must be a list if present."""
    data = {
        "exp_id": "exp-stack-001", "status": "completed",
        "config": {}, "metrics": {"loss": 0.5},
        "code_branches": "not-a-list",
    }
    result = validate_result(data)
    assert result["valid"] is False
    assert any("code_branches" in e for e in result["errors"])


def test_validate_result_stacking_order_must_be_int():
    """stacking_order must be a positive integer if present."""
    data = {
        "exp_id": "exp-stack-001", "status": "completed",
        "config": {}, "metrics": {"loss": 0.5},
        "stacking_order": "two",
    }
    result = validate_result(data)
    assert result["valid"] is False
    assert any("stacking_order" in e for e in result["errors"])


def test_validate_result_stacking_order_zero():
    """stacking_order of 0 fails validation (must be >= 1)."""
    data = {
        "exp_id": "exp-stack-001", "status": "completed",
        "config": {}, "metrics": {"loss": 0.5},
        "stacking_order": 0,
    }
    result = validate_result(data)
    assert result["valid"] is False
    assert any("stacking_order" in e for e in result["errors"])


def test_validate_result_code_branches_elements_must_be_strings():
    """code_branches elements must be strings."""
    data = {
        "exp_id": "exp-stack-001", "status": "completed",
        "config": {}, "metrics": {"loss": 0.5},
        "code_branches": [1, 2, 3],
    }
    result = validate_result(data)
    assert result["valid"] is False
    assert any("code_branches" in e for e in result["errors"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `eval "$(conda shell.bash hook 2>/dev/null)" && conda activate base && python -m pytest tests/test_schema_validator.py::test_validate_result_stacked_method_tiers tests/test_schema_validator.py::test_validate_result_code_branches_must_be_list tests/test_schema_validator.py::test_validate_result_stacking_order_must_be_int -v`

Expected: FAIL — `stacked_default_hp` not in `VALID_METHOD_TIERS`, no `code_branches` validation.

- [ ] **Step 3: Implement schema changes**

In `scripts/schema_validator.py`:

1. Add stacked tiers to the valid list (line 24):
```python
VALID_METHOD_TIERS = ["baseline", "method_default_hp", "method_tuned_hp", "stacked_default_hp", "stacked_tuned_hp"]
```

2. Add stacking fields to the optional list (line 19-23):
```python
EXPERIMENT_RESULT_OPTIONAL = [
    "gpu_id", "duration_seconds", "log_file", "script_file",
    "code_branch", "code_proposal", "notes",
    "method_tier", "proposal_source", "iteration",
    "code_branches", "stacking_order", "stack_base_exp",
]
```

3. Add validation for new fields inside `validate_result()`, after the `method_tier` check (after line 110):
```python
    if "code_branches" in data:
        if not isinstance(data["code_branches"], list):
            errors.append("'code_branches' must be a list")
        elif not all(isinstance(b, str) for b in data["code_branches"]):
            errors.append("'code_branches' elements must be strings")

    if "stacking_order" in data:
        if not isinstance(data["stacking_order"], int) or data["stacking_order"] < 1:
            errors.append("'stacking_order' must be a positive integer")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `eval "$(conda shell.bash hook 2>/dev/null)" && conda activate base && python -m pytest tests/test_schema_validator.py -v`

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/schema_validator.py tests/test_schema_validator.py
git commit -m "feat: add stacked method tier values and stacking fields to schema"
```

---

### Task 2: Add rank_methods_for_stacking() to result_analyzer

**Files:**
- Modify: `scripts/result_analyzer.py:211` (after `group_by_method_tier`)
- Test: `tests/test_result_analyzer.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_result_analyzer.py`:
1. Add `rank_methods_for_stacking` to the existing import on line 9 (append to the existing `from result_analyzer import ...` statement).
2. Add the following tests at the end of the file:

```python
# (rank_methods_for_stacking is already imported at the top of the file)


def test_rank_methods_for_stacking_basic(tmp_path):
    """Methods are ranked by improvement magnitude over baseline."""
    _write_results(tmp_path, {
        "baseline": {"metrics": {"loss": 1.0}, "config": {"lr": 0.01}},
        "exp-001": {"metrics": {"loss": 0.8}, "config": {"lr": 0.01},
                    "code_branch": "ml-opt/method-a", "code_proposal": "method-a",
                    "method_tier": "method_tuned_hp", "status": "completed"},
        "exp-002": {"metrics": {"loss": 0.6}, "config": {"lr": 0.001},
                    "code_branch": "ml-opt/method-b", "code_proposal": "method-b",
                    "method_tier": "method_tuned_hp", "status": "completed"},
        "exp-003": {"metrics": {"loss": 0.9}, "config": {"lr": 0.01},
                    "code_branch": "ml-opt/method-c", "code_proposal": "method-c",
                    "method_tier": "method_tuned_hp", "status": "completed"},
    })
    results = load_results(str(tmp_path))
    ranked = rank_methods_for_stacking(results, "loss", lower_is_better=True)
    # method-b improved most (0.6 vs 1.0), then method-a (0.8), then method-c (0.9)
    assert len(ranked) == 3
    assert ranked[0]["code_proposal"] == "method-b"
    assert ranked[1]["code_proposal"] == "method-a"
    assert ranked[2]["code_proposal"] == "method-c"


def test_rank_methods_excludes_non_improvements(tmp_path):
    """Methods that didn't improve over baseline are excluded."""
    _write_results(tmp_path, {
        "baseline": {"metrics": {"loss": 1.0}, "config": {"lr": 0.01}},
        "exp-001": {"metrics": {"loss": 0.8}, "config": {"lr": 0.01},
                    "code_branch": "ml-opt/method-a", "code_proposal": "method-a",
                    "method_tier": "method_tuned_hp", "status": "completed"},
        "exp-002": {"metrics": {"loss": 1.2}, "config": {"lr": 0.001},
                    "code_branch": "ml-opt/method-b", "code_proposal": "method-b",
                    "method_tier": "method_tuned_hp", "status": "completed"},
    })
    results = load_results(str(tmp_path))
    ranked = rank_methods_for_stacking(results, "loss", lower_is_better=True)
    assert len(ranked) == 1
    assert ranked[0]["code_proposal"] == "method-a"


def test_rank_methods_picks_best_per_branch(tmp_path):
    """When a branch has multiple experiments, uses the best result."""
    _write_results(tmp_path, {
        "baseline": {"metrics": {"loss": 1.0}, "config": {"lr": 0.01}},
        "exp-001": {"metrics": {"loss": 0.9}, "config": {"lr": 0.01},
                    "code_branch": "ml-opt/method-a", "code_proposal": "method-a",
                    "method_tier": "method_default_hp", "status": "completed"},
        "exp-002": {"metrics": {"loss": 0.7}, "config": {"lr": 0.001},
                    "code_branch": "ml-opt/method-a", "code_proposal": "method-a",
                    "method_tier": "method_tuned_hp", "status": "completed"},
    })
    results = load_results(str(tmp_path))
    ranked = rank_methods_for_stacking(results, "loss", lower_is_better=True)
    assert len(ranked) == 1
    assert ranked[0]["best_metric"] == 0.7
    assert ranked[0]["best_config"] == {"lr": 0.001}


def test_rank_methods_higher_is_better(tmp_path):
    """Ranking works with higher-is-better metrics."""
    _write_results(tmp_path, {
        "baseline": {"metrics": {"accuracy": 0.7}, "config": {}},
        "exp-001": {"metrics": {"accuracy": 0.9}, "config": {},
                    "code_branch": "ml-opt/method-a", "code_proposal": "method-a",
                    "method_tier": "method_tuned_hp", "status": "completed"},
        "exp-002": {"metrics": {"accuracy": 0.8}, "config": {},
                    "code_branch": "ml-opt/method-b", "code_proposal": "method-b",
                    "method_tier": "method_tuned_hp", "status": "completed"},
    })
    results = load_results(str(tmp_path))
    ranked = rank_methods_for_stacking(results, "accuracy", lower_is_better=False)
    assert ranked[0]["code_proposal"] == "method-a"  # 0.9 > 0.8


def test_rank_methods_empty_results(tmp_path):
    """Returns empty list when no methods improved."""
    _write_results(tmp_path, {
        "baseline": {"metrics": {"loss": 1.0}, "config": {}},
    })
    results = load_results(str(tmp_path))
    ranked = rank_methods_for_stacking(results, "loss", lower_is_better=True)
    assert ranked == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `eval "$(conda shell.bash hook 2>/dev/null)" && conda activate base && python -m pytest tests/test_result_analyzer.py::test_rank_methods_for_stacking_basic -v`

Expected: FAIL — `ImportError: cannot import name 'rank_methods_for_stacking'`

- [ ] **Step 3: Implement rank_methods_for_stacking()**

In `scripts/result_analyzer.py`, add after `group_by_method_tier()` (after line 211):

```python
def rank_methods_for_stacking(
    results: dict[str, dict],
    metric: str,
    lower_is_better: bool = True,
) -> list[dict]:
    """Rank methods by improvement over baseline for stacking.

    For each code_branch, finds the best experiment result. Excludes methods
    that didn't improve over baseline. Returns a list sorted by improvement
    magnitude (most improved first).

    Each entry contains: code_branch, code_proposal, best_metric,
    best_config, best_exp_id, improvement_pct.
    """
    # Get baseline metric
    baseline = results.get("baseline", {})
    baseline_metrics = baseline.get("metrics", baseline)
    if metric not in baseline_metrics:
        return []
    baseline_val = baseline_metrics[metric]

    # Group by code_branch, find best per branch
    branch_best: dict[str, dict] = {}
    for exp_id, data in results.items():
        if exp_id == "baseline":
            continue
        branch = data.get("code_branch")
        if not branch:
            continue
        # Skip failed/non-completed experiments
        status = data.get("status")
        if status is not None and status != "completed":
            continue
        exp_metrics = data.get("metrics", data)
        if metric not in exp_metrics:
            continue
        val = exp_metrics[metric]
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            continue

        if branch not in branch_best:
            branch_best[branch] = {
                "code_branch": branch,
                "code_proposal": data.get("code_proposal", branch.removeprefix("ml-opt/")),
                "best_metric": val,
                "best_config": data.get("config", {}),
                "best_exp_id": exp_id,
            }
        else:
            current = branch_best[branch]["best_metric"]
            better = val < current if lower_is_better else val > current
            if better:
                branch_best[branch]["best_metric"] = val
                branch_best[branch]["best_config"] = data.get("config", {})
                branch_best[branch]["best_exp_id"] = exp_id

    # Filter to methods that improved over baseline and compute improvement
    improved = []
    for entry in branch_best.values():
        val = entry["best_metric"]
        if lower_is_better:
            improved_over_baseline = val < baseline_val
        else:
            improved_over_baseline = val > baseline_val
        if not improved_over_baseline:
            continue
        if abs(baseline_val) < 1e-8:
            pct = None
        else:
            delta = baseline_val - val if lower_is_better else val - baseline_val
            pct = round(delta / abs(baseline_val) * 100, 2)
        entry["improvement_pct"] = pct
        improved.append(entry)

    # Sort by improvement magnitude (most improved first)
    def _sort_key(e):
        pct = e.get("improvement_pct")
        return pct if pct is not None else 0.0

    improved.sort(key=_sort_key, reverse=True)
    return improved
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `eval "$(conda shell.bash hook 2>/dev/null)" && conda activate base && python -m pytest tests/test_result_analyzer.py -v`

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/result_analyzer.py tests/test_result_analyzer.py
git commit -m "feat: add rank_methods_for_stacking() for sequential accumulation"
```

---

### Task 3: Extend group_by_method_tier for stacked tiers

**Files:**
- Modify: `scripts/result_analyzer.py:201-211`
- Test: `tests/test_result_analyzer.py`

- [ ] **Step 1: Write the failing test**

```python
def test_group_by_method_tier_stacked(tmp_path):
    """group_by_method_tier handles stacked tier values."""
    _write_results(tmp_path, {
        "baseline": {"metrics": {"loss": 1.0}, "method_tier": "baseline"},
        "exp-001": {"metrics": {"loss": 0.8}, "method_tier": "method_tuned_hp"},
        "exp-stack-001": {"metrics": {"loss": 0.6}, "method_tier": "stacked_default_hp",
                          "code_branches": ["ml-opt/a", "ml-opt/b"], "stacking_order": 2},
        "exp-stack-002": {"metrics": {"loss": 0.5}, "method_tier": "stacked_tuned_hp",
                          "code_branches": ["ml-opt/a", "ml-opt/b"], "stacking_order": 2},
    })
    results = load_results(str(tmp_path))
    groups = group_by_method_tier(results)
    assert "stacked_default_hp" in groups
    assert "stacked_tuned_hp" in groups
    assert len(groups["stacked_default_hp"]) == 1
    assert len(groups["stacked_tuned_hp"]) == 1
```

- [ ] **Step 2: Run test to verify it passes** (should already pass since `group_by_method_tier` is generic)

Run: `eval "$(conda shell.bash hook 2>/dev/null)" && conda activate base && python -m pytest tests/test_result_analyzer.py::test_group_by_method_tier_stacked -v`

Expected: PASS (existing implementation is generic enough). If so, commit directly.

- [ ] **Step 3: Commit**

```bash
git add tests/test_result_analyzer.py
git commit -m "test: verify group_by_method_tier handles stacked tiers"
```

---

### Task 4: Extend progress chart description for stacked experiments

**Files:**
- Modify: `scripts/result_analyzer.py` (`build_experiment_description`)
- Test: `tests/test_result_analyzer.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_description_stacked_experiment():
    """Stacked experiments show combined method names."""
    data = {
        "code_branches": ["ml-opt/perceptual-loss", "ml-opt/cosine-scheduler"],
        "stacking_order": 2,
        "config": {"lr": 0.003},
    }
    desc = build_experiment_description("exp-stack-001", data)
    assert "perceptual-loss" in desc
    assert "cosine-scheduler" in desc


def test_build_description_stacked_truncation():
    """Long stacked descriptions are truncated."""
    data = {
        "code_branches": [f"ml-opt/method-{c}" for c in "abcdefghij"],
        "config": {},
    }
    desc = build_experiment_description("exp-stack-010", data, max_len=30)
    assert len(desc) <= 30
    assert desc.endswith("...")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `eval "$(conda shell.bash hook 2>/dev/null)" && conda activate base && python -m pytest tests/test_result_analyzer.py::test_build_description_stacked_experiment -v`

Expected: FAIL — current `build_experiment_description` doesn't handle `code_branches`.

- [ ] **Step 3: Update build_experiment_description**

In `scripts/result_analyzer.py`, in `build_experiment_description()`, **replace** the existing method-name block (the 4 lines starting with `proposal = data.get("code_proposal")`) with this if/else block:

```python
    # Stacked methods (multiple branches combined)
    branches = data.get("code_branches")
    if branches and isinstance(branches, list):
        names = [b.removeprefix("ml-opt/") for b in branches]
        parts.append(" + ".join(names))
    else:
        # Single method
        proposal = data.get("code_proposal") or data.get("code_branch", "")
        if proposal:
            proposal = proposal.removeprefix("ml-opt/")
            parts.append(proposal)
```

The old code to remove is:
```python
    proposal = data.get("code_proposal") or data.get("code_branch", "")
    if proposal:
        proposal = proposal.removeprefix("ml-opt/")
        parts.append(proposal)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `eval "$(conda shell.bash hook 2>/dev/null)" && conda activate base && python -m pytest tests/test_result_analyzer.py -v`

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/result_analyzer.py tests/test_result_analyzer.py
git commit -m "feat: progress chart descriptions for stacked experiments"
```

---

## Chunk 2: Pipeline State & Skill Definitions

### Task 5: Add stacking state to pipeline_state.py

**Files:**
- Modify: `scripts/pipeline_state.py:104-161` (`save_state`)
- Test: `tests/test_pipeline_state.py`

- [ ] **Step 1: Write the failing test**

```python
def test_save_and_load_stacking_state(tmp_path):
    """Stacking state is preserved through save/load cycle."""
    stacking = {
        "ranked_methods": ["perceptual-loss", "cosine-scheduler", "mixup"],
        "current_stack_order": 2,
        "stack_base_branch": "ml-opt/stack-2",
        "stack_base_exp": "exp-015",
        "skipped_methods": ["mixup"],
        "stacked_methods": ["perceptual-loss", "cosine-scheduler"],
    }
    save_state(6, 5, [], str(tmp_path), user_choices={"stacking": stacking})
    loaded = load_state(str(tmp_path))
    assert loaded is not None
    assert loaded["user_choices"]["stacking"] == stacking
```

- [ ] **Step 2: Run test — should already pass** (stacking state is just a dict inside `user_choices`)

Run: `eval "$(conda shell.bash hook 2>/dev/null)" && conda activate base && python -m pytest tests/test_pipeline_state.py::test_save_and_load_stacking_state -v`

Expected: PASS (the existing `save_state` serializes any dict in `user_choices`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline_state.py
git commit -m "test: verify stacking state round-trips through pipeline_state"
```

---

### Task 6: Add Phase 6.5 stacking phase to orchestrate skill

**Files:**
- Modify: `skills/orchestrate/SKILL.md`

This is the core skill definition change. Insert a new top-level `## Phase 6.5` section between Phase 6 and Phase 7.

**Note:** The existing Phase 6 has an internal sub-step labeled "6.5. Mid-loop method proposal trigger" (line 685). This is a *sub-step* within Phase 6's loop, not a top-level phase. Our new `## Phase 6.5` is a top-level phase heading (same level as `## Phase 5.1`), so there is no naming collision.

**Non-git fallback guard:** The stacking phase requires git branches. If the implementation manifest uses `file_backup` strategy, skip stacking entirely.

- [ ] **Step 1: Read the current Phase 6 ending and Phase 7 beginning**

Read `skills/orchestrate/SKILL.md`. The insertion point is immediately **before** `## Phase 7: Report` (line 857). Insert the new section between the end of Phase 6 content and the `## Phase 7` heading.

- [ ] **Step 2: Add Phase 6.5 section**

Insert immediately before the line `## Phase 7: Report`:

```markdown
## Phase 6.5: Method Stacking (Sequential Accumulation)

**Pre-check:** If the implementation manifest uses `strategy: "file_backup"` (non-git project), skip stacking entirely. Log to dev_notes: "Stacking requires git branches — skipped for file-backup projects." Proceed to Phase 7.

**Trigger:** When the experiment loop ends (analyze recommends `stop` or budget exhausted) AND `methods_with_improvement >= 5`.

Count `methods_with_improvement` by calling `rank_methods_for_stacking()` from `result_analyzer.py`:
```bash
python3 scripts/result_analyzer.py <results_dir> <metric> [baseline_id] [lower_is_better]
```
Then count entries in the result. If fewer than 5, skip to Phase 7.

**Checkpoint:**
- **Interactive mode:** Ask user: "{N} methods showed improvement over baseline. Would you like to stack them to find compound gains? The best methods will be merged sequentially."
  - If user declines → skip to Phase 7
- **Autonomous mode:** Auto-proceed. Log to dev_notes: "Auto-entering stacking phase with {N} improved methods."

### Stacking Loop

1. **Rank methods** by improvement magnitude (descending) using `rank_methods_for_stacking()`.

2. **Initialize stack:** The best method's branch becomes `ml-opt/stack-1`. No experiment needed — its existing best result serves as the stack baseline.

3. **For each remaining method** (rank 2, 3, ... N):

   a. **Create stack branch:**
   ```bash
   git checkout -b ml-opt/stack-<order> ml-opt/stack-<order-1>
   # (For order=2, branch from ml-opt/stack-1 which is the best method's branch)
   ```

   b. **Merge the next method:**
   ```bash
   git merge ml-opt/<method-slug> --no-ff --no-edit
   ```

   c. **If clean merge** → proceed to validation.

   d. **If merge conflicts** → dispatch implement-agent:
      - **Prompt:** "Resolve merge conflicts in the following files. Both methods must be preserved: [method-A description] and [method-B description]. The goal is to combine their functionality."
      - **Files:** List of conflicting files from `git diff --name-only --diff-filter=U`
      - If implement-agent succeeds → `git add .` and `git commit -m "Resolve merge conflicts for stack-<order>"`
      - If implement-agent fails → skip this method:
        - `git merge --abort`
        - Log to error tracker: `category: "implementation_error", message: "Stacking conflict unresolved for <method-slug>"`
        - Continue to next method

   e. **Validate** (syntax, import, forward pass — same as implement skill validation).
      - If validation fails → skip: delete branch, log reason, continue.

   f. **Run experiment** using the `ml-optimizer:experiment` skill:
      - `code_branch`: `ml-opt/stack-<order>`
      - `code_branches`: list of all methods in this stack
      - `method_tier`: `"stacked_default_hp"`
      - `stacking_order`: current order number
      - `stack_base_exp`: exp_id of the previous stack's best result
      - `config`: best HP config from the top method currently in the stack

   g. **Evaluate result:**
      - Compare to previous stack step's metric value.
      - **If improved:** Keep this stack step.
        - Update `stack_base_branch = ml-opt/stack-<order>`
        - **Optional HP-tune:** If the improvement is > 1% AND remaining budget allows, invoke `ml-optimizer:hp-tune` with:
          - `code_branches`: [current stack branch]
          - `iteration`: 1
          - `remaining_budget`: min(2, actual remaining)
          - `search_space`: narrowed to HPs the newly added method likely interacts with (LLM judgment)
        - If HP-tune improves further, record as `method_tier: "stacked_tuned_hp"`
      - **If worse or equal:** Skip this method.
        - Delete `ml-opt/stack-<order>` branch
        - Log: "Method <slug> skipped in stacking (metric degraded by X%)"
        - Continue to next method (next stack branch re-branches from last successful stack)

4. **Save stacking state** to `pipeline-state.json` after each stack step (for resumption):
   ```json
   {
     "stacking": {
       "ranked_methods": ["method-b", "method-a", "method-c", ...],
       "current_stack_order": 3,
       "stack_base_branch": "ml-opt/stack-2",
       "stack_base_exp": "exp-stack-002",
       "skipped_methods": ["method-c"],
       "stacked_methods": ["method-b", "method-a"]
     }
   }
   ```

5. **Final result:** The last successful `ml-opt/stack-<N>` branch is the compound best.
   Log to dev_notes: "Stacking complete. Final stack: [methods]. Compound gain: X% over baseline. Branch: ml-opt/stack-<N>"

### Stacking Phase Resumption

On pipeline restart, if `pipeline-state.json` contains a `stacking` key in `user_choices`:
1. Read stacking state
2. Resume from `current_stack_order + 1`
3. Continue with remaining methods in `ranked_methods` that aren't in `stacked_methods` or `skipped_methods`
```

- [ ] **Step 3: Commit**

```bash
git add skills/orchestrate/SKILL.md
git commit -m "feat: add Phase 6.5 method stacking to orchestrate skill"
```

---

### Task 7: Update analyze skill to report methods_with_improvement count

**Files:**
- Modify: `skills/analyze/SKILL.md`

- [ ] **Step 1: Add instruction to analyze output**

In `skills/analyze/SKILL.md`, in the output format section, add:

```markdown
### Stacking Readiness

Include in the analysis output:
- `methods_with_improvement`: Count of unique code_branches whose best result beats baseline.
  Compute using `rank_methods_for_stacking()` from `result_analyzer.py`.
- `stacking_candidates`: List of method names (code_proposal values) that improved, ranked by improvement magnitude.
```

- [ ] **Step 2: Commit**

```bash
git add skills/analyze/SKILL.md
git commit -m "feat: add methods_with_improvement count to analyze output"
```

---

### Task 8: Update hp-tune skill for stacked branch tuning

**Files:**
- Modify: `skills/hp-tune/SKILL.md`

- [ ] **Step 1: Add stacking HP-tune section**

In `skills/hp-tune/SKILL.md`, add a section after the existing iteration strategies:

```markdown
### HP-Tuning for Stacked Methods

When invoked during the stacking phase (identifiable by `method_tier: "stacked_default_hp"` in recent results):

1. **Starting point:** Use the HP config from the best individual method in the stack (passed as `baseline_config`).
2. **Narrow scope:** Only vary HPs that the newly added method likely interacts with. For example:
   - New loss function → vary `learning_rate`, `weight_decay`
   - New augmentation → vary `batch_size`, `learning_rate`
   - New scheduler → vary `learning_rate`, `warmup_steps`
3. **Budget:** Cap at 2 iterations maximum during stacking.
4. **Proposals:** Generate `min(max(num_gpus, 1), remaining_budget)` configs, all targeting the stack branch.
```

- [ ] **Step 2: Commit**

```bash
git add skills/hp-tune/SKILL.md
git commit -m "feat: add stacking HP-tune strategy to hp-tune skill"
```

---

### Task 9: Update experiment-agent result schema

**Files:**
- Modify: `agents/experiment-agent.md`

- [ ] **Step 1: Add stacking fields to result schema**

In `agents/experiment-agent.md`, in the result JSON schema section, add the new fields:

```markdown
- `code_branches` (array of strings, optional): For stacked experiments, lists all method branches combined in this experiment. Null/absent for single-method experiments.
- `stacking_order` (integer, optional): Position in the stacking accumulation chain (1 = best method alone, 2 = best + second, etc.).
- `stack_base_exp` (string, optional): Experiment ID of the previous stack step this builds on.
```

And update the `method_tier` field documentation to include `"stacked_default_hp"` and `"stacked_tuned_hp"`.

- [ ] **Step 2: Commit**

```bash
git add agents/experiment-agent.md
git commit -m "feat: add stacking fields to experiment-agent result schema"
```

---

### Task 10: Add stacking results table to report skill

**Files:**
- Modify: `skills/report/SKILL.md`

- [ ] **Step 1: Add stacking section to report format**

In `skills/report/SKILL.md`, after the three-tier comparison table section, add:

```markdown
### Method Stacking Results (if stacking phase was run)

If any results have `method_tier` of `"stacked_default_hp"` or `"stacked_tuned_hp"`, include a stacking table:

```markdown
## Method Stacking Results

| Stack | Methods Added | <Metric> | vs Baseline | vs Previous Stack | Status |
|-------|---------------|----------|-------------|-------------------|--------|
| 1 | <best-method> | X.XX | +N.N% | — | kept |
| 2 | + <second-method> | X.XX | +N.N% | +N.N% | kept |
| 3 | + <third-method> | X.XX | — | -N.N% | skipped |
| ... | ... | ... | ... | ... | ... |

Final stack: <method-a> + <method-b> + <method-d>
Compound gain: +N.N% over baseline
Branch: ml-opt/stack-<N>
```

Sort by `stacking_order`. Show both cumulative gain (vs baseline) and incremental gain (vs previous stack step). Mark skipped methods.
```

- [ ] **Step 2: Commit**

```bash
git add skills/report/SKILL.md
git commit -m "feat: add stacking results table to report skill"
```

---

## Chunk 3: Documentation & Final Verification

### Task 11: Update CLAUDE.md documentation

**Files:**
- Modify: `.claude/CLAUDE.md`

- [ ] **Step 1: Add stacking phase to the pipeline description**

In the "Skill Pipeline (Orchestrator Flow)" section, add Phase 6.5 between Phase 6 and Phase 7:

```
Phase 6.5: Method stacking (if 5+ methods improved):
           Sequential accumulation — merge best methods one by one
           LLM conflict resolution, skip-on-failure
           Optional HP-tune per stack step
```

- [ ] **Step 2: Add stacking to design patterns**

In "Key Design Patterns", add:

```markdown
- **Method stacking (Phase 6.5):** After independent method testing identifies ≥5 methods that improve over baseline, the orchestrator sequentially merges them in descending order of improvement. Each stack step creates `ml-opt/stack-<N>` by merging the next method into the current best stack. Clean merges proceed directly; conflicts are resolved by the implement-agent. If a combination degrades performance, that method is skipped. Optional HP-tuning (1-2 iterations, narrowed scope) is applied when a combo shows >1% improvement. Stacking state persists in `pipeline-state.json` for resumption. Requires git branch strategy — skipped for `file_backup` projects.
```

- [ ] **Step 3: Add stacking to three-tier tracking documentation**

Update the "Three-Tier Result Tracking" section to mention the two new tiers:

```markdown
Additionally, stacking experiments use:
- **`stacked_default_hp`**: Combined methods tested with best individual HPs
- **`stacked_tuned_hp`**: Combined methods after HP-tuning

Stacking experiments also carry `code_branches` (array of combined branches), `stacking_order`, and `stack_base_exp`.
```

- [ ] **Step 4: Commit**

```bash
git add .claude/CLAUDE.md
git commit -m "docs: document method stacking phase in CLAUDE.md"
```

---

### Task 12: Run full test suite and verify

- [ ] **Step 1: Run all tests**

Run: `eval "$(conda shell.bash hook 2>/dev/null)" && conda activate base && python -m pytest tests/ -v`

Expected: ALL PASS (868+ tests including new ones)

- [ ] **Step 2: Verify no regressions in existing tests**

Check that the test count increased (new tests added) and no existing tests broke.

- [ ] **Step 3: Final commit if any fixups needed**

```bash
git add -A && git commit -m "fix: address test regressions from stacking feature"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Schema: stacked tiers + code_branches validation | `schema_validator.py`, tests |
| 2 | Utility: `rank_methods_for_stacking()` | `result_analyzer.py`, tests |
| 3 | Test: stacked tiers in group_by_method_tier | tests |
| 4 | Progress chart descriptions for stacked experiments | `result_analyzer.py`, tests |
| 5 | Pipeline state round-trip test | tests |
| 6 | **Core:** Phase 6.5 in orchestrate skill | `orchestrate/SKILL.md` |
| 7 | Analyze output: methods_with_improvement count | `analyze/SKILL.md` |
| 8 | HP-tune: stacking strategy | `hp-tune/SKILL.md` |
| 9 | Experiment agent: stacking result fields | `experiment-agent.md` |
| 10 | Report: stacking results table | `report/SKILL.md` |
| 11 | Documentation update | `CLAUDE.md` |
| 12 | Full test suite verification | — |
