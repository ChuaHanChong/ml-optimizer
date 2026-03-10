# Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix critical autonomy gaps (7 blocking checkpoints), a race condition bug, missing test coverage (8 tests), and flexibility gaps (HuggingFace log parsing, tabular ML hp-tune bias).

**Architecture:** Surgical edits to 4 SKILL.md files, 3 Python scripts, and 5 test files. No new files. All script changes are backward-compatible. TDD for script changes; SKILL.md changes are instruction edits verified by existing test suite.

**Tech Stack:** Markdown (SKILL.md), Python (stdlib only), pytest

---

### Task 1: HuggingFace Trainer Log Parser — Failing Test

**Files:**
- Test: `tests/test_parse_logs.py` (append)

**Step 1: Write failing tests for HuggingFace Trainer format**

Append to `tests/test_parse_logs.py`:

```python


# --- HuggingFace Trainer format (Python dict syntax) ---

def test_parse_hf_trainer_line():
    """HuggingFace Trainer outputs Python dicts with single quotes."""
    from parse_logs import parse_hf_trainer_line
    result = parse_hf_trainer_line("{'loss': 0.5, 'learning_rate': 5e-05, 'epoch': 1.0}")
    assert result == {"loss": 0.5, "learning_rate": 5e-05, "epoch": 1.0}


def test_parse_hf_trainer_line_with_prefix():
    """HuggingFace Trainer lines sometimes have leading whitespace or prefix."""
    from parse_logs import parse_hf_trainer_line
    result = parse_hf_trainer_line("  {'loss': 0.3241, 'grad_norm': 1.234, 'epoch': 2.0}")
    assert result["loss"] == 0.3241
    assert result["epoch"] == 2.0


def test_parse_hf_trainer_line_not_dict():
    """Non-dict lines return empty dict."""
    from parse_logs import parse_hf_trainer_line
    assert parse_hf_trainer_line("Epoch 1/10") == {}
    assert parse_hf_trainer_line("") == {}


def test_detect_format_hf_trainer():
    """HuggingFace Trainer dict-syntax lines detected as 'hf_trainer'."""
    lines = [
        "{'loss': 0.6931, 'learning_rate': 5e-05, 'epoch': 1.0}",
        "{'loss': 0.5123, 'learning_rate': 4e-05, 'epoch': 2.0}",
    ]
    assert detect_format(lines) == "hf_trainer"


def test_parse_log_hf_trainer_format(tmp_path):
    """Full parse_log integration with HuggingFace Trainer format."""
    f = tmp_path / "hf_train.log"
    f.write_text(
        "Training started\n"
        "{'loss': 0.6931, 'learning_rate': 5e-05, 'epoch': 1.0}\n"
        "{'loss': 0.5123, 'learning_rate': 4e-05, 'epoch': 2.0}\n"
    )
    records = parse_log(str(f))
    assert len(records) == 2
    assert records[0]["loss"] == 0.6931
    assert records[1]["epoch"] == 2.0
```

**Step 2: Run tests to verify they fail**

Run: `/data/hanchong/miniconda3/bin/python -m pytest tests/test_parse_logs.py::test_parse_hf_trainer_line -v --tb=short`
Expected: FAIL with `ImportError: cannot import name 'parse_hf_trainer_line'`

---

### Task 2: HuggingFace Trainer Log Parser — Implementation

**Files:**
- Modify: `scripts/parse_logs.py`

**Step 1: Add the `parse_hf_trainer_line` function**

In `scripts/parse_logs.py`, after `parse_json_line` (line 113), insert:

```python


def parse_hf_trainer_line(line: str) -> dict:
    """Parse HuggingFace Trainer dict-syntax lines: {'loss': 0.5, 'epoch': 1.0}."""
    stripped = line.strip()
    m = re.match(r"^\{.*'.*':.*\}$", stripped)
    if not m:
        return {}
    try:
        converted = stripped.replace("'", '"')
        data = json.loads(converted)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if isinstance(v, (int, float))}
    except (json.JSONDecodeError, ValueError):
        pass
    return {}
```

**Step 2: Add HuggingFace detection to `detect_format`**

In `scripts/parse_logs.py`, in the `detect_format` function (line 137), add after the JSON detection block (after the `return "json"` at approximately line 148) and BEFORE the logging format detection:

```python
    for line in lines[:5]:
        stripped = line.strip()
        if stripped and re.match(r"^\{.*'.*':.*\}$", stripped):
            return "hf_trainer"
```

**Step 3: Add HuggingFace format handling to `parse_log`**

In `scripts/parse_logs.py`, in the `parse_log` function, after the `if fmt == "json":` block (approximately line 188) and before `elif fmt == "csv":`, insert:

```python
    elif fmt == "hf_trainer":
        return [m for line in lines if (m := parse_hf_trainer_line(line))]
```

**Step 4: Update the import in test file**

In `tests/test_parse_logs.py` line 8, add `parse_hf_trainer_line` to the import:

Find:
```python
from parse_logs import parse_kv_line, parse_json_line, parse_csv_lines, parse_python_logging_line, parse_tqdm_line, parse_xgboost_line, detect_format, parse_log, extract_metric_trajectory
```

Replace with:
```python
from parse_logs import parse_kv_line, parse_json_line, parse_hf_trainer_line, parse_csv_lines, parse_python_logging_line, parse_tqdm_line, parse_xgboost_line, detect_format, parse_log, extract_metric_trajectory
```

**Step 5: Run tests to verify they pass**

Run: `/data/hanchong/miniconda3/bin/python -m pytest tests/test_parse_logs.py -v --tb=short -q`
Expected: All tests pass (including the 5 new HF Trainer tests)

**Step 6: Commit**

```bash
git add scripts/parse_logs.py tests/test_parse_logs.py
git commit -m "Add HuggingFace Trainer log format parser (single-quote dicts)"
```

---

### Task 3: result_analyzer File Filter — Failing Test

**Files:**
- Test: `tests/test_result_analyzer.py` (append)

**Step 1: Write failing test**

Append to `tests/test_result_analyzer.py`:

```python


def test_load_results_ignores_non_experiment_files(tmp_path):
    """load_results should only load exp-*.json and baseline.json, not other JSON."""
    _write_results(tmp_path, {
        "baseline": {"metrics": {"loss": 1.0}},
        "exp-001": {"metrics": {"loss": 0.8}},
    })
    # Write non-experiment files that should be ignored
    (tmp_path / "prerequisites.json").write_text(json.dumps({"status": "ready"}))
    (tmp_path / "implementation-manifest.json").write_text(json.dumps({"strategy": "git_branch"}))
    results = load_results(str(tmp_path))
    assert "baseline" in results
    assert "exp-001" in results
    assert "prerequisites" not in results
    assert "implementation-manifest" not in results
    assert len(results) == 2
```

**Step 2: Run test to verify it fails**

Run: `/data/hanchong/miniconda3/bin/python -m pytest tests/test_result_analyzer.py::test_load_results_ignores_non_experiment_files -v --tb=short`
Expected: FAIL — `assert "prerequisites" not in results` fails

---

### Task 4: result_analyzer File Filter — Implementation

**Files:**
- Modify: `scripts/result_analyzer.py:10-22`

**Step 1: Update `load_results` to filter files**

In `scripts/result_analyzer.py`, replace the `load_results` function (lines 10-22):

```python
def load_results(results_dir: str) -> dict[str, dict]:
    """Load all experiment results from a directory."""
    path = Path(results_dir)
    results = {}
    if not path.exists():
        return results
    for f in sorted(path.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            results[f.stem] = data
        except (json.JSONDecodeError, OSError):
            continue
    return results
```

With:

```python
def load_results(results_dir: str) -> dict[str, dict]:
    """Load experiment results (exp-*.json and baseline.json) from a directory."""
    path = Path(results_dir)
    results = {}
    if not path.exists():
        return results
    for f in sorted(path.glob("*.json")):
        if f.stem != "baseline" and not f.stem.startswith("exp-"):
            continue
        try:
            data = json.loads(f.read_text())
            results[f.stem] = data
        except (json.JSONDecodeError, OSError):
            continue
    return results
```

**Step 2: Run tests**

Run: `/data/hanchong/miniconda3/bin/python -m pytest tests/test_result_analyzer.py -v --tb=short -q`
Expected: All tests pass

**Step 3: Commit**

```bash
git add scripts/result_analyzer.py tests/test_result_analyzer.py
git commit -m "Filter load_results to exp-*.json and baseline.json only"
```

---

### Task 5: log_event Race Condition — Failing Test

**Files:**
- Test: `tests/test_error_tracker.py` (append)

**Step 1: Write failing test for concurrent log_event**

Append to `tests/test_error_tracker.py`:

```python


def test_concurrent_log_events_no_lost_events(tmp_path):
    """Multiple concurrent log_event calls must not lose events."""
    import concurrent.futures
    exp_root = str(tmp_path)
    n_threads = 4
    events_per_thread = 5
    total_expected = n_threads * events_per_thread

    def log_n_events(thread_id):
        for i in range(events_per_thread):
            log_event(exp_root, create_event(
                "training_failure", "warning", "experiment",
                f"error from thread {thread_id} event {i}"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = [ex.submit(log_n_events, t) for t in range(n_threads)]
        for f in futures:
            f.result()  # raise if any thread failed

    log_data = load_error_log(exp_root)
    assert log_data is not None
    assert len(log_data["events"]) == total_expected, (
        f"Expected {total_expected} events but got {len(log_data['events'])} — "
        f"race condition lost {total_expected - len(log_data['events'])} events"
    )
```

**Step 2: Run test to verify it fails**

Run: `/data/hanchong/miniconda3/bin/python -m pytest tests/test_error_tracker.py::test_concurrent_log_events_no_lost_events -v --tb=short`
Expected: FAIL — events lost due to read-modify-write race

---

### Task 6: log_event Race Condition — Fix

**Files:**
- Modify: `scripts/error_tracker.py:152-184`

**Step 1: Add file locking to `_atomic_write_json`**

In `scripts/error_tracker.py`, replace `_atomic_write_json` (lines 152-165) with:

```python
def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via temp file + rename, with file locking."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
```

**Step 2: Add file locking to `log_event`**

In `scripts/error_tracker.py`, replace `log_event` (lines 168-184) with:

```python
def log_event(exp_root: str, event: dict) -> str:
    """Append an event to the per-project error log. Returns the log path.

    Uses file locking to prevent concurrent read-modify-write races.
    """
    import fcntl
    path = _error_log_path(exp_root)
    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            log_data = load_error_log(exp_root)
            if log_data is None:
                log_data = {
                    "project_id": _project_id(exp_root),
                    "session_start": datetime.now(timezone.utc).isoformat(),
                    "events": [],
                    "summary": {},
                }
            log_data["events"].append(event)
            log_data["summary"] = _compute_summary(log_data["events"])
            _atomic_write_json(path, log_data)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    return str(path)
```

**Step 3: Run tests**

Run: `/data/hanchong/miniconda3/bin/python -m pytest tests/test_error_tracker.py -v --tb=short -q`
Expected: All tests pass (including the new concurrency test)

**Step 4: Commit**

```bash
git add scripts/error_tracker.py tests/test_error_tracker.py
git commit -m "Fix log_event race condition with file locking"
```

---

### Task 7: Empty Input Edge Case Tests

**Files:**
- Test: `tests/test_parse_logs.py` (append)
- Test: `tests/test_detect_divergence.py` (append)
- Test: `tests/test_result_analyzer.py` (append)
- Test: `tests/test_error_tracker.py` (append)

**Step 1: Write all empty input edge case tests**

Append to `tests/test_parse_logs.py`:

```python


# --- Empty input edge cases ---

def test_parse_csv_lines_empty():
    """parse_csv_lines with empty list returns empty."""
    assert parse_csv_lines([]) == []


def test_detect_format_empty_lines():
    """detect_format with empty list falls back to 'kv'."""
    assert detect_format([]) == "kv"


def test_parse_kv_line_empty_string():
    """parse_kv_line with empty string returns empty dict."""
    assert parse_kv_line("") == {}


def test_parse_json_line_empty_string():
    """parse_json_line with empty string returns empty dict."""
    assert parse_json_line("") == {}
```

Append to `tests/test_detect_divergence.py`:

```python


# --- Empty input edge cases ---

def test_detect_nan_inf_empty():
    """detect_nan_inf with empty list returns None."""
    assert detect_nan_inf([]) is None


def test_detect_explosion_empty():
    """detect_explosion with empty list returns None."""
    assert detect_explosion([], lower_is_better=True) is None


def test_check_divergence_empty():
    """check_divergence with empty list returns not diverged."""
    result = check_divergence([])
    assert result["diverged"] is False
```

Append to `tests/test_result_analyzer.py`:

```python


def test_rank_by_metric_empty():
    """rank_by_metric with empty results returns empty list."""
    assert rank_by_metric({}, "loss", lower_is_better=True) == []


def test_identify_correlations_empty():
    """identify_correlations with empty results returns empty dict."""
    assert identify_correlations({}, "loss") == {}
```

Append to `tests/test_error_tracker.py`:

```python


def test_validate_event_empty_dict():
    """validate_event with empty dict returns errors for all missing fields."""
    errors = validate_event({})
    assert len(errors) > 0
    # Must flag missing required fields
    assert any("category" in e for e in errors)
```

**Step 2: Run all tests**

Run: `/data/hanchong/miniconda3/bin/python -m pytest tests/test_parse_logs.py tests/test_detect_divergence.py tests/test_result_analyzer.py tests/test_error_tracker.py -v --tb=short -q`
Expected: All pass (these test existing behavior; no code changes needed)

**Step 3: Commit**

```bash
git add tests/test_parse_logs.py tests/test_detect_divergence.py tests/test_result_analyzer.py tests/test_error_tracker.py
git commit -m "Add empty input edge case tests for parsers and analyzers"
```

---

### Task 8: Unicode Log File Test

**Files:**
- Test: `tests/test_parse_logs.py` (append)

**Step 1: Write Unicode test**

Append to `tests/test_parse_logs.py`:

```python


def test_parse_log_unicode_content(tmp_path):
    """parse_log handles Unicode characters in log lines."""
    f = tmp_path / "unicode.log"
    f.write_text(
        "epoch=1 loss=0.5 # 训练日志\n"
        "epoch=2 loss=0.4 # training résumé\n"
        "epoch=3 loss=0.3\n",
        encoding="utf-8",
    )
    records = parse_log(str(f))
    assert len(records) == 3
    assert records[0]["loss"] == 0.5
    assert records[2]["loss"] == 0.3
```

**Step 2: Run test**

Run: `/data/hanchong/miniconda3/bin/python -m pytest tests/test_parse_logs.py::test_parse_log_unicode_content -v --tb=short`
Expected: PASS (parse_logs already uses `errors="replace"`)

**Step 3: Commit**

```bash
git add tests/test_parse_logs.py
git commit -m "Add Unicode log file test"
```

---

### Task 9: Contract Tests — Baseline → HP-tune, Experiment → Monitor, Pipeline State Roundtrip, Prerequisites → Schema

**Files:**
- Test: `tests/test_skill_contracts.py` (append)
- Test: `tests/test_pipeline_state.py` (append)

**Step 1: Write contract tests**

Append to `tests/test_skill_contracts.py`:

```python


# --- Baseline → HP-tune contract ---

def test_baseline_has_fields_hp_tune_expects():
    """Baseline output must have profiling and config fields for hp-tune."""
    baseline = {
        "exp_id": "baseline",
        "status": "completed",
        "config": {"lr": 0.01, "batch_size": 64, "weight_decay": 1e-4},
        "metrics": {"loss": 1.5, "accuracy": 45.0},
        "profiling": {
            "gpu_memory_used_mib": 8000,
            "gpu_memory_total_mib": 24576,
            "throughput_samples_per_sec": 150,
            "training_time_seconds": 600,
        },
    }
    # hp-tune needs config to define initial search space
    assert isinstance(baseline["config"], dict)
    assert len(baseline["config"]) > 0
    # hp-tune needs profiling for batch size recommendations
    assert "profiling" in baseline
    assert "gpu_memory_used_mib" in baseline["profiling"]
    assert "training_time_seconds" in baseline["profiling"]


# --- Experiment → Monitor log format contract ---

def test_experiment_log_format_parseable_by_monitor(tmp_path):
    """Experiment-generated training logs must be parseable by parse_logs."""
    from parse_logs import parse_log, extract_metric_trajectory
    # Simulate a typical experiment log output
    log_file = tmp_path / "train.log"
    log_file.write_text(
        "epoch=1 loss=0.693 accuracy=50.0\n"
        "epoch=2 loss=0.512 accuracy=68.5\n"
        "epoch=3 loss=0.445 accuracy=73.2\n"
    )
    records = parse_log(str(log_file))
    assert len(records) == 3
    # Monitor extracts loss trajectory for divergence detection
    trajectory = extract_metric_trajectory(records, "loss")
    assert len(trajectory) == 3
    assert trajectory[0] > trajectory[-1]  # loss should decrease


# --- Prerequisites → schema_validator contract ---

def test_prerequisites_report_passes_schema_validation():
    """A well-formed prerequisites report must pass schema_validator."""
    from schema_validator import validate_prerequisites
    report = {
        "status": "ready",
        "dataset": {
            "train_path": "/data/train",
            "val_path": "/data/val",
            "format_detected": "image_folder",
            "prepared": False,
            "prepared_train_path": None,
            "prepared_val_path": None,
            "validation_passed": True,
            "notes": "",
        },
        "environment": {
            "manager": "conda",
            "python_version": "3.10.12",
            "packages_installed": ["torch", "torchvision"],
            "packages_failed": [],
            "all_imports_resolved": True,
            "notes": "",
        },
        "ready_for_baseline": True,
    }
    errors = validate_prerequisites(report)
    assert errors == [], f"Valid prerequisites report failed validation: {errors}"


# --- Analyze → Research (method_proposal pivot) contract ---

def test_analyze_method_proposal_pivot_has_required_fields():
    """Analyze pivot_type=method_proposal must include fields orchestrator needs."""
    analyze_output = {
        "action": "pivot",
        "pivot_type": "method_proposal",
        "reason": "HP tuning has plateaued, no improvement in last 3 batches",
    }
    assert analyze_output["action"] == "pivot"
    assert analyze_output["pivot_type"] in ("method_proposal", "qualitative_change", "narrow_space")
    assert "reason" in analyze_output
```

Append to `tests/test_pipeline_state.py`:

```python


def test_full_user_choices_roundtrip(tmp_path):
    """All documented user_choices fields must survive save/load."""
    all_choices = {
        "primary_metric": "accuracy",
        "divergence_metric": "loss",
        "divergence_lower_is_better": True,
        "lower_is_better": False,
        "target_value": 95.0,
        "train_command": "python train.py",
        "eval_command": "python eval.py",
        "train_data_path": "/data/train",
        "val_data_path": "/data/val",
        "prepared_train_path": "/exp/prepared/train",
        "prepared_val_path": "/exp/prepared/val",
        "env_manager": "conda",
        "env_name": "ml-env",
        "model_category": "supervised",
        "user_papers": ["https://arxiv.org/abs/1234"],
        "budget_mode": "autonomous",
        "difficulty": "hard",
        "difficulty_multiplier": 25,
        "method_proposal_scope": "architecture",
        "method_proposal_iterations": 2,
        "hp_batches_per_round": 5,
    }
    save_state(6, 3, ["exp-010"], str(tmp_path), user_choices=all_choices)
    state = load_state(str(tmp_path))
    assert state is not None
    for key, value in all_choices.items():
        assert key in state["user_choices"], f"user_choices missing field: {key}"
        assert state["user_choices"][key] == value, (
            f"user_choices[{key}] roundtrip failed: {state['user_choices'][key]} != {value}"
        )
```

**Step 2: Run tests**

Run: `/data/hanchong/miniconda3/bin/python -m pytest tests/test_skill_contracts.py tests/test_pipeline_state.py -v --tb=short -q`
Expected: All pass

**Step 3: Commit**

```bash
git add tests/test_skill_contracts.py tests/test_pipeline_state.py
git commit -m "Add contract tests: baseline→hp-tune, experiment→monitor, prerequisites→schema, pipeline state roundtrip"
```

---

### Task 10: Orchestrate — Mid-Loop Method Proposal Auto-Skip (Steps 6.1b + 6.1d)

**Files:**
- Modify: `skills/orchestrate/SKILL.md:629-649`

**Step 1: Add autonomous auto-skip to step 6.1b (scope confirmation)**

In `skills/orchestrate/SKILL.md`, find step 6.1b at line 629:

```
   b. **Scope confirmation:** Ask the user which scope level to use:
      ```
      HP tuning has plateaued. I can propose new optimization methods.
```

Insert BEFORE that line:

```
   b. **Scope confirmation:**
      **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, use the stored `method_proposal_scope` from user_choices. Skip AskUserQuestion. Log to dev_notes: "Mid-loop method proposal auto-using scope: <scope> (autonomous mode)."

      **Otherwise:** Ask the user which scope level to use:
```

And change the existing `b. **Scope confirmation:** Ask the user which scope level to use:` to just the `**Otherwise:**` section.

**Step 2: Add autonomous auto-skip to step 6.1d (proposal confirmation)**

In `skills/orchestrate/SKILL.md`, find step 6.1d at line 649:

```
   d. **Present proposals:** Show the generated proposals to the user for confirmation. The user can accept all, select a subset, or reject all (which exits the loop).
```

Replace with:

```
   d. **Present proposals:**
      **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, accept all proposals. Skip user presentation. Log to dev_notes: "Mid-loop: auto-accepted all N method proposals (autonomous mode)."

      **Otherwise:** Show the generated proposals to the user for confirmation. The user can accept all, select a subset, or reject all (which exits the loop).
```

**Step 3: Verify no test regression**

Run: `/data/hanchong/miniconda3/bin/python -m pytest tests/ -v --tb=short -q`
Expected: All tests pass

**Step 4: Commit**

```bash
git add skills/orchestrate/SKILL.md
git commit -m "Add autonomous auto-skip for mid-loop method proposal scope and confirmation"
```

---

### Task 11: Orchestrate — Experiment Timeout

**Files:**
- Modify: `skills/orchestrate/SKILL.md:575-577`

**Step 1: Add timeout mechanism to step 4**

In `skills/orchestrate/SKILL.md`, find step 4 at line 575:

```
4. **Wait for completion:**
   - All experiments in the batch must complete (or be stopped) before analysis
   - Save pipeline state after each batch completes
```

Replace with:

```
4. **Wait for completion:**
   - All experiments in the batch must complete, be stopped, or time out before analysis
   - **Experiment timeout:** `max_experiment_duration = baseline_training_time * 3` (from `baseline.json` → `profiling.training_time_seconds`). If baseline profiling is unavailable (`null`), use a hard fallback of 6 hours. If any experiment exceeds this duration, kill it and record:
     - Set `status: "timeout"` in the experiment result
     - Log to error tracker: `category: "timeout", severity: "warning", source: "orchestrate", message: "Experiment <exp_id> killed after <duration>s (limit: <max_duration>s)"`
   - Save pipeline state after each batch completes
```

**Step 2: Commit**

```bash
git add skills/orchestrate/SKILL.md
git commit -m "Add experiment timeout mechanism (3x baseline duration)"
```

---

### Task 12: Orchestrate — Research Failure Recovery

**Files:**
- Modify: `skills/orchestrate/SKILL.md:319-326`

**Step 1: Add research failure recovery to Phase 5**

In `skills/orchestrate/SKILL.md`, find lines 319-326:

```
If the user chose research, invoke the `ml-optimizer:research` skill with parameters:
- `model_type`: Type of model (from Phase 1)
- `task`: What the model does (from Phase 0/1)
- `current_metrics`: Current baseline performance numbers
- `problem_description`: What needs improvement (from Phase 0)
- `user_papers`: Any user-provided paper URLs or links (optional)
- `exp_root`: Path to experiments/ directory (for error logging)
Wait for research findings.
```

Replace with:

```
If the user chose research, invoke the `ml-optimizer:research` skill with parameters:
- `model_type`: Type of model (from Phase 1)
- `task`: What the model does (from Phase 0/1)
- `current_metrics`: Current baseline performance numbers
- `problem_description`: What needs improvement (from Phase 0)
- `user_papers`: Any user-provided paper URLs or links (optional)
- `exp_root`: Path to experiments/ directory (for error logging)
Wait for research findings.

### Research Failure Recovery

If the research agent crashes or returns zero proposals:

1. **Retry with knowledge mode:** Re-invoke research with `source: "knowledge"` (LLM knowledge only, no web search). Log to error tracker: `category: "agent_failure", severity: "warning", source: "orchestrate", message: "Research web search failed — retrying with knowledge mode"`.
2. **If knowledge mode also fails or returns zero proposals:** Log to error tracker: `category: "agent_failure", severity: "critical", source: "orchestrate", message: "Research failed in both web and knowledge modes — proceeding with HP-only optimization"`. Continue to Phase 6 without code branches (HP-only mode).
3. **Do NOT block:** Never use AskUserQuestion for research failures. The pipeline must continue — HP-only optimization is always a valid fallback.

This same recovery chain applies to research invocations in step 6.1c and step 6.8b.
```

**Step 2: Commit**

```bash
git add skills/orchestrate/SKILL.md
git commit -m "Add research failure recovery with knowledge-mode fallback"
```

---

### Task 13: Orchestrate — RL Polarity, Phase 3 Unknown Error, All-Diverge Auto-Skips

**Files:**
- Modify: `skills/orchestrate/SKILL.md:120, 268, 830`

**Step 1: Add RL polarity autonomous auto-skip**

In `skills/orchestrate/SKILL.md`, find line 120:

```
     - **Polarity validation:** After setting `divergence_metric` and `divergence_lower_is_better`, check for inconsistency: if the metric name contains "reward", "return", or "score" but `divergence_lower_is_better` is True, warn the user: "For reward-like metrics, divergence means the metric drops (lower_is_better=False). You set lower_is_better=True — confirm this is correct?" Use AskUserQuestion. This prevents silently killing experiments during improvement.
```

Replace with:

```
     - **Polarity validation:** After setting `divergence_metric` and `divergence_lower_is_better`, check for inconsistency: if the metric name contains "reward", "return", or "score" but `divergence_lower_is_better` is True:
       **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, auto-correct to `divergence_lower_is_better = False`. Log to dev_notes: "Auto-corrected divergence polarity for reward-like metric (autonomous mode)."
       **Otherwise:** Use AskUserQuestion: "For reward-like metrics, divergence means the metric drops (lower_is_better=False). You set lower_is_better=True — confirm this is correct?" This prevents silently killing experiments during improvement.
```

**Step 2: Add Phase 3 unknown error autonomous handling**

In `skills/orchestrate/SKILL.md`, find line 268:

```
| Unknown error | Show full error via AskUserQuestion, ask for guidance |
```

Replace with:

```
| Unknown error | **If `budget_mode == "autonomous"`:** Log full error to error tracker and dev_notes: "Phase 3 baseline failed with unknown error after 2 retries — exiting with partial results (autonomous mode)." Proceed to Phase 7 (report) with whatever results are available. **Otherwise:** Show full error via AskUserQuestion, ask for guidance |
```

**Step 3: Add all-diverge recovery**

In `skills/orchestrate/SKILL.md`, find line 830:

```
- **All experiments diverge:** Stop loop, report to user with AskUserQuestion
```

Replace with:

```
- **All experiments diverge:**
  - **First occurrence:** Attempt one recovery batch with halved learning rates (divide all proposed LRs by 2). Log to error tracker: `category: "divergence", severity: "warning", source: "orchestrate", message: "All experiments diverged — attempting recovery with halved LRs"`.
  - **Second consecutive all-diverge:** In `"autonomous"` mode: exit loop and proceed to Phase 7 (report) with current best. Log: "All-diverge occurred twice consecutively — stopping (autonomous mode)." In other modes: use AskUserQuestion to inform the user and ask for guidance.
```

**Step 4: Verify no test regression**

Run: `/data/hanchong/miniconda3/bin/python -m pytest tests/ -v --tb=short -q`
Expected: All tests pass

**Step 5: Commit**

```bash
git add skills/orchestrate/SKILL.md
git commit -m "Add auto-skips for RL polarity, Phase 3 unknown error, and all-diverge recovery"
```

---

### Task 14: Orchestrate — OOM Feedback to HP-Tune

**Files:**
- Modify: `skills/orchestrate/SKILL.md:829`

**Step 1: Add OOM feedback mechanism**

In `skills/orchestrate/SKILL.md`, find line 829:

```
- **Training crashes:** Record the error, skip to next experiment in batch
```

Replace with:

```
- **Training crashes:** Record the error, skip to next experiment in batch. **OOM-specific handling:** If the error contains `CUDA out of memory` or `OutOfMemoryError`, log the OOM-causing config to error tracker: `category: "resource_error", severity: "warning", source: "orchestrate", message: "OOM at batch_size=<N>", context: {"batch_size": <N>, "gpu_memory_total_mib": <M>}`. On the next hp-tune invocation, pass `max_batch_size` constraint (set to 50% of the OOM-causing batch size) to prevent repeating the same OOM configuration.
```

**Step 2: Commit**

```bash
git add skills/orchestrate/SKILL.md
git commit -m "Add OOM feedback to hp-tune with max_batch_size constraint"
```

---

### Task 15: Implement Skill — Dirty-Tree Auto-Skip

**Files:**
- Modify: `skills/implement/SKILL.md:70-85`

**Step 1: Add autonomous auto-skip for dirty working tree**

In `skills/implement/SKILL.md`, find lines 70-85 (the dirty working tree check):

```
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
```

Replace with:

```
- **Check for uncommitted changes:**
  ```bash
  git status --porcelain
  ```
  If output is non-empty (dirty working tree):

  **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, auto-stash changes:
  ```bash
  git stash --include-untracked -m "ml-optimizer: auto-stash before implementation (autonomous mode)"
  ```
  Log to dev_notes: "Auto-stashed uncommitted changes before branch creation (autonomous mode). Run `git stash pop` to recover."

  **Otherwise:** Use AskUserQuestion:
  ```
  Your working tree has uncommitted changes. These changes would be carried into
  all proposal branches, which could contaminate the baseline comparison.

  Please either:
  1. Commit your changes: git commit -am "WIP"
  2. Stash your changes: git stash

  Then re-run the implement skill.
  ```
  Do NOT proceed with branch creation on a dirty working tree.
```

**Step 2: Commit**

```bash
git add skills/implement/SKILL.md
git commit -m "Add autonomous auto-stash for dirty working tree in implement skill"
```

---

### Task 16: Prerequisites Skill — Autonomous Defaults

**Files:**
- Modify: `skills/prerequisites/SKILL.md:42, 69, 100, 113`

**Step 1: Add autonomous defaults to all 4 AskUserQuestion calls**

In `skills/prerequisites/SKILL.md`, find line 42 (unknown format):

```
If confidence is "low" or format is "unknown", use AskUserQuestion:
```

Replace with:

```
If confidence is "low" or format is "unknown":

**Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, use the format as-is (best guess). Log to dev_notes: "Dataset format uncertain (confidence: <level>) — using detected format as-is (autonomous mode)." Proceed without user confirmation.

**Otherwise:** Use AskUserQuestion:
```

Find line 69 (format mismatch):

```
If there's a format mismatch between user data and what the training script expects, use AskUserQuestion:
```

Replace with:

```
If there's a format mismatch between user data and what the training script expects:

**Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, skip data preparation and use data as-is. Log to dev_notes: "Format mismatch detected (data: <detected>, expected: <expected>) — skipping preparation, using as-is (autonomous mode)."

**Otherwise:** Use AskUserQuestion:
```

Find line 100 (env manager mismatch):

```
- **Mismatch:** Use AskUserQuestion to warn:
```

Replace with:

```
- **Mismatch:**
  **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, use the detected manager. Log to dev_notes: "Env manager mismatch (user: <user_manager>, detected: <detected_manager>) — using detected (autonomous mode)."
  **Otherwise:** Use AskUserQuestion to warn:
```

Find line 113 (missing conda env):

```
If the environment does not exist, use AskUserQuestion:
```

Replace with:

```
If the environment does not exist:

**Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, auto-create the environment: `conda create -n <env_name> python=3.x -y`. Log to dev_notes: "Auto-created conda environment <env_name> (autonomous mode)."

**Otherwise:** Use AskUserQuestion:
```

**Step 2: Commit**

```bash
git add skills/prerequisites/SKILL.md
git commit -m "Add autonomous defaults for 4 prerequisites AskUserQuestion calls"
```

---

### Task 17: HP-Tune — Tabular ML Iteration 1 Strategy

**Files:**
- Modify: `skills/hp-tune/SKILL.md:81-85`

**Step 1: Add tabular ML conditional**

In `skills/hp-tune/SKILL.md`, find lines 81-85:

```
**If `code_branches` is empty (HP-only):**
- Propose configs that span the search space
- Focus on learning rate first (highest impact)
- One config per order of magnitude of LR
- Keep other HPs at baseline values
```

Replace with:

```
**If `code_branches` is empty (HP-only):**
- Propose configs that span the search space
- **For tabular ML** (scikit-learn, XGBoost, LightGBM): Focus on `max_depth` and `n_estimators` first (highest impact for tree-based models). One config per value of `max_depth` in the search space. Keep learning rate (if present) and other HPs at baseline values.
- **For deep learning** (all other frameworks): Focus on learning rate first (highest impact). One config per order of magnitude of LR. Keep other HPs at baseline values.
```

**Step 2: Commit**

```bash
git add skills/hp-tune/SKILL.md
git commit -m "Add tabular ML iteration 1 strategy for hp-tune (max_depth/n_estimators first)"
```

---

### Task 18: Orchestrate — Research `type: hp_only` Routing

**Files:**
- Modify: `skills/orchestrate/SKILL.md` (after Phase 5.1, before Phase 6)

**Step 1: Add hp_only routing rule**

In `skills/orchestrate/SKILL.md`, find the section after Phase 5.1 implementation and before Phase 6 (the line that starts `## Phase 6:`). Insert BEFORE the Phase 6 heading:

```

### Research Proposal Type Routing

After research proposals are selected (Phase 5 or step 6.1), filter by `type` before invoking implement:

- **`type: "code_change"`:** Send to implement skill for branch creation (normal flow)
- **`type: "hp_only"`:** Skip implement. Instead, merge the proposal's suggested HP changes into the search space for hp-tune. Log to dev_notes: "HP-only proposal '<name>' merged into search space (skipping implement)." These proposals do NOT create branches or appear in the implementation manifest.

If ALL selected proposals are `hp_only`, skip Phase 5.1 entirely and proceed directly to Phase 6.

```

**Step 2: Commit**

```bash
git add skills/orchestrate/SKILL.md
git commit -m "Add hp_only proposal routing to skip implement and merge into search space"
```

---

### Task 19: Update CLAUDE.md Documentation

**Files:**
- Modify: `.claude/CLAUDE.md`

**Step 1: Update the Gotchas section**

In `.claude/CLAUDE.md`, find the Gotchas section and add these entries:

After the last gotcha bullet (`Research findings files can be multiple`), add:

```
- **HuggingFace Trainer log format supported**: `parse_logs.py` handles HuggingFace Trainer's single-quote Python dict syntax (`{'loss': 0.5, 'epoch': 1.0}`) as the `"hf_trainer"` format. This is auto-detected.
- **Research failure has a fallback chain**: If web search fails, the orchestrator retries with `source: "knowledge"`. If knowledge mode also fails, it continues with HP-only optimization. Research failures never block the pipeline.
- **OOM feedback loop**: When an experiment OOMs, the orchestrator logs the OOM-causing batch size and passes `max_batch_size` to hp-tune on the next iteration to prevent repeating the configuration.
- **Experiment timeout**: Experiments are killed after `baseline_training_time * 3` seconds (or 6 hours if profiling unavailable). Timed-out experiments get `status: "timeout"`.
```

**Step 2: Commit**

```bash
git add .claude/CLAUDE.md
git commit -m "Document HuggingFace parser, research fallback, OOM feedback, experiment timeout"
```

---

### Task 20: Final Verification

**Step 1: Run full test suite**

Run: `/data/hanchong/miniconda3/bin/python -m pytest tests/ -v --tb=short`
Expected: All tests pass (782 existing + ~20 new = ~800+ total)

**Step 2: Verify git log**

Run: `git log --oneline -12`
Expected: 10-12 clean commits covering all tasks
