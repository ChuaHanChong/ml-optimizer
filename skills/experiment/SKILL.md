---
name: experiment
description: "Run a single ML training experiment. Generates bash scripts, executes training on a specified GPU, and parses results. Use when: need to run a training experiment with a specific configuration."
disable-model-invocation: true
user-invocable: false
---

# Experiment Runner

Runs a single training experiment with a specified configuration.

## Inputs Expected

From the orchestrator or hp-tune skill:
- `exp_id`: Experiment identifier (e.g., "exp-001")
- `config`: Dictionary of hyperparameters
- `gpu_id`: GPU index to use
- `project_root`: Project root directory
- `train_command`: Base training command (from baseline)
- `eval_command`: Evaluation command (optional)
- `code_branch`: Git branch with code changes (optional, from implement manifest)
- `code_proposal`: Name of the research proposal (optional, for tagging results)
- `proposal_source`: Origin of the proposal — `"paper"`, `"llm_knowledge"`, or `null` (pass-through from hp-tune)
- `method_tier`: Which tier this experiment belongs to — `"baseline"`, `"method_default_hp"`, `"method_tuned_hp"`, `"stacked_default_hp"`, or `"stacked_tuned_hp"` (pass-through from hp-tune)
- `iteration`: HP tuning iteration that produced this config (integer, from hp-tune proposed config)
- `prepared_train_path`: Path to prepared training data (optional, from prerequisites)
- `prepared_val_path`: Path to prepared validation data (optional, from prerequisites)
- `code_branches`: List of method branches combined in this stacking experiment (optional, from orchestrator Phase 8)
- `stacking_order`: Position in the stacking chain — 1 = best method alone, 2 = best + second, etc. (optional, integer)
- `stack_base_exp`: Experiment ID of the previous stack step this builds on (optional)

## Reference

- Script templates: `references/script-templates.md` (in this skill's directory)

## Step 1: Set Up Code Environment

If `code_branch` is provided (from implementation manifest):

1. **Use git worktree** instead of checkout (avoids conflicts with parallel experiments):
   ```bash
   git worktree add experiments/worktrees/<exp_id> <code_branch>
   ```
2. Verify the branch exists and the expected modified files are present in the worktree
3. Run training commands from within the worktree directory
4. **After training completes**, remove the worktree:
   ```bash
   git worktree remove experiments/worktrees/<exp_id>
   ```

If no `code_branch` is provided: use the current code as-is (HP-only experiment). Skip this step.

**Fallback:** If `git worktree` is not available (old git version), fall back to `git checkout` with a warning that parallel experiments on different branches will conflict.

## Step 1.1: Pre-Flight Checks

Before building the training command, verify:

1. **Disk space:** Check that the target filesystem has sufficient free space for logs and checkpoints:
   ```bash
   df -h <project_root> | tail -1
   ```
   Warn if less than 5 GB free.

2. **Timeout enforcement:**
   - **If `fixed_time_budget` is set** (from Phase 0 user_choices): use it directly as `timeout_seconds`. All experiments train for exactly this many seconds. When the budget expires (exit code 124 from `timeout`), this is NOT an error — set `status: "completed"`. Include `"time_budget_seconds": <value>` in the result JSON.
   - **Otherwise:** If the orchestrator passes a `timeout_seconds` value, use it directly (it computes `baseline_training_time * 3`). Otherwise, compute a timeout:
     - If `baseline.json` has `profiling.estimated_timeout_seconds` (tabular ML): `timeout_seconds = profiling.estimated_timeout_seconds`
     - Else if `baseline.json` has `profiling.throughput_samples_per_sec` (iterative DL): `timeout_seconds = int(1.5 × (dataset_size × epochs) / throughput)`
     - If neither available: `timeout_seconds = 14400` (4 hours default)
   - Cap at 86400 (24 hours maximum)
   - Store `timeout_seconds` for use in Step 3 script generation

## Step 2: Build Training Command

Construct the full training command by overriding the base command with experiment-specific config:

1. Read the base training command from `experiments/results/baseline.json`
1.4. **Validate prepared data paths:** If `prepared_train_path` or `prepared_val_path` was provided:
   - Verify each path exists on disk (file or directory)
   - If a path does not exist, log a warning to `experiments/dev_notes.md` and fall back to the original `train_data_path`/`val_data_path`
   - Log to error tracker with `category: "config_error"`, `severity: "warning"`, `source: "experiment"`
1.5. **Apply prepared data paths:** If `prepared_train_path` or `prepared_val_path` was provided (and validated in 1.4):
   a. **CLI substitution:** Check if the original `train_data_path`/`val_data_path` appears as a literal substring in the train_command. If found, replace it with the prepared path.
   b. **Config file substitution:** If not found in the train_command, read the training config file (YAML/JSON) and search for the original data path. Create a modified config copy at `experiments/logs/<exp_id>/config_modified.yaml` with the path updated.
   c. **No match:** Log a warning to dev_notes.md: "Could not find original data path in train_command or config — proceeding with original paths." Pass the prepared paths as additional CLI args if the training script accepts generic data path arguments (detected in Phase 1).
2. Determine how the project accepts config overrides:
   - **CLI args:** `python train.py --lr 0.001 --batch_size 16`
   - **Config file:** Modify a YAML/JSON config, then `python train.py --config <path>`
   - **Environment vars:** `LR=0.001 python train.py`
3. Build the override command

### Config Override Validation

After building the override command, verify it actually takes effect:
1. Run a 1-step dry run (if the training script supports `--max_steps 1` or similar)
2. Parse the first log output to verify the config values match what was intended
3. If the override didn't take effect (e.g., training script ignores `--lr` arg), try an alternative override method

### Config Override Strategy

Read the training script to determine the override method:
- If it uses `argparse`: use CLI argument overrides
- If it uses a config file (OmegaConf, yaml.load): create a modified config copy
- If it uses environment variables: set them in the script

For config file approach, write a modified config to:
`experiments/logs/<exp_id>/config.yaml`

## Step 2.1: Artifact Storage

Save model checkpoints, intermediate outputs, and visualizations to:
```
experiments/artifacts/<exp-id>/
```

Create the per-experiment subdirectory before training:
```bash
mkdir -p experiments/artifacts/<exp_id>
```

If the training command produces checkpoint files (`*.pt`, `*.pth`, `*.ckpt`, `*.h5`, `*.pkl`, `*.safetensors`), configure the save path to point here. Add the artifact path to the generated training script via `--checkpoint_dir`, `--save_dir`, `--output_dir`, or whichever flag the training script uses.

## Step 3: Generate Bash Script

Use the experiment setup script:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/experiment_setup.py \
  <project_root> \
  "<full_train_command>" \
  <gpu_id> \
  '<config_json>'
```

Or write the script manually using the Write tool, following templates in `references/script-templates.md`.

**Timeout wrapper:** The training command in the bash script must be wrapped with `timeout`:
```bash
timeout --signal=SIGTERM --kill-after=60 {timeout_seconds} {train_command} 2>&1 | tee experiments/logs/{exp_id}/train.log
EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 124 ]; then
    echo "TIMEOUT: Training exceeded {timeout_seconds}s limit" >> experiments/logs/{exp_id}/train.log
fi
```

The script must:
- Set `CUDA_VISIBLE_DEVICES=<gpu_id>`
- Create the log directory
- Run training with output logged to `experiments/logs/<exp_id>/train.log`
- Include any environment variables needed

Save to: `experiments/scripts/<exp_id>/<exp_id>.sh`

## Step 3.1: Write Placeholder Result

Before starting training, write a placeholder result file so the monitor and `cleanup_stale` can track this experiment:

```json
{
  "exp_id": "<exp_id>",
  "status": "running",
  "config": <config>,
  "metrics": {},
  "gpu_id": <gpu_id>,
  "log_file": "experiments/logs/<exp_id>/train.log",
  "script_file": "experiments/scripts/<exp_id>/<exp_id>.sh",
  "code_branch": "<code_branch or null>",
  "code_proposal": "<code_proposal or null>",
  "proposal_source": "<proposal_source or null>",
  "method_tier": "<method_tier or null>",
  "iteration": <iteration>,
  "timestamp": "<ISO 8601 UTC timestamp>",
  "notes": "Training in progress"
}
```

Write to: `experiments/results/<exp_id>.json`

**Why:** This prevents a race condition where the monitor detects divergence and writes a minimal result file (missing metadata like `code_branch`, `method_tier`, `iteration`) before the experiment agent has written anything. With the placeholder, the monitor sees `status: "running"` and updates to `"diverged"` while preserving all metadata fields.

Validate the placeholder:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/schema_validator.py \
  experiments/results/<exp_id>.json result
```

## Step 4: Execute Training

Run the experiment:
```bash
bash experiments/scripts/<exp_id>/<exp_id>.sh
```

**For foreground execution** (when called directly):
- Run via Bash tool and wait for completion
- Monitor output for early signs of problems

**For background execution** (when called by orchestrator in parallel):
- Run via Bash tool with `run_in_background: true`
- The monitor skill will handle divergence detection

## Step 4.1: Early Abort Check

After training starts, perform a fast sanity check on the first few log entries — **independent of the monitor skill**:

1. Wait for the first 5-10 training steps to appear in the log (poll `experiments/logs/<exp_id>/train.log` briefly)
2. Parse the initial loss values using:
   ```bash
   python3 ~/.claude/plugins/ml-optimizer/scripts/parse_logs.py experiments/logs/<exp_id>/train.log
   ```
3. **Abort immediately** if any of these conditions are met:
   - Loss is `NaN` or `Inf` in the first 10 steps
   - Loss exceeds 10× the baseline's initial loss (read from `experiments/results/baseline.json` → `metrics.loss` or first logged loss value)
   - Training process already exited with non-zero code

4. If aborting:
   - Kill the training process (if still running)
   - Write results with `"status": "failed"` and note: `"Early abort: <reason> in first 10 steps"`
   - Log to error tracker:
     ```bash
     python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"training_failure","severity":"warning","source":"experiment","message":"Early abort: <reason>","exp_id":"<exp_id>","config":<config_json>,"context":{"abort_step":<step>,"loss_value":<value>}}'
     ```
   - Skip to Step 6 (Write Results) — do not wait for full training

5. If the first steps look healthy, continue waiting for training to complete normally.

**Note:** This check is a fast pre-filter, not a replacement for the monitor skill. The monitor handles gradual divergence (plateau, slow explosion). This handles obvious failures that waste training time.

## Step 5: Parse Results

After training completes:

1. Parse the training log:
   ```bash
   python3 ~/.claude/plugins/ml-optimizer/scripts/parse_logs.py experiments/logs/<exp_id>/train.log
   ```

2. **If an eval command was provided, run evaluation** (mandatory — the primary_metric often comes from eval output):
   ```bash
   <eval_command>
   ```
   Parse eval output for final metrics.

   **Worktree experiments:** Evaluation MUST run inside the worktree directory (before `git worktree remove`). Copy model checkpoints/artifacts from the worktree to `experiments/artifacts/<exp_id>/` BEFORE removing the worktree.

3. Extract key metrics:
   - Final loss value
   - Best metric value (PSNR, accuracy, etc.)
   - Training duration
   - Any other relevant metrics

4. **Validate required metrics:** Ensure `metrics` includes the `divergence_metric` (for monitor) and `primary_metric` (for analyze/hp-tune). If either is missing from parsed output, check the raw log for alternative names (e.g., `train_loss`, `val_loss`). If a match is found, include it under both the original and canonical name. If not found, set to `null` and log a warning.

## Step 6: Write Results

**Note:** This overwrites the placeholder result from Step 3.1. If the monitor has already updated the placeholder to `status: "diverged"`, check the current file status first — if the experiment completed successfully despite the monitor's divergence call, use `status: "completed"` (the experiment's own metrics are authoritative).

Write experiment results to `experiments/results/<exp_id>.json`:

```json
{
  "exp_id": "<exp_id>",
  "status": "completed",
  "config": {
    "lr": <value>,
    "batch_size": <value>,
    ...
  },
  "metrics": {
    "loss": <final_loss>,
    "<primary_metric>": <best_value>,
    ...
  },
  "gpu_id": <gpu_id>,
  "duration_seconds": <training_time>,
  "log_file": "experiments/logs/<exp_id>/train.log",
  "script_file": "experiments/scripts/<exp_id>/<exp_id>.sh",
  "code_branch": "<branch name or null>",
  "code_proposal": "<proposal name or null>",
  "proposal_source": "<paper|llm_knowledge|null>",
  "method_tier": "<baseline|method_default_hp|method_tuned_hp|stacked_default_hp|stacked_tuned_hp>",
  "iteration": <tuning_iteration>,
  "code_branches": ["<branch1>", "<branch2>"],
  "stacking_order": <integer>,
  "stack_base_exp": "<exp_id of previous stack step>",
  "notes": "<any observations>"
}
```

## Step 6.1: Validate Output

```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/schema_validator.py \
  experiments/results/<exp_id>.json result
```

If validation fails, read the errors, fix the JSON file, and re-validate. Do not proceed to Step 7 until validation passes.

## Step 7: Report Back

Return to the orchestrator:
- Experiment ID
- Status (completed/failed/diverged)
- Key metrics
- Path to results file
- Any issues encountered

## Error Handling — Auto-Repair Loop

When training fails, classify the error and either retry (up to 3 attempts) or report immediately:

### Non-Retryable (report immediately, no retry):
- **OOM** (`CUDA out of memory`): deterministic for same config — retrying wastes time. Write `status: "failed"`, log with `error_type: "oom"`
- **Divergence** (detected by monitor): config is inherently unstable. Write `status: "diverged"`
- **Timeout**: config takes too long. Write `status: "timeout"`
- **SyntaxError / IndentationError**: code bug, not fixable by retry
- **Identical error on retry**: if attempt 2 produces the same stderr (first 200 chars match), skip attempt 3

### Retryable (auto-repair up to 3 attempts):
1. **Attempt 1 fails:** Capture stderr, classify error
2. **Diagnose and fix:**
   - `FileNotFoundError` on checkpoint/data → verify paths, check worktree setup
   - `RuntimeError: NCCL` / distributed error → retry with `CUDA_VISIBLE_DEVICES` single GPU
   - `ImportError` / `ModuleNotFoundError` → install missing package
   - `ValueError` / `TypeError` in config → check config override syntax (string vs number)
   - `PermissionError` → fix file permissions
   - `ConnectionError` / `HTTPError` → transient network error, wait 5s, retry
3. **Log each retry:** `category: "training_failure", severity: "warning", source: "experiment", context: {"original_error": "<error>", "fix": "<description>", "attempt": <N>}`
4. **Attempt 2** with fix → if fails with new error, apply new fix → **Attempt 3**
5. **All 3 fail:** Write `status: "failed"`, include all error history in `notes`

Retry time counts toward `duration_seconds` (single experiment, not separate entries).

### Specific error handling:

- **Training crashes:**
  - Capture the error output
  - Apply auto-repair loop (above) for retryable errors
  - Write results with `"status": "failed"` and error message in notes
  - Log to error tracker:
    ```bash
    python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"training_failure","severity":"critical","source":"experiment","message":"<error description>","exp_id":"<exp_id>","config":<config_json>,"stack_trace":"<last 20 lines of stderr>"}'
    ```

- **Divergence detected (by monitor):**
  - Training is killed by the monitor skill
  - Write results with `"status": "diverged"` and divergence details

- **GPU out of memory:**
  - Common cause: batch size too large
  - Write results with `"status": "failed"` and note the OOM error
  - The hp-tune skill will adjust batch size in next iteration
  - Log to error tracker:
    ```bash
    python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"training_failure","severity":"critical","source":"experiment","message":"GPU OOM with batch_size=<batch_size>","exp_id":"<exp_id>","config":<config_json>,"context":{"error_type":"oom","batch_size":<batch_size>}}'
    ```

- **Config override not working:**
  - If CLI args don't override correctly, try config file approach
  - If neither works, report the issue back to orchestrator
  - Log to error tracker:
    ```bash
    python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"config_error","severity":"warning","source":"experiment","message":"Config override failed: <method tried>","exp_id":"<exp_id>"}'
    ```

- **Training timeout:**
  - The `timeout` command kills the process with SIGTERM (then SIGKILL after 60s)
  - Parse any partial results from the log before the timeout
  - Write results with `"status": "timeout"` and note the timeout duration
  - Log to error tracker with `category: "timeout"`, `severity: "warning"`, `source: "experiment"`
  - The monitor skill will also detect the process death and mark accordingly
