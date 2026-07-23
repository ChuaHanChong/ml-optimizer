---
name: experiment
description: "Run a single ML training experiment. Generates bash scripts, executes training on a specified GPU, and parses results. Use when: need to run a training experiment with a specific configuration."
user-invocable: false
---

# Experiment Runner

> **Path convention:** All `<exp_root>/...` paths refer to the `exp_root` dispatch parameter. The plugin does not hardcode the output directory name.

## Reference

- Script templates: `${CLAUDE_SKILL_DIR}/references/script-templates.md` (in this skill's directory)

## Inputs Expected

From the orchestrator or hp-tune skill:
- `exp_id`: experiment identifier (e.g., "exp-001")
- `config`: dictionary of hyperparameters
- `gpu_id`: GPU index to use
- `project_root`: project root directory
- `train_command`: base training command (from baseline)
- `eval_command`: evaluation command (optional)
- `code_branch`: git branch with code changes (optional, from implement manifest)
- `code_proposal`: name of the research proposal (optional, for tagging results)
- `proposal_source`: origin of the proposal — `"paper"`, `"llm_knowledge"`, or `null` (pass-through from hp-tune)
- `method_tier`: which tier this experiment belongs to — `"baseline"`, `"method_default_hp"`, `"method_tuned_hp"`, `"stacked_default_hp"`, or `"stacked_tuned_hp"` (pass-through from hp-tune)
- `iteration`: HP tuning iteration that produced this config (integer, from hp-tune proposed config)
- `prepared_train_path`: path to prepared training data (optional, from prerequisites)
- `prepared_val_path`: path to prepared validation data (optional, from prerequisites)
- `code_branches`: list of method branches combined in this stacking experiment (optional, from orchestrator Phase 8)
- `stacking_order`: position in the stacking chain — 1 = best method alone, 2 = best + second, etc. (optional, integer)
- `stack_base_exp`: experiment ID of the previous stack step this builds on (optional)
- `checkpoint_source`: checkpoint warm-start info (optional). Dict with `exp_id` (source experiment) and `checkpoint_path` (checkpoint file). When provided, the experiment warm-starts from this checkpoint.
- `round_dir`: current round directory name (e.g., `"round-3-hp"`). **Required.** Passed by the orchestrator after `round_manager.py create-round`. All experiment result JSONs and proposed-config JSONs MUST be written inside this round's subdirectory. If missing from the dispatch context, fetch it:

```bash
round_dir=$(python3 ${CLAUDE_PLUGIN_ROOT}/scripts/round_manager.py <exp_root> current-round | python3 -c "import json,sys; print(json.load(sys.stdin)['dir'])")
```

**Why this matters:** a PreToolUse hook (`validate_experiment_write.py`) blocks any Write/Edit of `exp-*.json` outside a `round-N-<type>/` subdirectory. Writing to the flat `results/` path will fail.

## Step 1: Set Up Code Environment

> **Goal check:** check for frozen parameters and resource constraints from the optimization goals before running.

If `code_branch` is provided (from implementation manifest):

1. **Compute a safe worktree path OUTSIDE `<exp_root>/`** to avoid a cleanup
   race condition. A parallel experiment-agent dispatch can wipe sibling
   `<exp_root>/*` subdirs if any has a worktree under `<exp_root>/`.
   Put the worktree in a system temp dir instead:
   ```bash
   PROJECT_HASH=$(echo "<project_root>" | sha1sum | cut -c1-8)
   WORKTREE_ROOT="/tmp/ml-opt-worktrees-${PROJECT_HASH}"
   mkdir -p "$WORKTREE_ROOT"
   WORKTREE_PATH="$WORKTREE_ROOT/<exp_id>"
   # --detach: lets multiple parallel experiments share the same <code_branch>
   # (plain `git worktree add <path> <branch>` fails "already checked out" for the 2nd+).
   # Experiments only read the branch to run training, so a detached checkout is correct.
   git worktree add --detach "$WORKTREE_PATH" <code_branch>
   ```
   **Never** create worktrees under `<exp_root>/` or under `<project_root>/`
   itself. See the "Worktree race" lesson in `dev_notes.md` if you see
   unexplained file loss mid-run.
2. Verify the branch exists and the expected modified files are present in the worktree
3. Run training commands from within the worktree directory
4. **Copy artifacts BEFORE cleanup** — model checkpoints, eval reports, and
   any files to preserve must be copied out of the worktree to
   `<exp_root>/artifacts/<round_dir>/<exp_id>/` while the worktree is still alive.
5. **After training completes**, validate-then-remove the worktree. Never
   `rm -rf` a worktree — always let git scope the cleanup:
   ```bash
   # Assert the path is registered before removing
   if git worktree list --porcelain | grep -q "worktree $WORKTREE_PATH$"; then
       git worktree remove "$WORKTREE_PATH"
   else
       # Stale/unregistered — prune instead of blind rm
       git worktree prune
   fi
   ```
   If `git worktree remove` reports "contains modified or untracked files"
   you haven't copied artifacts out yet (Step 4). Go back to Step 4 — do NOT
   add `--force`, it may clobber work from a parallel experiment.

If no `code_branch` is provided: use the current code as-is (HP-only experiment). Skip this step.

**Fallback:** if `git worktree` is unavailable (old git version), fall back to `git checkout` with a warning that parallel experiments on different branches will conflict.

## Step 1.1: Pre-Flight Checks

Before building the training command, verify:

1. **Disk space:** check the target filesystem has enough free space for logs and checkpoints:
   ```bash
   df -h <project_root> | tail -1
   ```
   Warn if less than 5 GB free.

2. **Timeout enforcement:**
   - **If `fixed_time_budget` is set** (from Phase 0 user_choices): use it directly as `timeout_seconds`. All experiments train for exactly this many seconds. When the budget expires (exit code 124 from `timeout`), this is NOT an error — set `status: "completed"`. Include `"time_budget_seconds": <value>` in the result JSON. **Checkpoint pre-flight:** verify the training script checkpoints periodically (look for `save_freq`, `checkpoint_interval`, `ModelCheckpoint`, periodic `save_checkpoint` calls) or use a framework-native time limit (Lightning `--max_time`, HF `TrainingArguments`). If NEITHER exists, the SIGTERM kill leaves no final checkpoint — warn and record `"eval_checkpoint_missing": true` in the result notes rather than scoring a stale checkpoint.
   - **If `fixed_epoch_budget` is set** (from Phase 0 user_choices): override the epoch count in the training command (e.g., `--epochs <fixed_epoch_budget>`). All experiments train for exactly this many epochs. Include `"epoch_budget": <value>` in the result JSON. Still apply the safety timeout below.
   - **If `fixed_step_budget` is set** (from Phase 0 user_choices — environment timesteps, RL): override the timestep count via the framework's flag (e.g., `--total_timesteps <fixed_step_budget>`). All experiments train for exactly this many environment steps. Include `"step_budget": <value>` in the result JSON. Still apply the safety timeout below.
   - **Otherwise:** if the orchestrator passes a `timeout_seconds` value, use it directly (it computes `baseline_training_time * 3`). Otherwise compute a timeout:
     - If `baseline.json` has `profiling.estimated_timeout_seconds`: `timeout_seconds = profiling.estimated_timeout_seconds` (recorded by the baseline skill for tabular ML and RL)
     - Else if `baseline.json` has `profiling.training_duration_seconds`: `timeout_seconds = profiling.training_duration_seconds * 3`
     - Else if `baseline.json` has `profiling.throughput_samples_per_sec` (iterative DL): `timeout_seconds = int(1.5 × (dataset_size × epochs) / throughput)`
     - If none available: `timeout_seconds = 21600` (6 hours — the pipeline-wide fallback; matches CLAUDE.md "Experiment timeout" and phase-7-experiment-loop.md)
   - Cap at 86400 (24 hours maximum)
   - Store `timeout_seconds` for use in Step 3 script generation

### Step 1.2: Checkpoint Validation (if `checkpoint_source` provided)

1. Verify the checkpoint file exists at `checkpoint_source.checkpoint_path`
2. If missing, log warning to error tracker and fall back to from-scratch training
3. Read the training script to determine the checkpoint-loading flag (e.g., `--resume`, `--checkpoint`, `--init_checkpoint`)
4. If no loading mechanism found, set `CHECKPOINT_PATH` environment variable and proceed (the generated script already exports it)

## Step 1.3: Capture Reproducibility Metadata

Before running training, capture environment state for reproduction:

1. **Random seed**: if the training script doesn't already set one, generate a seed and set it:
   ```bash
   export PYTHONHASHSEED=<seed>
   ```
   Record the seed (or null if the training script manages its own seeding).

2. **Environment snapshot**:
   ```bash
   pip freeze > <exp_root>/logs/<round_dir>/<exp_id>/pip_freeze.txt 2>/dev/null || true
   ```

3. **Git state**: record `git rev-parse HEAD` (supplements `code_branch` with the exact commit).

4. **Framework version**: extract from pip freeze (e.g., `torch==2.x`, `tensorflow==2.x`).

Include in the result JSON under a `"reproducibility"` key:
```json
"reproducibility": {
  "random_seed": null,
  "environment_file": "<exp_root>/logs/<round_dir>/<exp_id>/pip_freeze.txt",
  "git_sha": "<sha>",
  "framework_version": "<version>"
}
```

## Step 2: Build Training Command

Construct the full training command by overriding the base command with experiment-specific config:

1. Read the base training command from `<exp_root>/results/baseline.json`
1.4. **Validate prepared data paths:** if `prepared_train_path` or `prepared_val_path` was provided:
   - Verify each path exists on disk (file or directory)
   - If a path does not exist, log a warning to `<exp_root>/dev_notes.md` and fall back to the original `train_data_path`/`val_data_path`
   - Log to error tracker with `category: "config_error"`, `severity: "warning"`, `source: "experiment"`
1.5. **Apply prepared data paths:** if `prepared_train_path` or `prepared_val_path` was provided (and validated in 1.4):
   a. **CLI substitution:** check if the original `train_data_path`/`val_data_path` appears as a literal substring in the train_command. If found, replace it with the prepared path.
   b. **Config file substitution:** if not found in the train_command, read the training config file (YAML/JSON) and search for the original data path. Create a modified config copy at `<exp_root>/logs/<round_dir>/<exp_id>/config_modified.yaml` with the path updated.
   c. **No match:** log a warning to dev_notes.md: "Could not find original data path in train_command or config — proceeding with original paths." Pass the prepared paths as additional CLI args if the training script accepts generic data path arguments (detected in Phase 1).
2. Determine how the project accepts config overrides:
   - **CLI args:** `python train.py --lr 0.001 --batch_size 16`
   - **Config file:** modify a YAML/JSON config, then `python train.py --config <path>`
   - **Environment vars:** `LR=0.001 python train.py`
3. Build the override command

### Config Override Validation

After building the override command, verify it actually takes effect:
1. Run a 1-step dry run (if the training script supports `--max_steps 1` or similar)
2. Parse the first log output to verify the config values match what was intended
3. If the override didn't take effect (e.g., training script ignores `--lr` arg), try an alternative override method

### Config Override Strategy

Read the training script to determine the override method:
- `argparse`: use CLI argument overrides
- config file (OmegaConf, yaml.load): create a modified config copy
- environment variables: set them in the script

For the config file approach, write a modified config to:
`<exp_root>/logs/<round_dir>/<exp_id>/config.yaml`

## Step 2.1: Artifact Storage

Save model checkpoints, intermediate outputs, and visualizations to:
```
<exp_root>/artifacts/<round_dir>/<exp-id>/
```

Create the per-experiment subdirectory before training:
```bash
mkdir -p <exp_root>/artifacts/<round_dir>/<exp_id>
```

If the training command produces checkpoint files (`*.pt`, `*.pth`, `*.ckpt`, `*.h5`, `*.pkl`, `*.safetensors`), configure the save path to point here. Add the artifact path to the generated training script via `--checkpoint_dir`, `--save_dir`, `--output_dir`, or whichever flag the training script uses.

## Step 3: Generate Bash Script

Use the experiment setup script:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/experiment_setup.py \
  <project_root> \
  "<full_train_command>" \
  <gpu_id> \
  '<config_json>' \
  <round_dir> \
  '<env_vars_json>'   # optional, e.g., '{"MUJOCO_GL": "egl"}' — from baseline.json profiling.sim_env
```

**Simulator env vars:** read `profiling.sim_env` from `<exp_root>/results/baseline.json` and pass it as the `env_vars_json` argument (it feeds `generate_script`'s `env_vars` param) so every experiment script exports the same headless-rendering env vars as baseline. Omit the argument when `sim_env` is empty or absent.

The `round_dir` argument tells `setup()` to create the placeholder result inside `results/<round_dir>/<exp_id>.json` instead of the flat `results/` path. This is MANDATORY — writes to the flat path are blocked by the PreToolUse hook.

Or write the script manually using the Write tool, following templates in `${CLAUDE_SKILL_DIR}/references/script-templates.md`.

**Timeout wrapper:** the training command in the bash script must be wrapped with `timeout`, launched in the background with output redirected to the log, its PID (`$!`) recorded, and the REAL exit code propagated:
```bash
timeout --signal=SIGTERM --kill-after=60 {timeout_seconds} {train_command} > <exp_root>/logs/{round_dir}/{exp_id}/train.log 2>&1 &
echo $! > <exp_root>/logs/{round_dir}/{exp_id}/pid
EXIT_CODE=0
wait $! || EXIT_CODE=$?
if [ $EXIT_CODE -eq 124 ]; then
    echo "TIMEOUT: Training exceeded {timeout_seconds}s limit" >> <exp_root>/logs/{round_dir}/{exp_id}/train.log
fi
exit $EXIT_CODE
```
Exit code 124 here means the SAFETY timeout fired → `status: "timeout"`. 124 is treated as success ONLY in the fixed-time-budget branch (Step 1.1).

The script must:
- Set `CUDA_VISIBLE_DEVICES=<gpu_id>`
- Create the log directory
- Run training with output logged to `<exp_root>/logs/<round_dir>/<exp_id>/train.log`
- Include any needed environment variables (including `profiling.sim_env` headless-simulator vars from baseline.json, forwarded via `env_vars_json`)

Save to: `<exp_root>/scripts/<round_dir>/<exp_id>/train.sh`

## Step 3.1: Write Placeholder Result

Before starting training, write a placeholder result file so the monitor and `cleanup_stale` can track this experiment:

```json
{
  "exp_id": "<exp_id>",
  "status": "running",
  "config": <config>,
  "metrics": {},
  "gpu_id": <gpu_id>,
  "log_file": "<exp_root>/logs/<round_dir>/<exp_id>/train.log",
  "script_file": "<exp_root>/scripts/<round_dir>/<exp_id>/train.sh",
  "code_branch": "<code_branch or null>",
  "code_proposal": "<code_proposal or null>",
  "proposal_source": "<proposal_source or null>",
  "method_tier": "<method_tier or null>",
  "iteration": <iteration>,
  "reproducibility": null,
  "timestamp": "<ISO 8601 UTC timestamp>",
  "notes": "Training in progress"
}
```

Write to: `<exp_root>/results/<round_dir>/<exp_id>.json`

**Why:** this prevents a race where the monitor detects divergence and writes a minimal result file (missing metadata like `code_branch`, `method_tier`, `iteration`) before the experiment agent has written anything. With the placeholder, the monitor sees `status: "running"` and updates to `"diverged"` while preserving all metadata fields.

Validate the placeholder:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/schema_validator.py \
  <exp_root>/results/<round_dir>/<exp_id>.json result
```

## Step 4: Execute Training

Run the experiment:
```bash
bash <exp_root>/scripts/<round_dir>/<exp_id>/train.sh
```

**For foreground execution** (when called directly):
- Run via Bash tool and wait for completion
- Monitor output for early signs of problems

**For background execution** (when dispatched from the Phase 7 workflow in parallel):
- Run via Bash tool with `run_in_background: true`
- **You own in-run monitoring** — there is no separate concurrent monitor dispatch by default. While training runs, poll the training log every 5 minutes:
  ```bash
  python3 ${CLAUDE_PLUGIN_ROOT}/scripts/parse_logs.py <exp_root>/logs/<round_dir>/<exp_id>/train.log
  python3 ${CLAUDE_PLUGIN_ROOT}/scripts/detect_divergence.py '<divergence_metric_values_json>' --model-category <model_category>
  ```
  using the dispatched `divergence_metric` values (add `--higher-is-better` when `divergence_lower_is_better` is false) and `model_category`.
- On `status: "diverged"` (NaN/Inf, explosion/crash, reward collapse): kill the training process and write `status: "diverged"` with the reason in notes.
- On `status: "plateaued"` (plateau/drift): do **NOT** kill — record the warning in notes so the analysis agent sees it. A plateaued run finishes its budget.
- Skip polling entirely when `divergence_metric` is null (tabular ML).

## Step 4.1: Early Abort Check

After training starts, run a fast sanity check on the first few log entries — **independent of the monitor skill**:

1. Wait for the first 5-10 training steps to appear in the log (poll `<exp_root>/logs/<round_dir>/<exp_id>/train.log` briefly)
2. Parse the initial loss values:
   ```bash
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/parse_logs.py <exp_root>/logs/<round_dir>/<exp_id>/train.log
   ```
3. **Abort immediately** if any of these hold:
   - ANY parsed metric is `NaN` or `Inf` in the first 10 steps — check every metric, not just the watched one:
     ```bash
     python3 ${CLAUDE_PLUGIN_ROOT}/scripts/detect_divergence.py --scan-records '<parsed_records_json>'
     ```
   - The divergence metric exceeds 10× the baseline's initial value (read from `<exp_root>/results/baseline.json`) — this magnitude rule applies ONLY when the divergence metric is lower-is-better AND its baseline value is positive. Never apply it to reward-like or negative-valued metrics, where 10× the magnitude is meaningless or even a good sign.
   - Training process already exited with non-zero code

4. If aborting:
   - Kill the training process (if still running)
   - Write results with `"status": "failed"` and note: `"Early abort: <reason> in first 10 steps"`
   - Log to error tracker:
     ```bash
     python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"training_failure","severity":"warning","source":"experiment","message":"Early abort: <reason>","exp_id":"<exp_id>","config":<config_json>,"context":{"abort_step":<step>,"loss_value":<value>}}'
     ```
   - Skip to Step 6 (Write Results) — do not wait for full training

5. If the first steps look healthy, continue waiting for training to complete normally.

**Note:** this is a fast pre-filter, not a replacement for the monitor skill. The monitor handles gradual divergence (plateau, slow explosion); this handles obvious failures that waste training time.

## Step 5: Parse Results

After training completes:

1. Parse the training log:
   ```bash
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/parse_logs.py <exp_root>/logs/<round_dir>/<exp_id>/train.log
   ```

2. **If an eval command was provided, run evaluation** (mandatory — the primary_metric often comes from eval output):
   ```bash
   <eval_command>
   ```
   Parse eval output for final metrics.

   **Worktree experiments:** evaluation MUST run inside the worktree directory (before `git worktree remove`). Copy model checkpoints/artifacts from the worktree to `<exp_root>/artifacts/<round_dir>/<exp_id>/` BEFORE removing the worktree.

   **Time-budget checkpoint validation:** when `fixed_time_budget` is set and evaluation loads a checkpoint, verify the checkpoint file's mtime falls within THIS run's window (between training start and end). A checkpoint older than the run start is stale (a previous run or the initial state) — do NOT score it: set the eval-derived metrics to null, record `"eval_checkpoint_missing": true` in notes, and log to the error tracker (`category: "config_error"`, `severity: "warning"`, `source: "experiment"`).

3. Extract key metrics:
   - Final loss value
   - Best metric value (PSNR, accuracy, etc.)
   - Training duration
   - Any other relevant metrics

4. **Validate required metrics:** ensure `metrics` includes the `divergence_metric` (for monitor) and `primary_metric` (for analyze/hp-tune). If either is missing from parsed output, check the raw log for alternative names (e.g., `train_loss`, `val_loss`). If a match is found, include it under both the original and canonical name. If not found, set to `null` and log a warning.

### RL Evaluation Symmetry

When `model_category = "rl"` (or evaluation is rollout-based):

1. **Same protocol as baseline:** run the SAME evaluation protocol the baseline used (baseline skill, "RL Baseline Evaluation"): the same number of evaluation episodes and a deterministic policy. A different episode count or a stochastic policy makes the comparison to baseline meaningless.
2. **Record mean AND std:** report the episode-reward mean and std in `metrics` (e.g., `episode_reward_mean`, `episode_reward_std`).
3. **Best-over-training-curve is FORBIDDEN as the `primary_metric` value for RL** — training-curve rewards are exploration-noised and single-episode peaks are luck. Use the final-eval mean (or the mean of the last N evaluations) instead.

## Step 6: Write Results

**Note:** this overwrites the placeholder from Step 3.1. If the monitor already updated the placeholder to `status: "diverged"`, check the current file status first — if the experiment completed successfully despite the monitor's divergence call, use `status: "completed"` (the experiment's own metrics are authoritative).

Write experiment results to `<exp_root>/results/<round_dir>/<exp_id>.json`:

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
  "log_file": "<exp_root>/logs/<round_dir>/<exp_id>/train.log",
  "script_file": "<exp_root>/scripts/<round_dir>/<exp_id>/train.sh",
  "code_branch": "<branch name or null>",
  "code_proposal": "<proposal name or null>",
  "proposal_source": "<paper|llm_knowledge|null>",
  "method_tier": "<baseline|method_default_hp|method_tuned_hp|stacked_default_hp|stacked_tuned_hp>",
  "iteration": <tuning_iteration>,
  "checkpoint_source": {"exp_id": "<source_exp>", "checkpoint_path": "<path>"} | null,
  "warm_started": true | false,
  "code_branches": ["<branch1>", "<branch2>"],
  "stacking_order": <integer>,
  "stack_base_exp": "<exp_id of previous stack step>",
  "reproducibility": {
    "random_seed": "<seed_or_null>",
    "environment_file": "<exp_root>/logs/<round_dir>/<exp_id>/pip_freeze.txt",
    "git_sha": "<sha>",
    "framework_version": "<version>"
  },
  "notes": "<any observations>"
}
```

## Step 6.1: Validate Output

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/schema_validator.py \
  <exp_root>/results/<round_dir>/<exp_id>.json result
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
- **OOM** (`CUDA out of memory`): deterministic for the same config — retrying wastes time. Write `status: "failed"`, log with `error_type: "oom"`
- **Divergence** (detected by monitor): config is inherently unstable. Write `status: "diverged"`
- **Timeout**: config takes too long. Write `status: "timeout"`
- **SyntaxError / IndentationError**: code bug, not fixable by retry
- **Identical error on retry**: if attempt 2 produces the same stderr (first 200 chars match), skip attempt 3

### Retryable (auto-repair up to 3 attempts):
1. **Attempt 1 fails:** capture stderr, classify error
2. **Diagnose and fix:**
   - `FileNotFoundError` on checkpoint/data → verify paths, check worktree setup
   - `RuntimeError: NCCL` / distributed error → retry with `CUDA_VISIBLE_DEVICES` single GPU
   - `ImportError` / `ModuleNotFoundError` → install missing package
   - `ValueError` / `TypeError` in config → check config override syntax (string vs number)
   - `PermissionError` → fix file permissions
   - `ConnectionError` / `HTTPError` → transient network error, wait 5s, retry
3. **Log each retry:** `category: "training_failure", severity: "warning", source: "experiment", context: {"original_error": "<error>", "fix": "<description>", "attempt": <N>}`
4. **Attempt 2** with fix → if it fails with a new error, apply a new fix → **Attempt 3**
5. **All 3 fail:** write `status: "failed"`, include all error history in `notes`

Retry time counts toward `duration_seconds` (single experiment, not separate entries).

### Specific error handling:

- **Training crashes:**
  - Capture the error output
  - Apply the auto-repair loop (above) for retryable errors
  - Write results with `"status": "failed"` and the error message in notes
  - Log to error tracker:
    ```bash
    python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"training_failure","severity":"critical","source":"experiment","message":"<error description>","exp_id":"<exp_id>","config":<config_json>,"stack_trace":"<last 20 lines of stderr>"}'
    ```

- **Divergence detected (by monitor):**
  - Training is killed by the monitor skill
  - Write results with `"status": "diverged"` and divergence details

- **GPU out of memory:**
  - Common cause: batch size too large
  - Write results with `"status": "failed"` and note the OOM error
  - The hp-tune skill will adjust batch size in the next iteration
  - Log to error tracker:
    ```bash
    python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"training_failure","severity":"critical","source":"experiment","message":"GPU OOM with batch_size=<batch_size>","exp_id":"<exp_id>","config":<config_json>,"context":{"error_type":"oom","batch_size":<batch_size>}}'
    ```

- **Config override not working:**
  - If CLI args don't override correctly, try the config file approach
  - If neither works, report the issue back to orchestrator
  - Log to error tracker:
    ```bash
    python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"config_error","severity":"warning","source":"experiment","message":"Config override failed: <method tried>","exp_id":"<exp_id>"}'
    ```

- **Training timeout:**
  - The `timeout` command kills the process with SIGTERM (then SIGKILL after 60s)
  - Parse any partial results from the log before the timeout
  - Write results with `"status": "timeout"` and note the timeout duration
  - Log to error tracker with `category: "timeout"`, `severity: "warning"`, `source: "experiment"`
  - The monitor skill will also detect the process death and mark accordingly
