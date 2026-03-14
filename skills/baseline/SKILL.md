---
name: baseline
description: "Establish baseline metrics for an ML model. Runs evaluation, profiles GPU memory and training throughput, and creates the experiments directory structure. Use when: need to measure current model performance before optimization."
disable-model-invocation: true
user-invocable: false
---

# Baseline Evaluation

Establish baseline performance metrics for the ML model. This is always the first step in optimization.

## Inputs Expected

The orchestrator provides:
- Project root path
- Model/training details (from understanding phase)
- `prepared_train_path` (optional): If prerequisites prepared data, use this path instead of the original data path in the training command
- `prepared_val_path` (optional): Same for validation data
- `model_category` (optional): From user_choices — `"supervised"`, `"rl"`, `"generative"`, or null. Controls RL-specific evaluation (see RL Baseline Evaluation section) and tabular ML GPU profiling skip.

## Step 1: Identify Evaluation Command

Search the project for evaluation scripts:

1. Use Glob to find candidates:
   - `**/eval*.py`, `**/test*.py`, `**/infer*.py`, `**/validate*.py`
   - `**/scripts/eval*`, `**/scripts/test*`
   - `Makefile`, `**/*.sh` (look for eval targets)

2. Read the training script to find validation/evaluation logic:
   - Look for functions named `evaluate`, `validate`, `test`, `infer`
   - Look for metric computation (PSNR, SSIM, loss, accuracy, F1, etc.)
   - **Lightning projects:** Look for `validation_step()` / `test_step()` methods and `self.log()` calls. Metrics may be in TensorBoard logs (`lightning_logs/`) — parse with `parse_logs.py`
   - **HuggingFace Trainer:** Look for `compute_metrics` function passed to `Trainer`. Metrics are logged to `runs/` or `output_dir`. Use `trainer.evaluate()` as eval command
   - **TF/Keras:** Look for `model.evaluate()` or custom callbacks. Metrics may be in `CSVLogger` output or TensorBoard

3. If no clear eval command found:

   **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, skip the user question. Instead:
   - Set `eval_command = null`
   - Use training output as the evaluation source — run training for the profiling duration and extract final metrics via `parse_logs.py`
   - If metrics containing the `primary_metric` keyword are found in training output: use those as baseline metrics
   - If no recognizable metrics found: look for checkpoint/log files (TensorBoard events, CSV logs, JSON summaries) and parse those
   - Log to dev_notes: "No eval command found — using training output metrics as baseline (autonomous mode)"
   - Log to error tracker: `category: "config_error", severity: "info", source: "baseline", message: "No eval command — falling back to training output metrics (autonomous mode)"`

   **Otherwise (interactive mode):** Use AskUserQuestion:
   ```
   I couldn't automatically identify an evaluation command.
   How do I evaluate this model? Please provide:
   - The command to run evaluation
   - What metrics it outputs
   ```

## Step 1.1: Apply Prepared Data Paths (If Applicable)

If the orchestrator passed `prepared_train_path` or `prepared_val_path`:
1. Identify how the training command references data paths (CLI args like `--data_dir`, `--train_path`, `--val_path`, or config file entries)
2. Substitute the prepared paths into the training/eval commands before running them
3. If data paths are in a config file, create a modified copy at `experiments/logs/baseline/config.yaml` with updated paths
4. Log in dev_notes which paths were substituted

If no prepared paths were provided, use the original commands as-is.

## Step 2: Set Up Experiment Directory

Run the experiment setup script:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/experiment_setup.py <project_root> "<train_command>"
```

This creates:
```
<project>/experiments/
  logs/
  reports/
  scripts/
  results/
  dev_notes.md
```

## Step 2.1: Auto-Repair Loop (for eval/train commands)

When executing evaluation or training commands (Steps 3 and 4), apply this retry pattern:

1. **Attempt 1:** Run the command normally
2. **If the command fails (non-zero exit code):**
   - Capture stderr and the last 50 lines of stdout
   - Log to error tracker: `category: "training_failure", severity: "warning", source: "baseline", message: "Command failed (attempt 1/3): <error_summary>", phase: 3, context: {"command": "<command>", "attempt": 1}`
   - **Diagnose the error:**
     - `ModuleNotFoundError` / `ImportError` → install the missing package, retry
     - `FileNotFoundError` → check if the path exists, fix path references, retry
     - `CUDA out of memory` → reduce batch size by 50% in the command, retry
     - `RuntimeError: CUDA` → try `CUDA_VISIBLE_DEVICES=0`, retry
     - `PermissionError` → fix file permissions, retry
     - `SyntaxError` / `IndentationError` → **do NOT retry** (code bug, report to orchestrator)
     - `KeyboardInterrupt` / `SIGTERM` → **do NOT retry** (intentional stop)
     - Other errors → read relevant source code, propose a fix, retry
   - **Attempt 2:** Re-run with fix applied
3. **If attempt 2 also fails:**
   - Log attempt 2 failure (same pattern, attempt: 2)
   - Try a different approach (different package version, broader file search, further batch reduction)
   - **Attempt 3:** Re-run with new fix
4. **If attempt 3 fails:**
   - Log with `severity: "critical"`, attempt: 3
   - Give up. Report full error history (all 3 attempts) to the orchestrator
   - The orchestrator's Phase 3 Failure Recovery table handles escalation

**Loop detection:** If attempt 2 produces the same error message (first 200 chars) as attempt 1, skip attempt 3.

## Step 3: Run Baseline Evaluation

1. Execute the evaluation command via Bash (within the auto-repair loop above)
2. Capture all output
3. Parse output for metrics using the log parser:
   ```bash
   python3 ~/.claude/plugins/ml-optimizer/scripts/parse_logs.py <output_file>
   ```
4. **Validate parse results:** Check that `parse_logs` returned non-empty records. If empty, the log format may be unrecognized — try forcing different formats (`--format kv`, `--format json`, `--format logging`, `--format tqdm`)
5. If metrics aren't parseable automatically, read the output and extract them manually

## Step 4: Profile Training

### Framework-Specific Profiling

**For non-iterative frameworks (scikit-learn, XGBoost without GPU, LightGBM without GPU):**
- Skip GPU memory profiling and throughput estimation below.
- Instead, measure total `fit()` wall-clock time and record as `profiling.fit_duration_seconds` in baseline.json.
- **Estimate experiment timeout from fit duration:** Read the model's configured iteration count (e.g., `n_estimators`, `max_iter`, `num_boost_round`) and the profiling iteration count. Compute: `estimated_timeout_seconds = fit_duration_seconds * (max_iterations_configured / profiling_iterations) * 2`. The `× 2` safety margin accounts for slower HP configs. If iteration counts cannot be determined, fall back to `fit_duration_seconds * 10`. Cap at 14400 (4 hours). Record as `profiling.estimated_timeout_seconds` in baseline.json.
- Set `profiling.throughput_samples_per_sec` and `profiling.estimated_max_batch_size` to `null`.
- If the framework supports GPU (XGBoost `tree_method="gpu_hist"`, LightGBM `device="gpu"`), still run `gpu_check.py`.

**For iterative frameworks (PyTorch, TensorFlow, JAX, Lightning, HuggingFace Trainer):**
- Proceed with the profiling steps below.

Run a short training session to measure GPU resource usage:

1. **GPU memory profiling:**
   ```bash
   # Start a short training run (1-2 epochs or ~100 steps)
   # Then check GPU memory:
   python3 ~/.claude/plugins/ml-optimizer/scripts/gpu_check.py
   ```

2. **Estimate throughput:**
   - Parse the training log for step timing
   - Calculate samples/second or steps/second

3. **Determine batch size limits:**
   - Current batch size and memory usage
   - Estimate max batch size based on available GPU memory
   - Note: this is a rough estimate; actual limits depend on model architecture

## Step 5: Write Baseline Results

Write `experiments/results/baseline.json`:
```json
{
  "exp_id": "baseline",
  "status": "completed",
  "config": {
    "lr": <current_lr>,
    "batch_size": <current_batch_size>,
    ...all current training params...
  },
  "metrics": {
    "<primary_metric>": <value>,
    ...all measured metrics (accuracy, f1, psnr, bleu, etc.)...
  },
  "code_branch": null,
  "code_proposal": null,
  "profiling": {
    "gpu_memory_used_mib": <value>,
    "gpu_memory_total_mib": <value>,
    "throughput_samples_per_sec": <value>,
    "estimated_max_batch_size": <value>
  },
  "eval_command": "<command used>",
  "train_command": "<command used>",
  "notes": "Baseline evaluation - current model state"
}
```

Use the Write tool to create this file.

**Nullable fields:** For non-iterative frameworks (scikit-learn, XGBoost, LightGBM), `profiling.throughput_samples_per_sec` and `profiling.estimated_max_batch_size` will be `null`. HP-tune must not use `estimated_max_batch_size` as a cap when null. Experiment must use `estimated_timeout_seconds` instead of throughput-based timeout.

## Step 5.1: Validate Output

```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/schema_validator.py \
  experiments/results/baseline.json baseline
```

If validation fails, fix and re-validate before proceeding.

## Step 5.2: Validate Metric Keys

After schema validation passes, verify the `metrics` dict contains required keys:

1. **Check primary_metric:** If the orchestrator specified `primary_metric`, verify `metrics` contains a matching key (case-insensitive). If not found, search for close matches (e.g., `"val_accuracy"` for `"accuracy"`). If a close match exists, log to dev_notes which key was used. If no match: log warning to error tracker with `category: "config_error"`.

2. **Check divergence metric:** If the orchestrator specified `divergence_metric` (default: `"loss"`), verify it exists in `metrics`. If not found, check aliases: `"train_loss"`, `"val_loss"`, `"total_loss"`, `"nll_loss"`. If found under an alias, log which alias to dev_notes — the orchestrator should pass this alias to the monitor skill. If not found at all: log warning — divergence monitoring may not work.

## Step 6: Write Dev Notes

Append to `experiments/dev_notes.md`:
```markdown
## <date> — Baseline

- **Goal:** Establish baseline metrics for optimization
- [metric 1]: [value]
- [metric 2]: [value]
- GPU Memory: [used]/[total] MiB
- Throughput: [X] samples/sec
- Max batch size estimate: [Y]
- **Next:** Awaiting user direction on optimization approach
```

## Output

Return to the orchestrator:
- Path to `baseline.json`
- Summary of baseline metrics
- GPU profiling results
- Any issues encountered

## RL Baseline Evaluation

When `model_category = "rl"`:

1. **Evaluation method:** Run N evaluation episodes (default: 100) with the current policy. Compute mean, std, min, max episode reward.
2. **Profiling:** Measure steps/second and episodes/hour. Set `throughput_samples_per_sec = null`. Record `steps_per_second` and `episodes_per_hour` in profiling.
3. **Timeout estimation:** Use `total_timesteps / steps_per_second` instead of epoch-based estimation.
4. **Config extraction:** Capture RL-specific HPs: `gamma`, `learning_rate`, `n_steps`/`buffer_size`, `batch_size`, `entropy_coef`, `clip_range` (PPO), `tau` (SAC/TD3).

## Error Handling

- **Eval command fails:** Report the error output, ask user for correct command
- **No GPU available:** Run CPU-only baseline, note that throughput estimates won't be representative
- **Metrics not parseable:** Show raw output, manually extract key numbers, note which metrics were found

## Error Tracking

At the following points, log an error event using the error tracker:

### When evaluation command fails:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"training_failure","severity":"critical","source":"baseline","message":"Baseline eval command failed: <error>","phase":3,"context":{"command":"<eval_command>","exit_code":<code>}}'
```

### When metrics are not parseable from output:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"config_error","severity":"warning","source":"baseline","message":"Could not parse metrics from baseline output","phase":3,"context":{"output_preview":"<first 200 chars>"}}'
```

### When GPU profiling fails:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"resource_error","severity":"info","source":"baseline","message":"GPU profiling failed — no GPU detected or nvidia-smi error","phase":3}'
```
