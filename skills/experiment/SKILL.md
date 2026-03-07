---
name: experiment
description: "Run a single ML training experiment. Generates bash scripts, executes training on a specified GPU, and parses results. Use when: need to run a training experiment with a specific configuration."
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

## Step 1.5: Pre-Flight Checks

Before building the training command, verify:

1. **Disk space:** Check that the target filesystem has sufficient free space for logs and checkpoints:
   ```bash
   df -h <project_root> | tail -1
   ```
   Warn if less than 5 GB free.

2. **Timeout estimation:** If baseline profiling data is available, estimate total training time:
   - Read `experiments/results/baseline.json` → `profiling.throughput_samples_per_sec`
   - Estimate: `total_time = (dataset_size * epochs) / throughput`
   - If estimated time exceeds 4 hours, warn the orchestrator

## Step 2: Build Training Command

Construct the full training command by overriding the base command with experiment-specific config:

1. Read the base training command from `experiments/results/baseline.json`
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

The script must:
- Set `CUDA_VISIBLE_DEVICES=<gpu_id>`
- Create the log directory
- Run training with output logged to `experiments/logs/<exp_id>/train.log`
- Include any environment variables needed

Save to: `experiments/scripts/<exp_id>.sh`

## Step 4: Execute Training

Run the experiment:
```bash
bash experiments/scripts/<exp_id>.sh
```

**For foreground execution** (when called directly):
- Run via Bash tool and wait for completion
- Monitor output for early signs of problems

**For background execution** (when called by orchestrator in parallel):
- Run via Bash tool with `run_in_background: true`
- The monitor skill will handle divergence detection

## Step 5: Parse Results

After training completes:

1. Parse the training log:
   ```bash
   python3 ~/.claude/plugins/ml-optimizer/scripts/parse_logs.py experiments/logs/<exp_id>/train.log
   ```

2. If an eval command exists, run evaluation:
   ```bash
   <eval_command>
   ```
   Parse eval output for final metrics.

3. Extract key metrics:
   - Final loss value
   - Best metric value (PSNR, accuracy, etc.)
   - Training duration
   - Any other relevant metrics

## Step 6: Write Results

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
  "script_file": "experiments/scripts/<exp_id>.sh",
  "code_branch": "<branch name or null>",
  "code_proposal": "<proposal name or null>",
  "notes": "<any observations>"
}
```

## Step 7: Report Back

Return to the orchestrator:
- Experiment ID
- Status (completed/failed/diverged)
- Key metrics
- Path to results file
- Any issues encountered

## Error Handling

- **Training crashes:**
  - Capture the error output
  - Write results with `"status": "failed"` and error message in notes
  - Do NOT retry automatically — let the orchestrator decide
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
