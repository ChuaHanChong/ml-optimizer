---
name: baseline
description: "Establish baseline metrics for an ML model. Runs evaluation, profiles GPU memory and training throughput, and creates the experiments directory structure. Use when: need to measure current model performance before optimization."
---

# Baseline Evaluation

Establish baseline performance metrics for the ML model. This is always the first step in optimization.

## Inputs Expected

The orchestrator provides:
- Project root path
- Model/training details (from understanding phase)

## Step 1: Identify Evaluation Command

Search the project for evaluation scripts:

1. Use Glob to find candidates:
   - `**/eval*.py`, `**/test*.py`, `**/infer*.py`, `**/validate*.py`
   - `**/scripts/eval*`, `**/scripts/test*`
   - `Makefile`, `**/*.sh` (look for eval targets)

2. Read the training script to find validation/evaluation logic:
   - Look for functions named `evaluate`, `validate`, `test`, `infer`
   - Look for metric computation (PSNR, SSIM, loss, accuracy, F1, etc.)

3. If no clear eval command found, use AskUserQuestion:
   ```
   I couldn't automatically identify an evaluation command.
   How do I evaluate this model? Please provide:
   - The command to run evaluation
   - What metrics it outputs
   ```

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

## Step 3: Run Baseline Evaluation

1. Execute the evaluation command via Bash
2. Capture all output
3. Parse output for metrics using the log parser:
   ```bash
   python3 ~/.claude/plugins/ml-optimizer/scripts/parse_logs.py <output_file>
   ```
4. **Validate parse results:** Check that `parse_logs` returned non-empty records. If empty, the log format may be unrecognized — try forcing different formats (`--format kv`, `--format json`, `--format logging`, `--format tqdm`)
5. If metrics aren't parseable automatically, read the output and extract them manually

## Step 4: Profile Training

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

## Error Handling

- **Eval command fails:** Report the error output, ask user for correct command
- **No GPU available:** Run CPU-only baseline, note that throughput estimates won't be representative
- **Metrics not parseable:** Show raw output, manually extract key numbers, note which metrics were found
