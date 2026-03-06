---
name: monitor
description: "Monitor running ML training experiments for divergence. Polls log files, detects NaN/explosion/plateau, and kills diverging processes. Use when: experiments are running and need to be watched for training issues."
---

# Experiment Monitor

Watches running training experiments for signs of divergence and takes corrective action.

## Inputs Expected

From the orchestrator:
- `log_files`: List of log file paths to monitor (one per running experiment)
- `exp_ids`: Corresponding experiment IDs
- `project_root`: Project root directory
- `poll_interval`: How often to check (default: 30 seconds)
- `metric_to_watch`: Which metric to monitor for divergence (default: "loss"). The orchestrator passes the user's chosen divergence metric from Phase 0. Common values: `"loss"`, `"train_loss"`, `"val_loss"`, `"objective"`, `"nll_loss"`, `"total_loss"`.
- `lower_is_better`: Whether lower values are better for the watched metric (default: true). The `primary_metric` (accuracy, PSNR, etc.) is used by analyze/hp-tune, not by monitor.
- `explosion_threshold`: Threshold multiplier for explosion detection (default: 5.0). Override based on model type.
- `plateau_patience`: Steps without improvement before plateau alarm (default: 20). Override based on model type.

## Step 1: Validate Inputs

1. Verify each log file path exists (or its directory exists — file may not be created yet)
2. Verify corresponding experiment scripts are running:
   ```bash
   # Check if training process is still running
   ps aux | grep "<exp_id>" | grep -v grep
   ```
   Or check for PID files in `experiments/logs/<exp_id>/pid`

## Step 2: Poll Loop

For each monitoring cycle:

### 2a: Read Latest Log Content

For each log file:
```bash
# Read the last N lines of the log file
tail -100 experiments/logs/<exp_id>/train.log
```

### 2b: Parse Metrics

Parse the log content for the watched metric:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/parse_logs.py experiments/logs/<exp_id>/train.log
```

Extract the metric trajectory (all values of the watched metric over time).

### 2b.1: Metric Name Fallback

If the watched metric is not found in the parsed records, attempt auto-detection:

1. **Case-insensitive match:** Try `metric_to_watch.lower()` against all keys lowercased
2. **Prefix variants:** Try `train_<metric>`, `val_<metric>`, `<metric>_train` (e.g., `loss` → `train_loss`, `val_loss`)
3. **Substring match:** Look for any key containing `metric_to_watch` as a substring (e.g., `"loss"` matches `"total_loss"`)
4. **Report missing:** If no match found after all fallbacks, log a warning and report the available metric names to the orchestrator. Do not treat this as divergence.

### 2c: Check for Divergence

Run divergence detection on the extracted trajectory, passing `lower_is_better`:
```bash
python3 -c "
import json, sys
sys.path.insert(0, '$HOME/.claude/plugins/ml-optimizer/scripts')
from detect_divergence import check_divergence
from parse_logs import parse_log, extract_metric_trajectory

records = parse_log('experiments/logs/<exp_id>/train.log')
values = extract_metric_trajectory(records, '<metric>')
result = check_divergence(values, lower_is_better=<lower_is_better>)
print(json.dumps(result))
"
```

### 2d: Take Action on Divergence

If divergence is detected:

1. **Kill the training process:**
   ```bash
   # Option 1: PID file (preferred — most reliable)
   kill $(cat experiments/logs/<exp_id>/pid) 2>/dev/null

   # Option 2: Safe pattern match — verify process is a training process before killing
   # First find candidates, then verify cmdline contains python/train before killing
   for pid in $(pgrep -f "<exp_id>"); do
     cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ')
     if echo "$cmdline" | grep -qE 'python|train'; then
       kill "$pid"
     fi
   done
   ```
   **Warning:** Never use bare `pkill -f "<exp_id>"` — it could match unrelated processes.

2. **Record the divergence:**
   - Read the current experiment result file
   - Update status to `"diverged"`
   - Add divergence details to notes:
     ```json
     {
       "status": "diverged",
       "notes": "Divergence detected: <reason> at step <step>"
     }
     ```
   - Write the updated result using the Write tool

3. **Log the event:**
   Append to `experiments/dev_notes.md`:
   ```
   ## Divergence Detected
   - Experiment: <exp_id>
   - Reason: <reason>
   - Step: <step>
   - Action: Training process killed
   ```

### 2e: Report Status

For healthy experiments, report status:
```
Monitoring status:
- exp-001: healthy (loss=0.45 at step 500, trending down)
- exp-002: DIVERGED (NaN at step 350) - process killed
- exp-003: healthy (loss=0.52 at step 480, trending down)
```

## Step 3: Completion

The monitor exits when:
1. All experiments have completed (log files stop being updated)
2. All experiments have been killed (divergence)
3. The orchestrator signals to stop

## Monitoring Heuristics

### Check frequency
- First 100 steps: check every 10 seconds (early divergence is common)
- Steps 100-1000: check every 30 seconds
- After step 1000: check every 60 seconds

### Divergence parameters (defaults, adjust based on model type)
- **NaN/Inf detection:** Always enabled, immediate kill
- **Explosion threshold:** 5x rolling average over 10-step window
- **Plateau patience:** 20 evaluation checkpoints with min_delta=1e-6

See also `hp-tune/references/tuning-strategy.md` for per-model-type HP guidance that informs threshold selection.

### Recommended thresholds by model type
| Model Type | Explosion Threshold | Plateau Patience | Notes |
|------------|-------------------|-----------------|-------|
| CNN (classification) | 5.0 | 20 | Standard defaults |
| Transformer | 10.0 | 30 | Loss can be spikier |
| GAN | 20.0 | 50 | Inherently noisy training |
| Diffusion model | 10.0 | 40 | Slow convergence is normal |
| Fine-tuning | 3.0 | 15 | Should converge faster |

### Common divergence patterns
- **NaN in first 10 steps:** Learning rate too high. Note this for hp-tune.
- **Loss explosion after good start:** Possible learning rate schedule issue or gradient accumulation bug.
- **Slow plateau:** Model capacity may be insufficient, or learning rate too low.

## Error Handling

- **Log file doesn't exist yet:** Wait up to 60 seconds for it to appear, then report error
- **Log file format unrecognized:** Try all parsers, report if none work
- **Process already dead:** Check exit code, mark as failed if non-zero
- **Permission errors:** Report and skip that experiment

## Output

Return to the orchestrator a dict per experiment:
- `exp_id`: Experiment identifier
- `status`: One of `healthy`, `diverged`, `completed`, `failed`
- `reason`: Divergence reason (if diverged) or `null`
- `step`: Step at which divergence was detected (if diverged) or `-1`
- `latest_metrics`: Dict of most recent metric values
- `metric_trajectory`: List of watched metric values over time
