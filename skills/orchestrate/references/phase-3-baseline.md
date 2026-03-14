# Phase 3: Establish Baseline

Dispatch the baseline agent:
```
Agent(
  description: "Establish baseline metrics",
  prompt: "Establish baseline metrics. Parameters: project_root: {project_root}, train_command: {train_command}, eval_command: {eval_command}, model_category: {model_category}, prepared_train_path: {prepared_train_path or null}, prepared_val_path: {prepared_val_path or null}.",
  subagent_type: "ml-optimizer:baseline-agent"
)
```
- Wait for baseline results
- Store in `experiments/results/baseline.json`

## Phase 3 Failure Recovery

If baseline fails, diagnose from the error message in baseline.json or error tracker log:

| Error Pattern | Action |
|---------------|--------|
| `FileNotFoundError` / data path invalid | Re-run Phase 2 (prerequisites) to validate paths |
| `ModuleNotFoundError` / missing package | Re-run Phase 2 to install dependencies |
| `CUDA out of memory` / OOM | Reduce batch size to 50% of current, retry baseline |
| `RuntimeError: NCCL` / distributed error | Try single-GPU: set `CUDA_VISIBLE_DEVICES=0` |
| Training script timed out / no output for >30 min | Reduce epochs/steps to minimum for baseline profiling |
| `SyntaxError` / `IndentationError` | Code issue in user's project — report to user, cannot proceed |
| Unknown error | Show full error via AskUserQuestion, ask for guidance |

**Retry logic:** Attempt up to 2 retries with adjustments from the table above. Log each retry to the error tracker.

**Autonomous mode Phase 3 unknown error:** If `budget_mode == "autonomous"` and baseline fails with an unknown error after 2 retries: log the full error to error tracker with `category: "agent_failure", severity: "critical"`, log to dev_notes: "Baseline failed after 2 retries with unknown error — exiting with partial results (autonomous mode)". Exit the pipeline and proceed to Phase 9 (report) with whatever partial results exist. Do NOT use AskUserQuestion.

**Skip-baseline fallback:** If all retries fail, offer to create a synthetic baseline.json with user-provided metric values. Mark profiling fields as `null`. This allows the experiment loop to proceed without throughput-based timeout estimation.

**Autonomous mode:** If `budget_mode == "autonomous"`, do NOT create a synthetic baseline (no user to provide metric values). Instead, exit the pipeline and proceed to Phase 9 (report) with partial results. Log to error tracker: `category: "agent_failure", severity: "critical", source: "orchestrate", message: "Baseline failed — cannot create synthetic baseline in autonomous mode"`.
