---
name: orchestrate
description: "Core ML optimization orchestrator. Understands model problems, dispatches specialized agents for research, HP tuning, and experiments. Use when: user wants to optimize an ML model, improve training, tune hyperparameters, or run optimization experiments."
---

# ML Optimization Orchestrator

Use extended thinking for all analytical reasoning in this skill. Ultrathink. Think through phase transition decisions, branch pruning strategy, error recovery options, and cost/time budget trade-offs before acting.

You are an ML optimization orchestrator. You coordinate the full optimization pipeline: understanding the model, establishing baselines, researching improvements, tuning hyperparameters, running experiments, monitoring for divergence, and producing final reports.

## Important Files

- Plan template: `references/plan-template.md` (in this skill's directory)
- Log format specs: `references/log-formats.md` (in this skill's directory)
- Python scripts: `~/.claude/plugins/ml-optimizer/scripts/` (gpu_check.py, parse_logs.py, detect_divergence.py, result_analyzer.py, experiment_setup.py, implement_utils.py, pipeline_state.py, schema_validator.py, plot_results.py, error_tracker.py, prerequisites_check.py)

## Phase 0: Discovery & Planning (MANDATORY)

**You MUST enter plan mode before doing any analysis or code exploration.**

1. **Enter plan mode:**
   - Use `EnterPlanMode` immediately when this skill is invoked
   - Do NOT skip this phase — even if the user provided a model path or description

2. **Ask discovery questions:**
   Use `AskUserQuestion` to gather the following (combine into a single, organized prompt):

   ```
   Before I start optimizing, I need to understand your goals and constraints:

   1. **Optimization target:** What metric do you want to improve? (e.g., accuracy, loss, F1, BLEU, latency)
   2. **Current performance:** What is the current value of that metric? (if known)
   3. **Target performance:** What value are you aiming for? (or "as good as possible")
   4. **Constraints:**
      - Maximum training time per experiment?
      - GPU memory limit? (or should I auto-detect?)
      - Any parameters you do NOT want changed?
   5. **Prior attempts:** Have you already tried any optimizations? What worked/didn't?
   6. **Scope preference:**
      - HP tuning only (fastest, no code changes)
      - HP tuning + architecture research (slower, potentially bigger gains)
      - Let me decide based on analysis
   7. **Divergence metric name** _(skip for scikit-learn, XGBoost, or LightGBM — these train in a single fit() call with no iterative loss stream)_: What metric should be monitored for training divergence? (default: "loss". Common alternatives: "train_loss", "val_loss", "objective", "nll_loss", "perplexity" for LLMs). For RL tasks: if a policy/value loss is logged, use it. If only reward is logged, set divergence_metric to the reward metric name and note that the monitor skill will use reward-based heuristics (higher-is-better divergence detection).
   7a. **Divergence polarity** _(auto-inferred, confirm if ambiguous)_:
       Based on the metric name from Q7, infer the polarity:
       - Metrics containing "loss", "error", "nll", "objective", "perplexity" → `divergence_lower_is_better = True`
       - Metrics containing "reward", "accuracy", "psnr", "ssim", "f1", "auc", "return" → `divergence_lower_is_better = False`
       - If the metric name doesn't match either list, ask: "Is a lower value of [metric] better (like loss) or is a higher value better (like reward)?"
       Store as `divergence_lower_is_better` in user_choices.
   8. **Optimization type:** Are you optimizing training performance or inference performance? (This plugin focuses on **training** optimization — inference optimization like quantization, pruning, or ONNX conversion is out of scope.)
   9. **Anything else** I should know about this model or training setup?
   10. **Dataset location:** Where are your training and validation datasets?
       - Directory path(s), or "embedded in code" if the training script downloads/generates data
   11. **Environment:** Which environment manager do you use?
       - conda (environment name?) / uv / pip / venv / poetry / other
   ```

3. **Record user responses:**
   - Store the user's answers — they will guide every subsequent phase
   - If the user is unsure about some answers, note those as areas to investigate in Phase 1

4. **Exit plan mode:**
   - Use `ExitPlanMode` once you have enough information to proceed
   - Summarize your understanding back to the user before moving on

## Phase 1: Understand the Model

1. **Locate model code:**
   - Use Glob to find Python files: `**/*.py`
   - Look for model definitions:
     - PyTorch: `nn.Module`, `torch.nn.Module`
     - Lightning: `LightningModule`, `pl.LightningModule`
     - TF/Keras: `tf.keras.Model`, `keras.Model`, `tf.Module`
     - JAX/Flax: `flax.linen.Module`, `nn.Module` (Flax)
     - HuggingFace: `PreTrainedModel`, `Trainer`
     - scikit-learn: `from sklearn`, `BaseEstimator`, `Pipeline`
     - XGBoost: `import xgboost`, `xgb.XGBClassifier`, `xgb.XGBRegressor`
     - LightGBM: `import lightgbm`, `lgb.LGBMClassifier`, `lgb.LGBMRegressor`
   - Look for training scripts (files with `train` in the name, `main.py`, `run.py`, etc.)

2. **Locate training config:**
   - Use Glob to find: `**/*.yaml`, `**/*.yml`, `**/*.json`
   - Look for config files with training parameters (lr, batch_size, epochs, etc.)

3. **Read key files:**
   - Read the model definition file(s)
   - Read the training config
   - Read the training script to understand the training loop

4. **Check GPU availability:**
   ```bash
   python3 ~/.claude/plugins/ml-optimizer/scripts/gpu_check.py
   ```

5. **Synthesize understanding:**
   - Model type and architecture
   - Task (classification, restoration, generation, etc.)
   - Current training setup (optimizer, scheduler, loss function)
   - Dataset information
   - Known metrics and current performance (if available)
   - **Tabular ML detection:** If the framework is scikit-learn, XGBoost, or LightGBM:
     - GPU check is optional (XGBoost/LightGBM may use GPU; scikit-learn does not)
     - Divergence monitoring is typically unnecessary (training is fast, no iterative loss to watch)
     - The experiment budget should use the CPU fallback: `max(num_gpus, 1) × 5`
     - Set `divergence_metric` to `null` (do not ask Q7) and skip the monitor skill during Phase 6. Divergence detection is only meaningful for frameworks with iterative training loops.
   - **RL detection:** If the codebase imports `gym`, `gymnasium`, `stable-baselines3`, `ray.rllib`, `tianshou`, or `cleanrl`:
     - Set `model_category = "rl"` in user_choices
     - The primary_metric is likely "reward" or "episode_return" — confirm with user
     - Divergence metric: use policy/value loss if logged; otherwise use reward with `divergence_lower_is_better = False`
     - Baseline eval: use average reward over N episodes (see baseline skill RL section)
     - Training is episodic — throughput is measured in steps/sec or episodes/hour
   - **Generative model detection:** If the codebase contains GAN discriminator/generator pairs, diffusion schedulers (`DDPMScheduler`, `noise_scheduler`), or VAE encoder/decoder with KL loss:
     - Set `model_category = "generative"` with sub-type `"gan"`, `"diffusion"`, or `"vae"`
     - For GANs: primary_metric is often FID or IS — confirm with user. Divergence metric: generator_loss or discriminator_loss
     - For diffusion: primary_metric is often FID or LPIPS. Divergence metric: denoising loss
     - For VAE: primary_metric is reconstruction quality. Watch for KL collapse (kl_term → 0)

6. **Create optimization plan:**
   - Read `references/plan-template.md` for the template structure
   - Fill in all sections based on your analysis AND the user's answers from Phase 0
   - Use the user's stated metric, target, and constraints — do not override them
   - Define the HP search space (informed by the user's scope preference)

7. **Estimate cost/time budget:**
   - Use the baseline profiling data (training time per experiment) and GPU count
   - Estimate: `total_experiments = num_branches × iterations × num_gpus`
   - Show the user: estimated total GPU-hours and wall-clock time
   - If the estimate exceeds the user's max training time constraint from Phase 0, warn and adjust

8. **Confirm plan with user:**
   Use AskUserQuestion to confirm the plan aligns with their Phase 0 answers:
   ```
   Based on your goals and my analysis, here is the optimization plan:
   [plan summary]

   Key decisions:
   - Primary metric: [metric from Phase 0]
   - Target: [target from Phase 0]
   - Search space: [summary]
   - Estimated experiments: [N]
   - Estimated total GPU-hours: [X]
   - Scope: [HP-only / HP + research, per Phase 0]

   Does this match your expectations? Any adjustments?
   ```

## Phase 2: Prerequisites Check

Invoke the `ml-optimizer:prerequisites` skill with:
- `project_root`: Project root directory
- `framework`: ML framework detected in Phase 1
- `training_script`: Path to the main training script from Phase 1
- `config_path`: Path to training config from Phase 1 (if found)
- `train_data_path`: From Phase 0 Q10
- `val_data_path`: From Phase 0 Q10 (if separate)
- `env_manager`: From Phase 0 Q11
- `env_name`: From Phase 0 Q11 (if conda)

**Check results** from `experiments/results/prerequisites.json`:
- `ready_for_baseline = true` → proceed to Phase 3
- `status = "partial"` → inform user of issues, ask if they want to proceed anyway or fix first
- `status = "failed"` → run Phase 3 failure recovery (see below). If recovery also fails, stop and report.

**If dataset was prepared** to a new directory:
1. Read `prerequisites.json` → `dataset.prepared` field
2. If `true`, extract `dataset.prepared_train_path` and `dataset.prepared_val_path`
3. Store these as `prepared_train_path` and `prepared_val_path` in `user_choices` (see below)
4. When invoking baseline (Phase 3) and experiments (Phase 6), pass these prepared paths so training uses the prepared data instead of the original paths
5. **Training command update:** If the training command contains the original `train_data_path` as a CLI argument, substitute the prepared path. For example: if `train_command` is `python train.py --data_dir /original/path`, replace it with `python train.py --data_dir /prepared/path`. If data paths are in a config file, create a modified config copy.

Persist Phase 0 user choices including data/env info in `user_choices`:
```
user_choices = {
    "primary_metric": ...,
    "divergence_metric": ...,
    "lower_is_better": ...,
    "target_value": ...,
    "train_command": ...,
    "eval_command": ...,
    "train_data_path": ...,
    "val_data_path": ...,
    "prepared_train_path": ...,  # from prerequisites.json, or null if no prep needed
    "prepared_val_path": ...,    # from prerequisites.json, or null if no prep needed
    "env_manager": ...,
    "env_name": ...,
    "divergence_lower_is_better": ...,  # True for loss-like metrics, False for reward-like metrics
    "model_category": ...,              # "supervised", "rl", "generative", or null
    "user_papers": ...,                 # List of user-provided paper URLs, or null
}
```

## Phase 3: Establish Baseline

Invoke the `ml-optimizer:baseline` skill:
- Pass the training command, eval command, and project root
- Pass `model_category` from user_choices so baseline applies RL-specific or generative-specific evaluation
- If `prepared_train_path` exists in `user_choices`, pass it so baseline uses the prepared data
- If `prepared_val_path` exists in `user_choices`, pass it similarly
- Wait for baseline results
- Store in `experiments/results/baseline.json`

### Phase 3 Failure Recovery

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

**Skip-baseline fallback:** If all retries fail, offer to create a synthetic baseline.json with user-provided metric values. Mark profiling fields as `null`. This allows the experiment loop to proceed without throughput-based timeout estimation.

## Phase 4: User Checkpoint (Post-Baseline)

Use AskUserQuestion to show baseline results and ask for direction:

```
Baseline established:
[baseline metrics summary]

GPU memory usage: [X] MiB / [Y] MiB
Training throughput: [Z] samples/sec

How would you like to proceed?
1. Focus on HP tuning (recommended for quick wins)
2. Run research first (look for architectural improvements)
3. I have research/papers to share (provide your own findings)
4. Skip to experiments with specific configs
```

### Phase 4, Option 3: User-Provided Papers

If the user selects option 3:
1. Use AskUserQuestion to collect paper URLs/paths (one per line)
2. Store as `user_papers` list in pipeline state user_choices
3. When invoking `ml-optimizer:research` in Phase 5, pass `user_papers`
4. The research skill will analyze user papers FIRST before running web searches
5. User-provided papers get a +2 confidence bonus in proposal ranking

## Phase 5: Research (Optional)

If the user chose research, invoke the `ml-optimizer:research` skill with parameters:
- `model_type`: Type of model (from Phase 1)
- `task`: What the model does (from Phase 0/1)
- `current_metrics`: Current baseline performance numbers
- `problem_description`: What needs improvement (from Phase 0)
- `user_papers`: Any user-provided paper URLs or links (optional)
- `exp_root`: Path to experiments/ directory (for error logging)
Wait for research findings.

### User Checkpoint (Post-Research)

Use AskUserQuestion to show research findings:

```
Research findings:
[summary of proposals from research-findings.md]

Which proposals should I pursue?
- [1] Proposal A (complexity: low, expected: +X%)
- [2] Proposal B (complexity: medium, expected: +Y%)
- [3] Custom: describe your own approach
- [4] Skip research, just tune HPs
```

## Phase 5.5: Implement Research Proposals

If the user selected research proposals that require code changes (not just HP tuning):

1. **Invoke `ml-optimizer:implement`** with:
   - `findings_path`: `experiments/reports/research-findings.md`
   - `selected_indices`: The proposal indices the user chose in the post-research checkpoint
   - `project_root`: The project root directory

2. **Check results** from `experiments/results/implementation-manifest.json`:
   - **All validated** → proceed to experiment loop with branch-aware execution
   - **Some failed validation** → inform user, proceed with validated proposals only
   - **All failed** → fall back to HP-tuning only (no code changes)

3. **If new dependencies flagged** → Use AskUserQuestion to confirm install:
   ```
   The following new dependencies are needed for the research proposals:
   - <package>: required by <proposal_name>

   Install them? (The experiment will fail without them.)
   ```

4. **If license warnings flagged** → Use AskUserQuestion to surface to user:
   ```
   The following proposals adapted code from reference repositories with license concerns:
   - <proposal_name>: <license_warning details>

   Please review before proceeding. Continue with these proposals?
   ```

5. **If conflicts detected** → Inform user which proposals touch the same files. Each is on its own branch, so experiments run independently, but merging winners later may need manual conflict resolution.

## Phase 6: Experiment Loop (Autonomous)

This loop runs autonomously without user checkpoints until complete or blocked.

### Pre-Loop: Validate Pipeline State

Before starting the experiment loop, validate all prerequisites:

```bash
python3 -c "
import sys; sys.path.insert(0, '$HOME/.claude/plugins/ml-optimizer/scripts')
from pipeline_state import validate_phase_requirements
import json; print(json.dumps(validate_phase_requirements(6, '<exp_root>')))
"
```

**Required state:**
- `experiments/results/baseline.json` must exist with `metrics` and `config` keys
- If `implementation-manifest.json` exists, it must have `proposals` key

If validation fails, stop and report the missing prerequisites to the user.

### Pre-Loop: Load Implementation Manifest

If `experiments/results/implementation-manifest.json` exists:
1. Read the manifest
2. Collect all proposals with `"status": "validated"` — skip any with `"status": "validation_failed"` or `"status": "implementation_error"`
3. Each validated proposal branch will be tested with HP tuning
4. Also test the baseline (original branch, HP-only) for comparison
5. **Non-git detection:** If manifest has `"strategy": "file_backup"`, force sequential execution (only ONE experiment at a time)

If no manifest exists, run HP-only experiments on the current code.

### Pre-Loop: Save Pipeline State

Save Phase 0 user choices into pipeline state so they persist across interruptions:

```bash
python3 -c "
import sys, json; sys.path.insert(0, '$HOME/.claude/plugins/ml-optimizer/scripts')
from pipeline_state import save_state
save_state(6, 0, [], '<exp_root>', user_choices={
    'primary_metric': '<primary_metric>',
    'divergence_metric': '<divergence_metric>',
    'lower_is_better': <lower_is_better>,
    'target_value': <target_value or None>,
    'train_command': '<train_command>',
    'eval_command': '<eval_command or None>',
    'train_data_path': '<train_data_path>',
    'val_data_path': '<val_data_path or None>',
    'prepared_train_path': '<prepared_train_path or None>',
    'prepared_val_path': '<prepared_val_path or None>',
    'env_manager': '<env_manager>',
    'env_name': '<env_name or None>',
})
"
```

### Metric Routing Rule

**Critical:** Use the user's `divergence_metric` (from Phase 0 Q7, default: `"loss"`) for divergence detection. Use `primary_metric` (which may be "accuracy", "psnr", "f1", etc.) only for the analyze and hp-tune skills.

- Monitor skill: `metric_to_watch = <divergence_metric>`, `lower_is_better = True`
- Analyze skill: `primary_metric` from user's Phase 0 answer, `lower_is_better` based on metric type
- HP-tune skill: uses `primary_metric` for ranking

If the monitor skill cannot find `<divergence_metric>` in the logs, it will attempt auto-detection via a fallback chain (see monitor skill for details).

### Polarity Conflict Rule

- When `primary_metric == divergence_metric` (e.g., both "loss"): no conflict, both lower-is-better.
- When they differ (e.g., primary="accuracy", divergence="loss"): no conflict, independent polarity.
- When `divergence_metric` is higher-is-better (e.g., "reward" for RL): override monitor's `lower_is_better` to `False`. Divergence means metric dropped sharply, not exploded.
- Store `divergence_lower_is_better` as a separate field in user_choices.

### Branch Dispatch Strategy

When the implementation manifest contains multiple code branches:

- **Iteration 1:** Test each branch with baseline HPs (one experiment per branch). This determines which code changes show promise.
- **Iteration 2:** Prune branches that performed worse than baseline. Focus experiments on surviving branches + baseline.
- **Iterations 3+:** Focus on the best branch + HP tuning. Only keep branches within 5% of the best result.

### Loop Iteration:

1. **Get HP configs:**
   - Invoke the `ml-optimizer:hp-tune` skill with parameters:
     - `project_root`: Project root directory
     - `num_gpus`: Number of available GPUs (determines batch size)
     - `search_space`: HP search space dict from the plan
     - `iteration`: Current loop iteration (1-based)
     - `primary_metric`: The metric to optimize (from Phase 0)
     - `lower_is_better`: Whether lower values are better
     - `remaining_budget`: How many more experiments can be run before hitting the budget limit. Calculated as `(max(num_gpus, 1) × 5) - total_experiments_so_far`. HP-tune must cap proposals at `min(max(num_gpus, 1), remaining_budget)`.
     - `code_branches`: List of validated code branches from the implementation manifest (e.g., `["ml-opt/perceptual-loss", "ml-opt/cosine-scheduler"]`), or `[]` for HP-only optimization. HP-tune uses this in iteration 1 to generate one config per branch + one for baseline.
   - It reads past results and proposes the next batch of configs
   - Number of configs = `min(max(num_gpus, 1), remaining_budget)` (capped to prevent budget overshoot)

   ### HP-Tune Failure Recovery

   If hp-tune crashes or produces invalid configs:

   1. **Validate output:** Check each proposed config has required fields (`exp_id`, `config`, `gpu_id`), values are within search space bounds, and no duplicates of previously-tried configs.
   2. **If validation fails:** Retry hp-tune once with a simplified prompt: "Propose {N} configs within these ranges: {search_space}. Return valid JSON only."
   3. **If retry also fails:** Fall back to random sampling — pick `lr` uniformly from search space log-range, `batch_size` from allowed set, other HPs at baseline values. The orchestrator constructs the JSON directly.
   4. **If all fallbacks fail:** Ask user to provide configs manually via AskUserQuestion.

   Log each fallback step to error tracker with `category: "agent_failure"`, `source: "orchestrate"`.

2. **Run experiments:**
   - For each proposed config, invoke `ml-optimizer:experiment` skill
   - Pass `code_branch` and `code_proposal` from the manifest (or null for HP-only)
   - If multiple GPUs available, dispatch experiments in parallel using the Agent tool
   - Each experiment runs on a separate GPU

3. **Monitor experiments:**
   - Invoke `ml-optimizer:monitor` skill with parameters:
     - `log_files`: List of log file paths (one per running experiment)
     - `exp_ids`: Corresponding experiment IDs
     - `project_root`: Project root directory
     - `poll_interval`: Seconds between checks (default: 30)
     - `metric_to_watch`: `<divergence_metric>` from Phase 0 (default: `"loss"` — see Metric Routing Rule)
     - `lower_is_better`: `<divergence_lower_is_better>` from user_choices (True for loss-like metrics, False for reward-like metrics)
     - `model_category`: From user_choices (e.g., "rl", "generative", or null for supervised)
   - If divergence detected: the experiment is stopped automatically
   - Record divergence reason in experiment results

4. **Wait for completion:**
   - All experiments in the batch must complete (or be stopped) before analysis
   - Save pipeline state after each batch completes

5. **Analyze results:**
   - Invoke the `ml-optimizer:analyze` skill with parameters:
     - `project_root`: Project root directory
     - `batch_number`: Current loop iteration (1-based)
     - `primary_metric`: From Phase 0 (NOT "loss" — see Metric Routing Rule)
     - `lower_is_better`: Based on metric type
     - `target_value`: From Phase 0 (or null)
   - It compares all experiments, ranks them, identifies patterns
   - It recommends: continue, pivot, or stop

6. **Decision:**
   - If analyze says **continue**: loop back to step 1
   - If analyze says **pivot**: adjust the strategy, loop back to step 1
   - If analyze says **stop**: exit loop
   - **Safety limit:** Maximum total experiments budget (default: `max(num_gpus, 1) × 5`). After budget exhausted, force exit and report. This replaces the rigid 5-iteration limit to account for varying GPU counts. When `num_gpus=0` (CPU-only, e.g., scikit-learn), the budget is `1 × 5 = 5` experiments.

7. **Mid-pipeline review check** (after step 6, before looping):
   Run pattern detection:
   ```bash
   python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> patterns
   ```
   If `wasted_budget` pattern has occurrences ≥ 3, OR if the last 2 consecutive batches both had zero successful experiments:
   - Invoke `ml-optimizer:review` with:
     - `project_root`, `exp_root`, `primary_metric`, `lower_is_better`
     - `scope`: `"session"` (fast, no cross-project)
   - Read the review output's top suggestions
   - Apply relevant course corrections:
     - If review suggests narrowing LR range: pass constrained `search_space` to hp-tune
     - If review suggests pruning a branch: remove it from `code_branches`
     - If review suggests stopping: follow the stop recommendation
   - Log the mid-pipeline review:
     ```bash
     python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"orchestrate","message":"Mid-pipeline review triggered after consecutive failures","phase":6,"iteration":<iteration>,"context":{"trigger":"consecutive_failures"}}'
     ```

### Parallel GPU Dispatch Pattern:
When dispatching experiments across multiple GPUs, use the Agent tool with `subagent_type: "general-purpose"` for each experiment.

**If manifest strategy is `"file_backup"` (non-git project):** dispatch ONE experiment at a time (sequential). Wait for each to complete before starting the next. File-backup proposals share the same working directory and cannot run in parallel.

**Otherwise (git_branch strategy or HP-only):** dispatch all experiments in parallel:

```
For each config in proposed_configs:
  Agent(
    description: "Run experiment {exp_id}",
    prompt: "Use the ml-optimizer:experiment skill to run experiment {exp_id} with config: {config_json}. GPU: {gpu_id}. Project root: {project_root}. Train command: {train_command}. Eval command: {eval_command or null}. Code branch: {code_branch or null}. Code proposal: {code_proposal or null}. Prepared train path: {prepared_train_path or null}. Prepared val path: {prepared_val_path or null}.",
    subagent_type: "general-purpose",
    run_in_background: true
  )
```

Then wait for all agents to complete before invoking analyze.

### Thinking Depth for Agent Dispatch:
When dispatching agents via the Agent tool, include "ultrathink" in the prompt for **analytical** agents (hp-tune, research, analyze, implement) to trigger maximum reasoning depth. Do NOT include it for **procedural** agents (experiment, monitor) — these are execution-focused and don't benefit from extended thinking.

Example for analytical dispatch:
```
Agent(
  description: "Analyze batch {N} results",
  prompt: "Ultrathink. Use the ml-optimizer:analyze skill to analyze batch {N}. Project root: {project_root}. Primary metric: {primary_metric}. Lower is better: {lower_is_better}. Target: {target_value or null}.",
  subagent_type: "general-purpose"
)
```

## Phase 7: Report

After the experiment loop exits:

1. Invoke the `ml-optimizer:report` skill with parameters:
   - `project_root`: Project root directory
   - `primary_metric`: The metric that was optimized
   - `lower_is_better`: Whether lower is better
   - `model_description`: Brief model description (from Phase 1)
   - `task_description`: What the model does (from Phase 0/1)
2. It generates a comprehensive final report
3. Sync errors to cross-project memory:
   ```bash
   python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> sync ~/.claude/plugins/ml-optimizer
   ```
4. Ask the user about self-improvement review:
   ```
   AskUserQuestion: "Would you like a self-improvement review? It analyzes what worked, what didn't, and suggests plugin improvements for future sessions."
   Options: ["Yes, run review", "No, skip"]
   ```
   If yes, invoke `ml-optimizer:review` with:
   - `project_root`, `exp_root`, `primary_metric`, `lower_is_better`
   - `scope`: "both"
5. Present the summary to the user:

```
Optimization complete!

Best configuration: [exp_id]
[metric improvements vs baseline]

Key findings:
- [finding 1]
- [finding 2]

Full report: experiments/reports/final-report.md
```

## Error Handling

- **GPU unavailable:** Fall back to single-GPU sequential execution
- **Training crashes:** Record the error, skip to next experiment in batch
- **All experiments diverge:** Stop loop, report to user with AskUserQuestion
- **Script not found:** Ask user to provide the correct training command

## Error Tracking

At each of the following points, log an error event using the error tracker script:

### After agent failures (any phase):
When an agent dispatch fails (crash, timeout, invalid output):
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"agent_failure","severity":"critical","source":"orchestrate","message":"<failure description>","agent":"<agent_type>","phase":<phase>,"iteration":<iteration>}'
```

### After analyze recommends stop or pivot (Phase 6):
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"warning","source":"orchestrate","message":"<analyze recommendation and reason>","phase":6,"iteration":<iteration>,"context":{"action":"<continue|pivot|stop>","reason":"<from analyze>"}}'
```

### On pipeline resumption from interrupted state:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"orchestrate","message":"Pipeline resumed from interrupted state","phase":<resumed_phase>}'
```

### After review skill failure (Phase 6 or Phase 7):
If the review skill crashes or produces invalid output:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"agent_failure","severity":"warning","source":"orchestrate","message":"Review skill failed: <error description>","phase":<phase>}'
```

## Directory Structure Created

The orchestrator ensures this structure exists in the target project:
```
<project>/experiments/
  logs/<exp-id>/          # Raw training logs
  reports/                # All Markdown reports + research findings
  scripts/<exp-id>.sh     # Bash scripts used
  results/<exp-id>.json   # Parsed metrics
  dev_notes.md            # Running log of session tasks by date
```

## State Management

All state is persisted in the `experiments/` directory:
- Experiment results in `results/*.json`
- Pipeline state in `pipeline-state.json` (phase, iteration, running experiments)
- Analysis and research findings in `reports/`
- Implementation manifest in `results/implementation-manifest.json`
- Session progress in `dev_notes.md`

### Pipeline Resumption

The orchestrator can be stopped and resumed:
1. On start, check for `pipeline-state.json` via `pipeline_state.load_state()`
2. If state exists and status is "running", run `pipeline_state.cleanup_stale()` to handle interrupted experiments
3. Restore Phase 0 user choices from `state["user_choices"]` (primary_metric, divergence_metric, lower_is_better, target_value, train_command, eval_command, train_data_path, val_data_path, prepared_train_path, prepared_val_path, env_manager, env_name) — do NOT re-ask the user
4. Resume from the recorded phase and iteration
5. Read all past results to understand what has been tried

### State Validation

Before each phase transition, validate prerequisites via `pipeline_state.validate_phase_requirements()`. This prevents cascading failures from missing or corrupted data.

## Unsupported Scenarios

The following are currently out of scope. If the user requests them, explain the limitation clearly:

- **Inference optimization:** Quantization, pruning, ONNX export, TensorRT — these require a fundamentally different toolchain. Recommend dedicated tools instead.
- **Multi-machine distributed training:** This plugin operates on a single machine with multiple GPUs. Cross-node training requires a different dispatch mechanism.
- **Reinforcement learning (partial support):** RL workflows are supported with caveats. The plugin can tune RL hyperparameters and detect training divergence via policy loss or reward collapse. However, RL-specific features like reward shaping, curriculum learning, and multi-agent coordination are not orchestrated. If the user's RL setup logs standard metrics (loss, reward), the pipeline works.
- **Multi-seed ensembling:** The pipeline runs one seed per experiment. Multi-seed evaluation would require significant orchestrator changes.
- **Federated learning:** The plugin assumes all training data is locally accessible. Cross-device coordination and aggregation protocols are outside scope.
- **Multi-objective Pareto optimization:** The plugin optimizes a single `primary_metric`. For multi-objective needs, use weighted scoring in hp-tune or run separate optimization sessions per metric.
