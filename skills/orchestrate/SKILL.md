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
- Python scripts: `~/.claude/plugins/ml-optimizer/scripts/` (gpu_check.py, parse_logs.py, detect_divergence.py, result_analyzer.py, experiment_setup.py, implement_utils.py, pipeline_state.py, schema_validator.py, plot_results.py)

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
   7. **Divergence metric name:** What metric should be monitored for training divergence? (default: "loss". Common alternatives: "train_loss", "val_loss", "objective", "nll_loss", "perplexity" for LLMs). **Must be a lower-is-better metric** — divergence detection assumes lower values mean better training. For RL tasks where "reward" is the primary metric, still use "loss" for divergence monitoring.
   8. **Optimization type:** Are you optimizing training performance or inference performance? (This plugin focuses on **training** optimization — inference optimization like quantization, pruning, or ONNX conversion is out of scope.)
   9. **Anything else** I should know about this model or training setup?
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

## Phase 2: Establish Baseline

Invoke the `ml-optimizer:baseline` skill:
- Pass the training command, eval command, and project root
- Wait for baseline results
- Store in `experiments/results/baseline.json`

## Phase 3: User Checkpoint (Post-Baseline)

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

## Phase 4: Research (Optional)

If the user chose research, invoke the `ml-optimizer:research` skill with parameters:
- `model_type`: Type of model (from Phase 1)
- `task`: What the model does (from Phase 0/1)
- `current_metrics`: Current baseline performance numbers
- `problem_description`: What needs improvement (from Phase 0)
- `user_papers`: Any user-provided paper URLs or links (optional)
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

## Phase 4.5: Implement Research Proposals

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

## Phase 5: Experiment Loop (Autonomous)

This loop runs autonomously without user checkpoints until complete or blocked.

### Pre-Loop: Validate Pipeline State

Before starting the experiment loop, validate all prerequisites:

```bash
python3 -c "
import sys; sys.path.insert(0, '$HOME/.claude/plugins/ml-optimizer/scripts')
from pipeline_state import validate_phase_requirements
import json; print(json.dumps(validate_phase_requirements(5, '<exp_root>')))
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
save_state(5, 0, [], '<exp_root>', user_choices={
    'primary_metric': '<primary_metric>',
    'divergence_metric': '<divergence_metric>',
    'lower_is_better': <lower_is_better>,
    'target_value': <target_value or None>,
    'train_command': '<train_command>',
    'eval_command': '<eval_command or None>',
})
"
```

### Metric Routing Rule

**Critical:** Use the user's `divergence_metric` (from Phase 0 Q7, default: `"loss"`) for divergence detection. Use `primary_metric` (which may be "accuracy", "psnr", "f1", etc.) only for the analyze and hp-tune skills.

- Monitor skill: `metric_to_watch = <divergence_metric>`, `lower_is_better = True`
- Analyze skill: `primary_metric` from user's Phase 0 answer, `lower_is_better` based on metric type
- HP-tune skill: uses `primary_metric` for ranking

If the monitor skill cannot find `<divergence_metric>` in the logs, it will attempt auto-detection via a fallback chain (see monitor skill for details).

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
     - `remaining_budget`: How many more experiments can be run before hitting the budget limit. Calculated as `(num_gpus × 5) - total_experiments_so_far`. HP-tune must cap proposals at `min(num_gpus, remaining_budget)`.
     - `code_branches`: List of validated code branches from the implementation manifest (e.g., `["ml-opt/perceptual-loss", "ml-opt/cosine-scheduler"]`), or `[]` for HP-only optimization. HP-tune uses this in iteration 1 to generate one config per branch + one for baseline.
   - It reads past results and proposes the next batch of configs
   - Number of configs = `min(num_gpus, remaining_budget)` (capped to prevent budget overshoot)

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
     - `lower_is_better`: `true` (always for loss-based divergence monitoring)
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
   - **Safety limit:** Maximum total experiments budget (default: `num_gpus × 5 iterations`). After budget exhausted, force exit and report. This replaces the rigid 5-iteration limit to account for varying GPU counts.

### Parallel GPU Dispatch Pattern:
When dispatching experiments across multiple GPUs, use the Agent tool with `subagent_type: "general-purpose"` for each experiment.

**If manifest strategy is `"file_backup"` (non-git project):** dispatch ONE experiment at a time (sequential). Wait for each to complete before starting the next. File-backup proposals share the same working directory and cannot run in parallel.

**Otherwise (git_branch strategy or HP-only):** dispatch all experiments in parallel:

```
For each config in proposed_configs:
  Agent(
    description: "Run experiment {exp_id}",
    prompt: "Use the ml-optimizer:experiment skill to run experiment {exp_id} with config: {config_json}. GPU: {gpu_id}. Project root: {project_root}. Train command: {train_command}. Eval command: {eval_command or null}. Code branch: {code_branch or null}. Code proposal: {code_proposal or null}.",
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

## Phase 6: Report

After the experiment loop exits:

1. Invoke the `ml-optimizer:report` skill with parameters:
   - `project_root`: Project root directory
   - `primary_metric`: The metric that was optimized
   - `lower_is_better`: Whether lower is better
   - `model_description`: Brief model description (from Phase 1)
   - `task_description`: What the model does (from Phase 0/1)
2. It generates a comprehensive final report
3. Present the summary to the user:

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
3. Restore Phase 0 user choices from `state["user_choices"]` (primary_metric, divergence_metric, lower_is_better, target_value, train_command, eval_command) — do NOT re-ask the user
4. Resume from the recorded phase and iteration
5. Read all past results to understand what has been tried

### State Validation

Before each phase transition, validate prerequisites via `pipeline_state.validate_phase_requirements()`. This prevents cascading failures from missing or corrupted data.

## Unsupported Scenarios

The following are currently out of scope. If the user requests them, explain the limitation clearly:

- **Inference optimization:** Quantization, pruning, ONNX export, TensorRT — these require a fundamentally different toolchain. Recommend dedicated tools instead.
- **Multi-machine distributed training:** This plugin operates on a single machine with multiple GPUs. Cross-node training requires a different dispatch mechanism.
- **Reinforcement learning:** The default tuning strategy assumes supervised/self-supervised training with a loss metric. RL workflows may work if a suitable divergence metric (e.g., negative reward, policy loss) is specified in Phase 0 Q7, but reward-shaped optimization may require a different approach for best results.
- **Multi-seed ensembling:** The pipeline runs one seed per experiment. Multi-seed evaluation would require significant orchestrator changes.
