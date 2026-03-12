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

0. **Cross-session memory lookup** (optional but recommended):
   Before analyzing the codebase, use `claude-mem:mem-search` to search for past optimization sessions involving similar model types, tasks, or frameworks. This may surface:
   - HP ranges that worked well for similar models
   - Optimization techniques that succeeded or failed for this type of task
   - Common pitfalls encountered in previous sessions
   Use any relevant findings to inform the optimization plan (Phase 1, step 6).

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
     - Set `divergence_metric` to `null` (do not ask Q7) and skip the monitor skill during Phase 7. Divergence detection is only meaningful for frameworks with iterative training loops.
   - **RL detection:** If the codebase imports `gym`, `gymnasium`, `stable-baselines3`, `ray.rllib`, `tianshou`, or `cleanrl`:
     - Set `model_category = "rl"` in user_choices
     - The primary_metric is likely "reward" or "episode_return" — confirm with user
     - Divergence metric: use policy/value loss if logged; otherwise use reward with `divergence_lower_is_better = False`
     - Baseline eval: use average reward over N episodes (see baseline skill RL section)
     - Training is episodic — throughput is measured in steps/sec or episodes/hour
     - **Polarity validation:** After setting `divergence_metric` and `divergence_lower_is_better`, check for inconsistency: if the metric name contains "reward", "return", or "score" but `divergence_lower_is_better` is True, or if the metric name contains "loss", "error", "nll" but `divergence_lower_is_better` is False:
       - **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, auto-infer the correct polarity from the metric name (reward/return/score → `False`, loss/error/nll → `True`). Log to dev_notes: "Auto-inferred divergence polarity for '<metric_name>': lower_is_better=<value> (autonomous mode)". Skip AskUserQuestion.
       - **Otherwise:** Warn the user: "For reward-like metrics, divergence means the metric drops (lower_is_better=False). You set lower_is_better=True — confirm this is correct?" Use AskUserQuestion. This prevents silently killing experiments during improvement.
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
   **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, skip the confirmation below. Log the plan summary to `experiments/dev_notes.md` with a note: "Plan auto-approved (autonomous mode)." Proceed directly to Phase 2.

   **Otherwise:** Use AskUserQuestion to confirm the plan aligns with their Phase 0 answers:
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

   Experiment budget mode:
   1. Auto (default): I'll judge the experiment budget based on task difficulty — [N] experiments estimated
   2. Autonomous (run until you interrupt — no experiment limit, continuous research every N batches)
   3. Custom: specify your own experiment limit

   If autonomous: How many HP tuning batches between research rounds? (default: 3)

   Does this match your expectations? Any adjustments?
   ```

   Store the budget mode as `budget_mode` in user_choices (`"auto"`, `"autonomous"`, or `"custom"`). Default: `"auto"`.

   **Note:** Autonomous mode skips ALL user checkpoints after Phase 0 discovery (Phase 4 direction, Phase 5 proposal selection, Phase 6 dependency/license approval, Pre-Loop method proposal selection). All decisions are auto-resolved with logging to dev_notes and error tracker for post-session review.

   If autonomous, also store `hp_batches_per_round` (default: 3). This controls how often the orchestrator auto-triggers a full research → implement cycle. Set to a higher value (e.g., 5-10) for slower research cadence.

   If custom, store `custom_budget` (user-specified number).

   **Adaptive budget calculation (auto mode):**

   In `"auto"` mode, the orchestrator judges task difficulty based on Phase 1 analysis and sets a `difficulty_multiplier`:

   | Difficulty | Multiplier | Criteria |
   |-----------|-----------|----------|
   | `easy`    | `× 8`    | Tabular ML (sklearn/XGBoost/LightGBM), ≤3 tunable HPs, HP-only (no code changes) |
   | `moderate`| `× 15`   | Standard supervised learning (CNN/MLP), moderate HP space (4-8 HPs), 1-2 research proposals |
   | `hard`    | `× 25`   | Complex architecture (transformers, GANs), large HP space (8+ HPs), 3+ proposals, RL, or generative tasks |

   Formula: `max_experiments = max(num_gpus, 1) × difficulty_multiplier`

   Store `difficulty` and `difficulty_multiplier` in user_choices. The user can override the budget at the Phase 4 checkpoint if they disagree with the assessment.

   **Budget by mode:**
   - `"auto"`: `max_experiments = max(num_gpus, 1) × difficulty_multiplier` (8, 15, or 25 based on assessed difficulty)
   - `"autonomous"`: `max_experiments = 999` (effectively unlimited — loop runs until user interrupts or context window exhaustion). In autonomous mode, the analyze skill's `"stop"` recommendation is logged but NOT enforced — the loop continues. The only hard stops are: user interruption, context window limit, or all reasonable approaches exhausted (analyze recommends stop 3 consecutive times). Research → implement cycles auto-trigger every `hp_batches_per_round` batches (see step 8).
   - `"custom"`: `max_experiments = custom_budget` (user-specified value)

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
- **If `budget_mode == "autonomous"`:**
  - `status = "partial"` → log warnings to `experiments/dev_notes.md`: "Prerequisites partial — proceeding anyway (autonomous mode)." Proceed to Phase 3.
  - `status = "failed"` → Classify the failure reason from `prerequisites.json`:
    - **Data path invalid / not found:** If `budget_mode == "autonomous"`, attempt auto-recovery: search the project for data files, check the training script for auto-download patterns (CIFAR, MNIST, HuggingFace `load_dataset`). If a plausible path is found, update `train_data_path`/`val_data_path` and re-run Phase 2. If not found: BLOCK with AskUserQuestion.
    - **Dependency install failed:** If `budget_mode == "autonomous"`, retry install once with `--no-deps`, then check if import still fails. If still fails: BLOCK with AskUserQuestion.
    - **Environment not found:** If `budget_mode == "autonomous"` and env_manager is conda, auto-create the environment. If creation also fails: BLOCK.
    - **Dry-run failed:** If `budget_mode == "autonomous"`, log error and attempt Phase 3 anyway (baseline may succeed where dry-run failed). If baseline also fails, exit via Phase 3 failure path.
    - **All other failures:** BLOCK with AskUserQuestion (unrecoverable without user input).
    - Log all auto-recovery attempts to dev_notes and error tracker with `category: "config_error", severity: "warning", source: "orchestrate"`.
- **Otherwise (interactive/auto/custom):**
  - `status = "partial"` → inform user of issues, ask if they want to proceed anyway or fix first
  - `status = "failed"` → diagnose from `prerequisites.json` error details:
    - Dataset not found / path invalid → ask user to verify `train_data_path`/`val_data_path` via AskUserQuestion
    - Dataset format unrecognized → ask user to specify format manually
    - Dependency install failed → show the failed packages/error, ask user to install manually
    - Environment not found → ask user to verify `env_manager`/`env_name`
    - If fixable, re-run Phase 2 after corrections. Otherwise stop and report.

**If dataset was prepared** to a new directory:
1. Read `prerequisites.json` → `dataset.prepared` field
2. If `true`, extract `dataset.prepared_train_path` and `dataset.prepared_val_path`
3. Store these as `prepared_train_path` and `prepared_val_path` in `user_choices` (see below)
4. When invoking baseline (Phase 3) and experiments (Phase 7), pass these prepared paths so training uses the prepared data instead of the original paths
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

**Autonomous mode Phase 3 unknown error:** If `budget_mode == "autonomous"` and baseline fails with an unknown error after 2 retries: log the full error to error tracker with `category: "agent_failure", severity: "critical"`, log to dev_notes: "Baseline failed after 2 retries with unknown error — exiting with partial results (autonomous mode)". Exit the pipeline and proceed to Phase 9 (report) with whatever partial results exist. Do NOT use AskUserQuestion.

**Skip-baseline fallback:** If all retries fail, offer to create a synthetic baseline.json with user-provided metric values. Mark profiling fields as `null`. This allows the experiment loop to proceed without throughput-based timeout estimation.

**Autonomous mode:** If `budget_mode == "autonomous"`, do NOT create a synthetic baseline (no user to provide metric values). Instead, exit the pipeline and proceed to Phase 9 (report) with partial results. Log to error tracker: `category: "agent_failure", severity: "critical", source: "orchestrate", message: "Baseline failed — cannot create synthetic baseline in autonomous mode"`.

## Phase 4: User Checkpoint (Post-Baseline)

**Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, skip the user question below. Auto-select option 5 (method proposals) with `method_proposal_scope = "architecture"` (balanced default). Log to dev_notes: `"Autonomous mode: auto-selected method proposals (scope: architecture)"`. Then proceed directly to the Pre-Loop method proposal section (within Phase 7).

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
5. Propose new optimization methods (method proposals — LLM knowledge + optional web search)
```

### Phase 4, Option 3: User-Provided Papers

If the user selects option 3:
1. Use AskUserQuestion to collect paper URLs/paths (one per line)
2. Store as `user_papers` list in pipeline state user_choices
3. When invoking `ml-optimizer:research` in Phase 5, pass `user_papers`
4. The research skill will analyze user papers FIRST before running web searches
5. User-provided papers get a +2 confidence bonus in proposal ranking

### Phase 4, Option 5: Method Proposals

If the user selects option 5:
1. Use AskUserQuestion to confirm scope level:
   ```
   What scope of changes should I propose?
   1. Training strategies only (optimizer, schedulers, regularization, augmentation) — safest
   2. Training + architecture changes (attention, normalization, activation variants) — bolder
   3. Full scope (everything including data pipeline and loss functions) — most aggressive
   ```
2. Store `method_proposal_scope` in pipeline state user_choices (values: `"training"`, `"architecture"`, `"full"`)
3. Skip Phase 5 (web research) — method proposals will be generated in Phase 7 Pre-Loop

## Phase 5: Research (Optional)

If the user chose research, invoke the `ml-optimizer:research` skill with parameters:
- `model_type`: Type of model (from Phase 1)
- `task`: What the model does (from Phase 0/1)
- `current_metrics`: Current baseline performance numbers
- `problem_description`: What needs improvement (from Phase 0)
- `user_papers`: Any user-provided paper URLs or links (optional)
- `exp_root`: Path to experiments/ directory (for error logging)
Wait for research findings.

### Research Failure Recovery

If the research skill fails (web search errors, timeout, or no results):

1. **First fallback:** Retry with `source: "knowledge"` (skip web search, use LLM training knowledge only). Log to error tracker: `category: "agent_failure", severity: "warning", message: "Research web search failed — retrying with knowledge-only mode"`.
2. **Second fallback:** If knowledge-only also fails, continue with HP-only optimization (no research proposals). Log to error tracker: `category: "agent_failure", severity: "warning", message: "Research failed entirely — continuing with HP-only optimization"`. Log to dev_notes: "Research failed — proceeding with HP tuning only."
3. **Each fallback step** is logged to the error tracker for post-session review.

### User Checkpoint (Post-Research)

**Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, skip the user question below. Auto-select all proposals. Log to dev_notes: `"Autonomous mode: auto-selected all N research proposals for implementation"`.

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

## Phase 6: Implement Research Proposals

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
   **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, auto-approve dependency installation. Log to error tracker: `category: "pipeline_inefficiency", severity: "info", message: "Autonomous mode: auto-approved installation of [packages]"`.

4. **If license warnings flagged** → Use AskUserQuestion to surface to user:
   ```
   The following proposals adapted code from reference repositories with license concerns:
   - <proposal_name>: <license_warning details>

   Please review before proceeding. Continue with these proposals?
   ```
   **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, auto-accept license warnings and proceed. Log to error tracker: `category: "pipeline_inefficiency", severity: "warning", message: "Autonomous mode: auto-accepted license warnings for [proposals]"`. Log to dev_notes for user review later.

5. **If conflicts detected** → Inform user which proposals touch the same files. Each is on its own branch, so experiments run independently, but merging winners later may need manual conflict resolution.

6. **Post-implementation quality review** (skip in autonomous mode):
   **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, skip code review to avoid blocking the pipeline. The experiment loop catches broken implementations via early abort. Log to dev_notes: "Skipping post-implementation code review (autonomous mode)."

   **Otherwise:**
   For validated proposals, dispatch `feature-dev:code-reviewer` to review each implementation branch for bugs, logic errors, and code quality issues. This catches problems before wasting experiment budget on broken implementations.
   - Only review proposals with `status: "validated"` in the manifest
   - If the reviewer flags critical issues, mark the proposal as `validation_failed` and skip it
   - If the reviewer flags minor issues (style, non-blocking), log them but proceed

## Phase 7: Experiment Loop (Autonomous)

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
   **Branch existence validation:** Before passing `code_branches` to hp-tune, verify each branch exists via `git rev-parse --verify <branch>`. Remove missing branches and log to error tracker.
4. Also test the baseline (original branch, HP-only) for comparison
5. **Non-git detection:** If manifest has `"strategy": "file_backup"`, force sequential execution (only ONE experiment at a time)

If no manifest exists, run HP-only experiments on the current code.

### Pre-Loop: Method Proposals (if user chose option 5 in Phase 4)

If `method_proposal_scope` is set in user_choices (i.e., user chose option 5 in Phase 4):

1. **Invoke `ml-optimizer:research`** with:
   - `source`: `"both"`
   - `scope_level`: from user_choices `method_proposal_scope`
   - `output_path`: `"experiments/reports/research-findings-method-proposals.md"`
   - All other standard inputs (`model_type`, `task`, `current_metrics`, `problem_description`, `exp_root`)

2. **Present proposals to user for confirmation** (same as Phase 5 post-research checkpoint):

   **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, skip the user question. Auto-select all proposals. Log to dev_notes: `"Autonomous mode: auto-selected all N method proposals for implementation"`.

   ```
   Method proposals (from LLM knowledge + web search):
   [summary of proposals from research-findings-method-proposals.md]

   Which proposals should I pursue?
   - [1] Proposal A (complexity: low, expected: +X%)
   - [2] Proposal B (complexity: medium, expected: +Y%)
   - [3] Custom: describe your own approach
   - [4] Skip, just tune HPs on existing code
   ```

3. **If user selects proposals:** Invoke `ml-optimizer:implement` with:
   - `findings_path`: `"experiments/reports/research-findings-method-proposals.md"`
   - `selected_indices`: The indices the user chose
   - `project_root`: Project root directory

4. **Check implementation results** from `experiments/results/implementation-manifest.json`:
   - Merge validated method proposal branches into the `code_branches` list
   - Follow the same handling as Phase 6 (failed proposals, dependencies, license warnings)

5. **Store method proposal state:**
   - `method_proposal_iterations`: 1 (initial)

### Pre-Loop: Route `hp_only` Research Proposals

When processing research proposals (from Phase 5 or mid-loop step 7), check each proposal's `type` field:
- **`type: "hp_only"`**: These proposals recommend search space modifications (e.g., "try cyclical learning rates", "increase weight decay range") rather than code changes. Route them directly to hp-tune as search space adjustments — skip the implement skill entirely. Merge the suggested HP ranges into the existing `search_space` dict.
- **`type: "code_change"` or no type field**: Route through implement as normal (create branches, validate, etc.).

This prevents unnecessary implementation overhead for proposals that only affect HP tuning parameters.

### Pre-Loop: Initialize Research Cadence

Initialize the research round counter for autonomous mode:
- `batches_since_last_research = 0`
- This counter tracks how many HP tuning batches have run since the last research → implement cycle
- In autonomous mode, when this counter reaches `hp_batches_per_round`, step 8 auto-triggers a new research round

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
    'divergence_lower_is_better': <divergence_lower_is_better>,
    'target_value': <target_value or None>,
    'train_command': '<train_command>',
    'eval_command': '<eval_command or None>',
    'train_data_path': '<train_data_path>',
    'val_data_path': '<val_data_path or None>',
    'prepared_train_path': '<prepared_train_path or None>',
    'prepared_val_path': '<prepared_val_path or None>',
    'env_manager': '<env_manager>',
    'env_name': '<env_name or None>',
    'model_category': '<model_category or None>',
    'user_papers': <user_papers or None>,
    'budget_mode': '<budget_mode>',
    'method_proposal_scope': '<method_proposal_scope or None>',
    'method_proposal_iterations': <method_proposal_iterations or 0>,
    'hp_batches_per_round': <hp_batches_per_round or 3>,
})
"
```

### Metric Routing Rule

**Critical:** Use the user's `divergence_metric` (from Phase 0 Q7, default: `"loss"`) for divergence detection. Use `primary_metric` (which may be "accuracy", "psnr", "f1", etc.) only for the analyze and hp-tune skills.

- Monitor skill: `metric_to_watch = <divergence_metric>`, `lower_is_better = <divergence_lower_is_better>`
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
- **Iterations 3+:** Focus on the best branch + HP tuning. Only keep branches whose best metric is within 5% relative of the overall best: `abs(branch_best - overall_best) / abs(overall_best) <= 0.05`. If `overall_best` is zero, keep all branches.

### Loop Iteration:

1. **Get HP configs** (use speculative proposals from previous iteration if available):
   - **If speculative proposals are available from the previous iteration's background hp-tune:**
     1. Validate speculative proposals before use (see "Speculative Proposal Validation" below)
     2. If valid → use them as this iteration's configs. Skip hp-tune invocation entirely.
     3. If invalid → discard them and invoke hp-tune synchronously as normal.
   - **Otherwise (first iteration, or speculative proposals were discarded):**
     - Invoke the `ml-optimizer:hp-tune` skill with parameters:
     - `project_root`: Project root directory
     - `num_gpus`: Number of available GPUs (determines batch size)
     - `search_space`: HP search space dict from the plan
     - `iteration`: Current loop iteration (1-based)
     - `primary_metric`: The metric to optimize (from Phase 0)
     - `lower_is_better`: Whether lower values are better
     - `remaining_budget`: How many more experiments can be run before hitting the budget limit. Calculated as `max_experiments - total_experiments_so_far` (where `max_experiments` is set by the adaptive difficulty assessment or user override). HP-tune must cap proposals at `min(max(num_gpus, 1), remaining_budget)`.
     - `code_branches`: List of validated code branches from the implementation manifest (e.g., `["ml-opt/perceptual-loss", "ml-opt/cosine-scheduler"]`), or `[]` for HP-only optimization. HP-tune uses this in iteration 1 to generate one config per branch + one for baseline.
     - `max_batch_size` *(optional)*: One step below the smallest OOM-causing batch size from error tracker. Omit if no OOM events have occurred. See "OOM feedback to hp-tune" in Error Handling.
   - It reads past results and proposes the next batch of configs
   - Number of configs = `min(max(num_gpus, 1), remaining_budget)` (capped to prevent budget overshoot)
   - **Check hp-tune recommendation:** If hp-tune output includes `"recommendation": "stop"`, log it to error tracker with `category: "pipeline_inefficiency"` and note it for the analyze step. Analyze makes the final continue/pivot/stop decision, but hp-tune's recommendation provides an early signal of search space exhaustion.

   ### HP-Tune Failure Recovery

   If hp-tune crashes or produces invalid configs:

   1. **Validate output:** Check each proposed config has required fields (`exp_id`, `config`, `gpu_id`), values are within search space bounds, and no duplicates of previously-tried configs.
   2. **If validation fails:** Retry hp-tune once with a simplified prompt: "Propose {N} configs within these ranges: {search_space}. Return valid JSON only."
   3. **If retry also fails:** Fall back to random sampling — pick `lr` uniformly from search space log-range, `batch_size` from allowed set, other HPs at baseline values. The orchestrator constructs the JSON directly.
   4. **If random sampling also fails** (construction error):
      - **Autonomous mode:** If `budget_mode == "autonomous"`, use the baseline config as-is for all experiments in this batch (re-validates baseline, keeps loop alive). Log to error tracker: `category: "agent_failure", severity: "critical", source: "orchestrate", message: "All HP-tune fallbacks failed — using baseline config as placeholder batch (autonomous mode)"`. Log to dev_notes: "HP-tune completely failed — running baseline-config batch as placeholder." Proceed to step 2 with baseline configs.
      - **Interactive mode:** Ask user to provide configs manually via AskUserQuestion.

   Log each fallback step to error tracker with `category: "agent_failure"`, `source: "orchestrate"`.

2. **Run experiments:**
   - For each proposed config, invoke `ml-optimizer:experiment` skill
   - Pass `code_branch` and `code_proposal` from the manifest (or null for HP-only)
   - If multiple GPUs available, dispatch experiments in parallel using the Agent tool
   - Each experiment runs on a separate GPU

3. **Monitor experiments:**
   - **If `divergence_metric` is not null**, invoke `ml-optimizer:monitor` skill with parameters:
     - `log_files`: List of log file paths (one per running experiment)
     - `exp_ids`: Corresponding experiment IDs
     - `project_root`: Project root directory
     - `poll_interval`: Seconds between checks (default: 30)
     - `metric_to_watch`: `<divergence_metric>` from Phase 0 (default: `"loss"` — see Metric Routing Rule)
     - `lower_is_better`: `<divergence_lower_is_better>` from user_choices (True for loss-like metrics, False for reward-like metrics)
     - `model_category`: From user_choices (e.g., "rl", "generative", or null for supervised)
   - Monitor status handling:
     - `healthy`: Training is progressing normally — continue waiting
     - `diverged`: Stop the experiment automatically, record divergence reason in experiment results
     - `completed`: Training finished naturally during monitoring — proceed to wait/analysis
     - `unmonitored`: The watched metric was not found in the logs after all fallback attempts. Warn the user once (via dev_notes) that divergence monitoring is disabled for this experiment. Continue without divergence checks — rely on the experiment's hard timeout (from baseline profiling) as the safety net.
     - `failed`: Monitor itself encountered an error — log as `agent_failure`, continue without monitoring for remaining experiments in this batch
     - `no_output`: Log file has no parseable data yet — continue monitoring (normal for early training)
   - **If `divergence_metric` is null** (tabular ML — scikit-learn, XGBoost, LightGBM): skip the monitor skill entirely. Wait for experiments to complete naturally without divergence monitoring.

   ### Early Batch Abort on Mass Divergence

   When monitoring a batch of experiments in parallel, track divergence timestamps:

   - **Trigger:** If `>= 2` experiments in the same batch diverge AND all divergences occurred within 60 seconds of their respective start times:
     1. Cancel remaining running experiments in the batch (kill processes)
     2. Mark cancelled experiments as `status: "cancelled"` with `notes: "Early batch abort — mass divergence detected"`
     3. Log to error tracker: `category: "training_failure", severity: "critical", source: "orchestrate", message: "Early batch abort: <N_diverged> of <batch_size> experiments diverged within 60s of start"`
     4. Log to dev_notes: "Early batch abort at iteration <N>: <diverged_count> experiments diverged within 60s. Cancelled <cancelled_count> remaining."
     5. Proceed directly to step 4 with partial results
   - **Rationale:** Divergence within 60 seconds indicates a systematic config problem (LR too high, NaN initialization) that will affect all similar configs. Two or more confirms it's systematic, not a fluke.

4. **Wait for completion:**
   - All experiments in the batch must complete (or be stopped) before analysis
   - **Experiment timeout:** Each experiment has a hard timeout computed as:
     - If `baseline.json` has `profiling.estimated_timeout_seconds` (tabular ML): use that value directly
     - Else if `baseline.json` has throughput profiling: `max_experiment_duration = baseline_training_time * 3`
     - Else (no profiling): fallback to 21600 seconds (6 hours)
     If an experiment exceeds the timeout:
     1. Kill the experiment process
     2. Set `status: "timeout"` in the experiment result JSON
     3. Log to error tracker: `category: "timeout", severity: "warning", message: "Experiment <exp_id> timed out after <duration>s (limit: <max_duration>s)"`
     4. Continue with the remaining experiments in the batch
   - Save pipeline state after each batch completes

5. **Analyze results + speculative hp-tune (parallel):**
   - **Start analyze synchronously:**
     - Invoke the `ml-optimizer:analyze` skill with parameters:
       - `project_root`: Project root directory
       - `batch_number`: Current loop iteration (1-based)
       - `primary_metric`: From Phase 0 (NOT "loss" — see Metric Routing Rule)
       - `lower_is_better`: Based on metric type
       - `target_value`: From Phase 0 (or null)
       - `remaining_budget`: `max_experiments - total_experiments_so_far` (analyze uses this in its pivot decision tree to gate research pivots)
     - It compares all experiments, ranks them, identifies patterns
     - It recommends: continue, pivot, or stop
   - **At the SAME TIME, start speculative hp-tune in background** (only if `remaining_budget > max(num_gpus, 1)`):
     ```
     Agent(
       description: "Speculative hp-tune for next batch",
       prompt: "Ultrathink. This is a SPECULATIVE proposal — the orchestrator may discard these results if analyze recommends stop or pivot. Use the ml-optimizer:hp-tune skill with: project_root: {project_root}, num_gpus: {num_gpus}, search_space: {search_space}, iteration: {iteration + 1}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, remaining_budget: {remaining_budget - current_batch_size}, code_branches: {code_branches}.",
       subagent_type: "general-purpose",
       run_in_background: true
     )
     ```
   - If `remaining_budget <= max(num_gpus, 1)`: skip speculative hp-tune (not enough budget for another full batch)
   - Analyze completes first (it's synchronous). Speculative hp-tune may still be running.
   - **Wait policy:** If analyze says "continue" and speculative hp-tune has not completed, wait up to 120 seconds. If it completes in time, validate and use. If it doesn't, discard and invoke hp-tune synchronously. Log timeout to error tracker with `category: "agent_failure", severity: "info"`.
   - If analyze says "pivot" or "stop", discard speculative hp-tune immediately — do not wait.

6. **Decision:**
   - If analyze says **continue** AND speculative proposals are available and valid:
     - Use speculative proposals → loop back to step 2 immediately (zero GPU idle time)
   - If analyze says **continue** BUT speculative proposals are invalid or unavailable:
     - Discard speculative proposals (if Agent failure, log to error tracker with `category: "agent_failure", source: "orchestrate"`). Invoke hp-tune synchronously → loop back to step 2
   - If analyze says **pivot**:
     - Discard speculative proposals. Apply pivot adjustments → invoke hp-tune synchronously → loop back to step 2
     **Pivot dispatch by type:**
     - `"branch_test"`: Pass analyze's suggestion to hp-tune. Generate configs for untested branches with baseline HPs. No research needed.
     - `"hp_expand"`: Widen the search space around the best config (extend LR range by 2× in each direction). Pass updated `search_space` to hp-tune.
     - `"narrow_space"`: Constrain the search space to the range around the best result (analyze's `suggestion` field contains bounds). Pass narrowed `search_space` to hp-tune.
     - `"regularization"`: Add regularization HPs (weight_decay, dropout) to the search space or expand their range. Pass updated `search_space` to hp-tune. No research needed.
     - `"research"`: Route to step 7 (same as `method_proposal`). Requires `remaining_budget >= 5`.
     - `"method_proposal"`, `"qualitative_change"`: Route to step 7 (existing handling). Requires `remaining_budget >= 3`.
     - **Unknown pivot_type:** Treat as `"hp_expand"` (safest default). Log to error tracker.
   - If analyze says **stop**:
     - Discard speculative proposals.
     - In `"auto"` or `"custom"` mode: exit loop
     - In `"autonomous"` mode: log the stop recommendation but continue the loop. Only force-stop if analyze recommends stop 3 consecutive times (indicating true convergence, not a one-off plateau). Track via `consecutive_stop_count` in pipeline state: increment on each "stop" recommendation, reset to 0 on "continue" or "pivot". Persist via `save_state()` at the end of each iteration. On pipeline resume, read from state (default 0).
   - **If analyze output is malformed or contains an unexpected action:** Treat as `agent_failure`. Log to error tracker. Retry analyze once with a simplified prompt: "Based on the experiment results, should we continue, pivot, or stop? Respond with exactly one of: continue, pivot, stop." If retry also fails, default to `continue` if remaining_budget > 0, or `stop` if budget exhausted.
   - **Safety limit:** Maximum experiments budget depends on `budget_mode` from Phase 1:
     - `"auto"` (default): `max(num_gpus, 1) × difficulty_multiplier` (easy=8, moderate=15, hard=25)
     - `"custom"`: user-specified `custom_budget`
     - `"autonomous"`: 999 (effectively unlimited — runs until interrupted or 3 consecutive stop recommendations)
     After budget exhausted, force exit and report. When `num_gpus=0` (CPU-only, e.g., scikit-learn), the multiplier applies to `1`.

7. **Mid-loop method proposal trigger** (when analyze recommends new methods):

   If analyze returns `pivot_type: "method_proposal"` or `pivot_type: "qualitative_change"`:

   a. **Budget gate:** If `remaining_budget < 3`, skip method proposals and recommend stop with current best result. Log:
      ```bash
      python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"orchestrate","message":"Method proposals skipped: remaining_budget (<N>) < 3","phase":7,"iteration":<iteration>}'
      ```

   b. **Scope confirmation:**
      **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, use stored `method_proposal_scope` from user_choices. Log to dev_notes: "Autonomous mode: using stored method_proposal_scope '<scope>' for mid-loop proposals". Skip AskUserQuestion.

      **Otherwise:** Ask the user which scope level to use:
      ```
      HP tuning has plateaued. I can propose new optimization methods.

      Scope options:
      1. Training strategies only (optimizers, schedulers, regularization, augmentation, loss functions)
      2. Training + architecture changes (attention, normalization, activations, block design)
      3. Full scope (training + architecture + data pipeline, distillation, ensemble)
      4. Skip — stop with current best result

      Which scope? (1/2/3/4)
      ```
      If user chooses 4 (skip), exit the loop and proceed to Phase 9 (report).

   c. **Generate proposals:** Invoke `ml-optimizer:research` with:
      - `source`: `"both"`
      - `scope_level`: user's choice (`"training"` / `"architecture"` / `"full"`)
      - `output_path`: `"experiments/reports/research-findings-method-proposals-iter<N>.md"` (where N = `method_proposal_iterations + 1`)
      - All other standard inputs (project_root, model description, primary_metric, etc.)

   d. **Present proposals:**
      **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, accept all proposals automatically. Log to dev_notes: "Autonomous mode: auto-accepted all N mid-loop method proposals". Skip AskUserQuestion.

      **Otherwise:** Show the generated proposals to the user for confirmation. The user can accept all, select a subset, or reject all (which exits the loop).

   e. **Implement proposals:** Invoke `ml-optimizer:implement` with the confirmed method proposal findings. This creates new `ml-opt/<slug>` branches.

   f. **Merge into experiment loop:** Add the new validated branches to `code_branches`. Reset the iteration counter for these new branches only (they start at iteration 1 = `method_default_hp` tier). Existing branches keep their iteration count.

   g. **Update state:**
      - Increment `method_proposal_iterations` in user_choices
      - Deduct expected cost from `remaining_budget`: `expected_cost = len(new_branches) * max(num_gpus, 1)` (accounts for GPU-parallel batch sizing per branch)
      - Save pipeline state

   h. **Continue loop:** Loop back to step 1 (hp-tune) with the expanded `code_branches` list. Reset `batches_since_last_research = 0`.

8. **Research round check** (autonomous mode only — cadence-based research trigger):

   This step auto-triggers research → implement on a regular cadence, independent of analyze's pivot recommendation. It only applies in autonomous mode with `method_proposal_scope` set.

   **Conditions (ALL must be true):**
   - `budget_mode == "autonomous"`
   - `method_proposal_scope` is set (user opted into method proposals)
   - `batches_since_last_research >= hp_batches_per_round`
   - Step 7 did NOT already trigger this iteration (avoid double research)

   **If conditions met:**

   a. **Log the trigger:**
      ```bash
      python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"orchestrate","message":"Autonomous research round triggered after <N> HP batches","phase":7,"iteration":<iteration>,"context":{"batches_since_last_research":<N>,"method_proposal_iterations":<M>}}'
      ```

   b. **Generate proposals:** Invoke `ml-optimizer:research` with:
      - `source`: `"both"`
      - `scope_level`: from user_choices `method_proposal_scope`
      - `output_path`: `"experiments/reports/research-findings-method-proposals-iter<N>.md"` (where N = `method_proposal_iterations + 1`)
      - All standard inputs (project_root, model description, primary_metric, etc.)

   c. **Check results:**
      - If research returns new proposals (not all filtered by deduplication): proceed to implement
      - If research returns **no new proposals** (all deduplicated): skip implement, double `hp_batches_per_round` (exponential backoff), log:
        ```bash
        python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"orchestrate","message":"Research round yielded no new proposals — increasing cadence to <new_value> batches","phase":7,"iteration":<iteration>}'
        ```

   d. **Implement proposals (no user confirmation):** In autonomous mode, ALL returned proposals are implemented automatically (the user opted into autonomous operation). Invoke `ml-optimizer:implement` with the research findings. This creates new `ml-opt/<slug>` branches.

   e. **Merge into experiment loop:** Same as step 7f — add new validated branches to `code_branches`, reset iteration counter for new branches.

   f. **Update state:**
      - Increment `method_proposal_iterations`
      - Reset `batches_since_last_research = 0`
      - Save pipeline state

   **If conditions NOT met:** Increment `batches_since_last_research` and continue.

9. **Mid-pipeline review check** (after step 6/7/8, before looping):
   Run pattern detection:
   ```bash
   python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> patterns
   ```
   If `wasted_budget` pattern has occurrences ≥ 3, OR if the last 2 consecutive batches both had zero successful experiments:

   **If `budget_mode == "autonomous"`:**
   - Start review in background via Agent with `run_in_background: true`:
     ```
     Agent(
       description: "Async mid-pipeline review",
       prompt: "Use the ml-optimizer:review skill. project_root: {project_root}, exp_root: {exp_root}, primary_metric: {primary_metric}, lower_is_better: {lower_is_better}, scope: session.",
       subagent_type: "general-purpose",
       run_in_background: true
     )
     ```
   - Continue to next loop iteration immediately (do NOT wait for review)
   - At the START of the next loop iteration (before step 1), check if the background review has completed:
     - If completed successfully: read suggestions and apply course corrections (narrow search space, prune branches, stop)
     - If completed with error or no output: log as `agent_failure` to error tracker, skip course corrections for this iteration
     - If still running: continue without waiting (suggestions will be applied in the following iteration)
   - **Design note:** Applying review suggestions one batch late is intentional — blocking the pipeline to wait for review would waste GPU time. Review suggestions are strategic (search space narrowing, branch pruning), so a one-batch delay is acceptable.
   - Log: "Mid-pipeline review started in background (autonomous mode)"

   **Otherwise (interactive/auto/custom):**
   - Invoke `ml-optimizer:review` synchronously with:
     - `project_root`, `exp_root`, `primary_metric`, `lower_is_better`
     - `scope`: `"session"` (fast, no cross-project)
   - Read the review output's top suggestions
   - Apply relevant course corrections:
     - If review suggests narrowing LR range: pass constrained `search_space` to hp-tune
     - If review suggests pruning a branch: remove it from `code_branches`
     - If review suggests stopping: follow the stop recommendation
   - Log the mid-pipeline review:
     ```bash
     python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"orchestrate","message":"Mid-pipeline review triggered after consecutive failures","phase":7,"iteration":<iteration>,"context":{"trigger":"consecutive_failures"}}'
     ```

10. **Loop back:** After steps 6/7/8/9, increment `batches_since_last_research` and return to step 1 (Get HP configs). The loop continues until the Decision step (6) or budget exhaustion forces an exit.

### Speculative Proposal Validation

Before using speculative proposals from a previous iteration's background hp-tune, verify ALL of these:

1. **Branch validity:** All `code_branch` values in proposals still exist in the active branch list (none were pruned by analyze)
2. **Budget compliance:** Number of proposals ≤ `remaining_budget`
3. **No search space conflict:** If analyze recommended narrowing the search space (via pivot), check that speculative proposals fall within the new bounds
4. **No duplicates:** Speculative proposals don't duplicate experiments from the just-completed batch

If ANY check fails, discard ALL speculative proposals and invoke hp-tune synchronously with updated parameters.

### Parallel GPU Dispatch Pattern:
When dispatching experiments across multiple GPUs, use the Agent tool with `subagent_type: "general-purpose"` for each experiment.

**If manifest strategy is `"file_backup"` (non-git project):** dispatch ONE experiment at a time (sequential). Wait for each to complete before starting the next. File-backup proposals share the same working directory and cannot run in parallel.

**Otherwise (git_branch strategy or HP-only):** dispatch all experiments in parallel:

```
For each config in proposed_configs:
  Agent(
    description: "Run experiment {exp_id}",
    prompt: "Use the ml-optimizer:experiment skill to run experiment {exp_id} with config: {config_json}. GPU: {gpu_id}. Project root: {project_root}. Train command: {train_command}. Eval command: {eval_command or null}. Code branch: {code_branch or null}. Code proposal: {code_proposal or null}. Proposal source: {proposal_source or null}. Method tier: {method_tier or null}. Iteration: {iteration}. Prepared train path: {prepared_train_path or null}. Prepared val path: {prepared_val_path or null}.",
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
  prompt: "Ultrathink. Use the ml-optimizer:analyze skill to analyze batch {N}. Project root: {project_root}. Primary metric: {primary_metric}. Lower is better: {lower_is_better}. Target: {target_value or null}. Remaining budget: {remaining_budget}.",
  subagent_type: "general-purpose"
)
```

## Phase 8: Method Stacking (Sequential Accumulation)

**Pre-check:** If the implementation manifest uses `strategy: "file_backup"` (non-git project), skip stacking entirely. Log to dev_notes: "Stacking requires git branches — skipped for file-backup projects." Proceed to Phase 9.

**Trigger:** When the experiment loop ends (analyze recommends `stop` or budget exhausted) AND `methods_with_improvement >= 5`.

Count `methods_with_improvement` by calling `rank_methods_for_stacking()` from `result_analyzer.py`:
```bash
python3 scripts/result_analyzer.py <results_dir> <metric> [baseline_id] [lower_is_better]
```
Then count entries in the result. If fewer than 5, skip to Phase 9.

**Checkpoint:**
- **Interactive mode:** Ask user: "{N} methods showed improvement over baseline. Would you like to stack them to find compound gains? The best methods will be merged sequentially."
  - If user declines → skip to Phase 9
- **Autonomous mode:** Auto-proceed. Log to dev_notes: "Auto-entering stacking phase with {N} improved methods."

### Stacking Loop

1. **Rank methods** by improvement magnitude (descending) using `rank_methods_for_stacking()`.

2. **Initialize stack:** The best method's branch becomes `ml-opt/stack-1`. No experiment needed — its existing best result serves as the stack baseline.

3. **For each remaining method** (rank 2, 3, ... N):

   a. **Create stack branch:**
   ```bash
   git checkout -b ml-opt/stack-<order> ml-opt/stack-<order-1>
   # (For order=2, branch from ml-opt/stack-1 which is the best method's branch)
   ```

   b. **Merge the next method:**
   ```bash
   git merge ml-opt/<method-slug> --no-ff --no-edit
   ```

   c. **If clean merge** → proceed to validation.

   d. **If merge conflicts** → dispatch implement-agent:
      - **Prompt:** "Resolve merge conflicts in the following files. Both methods must be preserved: [method-A description] and [method-B description]. The goal is to combine their functionality."
      - **Files:** List of conflicting files from `git diff --name-only --diff-filter=U`
      - If implement-agent succeeds → `git add .` and `git commit -m "Resolve merge conflicts for stack-<order>"`
      - If implement-agent fails → skip this method:
        - `git merge --abort`
        - Log to error tracker: `category: "implementation_error", message: "Stacking conflict unresolved for <method-slug>"`
        - Continue to next method

   e. **Validate** (syntax, import, forward pass — same as implement skill validation).
      - If validation fails → skip: delete branch, log reason, continue.

   f. **Run experiment** using the `ml-optimizer:experiment` skill:
      - `code_branch`: `ml-opt/stack-<order>`
      - `code_branches`: list of all methods in this stack
      - `method_tier`: `"stacked_default_hp"`
      - `stacking_order`: current order number
      - `stack_base_exp`: exp_id of the previous stack's best result
      - `config`: best HP config from the top method currently in the stack

   g. **Evaluate result:**
      - Compare to previous stack step's metric value.
      - **If improved:** Keep this stack step.
        - Update `stack_base_branch = ml-opt/stack-<order>`
        - **Optional HP-tune:** If the improvement is > 1% AND remaining budget allows, invoke `ml-optimizer:hp-tune` with:
          - `project_root`: Project root directory
          - `num_gpus`: Number of available GPUs
          - `primary_metric`: The metric to optimize (from Phase 0)
          - `lower_is_better`: Whether lower values are better
          - `code_branches`: [current stack branch]
          - `iteration`: 1
          - `remaining_budget`: min(2, actual remaining)
          - `search_space`: narrowed to HPs the newly added method likely interacts with (LLM judgment)
        - If HP-tune improves further, record as `method_tier: "stacked_tuned_hp"`
      - **If worse or equal:** Skip this method.
        - Delete `ml-opt/stack-<order>` branch
        - Log: "Method <slug> skipped in stacking (metric degraded by X%)"
        - Continue to next method (next stack branch re-branches from last successful stack)

4. **Save stacking state** to `pipeline-state.json` via `save_state(user_choices={"stacking": {...}})` after each stack step (for resumption):
   ```json
   {
     "user_choices": {
       "stacking": {
         "ranked_methods": ["method-b", "method-a", "method-c"],
         "current_stack_order": 3,
         "stack_base_branch": "ml-opt/stack-2",
         "stack_base_exp": "exp-stack-002",
         "skipped_methods": ["method-c"],
         "stacked_methods": ["method-b", "method-a"]
       }
     }
   }
   ```

5. **Final result:** The last successful `ml-opt/stack-<N>` branch is the compound best.
   Log to dev_notes: "Stacking complete. Final stack: [methods]. Compound gain: X% over baseline. Branch: ml-opt/stack-<N>"

### Stacking Phase Resumption

On pipeline restart, if `pipeline-state.json` contains a `stacking` key in `user_choices`:
1. Read stacking state
2. **Validate before resuming:**
   a. `current_stack_order < len(ranked_methods)` — if not, stacking is already complete; skip to Phase 9
   b. Verify `ml-opt/stack-<current_stack_order>` branch exists (`git branch --list`). If missing, log error to error tracker and skip to Phase 9 with partial results.
   c. Verify `stack_base_exp` result file is readable. If missing, fall back to the last known good stack result from `stacked_methods`.
3. Resume from `current_stack_order + 1`
4. Continue with remaining methods in `ranked_methods` that aren't in `stacked_methods` or `skipped_methods`

## Phase 9: Report

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
4. **Self-improvement review:**
   **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`, auto-run review with `scope: "session"`. Skip AskUserQuestion. Log to dev_notes: "Auto-running self-improvement review (autonomous mode)." Invoke `ml-optimizer:review` with:
   - `project_root`, `exp_root`, `primary_metric`, `lower_is_better`
   - `scope`: "session"

   **Otherwise:** Ask the user:
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
- **All experiments diverge in a batch:**
  - **Recovery attempt:** Before stopping, attempt a recovery batch with halved learning rates (divide all LR values by 2). Log to error tracker: `category: "training_failure", severity: "warning", message: "All experiments diverged — attempting recovery with halved LRs"`.
  - If the recovery batch also all-diverges: stop the loop and report to user. In autonomous mode, log to dev_notes and proceed to Phase 9 (report). In interactive mode, use AskUserQuestion to inform user.
- **OOM feedback to hp-tune:** When an experiment fails with `CUDA out of memory`:
  1. Record the OOM-causing batch size in the error tracker: `category: "training_failure", context: {"oom_batch_size": <batch_size>}`
  2. On the next hp-tune invocation, pass `max_batch_size` constraint (one step below the OOM-causing batch size) so hp-tune avoids proposing configs that will OOM again
  3. If multiple OOM events occur, use the smallest OOM-causing batch size as the constraint
- **Script not found:** Ask user to provide the correct training command

## Error Tracking

At each of the following points, log an error event using the error tracker script:

### After agent failures (any phase):
When an agent dispatch fails (crash, timeout, invalid output):
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"agent_failure","severity":"critical","source":"orchestrate","message":"<failure description>","agent":"<agent_type>","phase":<phase>,"iteration":<iteration>}'
```

### After analyze recommends stop or pivot (Phase 7):
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"warning","source":"orchestrate","message":"<analyze recommendation and reason>","phase":7,"iteration":<iteration>,"context":{"action":"<continue|pivot|stop>","reason":"<from analyze>"}}'
```

### On pipeline resumption from interrupted state:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"pipeline_inefficiency","severity":"info","source":"orchestrate","message":"Pipeline resumed from interrupted state","phase":<resumed_phase>}'
```

### After review skill failure (Phase 7 or Phase 9):
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
2. If state exists and status is "running", run `pipeline_state.cleanup_stale()` to handle interrupted experiments. This uses a 2-hour timeout: any experiment with status: "running" last modified >2 hours ago is marked status: "failed" with notes: "Marked failed by cleanup_stale — presumed interrupted". Log cleaned-up items to dev_notes before resuming.
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
