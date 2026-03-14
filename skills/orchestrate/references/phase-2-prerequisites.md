# Phase 2: Prerequisites Check

Dispatch the prerequisites agent:
```
Agent(
  description: "Check prerequisites",
  prompt: "Check prerequisites for ML project. Parameters: project_root: {project_root}, framework: {framework}, training_script: {training_script}, config_path: {config_path}, train_data_path: {train_data_path}, val_data_path: {val_data_path}, env_manager: {env_manager}, env_name: {env_name}.",
  subagent_type: "ml-optimizer:prerequisites-agent"
)
```

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
