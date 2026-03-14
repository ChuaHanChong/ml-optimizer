---
name: prerequisites
description: "Check and prepare prerequisites before ML experiments. Validates dataset availability and format, sets up the Python environment, and installs missing dependencies. Use when: Phase 1 (understand model) is complete and the project needs verification before running baseline."
disable-model-invocation: true
user-invocable: false
---

# Prerequisites Check

Verify that the user's project is ready for training experiments. This skill validates dataset paths and format, then checks and sets up the Python environment.

## Reference

- Dataset formats guide: `references/dataset-formats.md` (in this skill's directory)

## Inputs Expected

The orchestrator provides:
- Project root path
- ML framework detected in Phase 1 (pytorch, tensorflow, jax, etc.)
- Training script path (from Phase 1)
- Config file path (from Phase 1, if found)
- User-provided data paths (from Phase 0 Q10: `train_data_path`, `val_data_path`)
- User-specified environment manager (from Phase 0 Q11: `env_manager`, `env_name`)

## Step 1: Gather Phase 1 Context

Read the training script and config file identified in Phase 1:
- Identify data loading patterns (DataLoader calls, dataset classes, data paths in config)
- Note the ML framework and its version requirements
- Identify any command-line arguments related to data paths
- Look for preprocessing scripts (files matching `preprocess*`, `prepare*`, `setup_data*`) that may need to be run before training
- Check `README.md` or `SETUP.md` for data preparation instructions

## Step 2: Analyze Dataset Requirements

Run the project-level format detection (follows imports to find data modules):
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/prerequisites_check.py detect-format-project <project_root> <training_script>
```

This scans the training script AND any local modules it imports for data-loading patterns. It returns the expected format (image_folder, csv, hdf5, cifar, etc.), patterns found, data-related CLI arguments, and confidence level.

If confidence is "low" or format is "unknown":

**Autonomous mode auto-skip:** If `budget_mode == "autonomous"`: use the dataset as-is without format conversion. Log warning to dev_notes: "Unknown dataset format — using as-is (autonomous mode)". Skip AskUserQuestion.

Otherwise, use AskUserQuestion:
```
I couldn't automatically determine the expected dataset format from the training code.

Please describe:
1. What format is your data in? (images in folders, CSV, HDF5, etc.)
2. Does the data need any preprocessing before training?
3. Should I just skip dataset preparation and use the data as-is?
```

## Step 3: Validate User-Provided Data Paths

For each data path provided by the user (training, validation):

```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/prerequisites_check.py validate-data <path> <format>
```

Check that:
- Path exists and is readable
- Path is non-empty
- Data format matches what the training script expects

If the user said "embedded in code" (e.g., CIFAR10 auto-download), skip validation and note that data will be downloaded during training.

## Step 4: Prepare Dataset (If Needed)

If there's a format mismatch between user data and what the training script expects:

**Autonomous mode auto-skip:** If `budget_mode == "autonomous"`: skip data preparation/conversion. Log warning to dev_notes: "Dataset format mismatch — skipping preparation (autonomous mode)". Skip AskUserQuestion.

Otherwise, use AskUserQuestion:
```
Your data appears to be in [detected format] but the training script expects [expected format].

Options:
1. Let me restructure the data into the expected format (I'll create a new folder, originals untouched)
2. Skip preparation — I'll use the data as-is
3. The data is already correct — the detection was wrong
```

If restructuring is requested:
1. Create `experiments/prepared-data/` directory
2. Perform the necessary restructuring (directory layout, symlinks, format conversion)
3. **Never modify the original data files**
4. Re-validate the prepared data
5. Update the data paths that will be passed to baseline

Common preparations:
- **ImageFolder restructure:** Create class subdirectories, move/symlink images
- **Train/val split:** Split a single dataset directory into train/ and val/ subsets
- **CSV column rename:** Create a new CSV with columns matching what the training code expects

## Step 4.1: Validate Environment Manager

Run environment detection to validate the user's Phase 0 answer:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/prerequisites_check.py detect-env <project_root>
```

Compare the detected manager with the user's specified manager (`env_manager` from Phase 0):
- **Match:** Proceed as normal
- **Mismatch:**

  **Autonomous mode auto-skip:** If `budget_mode == "autonomous"`: use the detected environment manager instead. Log to dev_notes: "Env manager mismatch — using detected '<detected_manager>' (autonomous mode)". Skip AskUserQuestion.

  Otherwise, use AskUserQuestion to warn:
  ```
  I detected [detected_manager] (found [config_file]) but you specified [user_manager].
  Which should I use for package installation?
  Options: [detected_manager, user_manager]
  ```
- **User said "unknown" or skipped Q11:** Use the detected manager automatically
- **Detected "unknown":** Trust the user's answer

**Conda environment existence check:** If `env_manager` is `conda` and `env_name` is provided, verify the environment exists:
```bash
conda env list | grep -w <env_name>
```
If the environment does not exist:

**Autonomous mode auto-skip:** If `budget_mode == "autonomous"`: auto-create the conda environment with `conda create -n <env_name> python=<detected_python_version> -y`. Log to dev_notes: "Auto-created conda env '<env_name>' (autonomous mode)". Skip AskUserQuestion.

Otherwise, use AskUserQuestion:
```
Conda environment "<env_name>" does not exist.
Options:
1. Create it now: conda create -n <env_name> python=3.x -y
2. Use a different environment name
3. Use the base environment instead
```

## Step 5: Check Environment

Scan the project for required packages:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/prerequisites_check.py scan-imports <project_root>
```

Then check which third-party packages are missing, using the user's Python executable:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/prerequisites_check.py check-packages '<third_party_json>' <python_executable>
```

Where `<python_executable>` is:
- For conda: the python inside the conda env (run `conda run -n <env_name> which python` to find it)
- For venv: `<venv_path>/bin/python`
- For system: `python3` (the default if omitted)

## Step 5.1: Bulk Install from Dependency Files

Before installing packages individually, check if the project has a dependency specification:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/prerequisites_check.py bulk-install-cmd <project_root> <env_manager> [env_name]
```
Pass `<env_name>` when using conda so the generated install command targets the correct environment.

If `has_deps_file` is `true`:
1. Run the `install_command` from the output
2. Re-run the package check (Step 5) to see what's still missing
3. Only install remaining missing packages individually (Step 6)

If `has_deps_file` is `false`, skip to Step 6.

## Step 6: Install Missing Packages

**GPU-aware installation (CRITICAL for torch/tensorflow):**

Before installing `torch`, `torchvision`, `torchaudio`, `tensorflow`, `jax`, or `jaxlib`, detect the correct CUDA variant:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/prerequisites_check.py gpu-install-cmd <package> [env_manager] [env_name]
```
Pass `<env_manager>` and `<env_name>` when using conda — the output command will be wrapped with `conda run --no-banner -n <env_name>` so the package installs into the correct environment. **Never use bare `pip install torch` or `pip install jax`** — these install CPU-only versions, causing silent performance failure on GPU machines.

For all other packages, install using the user's preferred manager:

| Manager | Install Command |
|---------|----------------|
| conda   | `conda install -y -n <env_name> <package>` (try `conda install -y -n <env_name> -c conda-forge <package>` if default fails) |
| uv      | `uv pip install <package>` |
| pip     | `pip install <package>` (or use the command from `gpu-install-cmd` for GPU packages) |
| poetry  | `poetry add <package>` |

**Note:** Some import names differ from pip package names. Use the `IMPORT_TO_PACKAGE` mapping in `prerequisites_check.py` (e.g., `cv2` → `opencv-python`, `PIL` → `Pillow`, `yaml` → `PyYAML`).

After installation, re-run the package check to verify:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/prerequisites_check.py check-packages '<still_missing_json>'
```

**Classify failures:**
- **Critical** (torch, tensorflow, jax, keras, lightning, transformers): Set `ready_for_baseline: false`
- **Non-critical** (wandb, tensorboard, mlflow, comet_ml, neptune): Set `ready_for_baseline: true` with warning

## Step 6.1: Dry-Run Validation

After all dependencies are installed and data is prepared, verify the training command actually executes:

```bash
# Run the training command with minimal steps to check it works
timeout 120 <train_command_with_minimal_steps>
```

**How to limit steps:** Modify the training command based on the framework:
- **PyTorch/Lightning:** Add `--max_steps 1` or `--max_epochs 1` (check if the script accepts these flags by reading its argparse)
- **TensorFlow/Keras:** Add `--epochs 1` or modify config to set `epochs: 1`
- **scikit-learn/XGBoost:** These are typically fast enough to run the full command

If the script doesn't accept step-limiting flags, run it with a 120-second timeout — the goal is just to verify the process starts without errors, not to complete training.

**If dry-run fails:**
- Parse the error message (FileNotFoundError, ModuleNotFoundError, SyntaxError, etc.)
- Apply the same classification as baseline failure recovery (see orchestrate Phase 3)
- Log the error and set `ready_for_baseline: false` with the dry-run error details
- This catches training command typos, missing configs, and environment issues BEFORE baseline

**If dry-run succeeds:** Clean up any partial outputs (checkpoints, logs) created during the dry run.

## Step 7: Write Prerequisites Report

Write `experiments/results/prerequisites.json`:
```json
{
  "status": "ready|partial|failed",
  "dataset": {
    "train_path": "<original data path>",
    "val_path": "<original data path, or null>",
    "format_detected": "<format name>",
    "prepared": true|false,
    "prepared_train_path": "<prepared train path, or null>",
    "prepared_val_path": "<prepared val path, or null>",
    "validation_passed": true|false,
    "notes": "<any issues or info>"
  },
  "environment": {
    "manager": "conda|uv|pip|poetry|other",
    "python_version": "3.x.y",
    "packages_installed": ["<newly installed packages>"],
    "packages_failed": ["<packages that failed to install>"],
    "all_imports_resolved": true|false,
    "notes": "<any issues or info>"
  },
  "ready_for_baseline": true|false
}
```

## Step 7.1: Validate Output

```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/schema_validator.py \
  experiments/results/prerequisites.json prerequisites
```

If validation fails, fix and re-validate before proceeding.

Append to `experiments/dev_notes.md`:
```markdown
## <date> — Prerequisites Check

- **Dataset:** [format] at [path] — [validated/prepared/skipped]
- **Environment:** [manager] — [N] packages installed, [M] failed
- **Status:** [ready/partial/failed]
- **Next:** [Proceed to baseline / Fix issues first]
```

## Output

Return to the orchestrator:
- Path to `prerequisites.json`
- Whether the project is ready for baseline (`ready_for_baseline`)
- Summary of any issues found
- If data was prepared, the `prepared_train_path` and `prepared_val_path` to pass to baseline

## Error Handling

- **Data path doesn't exist:** Set `status: "failed"`, report to user
- **Format detection unknown:** Ask user, fall back to "use as-is"
- **Package install fails:** Record error, classify as critical/non-critical
- **Permission errors:** Report and suggest user fix manually
- **No internet for auto-download datasets:** Warn that CIFAR10/MNIST etc. will need network access during training

## Error Tracking

At the following points, log an error event using the error tracker:

### When data path doesn't exist or validation fails:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"resource_error","severity":"critical","source":"prerequisites","message":"Data path does not exist: <path>","phase":2,"context":{"path":"<path>","path_type":"<train|val>"}}'
```

### When data format validation fails:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"resource_error","severity":"warning","source":"prerequisites","message":"Data format validation failed: <reason>","phase":2,"context":{"format_detected":"<format>","validation_error":"<reason>"}}'
```

### When package installation fails:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"config_error","severity":"<critical|warning>","source":"prerequisites","message":"Package install failed: <package>","phase":2,"context":{"package":"<package>","manager":"<env_manager>","is_critical":<true|false>}}'
```

### When environment detection or setup fails:
```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/error_tracker.py <exp_root> log '{"category":"resource_error","severity":"warning","source":"prerequisites","message":"Environment detection failed: <error>","phase":2,"context":{"env_manager":"<env_manager>","env_name":"<env_name>"}}'
```
