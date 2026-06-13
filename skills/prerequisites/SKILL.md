---
name: prerequisites
description: "Check and prepare prerequisites before ML experiments. Validates dataset availability and format, sets up the Python environment, and installs missing dependencies. Use when: Phase 1 (understand model) is complete and the project needs verification before running baseline."
user-invocable: false
---

# Prerequisites Check

Verify that the user's project is ready for training experiments. This skill validates the required GitNexus code-graph tooling, dataset paths and format, then checks and sets up the Python environment.

> **Path convention:** All paths written as `<exp_root>/...` refer to the `exp_root` parameter from your dispatch. The plugin does not hardcode the output directory name.

## Reference

- Dataset formats guide: `${CLAUDE_SKILL_DIR}/references/dataset-formats.md` (in this skill's directory)

## Inputs Expected

The orchestrator provides:
- Project root path
- ML framework detected in Phase 1 (pytorch, tensorflow, jax, etc.)
- Training script path (from Phase 1)
- Config file path (from Phase 1, if found)
- User-provided data paths (from Phase 0 Q10: `train_data_path`, `val_data_path`)
- User-specified environment manager (from Phase 0 Q11: `env_manager`, `env_name`)

## Step 0: Verify GitNexus (REQUIRED) and Index the Target Project

GitNexus is a **hard prerequisite** — on par with git and a working training command. It is **not optional** and there is **no grep/analyze fallback** for code understanding. Every downstream code-understanding agent (implement, research) relies on the GitNexus code graph to reason about the codebase before editing or adapting it.

**Step 0.1 — Check availability (BLOCKING):**
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py available
```
If the output reports `"available": false`, this is an **unrecoverable prerequisite failure** — treat it exactly like "Phase 2 failed blocks the pipeline". Do NOT continue and do NOT fall back to grep/analyze. Set `status: "failed"` and `ready_for_baseline: false`, record the failure in the report (`code_graph.available: false`), log an error event (see Error Tracking), and halt with these install instructions for the user:
```
GitNexus is a required dependency for ML-Optimizer and was not found on PATH.
Install it, then re-run:

  npm install -g gitnexus && gitnexus setup
```
`gitnexus setup` auto-registers the gitnexus MCP server for Claude Code (and also installs gitnexus's own global skills and PreToolUse/PostToolUse hooks into `~/.claude/`). If MCP registration needs to be done manually, the fallback is `claude mcp add --transport stdio --scope user gitnexus gitnexus mcp`.

**Step 0.2 — Check MCP-server registration (BEST-EFFORT WARNING, not a block):**
The CLI being on PATH does not guarantee the gitnexus MCP server is registered with Claude Code. Because querying the code graph is **MCP-only** (agents query via the `mcp__gitnexus__context` / `mcp__gitnexus__query` / `mcp__gitnexus__impact` tools — there is no CLI query path), an unregistered server means downstream agents cannot query the graph. After the CLI check passes, run:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py mcp-registered
```
Interpret the `{"registered": ...}` output:
- `false` → emit a **WARNING** (do NOT fail): the gitnexus CLI is installed but its MCP server is not registered with Claude Code, so downstream agents will not be able to query the code graph. Guide the user to run `gitnexus setup` (or the manual fallback `claude mcp add --transport stdio --scope user gitnexus gitnexus mcp`). A freshly-registered MCP server only becomes available after a Claude Code session restart. Proceed to Step 0.3.
- `null` → cannot determine (the `claude` CLI is not on PATH); proceed silently. The real failure, if any, is caught downstream as a hard error.
- `true` → good; proceed.

Record the probe result under `code_graph.mcp_registered` (`true`/`false`/`null`) in the prerequisites report (Step 7).

**Step 0.3 — Index the target project (BLOCKING):**
Once availability passes, index the target project **once** so the code graph is available to all downstream agents:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py index <project_root>
```
The wrapper runs `gitnexus analyze <project_root> --index-only`, which writes the code knowledge graph to `<project_root>/.gitnexus`. Indexing is **non-invasive**: `--index-only` keeps the index pure — it does NOT inject a GitNexus section into the project's `CLAUDE.md` / `AGENTS.md` and does NOT install `.claude/` skills, so the target project is never contaminated. If the output reports `"success": false`, treat it as a **prerequisite failure** (halt with the script's `error` text as repair guidance): set `status: "failed"`, `ready_for_baseline: false`, record `code_graph.target_indexed: false` in the report, log an error event, and do not proceed. Common repairs: ensure `gitnexus` runs cleanly on `<project_root>` (check the reported error), confirm disk space, then re-run the wrapper index command (optionally with `--force` to rebuild a stale index).

**Do NOT commit `<project_root>/.gitnexus`.** The code graph is a local build artifact — the wrapper auto-adds it to the repo git exclude on success; never `git add` it.

Record the GitNexus availability and target-index status in the prerequisites report (Step 7) under the `code_graph` key so downstream phases know the index exists.

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
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/prerequisites_check.py detect-format-project <project_root> <training_script>
```

This scans the training script AND any local modules it imports for data-loading patterns. It returns the expected format (image_folder, csv, hdf5, cifar, etc.), patterns found, data-related CLI arguments, and confidence level.

If confidence is "low" or format is "unknown":

Use the dataset as-is without format conversion. Log warning to dev_notes: "Unknown dataset format — using as-is". If the user is available, ask for clarification via AskUserQuestion:
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
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/prerequisites_check.py validate-data <path> <format>
```

Check that:
- Path exists and is readable
- Path is non-empty
- Data format matches what the training script expects

If the user said "embedded in code" (e.g., CIFAR10 auto-download), skip validation and note that data will be downloaded during training.

## Step 4: Prepare Dataset (If Needed)

If there's a format mismatch between user data and what the training script expects:

Use AskUserQuestion:
```
Your data appears to be in [detected format] but the training script expects [expected format].

Options:
1. Let me restructure the data into the expected format (I'll create a new folder, originals untouched)
2. Skip preparation — I'll use the data as-is
3. The data is already correct — the detection was wrong
```

If restructuring is requested:
1. Create `<exp_root>/prepared-data/` directory
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
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/prerequisites_check.py detect-env <project_root>
```

Compare the detected manager with the user's specified manager (`env_manager` from Phase 0):
- **Match:** Proceed as normal
- **Mismatch:**

  Use the detected environment manager instead. Log to dev_notes: "Env manager mismatch — using detected '<detected_manager>'". If the user is available, confirm via AskUserQuestion:
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

Auto-create the conda environment with `conda create -n <env_name> python=<detected_python_version> -y`. Log to dev_notes: "Auto-created conda env '<env_name>'". If auto-creation fails, use AskUserQuestion:
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
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/prerequisites_check.py scan-imports <project_root>
```

Then check which third-party packages are missing, using the user's Python executable:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/prerequisites_check.py check-packages '<third_party_json>' <python_executable>
```

Where `<python_executable>` is:
- For conda: the python inside the conda env (run `conda run -n <env_name> which python` to find it)
- For venv: `<venv_path>/bin/python`
- For system: `python3` (the default if omitted)

## Step 5.1: Bulk Install from Dependency Files

Before installing packages individually, check if the project has a dependency specification:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/prerequisites_check.py bulk-install-cmd <project_root> <env_manager> [env_name]
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
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/prerequisites_check.py gpu-install-cmd <package> [env_manager] [env_name]
```
Pass `<env_manager>` and `<env_name>` when using conda — the output command will be wrapped with `conda run --no-banner -n <env_name>` so the package installs into the correct environment. **Never use bare `pip install torch` or `pip install jax`** — these install CPU-only versions, causing silent performance failure on GPU machines.

For all other packages, install using the user's preferred manager:

| Manager | Install Command |
|---------|----------------|
| conda   | `conda install -y -n <env_name> <package>` (try `conda install -y -n <env_name> -c conda-forge <package>` if default fails) |
| uv      | `uv pip install <package>` |
| pip     | `pip install <package>` (or use the command from `gpu-install-cmd` for GPU packages) |
| poetry  | `poetry add <package>` |

**Note:** Some import names differ from pip package names. Use the `IMPORT_TO_PACKAGE` mapping in `${CLAUDE_PLUGIN_ROOT}/scripts/prerequisites_check.py` (e.g., `cv2` → `opencv-python`, `PIL` → `Pillow`, `yaml` → `PyYAML`).

After installation, re-run the package check to verify:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/prerequisites_check.py check-packages '<still_missing_json>'
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

Write `<exp_root>/results/prerequisites.json`:
```json
{
  "status": "ready|partial|failed",
  "code_graph": {
    "available": true|false,
    "mcp_registered": true|false|null,
    "target_indexed": true|false,
    "graph_path": "<project_root>/.gitnexus",
    "notes": "<any issues or info>"
  },
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

GitNexus is required: if `code_graph.available` is `false` or `code_graph.target_indexed` is `false`, `status` MUST be `failed` and `ready_for_baseline` MUST be `false`.

## Step 7.1: Validate Output

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/schema_validator.py \
  <exp_root>/results/prerequisites.json prerequisites
```

If validation fails, fix and re-validate before proceeding.

Append to `<exp_root>/dev_notes.md`:
```markdown
## <date> — Prerequisites Check

- **GitNexus:** [available/MISSING] — MCP server [registered/NOT registered/unknown] — target indexed at <project_root>/.gitnexus [yes/no]
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

- **GitNexus not available:** Unrecoverable — set `status: "failed"`, `ready_for_baseline: false`, halt with install instructions: `npm install -g gitnexus && gitnexus setup` (`gitnexus setup` auto-registers the gitnexus MCP server for Claude Code; manual MCP fallback `claude mcp add --transport stdio --scope user gitnexus gitnexus mcp`). No grep/analyze fallback.
- **GitNexus MCP server not registered:** Best-effort WARNING only (not a block) — the CLI is installed but agents cannot query the MCP-only code graph; guide the user to run `gitnexus setup` (or `claude mcp add --transport stdio --scope user gitnexus gitnexus mcp`) and restart the Claude Code session. Proceed.
- **GitNexus target indexing fails:** Unrecoverable — set `status: "failed"`, `ready_for_baseline: false`, halt with the script's `error` text as repair guidance.
- **Data path doesn't exist:** Set `status: "failed"`, report to user
- **Format detection unknown:** Ask user, fall back to "use as-is"
- **Package install fails:** Record error, classify as critical/non-critical
- **Permission errors:** Report and suggest user fix manually
- **No internet for auto-download datasets:** Warn that CIFAR10/MNIST etc. will need network access during training

## Error Tracking

At the following points, log an error event using the error tracker:

### When GitNexus is not available (REQUIRED dependency missing):
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"config_error","severity":"critical","source":"prerequisites","message":"GitNexus not installed (required dependency)","phase":2,"context":{"dependency":"gitnexus","install":"npm install -g gitnexus && gitnexus setup"}}'
```

### When GitNexus target-project indexing fails:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"config_error","severity":"critical","source":"prerequisites","message":"GitNexus failed to index target project: <error>","phase":2,"context":{"dependency":"gitnexus","project_root":"<project_root>","error":"<error>"}}'
```

### When data path doesn't exist or validation fails:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"resource_error","severity":"critical","source":"prerequisites","message":"Data path does not exist: <path>","phase":2,"context":{"path":"<path>","path_type":"<train|val>"}}'
```

### When data format validation fails:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"resource_error","severity":"warning","source":"prerequisites","message":"Data format validation failed: <reason>","phase":2,"context":{"format_detected":"<format>","validation_error":"<reason>"}}'
```

### When package installation fails:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"config_error","severity":"<critical|warning>","source":"prerequisites","message":"Package install failed: <package>","phase":2,"context":{"package":"<package>","manager":"<env_manager>","is_critical":<true|false>}}'
```

### When environment detection or setup fails:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py <exp_root> log '{"category":"resource_error","severity":"warning","source":"prerequisites","message":"Environment detection failed: <error>","phase":2,"context":{"env_manager":"<env_manager>","env_name":"<env_name>"}}'
```
