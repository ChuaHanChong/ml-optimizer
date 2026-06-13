# Phase 2: Prerequisites Check

**Phase gate:** Run `pipeline_state.py <exp_root> gate 1 2` before entering. On completion: `pipeline_state.py <exp_root> log-gate 2 completed "<summary>"`.

Dispatch the prerequisites agent:
```
Agent(
  description: "Check prerequisites",
  prompt: "Check prerequisites for ML project. Parameters: project_root: {project_root}, framework: {framework}, training_script: {training_script}, config_path: {config_path}, train_data_path: {train_data_path}, val_data_path: {val_data_path}, env_manager: {env_manager}, env_name: {env_name}.",
  subagent_type: "ml-optimizer:prerequisites-agent"
)
```

**Check results** from `<exp_root>/results/prerequisites.json`:
- `ready_for_baseline = true` → proceed to Phase 3
- `status = "partial"` → log warnings to `<exp_root>/dev_notes.md`: "Prerequisites partial — proceeding anyway." Proceed to Phase 3.
- `status = "failed"` → Classify the failure reason from `prerequisites.json`:
  - **GitNexus missing or target index failed** (`code_graph.available = false` or `code_graph.target_indexed = false`): **Unrecoverable BLOCK** — GitNexus is a hard prerequisite (on par with "Phase 2 failed blocks the pipeline"); there is no grep/analyze fallback for code understanding. Do NOT proceed to Phase 3. Surface the install/repair guidance to the user:
    ```
    npm install -g gitnexus && gitnexus setup
    ```
    `gitnexus setup` auto-registers the gitnexus MCP server for Claude Code; the manual MCP fallback is `claude mcp add --transport stdio --scope user gitnexus gitnexus mcp`. If `code_graph.available` is true but `code_graph.target_indexed` is false, relay the `code_graph.notes` / error text as repair guidance (re-run the wrapper index: `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/gitnexus_utils.py index <project_root>`, optionally with `--force` to rebuild a stale index). Halt until the user installs/fixes GitNexus and re-runs Phase 2.
  - **GitNexus MCP server not registered** (`code_graph.available = true`, `code_graph.mcp_registered = false`): **NOT a block** — `status` will not be `failed` for this alone. The CLI is installed but the MCP server is not registered with Claude Code; because querying the code graph is MCP-only, downstream agents (implement, research) will be unable to query it. Surface a warning and guide the user to run `gitnexus setup` (or `claude mcp add --transport stdio --scope user gitnexus gitnexus mcp`) and restart the Claude Code session so the MCP tools load. Log to dev_notes and proceed. (`code_graph.mcp_registered = null` means the check could not run — the `claude` CLI is absent — proceed silently.)
  - **Data path invalid / not found:** Attempt auto-recovery: search the project for data files, check the training script for auto-download patterns (CIFAR, MNIST, HuggingFace `load_dataset`). If a plausible path is found, update `train_data_path`/`val_data_path` and re-run Phase 2. If not found: BLOCK with AskUserQuestion.
  - **Dependency install failed:** Retry install once with `--no-deps`, then check if import still fails. If still fails: BLOCK with AskUserQuestion.
  - **Environment not found:** If env_manager is conda, auto-create the environment. If creation also fails: BLOCK.
  - **Dry-run failed:** Log error and attempt Phase 3 anyway (baseline may succeed where dry-run failed). If baseline also fails, exit via Phase 3 failure path.
  - **All other failures:** BLOCK with AskUserQuestion (unrecoverable without user input).
  - Log all auto-recovery attempts to dev_notes and error tracker with `category: "config_error", severity: "warning", source: "orchestrate"`.

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
