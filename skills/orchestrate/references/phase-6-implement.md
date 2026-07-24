# Phase 6: Implement Research Proposals

**Phase gate:** Run `pipeline_state.py <exp_root> gate 5 6` before entering. On completion: `pipeline_state.py <exp_root> log-gate 6 completed "<summary>"`.

If the user selected research proposals that require code changes (not just HP tuning), Phase 6 runs as a **dynamic workflow**:

1. **Launch the implement workflow** — build the args and call it once. The workflow dispatches `ml-optimizer:implement-agent` internally (worktree-isolated, one `ml-opt/<slug>` branch per proposal, implement skill Step 4) and also runs the post-implementation reviewers internally (see step 6):
   ```
   result = Workflow({
     scriptPath: "${CLAUDE_PLUGIN_ROOT}/skills/orchestrate/workflows/phase-6-implement.js",
     args: {
       exp_root,
       project_root,
       findings_path: "<exp_root>/reports/research-findings.md",
       selected_indices,
       strategy: "git_branch"   # or "file_backup" for non-git projects
     }
   })
   ```
   The workflow writes `results/implementation-manifest.json` + the git branches and returns:
   ```
   {
     manifest_path: "<exp_root>/results/implementation-manifest.json",
     branches: [{slug, branch, status, validation, reviews}, ...]
   }
   ```

2. **Check results** from `result.branches` / `<exp_root>/results/implementation-manifest.json`:
   - **All validated** → proceed to experiment loop with branch-aware execution
   - **Some failed validation** → inform user, proceed with validated proposals only
   - **All failed** → fall back to HP-tuning only (no code changes)

3. **If new dependencies flagged** → Use AskUserQuestion to confirm install:
   ```
   The following new dependencies are needed for the research proposals:
   - <package>: required by <proposal_name>

   Install them? (The experiment will fail without them.)
   ```
   Auto-approve dependency installation if the user is not available. Log to error tracker: `category: "pipeline_inefficiency", severity: "info", source: "orchestrate", message: "Auto-approved installation of [packages]"`.

4. **If license warnings flagged** → Use AskUserQuestion to surface to user:
   ```
   The following proposals adapted code from reference repositories with license concerns:
   - <proposal_name>: <license_warning details>

   Please review before proceeding. Continue with these proposals?
   ```
   Auto-accept license warnings if the user is not available. Log to error tracker: `category: "pipeline_inefficiency", severity: "warning", source: "orchestrate", message: "Auto-accepted license warnings for [proposals]"`. Log to dev_notes for user review later.

5. **If conflicts detected** → Inform user which proposals touch the same files. Each is on its own branch, so experiments run independently, but merging winners later may need manual conflict resolution.

6. **Post-implementation quality review (run inside the workflow):**
   The phase-6 workflow dispatches two reviewers per validated implementation branch (in parallel, via `agentType`) to catch problems before running experiments on broken implementations — the results are folded into each branch's `reviews` field in the returned manifest:
   - `pr-review-toolkit:code-reviewer` — bugs, logic errors, and general code quality issues.
   - `pr-review-toolkit:silent-failure-hunter` — swallowed errors, inadequate error handling, and inappropriate fallbacks. Especially important for ML code: silently-caught NaN losses, failed CUDA/optimizer ops that fall through, or `except: pass` around training/eval steps will corrupt experiment results without surfacing.

   The workflow applies the findings:
   - Only review proposals with `status: "validated"` in the manifest
   - If either reviewer flags a critical issue (a real bug, or a silent failure that would invalidate metrics), the proposal is marked `validation_failed` and skipped
   - If a reviewer flags minor issues (style, non-blocking), they are logged to dev_notes and the workflow proceeds

   After the workflow returns, read the per-branch `reviews` field; surface any `validation_failed` branches to the user with the reason.

7. **Test coverage check (run inside the workflow):**
   The implement-agent writes a focused unit test per proposal (`<exp_root>/tests/test_<slug>.py`, implement skill step 4f). For validated proposals whose `validation.unit_tests` is `"pass"`, the workflow dispatches `pr-review-toolkit:pr-test-analyzer` on the test + the changed files to assess whether the test actually exercises the new behavior (not a token/placeholder test) and to surface missing edge cases.
   - This is **advisory** — weak coverage does NOT block experimentation (these are ML proposals, not production code), but the findings are logged to dev_notes and inform later analysis.
   - Skip when `validation.unit_tests` is `"skipped"` (no meaningful test to analyze).
