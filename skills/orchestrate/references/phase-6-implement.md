# Phase 6: Implement Research Proposals

**Phase gate:** Run `pipeline_state.py <exp_root> gate 5 6` before entering. On completion: `pipeline_state.py <exp_root> log-gate 6 completed "<summary>"`.

If the user selected research proposals that require code changes (not just HP tuning):

1. **Dispatch the implement agent:**
   ```
   Agent(
     description: "Implement research proposals",
     prompt: "Ultrathink. Implement research proposals. Parameters: findings_path: <exp_root>/reports/research-findings.md, selected_indices: {selected_indices}, project_root: {project_root}.",
     subagent_type: "ml-optimizer:implement-agent"
   )
   ```
   → Save the returned agentId to `agent_registry["implement"]`
   → Persist registry: `save_state(..., agent_registry=agent_registry)`

2. **Check results** from `<exp_root>/results/implementation-manifest.json`:
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

6. **Post-implementation quality review:**
   For validated proposals, dispatch `feature-dev:code-reviewer` to review each implementation branch for bugs, logic errors, and code quality issues. This catches problems before running experiments on broken implementations.
   - Only review proposals with `status: "validated"` in the manifest
   - If the reviewer flags critical issues, mark the proposal as `validation_failed` and skip it
   - If the reviewer flags minor issues (style, non-blocking), log them but proceed
