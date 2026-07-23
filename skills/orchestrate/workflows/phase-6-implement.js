export const meta = {
  name: 'phase-6-implement',
  description: 'Phase 6: implement selected research proposals as ml-opt/<slug> git branches (one worktree-isolated implement-agent per proposal, in parallel), review each branch (code-reviewer + silent-failure-hunter), then assemble results/implementation-manifest.json.',
  phases: [
    { title: 'Implement', detail: 'pipeline: one worktree-isolated implement-agent per selected proposal -> manifest entry' },
    { title: 'Review', detail: 'per branch: code-reviewer + silent-failure-hunter in parallel; downgrade on critical findings' },
    { title: 'Assemble', detail: 'merge entries into results/implementation-manifest.json and validate the schema' },
  ],
}

// -------------------- args (CONTRACT phase-6-implement) --------------------
// { exp_root, project_root, findings_path, selected_indices:[int], strategy }
const expRoot = args.exp_root
const projectRoot = args.project_root
const findingsPath = args.findings_path || `${expRoot}/reports/research-findings.md`
const selectedIndices = Array.isArray(args.selected_indices) ? args.selected_indices : []
const strategy = args.strategy || 'git_branch'
const manifestPath = `${expRoot}/results/implementation-manifest.json`

// file_backup (non-git) projects share one working tree (restore-before-apply validation against
// a restored baseline), so they must run strictly sequentially; git projects get one worktree per
// branch and run concurrently. isGit drives BOTH the isolation flag and parallel-vs-sequential.
const isGit = strategy === 'git_branch'

// -------------------- structured return schemas --------------------
// One manifest entry per implemented proposal (matches implementation-manifest.json proposal shape).
const VALIDATION_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    syntax: { enum: ['pass', 'fail'] },
    import: { enum: ['pass', 'fail'] },
    model_instantiate: { enum: ['pass', 'fail', 'skipped'] },
    forward_pass: { enum: ['pass', 'fail', 'skipped'] },
    unit_tests: { enum: ['pass', 'fail', 'skipped'] },
  },
}

const MANIFEST_ENTRY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['name', 'slug', 'status'],
  properties: {
    index: { type: 'number', description: '1-based proposal index from research-findings.md' },
    name: { type: 'string' },
    slug: { type: 'string' },
    branch: { type: 'string', description: 'ml-opt/<slug> (git) or backup tag (file_backup)' },
    status: { enum: ['validated', 'validation_failed', 'implementation_error', 'preflight_failed'] },
    files_modified: { type: 'array', items: { type: 'string' } },
    files_created: { type: 'array', items: { type: 'string' } },
    complexity: { type: 'string' },
    implementation_strategy: { enum: ['from_scratch', 'from_reference'] },
    reference_repo: { type: 'string' },
    reference_files_used: { type: 'array', items: { type: 'string' } },
    adaptation_notes: { type: 'string' },
    license_warning: { type: ['string', 'null'] },
    validation: VALIDATION_SCHEMA,
    test_file: { type: ['string', 'null'] },
    explanation: { type: 'string' },
    commit_sha: { type: 'string' },
    new_dependencies: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
}

const REVIEW_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['critical', 'summary'],
  properties: {
    critical: { type: 'boolean', description: 'true if a real bug or metric-invalidating silent failure was found' },
    summary: { type: 'string', description: 'concise findings' },
    findings: { type: 'array', items: { type: 'string' } },
  },
}

const ASSEMBLE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['manifest_path', 'validated_count', 'total'],
  properties: {
    manifest_path: { type: 'string' },
    validated_count: { type: 'number' },
    total: { type: 'number' },
    schema_valid: { type: 'boolean' },
    new_dependencies: { type: 'array', items: { type: 'string' } },
    conflicts: { type: 'array', items: {} },
  },
}

const IMPLEMENT_AGENT = 'ml-optimizer:implement-agent'
const CODE_REVIEWER = 'feature-dev:code-reviewer'
const SILENT_HUNTER = 'pr-review-toolkit:silent-failure-hunter'
const TEST_ANALYZER = 'pr-review-toolkit:pr-test-analyzer'

const baseCtx = `You are the ml-optimizer implement-agent running inside the Phase 6 implementation workflow.
exp_root: ${expRoot}
project_root: ${projectRoot}
findings_path: ${findingsPath}
strategy: ${strategy}
Respect optimization-goals.json scope_level — do NOT make changes outside the proposal's scope.`

// Output contract for the implement-agent (mirrors output_contract.py "implement-agent").
// Workflow-dispatched agents may skip the SubagentStart/SubagentStop hooks, so the assemble
// prompt states the output path and the completeness check below verifies the manifest on disk.
const OUTPUT_CONTRACT = `REQUIRED OUTPUT (implement-agent, assemble step):
  ${manifestPath}  — validated proposal branches manifest (PreToolUse hook BLOCKS writes to the wrong path/schema).
${isGit ? 'Each validated proposal must also leave a persisted ml-opt/<slug> git branch.' : 'Each validated proposal must leave a restore-before-apply file_backup of its modified state.'}
Before finishing, verify ${manifestPath} exists and passes schema_validator.py manifest with the Read tool; re-write it if missing or invalid.`

if (!selectedIndices.length) {
  log('Phase 6: no selected_indices — nothing to implement.')
  return { manifest_path: manifestPath, branches: [] }
}

// -------------------- Phase 1+2: implement -> review per proposal --------------------
phase('Implement')

// stage 1 implements one proposal (worktree-isolated for git so parallel implements don't collide);
// stage 2 reviews that proposal's branch. git = concurrent pipeline; file_backup = sequential (below).
const stageImplement = (_prev, idx, i) => agent(
  `${baseCtx}

Implement EXACTLY ONE proposal: index ${idx} from ${findingsPath} (use the implement skill).
- Parse this single proposal (implement_utils.py with selected_indices [${idx}]).
- ${isGit
    ? 'git strategy: implement inside the implement skill Step 3.1 worktree (outside exp_root). Create branch ml-opt/<slug> off the original branch, edit, validate (syntax/import/LSP), write a focused unit test, commit, then remove the worktree leaving the branch.'
    : 'file_backup strategy (non-git): restore the baseline backup, apply ONLY this proposal\'s changes, validate, then back up the modified state (restore-before-apply). Do NOT touch other proposals\' files.'}
- For from_reference proposals, scope edits using implement_utils.py analyze / grep on the cloned repo and the target project.
- Run pre-flight file-existence validation; if all target files are missing, return status "preflight_failed".
Return ONE manifest entry (the schema) describing the outcome for this proposal, including index=${idx}, slug, branch, status, validation block, files_modified/created, test_file, explanation, commit_sha. Do NOT write the manifest file (the workflow assembles it).`,
  { agentType: IMPLEMENT_AGENT, isolation: isGit ? 'worktree' : undefined, label: `implement:idx-${idx}`, phase: 'Implement', schema: MANIFEST_ENTRY_SCHEMA })

phase('Review')

// stage 2: only review proposals that produced a branch (status validated). Reviews run on the
// persisted branch via git (no shared mutable working tree) so no worktree isolation is needed.
const stageReview = (entry, idx) => {
  if (!entry) return null
  if (entry.status !== 'validated' || !entry.branch) return entry // nothing to review; pass through
  const branch = entry.branch
  // Strategy-aware review target: git projects diff the branch; file_backup projects have NO git
  // branch (entry.branch is a backup tag), so review the modified files directly.
  const touched = JSON.stringify((entry.files_modified || []).concat(entry.files_created || []))
  const reviewCtx = isGit
    ? `Review the ml-optimizer implementation on git branch "${branch}" in project_root ${projectRoot} (diff it against the original branch). This is an ML research proposal implementation, not production code.`
    : `Review the ml-optimizer implementation for proposal "${entry.slug}" in project_root ${projectRoot}. This is a file_backup / non-git project — there is NO git branch; inspect the modified files DIRECTLY (Read/Grep): ${touched}. This is an ML research proposal implementation, not production code.`
  // code-reviewer + silent-failure-hunter GATE status; pr-test-analyzer is ADVISORY (test
  // coverage) and runs only when the implement-agent wrote a test_file — it never blocks.
  const hasTest = !!entry.test_file
  const thunks = [
    () => agent(
      `${reviewCtx}
Focus: bugs, logic errors, incorrect tensor/shape handling, and general correctness. Report whether any finding is CRITICAL (a real bug that would corrupt training/eval results). Return the schema.`,
      { agentType: CODE_REVIEWER, label: `review:code:${entry.slug}`, phase: 'Review', schema: REVIEW_SCHEMA }),
    () => agent(
      `${reviewCtx}
Focus: silent failures — swallowed NaN/Inf losses, failed CUDA/optimizer ops that fall through, bare except/except:pass around training or eval, inappropriate fallbacks that hide errors. Any of these invalidates experiment metrics silently, so flag them CRITICAL. Return the schema.`,
      { agentType: SILENT_HUNTER, label: `review:silent:${entry.slug}`, phase: 'Review', schema: REVIEW_SCHEMA }),
  ]
  if (hasTest) {
    thunks.push(() => agent(
      `${reviewCtx}
ADVISORY test-coverage review of the unit test ${entry.test_file}. Assess whether it actually exercises the new behavior from this proposal (not a placeholder/token test) and covers the key edge cases. This is advisory only — set critical=false regardless. Return the schema.`,
      { agentType: TEST_ANALYZER, label: `review:test:${entry.slug}`, phase: 'Review', schema: REVIEW_SCHEMA }))
  }
  return parallel(thunks).then((reviews) => {
    const got = (reviews || []).filter(Boolean)
    const testRes = hasTest ? reviews[2] : null // advisory, excluded from the critical gate
    const gating = got.filter((r) => r && r !== testRes)
    const critical = gating.some((r) => r && r.critical === true)
    const reviewNotes = gating.map((r) => r && r.summary).filter(Boolean)
    if (testRes && testRes.summary) reviewNotes.push(`test-coverage (advisory): ${testRes.summary}`)
    // Downgrade a branch whose review found a metric-invalidating bug so the experiment
    // loop skips it (mirrors phase-6 reference step 6).
    const status = critical ? 'validation_failed' : entry.status
    return {
      ...entry,
      status,
      reviews: got,
      notes: [entry.notes, critical ? `Review flagged critical issue(s): ${reviewNotes.join(' | ')}` : ''].filter(Boolean).join(' '),
    }
  })
}

let entries
if (isGit) {
  // git: worktree-isolated implements run concurrently; each is reviewed as it finishes.
  entries = (await pipeline(selectedIndices, stageImplement, stageReview)).filter(Boolean)
} else {
  // file_backup: proposals share the one working tree, so implement+review STRICTLY one at a time.
  // A concurrent pipeline would let one agent's restore-before-apply wipe another's changes.
  entries = []
  for (let i = 0; i < selectedIndices.length; i++) {
    const idx = selectedIndices[i]
    const entry = await stageImplement(null, idx, i)
    entries.push(await stageReview(entry, idx))
  }
  entries = entries.filter(Boolean)
}
log(`Phase 6: ${entries.length} proposal(s) processed; ${entries.filter((e) => e.status === 'validated').length} validated after review.`)

// -------------------- Phase 3: assemble + validate manifest --------------------
phase('Assemble')

// A final implement-agent merges all per-proposal entries into the canonical manifest and
// validates the schema — it owns the exact implementation-manifest.json format.
const assemble = await agent(
  `${baseCtx}

${OUTPUT_CONTRACT}

Assemble the implementation manifest from these per-proposal entries (already implemented + reviewed; reviews already applied to status):
${JSON.stringify(entries, null, 2)}

1. Write ${manifestPath} in the implementation-manifest.json format (implement skill Step 5): top-level original_branch, strategy="${strategy}", proposals (one per entry above — keep their index/slug/branch/status/validation/files_modified/etc., drop the transient "reviews" field but fold critical review summaries into each proposal's notes), conflicts (proposals touching the same files), and new_dependencies (union across proposals).
2. Validate it: python3 \${CLAUDE_PLUGIN_ROOT}/scripts/schema_validator.py ${manifestPath} manifest — fix and re-validate until it passes.
3. Verify the file exists with Read.
Return the schema: manifest_path, validated_count (proposals with status "validated"), total, schema_valid, new_dependencies, conflicts.`,
  { agentType: IMPLEMENT_AGENT, label: 'implement:assemble-manifest', phase: 'Assemble', schema: ASSEMBLE_SCHEMA })

// -------------------- Completeness check (SubagentStop 'verify' role) --------------------
// Workflow-dispatched agents may skip the SubagentStop hook, so verify the manifest exists on disk
// (and, for git, that validated branches resolve). If missing, one repair attempt; else fail loudly.
const validatedBranches = entries
  .filter((e) => e.status === 'validated' && e.branch)
  .map((e) => e.branch)

const VERIFY_SCHEMA = {
  type: 'object',
  required: ['manifest_exists'],
  properties: {
    manifest_exists: { type: 'boolean' },
    schema_valid: { type: 'boolean' },
    missing_branches: { type: 'array', items: { type: 'string' } },
    repaired: { type: 'boolean' },
  },
}

const verify = await agent(
  `${baseCtx}

Completeness check for Phase 6 implementation output. Use Read/Bash ONLY (do NOT re-implement proposals):
1. Check whether ${manifestPath} exists and passes \`python3 \${CLAUDE_PLUGIN_ROOT}/scripts/schema_validator.py ${manifestPath} manifest\`.
2. ${isGit
    ? `For each of these validated branches verify it resolves: ${JSON.stringify(validatedBranches)} via \`git -C ${projectRoot} rev-parse --verify <branch>\`. Collect any that do NOT resolve into missing_branches.`
    : 'file_backup strategy: skip git branch verification; set missing_branches=[].'}
3. If ${manifestPath} is MISSING, re-assemble it from these entries: ${JSON.stringify(entries)} (same implementation-manifest.json format as the assemble step), validate it, set repaired=true. If it merely fails schema validation, fix and re-validate, set repaired=true.
4. Log any gap: \`python3 \${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py ${expRoot} log '{"category":"agent_failure","severity":"warning","source":"orchestrate","message":"implementation-manifest.json missing or invalid after assemble","phase":6}'\`.
Return {manifest_exists, schema_valid, missing_branches, repaired}.`,
  { agentType: IMPLEMENT_AGENT, label: 'implement:verify-manifest', phase: 'Assemble', schema: VERIFY_SCHEMA })

if (!verify || verify.manifest_exists !== true) {
  log(`Phase 6 completeness FAILED: ${manifestPath} could not be confirmed (verifier returned ${JSON.stringify(verify)}).`)
} else if (verify.repaired) {
  log(`Phase 6 completeness: ${manifestPath} was missing/invalid and has been repaired.`)
}
if (verify && Array.isArray(verify.missing_branches) && verify.missing_branches.length) {
  log(`Phase 6 completeness WARNING: validated branches that do not resolve: ${verify.missing_branches.join(', ')}.`)
}

// CONTRACT phase-6 return: { manifest_path, branches:[{slug,branch,status,validation,reviews}] }
const branches = entries.map((e) => ({
  slug: e.slug,
  branch: e.branch || null,
  status: e.status,
  validation: e.validation || null,
  reviews: e.reviews || [],
}))

return {
  manifest_path: (assemble && assemble.manifest_path) || manifestPath,
  branches,
}
