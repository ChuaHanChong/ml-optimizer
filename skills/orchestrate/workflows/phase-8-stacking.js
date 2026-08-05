export const meta = {
  name: "phase-8-stacking",
  description: "Sequential method-stacking accumulation. Merge improved methods one at a time into ml-opt/stack-N, run/assess each, skip-on-failure, optional code_evolution and narrowed HP-tune. Returns the compound-best stack branch and per-step ledger.",
  phases: [
    { title: "Pre-check", detail: "Skip entirely for non-git (file_backup) projects or when no stacking candidates exist." },
    { title: "Rank & init", detail: "Rank candidate methods by improvement (descending); the top method's own existing branch becomes the logical stack-1 base (no new branch created for it) -- the first real branch cut is ml-opt/stack-2." },
    { title: "Accumulate", detail: "For each next method: implement-agent merges it into ml-opt/stack-N in a worktree (resolving conflicts), experiment-agent runs it, analysis-agent assesses." },
    { title: "Repair interference", detail: "If the stacked gain trails the best individual gain, optionally run code_evolution (tuning evolve HPs + implement evolve skill); if >1% improvement, optional narrowed HP-tune." },
    { title: "Finalize", detail: "Return the last kept ml-opt/stack-N as the compound best with its metric and a per-step kept/skipped ledger." }
  ]
};

// ---------------------------------------------------------------------------
// Schemas (align with scripts/schema_validator.py experiment-result + relay shapes)
// ---------------------------------------------------------------------------

// implement-agent merge result for one stack step
const STACK_MERGE_SCHEMA = {
  type: "object",
  required: ["status"],
  properties: {
    status: { type: "string", enum: ["merged_clean", "merged_resolved", "conflict_unresolved", "validation_failed"] },
    branch: { type: "string" },
    conflicting_files: { type: "array", items: { type: "string" } },
    notes: { type: "string" }
  }
};

// experiment-agent result for one stacked run (overlaps with EXPERIMENT_RESULT_SCHEMA
// in phase-7-experiment.js, plus stacking-specific fields: round_dir, code_branches,
// stacking_order, stack_base_exp)
const STACK_EXPERIMENT_SCHEMA = {
  type: "object",
  required: ["exp_id", "status", "metrics"],
  properties: {
    exp_id: { type: "string" },
    status: { type: "string", enum: ["completed", "failed", "diverged", "timeout", "pending"] },
    round_dir: { type: "string" },
    metrics: { type: "object" },
    code_branch: { type: "string" },
    code_branches: { type: "array", items: { type: "string" } },
    stacking_order: { type: "integer" },
    stack_base_exp: { type: "string" },
    method_tier: { type: "string" },
    notes: { type: "string" }
  }
};

// analysis-agent decision for a stacked result (mirrors analyze_to_tuning relay)
const STACK_ANALYSIS_SCHEMA = {
  type: "object",
  required: ["recommendation"],
  properties: {
    recommendation: { type: "string", enum: ["continue", "code_evolution", "stop"] },
    pivot_type: { type: "string" },
    best_metric_value: { type: "number" },
    interference_detected: { type: "boolean" },
    reasoning: { type: "string" }
  }
};

// code-reviewer / silent-failure-hunter verdict for a merged stack branch — reviewed at the
// merge boundary before any GPU time (mirrors the Phase 6 per-branch review).
const STACK_REVIEW_SCHEMA = {
  type: "object",
  required: ["critical", "summary"],
  properties: {
    critical: { type: "boolean", description: "true if the merge introduced a real bug or a metric-invalidating silent failure" },
    summary: { type: "string" },
    findings: { type: "array", items: { type: "string" } }
  }
};

// tuning-agent evolve-HP recommendation
const EVOLVE_HP_SCHEMA = {
  type: "object",
  required: ["evolve_recommendation"],
  properties: {
    evolve_recommendation: {
      type: "object",
      required: ["num_generations", "population_size"],
      properties: {
        num_generations: { type: "integer" },
        population_size: { type: "integer" },
        reasoning: { type: "string" }
      }
    }
  }
};

// implement-agent evolve-skill result
const EVOLVE_RESULT_SCHEMA = {
  type: "object",
  required: ["status"],
  properties: {
    status: { type: "string", enum: ["validated", "validation_failed", "shinkaevolve_unavailable"] },
    branch: { type: "string" },
    notes: { type: "string" }
  }
};

// tuning-agent narrowed HP proposal batch
const HP_TUNE_SCHEMA = {
  type: "object",
  required: ["configs"],
  properties: {
    configs: {
      type: "array",
      items: {
        type: "object",
        required: ["exp_id", "config"],
        properties: {
          exp_id: { type: "string" },
          config: { type: "object" },
          reasoning: { type: "string" }
        }
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Helpers (pure — no time/random)
// ---------------------------------------------------------------------------

function primaryMetricOf(result, primaryMetric) {
  // NO first-numeric-metric fallback: a result missing the primary metric is
  // null (isBetter(null, ...) === false -> safe skip), never a random metric.
  if (!result || !result.metrics) return null;
  const v = result.metrics[primaryMetric];
  if (typeof v === "number" && isFinite(v)) return v;
  log(`primary_metric "${primaryMetric}" missing or non-numeric in result ${result.exp_id || "?"} — treating as null (safe skip).`);
  return null;
}

// Is `candidate` better than `current` under the optimization direction?
function isBetter(candidate, current, lowerIsBetter) {
  if (candidate === null || !isFinite(candidate)) return false;
  if (current === null || !isFinite(current)) return true;
  return lowerIsBetter ? candidate < current : candidate > current;
}

// Percent improvement of `value` over `baseline`, sign-normalized so positive = better.
function improvementPct(value, baseline, lowerIsBetter) {
  if (value === null || baseline === null || baseline === 0) return 0;
  const raw = ((value - baseline) / Math.abs(baseline)) * 100;
  return lowerIsBetter ? -raw : raw;
}

// Per-step completeness verifier (SubagentStop 'verify' role moved into the script, since
// workflow-dispatched agents may skip that hook). Confirms the kept experiment's outputs +
// the step analysis file exist via scripts/output_contract.py. Logs loud on a gap; non-fatal.
async function verifyStackStep(expRoot, plugin, keptExpId, stepOrder) {
  const VERIFY_SCHEMA = {
    type: "object",
    required: ["all_present"],
    properties: {
      all_present: { type: "boolean" },
      missing: { type: "array", items: { type: "string" } }
    }
  };
  const res = await _dispatch(
    [
      `Completeness check for kept stacking step ${stepOrder}. Use Bash/Read ONLY (do NOT re-run anything).`,
      `Find the round directory holding result exp_id "${keptExpId}" under ${expRoot}/results/ (a round-*-stacked dir), then verify the experiment-agent outputs:`,
      `python3 ${plugin}/scripts/output_contract.py check ${expRoot} experiment-agent --round-dir <that round dir> --exp-id ${keptExpId}`,
      `(exit code 2 means missing outputs — capture its "missing" array). Also confirm ${expRoot}/reports/batch-stack-${stepOrder}-analysis.md exists.`,
      `Collect every missing path into a flat "missing" array. If anything is missing, log it once:`,
      `python3 ${plugin}/scripts/error_tracker.py ${expRoot} log '{"category":"agent_failure","severity":"warning","source":"orchestrate","message":"stack step ${stepOrder} missing required outputs","phase":8}'.`,
      `Do NOT fabricate outputs. Return {all_present, missing}.`
    ].join(" "),
    { schema: VERIFY_SCHEMA, agentType: "ml-optimizer:experiment-agent", label: `stack-verify-${stepOrder}`, phase: "Accumulate" }
  );
  if (!res || res.all_present !== true) {
    const missing = res && Array.isArray(res.missing) ? res.missing : ["<verifier failed>"];
    log(`Phase 8 stack step ${stepOrder} completeness WARNING — missing outputs: ${missing.join(", ")}.`);
  }
  return res;
}

// ---------------------------------------------------------------------------
// Workflow body
// ---------------------------------------------------------------------------

phase("Pre-check");

// Workflow runtime may deliver `args` as a JSON string — parse-if-string so
// destructured fields + arg reads don't silently become undefined.
const A = typeof args === 'string' ? JSON.parse(args) : (args || {})
// opts.model override for this run only (via args.model_override); every workflow-internal
// dispatch routes through _dispatch so a run-scoped model override applies uniformly without
// touching each agentType call site's opts literal.
const _dispatch = (prompt, opts) => agent(prompt, A.model_override ? { ...opts, model: A.model_override } : opts)
const {
  exp_root,
  project_root,
  primary_metric,
  lower_is_better,
  stacking_candidates,
  scope_level,
  baseline_metric,
  divergence_metric,
  divergence_lower_is_better,
  model_category
} = A;

const lowerIsBetter = !!lower_is_better;
const PLUGIN = "${CLAUDE_PLUGIN_ROOT}";
// Remote GPU host, same contract as phase-7: stacked runs must execute on the machine the
// experiments they are compared against ran on, or the comparison is across hardware.
const remote = A.remote && A.remote.host ? A.remote : null;
const remoteBlock = remote
  ? `
REMOTE EXECUTION — the GPUs are on another machine. Do NOT run training locally.
Wrap the training command as the experiment skill's "Running on a remote GPU host" section describes:
  \${CLAUDE_PLUGIN_ROOT}/scripts/remote_train.sh --host ${remote.host} --workdir ${remote.workdir}/<exp_id> --gpu <gpu_id> --env-python ${remote.env_python} --sync <worktree path> -- <the training command>
Pass the GPU via --gpu, not a local CUDA_VISIBLE_DEVICES. Query free GPUs with \`python3 \${CLAUDE_PLUGIN_ROOT}/scripts/gpu_check.py 30 80 ${remote.host}\`.`
  : "";
// code_evolution (ShinkaEvolve) only at scope_level "full" (mirrors phase-7's gate) — an
// "architecture"-scope run never triggers ShinkaEvolve.
const codeEvolutionEnabled = scope_level === "full";
// Training budget — the SAME cap baseline/phase-7 used so stacked metrics stay comparable
// (read via A.*; a legacy scalar `budget` is tolerated as seconds).
const fixedTimeBudget =
  A.fixed_time_budget != null ? A.fixed_time_budget
    : typeof A.budget === "number" ? A.budget
    : (A.budget && A.budget.fixed_time_budget) || null;
const fixedEpochBudget =
  A.fixed_epoch_budget != null ? A.fixed_epoch_budget
    : (A.budget && A.budget.fixed_epoch_budget) || null;
const fixedStepBudget = A.fixed_step_budget != null ? A.fixed_step_budget : null;
const budgetClause =
  fixedStepBudget != null
    ? `Training budget: cap this run at fixed_step_budget=${fixedStepBudget} environment timesteps — map it to the framework's timestep flag (e.g. --total_timesteps), the SAME cap the baseline used so metrics are comparable. Record step_budget in the result JSON.`
    : fixedTimeBudget != null
    ? `Cap this run at fixed_time_budget=${fixedTimeBudget} seconds (wrap training with \`timeout\` or a framework-native time limit) — the SAME cap the baseline used so metrics are comparable.`
    : fixedEpochBudget != null
    ? `Cap this run at fixed_epoch_budget=${fixedEpochBudget} epochs — the SAME cap the baseline used so metrics are comparable.`
    : `No fixed training budget — use the natural training schedule.`;

const modelCategory = model_category || null;
// metric-routing: divergence uses divergence_metric (as phase-7); analysis + isBetter use primary_metric.
const divergenceClause =
  divergence_metric === null || divergence_metric === undefined
    ? "Divergence monitoring is DISABLED (tabular ML / null divergence_metric) — do not run detect_divergence; rely on the hard timeout."
    : `Divergence metric: ${divergence_metric} (lower_is_better=${divergence_lower_is_better}). You own in-run monitoring: poll your training log with detect_divergence.py every 5 minutes (model_category=${modelCategory} for thresholds). status="diverged" (NaN/Inf/explosion/collapse) -> kill the run and mark status="diverged". status="plateaued" (plateau/drift) -> NEVER kill; record the warning in notes for analysis.`;

// Skip non-git projects: stacking needs branch merges. Guard defensively even though the
// orchestrator only launches this for git projects — file_backup surfaces as zero candidates.
const candidates = (stacking_candidates || [])
  .filter(Boolean)
  .filter((c) => c && typeof c.branch === "string" && c.branch.length > 0);

if (candidates.length === 0) {
  log("Phase 8 skipped: no git-branch stacking candidates (file_backup project or nothing improved over baseline).");
  return {
    best_stack_branch: null,
    best_stack_metric: null,
    steps: []
  };
}

// Rank by improvement_pct descending — largest improvement stacks first.
const ranked = candidates
  .slice()
  .sort((a, b) => (b.improvement_pct || 0) - (a.improvement_pct || 0));

log(`Phase 8 stacking: ${ranked.length} candidate methods ranked by improvement.`);
ranked.forEach((c, i) => log(`  rank ${i + 1}: ${c.branch} (+${(c.improvement_pct || 0).toFixed(2)}% over baseline)`));

// Best individual gain — the bar a stacked combo must clear to avoid interference repair.
const bestIndividualGain = ranked[0].improvement_pct || 0;

phase("Rank & init");

// Seed the stack with the strongest method. No experiment needed: its existing best
// result is the stack baseline. Hold the current best stack in script variables.
let stackOrder = 1;
let stackBaseBranch = ranked[0].branch; // logical base = first method's branch as ml-opt/stack-1
// Derive the seed (best-single) method's absolute metric from its known improvement over baseline,
// so the FIRST accumulation is compared against it instead of being force-kept. Null if we can't
// derive it (no baseline_metric passed) — then the old "first re-run always kept" behavior stands.
const seedMetric =
  typeof baseline_metric === "number" && typeof ranked[0].improvement_pct === "number"
    ? baseline_metric + (lowerIsBetter ? -1 : 1) * (ranked[0].improvement_pct / 100) * Math.abs(baseline_metric)
    : null;
let stackBaseMetric = seedMetric;       // best-single-method metric when derivable, else null
let stackBaseExp = null;
const stackedMethods = [ranked[0].branch];

// The seed "stack-1" is logically the strongest method's own branch — no new branch is
// cut for it (the first accumulation merges into it, producing ml-opt/stack-2). Record the
// real branch, not a phantom ml-opt/stack-1 that never gets created.
const steps = [
  { method: ranked[0].branch, branch: ranked[0].branch, kept: true }
];

log(`Initialized stack-1 from ${ranked[0].branch} (best individual: +${bestIndividualGain.toFixed(2)}%).`);

phase("Accumulate");

// Accumulate the remaining methods one at a time.
for (let i = 1; i < ranked.length; i++) {
  if (budget.remaining() <= 0) {
    log("Budget exhausted; stopping stacking accumulation.");
    break;
  }

  const method = ranked[i];
  stackOrder += 1;
  const stackBranch = `ml-opt/stack-${stackOrder}`;

  log(`Stack step ${stackOrder}: merge ${method.branch} into ${stackBaseBranch} -> ${stackBranch}`);

  // ---- Step a/b/c/d: implement-agent merges next method into a new stack branch ----
  const merge = await _dispatch(
    [
      `Method-stacking merge — stack step ${stackOrder}.`,
      `Project root: ${project_root}. exp_root: ${exp_root}.`,
      `Create branch ${stackBranch} from ${stackBaseBranch} (git checkout -b ${stackBranch} ${stackBaseBranch}),`,
      `then merge the next method branch ${method.branch} into it (git merge ${method.branch} --no-ff --no-edit).`,
      `Work inside an isolated git worktree outside ${exp_root}/ so the main tree is untouched.`,
      `Already stacked (must all be preserved): ${stackedMethods.join(", ")}.`,
      `If the merge is clean -> status "merged_clean".`,
      `If there are conflicts, resolve them so BOTH the existing stack and ${method.branch} keep their functionality`,
      `(git diff --name-only --diff-filter=U for the conflict set), commit, and report status "merged_resolved".`,
      `If conflicts cannot be reconciled, git merge --abort and report "conflict_unresolved".`,
      `Then validate (syntax, imports, a forward/smoke pass); on failure report "validation_failed".`,
      `Inspect the touched files (Read / Grep) to keep the merge surgical.`,
      `Return the merge status, the branch name, any conflicting_files, and notes.`
    ].join(" "),
    {
      agentType: "ml-optimizer:implement-agent",
      isolation: "worktree",
      schema: STACK_MERGE_SCHEMA,
      label: `stack-merge-${stackOrder}`,
      phase: "Accumulate"
    }
  );

  if (!merge || merge.status === "conflict_unresolved" || merge.status === "validation_failed") {
    log(`Stack step ${stackOrder}: ${method.branch} skipped (${merge ? merge.status : "merge failed"}).`);
    steps.push({ method: method.branch, branch: stackBranch, kept: false });
    // Keep stackOrder as-is: the merge agent already ran `git checkout -b ${stackBranch}`,
    // so this number is spent. Reusing it would collide on the next attempt. Gaps are harmless.
    continue;
  }

  // ---- Step e: review the merged branch (mirror Phase 6 per-branch review) ----
  // The merge boundary is where silent failures sneak in: a conflict resolution that drops a
  // NaN/CUDA guard, or two methods both wrapping the loss. Review BEFORE spending GPU time so a
  // corrupted-metrics experiment never runs; a critical finding skips this stack step.
  const stackReviewCtx = `Review the merged stacking branch ${stackBranch} in project_root ${project_root}. Diff it against the prior stack base ${stackBaseBranch} AND the newly merged method ${method.branch} to focus on the MERGE BOUNDARY (combining: ${stackedMethods.concat([method.branch]).join(", ")}). This is an ML research stack, not production code.`;
  const stackReviews = (await parallel([
    () => _dispatch(
      `${stackReviewCtx} Focus: bugs, logic errors, incorrect tensor/shape handling, and whether the merge preserved BOTH methods' functionality. Set critical=true for a real bug that would corrupt training/eval. Return the schema.`,
      { agentType: "pr-review-toolkit:code-reviewer", label: `stack-review-code-${stackOrder}`, phase: "Accumulate", schema: STACK_REVIEW_SCHEMA }),
    () => _dispatch(
      `${stackReviewCtx} Focus: silent failures introduced by the merge/conflict resolution — swallowed NaN/Inf losses, failed CUDA/optimizer ops that fall through, bare except/except:pass around training or eval, or a resolution that dropped one method's error handling. Any of these silently invalidates the stacked metric, so set critical=true. Return the schema.`,
      { agentType: "pr-review-toolkit:silent-failure-hunter", label: `stack-review-silent-${stackOrder}`, phase: "Accumulate", schema: STACK_REVIEW_SCHEMA }),
  ])).filter(Boolean);
  if (stackReviews.some((r) => r && r.critical === true)) {
    const why = stackReviews.map((r) => r && r.summary).filter(Boolean).join(" | ");
    log(`Stack step ${stackOrder}: ${method.branch} skipped — merge review flagged a critical issue: ${why}`);
    steps.push({ method: method.branch, branch: stackBranch, kept: false });
    continue;
  }

  // ---- Step e2: the experiment-agent creates the stacking round (round_manager.py
  // create-round stacked --branch <stackBranch>) as a side effect and writes under it. ----

  // ---- Step f: experiment-agent runs the stacked branch ----
  const codeBranches = stackedMethods.concat([method.branch]);

  const exp = await _dispatch(
    [
      `Run stacking experiment for ${stackBranch} (stack step ${stackOrder}).`,
      `Project root: ${project_root}. exp_root: ${exp_root}.`,
      `Before writing results, create a stacked round:`,
      `python3 \${CLAUDE_PLUGIN_ROOT}/scripts/round_manager.py ${exp_root} create-round stacked --branch ${stackBranch}`,
      `REQUIRED OUTPUTS (experiment-agent, mirrors scripts/output_contract.py — ALL must exist before you finish, under that round dir <RD>):`,
      `results/<RD>/exp-*.json (the result JSON; PreToolUse hook BLOCKS any other path), logs/<RD>/<exp_id>/train.log,`,
      `scripts/<RD>/<exp_id>/ (training script dir), artifacts/<RD>/<exp_id>/ (checkpoint dir).`,
      `Code branch: ${stackBranch}. Method tier: stacked_default_hp.`,
      `Code branches (combined): ${JSON.stringify(codeBranches)}. Stacking order: ${stackOrder}.`,
      `The stacked result JSON MUST include code_branches and stacking_order (completeness fields for stacked results).`,
      stackBaseExp ? `Stack base exp: ${stackBaseExp}.` : `This is the first re-run of an accumulated stack.`,
      `Primary metric: ${primary_metric}. Run inside an isolated git worktree on ${stackBranch} using \`--detach\` (parallel runs on the same branch must not collide). ${budgetClause} ${divergenceClause}`,
      `After the run completes, register and close the round via round_manager.py.`,
      `Return exp_id, status, metrics, code_branch, code_branches, stacking_order, stack_base_exp, method_tier, notes.`,
      remoteBlock
    ].join(" "),
    {
      agentType: "ml-optimizer:experiment-agent",
      isolation: "worktree",
      schema: STACK_EXPERIMENT_SCHEMA,
      label: `stack-exp-${stackOrder}`,
      phase: "Accumulate"
    }
  );

  if (!exp || exp.status !== "completed") {
    log(`Stack step ${stackOrder}: experiment on ${stackBranch} did not complete (${exp ? exp.status : "no result"}); skipping ${method.branch}.`);
    steps.push({ method: method.branch, branch: stackBranch, kept: false });
    continue;
  }

  const stackedMetric = primaryMetricOf(exp, primary_metric);

  // ---- Step g: analysis-agent assesses the stacked result ----
  const analysis = await _dispatch(
    [
      `Assess a stacked experiment result.`,
      `exp_root: ${exp_root}. project_root: ${project_root}.`,
      `batch_number: stack-${stackOrder}. primary_metric: ${primary_metric}. lower_is_better: ${lowerIsBetter}. scope_level: ${scope_level} (only recommend code_evolution when scope_level=="full").`,
      `Stacked branch: ${stackBranch}. Methods combined: ${JSON.stringify(codeBranches)}. Stacking order: ${stackOrder}.`,
      `Stacked ${primary_metric}: ${stackedMetric}. Best individual gain over baseline: ${bestIndividualGain.toFixed(2)}%.`,
      `REQUIRED OUTPUT (analysis-agent, mirrors scripts/output_contract.py): write your assessment to ${exp_root}/reports/batch-stack-${stackOrder}-analysis.md (this file MUST exist before you finish — it satisfies the analysis-agent batch-analysis contract).`,
      `Read the round's result JSON and the latest reports/batch-*-analysis.md for context.`,
      `Compare the stacked gain to the best individual method gain. If the stack UNDERPERFORMS the`,
      `best individual method (methods interfering), recommend code_evolution. If the stacked result`,
      `improved over the current stack base, recommend continue. If the combination is unproductive`,
      `and cannot be fixed, recommend stop.`,
      `Return recommendation (continue|code_evolution|stop), pivot_type, best_metric_value,`,
      `interference_detected, and reasoning.`
    ].join(" "),
    {
      agentType: "ml-optimizer:analysis-agent",
      schema: STACK_ANALYSIS_SCHEMA,
      label: `stack-analyze-${stackOrder}`,
      phase: "Accumulate"
    }
  );

  const recommendation = analysis ? analysis.recommendation : "stop";

  // Regression / unproductive -> skip this method, keep the previous stack base.
  // First re-run (stackBaseMetric === null) is always kept if it completed.
  const improvedOverBase = stackBaseMetric === null
    ? true
    : isBetter(stackedMetric, stackBaseMetric, lowerIsBetter);

  if (recommendation === "stop" || !improvedOverBase) {
    log(`Stack step ${stackOrder}: ${method.branch} skipped (${recommendation === "stop" ? "analysis stop" : "regression vs current stack"}).`);
    steps.push({ method: method.branch, branch: stackBranch, kept: false });
    continue;
  }

  // Method is kept. Promote it to the current stack base.
  let keptBranch = stackBranch;
  let keptMetric = stackedMetric;
  let keptExp = exp.exp_id;
  stackedMethods.push(method.branch);

  // ---- Step: interference repair via code_evolution when stacked gain < best individual ----
  // Scope-gated: ShinkaEvolve only at scope_level "full" (mirrors phase-7); a narrower scope
  // skips the repair and keeps the merged stack.
  if (recommendation === "code_evolution" && !codeEvolutionEnabled) {
    log(`Stack step ${stackOrder}: code_evolution recommended but scope_level=${scope_level} != "full" — skipping ShinkaEvolve repair, keeping the merged stack.`);
  } else if (recommendation === "code_evolution") {
    phase("Repair interference");
    log(`Stack step ${stackOrder}: interference detected; attempting code_evolution on ${stackBranch}.`);

    // (1) tuning-agent proposes evolve HPs
    const evolveHp = await _dispatch(
      [
        `Propose ShinkaEvolve evolution HPs for a stacking code_evolution step.`,
        `exp_root: ${exp_root}. project_root: ${project_root}.`,
        `Stacked methods: ${JSON.stringify(stackedMethods)}. Stacked branch: ${stackBranch}.`,
        `Read learned-behaviors.json category "evolve_hp" for prior outcomes.`,
        `Return evolve_recommendation: { num_generations, population_size, reasoning }.`,
        `Defaults if no prior signal: 5 generations, population 2 (stacking is narrower than Phase 7 evolve).`
      ].join(" "),
      {
        agentType: "ml-optimizer:tuning-agent",
        schema: EVOLVE_HP_SCHEMA,
        label: `stack-evolve-hp-${stackOrder}`,
        phase: "Repair interference"
      }
    );

    const evolveRec = evolveHp && evolveHp.evolve_recommendation
      ? evolveHp.evolve_recommendation
      : { num_generations: 5, population_size: 2, reasoning: "default stacking evolve HPs" };

    // (2) implement-agent runs the evolve skill on the stacked branch
    const evolved = await _dispatch(
      [
        `Run ShinkaEvolve code evolution on a stacked branch via Skill("ml-optimizer:evolve").`,
        `project_root: ${project_root}. exp_root: ${exp_root}.`,
        `parent_branch: ${stackBranch}. parent_metrics: ${JSON.stringify(exp.metrics)}.`,
        `primary_metric: ${primary_metric}. lower_is_better: ${lowerIsBetter}. scope_level: ${scope_level}.`,
        `evolve_recommendation: ${JSON.stringify(evolveRec)}.`,
        `feedback_context: before evolving, read the dead-end catalog (python3 ${PLUGIN}/scripts/error_tracker.py ${exp_root} dead-end list), recurring error patterns (python3 ${PLUGIN}/scripts/error_tracker.py ${exp_root} patterns), the most recent ${exp_root}/reports/batch-*-analysis.md, and ${exp_root}/learned-behaviors.json — pass them as feedback_context: {dead_ends, error_patterns, batch_analysis, learned_behaviors} per the evolve skill's own Input Parameters contract. Do NOT propose mutations matching a dead_ends technique.`,
        `Goal: optimize code-level interactions between the stacked methods ${JSON.stringify(stackedMethods)}.`,
        `Commit the best mutation as ml-opt/evolved-stack-${stackOrder}.`,
        `If ShinkaEvolve is unavailable return status "shinkaevolve_unavailable".`,
        `Return status, branch, notes.`
      ].join(" "),
      {
        agentType: "ml-optimizer:implement-agent",
        isolation: "worktree",
        schema: EVOLVE_RESULT_SCHEMA,
        label: `stack-evolve-${stackOrder}`,
        phase: "Repair interference"
      }
    );

    if (evolved && evolved.status === "validated" && evolved.branch) {
      // Re-run the evolved branch and keep it only if it beats the pre-evolution stack.
      const evolvedExp = await _dispatch(
        [
          `Run experiment on an evolved stacked branch.`,
          `Project root: ${project_root}. exp_root: ${exp_root}.`,
          `Create a stacked round (round_manager.py create-round stacked --branch ${evolved.branch}).`,
          `REQUIRED OUTPUTS (experiment-agent — ALL must exist before you finish, under that round dir <RD>):`,
          `results/<RD>/exp-*.json, logs/<RD>/<exp_id>/train.log, scripts/<RD>/<exp_id>/, artifacts/<RD>/<exp_id>/.`,
          `Code branch: ${evolved.branch}. Method tier: stacked_default_hp.`,
          `Code branches (combined): ${JSON.stringify(codeBranches)}. Stacking order: ${stackOrder}.`,
          `The result JSON MUST include code_branches and stacking_order.`,
          `Primary metric: ${primary_metric}. Run inside an isolated git worktree on ${evolved.branch} using \`--detach\`. ${budgetClause} ${divergenceClause}`,
          `Register and close the round when done. Return exp_id, status, metrics, code_branch,`,
          `code_branches, stacking_order, method_tier, notes.`
        ].join(" "),
        {
          agentType: "ml-optimizer:experiment-agent",
          isolation: "worktree",
          schema: STACK_EXPERIMENT_SCHEMA,
          label: `stack-evolved-exp-${stackOrder}`,
          phase: "Repair interference"
        }
      );

      if (evolvedExp && evolvedExp.status === "completed") {
        const evolvedMetric = primaryMetricOf(evolvedExp, primary_metric);
        if (isBetter(evolvedMetric, keptMetric, lowerIsBetter)) {
          log(`Stack step ${stackOrder}: evolved branch ${evolved.branch} improved over pre-evolution stack; promoting.`);
          keptBranch = evolved.branch;
          keptMetric = evolvedMetric;
          keptExp = evolvedExp.exp_id;
        } else {
          log(`Stack step ${stackOrder}: evolved branch did not beat pre-evolution stack; discarding evolution.`);
        }
      } else {
        log(`Stack step ${stackOrder}: evolved experiment did not complete; keeping pre-evolution stack.`);
      }
    } else {
      log(`Stack step ${stackOrder}: evolution unavailable or failed (${evolved ? evolved.status : "no result"}); continuing without it.`);
    }
  }

  // ---- Optional narrowed HP-tune when the combo shows >1% improvement ----
  // Gate on the kept stack's gain vs the prior stack base when known, else this method's
  // candidate improvement signal (first accumulated re-run).
  const combinedGainPct = stackBaseMetric === null
    ? (method.improvement_pct || bestIndividualGain)
    : improvementPct(keptMetric, stackBaseMetric, lowerIsBetter);

  if (combinedGainPct > 1 && budget.remaining() > 0) {
    log(`Stack step ${stackOrder}: combo gain ${combinedGainPct.toFixed(2)}% > 1%; running narrowed HP-tune on ${keptBranch}.`);

    const hp = await _dispatch(
      [
        `Propose a narrowed HP-tune batch for a stacked branch (1 small round).`,
        `exp_root: ${exp_root}. project_root: ${project_root}.`,
        `code_branches: ${JSON.stringify([keptBranch])}. primary_metric: ${primary_metric}.`,
        `lower_is_better: ${lowerIsBetter}. iteration: 1.`,
        `Narrow the search space around the best HPs of the stacked methods — do not re-explore widely.`,
        `Return configs: [{ exp_id, config, reasoning }].`
      ].join(" "),
      {
        agentType: "ml-optimizer:tuning-agent",
        schema: HP_TUNE_SCHEMA,
        label: `stack-hptune-${stackOrder}`,
        phase: "Repair interference"
      }
    );

    const cfgs = hp && Array.isArray(hp.configs) ? hp.configs : [];
    if (cfgs.length > 0) {
      const hpRuns = await parallel(
        cfgs.map((cfg) => () =>
          _dispatch(
            [
              `Run a narrowed stacked HP-tune experiment.`,
              `Project root: ${project_root}. exp_root: ${exp_root}.`,
              `Create a stacked round (round_manager.py create-round stacked --branch ${keptBranch}).`,
              `REQUIRED OUTPUTS (experiment-agent — ALL must exist before you finish, under that round dir <RD>):`,
              `results/<RD>/exp-*.json, logs/<RD>/<exp_id>/train.log, scripts/<RD>/<exp_id>/, artifacts/<RD>/<exp_id>/.`,
              `Code branch: ${keptBranch}. Method tier: stacked_tuned_hp.`,
              `Code branches (combined): ${JSON.stringify(codeBranches)}. Stacking order: ${stackOrder}.`,
              `The result JSON MUST include code_branches and stacking_order.`,
              `exp_id: ${cfg.exp_id}. Config: ${JSON.stringify(cfg.config)}.`,
              `Primary metric: ${primary_metric}. Run inside an isolated git worktree on ${keptBranch} using \`--detach\` (these HP-tune runs execute in parallel on the SAME branch, so --detach is required to avoid a worktree checkout collision). ${budgetClause} ${divergenceClause}`,
              `Register and close the round when done. Return exp_id, status, metrics, code_branch,`,
              `code_branches, stacking_order, method_tier, notes.`
            ].join(" "),
            {
              agentType: "ml-optimizer:experiment-agent",
              isolation: "worktree",
              schema: STACK_EXPERIMENT_SCHEMA,
              label: `stack-hp-exp-${stackOrder}-${cfg.exp_id}`,
              phase: "Repair interference"
            }
          )
        )
      );

      for (const r of hpRuns.filter(Boolean)) {
        if (r.status !== "completed") continue;
        const tunedMetric = primaryMetricOf(r, primary_metric);
        if (isBetter(tunedMetric, keptMetric, lowerIsBetter)) {
          log(`Stack step ${stackOrder}: tuned HP ${r.exp_id} improved the stack; promoting.`);
          keptMetric = tunedMetric;
          keptExp = r.exp_id;
          // keptBranch unchanged — tuned HPs run on the same branch.
        }
      }
    }
  }

  // Commit this step into the running best stack.
  stackBaseBranch = keptBranch;
  stackBaseMetric = keptMetric;
  stackBaseExp = keptExp;
  steps.push({ method: method.branch, branch: keptBranch, kept: true });
  log(`Stack step ${stackOrder}: KEPT ${method.branch} -> ${keptBranch} (metric ${keptMetric}).`);

  // Completeness check for this kept step's outputs (SubagentStop 'verify' role).
  await verifyStackStep(exp_root, PLUGIN, keptExp, stackOrder);
}

phase("Finalize");

log(`Stacking complete. Final stack branch: ${stackBaseBranch}. Kept ${steps.filter((s) => s.kept).length}/${steps.length} steps.`);

return {
  best_stack_branch: stackBaseBranch,
  best_stack_metric: stackBaseMetric,
  steps
};
