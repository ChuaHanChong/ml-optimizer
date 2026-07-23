export const meta = {
  name: "phase-7-experiment",
  description:
    "Autonomous experiment loop: per round, tuning-agent proposes HP configs, experiment-agents run in parallel (worktree-isolated, --detach) on their code branches with divergence handled per-batch, analysis-agent writes batch-N-analysis.md and returns a decision, the script acts on the decision tree (continue/branch_test/hp_expand/narrow_space/regularization/stop/method_proposal/code_evolution). On stop it runs the stuck protocol (research-agent) and the fixpoint exit judgment. Returns best experiment + stacking candidates that beat baseline.",
  phases: [
    { title: "Pre-loop", detail: "Validate baseline integrity, load implementation manifest, seed code_branches + search space, read agenda/dead-ends context files." },
    { title: "Round", detail: "round_manager create-round; tuning-agent proposes configs reading agenda/dead-ends/prior batch-N-analysis.md." },
    { title: "Experiments", detail: "parallel experiment-agent runs (worktree, --detach) per code_branch; experiment agent runs detect_divergence.py; optional monitor-agent per batch." },
    { title: "Analysis", detail: "analysis-agent writes reports/batch-N-analysis.md and returns a structured decision over the primary_metric." },
    { title: "Decision", detail: "continue/pivots loop; stop triggers stuck protocol (research-agent) + fixpoint check; method_proposal triggers research+implement; code_evolution triggers tuning(evolve HPs)+implement(evolve skill)." },
    { title: "Exit", detail: "Return best_exp_id/best_metric/rounds_completed/exit_reason and stacking_candidates (branches that beat baseline)." },
  ],
};

// Schemas (aligned with scripts/schema_validator.py); the runtime validates agent() returns against these.

const HP_CONFIG_SCHEMA = {
  type: "object",
  properties: {
    round_dir: { type: "string" },
    recommendation: { type: "string" }, // hp-tune's own early signal: "continue" | "stop"
    research_requested: { type: ["boolean", "null"] }, // hp-tune: no research-derived priors exist yet — route one research round
    configs: {
      type: "array",
      items: {
        type: "object",
        properties: {
          exp_id: { type: "string" },
          config: { type: "object" },
          gpu_id: { type: ["integer", "string", "null"] },
          code_branch: { type: ["string", "null"] },
          code_proposal: { type: ["string", "null"] },
          proposal_source: { type: ["string", "null"] },
          method_tier: { type: ["string", "null"] },
          iteration: { type: ["integer", "null"] },
          checkpoint_source: { type: ["object", "null"] },
          warm_started: { type: ["boolean", "null"] },
        },
        required: ["exp_id", "config"],
      },
    },
  },
  required: ["configs"],
};

const EXPERIMENT_RESULT_SCHEMA = {
  type: "object",
  properties: {
    exp_id: { type: "string" },
    status: { type: "string" }, // completed | failed | diverged | timeout
    metrics: { type: "object" },
    config: { type: "object" },
    code_branch: { type: ["string", "null"] },
    method_tier: { type: ["string", "null"] },
    duration_seconds: { type: ["number", "null"] },
    primary_metric_value: { type: ["number", "null"] },
    diverged: { type: ["boolean", "null"] },
    notes: { type: ["string", "null"] },
    result_path: { type: ["string", "null"] },
  },
  required: ["exp_id", "status"],
};

const ANALYSIS_DECISION_SCHEMA = {
  type: "object",
  properties: {
    decision: { type: "string" }, // continue | branch_test | hp_expand | narrow_space | regularization | stop | method_proposal | code_evolution
    pivot_type: { type: ["string", "null"] },
    batch_number: { type: "integer" },
    best_exp_id: { type: ["string", "null"] },
    best_metric_value: { type: ["number", "null"] },
    improved_since_last_stop: { type: ["boolean", "null"] },
    search_space: { type: ["object", "null"] }, // narrowed/expanded space to hand to next tuning round
    branch_scores: { type: ["object", "null"] },
    correlations: { type: ["object", "null"] },
    untested_branches: { type: ["array", "null"], items: { type: "string" } },
    surviving_branches: { type: ["array", "null"], items: { type: "string" } },
    max_batch_size: { type: ["integer", "null"] },
    reason: { type: ["string", "null"] },
    analysis_path: { type: ["string", "null"] },
    stacking_candidates: {
      type: ["array", "null"],
      items: {
        type: "object",
        properties: {
          branch: { type: "string" },
          improvement_pct: { type: "number" },
        },
        required: ["branch"],
      },
    },
  },
  required: ["decision", "batch_number"],
};

const RESEARCH_PROPOSALS_SCHEMA = {
  type: "object",
  properties: {
    findings_path: { type: "string" },
    proposals: {
      type: "array",
      items: {
        type: "object",
        properties: {
          index: { type: "integer" },
          title: { type: "string" },
          impact: { type: ["number", "null"] },
          confidence: { type: ["number", "null"] },
          feasibility: { type: ["number", "null"] },
          scope: { type: ["string", "null"] },
          type: { type: ["string", "null"] }, // hp_only | code_change
          implementation_strategy: { type: ["string", "null"] },
          files_to_modify: { type: ["array", "null"], items: { type: "string" } },
        },
        required: ["index", "title"],
      },
    },
    new_in_scope_count: { type: ["integer", "null"] },
    agenda_has_untried: { type: ["boolean", "null"] },
  },
  required: ["proposals"],
};

const MANIFEST_BRANCHES_SCHEMA = {
  type: "object",
  properties: {
    manifest_path: { type: "string" },
    branches: {
      type: "array",
      items: {
        type: "object",
        properties: {
          slug: { type: ["string", "null"] },
          branch: { type: "string" },
          status: { type: "string" },
          type: { type: ["string", "null"] },
        },
        required: ["branch", "status"],
      },
    },
  },
  required: ["branches"],
};

const ROUND_SCHEMA = {
  type: "object",
  properties: { round_dir: { type: "string" } },
  required: ["round_dir"],
};

const EVOLVE_HP_SCHEMA = {
  type: "object",
  properties: {
    num_generations: { type: ["integer", "null"] },
    population_size: { type: ["integer", "null"] },
    reasoning: { type: ["string", "null"] },
  },
};

const EVOLVE_RESULT_SCHEMA = {
  type: "object",
  properties: {
    status: { type: "string" }, // validated | validation_failed | shinkaevolve_unavailable
    branch: { type: ["string", "null"] },
    best_combined_score: { type: ["number", "null"] },
    notes: { type: ["string", "null"] },
  },
  required: ["status"],
};

// Workflow body

// Workflow runtime may deliver `args` as a JSON string — parse-if-string so
// destructured fields + arg reads don't silently become undefined.
const A = typeof args === 'string' ? JSON.parse(args) : (args || {})
const {
  exp_root,
  project_root,
  baseline,
  primary_metric,
  divergence_metric,
  lower_is_better,
  target_value,
  scope_level,
  hp_batches_per_round,
  method_proposal_scope,
  method_proposal_iterations,
} = A;

const PLUGIN = "${CLAUDE_PLUGIN_ROOT}";
const lowerIsBetter = lower_is_better === true;
const divergenceLowerIsBetter = A.divergence_lower_is_better;
const cadence = Number(hp_batches_per_round) || 3;
// Secondary metrics: [{name, lower_is_better, role}] from Phase 0 — threaded into the analysis Step 2.3 guardrail check.
const secondaryMetrics = Array.isArray(A.secondary_metrics) ? A.secondary_metrics : [];

// Seed replicates: pre-authorized at Phase 4 — hp-tune proposes each config seeds_per_config times with distinct random_seed values.
const seedsPerConfig = A.seeds_per_config || null;

// Training budget — the SAME fixed cap the baseline used, so experiment metrics stay comparable.
// Read via A.* and NEVER destructured as `budget`: that identifier is the runtime's token-budget
// global the loop uses to self-bound (budget.remaining()). Accept typed fields; also tolerate a
// legacy scalar `budget` (seconds). Forwarded into every experiment-agent prompt via budgetClause.
const fixedTimeBudget =
  A.fixed_time_budget != null
    ? A.fixed_time_budget
    : typeof A.budget === "number"
    ? A.budget
    : (A.budget && A.budget.fixed_time_budget) || null;
const fixedEpochBudget =
  A.fixed_epoch_budget != null
    ? A.fixed_epoch_budget
    : (A.budget && A.budget.fixed_epoch_budget) || null;
const fixedStepBudget = A.fixed_step_budget != null ? A.fixed_step_budget : null;
const budgetClause =
  fixedStepBudget != null
    ? `Training budget: cap this run at fixed_step_budget=${fixedStepBudget} environment timesteps — map it to the framework's timestep flag (e.g. --total_timesteps), the SAME cap the baseline used so metrics are comparable. Record step_budget in the result JSON.`
    : fixedTimeBudget != null
    ? `Training budget: cap this run at fixed_time_budget=${fixedTimeBudget} seconds — wrap training with \`timeout\` (or a framework-native time limit, e.g. Lightning --max_time / HF TrainingArguments), the SAME cap the baseline used so metrics are comparable. Record time_budget_seconds in the result JSON.`
    : fixedEpochBudget != null
    ? `Training budget: cap this run at fixed_epoch_budget=${fixedEpochBudget} epochs — the SAME cap the baseline used so metrics are comparable.`
    : `No fixed training budget — use the natural training schedule (the hard timeout baseline_training_time*3 still applies).`;
// scope gating per the analysis decision tree:
//   "training"     -> HP-only: no research / code_evolution pivots
//   "architecture" -> research enabled, ShinkaEvolve disabled
//   "full"         -> everything, incl. code_evolution
const methodProposalsEnabled =
  !!method_proposal_scope && scope_level !== "training";
const codeEvolutionEnabled = scope_level === "full";

const baselineMetricValue =
  baseline && baseline.metrics ? baseline.metrics[primary_metric] : null;

phase("Pre-loop");

// Pre-loop setup runs through an experiment-agent acting as a CLI runner: verify baseline
// integrity, sync behavioral memory, read the manifest + agenda/dead-ends, and produce the
// initial code_branches + search space. Context flows via args + files under exp_root.
const PRELOOP_SCHEMA = {
  type: "object",
  properties: {
    baseline_ok: { type: "boolean" },
    strategy: { type: "string" }, // git_branch | file_backup | hp_only
    sequential: { type: "boolean" }, // true for file_backup (no parallel runs)
    code_branches: { type: "array", items: { type: "string" } },
    search_space: { type: "object" },
    model_category: { type: ["string", "null"] },
    num_gpus: { type: ["integer", "null"] },
    warm_start_enabled: { type: ["boolean", "null"] },
    halt: { type: ["boolean", "null"] }, // baseline checksum mismatch -> hard stop
    halt_reason: { type: ["string", "null"] },
    original_branch: { type: ["string", "null"] }, // baseline/original git branch (excluded from stacking)
  },
  required: ["baseline_ok", "strategy", "code_branches", "search_space"],
};

const pre = await agent(
  `Pre-loop setup for the Phase 7 experiment loop. You are a CLI runner — run the commands below and return the structured result. Do NOT start experiments.

exp_root: ${exp_root}
project_root: ${project_root}
primary_metric: ${primary_metric}

Steps:
1. Verify baseline integrity: run \`python3 ${PLUGIN}/scripts/pipeline_state.py ${exp_root} verify-baseline\`. If exit code is non-zero (checksum mismatch), set halt=true, halt_reason to the message, and STOP — do not continue. Log: \`python3 ${PLUGIN}/scripts/error_tracker.py ${exp_root} log '{"category":"config_error","severity":"critical","source":"orchestrate","message":"Baseline integrity check FAILED","phase":7}'\`. A legacy/no-checksum warning is fine — continue.
2. Sync behavioral memory: \`python3 ${PLUGIN}/scripts/goal_memory.py ${exp_root} sync-from-errors\`.
3. Load implementation manifest at ${exp_root}/results/implementation-manifest.json if it exists. Collect proposals with status=="validated"; skip validation_failed/implementation_error. For each validated branch, verify it exists via \`git -C ${project_root} rev-parse --verify <branch>\`; drop missing ones (log to error_tracker). Always include the baseline (original) branch for HP-only comparison, and return its name as original_branch (the current branch of ${project_root} via \`git -C ${project_root} rev-parse --abbrev-ref HEAD\`, or null for non-git). If manifest strategy=="file_backup", set sequential=true and strategy="file_backup". If no manifest exists, set strategy="hp_only", code_branches=[] (HP-only on current code).
4. Read the research agenda and dead-end catalog for context (no action, just confirm they exist): \`python3 ${PLUGIN}/scripts/error_tracker.py ${exp_root} agenda list\` and \`... dead-end list\`.
5. Build the initial search_space dict for HP tuning from baseline config (${exp_root}/results/baseline.json) — sensible ranges around the baseline lr / batch_size / weight_decay etc. For tree-based frameworks (sklearn/XGBoost/LightGBM) prioritize max_depth / n_estimators. When model_category=rl, seed ranges from baseline.json's captured RL HPs — gamma, clip_range, ent_coef, n_steps — around the baseline values; do NOT apply supervised defaults.
6. Detect num_gpus via \`python3 ${PLUGIN}/scripts/gpu_check.py\` and model_category from baseline.json if present.

Return baseline_ok, strategy, sequential, code_branches, search_space, model_category, num_gpus, warm_start_enabled, halt, halt_reason, original_branch.`,
  { schema: PRELOOP_SCHEMA, agentType: "ml-optimizer:experiment-agent", phase: "Pre-loop", label: "phase7-preloop" }
);

if (!pre || pre.halt) {
  log(`Phase 7 halted before loop: ${pre ? pre.halt_reason : "pre-loop agent failed"}`);
  return {
    best_exp_id: null,
    best_metric: baselineMetricValue,
    rounds_completed: 0,
    exit_reason: pre ? "baseline_integrity_halt" : "preloop_failed",
    stacking_candidates: [],
  };
}

let codeBranches = Array.isArray(pre.code_branches) ? pre.code_branches : [];
let searchSpace = pre.search_space || {};
const strategy = pre.strategy || "hp_only";
const sequential = strategy === "file_backup" || pre.sequential === true;
// model_category: orchestrator-supplied user_choices value wins; baseline.json (pre-loop) is the fallback.
const modelCategory = A.model_category || pre.model_category || null;
const originalBranch = pre.original_branch || (baseline && baseline.original_branch) || null;
const numGpus = Math.max(1, Number(pre.num_gpus) || 1);
const warmStartEnabled = pre.warm_start_enabled === true;
// CPU-bound parallelism: Phase-4-authorized experiments_per_gpu (default 1) multiplies the
// per-round batch size; each experiment then gets a CPU-core slice via env_vars.
const experimentsPerGpu = Math.max(1, Number(A.experiments_per_gpu) || 1);
// number of configs per batch = numGpus * experiments_per_gpu; file_backup forces 1 at a time.
const configsPerBatch = sequential ? 1 : numGpus * experimentsPerGpu;
const cpuClause =
  experimentsPerGpu > 1
    ? `experiments_per_gpu=${experimentsPerGpu}: multiple experiments share each GPU. Slice CPU cores per experiment — pass env_vars={"OMP_NUM_THREADS": max(1, floor(nproc / ${configsPerBatch}))} to generate_script (optionally pin with taskset) so parallel runs do not thrash the CPU.`
    : "";

// Loop state ------------------------------------------------------------------
let iteration = 0;
let batchNumber = 0;
let batchesSinceLastResearch = 0;
// Budget cap on mid-loop research/evolution rounds — method_proposal, code_evolution, and cadence
// research all draw from this pool. Phase 4 pre-authorizes it; 0 disables mid-loop proposals. When
// exhausted, costly pivots fall back to hp_expand instead of running unbounded.
const methodProposalBudget = Number(method_proposal_iterations) || 0;
let methodProposalRoundsUsed = 0;
const methodProposalBudgetLeft = () => methodProposalRoundsUsed < methodProposalBudget;
let researchRoundIndex = 0;

let bestExpId = null;
let bestMetric = baselineMetricValue;
let lastStopBestMetric = baselineMetricValue;
let branchScores = {};
let correlations = {};
let maxBatchSize = null; // OOM constraint fed back into tuning

let consecutiveStopCount = 0;
let stuckProtocolTriggered = false;
// Default = budget_exhausted (while-condition falls through when budget hits 0 at the top of a
// round). Explicit breaks below set target_reached / fixpoint.
let exitReason = "budget_exhausted";

// stacking candidates: branches that beat baseline (deduped, best improvement kept)
const stackingByBranch = new Map();

function recordStackingCandidate(branch, improvementPct) {
  // Exclude the baseline/original branch: an HP-only win on it is not a stackable *code* method.
  if (!branch || branch === originalBranch) return;
  if (typeof improvementPct !== "number") return;
  if (improvementPct <= 0) return;
  const prev = stackingByBranch.get(branch);
  if (prev === undefined || improvementPct > prev) {
    stackingByBranch.set(branch, improvementPct);
  }
}

function isImprovement(candidate, reference) {
  if (typeof candidate !== "number" || typeof reference !== "number") return false;
  return lowerIsBetter ? candidate < reference : candidate > reference;
}

function targetReached(metric) {
  if (target_value === null || target_value === undefined) return false;
  if (typeof metric !== "number") return false;
  return lowerIsBetter ? metric <= target_value : metric >= target_value;
}

// metric routing: divergence uses divergence_metric (lower-is-better unless divergence_lower_is_better
// overrides for reward-like RL metrics); analysis + tuning use primary_metric. Both are passed to the agents.
const divergenceClause =
  divergence_metric === null || divergence_metric === undefined
    ? "Divergence monitoring is DISABLED (tabular ML / null divergence_metric) — do not run detect_divergence; rely on the hard timeout."
    : `Divergence metric: ${divergence_metric} (lower_is_better=${divergenceLowerIsBetter}). You own in-run monitoring: poll your training log with detect_divergence.py every 5 minutes (model_category=${modelCategory} for thresholds). status="diverged" (NaN/Inf/explosion/collapse) -> kill the run and mark status="diverged". status="plateaued" (plateau/drift) -> NEVER kill; record the warning in notes for analysis.`;

// Main autonomous loop — bounded by budget.remaining(). Per-iteration progress phases
// (Round/Experiments/Analysis/Decision) are set via each agent()'s phase: opt (no top-level
// marker here — it would not map to meta.phases[]).

while (budget.remaining() > 0) {
  iteration += 1;
  batchNumber += 1;

  // -- Step 0: create round ---------------------------------------------------
  const roundType = "hp";
  const roundRes = await agent(
    `Create a new experiment round and return its directory. Run:
\`python3 ${PLUGIN}/scripts/round_manager.py ${exp_root} create-round ${roundType}\`
Parse the JSON output and return round_dir = the "dir" field (e.g. "round-${iteration}-hp"). If the output has an "error", still return that dir field if present.`,
    { schema: ROUND_SCHEMA, agentType: "ml-optimizer:experiment-agent", phase: "Round", label: `round-${iteration}` }
  );
  const roundDir = roundRes && roundRes.round_dir ? roundRes.round_dir : `round-${iteration}-hp`;

  // -- Step 1: tuning-agent proposes configs ----------------------------------
  // Branch dispatch: iter 1 = one exp per branch at baseline HPs; iter 2 prunes losers; iter 3+
  // focuses best branch + HP tuning. The agent reads agenda/dead-ends/prior batch-N-analysis.md files.
  const branchPhase =
    iteration === 1
      ? "Iteration 1: test EACH code branch with baseline HPs (one config per branch) to find promising code changes."
      : iteration === 2
      ? "Iteration 2: prune branches worse than baseline; focus on surviving branches + baseline."
      : "Iteration 3+: focus the best branch(es) + HP tuning. branch_scores guide slot allocation.";

  const tuneRes = await agent(
    `Propose the next batch of HP configurations for the experiment loop.

Context files to read under ${exp_root}: reports/research-agenda.* (untried high-priority techniques), reports/dead-ends.* (DO NOT re-propose these), the most recent reports/batch-*-analysis.md (prior batch findings), learned-behaviors.json (OOM limits, divergence patterns).

Parameters:
- project_root: ${project_root}
- exp_root: ${exp_root}
- num_gpus: ${numGpus}
- num_configs: ${configsPerBatch}
- search_space: ${JSON.stringify(searchSpace)}
- iteration: ${iteration}
- primary_metric: ${primary_metric}
- lower_is_better: ${lowerIsBetter}
- code_branches: ${JSON.stringify(codeBranches)}
- branch_scores: ${JSON.stringify(branchScores)}
- correlations (from last analysis): ${JSON.stringify(correlations)}
${maxBatchSize !== null ? `- max_batch_size: ${maxBatchSize} (OOM constraint — do not propose configs that exceed this)` : "- (no OOM constraint)"}
- warm_start_enabled: ${warmStartEnabled}
- round_dir: ${roundDir}
- model_category: ${modelCategory}
- seeds_per_config: ${seedsPerConfig || 1}

${branchPhase}

REQUIRED OUTPUT (tuning-agent, mirrors scripts/output_contract.py): write one proposed-config JSON per config to ${exp_root}/proposed-configs/${roundDir}/exp-*.json (the PreToolUse hook BLOCKS any other path; the round dir must already exist — it was created in Step 0). At least one file must exist under that directory before you finish.
Return {round_dir, recommendation, configs:[{exp_id, config, gpu_id, code_branch, code_proposal, proposal_source, method_tier, iteration, checkpoint_source, warm_started}]}. recommendation is your own early signal ("continue" or "stop") — analysis makes the final call.`,
    { schema: HP_CONFIG_SCHEMA, agentType: "ml-optimizer:tuning-agent", phase: "Round", label: `tune-${iteration}` }
  );

  let configs = tuneRes && Array.isArray(tuneRes.configs) ? tuneRes.configs : [];
  // HP-tune failure recovery: if no valid configs, fall back to a single
  // baseline-config experiment to keep the loop alive.
  if (configs.length === 0) {
    log(`hp-tune returned no configs at iteration ${iteration} — falling back to a baseline-config batch.`);
    configs = [
      {
        exp_id: `exp-fallback-${iteration}`,
        config: baseline && baseline.config ? baseline.config : {},
        gpu_id: 0,
        code_branch: codeBranches[0] || null,
        method_tier: "method_default_hp",
        iteration,
      },
    ];
  }

  // hp-tune requested research-derived priors (Task 14): no cited search_space
  // priors exist yet for this model class. Route ONE research round via the
  // existing mid-loop research→implement path before broad HP exploration.
  if (tuneRes && tuneRes.research_requested === true) {
    if (!methodProposalsEnabled) {
      log(`hp-tune set research_requested=true but scope_level=${scope_level} disallows research — proceeding with baseline-anchored configs only.`);
    } else if (!methodProposalBudgetLeft()) {
      log(`hp-tune set research_requested=true but the method_proposal_iterations budget (${methodProposalBudget}) is exhausted — proceeding with baseline-anchored configs only.`);
    } else {
      log(`hp-tune set research_requested=true — routing one research round for cited HP priors before broad HP exploration.`);
      researchRoundIndex += 1;
      methodProposalRoundsUsed += 1;
      const added = await runResearchImplement(`hp-tune-requested prior research (iter ${researchRoundIndex})`, researchRoundIndex);
      if (added.length) {
        const merged = new Set(codeBranches);
        for (const b of added) merged.add(b);
        codeBranches = Array.from(merged);
      }
      batchesSinceLastResearch = 0;
    }
  }

  // -- Step 2: run experiments in parallel ------------------------------------
  // Each experiment-agent runs in a worktree (--detach) on its own code_branch so parallel runs
  // never collide on files. File-backup forces sequential.
  const runOne = (cfg, idx) => async () => {
    const cfgJson = JSON.stringify(cfg.config || {});
    return agent(
      `Run a single training experiment, then evaluate and write the result JSON.

exp_id: ${cfg.exp_id}
config: ${cfgJson}
gpu_id: ${cfg.gpu_id !== undefined && cfg.gpu_id !== null ? cfg.gpu_id : idx % numGpus}
project_root: ${project_root}
exp_root: ${exp_root}
round_dir: ${roundDir}
code_branch: ${cfg.code_branch || "null"}
code_proposal: ${cfg.code_proposal || "null"}
proposal_source: ${cfg.proposal_source || "null"}
method_tier: ${cfg.method_tier || "method_default_hp"}
iteration: ${iteration}
checkpoint_source: ${cfg.checkpoint_source ? JSON.stringify(cfg.checkpoint_source) : "null"}
primary_metric: ${primary_metric}
model_category: ${modelCategory}

Run inside a git worktree on the given code_branch using \`--detach\` so this run does not disturb the main tree or other parallel runs. ${divergenceClause} ${budgetClause} ${cpuClause}

REQUIRED OUTPUTS (experiment-agent, mirrors scripts/output_contract.py — ALL of these must exist before you finish):
  1. ${exp_root}/results/${roundDir}/${cfg.exp_id}.json — the result JSON (the PreToolUse hook BLOCKS any other path / bad schema).
  2. ${exp_root}/logs/${roundDir}/${cfg.exp_id}/train.log — the training log (eval.log too if eval is separate).
  3. ${exp_root}/scripts/${roundDir}/${cfg.exp_id}/ — the training script directory (train.sh, eval.sh if separate).
  4. ${exp_root}/artifacts/${roundDir}/${cfg.exp_id}/ — the checkpoint/artifacts directory.
The result JSON must include status, config, metrics, code_branch, method_tier, duration_seconds, iteration, and the primary_metric_value (${primary_metric} value). On OOM: do NOT retry; record the offending batch size to the error tracker and set status="failed" with notes. On timeout: kill and set status="timeout". Failed/diverged/timeout results still require notes. Return {exp_id, status, metrics, config, code_branch, method_tier, duration_seconds, primary_metric_value, diverged, notes, result_path}.`,
      {
        schema: EXPERIMENT_RESULT_SCHEMA,
        agentType: "ml-optimizer:experiment-agent",
        isolation: "worktree",
        phase: "Experiments",
        label: `exp-${cfg.exp_id}`,
      }
    );
  };

  let results;
  if (sequential) {
    // file_backup: one experiment at a time
    results = [];
    for (let i = 0; i < configs.length; i++) {
      results.push(await runOne(configs[i], i)());
    }
  } else {
    results = await parallel(configs.map((cfg, i) => runOne(cfg, i)));
  }
  const completed = results.filter(Boolean);

  // -- Step 3: batch monitor skipped -----------------------------------------
  // Per-run divergence is handled inside the experiment agent; the agents have already
  // completed here, so a separate live monitor adds nothing — analysis reads the
  // diverged/timeout statuses from results.

  // -- Step 4: analysis-agent -------------------------------------------------
  const completedSummary = completed.map((r) => ({
    exp_id: r.exp_id,
    status: r.status,
    code_branch: r.code_branch || null,
    metric: r.primary_metric_value,
  }));
  const allDiverged =
    completed.length > 0 &&
    completed.every((r) => r.status === "diverged" || r.diverged === true);

  const analyzeRes = await agent(
    `Analyze batch ${batchNumber} results and decide the next action.

exp_root: ${exp_root}
project_root: ${project_root}
batch_number: ${batchNumber}
iteration: ${iteration}
primary_metric: ${primary_metric}
lower_is_better: ${lowerIsBetter}
target_value: ${target_value === null || target_value === undefined ? "null" : target_value}
scope_level: ${scope_level}
model_category: ${modelCategory}
baseline_metric (${primary_metric}): ${baselineMetricValue}
current_best_metric: ${bestMetric}
secondary_metrics: ${JSON.stringify(secondaryMetrics)} (each {name, lower_is_better, role}; role "guardrail" entries must not regress vs baseline in top-ranked results — apply your Step 2.3 guardrail check; [] = skip)
code_branches: ${JSON.stringify(codeBranches)}
batch results: ${JSON.stringify(completedSummary)}
${allDiverged ? "NOTE: ALL experiments in this batch diverged — recommend a recovery action (halve LRs / narrow_space)." : ""}

Read all exp-*.json under ${exp_root}/results/${roundDir}/ and prior rounds. Compute rankings, branch_scores, HP correlations/interactions, effect sizes, and any stacking candidates (branches whose best result beats baseline, with improvement_pct).

REQUIRED OUTPUT (analysis-agent, mirrors scripts/output_contract.py): write your analysis to ${exp_root}/reports/batch-${batchNumber}-analysis.md (this file MUST exist before you finish) and update the research agenda + dead-end catalog (error_tracker.py).

Decision tree (respect scope_level — "training" disables research/code_evolution pivots; "architecture" enables research but NOT code_evolution; "full" enables everything):
- continue — keep tuning the same space
- branch_test — test untested branches at baseline HPs (set untested_branches)
- hp_expand — widen the search space around the best config (return widened search_space)
- narrow_space — constrain to the best region (return narrowed search_space)
- regularization — add/expand weight_decay/dropout in search_space (for model_category=rl: entropy coefficient / KL penalty)
- method_proposal — HP tuning plateaued; propose new methods via research (only if scope_level != "training")
- code_evolution — diminishing HP returns + method improved (|rho|<0.3); evolve code (only if scope_level == "full")
- stop — approaches look exhausted (the workflow then runs the stuck protocol + fixpoint check)

Return {decision, pivot_type, batch_number, best_exp_id, best_metric_value, improved_since_last_stop, search_space, branch_scores, correlations, untested_branches, surviving_branches, max_batch_size, reason, analysis_path, stacking_candidates}.`,
    { schema: ANALYSIS_DECISION_SCHEMA, agentType: "ml-optimizer:analysis-agent", phase: "Analysis", label: `analyze-${batchNumber}` }
  );

  // Malformed analysis -> default to continue (per the reference doc).
  let decision = analyzeRes && analyzeRes.decision ? analyzeRes.decision : "continue";

  // qualitative_change is a method-proposal-class pivot_type. Route it to method_proposal
  // ONLY when the agent didn't already pick a more specific decision — otherwise a
  // decision like "stop" or "code_evolution" that also carries pivot_type=qualitative_change
  // would be silently rerouted to method_proposal, bypassing the stuck protocol / evolution.
  if (
    analyzeRes && analyzeRes.pivot_type === "qualitative_change" &&
    decision !== "stop" && decision !== "code_evolution" && decision !== "method_proposal"
  ) {
    decision = "method_proposal";
  }

  // Update best metric + branch scores from this batch.
  if (analyzeRes) {
    if (typeof analyzeRes.best_metric_value === "number" && isImprovement(analyzeRes.best_metric_value, bestMetric)) {
      bestMetric = analyzeRes.best_metric_value;
      bestExpId = analyzeRes.best_exp_id || bestExpId;
    } else if (analyzeRes.best_exp_id && bestExpId === null) {
      bestExpId = analyzeRes.best_exp_id;
    }
    if (analyzeRes.branch_scores) branchScores = analyzeRes.branch_scores;
    if (analyzeRes.correlations) correlations = analyzeRes.correlations;
    if (typeof analyzeRes.max_batch_size === "number") maxBatchSize = analyzeRes.max_batch_size;
    if (Array.isArray(analyzeRes.stacking_candidates)) {
      for (const c of analyzeRes.stacking_candidates) recordStackingCandidate(c.branch, c.improvement_pct);
    }
  }
  // Also harvest stacking candidates directly from completed results vs baseline.
  for (const r of completed) {
    if (r.status === "completed" && r.code_branch && typeof r.primary_metric_value === "number" && typeof baselineMetricValue === "number") {
      if (isImprovement(r.primary_metric_value, baselineMetricValue)) {
        const denom = Math.abs(baselineMetricValue) > 1e-12 ? Math.abs(baselineMetricValue) : 1;
        const improvement = lowerIsBetter
          ? ((baselineMetricValue - r.primary_metric_value) / denom) * 100
          : ((r.primary_metric_value - baselineMetricValue) / denom) * 100;
        recordStackingCandidate(r.code_branch, improvement);
      }
    }
  }

  // -- Step 4.5: per-batch completeness check (SubagentStop 'verify' role) -----
  // Workflow-dispatched agents may skip the SubagentStop hook, so verify the round's required
  // outputs exist on disk via output_contract.py (same source the hooks use). Missing outputs are
  // logged (loud) but do not abort — analysis already returned its decision; a partial batch is
  // still actionable.
  await verifyBatchOutputs(roundDir, completed, batchNumber);

  // Close the round.
  await agent(
    `Close the current experiment round with a summary. Run:
\`python3 ${PLUGIN}/scripts/round_manager.py ${exp_root} close-round --summary "Iteration ${iteration}: best ${primary_metric}=${bestMetric}, ${completed.length}/${configs.length} completed"\`
Also regenerate the live dashboard: \`python3 ${PLUGIN}/scripts/dashboard.py ${exp_root} --live\`.
Return nothing structured.`,
    { agentType: "ml-optimizer:experiment-agent", phase: "Analysis", label: `close-${iteration}` }
  );

  // -- Target reached? --------------------------------------------------------
  if (targetReached(bestMetric)) {
    exitReason = "target_reached";
    log(`Target ${primary_metric}=${target_value} reached (best=${bestMetric}). Exiting loop.`);
    break;
  }

  // -- Step 4.7: adversarial second opinion before a COSTLY pivot ------------
  // method_proposal / code_evolution each spend a full research/implement/evolve cycle. Before one,
  // dispatch independent skeptics to REFUTE the decision from the run's files; a credible refutation
  // downgrades the pivot to "continue" so a single mis-read does not burn a cycle. "stop" is NOT
  // second-guessed here — it is already guarded by the stuck protocol + fixpoint below, which
  // preserves the loop's termination guarantee.
  const isCostlyPivot = decision === "code_evolution" || decision === "method_proposal";
  if (isCostlyPivot) {
    const REFUTE_SCHEMA = { type: "object", required: ["refuted"], properties: { refuted: { type: "boolean" }, reason: { type: "string" } } };
    const lenses = [
      "an untried, in-scope, non-dead-end HP direction still exists that is cheaper than this pivot (check reports/research-agenda.* and the search-space coverage in the latest reports/batch-*-analysis.md) — if so the pivot is premature",
      "the HP plateau is within noise rather than a real signal (re-examine the batch result spread / effect sizes in the latest reports/batch-*-analysis.md) — if so keep tuning before pivoting",
    ];
    const verdicts = (await parallel(lenses.map((lens, li) => () => agent(
      `You are an independent skeptic reviewing the analysis-agent's costly "${decision}" pivot (pivot_type=${(analyzeRes && analyzeRes.pivot_type) || "none"}) at iteration ${iteration}. Read ${exp_root}/reports/batch-${batchNumber}-analysis.md, the results under ${exp_root}/results/, reports/research-agenda.*, and reports/dead-ends.*. Try to REFUTE the pivot via this lens: ${lens}. Default refuted=false unless you find concrete, file-grounded evidence. Return {refuted, reason}.`,
      { agentType: "ml-optimizer:analysis-agent", phase: "Decision", label: `refute-pivot-${li + 1}-${iteration}`, schema: REFUTE_SCHEMA }))
    )).filter(Boolean);
    if (verdicts.some((v) => v && v.refuted === true)) {
      const why = verdicts.filter((v) => v && v.refuted).map((v) => v.reason).filter(Boolean).join(" | ");
      log(`Adversarial check overrode costly pivot "${decision}" -> "continue": ${why}`);
      decision = "continue";
    }
  }

  // -- Step 5: act on the decision -------------------------------------------
  if (decision === "continue") {
    batchesSinceLastResearch += 1;
  } else if (decision === "branch_test") {
    // hand untested branches into next tuning round (codeBranches already holds them)
    if (analyzeRes && Array.isArray(analyzeRes.untested_branches) && analyzeRes.untested_branches.length) {
      const merged = new Set(codeBranches);
      for (const b of analyzeRes.untested_branches) merged.add(b);
      codeBranches = Array.from(merged);
    }
    batchesSinceLastResearch += 1;
  } else if (decision === "hp_expand" || decision === "narrow_space" || decision === "regularization") {
    if (analyzeRes && analyzeRes.search_space) searchSpace = analyzeRes.search_space;
    batchesSinceLastResearch += 1;
  } else if (decision === "method_proposal") {
    // Mid-loop method proposal — pre-authorized at Phase 4 (NO user prompt).
    if (!methodProposalsEnabled) {
      log(`method_proposal requested but scope_level=${scope_level} disallows it — treating as continue.`);
      batchesSinceLastResearch += 1;
    } else if (!methodProposalBudgetLeft()) {
      log(`method_proposal requested but method_proposal_iterations budget (${methodProposalBudget}) is exhausted — falling back to hp_expand.`);
      if (analyzeRes && analyzeRes.search_space) searchSpace = analyzeRes.search_space;
      batchesSinceLastResearch += 1;
    } else {
      researchRoundIndex += 1;
      methodProposalRoundsUsed += 1; // count the round when it runs, per the doc's budget
      const added = await runResearchImplement(`mid-loop method proposal (iter ${researchRoundIndex})`, researchRoundIndex);
      if (added.length) {
        const merged = new Set(codeBranches);
        for (const b of added) merged.add(b);
        codeBranches = Array.from(merged);
        stuckProtocolTriggered = false;
        consecutiveStopCount = 0;
      }
      batchesSinceLastResearch = 0;
    }
  } else if (decision === "code_evolution") {
    if (!codeEvolutionEnabled) {
      log(`code_evolution requested but scope_level=${scope_level} != "full" — treating as continue.`);
      batchesSinceLastResearch += 1;
    } else if (!methodProposalBudgetLeft()) {
      log(`code_evolution requested but method_proposal_iterations budget (${methodProposalBudget}) is exhausted — falling back to hp_expand.`);
      if (analyzeRes && analyzeRes.search_space) searchSpace = analyzeRes.search_space;
      batchesSinceLastResearch += 1;
    } else {
      methodProposalRoundsUsed += 1;
      const evolvedBranch = await runCodeEvolution(analyzeRes);
      if (evolvedBranch) {
        const merged = new Set(codeBranches);
        merged.add(evolvedBranch);
        codeBranches = Array.from(merged);
      }
      batchesSinceLastResearch += 1;
    }
  } else if (decision === "stop") {
    // -- Stuck protocol + fixpoint exit judgment ------------------------------
    consecutiveStopCount += 1;
    const improvedSinceLastStop = isImprovement(bestMetric, lastStopBestMetric);
    log(`Analysis recommended stop (count=${consecutiveStopCount}). Running stuck protocol.`);

    let newProposalsAdded = [];
    let agendaHasUntried = false;
    let researchNewInScope = 0;

    if (methodProposalsEnabled) {
      researchRoundIndex += 1;
      const stuck = await runStuckResearch(researchRoundIndex);
      researchNewInScope = stuck.newInScope;
      agendaHasUntried = stuck.agendaHasUntried;
      if (stuck.selectedIndices.length) {
        newProposalsAdded = await runImplementOnly(stuck.findingsPath, stuck.selectedIndices);
        if (newProposalsAdded.length) {
          const merged = new Set(codeBranches);
          for (const b of newProposalsAdded) merged.add(b);
          codeBranches = Array.from(merged);
        }
      }
    } else {
      log(`Scope_level=${scope_level} disables research — stuck protocol can only check for metric improvement.`);
    }

    // Exit judgment: CONTINUE if any new direction exists; EXIT only at fixpoint.
    const continueLoop =
      newProposalsAdded.length > 0 ||
      researchNewInScope > 0 ||
      agendaHasUntried ||
      improvedSinceLastStop;

    await agent(
      `Log the loop-exit judgment decision. Run:
\`python3 ${PLUGIN}/scripts/pipeline_state.py ${exp_root} log-decision '${JSON.stringify({
        phase: 7,
        agent: "workflow",
        decision_type: "loop_exit_judgment",
        decision: continueLoop ? "continue" : "exit",
        iteration,
        reasoning: `best_metric=${bestMetric}, improved_since_last_stop=${improvedSinceLastStop}, agenda_untried=${agendaHasUntried}, new_proposals=${newProposalsAdded.length + researchNewInScope}`,
      }).replace(/'/g, "'\\''")}'\`
Return nothing structured.`,
      { agentType: "ml-optimizer:experiment-agent", phase: "Decision", label: `exit-judgment-${iteration}` }
    );

    if (continueLoop) {
      stuckProtocolTriggered = false;
      consecutiveStopCount = 0;
      lastStopBestMetric = bestMetric;
      batchesSinceLastResearch = 0;
      log("Stuck protocol found a fresh direction — continuing loop.");
    } else {
      stuckProtocolTriggered = true;
      lastStopBestMetric = bestMetric;
      exitReason = "fixpoint";
      log("Fixpoint reached (no new in-scope proposals, empty agenda, flat metric). Exiting to Phase 9.");
      break;
    }
  } else {
    // Unknown decision -> treat as hp_expand (safest default per the reference).
    log(`Unknown analysis decision '${decision}' — defaulting to hp_expand behavior.`);
    if (analyzeRes && analyzeRes.search_space) searchSpace = analyzeRes.search_space;
    batchesSinceLastResearch += 1;
  }

  // -- Step 6: cadence-based research round (independent of pivot) -------------
  if (
    methodProposalsEnabled &&
    methodProposalBudgetLeft() &&
    decision !== "method_proposal" &&
    decision !== "code_evolution" && // code_evolution is a step-7 pivot — don't double-fire cadence research the same iteration
    decision !== "stop" &&
    batchesSinceLastResearch >= cadence
  ) {
    researchRoundIndex += 1;
    methodProposalRoundsUsed += 1;
    const added = await runResearchImplement(`cadence research round (iter ${researchRoundIndex})`, researchRoundIndex);
    if (added.length) {
      const merged = new Set(codeBranches);
      for (const b of added) merged.add(b);
      codeBranches = Array.from(merged);
    }
    batchesSinceLastResearch = 0;
  }

  if (budget.remaining() <= 0) {
    exitReason = "budget_exhausted";
    log("Budget exhausted — exiting loop.");
    break;
  }
}

phase("Exit");

const stackingCandidates = Array.from(stackingByBranch.entries())
  .map(([branch, improvement_pct]) => ({ branch, improvement_pct }))
  .sort((a, b) => b.improvement_pct - a.improvement_pct);

return {
  best_exp_id: bestExpId,
  best_metric: bestMetric,
  rounds_completed: iteration,
  exit_reason: exitReason,
  stacking_candidates: stackingCandidates,
};

// Helpers — function declarations (they hoist above their use) closing over loop state.

// Per-batch completeness verifier (SubagentStop 'verify' role moved into the script). Uses
// scripts/output_contract.py — the SAME contract source the hooks use — to confirm each completed
// experiment's required outputs (result JSON, train.log, scripts/artifacts dirs) and the batch
// analysis file exist on disk. Logs (loud) on any gap; non-fatal.
async function verifyBatchOutputs(roundDir, completedResults, batchNum) {
  const expIds = (completedResults || []).map((r) => r && r.exp_id).filter(Boolean);
  const VERIFY_SCHEMA = {
    type: "object",
    required: ["all_present"],
    properties: {
      all_present: { type: "boolean" },
      missing: { type: "array", items: { type: "string" } },
    },
  };
  const res = await agent(
    `Completeness check for Phase 7 batch ${batchNum} (round ${roundDir}). Use Bash/Read ONLY (do NOT re-run experiments). For EACH exp_id in ${JSON.stringify(expIds)} run:
\`python3 ${PLUGIN}/scripts/output_contract.py check ${exp_root} experiment-agent --round-dir ${roundDir} --exp-id <exp_id>\`
(exit code 2 means missing outputs — capture the "missing" array from its JSON). Also confirm the batch analysis file exists: ${exp_root}/reports/batch-${batchNum}-analysis.md.
Collect every missing path across all exp_ids (and the analysis file if absent) into a flat "missing" array. If anything is missing, log it once: \`python3 ${PLUGIN}/scripts/error_tracker.py ${exp_root} log '{"category":"agent_failure","severity":"warning","source":"orchestrate","message":"batch ${batchNum} missing required outputs","phase":7}'\`. Do NOT fabricate outputs.
Return {all_present, missing}.`,
    { schema: VERIFY_SCHEMA, agentType: "ml-optimizer:experiment-agent", phase: "Analysis", label: `verify-batch-${batchNum}` }
  );
  if (!res || res.all_present !== true) {
    const missing = res && Array.isArray(res.missing) ? res.missing : ["<verifier failed>"];
    log(`Phase 7 batch ${batchNum} completeness WARNING — missing outputs: ${missing.join(", ")}.`);
  }
  return res;
}

// Research round (proposals) + implement the in-scope ones. Returns new branches.
async function runResearchImplement(reasonLabel, idx) {
  const findingsPath = `${exp_root}/reports/research-findings-method-proposals-iter${idx}.md`;
  const research = await agent(
    `${reasonLabel}. Research ML optimization techniques and write in-scope proposals.

Parameters:
- source: both
- scope_level: ${method_proposal_scope}
- output_path: ${findingsPath}
- exp_root: ${exp_root}
- project_root: ${project_root}
- primary_metric: ${primary_metric}
- current best ${primary_metric}: ${bestMetric}
- model_category: ${modelCategory}

REQUIRED OUTPUT (research-agent): write the proposals to ${findingsPath} (this file MUST exist before you finish; the PreToolUse hook guards the path/schema).
Read ${exp_root}/reports/dead-ends.* and the research agenda first — DO NOT re-propose dead ends or already-tried techniques (dedupe against all research-findings*.md). Initialize/update the research agenda. Return {findings_path, proposals, new_in_scope_count, agenda_has_untried}.`,
    { schema: RESEARCH_PROPOSALS_SCHEMA, agentType: "ml-optimizer:research-agent", phase: "Decision", label: `research-${idx}` }
  );
  if (!research || !Array.isArray(research.proposals) || research.proposals.length === 0) return [];
  // ALL returned in-scope proposals are implemented automatically (pre-authorized).
  const indices = research.proposals.map((p) => p.index).filter((i) => Number.isInteger(i));
  if (indices.length === 0) return [];
  return runImplementOnly(research.findings_path || findingsPath, indices);
}

// Stuck-protocol research (failure-context aware). Returns counts + selection.
async function runStuckResearch(idx) {
  const findingsPath = `${exp_root}/reports/research-findings-method-proposals-iter${idx}.md`;
  const research = await agent(
    `STUCK PROTOCOL — the optimization is stuck; find new approaches NOT yet tried.

Read first (FILES under ${exp_root}): error patterns (\`python3 ${PLUGIN}/scripts/error_tracker.py ${exp_root} patterns\`), success metrics, the dead-end catalog (\`... dead-end list\` — DO NOT re-propose any of these), and the research agenda (\`... agenda list\`).

Parameters:
- source: both
- scope_level: ${method_proposal_scope || "architecture"}
- output_path: ${findingsPath}
- exp_root: ${exp_root}
- project_root: ${project_root}
- primary_metric: ${primary_metric}
- current best ${primary_metric}: ${bestMetric}
- model_category: ${modelCategory}

REQUIRED OUTPUT (research-agent): write the proposals to ${findingsPath} (this file MUST exist before you finish; the PreToolUse hook guards the path/schema).
Focus on techniques NOT in the dead-end catalog. Return {findings_path, proposals (only NEW in-scope ones after dead-end + suggestion-history filtering), new_in_scope_count, agenda_has_untried}.`,
    { schema: RESEARCH_PROPOSALS_SCHEMA, agentType: "ml-optimizer:research-agent", phase: "Decision", label: `stuck-research-${idx}` }
  );
  if (!research) return { newInScope: 0, agendaHasUntried: false, selectedIndices: [], findingsPath };
  const indices = (research.proposals || []).map((p) => p.index).filter((i) => Number.isInteger(i));
  return {
    newInScope: typeof research.new_in_scope_count === "number" ? research.new_in_scope_count : indices.length,
    agendaHasUntried: research.agenda_has_untried === true,
    selectedIndices: indices,
    findingsPath: research.findings_path || findingsPath,
  };
}

// Implement selected proposals -> new validated branches.
async function runImplementOnly(findingsPath, selectedIndices) {
  const impl = await agent(
    `Implement the selected research proposals as git branches (ml-opt/<slug>), one per proposal, sequentially inside a git worktree outside ${exp_root}.

Parameters:
- findings_path: ${findingsPath}
- selected_indices: ${JSON.stringify(selectedIndices)}
- project_root: ${project_root}
- exp_root: ${exp_root}

REQUIRED OUTPUT (implement-agent): update ${exp_root}/results/implementation-manifest.json (this file MUST exist and pass schema_validator.py manifest before you finish; each validated proposal leaves a persisted ml-opt/<slug> branch).
Validate each branch (pre-flight file checks, LSP/pyright). Return {manifest_path, branches:[{slug, branch, status, type}]}.`,
    { schema: MANIFEST_BRANCHES_SCHEMA, agentType: "ml-optimizer:implement-agent", phase: "Decision", label: "implement-midloop" }
  );
  if (!impl || !Array.isArray(impl.branches)) return [];
  return impl.branches.filter((b) => b.status === "validated").map((b) => b.branch);
}

// code_evolution: tuning(evolve HPs) -> implement(evolve skill). Returns branch or null.
async function runCodeEvolution(analysis) {
  const parentBranch =
    (analysis && analysis.surviving_branches && analysis.surviving_branches[0]) ||
    codeBranches[0] ||
    null;
  const evolveHp = await agent(
    `Propose ShinkaEvolve evolution HPs for a code_evolution step. Read learned-behaviors.json category "evolve_hp" under ${exp_root} for prior outcomes (how many generations produced the best mutation, whether more population diversity helped). Defaults: num_generations=10, population_size=2. Return {num_generations, population_size, reasoning}.`,
    { schema: EVOLVE_HP_SCHEMA, agentType: "ml-optimizer:tuning-agent", phase: "Decision", label: "evolve-hp" }
  );
  const numGen = evolveHp && evolveHp.num_generations ? evolveHp.num_generations : 10;
  const popSize = evolveHp && evolveHp.population_size ? evolveHp.population_size : 2;

  const evolve = await agent(
    `Run the ml-optimizer:evolve skill (ShinkaEvolve) to evolve code on the best branch.

Skill("ml-optimizer:evolve") parameters:
- project_root: ${project_root}
- exp_root: ${exp_root}
- parent_branch: ${parentBranch}
- parent_metrics: best ${primary_metric}=${bestMetric}
- primary_metric: ${primary_metric}
- lower_is_better: ${lowerIsBetter}
- scope_level: ${scope_level}
- evolve_recommendation: {num_generations: ${numGen}, population_size: ${popSize}}

The evolve skill runs shinka-convert -> shinka-run (file handoff, SHINKA_PROVIDER=claude_code) -> shinka-inspect -> commit best as ml-opt/evolved-<slug>. If ShinkaEvolve is unavailable, return status="shinkaevolve_unavailable". Return {status, branch, best_combined_score, notes}.`,
    { schema: EVOLVE_RESULT_SCHEMA, agentType: "ml-optimizer:implement-agent", phase: "Decision", label: "evolve-run" }
  );
  if (evolve && evolve.status === "validated" && evolve.branch) return evolve.branch;
  if (evolve && evolve.status === "shinkaevolve_unavailable") {
    log("ShinkaEvolve unavailable — falling back to research->implement path.");
    if (methodProposalsEnabled) {
      researchRoundIndex += 1;
      const added = await runResearchImplement("code_evolution fallback (research)", researchRoundIndex);
      return added.length ? added[0] : null;
    }
  }
  return null;
}
