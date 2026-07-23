export const meta = {
  name: 'phase-5-research',
  description: 'Phase 5: fan-out ML optimization research across domain angles, dedup + adversarially vet proposals, write reports/research-findings.md + init the research agenda, and return ranked proposals.',
  phases: [
    { title: 'Fan-out', detail: 'parallel research-agent calls across domain-specific + generic angles + user papers' },
    { title: 'Vet', detail: 'pipeline deep-read + adversarial feasibility/novelty check per candidate' },
    { title: 'Synthesize', detail: 'one research-agent writes research-findings.md and initializes the research agenda' },
  ],
}

// -------------------- args (CONTRACT phase-5-research) --------------------
// { exp_root, primary_metric, model_category, scope_level, source, user_papers }
const expRoot = args.exp_root
const primaryMetric = args.primary_metric
const modelCategory = args.model_category || 'supervised'
const scopeLevel = args.scope_level || 'training'
const source = args.source || 'web'
const userPapers = Array.isArray(args.user_papers) ? args.user_papers : (args.user_papers ? [args.user_papers] : [])
const findingsPath = `${expRoot}/reports/research-findings.md`

// -------------------- structured return schemas --------------------
const CANDIDATE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['proposals'],
  properties: {
    proposals: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'impact', 'confidence', 'feasibility', 'scope', 'type', 'implementation_strategy'],
        properties: {
          title: { type: 'string' },
          impact: { type: 'number', description: '1-10 expected improvement' },
          confidence: { type: 'number', description: '1-10 confidence in the outcome' },
          feasibility: { type: 'number', description: '1-10 ease of implementation' },
          scope: { enum: ['training', 'architecture', 'full'] },
          type: { enum: ['code_change', 'hp_only'] },
          implementation_strategy: { enum: ['from_scratch', 'from_reference'] },
          files_to_modify: { type: 'array', items: { type: 'string' } },
          source: { type: 'string', description: 'paper title/URL, "user", or "LLM knowledge"' },
          reference_repo: { type: 'string' },
          notes: { type: 'string' },
        },
      },
    },
  },
}

const VET_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['title', 'keep', 'impact', 'confidence', 'feasibility', 'scope', 'type', 'implementation_strategy'],
  properties: {
    title: { type: 'string' },
    keep: { type: 'boolean', description: 'false if a dead-end, out-of-scope, infeasible, or non-novel duplicate' },
    reason: { type: 'string', description: 'why kept or dropped' },
    impact: { type: 'number' },
    confidence: { type: 'number' },
    feasibility: { type: 'number' },
    scope: { enum: ['training', 'architecture', 'full'] },
    type: { enum: ['code_change', 'hp_only'] },
    implementation_strategy: { enum: ['from_scratch', 'from_reference'] },
    files_to_modify: { type: 'array', items: { type: 'string' } },
    source: { type: 'string' },
    reference_repo: { type: 'string' },
  },
}

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['findings_path', 'agenda_initialized', 'proposals'],
  properties: {
    findings_path: { type: 'string' },
    agenda_initialized: { type: 'boolean' },
    proposals: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['index', 'title', 'impact', 'confidence', 'feasibility', 'scope', 'type', 'implementation_strategy'],
        properties: {
          index: { type: 'number', description: '1-based position in research-findings.md' },
          title: { type: 'string' },
          impact: { type: 'number' },
          confidence: { type: 'number' },
          feasibility: { type: 'number' },
          scope: { enum: ['training', 'architecture', 'full'] },
          type: { enum: ['code_change', 'hp_only'] },
          implementation_strategy: { enum: ['from_scratch', 'from_reference'] },
          files_to_modify: { type: 'array', items: { type: 'string' } },
        },
      },
    },
  },
}

const RESEARCH_AGENT = 'ml-optimizer:research-agent'

const baseCtx = `You are the ml-optimizer research-agent running inside the Phase 5 research workflow.
exp_root: ${expRoot}
primary_metric (optimize this): ${primaryMetric}
model_category: ${modelCategory}
scope_level: ${scopeLevel}  (NEVER propose changes outside this scope)
source: ${source}
Respect optimization-goals.json constraints and the dead-end catalog under ${expRoot} — do NOT re-propose dead-end or out-of-scope techniques.`

// Output contract for the research-agent (mirrors output_contract.py "research-agent").
// Workflow-dispatched agents may skip the SubagentStart/SubagentStop hooks, so the synthesis
// prompt states the output path and verifyFindings() checks it on disk after the agents run.
const OUTPUT_CONTRACT = `REQUIRED OUTPUT (research-agent):
  ${findingsPath}  — research findings (the Phase 5 canonical findings file; PreToolUse hook BLOCKS writes to the wrong path/schema).
Before finishing, verify ${findingsPath} exists and is non-empty with the Read tool; re-write it if missing.`

// Completeness check (SubagentStop 'verify' role, in-script): a verifier agent confirms the
// findings file exists on disk, else logs the gap and writes a minimal stub for downstream phases.
async function verifyFindings(label) {
  const VERIFY_SCHEMA = { type: 'object', required: ['exists'], properties: { exists: { type: 'boolean' }, repaired: { type: 'boolean' } } }
  const res = await agent(
    `Completeness check for Phase 5 research output. Using the Read/Glob tools ONLY (do not re-run research):
1. Check whether ${findingsPath} exists and is non-empty.
2. If it is MISSING or empty, log the gap (\`python3 \${CLAUDE_PLUGIN_ROOT}/scripts/error_tracker.py ${expRoot} log '{"category":"agent_failure","severity":"warning","source":"orchestrate","message":"research-findings.md missing after synthesis","phase":5}'\`) and write a minimal ${findingsPath} with a Problem Statement and an empty "## Proposals (Ranked by Priority)" section so downstream phases can proceed. Then set repaired=true.
3. If it already exists and is non-empty, set exists=true, repaired=false.
Return {exists, repaired}.`,
    { agentType: RESEARCH_AGENT, label, phase: 'Synthesize', schema: VERIFY_SCHEMA })
  if (!res || res.exists !== true) {
    log(`Phase 5 completeness: ${findingsPath} could not be confirmed (verifier returned ${JSON.stringify(res)}).`)
  } else if (res.repaired) {
    log(`Phase 5 completeness: ${findingsPath} was missing and has been repaired with a stub.`)
  }
  return res
}

// -------------------- Phase 1: fan-out across research angles --------------------
phase('Fan-out')

// Domain-specific angles per model_category, plus an always-on generic angle.
const ANGLES_BY_CATEGORY = {
  supervised: ['architecture & training strategy', 'loss functions & regularization'],
  generative: ['architecture & sampling/inference', 'loss functions & training stability'],
  rl: ['policy optimization & value estimation', 'exploration & reward shaping'],
  vla: ['imitation-learning objectives & action chunking', 'demonstration augmentation & co-training'],
  nlp: ['attention/architecture & fine-tuning (LoRA/PEFT)', 'tokenization & embeddings'],
  cv: ['task-specific architectures', 'data augmentation & regularization'],
  timeseries: ['temporal encoding & patching', 'forecasting architectures'],
  tabular: ['feature engineering & selection', 'ensembling & tree-model tuning'],
}
if (!ANGLES_BY_CATEGORY[modelCategory]) {
  log(`Phase 5: unknown model_category "${modelCategory}" — falling back to supervised research angles.`)
}
const domainAngles = ANGLES_BY_CATEGORY[modelCategory] || ANGLES_BY_CATEGORY.supervised
const angles = [...domainAngles, 'generic cross-domain optimization techniques']

// Each angle is an independent research-agent call. user_papers (if any) go to the
// first angle so the user-paper priority bonus is applied exactly once.
const angleThunks = angles.map((angle, i) => () => agent(
  `${baseCtx}

Research ANGLE ${i + 1}/${angles.length}: "${angle}".
Use the research skill's web-search / paper-analysis tooling for source="${source}" (parallel WebSearch + alphaxiv; assess candidate reference repos via alphaxiv / grep). ${i === 0 && userPapers.length ? `Also analyze these user-provided papers and apply the user-paper priority bonus: ${JSON.stringify(userPapers)}.` : 'Do NOT analyze user papers in this angle (handled elsewhere).'}
Do NOT write reports/research-findings.md yet and do NOT initialize the agenda yet — that is a later synthesis step.
Return only candidate proposals scoped to "${angle}" as the schema. Cap knowledge-mode confidence at 7.`,
  { agentType: RESEARCH_AGENT, label: `research:angle-${i + 1}`, phase: 'Fan-out', schema: CANDIDATE_SCHEMA }))

const angleResults = (await parallel(angleThunks)).filter(Boolean)

// Flatten + dedup candidates by normalized title (case-insensitive, strip trailing
// "loss"/"function"/"scheduler"/etc.) — the same fuzzy rule the research skill uses.
function normTitle(t) {
  let s = String(t || '').toLowerCase().trim()
  s = s.replace(/[-_]+/g, ' ').replace(/\s+/g, ' ').trim()
  s = s.replace(/\b(loss|function|scheduler|strategy|method|technique)\b/g, '').trim()
  return s
}
const seen = new Set()
const candidates = []
for (const r of angleResults) {
  for (const p of (r && r.proposals ? r.proposals : [])) {
    const key = normTitle(p.title)
    if (!key || seen.has(key)) continue
    seen.add(key)
    candidates.push(p)
  }
}
log(`Phase 5: ${angleResults.length} angle(s) returned, ${candidates.length} unique candidate proposal(s) after dedup.`)

// -------------------- Phase 2: adversarial vet per candidate --------------------
phase('Vet')

// pipeline: one stage — deep-read + adversarial feasibility/novelty check per candidate.
// No barrier between candidates; each is independently vetted (concurrency auto-capped).
const vetted = candidates.length
  ? (await pipeline(
      candidates,
      (_prev, cand, idx) => agent(
        `${baseCtx}

ADVERSARIALLY VET this candidate proposal (#${idx + 1}) before it reaches the user:
${JSON.stringify(cand)}

Steps:
1. Deep-read the cited source (alphaxiv get_paper_content / answer_pdf_queries, or WebFetch; for from_reference, assess the repo via alphaxiv read_files_from_github_repository / grep).
2. Feasibility check FOR THIS PROJECT: compatible with the model/framework, fits the budget, implementable without major refactor. Set files_to_modify to concrete target paths under the project when known.
3. Novelty/dead-end check: drop (keep=false) if it duplicates an existing proposal, is in the dead-end catalog, or violates scope_level "${scopeLevel}".
4. Re-score impact/confidence/feasibility honestly (1-10; cap knowledge-mode confidence at 7). Set type=hp_only when the change is purely a search-space tweak, else code_change.
Return the schema with keep + final scores. Do NOT write files.`,
        { agentType: RESEARCH_AGENT, label: `research:vet-${idx + 1}`, phase: 'Vet', schema: VET_SCHEMA }))
    ).filter(Boolean).filter((v) => v && v.keep !== false)
  : []

log(`Phase 5: ${vetted.length}/${candidates.length} candidate(s) survived adversarial vetting.`)

// Rank survivors by the plugin's priority formula: (impact * confidence) / (11 - clamp(feasibility,1,10)).
function priority(p) {
  const impact = Number(p.impact) || 0
  const conf = Number(p.confidence) || 0
  const feas = Math.min(Math.max(Number(p.feasibility) || 1, 1), 10)
  return (impact * conf) / (11 - feas)
}
const ranked = vetted.slice().sort((a, b) => priority(b) - priority(a))

// -------------------- Phase 3: synthesize findings + init agenda --------------------
phase('Synthesize')

if (!ranked.length) {
  // No actionable proposals — still emit an honest findings file so downstream phases
  // can detect the empty result; the research-agent logs the research_failure event.
  await agent(
    `${baseCtx}

${OUTPUT_CONTRACT}

Research produced NO actionable, in-scope proposals after fan-out + adversarial vetting.
Write ${findingsPath} with the standard sections (Problem Statement, Current Performance, Sources Consulted, an empty "Proposals (Ranked by Priority)" section noting none survived, and a "Not Recommended" section explaining why). Log a research_failure event via error_tracker.py. Do NOT initialize the agenda (no proposals).
Verify the file exists with Read before finishing.`,
    { agentType: RESEARCH_AGENT, label: 'research:write-empty', phase: 'Synthesize' })
  await verifyFindings('research:write-empty-recheck')
  return { findings_path: findingsPath, proposals: [], agenda_initialized: false }
}

// One research-agent writes the canonical findings file (Step 5) and initializes the
// agenda (Step 5.1) from the already-vetted, ranked proposals — it owns the exact format.
const synth = await agent(
  `${baseCtx}

${OUTPUT_CONTRACT}

These ${ranked.length} proposals have already been fan-out-researched and adversarially vetted/ranked (highest priority first). Do NOT re-search — synthesize and persist them:
${JSON.stringify(ranked, null, 2)}

1. Write ${findingsPath} in the EXACT research-findings.md format (research skill Step 5): Problem Statement, Current Performance, Sources Consulted (tag each source channel), then "## Proposals (Ranked by Priority)" with one "### Proposal N: <title> (Priority: X/10)" block per proposal IN THIS ORDER (index 1..N), each carrying Proposal source, Type, Source, What to change, Expected improvement, Complexity, Risk, Implementation steps, Implementation strategy, and (for from_reference) Reference repo/files. Then Recommendations and Not Recommended.
2. Initialize the research agenda from these proposals (research skill Step 5.1 — error_tracker.py agenda init, or agenda add if one already exists). Use each proposal's priority score and scope.
3. Verify ${findingsPath} exists and is non-empty with the Read tool; re-write if missing.
Return the schema: findings_path, agenda_initialized=true, and the proposals array with each proposal's 1-based index matching its order in the file.`,
  { agentType: RESEARCH_AGENT, label: 'research:synthesize', phase: 'Synthesize', schema: FINDINGS_SCHEMA })

// Prefer the synthesizer's authoritative index mapping; fall back to local ranking if it
// returned nothing (agent skipped/died).
const proposals = (synth && Array.isArray(synth.proposals) && synth.proposals.length)
  ? synth.proposals
  : ranked.map((p, i) => ({
      index: i + 1,
      title: p.title,
      impact: Number(p.impact) || 0,
      confidence: Number(p.confidence) || 0,
      feasibility: Number(p.feasibility) || 1,
      scope: p.scope || scopeLevel,
      type: p.type || 'code_change',
      implementation_strategy: p.implementation_strategy || 'from_scratch',
      files_to_modify: Array.isArray(p.files_to_modify) ? p.files_to_modify : [],
    }))

// Completeness check: confirm the canonical findings file exists before returning.
await verifyFindings('research:verify-findings')

return {
  findings_path: (synth && synth.findings_path) || findingsPath,
  proposals,
  agenda_initialized: synth ? synth.agenda_initialized === true : false,
}
