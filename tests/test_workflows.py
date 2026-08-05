"""Validate the dynamic workflow scripts for phases 5-8.

Phases 5-8 run as dynamic workflows held in
`skills/orchestrate/workflows/phase-{5,6,7,8}-*.js`. They live inside the skill
folder (not `.claude/workflows/`) so they stay out of the user `/slash-command`
namespace and are launched by the orchestrator via `scriptPath`.
Each workflow script:
  - exists on disk,
  - begins with `export const meta` (the workflow contract),
  - is valid ESM (checked with `node --input-type=module --check` when node is
    available; skipped otherwise),
and the corresponding orchestrate phase reference doc dispatches it via
`Workflow(` (scriptPath form).

Run:
    python -m pytest tests/test_workflows.py -v
"""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = PLUGIN_ROOT / "skills" / "orchestrate" / "workflows"
REFERENCES_DIR = PLUGIN_ROOT / "skills" / "orchestrate" / "references"

# Map each workflow phase number to (workflow filename, orchestrate phase doc).
WORKFLOW_PHASES = {
    5: ("phase-5-research.js", "phase-5-research.md"),
    6: ("phase-6-implement.js", "phase-6-implement.md"),
    7: ("phase-7-experiment.js", "phase-7-experiment-loop.md"),
    8: ("phase-8-stacking.js", "phase-8-stacking.md"),
}

WORKFLOW_FILES = [fname for fname, _ in WORKFLOW_PHASES.values()]

_NODE = shutil.which("node")


# ---------------------------------------------------------------------------
# Existence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("workflow_file", WORKFLOW_FILES)
def test_workflow_file_exists(workflow_file):
    """Each phase-{5,6,7,8} workflow script exists in skills/orchestrate/workflows/."""
    path = WORKFLOWS_DIR / workflow_file
    assert path.is_file(), f"Missing workflow script: {path}"


def test_no_unexpected_workflow_files():
    """Only the four phase-5/6/7/8 workflow scripts live in skills/orchestrate/workflows/."""
    assert WORKFLOWS_DIR.is_dir(), f"Missing workflows dir: {WORKFLOWS_DIR}"
    actual = {p.name for p in WORKFLOWS_DIR.glob("*.js")}
    extra = actual - set(WORKFLOW_FILES)
    assert not extra, f"Unexpected workflow scripts: {extra}"


# ---------------------------------------------------------------------------
# Workflow contract: begins with `export const meta`
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("workflow_file", WORKFLOW_FILES)
def test_workflow_begins_with_export_const_meta(workflow_file):
    """Each workflow script begins with the `export const meta` contract."""
    text = (WORKFLOWS_DIR / workflow_file).read_text()
    assert text.lstrip().startswith("export const meta"), (
        f"{workflow_file} must begin with 'export const meta' (workflow contract)"
    )


# ---------------------------------------------------------------------------
# Syntax check (requires node)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("workflow_file", WORKFLOW_FILES)
def test_workflow_passes_node_syntax_check(workflow_file):
    """Each workflow script passes `node --check`.

    Workflow scripts begin with `export const meta` (ESM module markers) but
    the Workflow runtime wraps the script body in an async function, so they
    legitimately use top-level `return`/`await`. A bare
    `node --input-type=module --check` rejects that valid runtime style with
    "Illegal return statement"; `node --check <file>` validates the syntax
    while permitting the runtime-wrapped top-level statements. Skipped when
    node is not available in the environment.
    """
    if _NODE is None:
        pytest.skip("node not available — cannot run syntax check")
    path = WORKFLOWS_DIR / workflow_file
    result = subprocess.run(
        [_NODE, "--check", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"{workflow_file} failed node --check syntax validation:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Orchestrate phase docs dispatch via Workflow(...)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phase_num", sorted(WORKFLOW_PHASES))
def test_phase_doc_dispatches_via_workflow(phase_num):
    """The phase-{5,6,7,8} orchestrate reference doc dispatches via Workflow(...).

    The workflow scripts moved into the skill folder, so the orchestrator
    launches them by `scriptPath` (keeping them out of the `/slash-command`
    namespace) rather than by saved `name`. Accept either the scriptPath form
    (preferred, referencing skills/orchestrate/workflows/) or the legacy
    `name:` form.
    """
    workflow_file, doc_name = WORKFLOW_PHASES[phase_num]
    path = REFERENCES_DIR / doc_name
    assert path.is_file(), f"Missing orchestrate phase doc: {path}"
    text = path.read_text()
    assert "Workflow(" in text, (
        f"{doc_name} should dispatch phase {phase_num} via Workflow(...)"
    )
    workflow_name = workflow_file.removesuffix(".js")
    has_script_path = (
        "scriptPath" in text
        and f"skills/orchestrate/workflows/{workflow_file}" in text
    )
    has_legacy_name = f'name: "{workflow_name}"' in text
    assert has_script_path or has_legacy_name, (
        f"{doc_name} should dispatch phase {phase_num} via Workflow(...) using "
        f"the scriptPath form (skills/orchestrate/workflows/{workflow_file}) "
        f'or the legacy name form (name: "{workflow_name}")'
    )


# ---------------------------------------------------------------------------
# Batch B: model_category / divergence threading + metric fallback deletion
# ---------------------------------------------------------------------------


def _wf(name):
    return (WORKFLOWS_DIR / name).read_text()


def test_phase7_model_category_arg_precedence():
    """Phase 7 reads model_category from args first, baseline pre-loop second."""
    assert "model_category || pre.model_category" in _wf("phase-7-experiment.js")


def test_phase7_prompts_thread_model_category():
    """model_category reaches tuning + analysis + experiment prompts (plus the
    two research prompts that already carried it: 5 total)."""
    assert _wf("phase-7-experiment.js").count("model_category: ${modelCategory}") >= 5


def test_phase8_divergence_args_and_clause():
    """Phase 8 takes divergence args and appends the clause to all three
    experiment prompts (stack run, evolved run, HP-tune runs)."""
    text = _wf("phase-8-stacking.js")
    for arg in ("divergence_metric", "divergence_lower_is_better", "model_category"):
        assert arg in text, arg
    assert "const divergenceClause" in text
    assert text.count("${divergenceClause}") >= 3


def test_phase8_no_first_numeric_metric_fallback():
    """primaryMetricOf must return null (safe skip) + loud log, never the
    first numeric metric in the dict."""
    text = _wf("phase-8-stacking.js")
    assert "function metricOf" not in text
    assert "missing or non-numeric" in text


def test_args_contracts_document_model_category():
    """Both phase docs and the orchestrate SKILL.md document the new args."""
    p7 = (REFERENCES_DIR / "phase-7-experiment-loop.md").read_text()
    p8 = (REFERENCES_DIR / "phase-8-stacking.md").read_text()
    skill = (WORKFLOWS_DIR.parent / "SKILL.md").read_text()
    assert "divergence_lower_is_better, model_category" in p7
    assert "divergence_lower_is_better, model_category" in p8
    assert skill.count("divergence_lower_is_better, model_category") >= 2  # rows + launch snippets


# ---------------------------------------------------------------------------
# Batch B: RL budget unit (fixed_step_budget)
# ---------------------------------------------------------------------------


def test_budget_clauses_support_fixed_step_budget():
    for fname in ("phase-7-experiment.js", "phase-8-stacking.js"):
        text = _wf(fname)
        assert "fixed_step_budget" in text, fname
        assert "total_timesteps" in text, fname


def test_docs_document_fixed_step_budget():
    for doc in (REFERENCES_DIR / "phase-7-experiment-loop.md",
                REFERENCES_DIR / "phase-8-stacking.md",
                WORKFLOWS_DIR.parent / "SKILL.md"):
        assert "fixed_step_budget" in doc.read_text(), doc.name


def test_workflow_prompts_state_poll_cadence():
    """Both workflows' divergenceClauses state the in-run polling cadence
    and the non-kill plateaued handling."""
    for fname in ("phase-7-experiment.js", "phase-8-stacking.js"):
        text = _wf(fname)
        assert "every 5 minutes" in text, fname
        assert "plateaued" in text, fname


# ---------------------------------------------------------------------------
# Batch C: research-based HP priors in the phase-7 workflow
# ---------------------------------------------------------------------------


def test_phase7_preloop_seeds_rl_priors_from_baseline():
    """Pre-loop search-space seeding uses baseline-captured RL HPs for model_category=rl."""
    text = (WORKFLOWS_DIR / "phase-7-experiment.js").read_text()
    assert "gamma, clip_range, ent_coef, n_steps" in text


def test_phase7_regularization_line_names_rl_knobs():
    """The analysis decision-tree regularization line names the RL regularizers."""
    text = (WORKFLOWS_DIR / "phase-7-experiment.js").read_text()
    assert "entropy coefficient / KL penalty" in text


def test_phase7_routes_hp_tune_research_request():
    """hp-tune's research_requested flag routes one mid-loop research round."""
    text = (WORKFLOWS_DIR / "phase-7-experiment.js").read_text()
    assert "research_requested" in text
    assert "tuneRes.research_requested === true" in text


def test_phase7_threads_seeds_per_config():
    """seeds_per_config is read from args and passed to the tuning-agent prompt."""
    text = (WORKFLOWS_DIR / "phase-7-experiment.js").read_text()
    assert "seeds_per_config" in text
    assert "seeds_per_config: ${seedsPerConfig" in text


# ---------------------------------------------------------------------------
# Phase 5 research angles (Batch E, Task 22)
# ---------------------------------------------------------------------------


class TestPhase5ResearchAngles:
    """ANGLES_BY_CATEGORY covers vla; unknown categories fall back with a log."""

    def test_vla_angle_entry(self):
        text = (WORKFLOWS_DIR / "phase-5-research.js").read_text()
        assert "vla:" in text
        assert "action chunking" in text

    def test_unknown_category_fallback_logged(self):
        text = (WORKFLOWS_DIR / "phase-5-research.js").read_text()
        assert "unknown model_category" in text


def test_phase7_threads_secondary_metrics():
    """secondary_metrics flow from args into the analysis-agent prompt."""
    text = (WORKFLOWS_DIR / "phase-7-experiment.js").read_text()
    assert "secondary_metrics" in text
    assert "secondary_metrics: ${JSON.stringify(secondaryMetrics)}" in text


# ---------------------------------------------------------------------------
# Phase 7 CPU-bound parallelism (Batch E, Task 26)
# ---------------------------------------------------------------------------


class TestPhase7CpuParallelism:
    """experiments_per_gpu multiplies batch size; CPU cores are sliced via env_vars."""

    def test_configs_per_batch_uses_experiments_per_gpu(self):
        text = (WORKFLOWS_DIR / "phase-7-experiment.js").read_text()
        assert "experiments_per_gpu" in text
        assert "numGpus * experimentsPerGpu" in text

    def test_cpu_core_slice_via_env_vars(self):
        text = (WORKFLOWS_DIR / "phase-7-experiment.js").read_text()
        assert "OMP_NUM_THREADS" in text
        assert "taskset" in text


# ---------------------------------------------------------------------------
# Real-execution coverage: actually run the literal arg-derivation lines with
# node, instead of just asserting the string is present in the source. This
# catches precedence/typo/evaluation bugs a substring check cannot.
# ---------------------------------------------------------------------------

_DERIVATION_PATTERNS = {
    "evalTasks": r"const evalTasks = .*?;",
    "seedsPerConfig": r"const seedsPerConfig = .*?;",
    "modelCategory": r"const modelCategory = .*?;",
}


def _extract_derivation_lines():
    """Pull the exact literal derivation lines out of phase-7-experiment.js.

    Fails loudly (not skip) if the source moved — a skip here would silently
    defeat the point of proving real execution against the current source.
    """
    text = _wf("phase-7-experiment.js")
    lines = {}
    for name, pattern in _DERIVATION_PATTERNS.items():
        match = re.search(pattern, text)
        assert match is not None, (
            f"{name} derivation line not found in phase-7-experiment.js — did the source move?"
        )
        lines[name] = match.group(0)
    return lines


def _run_derivation(A, pre):
    """Execute the extracted lines verbatim under real node for given A/pre."""
    if _NODE is None:
        pytest.skip("node not available — cannot execute real derivation lines")
    lines = _extract_derivation_lines()
    script = (
        f"const A = {json.dumps(A)};\n"
        f"const pre = {json.dumps(pre)};\n"
        f"{lines['evalTasks']}\n"
        f"{lines['seedsPerConfig']}\n"
        f"{lines['modelCategory']}\n"
        "console.log(JSON.stringify({evalTasks, seedsPerConfig, modelCategory}));\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write(script)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [_NODE, tmp_path], capture_output=True, text=True, timeout=15,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    assert result.returncode == 0, (
        f"node execution of extracted derivation lines failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return json.loads(result.stdout.strip())


class TestArgDerivationRealExecution:
    """Actually execute the 3 literal arg-derivation lines from
    phase-7-experiment.js under node, for real A/pre inputs, and assert on
    the real output — not on substring presence in the source text."""

    def test_model_category_arg_takes_precedence_over_pre(self):
        out = _run_derivation({"model_category": "vla"}, {"model_category": "rl"})
        assert out["modelCategory"] == "vla"

    def test_model_category_falls_back_to_pre_when_arg_missing(self):
        out = _run_derivation({}, {"model_category": "rl"})
        assert out["modelCategory"] == "rl"

    def test_model_category_falls_back_to_pre_when_arg_null(self):
        out = _run_derivation({"model_category": None}, {"model_category": "rl"})
        assert out["modelCategory"] == "rl"

    def test_model_category_null_when_neither_set(self):
        out = _run_derivation({}, {})
        assert out["modelCategory"] is None

    def test_eval_tasks_real_array_kept_as_is(self):
        out = _run_derivation({"eval_tasks": ["pick", "place"]}, {})
        assert out["evalTasks"] == ["pick", "place"]

    def test_eval_tasks_absent_becomes_empty_list(self):
        out = _run_derivation({}, {})
        assert out["evalTasks"] == []

    def test_eval_tasks_null_becomes_empty_list(self):
        out = _run_derivation({"eval_tasks": None}, {})
        assert out["evalTasks"] == []

    def test_eval_tasks_non_array_string_becomes_empty_list(self):
        out = _run_derivation({"eval_tasks": "pick"}, {})
        assert out["evalTasks"] == []

    def test_seeds_per_config_number_kept(self):
        out = _run_derivation({"seeds_per_config": 3}, {})
        assert out["seedsPerConfig"] == 3

    def test_seeds_per_config_absent_becomes_null(self):
        out = _run_derivation({}, {})
        assert out["seedsPerConfig"] is None


# ---------------------------------------------------------------------------
# Phase 7 remote GPU execution
# ---------------------------------------------------------------------------

class TestPhase7RemoteExecution:
    """Phase 7 must be able to send training to a remote GPU host.

    The `remote` object from Phase 0 (`{host, workdir, env_python}`) is the only switch:
    present means experiments wrap their training command in remote_train.sh, absent means
    everything stays local. A silent failure here is expensive — experiments would run on
    the orchestrating machine, which in a remote setup has no GPU.
    """

    SRC = (WORKFLOWS_DIR / "phase-7-experiment.js").read_text()

    def test_remote_is_read_from_args_and_requires_a_host(self):
        assert "const remote = A.remote && A.remote.host ? A.remote : null;" in self.SRC

    def test_local_runs_are_unchanged(self):
        """Without `remote`, the injected block must be empty — no prompt drift."""
        m = re.search(r'const remoteBlock = remote\s*\?(.*?)\n  : "";', self.SRC, re.S)
        assert m, "remoteBlock ternary missing"
        assert m.group(0).rstrip().endswith(': "";')

    def test_remote_block_carries_every_field_the_wrapper_needs(self):
        m = re.search(r'const remoteBlock = remote\s*\?(.*?)\n  : "";', self.SRC, re.S)
        branch = m.group(1)
        for needle in ("remote_train.sh", "--host ${remote.host}",
                       "--env-python ${remote.env_python}", "--gpu <gpu_id>"):
            assert needle in branch, f"remote branch is missing {needle}"

    def test_gpu_scheduling_targets_the_remote_host(self):
        """Scheduling against the local box would find zero GPUs and stall the round."""
        assert "gpu_check.py 30 80 ${remote.host}" in self.SRC

    def test_remote_block_reaches_the_experiment_prompt(self):
        """Defining the block without interpolating it is the likely refactor slip."""
        assert "${cpuClause}${remoteBlock}" in self.SRC
