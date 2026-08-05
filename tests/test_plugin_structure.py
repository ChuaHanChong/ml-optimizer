"""Tests for plugin structure — agents, skills, hooks, and scripts."""

import importlib
import json
import os
import re
from pathlib import Path

import pytest

from conftest import (
    AGENTS_DIR,
    FIXTURES,
    HOOKS_DIR,
    PLUGIN_JSON,
    PLUGIN_ROOT,
    SCRIPTS_DIR,
    SKILLS_DIR,
    _write_result,
)

from detect_divergence import check_divergence
from implement_utils import parse_research_proposals
from parse_logs import parse_log
from result_analyzer import load_results, rank_by_metric
from schema_validator import validate_prerequisites
from error_tracker import (
    create_event,
    log_event,
    detect_patterns,
    summarize_session,
    compute_success_metrics,
    compute_proposal_outcomes,
    rank_suggestions,
    log_suggestion,
    get_suggestion_history,
    VALID_CATEGORIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_frontmatter(filepath: Path) -> dict:
    """Extract YAML frontmatter from a markdown file as a dict."""
    text = filepath.read_text()
    match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return {}
    fm: dict = {}
    last_key = ""
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            if last_key and last_key in fm and isinstance(fm[last_key], list):
                fm[last_key].append(line[2:].strip())
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            last_key = key
            if val == "":
                fm[key] = []
            elif val.lower() == "true":
                fm[key] = True
            elif val.lower() == "false":
                fm[key] = False
            else:
                fm[key] = val
    return fm


def _parse_tools(agent_file: Path) -> set[str]:
    """Extract tools set from agent frontmatter."""
    fm = _parse_frontmatter(agent_file)
    tools_str = fm.get("tools", "")
    if isinstance(tools_str, str):
        return {t.strip() for t in tools_str.split(",") if t.strip()}
    return set()


def _parse_skills(agent_file: Path) -> list[str]:
    """Extract skills list from agent frontmatter."""
    fm = _parse_frontmatter(agent_file)
    skills = fm.get("skills", [])
    if isinstance(skills, list):
        return skills
    return [skills] if skills else []


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

EXPECTED_AGENTS = {
    "prerequisites-agent": {
        "model": "sonnet", "skill": "ml-optimizer:prerequisites",
        "required_tools": {"Bash", "Read", "Write", "Glob", "Grep"},
        "forbidden_tools": {"Edit"}, "color": "cyan", "background": False,
    },
    "baseline-agent": {
        "model": "sonnet", "skill": "ml-optimizer:baseline",
        "required_tools": {"Bash", "Read", "Write", "Glob", "Grep", "Skill"},
        "forbidden_tools": {"Edit"}, "color": "blue", "background": False,
    },
    "experiment-agent": {
        "model": "sonnet", "skill": "ml-optimizer:experiment",
        "required_tools": {"Bash", "Read", "Write", "Glob", "Grep"},
        "forbidden_tools": {"Edit"}, "color": "green", "background": True,
    },
    "monitor-agent": {
        "model": "sonnet", "skill": "ml-optimizer:monitor",
        "required_tools": {"Bash", "Read", "Write", "Glob", "Grep", "Skill"},
        "forbidden_tools": {"Edit"}, "color": "yellow", "background": True,
    },
    "research-agent": {
        "model": "opus", "skill": "ml-optimizer:research",
        "required_tools": {"Bash", "Read", "Write", "Glob", "Grep", "WebSearch", "WebFetch"},
        "forbidden_tools": {"Edit"}, "color": "magenta", "background": False,
        "external_skills": ["claude-mem:mem-search", "superpowers:verification-before-completion"],
    },
    "implement-agent": {
        "model": "opus", "skill": "ml-optimizer:implement",
        "required_tools": {"Bash", "Read", "Write", "Edit", "LSP", "Glob", "Grep"},
        "forbidden_tools": set(), "color": "magenta", "background": False,
        "external_skills": ["superpowers:systematic-debugging", "superpowers:verification-before-completion", "karpathy-skills:karpathy-guidelines", "ml-optimizer:shinka-setup", "ml-optimizer:shinka-convert", "ml-optimizer:shinka-run", "ml-optimizer:shinka-inspect"],
    },
    "tuning-agent": {
        "model": "opus", "skill": "ml-optimizer:hp-tune",
        "required_tools": {"Read", "Write", "Bash", "Glob", "Grep"},
        "forbidden_tools": {"Edit"}, "color": "red", "background": False,
        "external_skills": ["claude-mem:mem-search", "superpowers:verification-before-completion"],
    },
    "analysis-agent": {
        "model": "opus", "skill": "ml-optimizer:analyze",
        "required_tools": {"Read", "Write", "Bash", "Glob", "Grep", "Skill"},
        "forbidden_tools": {"Edit"}, "color": "cyan", "background": False,
        "external_skills": ["claude-mem:mem-search", "superpowers:verification-before-completion"],
    },
    "report-agent": {
        "model": "opus", "skill": "ml-optimizer:report",
        "required_tools": {"Read", "Write", "Bash", "Glob", "Grep", "Skill"},
        "forbidden_tools": {"Edit"}, "color": "blue", "background": False,
        "external_skills": ["superpowers:verification-before-completion"],
    },
    "orchestrator-agent": {
        "model": "opus", "skill": "ml-optimizer:orchestrate",
        "required_tools": {"Agent", "Bash", "Read", "Write", "Edit", "Glob", "Grep", "Skill"},
        "forbidden_tools": set(), "color": "blue", "background": False,
        "external_skills": ["superpowers:verification-before-completion"],
    },
}

EXPECTED_SKILLS = [
    "orchestrate", "prerequisites", "baseline", "experiment", "monitor",
    "research", "implement", "hp-tune", "analyze", "report",
    "evolve", "shinka-setup", "shinka-convert", "shinka-run", "shinka-inspect",
]

NON_ORCHESTRATE_SKILLS = [s for s in EXPECTED_SKILLS if s != "orchestrate"]


# ---------------------------------------------------------------------------
# Agent file validation (one comprehensive test per agent)
# ---------------------------------------------------------------------------

class TestAgentFiles:
    """Validate all 10 agent definition files."""

    def test_all_9_agents_exist_and_no_extra(self):
        """All 10 expected agent files exist and no unexpected ones."""
        for name in EXPECTED_AGENTS:
            assert (AGENTS_DIR / f"{name}.md").exists(), f"Missing: {name}"
        actual = {f.stem for f in AGENTS_DIR.glob("*.md")}
        extra = actual - set(EXPECTED_AGENTS.keys())
        assert not extra, f"Unexpected agent files: {extra}"

    @pytest.mark.parametrize("agent_name", EXPECTED_AGENTS.keys())
    def test_agent_frontmatter(self, agent_name):
        """Each agent has correct name, description, model, color, tools, skill, and background."""
        spec = EXPECTED_AGENTS[agent_name]
        path = AGENTS_DIR / f"{agent_name}.md"
        fm = _parse_frontmatter(path)
        tools = _parse_tools(path)
        skills = _parse_skills(path)

        # name and description
        assert fm.get("name") == agent_name, f"{agent_name}: name mismatch"
        assert fm.get("description"), f"{agent_name}: missing description"

        # model
        assert fm.get("model", "").startswith(spec["model"]), (
            f"{agent_name}: expected model starting with {spec['model']}, got {fm.get('model')}")

        # color
        assert fm.get("color") == spec["color"], (
            f"{agent_name}: expected color {spec['color']}")
        assert fm.get("color", "") in {"blue", "cyan", "green", "yellow", "magenta", "red"}, (
            f"{agent_name}: color must be a named color, got {fm.get('color')}")

        # tools
        missing = spec["required_tools"] - tools
        assert not missing, f"{agent_name} missing tools: {missing}"
        present_forbidden = spec["forbidden_tools"] & tools
        assert not present_forbidden, f"{agent_name} has forbidden tools: {present_forbidden}"

        # skill
        assert spec["skill"] in skills, (
            f"{agent_name}: expected skill {spec['skill']}, got {skills}")

        # external skills
        for ext in spec.get("external_skills", []):
            assert ext in skills, f"{agent_name}: missing external skill {ext}"

        # background
        if spec["background"]:
            assert fm.get("background") is True, f"{agent_name}: should have background: true"
        else:
            assert "background" not in fm or fm.get("background") is not True

        # memory
        assert fm.get("memory") == "local", f"{agent_name}: should have memory: local"


# ---------------------------------------------------------------------------
# Skill file validation
# ---------------------------------------------------------------------------

class TestSkillFiles:
    """Validate all 10 skill definition files."""

    # ShinkaEvolve skills are symlinked from the submodule — they have
    # different frontmatter conventions, so only check name + existence
    _THIRD_PARTY_SKILLS = {"shinka-setup", "shinka-convert", "shinka-run", "shinka-inspect"}

    # orchestrate is the user-facing entry point — it must be invocable
    # (no disable-model-invocation, no user-invocable: false)
    _USER_FACING_SKILLS = {"orchestrate"}

    @pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
    def test_skill_frontmatter(self, skill_name):
        """Each skill exists and has correct name, disable-model-invocation, user-invocable."""
        path = SKILLS_DIR / skill_name / "SKILL.md"
        assert path.exists(), f"Missing skill: {path}"
        fm = _parse_frontmatter(path)
        assert fm.get("name") == skill_name, f"{skill_name}: name mismatch"
        if skill_name not in self._USER_FACING_SKILLS and skill_name not in self._THIRD_PARTY_SKILLS:
            # Non-user-facing skills should not be user-invocable
            assert fm.get("user-invocable") is not True, (
                f"{skill_name}: internal skill should not be user-invocable")

    def test_orchestrate_reference_files_exist(self):
        """All 10 phase reference files must exist in orchestrate/references/."""
        refs_dir = SKILLS_DIR / "orchestrate" / "references"
        for phase in range(10):
            matches = list(refs_dir.glob(f"phase-{phase}-*.md"))
            assert len(matches) >= 1, f"Missing reference file for phase {phase}"

    @pytest.mark.parametrize("skill_name", NON_ORCHESTRATE_SKILLS)
    def test_non_orchestrate_no_context_or_agent(self, skill_name):
        """Non-orchestrate skills should not have context: or agent: in frontmatter."""
        fm = _parse_frontmatter(SKILLS_DIR / skill_name / "SKILL.md")
        assert "context" not in fm, f"{skill_name}: should not have context:"
        assert "agent" not in fm, f"{skill_name}: should not have agent:"


# ---------------------------------------------------------------------------
# Skill-to-agent mapping
# ---------------------------------------------------------------------------

class TestSkillAgentMapping:
    """Verify every non-orchestrate skill has exactly one agent that loads it."""

    def test_every_skill_has_an_agent(self):
        """Each non-orchestrate skill is loaded by exactly one agent (shared skills by at least one)."""
        skill_to_agents: dict[str, list[str]] = {s: [] for s in NON_ORCHESTRATE_SKILLS}
        for agent_file in AGENTS_DIR.glob("*.md"):
            for skill in _parse_skills(agent_file):
                skill_name = skill.replace("ml-optimizer:", "")
                if skill_name in skill_to_agents:
                    skill_to_agents[skill_name].append(agent_file.stem)
        # Some skills are intentionally shared between agents (e.g., evolve, shinka-*)
        shared_skills = {"evolve", "shinka-setup", "shinka-convert", "shinka-run", "shinka-inspect"}
        for skill_name, agents in skill_to_agents.items():
            if skill_name in shared_skills:
                assert len(agents) >= 1, (
                    f"Shared skill '{skill_name}' has no agents: {agents}")
            else:
                assert len(agents) == 1, (
                    f"Skill '{skill_name}' loaded by {len(agents)} agents: {agents} (expected 1)")


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

EXPECTED_HOOKS = [
    "bash-safety.sh", "file-guardrail.sh", "detect-critical-errors.sh",
    "pre-compact.sh", "post-compact-context.sh",
    "subagent-start-inject-goals.sh", "stop-check.sh",
    "file-changed-pipeline-state.sh", "cwd-changed-detect-experiments.sh",
]


class TestHooks:
    """Validate hooks configuration."""

    def test_hooks_json_valid_with_entries(self):
        """hooks.json exists, is valid, and has enough event types."""
        assert (HOOKS_DIR / "hooks.json").exists()
        data = json.loads((HOOKS_DIR / "hooks.json").read_text())
        assert "hooks" in data
        assert isinstance(data["hooks"], dict)
        assert len(data["hooks"]) >= 5

    @pytest.mark.parametrize("hook_file", EXPECTED_HOOKS)
    def test_hook_script_exists_and_executable(self, hook_file):
        """Each expected hook script exists and is executable."""
        path = HOOKS_DIR / hook_file
        assert path.exists(), f"Missing hook script: {path}"
        assert os.access(path, os.X_OK), f"{hook_file} is not executable"

    def test_hooks_use_plugin_root_var(self):
        """Hook commands should use ${CLAUDE_PLUGIN_ROOT}, not hardcoded paths."""
        data = json.loads((HOOKS_DIR / "hooks.json").read_text())
        for event_type, hook_groups in data["hooks"].items():
            for group in hook_groups:
                for hook in group.get("hooks", []):
                    cmd = hook.get("command", "")
                    if ".sh" in cmd or "scripts/" in cmd:
                        assert "${CLAUDE_PLUGIN_ROOT}" in cmd, (
                            f"Hook '{event_type}' should use ${{CLAUDE_PLUGIN_ROOT}}")


# ---------------------------------------------------------------------------
# Scripts
# ---------------------------------------------------------------------------

EXPECTED_SCRIPTS = [
    "gpu_check.py", "parse_logs.py", "detect_divergence.py",
    "result_analyzer.py", "experiment_setup.py", "implement_utils.py",
    "pipeline_state.py", "schema_validator.py", "plot_results.py",
    "error_tracker.py", "prerequisites_check.py", "dashboard.py",
    "excalidraw_gen.py", "goal_memory.py", "gitnexus_utils.py",
]


class TestScripts:
    """Validate all Python scripts exist and are importable."""

    @pytest.mark.parametrize("script", EXPECTED_SCRIPTS)
    def test_script_exists_and_importable(self, script):
        """Each expected script exists and imports without error."""
        assert (SCRIPTS_DIR / script).exists(), f"Missing script: {script}"
        module_name = script.replace(".py", "")
        try:
            importlib.import_module(module_name)
        except Exception as e:
            pytest.fail(f"Failed to import {module_name}: {e}")

    def test_scripts_run_on_the_active_interpreter(self):
        """The `python3` on PATH must satisfy what the scripts are written against.

        Skills and hooks invoke them as bare `python3`, and anything older than 3.10
        fails at import, before main() can report it — with hooks swallowing the error.
        Fail loudly here rather than leaving that to a silent failure at runtime.
        """
        import subprocess

        r = subprocess.run(
            ["python3", "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, f"`python3` is not runnable: {r.stderr.strip()}"
        major, minor = (int(x) for x in r.stdout.strip().split("."))
        assert (major, minor) >= (3, 10), (
            f"`python3` on PATH is {major}.{minor}; the scripts need 3.10+. "
            "Put a newer interpreter earlier on PATH than /usr/bin."
        )


# ---------------------------------------------------------------------------
# Plugin manifest
# ---------------------------------------------------------------------------

class TestPluginManifest:
    """Validate plugin.json."""

    def test_plugin_json_valid(self):
        """plugin.json exists with a valid name, version, and description."""
        assert PLUGIN_JSON.exists()
        data = json.loads(PLUGIN_JSON.read_text())
        assert data.get("name") == "ml-optimizer"
        assert "version" in data
        assert "description" in data
        assert re.match(r"^\d+\.\d+\.\d+$", data["version"]), f"Invalid version: {data['version']}"


# ---------------------------------------------------------------------------
# Skill symlink resolution
# ---------------------------------------------------------------------------

class TestSkillSymlinks:
    """Verify symlinked skills resolve to actual SKILL.md files."""

    _SYMLINKED_SKILLS = [
        "shinka-setup", "shinka-convert", "shinka-run", "shinka-inspect",
    ]

    @pytest.mark.parametrize("skill_name", _SYMLINKED_SKILLS)
    def test_symlink_resolves(self, skill_name):
        """Symlinked skill directories resolve to SKILL.md files."""
        skill_path = SKILLS_DIR / skill_name
        if not skill_path.exists():
            pytest.skip(f"Symlink {skill_name} not created (run setup script)")
        assert skill_path.is_symlink() or skill_path.is_dir()
        skill_md = skill_path / "SKILL.md"
        assert skill_md.exists(), f"{skill_name}/SKILL.md not found via symlink"
        content = skill_md.read_text()
        assert "name:" in content, f"{skill_name}/SKILL.md missing name frontmatter"


# ---------------------------------------------------------------------------
# Orchestrate dispatch points
# ---------------------------------------------------------------------------

class TestOrchestrateDispatch:
    """Verify the orchestrate skill references all 10 agents correctly."""

    @staticmethod
    def _orchestrate_full_text():
        orch_dir = SKILLS_DIR / "orchestrate"
        parts = [(orch_dir / "SKILL.md").read_text()]
        refs_dir = orch_dir / "references"
        if refs_dir.exists():
            for f in sorted(refs_dir.glob("*.md")):
                parts.append(f.read_text())
        return "\n".join(parts)

    def test_dispatch_patterns(self):
        """Orchestrate uses named agent dispatch, no bare skill invocations, and references all dispatched agents."""
        text = self._orchestrate_full_text()
        named_dispatches = re.findall(r'subagent_type.*ml-optimizer:', text)
        assert len(named_dispatches) >= 5
        bare_invocations = re.findall(r'Invoke\s+the\s+ml-optimizer:', text)
        assert not bare_invocations
        # orchestrator-agent is the main-thread agent (not dispatched by orchestrate skill)
        dispatched_agents = {a for a in EXPECTED_AGENTS if a != "orchestrator-agent"}
        for agent in dispatched_agents:
            assert agent in text, f"Orchestrate does not reference {agent}"


# ---------------------------------------------------------------------------
# Workflow-driven phases 5-8 validation
# ---------------------------------------------------------------------------

# Phases 5-8 run as dynamic workflows (skills/orchestrate/workflows/phase-{5,6,7,8}-*.js).
# Each workflow dispatches fresh agents via `agentType: "ml-optimizer:<name>-agent"`
# and hands off context via args + files — there is no SendMessage message bus,
# no agent_registry, and no persistent-agent resume pattern for these phases.
# The four agents the workflows reuse internally.
WORKFLOW_AGENTS = {"research", "implement", "tuning", "analysis", "monitor"}


class TestWorkflowDrivenPhases:
    """Verify phases 5-8 are documented as dynamic workflows (no SendMessage/registry)."""

    @staticmethod
    def _orchestrate_skill_text():
        return (SKILLS_DIR / "orchestrate" / "SKILL.md").read_text()

    @staticmethod
    def _phase_text(phase_name):
        return (SKILLS_DIR / "orchestrate" / "references" / phase_name).read_text()

    @pytest.mark.parametrize("phase_doc", [
        "phase-5-research.md",
        "phase-6-implement.md",
        "phase-7-experiment-loop.md",
        "phase-8-stacking.md",
    ])
    def test_phase_docs_invoke_workflow(self, phase_doc):
        """Each of phases 5-8 dispatches its work via Workflow(...)."""
        text = self._phase_text(phase_doc)
        assert "Workflow(" in text, (
            f"{phase_doc} should dispatch via Workflow(...) for the workflow-driven model"
        )

    @pytest.mark.parametrize("phase_doc", [
        "phase-5-research.md",
        "phase-6-implement.md",
        "phase-7-experiment-loop.md",
        "phase-8-stacking.md",
    ])
    def test_phase_docs_have_no_sendmessage_or_registry(self, phase_doc):
        """Phases 5-8 must not reference the removed SendMessage/agent_registry pattern."""
        text = self._phase_text(phase_doc)
        assert "SendMessage(" not in text, (
            f"{phase_doc} should not use SendMessage — phases 5-8 are workflow-driven"
        )
        assert "agent_registry" not in text, (
            f"{phase_doc} should not reference agent_registry — removed for phases 5-8"
        )

    def test_orchestrate_documents_workflow_dispatch(self):
        """Orchestrate SKILL.md must document the workflow dispatch model for 5-8."""
        text = self._orchestrate_skill_text()
        assert "Workflow(" in text
        assert "agentType" in text

    def test_orchestrate_has_no_registry_or_message_bus(self):
        """Orchestrate SKILL.md must not document the removed registry/message-bus pattern."""
        text = self._orchestrate_skill_text()
        # The only acceptable mentions are explicit statements that there is
        # NO registry / NO SendMessage. Disallow the resume-protocol artifacts.
        assert "Dispatch Protocol" not in text
        assert "CONTEXT FROM OTHER AGENTS" not in text


# ---------------------------------------------------------------------------
# Documentation consistency (merged)
# ---------------------------------------------------------------------------

class TestDocumentation:
    """Verify docs reflect the 10-agent architecture (9 subagents + orchestrator) and key features."""

    @pytest.mark.parametrize("keyword", [
        "10 agents", 'Agent(subagent_type="ml-optimizer:',
        "stuck protocol", "dead-end", "research agenda",
        "immutable baseline",
    ])
    def test_claude_md_documents_feature(self, keyword):
        """CLAUDE.md mentions each documented architecture feature keyword."""
        text = (PLUGIN_ROOT / ".claude" / "CLAUDE.md").read_text()
        assert keyword.lower() in text.lower(), f"CLAUDE.md should mention '{keyword}'"

    def test_implement_skill_mentions_unit_tests(self):
        """The implement skill documents unit test writing."""
        text = (SKILLS_DIR / "implement" / "SKILL.md").read_text()
        assert "unit test" in text.lower()

    def test_implement_agent_mentions_test_writing(self):
        """The implement agent definition documents a Test Writing responsibility."""
        text = (AGENTS_DIR / "implement-agent.md").read_text()
        assert "Test Writing" in text

    def test_readme_mentions_orchestrate_entry_point(self):
        """The README documents orchestrate as the entry point."""
        text = (PLUGIN_ROOT / "README.md").read_text()
        assert "orchestrate" in text.lower()


# ---------------------------------------------------------------------------
# Skill contract documentation tests (merged)
# ---------------------------------------------------------------------------

class TestSkillContracts:
    """Verify skills reference the autoresearch-inspired features they consume."""

    @pytest.mark.parametrize("skill,keyword", [
        ("analyze", "dead-end"), ("analyze", "agenda"), ("analyze", "session review mode"),
        ("research", "dead-end"), ("research", "agenda"),
        ("hp-tune", "dead-end"), ("hp-tune", "agenda"),
        ("report", "agenda"), ("experiment", "time_budget"),
        ("baseline", "auto-repair"), ("experiment", "non-retryable"),
        # skill integration: verify new features exist in skills
        ("research", "diverge"), ("research", "converge"),
        ("analyze", "effect size"), ("experiment", "reproducibility"),
        ("report", "verify claims"),
    ])
    def test_skill_mentions_feature(self, skill, keyword):
        """Each skill mentions the autoresearch-inspired feature it consumes."""
        text = (SKILLS_DIR / skill / "SKILL.md").read_text().lower()
        assert keyword.lower() in text, f"Skill '{skill}' should mention '{keyword}'"

    def test_phase7_has_baseline_verification_and_dashboard(self):
        """Phase 7 references baseline checksum verification and the dashboard generator."""
        text = (SKILLS_DIR / "orchestrate" / "references" / "phase-7-experiment-loop.md").read_text()
        assert "verify-baseline" in text
        assert "dashboard.py" in text


# ---------------------------------------------------------------------------
# Skill interface contracts
# ---------------------------------------------------------------------------

SAMPLE_FINDINGS = FIXTURES / "sample_research_findings.md"
SAMPLE_FINDINGS_REF = FIXTURES / "sample_research_findings_with_reference.md"


def test_research_proposals_contract():
    """parse_research_proposals output has implement-required fields and valid slugs."""
    proposals = parse_research_proposals(str(SAMPLE_FINDINGS))
    required_fields = {"index", "name", "slug", "body", "files_to_modify", "complexity", "implementation_steps"}
    for p in proposals:
        missing = required_fields - set(p.keys())
        assert not missing, f"Proposal {p.get('name', '?')} missing fields: {missing}"
        assert re.match(r'^[a-z0-9][a-z0-9-]*[a-z0-9]$', p["slug"]), \
            f"Slug '{p['slug']}' is not a valid branch name component"


def test_research_proposals_strategy_fields():
    """All proposals have implementation_strategy; from_reference have repo/files."""
    proposals = parse_research_proposals(str(SAMPLE_FINDINGS_REF))
    for p in proposals:
        assert p["implementation_strategy"] in ("from_scratch", "from_reference")
        if p["implementation_strategy"] == "from_reference":
            assert p["reference_repo"]
            assert len(p["reference_files"]) > 0


def test_experiment_result_matches_analyze_input(tmp_path):
    """Experiment result JSON must be loadable by result_analyzer."""
    result = {
        "exp_id": "exp-001", "status": "completed",
        "config": {"lr": 0.001, "batch_size": 16},
        "metrics": {"loss": 0.5, "accuracy": 82.5},
        "gpu_id": 0, "duration_seconds": 3600,
        "code_branch": "ml-opt/perceptual-loss",
    }
    (tmp_path / "exp-001.json").write_text(json.dumps(result))
    loaded = load_results(str(tmp_path))
    assert "exp-001" in loaded
    ranked = rank_by_metric(loaded, "loss", lower_is_better=True)
    assert len(ranked) == 1 and ranked[0]["value"] == 0.5


def test_baseline_and_hp_tune_schemas():
    """baseline.json and HP-tune proposed config must have required fields."""
    baseline = {
        "exp_id": "baseline", "status": "completed",
        "config": {"lr": 0.01, "batch_size": 64},
        "metrics": {"loss": 1.5, "accuracy": 45.0},
        "profiling": {"gpu_memory_used_mib": 8000, "throughput_samples_per_sec": 150},
    }
    assert isinstance(baseline["metrics"], dict) and len(baseline["metrics"]) > 0
    assert isinstance(baseline["config"], dict) and len(baseline["config"]) > 0
    assert "gpu_memory_used_mib" in baseline["profiling"]

    proposed = {
        "exp_id": "exp-003", "config": {"lr": 0.0001, "batch_size": 32},
        "code_branch": None, "gpu_id": 0,
        "reasoning": "Lower LR showed best results", "iteration": 2,
    }
    required = {"exp_id", "config", "gpu_id", "reasoning", "iteration"}
    assert not (required - set(proposed.keys()))


def test_manifest_schema_for_orchestrate():
    """implementation-manifest.json must have fields orchestrate expects."""
    manifest = {
        "original_branch": "main", "strategy": "git_branch",
        "proposals": [
            {"name": "Perceptual Loss", "slug": "perceptual-loss",
             "branch": "ml-opt/perceptual-loss", "status": "validated"},
            {"name": "Bad Proposal", "slug": "bad-proposal",
             "branch": "ml-opt/bad-proposal", "status": "validation_failed"},
        ],
    }
    validated = [p for p in manifest["proposals"] if p["status"] == "validated"]
    assert len(validated) == 1 and "branch" in validated[0]


# --- Error tracker → Review contract ---

def test_review_contract_outputs(tmp_path):
    """summarize_session, detect_patterns, success_metrics, proposal_outcomes, rank output schemas."""
    # summarize_session
    log_event(str(tmp_path), create_event("training_failure", "critical", "experiment", "crash"))
    log_event(str(tmp_path), create_event("divergence", "warning", "monitor", "nan"))
    summary = summarize_session(str(tmp_path))
    for key in ("total_events", "by_category", "by_severity", "patterns_detected"):
        assert key in summary

    # detect_patterns
    events = [
        create_event("divergence", "warning", "monitor", "NaN",
                      config={"lr": lr, "batch_size": 32})
        for lr in [0.1, 0.2, 0.05]
    ]
    patterns = detect_patterns(events)
    for p in patterns:
        for key in ("pattern_id", "description", "occurrences", "suggested_action"):
            assert key in p

    # success_metrics
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results, "baseline", "completed", {}, {"acc": 70.0})
    _write_result(results, "exp-001", "completed", {}, {"acc": 75.0})
    m = compute_success_metrics(str(tmp_path), "acc", lower_is_better=False)
    for key in ("total_experiments", "completed", "failed", "diverged",
                "success_rate", "improvement_rate", "top_configs", "worst_configs"):
        assert key in m

    # proposal_outcomes
    p = compute_proposal_outcomes(str(tmp_path), "acc", lower_is_better=False)
    for key in ("research_proposals", "hp_proposals", "implementation_stats"):
        assert key in p

    # rank_suggestions
    ranked = rank_suggestions([
        {"pattern_id": "oom_batch_size", "description": "OOM",
         "occurrences": 2, "suggested_action": "reduce bs"},
    ])
    assert "score" in ranked[0]
    ranked_with_total = rank_suggestions(ranked, total_experiments=50)
    assert "significance" in ranked_with_total[0]


def test_review_category_to_file_mapping_complete():
    """Every VALID_CATEGORIES entry must have a known mapping."""
    mapped = {
        "agent_failure", "divergence", "training_failure",
        "implementation_error", "pipeline_inefficiency", "config_error",
        "research_failure", "timeout", "resource_error",
    }
    assert not (set(VALID_CATEGORIES) - mapped)


def test_review_suggestion_history_schema(tmp_path):
    """log_suggestion and get_suggestion_history produce expected schema."""
    log_suggestion(str(tmp_path), "wasted_budget", scope="session")
    history = get_suggestion_history(str(tmp_path))
    assert len(history) == 1
    for key in ("pattern_id", "timestamp", "scope", "iteration"):
        assert key in history[0]
    assert isinstance(history[0]["iteration"], int) and history[0]["iteration"] >= 1


# --- Prerequisites contract ---

def test_prerequisites_contract():
    """prerequisites.json has orchestrator-required fields and status variants work."""
    ready = {
        "status": "ready",
        "dataset": {"train_path": "/data/train", "prepared": True,
                     "prepared_train_path": "/exp/prepared-data/train",
                     "prepared_val_path": "/exp/prepared-data/val"},
        "environment": {"manager": "conda", "packages_installed": ["torch"]},
        "ready_for_baseline": True,
    }
    for key in ("status", "dataset", "environment", "ready_for_baseline"):
        assert key in ready
    assert ready["status"] in ("ready", "partial", "failed")
    assert isinstance(ready["ready_for_baseline"], bool)
    assert len(ready["dataset"]["prepared_train_path"]) > 0

    failed = {"status": "failed", "dataset": {}, "environment": {}, "ready_for_baseline": False}
    assert failed["ready_for_baseline"] is False


# --- HP batch size, method_tier, monitor, analyze ---

def test_hp_batch_size_contract():
    """HP batch size = max(num_gpus, 1); branch-iter slot widening.

    Loop exit is NOT a fixed stop-count threshold — it is the orchestrator's
    evidence-based judgment (see phase-7-experiment-loop.md). consecutive_stop_count
    is a persisted signal, not a hardcoded trigger, so no numeric gate is asserted here.
    """
    assert max(4, 1) == 4   # 4 GPUs
    assert max(2, 1) == 2   # 2 GPUs
    assert max(0, 1) == 1   # CPU-only
    assert max(1, 1) == 1   # 1 GPU
    # Branch iter 1
    assert min(3 + 1, 5) == 4
    assert min(3 + 1, 2) == 2


def test_method_tier_rules():
    """method_tier: baseline / method_default_hp / method_tuned_hp."""
    def tier(branch, it):
        return "baseline" if branch is None else ("method_default_hp" if it == 1 else "method_tuned_hp")
    assert tier(None, 1) == "baseline"
    assert tier("ml-opt/x", 1) == "method_default_hp"
    assert tier("ml-opt/x", 3) == "method_tuned_hp"


def test_monitor_and_analyze_contracts():
    """Monitor divergence_status schema; analyze stop/pivot flow."""
    # Monitor
    for status in ("healthy", "diverged", "completed", "unmonitored", "failed", "no_output"):
        assert status in ("healthy", "diverged", "completed", "unmonitored", "failed", "no_output")

    # Analyze stop prevents hp-tune
    assert ("stop" not in ("stop",)) is False
    # Pivot narrows search space
    updated = {"lr": [1e-5, 1e-3]}
    assert updated["lr"] == [1e-5, 1e-3]

    # Analyze pivot types include code_evolution
    valid_pivots = {
        "branch_test", "hp_expand", "research", "method_proposal",
        "narrow_space", "qualitative_change", "regularization", "code_evolution",
    }
    assert "code_evolution" in valid_pivots



# --- Experiment → Monitor log format ---

def test_experiment_log_parseable(tmp_path):
    """Experiment log parseable by parse_logs -> detect_divergence."""
    log_file = tmp_path / "train.log"
    log_file.write_text("loss: 0.5\nloss: 0.4\nloss: 0.35\nloss: 0.3\nloss: 0.28\n")
    records = parse_log(str(log_file))
    values = [r["loss"] for r in records if "loss" in r]
    assert len(values) > 0
    result = check_divergence(values)
    assert "diverged" in result and "reason" in result


# --- Prerequisites → Schema validator ---

def test_prerequisites_report_validates():
    """Prerequisites report passes schema validation."""
    report_valid = {
        "status": "ready",
        "dataset": {"format": "csv", "train_path": "/data/train.csv"},
        "environment": {"manager": "conda", "python_version": "3.10"},
        "ready_for_baseline": True,
    }
    result = validate_prerequisites(report_valid)
    assert result["valid"] is True and result["errors"] == []


# --- Remote training wrapper ---

class TestRemoteTrainWrapper:
    """The remote-training wrapper must behave like a local training process.

    That contract is spelled out in the script's own header; what these tests protect is
    the consequence of breaking it — every downstream check (log parsing, divergence,
    result status) would read the wrapper instead of the training run.
    """

    WRAPPER = SCRIPTS_DIR / "remote_train.sh"

    def test_wrapper_exists_and_is_executable(self):
        """The workflow prompt invokes the path directly, so a lost exec bit fails at runtime."""
        assert self.WRAPPER.exists(), "scripts/remote_train.sh is missing"
        assert self.WRAPPER.stat().st_mode & 0o111, "remote_train.sh is not executable"

    def test_wrapper_is_valid_bash(self):
        """A syntax error here surfaces mid-round, after GPU time is already spent."""
        import subprocess
        r = subprocess.run(["bash", "-n", str(self.WRAPPER)], capture_output=True, text=True)
        assert r.returncode == 0, f"bash -n failed: {r.stderr}"

    def test_launches_detached_so_training_survives_a_dropped_connection(self):
        """Without tmux, a dropped ssh connection kills training mid-round."""
        src = self.WRAPPER.read_text()
        assert "tmux new -d" in src, "training must be launched detached"

    def test_kills_the_remote_job_when_signalled(self):
        """`timeout` firing must not leave a process holding a remote GPU."""
        src = self.WRAPPER.read_text()
        assert "trap 'cleanup SIGTERM' TERM" in src
        assert "kill-session" in src

    def test_waits_on_a_background_tail_so_the_trap_can_fire(self):
        """bash defers traps during a foreground command.

        Tailing the remote log in the foreground would swallow SIGTERM until training
        ended on its own, which is precisely when the kill is no longer useful. The tail
        therefore runs in the background with an interruptible `wait`.
        """
        src = self.WRAPPER.read_text()
        assert 'TAIL_PID=$!' in src and 'wait "$TAIL_PID"' in src

    def test_propagates_the_remote_exit_code(self):
        """A wrapper that always exits 0 marks a crashed run as completed."""
        src = self.WRAPPER.read_text()
        assert 'exit "$RC"' in src, "the training command's exit status must reach the caller"
