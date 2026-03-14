"""Comprehensive plugin structure validation.

Validates all 10 agents, 11 skills, hooks, and scripts are correctly
configured for the agent-based dispatch architecture. Run anytime:

    python -m pytest tests/test_plugin_structure.py -v
"""

import json
import re
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).parent.parent
AGENTS_DIR = PLUGIN_ROOT / "agents"
SKILLS_DIR = PLUGIN_ROOT / "skills"
HOOKS_DIR = PLUGIN_ROOT / "hooks"
SCRIPTS_DIR = PLUGIN_ROOT / "scripts"
PLUGIN_JSON = PLUGIN_ROOT / ".claude-plugin" / "plugin.json"


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
            # YAML list item — append to last key
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
    # Procedural agents (model: sonnet)
    "prerequisites-agent": {
        "model": "sonnet",
        "skill": "ml-optimizer:prerequisites",
        "required_tools": {"Bash", "Read", "Write", "Glob", "Grep"},
        "forbidden_tools": {"Edit"},
    },
    "baseline-agent": {
        "model": "sonnet",
        "skill": "ml-optimizer:baseline",
        "required_tools": {"Bash", "Read", "Write", "Glob", "Grep", "Skill"},
        "forbidden_tools": {"Edit"},
    },
    "experiment-agent": {
        "model": "sonnet",
        "skill": "ml-optimizer:experiment",
        "required_tools": {"Bash", "Read", "Write", "Glob", "Grep"},
        "forbidden_tools": {"Edit"},
    },
    "monitor-agent": {
        "model": "sonnet",
        "skill": "ml-optimizer:monitor",
        "required_tools": {"Bash", "Read", "Write", "Glob", "Grep", "Skill"},
        "forbidden_tools": {"Edit"},
    },
    # Analytical agents (model: opus)
    "research-agent": {
        "model": "opus",
        "skill": "ml-optimizer:research",
        "required_tools": {"Bash", "Read", "Write", "Glob", "Grep", "WebSearch", "WebFetch"},
        "forbidden_tools": {"Edit"},
    },
    "implement-agent": {
        "model": "opus",
        "skill": "ml-optimizer:implement",
        "required_tools": {"Bash", "Read", "Write", "Edit", "Glob", "Grep"},
        "forbidden_tools": set(),
    },
    "tuning-agent": {
        "model": "opus",
        "skill": "ml-optimizer:hp-tune",
        "required_tools": {"Read", "Write", "Bash", "Glob", "Grep"},
        "forbidden_tools": {"Edit"},
    },
    "analysis-agent": {
        "model": "opus",
        "skill": "ml-optimizer:analyze",
        "required_tools": {"Read", "Write", "Bash", "Glob", "Grep", "Skill"},
        "forbidden_tools": {"Edit"},
    },
    "report-agent": {
        "model": "opus",
        "skill": "ml-optimizer:report",
        "required_tools": {"Read", "Write", "Bash", "Glob", "Grep", "Skill"},
        "forbidden_tools": {"Edit"},
    },
    "review-agent": {
        "model": "opus",
        "skill": "ml-optimizer:review",
        "required_tools": {"Read", "Write", "Bash", "Glob", "Grep", "Skill"},
        "forbidden_tools": {"Edit"},
    },
}

PROCEDURAL_AGENTS = {"prerequisites-agent", "baseline-agent", "experiment-agent", "monitor-agent"}
ANALYTICAL_AGENTS = {"research-agent", "implement-agent", "tuning-agent", "analysis-agent", "report-agent", "review-agent"}

EXPECTED_COLORS = {
    "prerequisites-agent": "#6B7280",
    "baseline-agent": "#3B82F6",
    "experiment-agent": "#10B981",
    "monitor-agent": "#F59E0B",
    "research-agent": "#8B5CF6",
    "implement-agent": "#EC4899",
    "tuning-agent": "#F97316",
    "analysis-agent": "#06B6D4",
    "report-agent": "#6366F1",
    "review-agent": "#EF4444",
}

BACKGROUND_AGENTS = {"experiment-agent", "monitor-agent", "review-agent"}
FOREGROUND_AGENTS = set(EXPECTED_AGENTS.keys()) - BACKGROUND_AGENTS

EXPECTED_EXTERNAL_SKILLS = {
    "research-agent": ["claude-mem:mem-search"],
    "implement-agent": ["superpowers:systematic-debugging"],
}


class TestAgentFiles:
    """Validate all 10 agent definition files."""

    def test_all_10_agents_exist(self):
        for name in EXPECTED_AGENTS:
            path = AGENTS_DIR / f"{name}.md"
            assert path.exists(), f"Missing agent file: {path}"

    def test_no_extra_agents(self):
        actual = {f.stem for f in AGENTS_DIR.glob("*.md")}
        expected = set(EXPECTED_AGENTS.keys())
        extra = actual - expected
        assert not extra, f"Unexpected agent files: {extra}"

    @pytest.mark.parametrize("agent_name", EXPECTED_AGENTS.keys())
    def test_agent_has_name(self, agent_name):
        fm = _parse_frontmatter(AGENTS_DIR / f"{agent_name}.md")
        assert fm.get("name") == agent_name, f"{agent_name}: name mismatch"

    @pytest.mark.parametrize("agent_name", EXPECTED_AGENTS.keys())
    def test_agent_has_description(self, agent_name):
        fm = _parse_frontmatter(AGENTS_DIR / f"{agent_name}.md")
        assert fm.get("description"), f"{agent_name}: missing description"

    @pytest.mark.parametrize("agent_name", EXPECTED_AGENTS.keys())
    def test_agent_has_required_tools(self, agent_name):
        tools = _parse_tools(AGENTS_DIR / f"{agent_name}.md")
        required = EXPECTED_AGENTS[agent_name]["required_tools"]
        missing = required - tools
        assert not missing, f"{agent_name} missing tools: {missing}"

    @pytest.mark.parametrize("agent_name", EXPECTED_AGENTS.keys())
    def test_agent_no_forbidden_tools(self, agent_name):
        tools = _parse_tools(AGENTS_DIR / f"{agent_name}.md")
        forbidden = EXPECTED_AGENTS[agent_name]["forbidden_tools"]
        present = forbidden & tools
        assert not present, f"{agent_name} has forbidden tools: {present}"

    @pytest.mark.parametrize("agent_name", EXPECTED_AGENTS.keys())
    def test_agent_has_correct_skill(self, agent_name):
        skills = _parse_skills(AGENTS_DIR / f"{agent_name}.md")
        expected_skill = EXPECTED_AGENTS[agent_name]["skill"]
        assert expected_skill in skills, (
            f"{agent_name}: expected skill {expected_skill}, got {skills}"
        )

    @pytest.mark.parametrize("agent_name", sorted(PROCEDURAL_AGENTS))
    def test_procedural_agent_has_sonnet_model(self, agent_name):
        fm = _parse_frontmatter(AGENTS_DIR / f"{agent_name}.md")
        assert fm.get("model") == "sonnet", (
            f"{agent_name}: procedural agent should have model: sonnet"
        )

    @pytest.mark.parametrize("agent_name", sorted(ANALYTICAL_AGENTS))
    def test_analytical_agent_model_opus(self, agent_name):
        fm = _parse_frontmatter(AGENTS_DIR / f"{agent_name}.md")
        assert fm.get("model") == "opus", (
            f"{agent_name}: analytical agent must have model: opus"
        )

    @pytest.mark.parametrize("agent_name", EXPECTED_AGENTS.keys())
    def test_agent_has_color(self, agent_name):
        fm = _parse_frontmatter(AGENTS_DIR / f"{agent_name}.md")
        expected_color = EXPECTED_COLORS[agent_name]
        assert fm.get("color") == expected_color, (
            f"{agent_name}: expected color {expected_color}, got {fm.get('color')}"
        )

    @pytest.mark.parametrize("agent_name", EXPECTED_AGENTS.keys())
    def test_agent_color_is_valid_hex(self, agent_name):
        fm = _parse_frontmatter(AGENTS_DIR / f"{agent_name}.md")
        color = fm.get("color", "")
        assert re.match(r"^#[0-9A-Fa-f]{6}$", color), (
            f"{agent_name}: color '{color}' is not a valid hex color"
        )

    @pytest.mark.parametrize("agent_name", sorted(BACKGROUND_AGENTS))
    def test_background_agent_has_flag(self, agent_name):
        fm = _parse_frontmatter(AGENTS_DIR / f"{agent_name}.md")
        assert fm.get("background") is True, (
            f"{agent_name}: should have background: true"
        )

    @pytest.mark.parametrize("agent_name", sorted(FOREGROUND_AGENTS))
    def test_foreground_agent_no_background(self, agent_name):
        fm = _parse_frontmatter(AGENTS_DIR / f"{agent_name}.md")
        assert "background" not in fm or fm.get("background") is not True, (
            f"{agent_name}: foreground agent should NOT have background: true"
        )

    @pytest.mark.parametrize("agent_name,expected_skills", sorted(EXPECTED_EXTERNAL_SKILLS.items()))
    def test_agent_has_external_skills(self, agent_name, expected_skills):
        skills = _parse_skills(AGENTS_DIR / f"{agent_name}.md")
        for ext_skill in expected_skills:
            assert ext_skill in skills, (
                f"{agent_name}: missing external skill {ext_skill}, got {skills}"
            )


# ---------------------------------------------------------------------------
# Skill definitions
# ---------------------------------------------------------------------------

EXPECTED_SKILLS = [
    "orchestrate", "prerequisites", "baseline", "experiment", "monitor",
    "research", "implement", "hp-tune", "analyze", "report", "review",
]

NON_ORCHESTRATE_SKILLS = [s for s in EXPECTED_SKILLS if s != "orchestrate"]


class TestSkillFiles:
    """Validate all 11 skill definition files."""

    @pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
    def test_skill_exists(self, skill_name):
        path = SKILLS_DIR / skill_name / "SKILL.md"
        assert path.exists(), f"Missing skill: {path}"

    @pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
    def test_skill_has_name(self, skill_name):
        fm = _parse_frontmatter(SKILLS_DIR / skill_name / "SKILL.md")
        assert fm.get("name") == skill_name, f"{skill_name}: name mismatch"

    @pytest.mark.parametrize("skill_name", NON_ORCHESTRATE_SKILLS)
    def test_non_orchestrate_skill_disabled(self, skill_name):
        fm = _parse_frontmatter(SKILLS_DIR / skill_name / "SKILL.md")
        assert fm.get("disable-model-invocation") is True, (
            f"{skill_name}: must have disable-model-invocation: true"
        )

    def test_all_skills_disabled(self):
        """All skills (including orchestrate) must have disable-model-invocation: true.
        The only entry point is the /optimize command."""
        for skill_dir in SKILLS_DIR.iterdir():
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                fm = _parse_frontmatter(skill_file)
                assert fm.get("disable-model-invocation") is True, (
                    f"{skill_dir.name} must have disable-model-invocation: true"
                )

    def test_all_skills_not_user_invocable(self):
        """All skills must have user-invocable: false.
        The only entry point is the /optimize command."""
        for skill_dir in SKILLS_DIR.iterdir():
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                fm = _parse_frontmatter(skill_file)
                assert fm.get("user-invocable") is False, (
                    f"{skill_dir.name} must have user-invocable: false"
                )

    def test_orchestrate_reference_files_exist(self):
        """All 10 phase reference files must exist in orchestrate/references/."""
        refs_dir = SKILLS_DIR / "orchestrate" / "references"
        for phase in range(10):
            matches = list(refs_dir.glob(f"phase-{phase}-*.md"))
            assert len(matches) >= 1, f"Missing reference file for phase {phase}"

    @pytest.mark.parametrize("skill_name", NON_ORCHESTRATE_SKILLS)
    def test_non_orchestrate_no_context_fork(self, skill_name):
        fm = _parse_frontmatter(SKILLS_DIR / skill_name / "SKILL.md")
        assert "context" not in fm, f"{skill_name}: should not have context: in frontmatter"

    @pytest.mark.parametrize("skill_name", NON_ORCHESTRATE_SKILLS)
    def test_non_orchestrate_no_agent_field(self, skill_name):
        fm = _parse_frontmatter(SKILLS_DIR / skill_name / "SKILL.md")
        assert "agent" not in fm, f"{skill_name}: should not have agent: in frontmatter"


# ---------------------------------------------------------------------------
# Skill-to-agent mapping
# ---------------------------------------------------------------------------

class TestSkillAgentMapping:
    """Verify every non-orchestrate skill has exactly one agent that loads it."""

    def test_every_skill_has_an_agent(self):
        """Each non-orchestrate skill should be preloaded by exactly one agent."""
        skill_to_agents: dict[str, list[str]] = {s: [] for s in NON_ORCHESTRATE_SKILLS}
        for agent_file in AGENTS_DIR.glob("*.md"):
            for skill in _parse_skills(agent_file):
                skill_name = skill.replace("ml-optimizer:", "")
                if skill_name in skill_to_agents:
                    skill_to_agents[skill_name].append(agent_file.stem)
        for skill_name, agents in skill_to_agents.items():
            assert len(agents) == 1, (
                f"Skill '{skill_name}' loaded by {len(agents)} agents: {agents} (expected 1)"
            )


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

EXPECTED_HOOKS = [
    "bash-safety.sh",
    "file-guardrail.sh",
    "detect-critical-errors.sh",
    "subagent-stop-hook.sh",
    "pre-compact.sh",
    "post-compact-context.sh",
]


class TestHooks:
    """Validate hooks configuration."""

    def test_hooks_json_exists(self):
        assert (HOOKS_DIR / "hooks.json").exists()

    def test_hooks_json_valid(self):
        data = json.loads((HOOKS_DIR / "hooks.json").read_text())
        assert "hooks" in data, "hooks.json missing 'hooks' key"
        assert isinstance(data["hooks"], dict), "hooks must be a dict keyed by event type"

    def test_hooks_json_has_entries(self):
        data = json.loads((HOOKS_DIR / "hooks.json").read_text())
        events = data["hooks"]
        assert len(events) >= 5, f"Expected at least 5 event types, got {len(events)}"

    @pytest.mark.parametrize("hook_file", EXPECTED_HOOKS)
    def test_hook_script_exists(self, hook_file):
        path = HOOKS_DIR / hook_file
        assert path.exists(), f"Missing hook script: {path}"

    @pytest.mark.parametrize("hook_file", EXPECTED_HOOKS)
    def test_hook_script_executable(self, hook_file):
        import os
        path = HOOKS_DIR / hook_file
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
                            f"Hook '{event_type}' should use ${{CLAUDE_PLUGIN_ROOT}}, "
                            f"not hardcoded paths: {cmd}"
                        )


# ---------------------------------------------------------------------------
# Scripts
# ---------------------------------------------------------------------------

EXPECTED_SCRIPTS = [
    "gpu_check.py",
    "parse_logs.py",
    "detect_divergence.py",
    "result_analyzer.py",
    "experiment_setup.py",
    "implement_utils.py",
    "pipeline_state.py",
    "schema_validator.py",
    "plot_results.py",
    "error_tracker.py",
    "prerequisites_check.py",
]


class TestScripts:
    """Validate all Python scripts exist and are importable."""

    @pytest.mark.parametrize("script", EXPECTED_SCRIPTS)
    def test_script_exists(self, script):
        assert (SCRIPTS_DIR / script).exists(), f"Missing script: {script}"

    @pytest.mark.parametrize("script", EXPECTED_SCRIPTS)
    def test_script_importable(self, script):
        """Each script should be importable without errors."""
        import importlib
        module_name = script.replace(".py", "")
        try:
            importlib.import_module(module_name)
        except Exception as e:
            pytest.fail(f"Failed to import {module_name}: {e}")


# ---------------------------------------------------------------------------
# Plugin manifest
# ---------------------------------------------------------------------------

class TestPluginManifest:
    """Validate plugin.json."""

    def test_plugin_json_exists(self):
        assert PLUGIN_JSON.exists()

    def test_plugin_json_valid(self):
        data = json.loads(PLUGIN_JSON.read_text())
        assert data.get("name") == "ml-optimizer"
        assert "version" in data
        assert "description" in data

    def test_plugin_json_version_format(self):
        data = json.loads(PLUGIN_JSON.read_text())
        version = data["version"]
        assert re.match(r"^\d+\.\d+\.\d+$", version), f"Invalid version: {version}"


# ---------------------------------------------------------------------------
# Orchestrate dispatch points
# ---------------------------------------------------------------------------

class TestOrchestrateDispatch:
    """Verify the orchestrate skill references all 10 agents correctly."""

    @staticmethod
    def _orchestrate_full_text():
        """Read orchestrate SKILL.md + all reference files as combined text."""
        orch_dir = SKILLS_DIR / "orchestrate"
        parts = [(orch_dir / "SKILL.md").read_text()]
        refs_dir = orch_dir / "references"
        if refs_dir.exists():
            for f in sorted(refs_dir.glob("*.md")):
                parts.append(f.read_text())
        return "\n".join(parts)

    def test_no_general_purpose_dispatch(self):
        """Named agent dispatch points should not use general-purpose.

        Note: speculative hp-tune and parallel experiment dispatch are allowed
        to use general-purpose since they're dynamic/templated dispatches.
        """
        text = self._orchestrate_full_text()
        # Verify named agent types are used (not just general-purpose everywhere)
        named_dispatches = re.findall(r'subagent_type.*ml-optimizer:', text)
        assert len(named_dispatches) >= 5, (
            f"Expected at least 5 named agent dispatches, found {len(named_dispatches)}"
        )

    def test_no_bare_skill_invocation(self):
        """Orchestrate should dispatch agents, not invoke skills directly."""
        text = self._orchestrate_full_text()
        bare_invocations = re.findall(r'Invoke\s+the\s+ml-optimizer:', text)
        assert not bare_invocations, (
            f"Found bare skill invocations (should use Agent dispatch): {bare_invocations}"
        )

    def test_references_all_agent_types(self):
        """Orchestrate should reference all 10 agent types by name."""
        text = self._orchestrate_full_text()
        expected_agents = [
            "prerequisites-agent", "baseline-agent", "experiment-agent",
            "monitor-agent", "research-agent", "implement-agent",
            "tuning-agent", "analysis-agent", "report-agent", "review-agent",
        ]
        for agent in expected_agents:
            assert agent in text, (
                f"Orchestrate does not reference {agent}"
            )


# ---------------------------------------------------------------------------
# Documentation consistency
# ---------------------------------------------------------------------------

class TestDocumentation:
    """Verify docs reflect 10-agent architecture."""

    def test_claude_md_says_10_agents(self):
        text = (PLUGIN_ROOT / ".claude" / "CLAUDE.md").read_text()
        assert "10 subagent definitions" in text, "CLAUDE.md should say '10 subagent definitions'"

    def test_claude_md_agent_dispatch_pattern(self):
        text = (PLUGIN_ROOT / ".claude" / "CLAUDE.md").read_text()
        assert 'Agent(subagent_type="ml-optimizer:' in text, (
            "CLAUDE.md should reference Agent(subagent_type=...) dispatch pattern"
        )

    def test_workflow_diagram_10_types(self):
        path = PLUGIN_ROOT / "docs" / "workflow-diagram.txt"
        if path.exists():
            text = path.read_text()
            assert "10 types" in text, "workflow-diagram.txt should say '10 types'"

    def test_implement_skill_mentions_unit_tests(self):
        text = (SKILLS_DIR / "implement" / "SKILL.md").read_text()
        assert "unit test" in text.lower(), (
            "Implement skill should mention unit test writing"
        )

    def test_implement_agent_mentions_test_writing(self):
        text = (AGENTS_DIR / "implement-agent.md").read_text()
        assert "Test Writing" in text, (
            "Implement agent should have a Test Writing section"
        )

    def test_readme_disable_model_invocation(self):
        text = (PLUGIN_ROOT / "README.md").read_text()
        assert "disable-model-invocation" in text, (
            "README.md should mention disable-model-invocation"
        )
