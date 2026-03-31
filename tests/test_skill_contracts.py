#!/usr/bin/env python3
"""Prompt contract tests for skill .md files.

Static analysis tests that read skill and agent .md files and verify they
reference required reliability protocols (phase gates, relay validation,
context budget, decision logging, meta-patch validation).

These tests verify the *contracts* defined in Markdown, not Python code.
"""

import json
import os
import re
import sys
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

PLUGIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SKILLS_DIR = os.path.join(PLUGIN_ROOT, "skills")
AGENTS_DIR = os.path.join(PLUGIN_ROOT, "agents")
REFERENCES_DIR = os.path.join(SKILLS_DIR, "orchestrate", "references")
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

# Add scripts to path for imports
sys.path.insert(0, os.path.join(PLUGIN_ROOT, "scripts"))


def _read_file(path):
    """Read a file and return its content, or None if not found."""
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return None


def _list_agent_files():
    """Return sorted list of .md file paths in agents/."""
    if not os.path.isdir(AGENTS_DIR):
        return []
    return sorted(
        os.path.join(AGENTS_DIR, f)
        for f in os.listdir(AGENTS_DIR)
        if f.endswith(".md")
    )


def _list_skill_dirs():
    """Return sorted list of skill directory names under skills/.

    Only includes directories that actually exist on disk (resolves symlinks).
    Broken symlinks are excluded.
    """
    if not os.path.isdir(SKILLS_DIR):
        return []
    result = []
    for entry in sorted(os.listdir(SKILLS_DIR)):
        full = os.path.join(SKILLS_DIR, entry)
        # os.path.isdir follows symlinks; broken symlinks return False
        if os.path.isdir(full):
            result.append(entry)
    return result


def _extract_frontmatter(content):
    """Extract YAML frontmatter from a Markdown file as raw text.

    Returns the frontmatter string (between the --- delimiters) or None.
    """
    if not content or not content.startswith("---"):
        return None
    end = content.find("---", 3)
    if end == -1:
        return None
    return content[3:end]


def _extract_skills_from_frontmatter(frontmatter):
    """Extract skill references from YAML frontmatter text.

    Parses the skills: list manually (no PyYAML dependency).
    Returns a list of skill strings like "ml-optimizer:research".
    """
    if not frontmatter:
        return []
    skills = []
    in_skills = False
    for line in frontmatter.splitlines():
        stripped = line.strip()
        if stripped.startswith("skills:"):
            in_skills = True
            continue
        if in_skills:
            if stripped.startswith("- "):
                skills.append(stripped[2:].strip())
            elif stripped and not stripped.startswith("#"):
                # Reached a different frontmatter key
                break
    return skills


# Persistent agent names (from CLAUDE.md)
PERSISTENT_AGENTS = [
    "research-agent",
    "tuning-agent",
    "analysis-agent",
    "implement-agent",
    "monitor-agent",
    "hyperagent-agent",
]

# Relay routes defined in the orchestrate SKILL.md (key relay routes section)
DOCUMENTED_RELAY_ROUTES = [
    "analyze_to_tuning",
    "analyze_to_research",
    "monitor_to_tuning",
    "research_to_implement",
    "experiments_to_analyze",
]

# Additional relay routes that exist for the hyperagent integration
HYPERAGENT_RELAY_ROUTES = [
    "analyze_to_hyperagent",
    "hyperagent_to_tuning",
]


# ===========================================================================
# TestRelaySchemaCompleteness
# ===========================================================================


class TestRelaySchemaCompleteness:
    """Verify that every relay route mentioned in skill docs is defined in RELAY_SCHEMAS."""

    def test_all_relay_routes_defined(self):
        """Every relay route documented in SKILL.md or phase-7 reference has a schema."""
        from schema_validator import RELAY_SCHEMAS

        all_documented = DOCUMENTED_RELAY_ROUTES + HYPERAGENT_RELAY_ROUTES

        missing = []
        for route in all_documented:
            if route not in RELAY_SCHEMAS:
                missing.append(route)

        assert not missing, (
            f"Relay routes documented in skill files but missing from RELAY_SCHEMAS: {missing}"
        )

    def test_relay_schemas_have_required_keys(self):
        """Each RELAY_SCHEMAS entry has the expected structure."""
        from schema_validator import RELAY_SCHEMAS

        for route, schema in RELAY_SCHEMAS.items():
            assert "required" in schema, f"Route '{route}' missing 'required' key"
            assert "optional" in schema, f"Route '{route}' missing 'optional' key"
            assert isinstance(schema["required"], list), (
                f"Route '{route}' 'required' must be a list"
            )
            assert isinstance(schema["optional"], list), (
                f"Route '{route}' 'optional' must be a list"
            )

    def test_context_from_other_agents_has_matching_routes(self):
        """Each 'CONTEXT FROM OTHER AGENTS' block in SKILL.md maps to a known route.

        Checks that the relay routes described in prose (e.g. 'analyze -> tuning')
        correspond to entries in RELAY_SCHEMAS.
        """
        from schema_validator import RELAY_SCHEMAS

        content = _read_file(os.path.join(SKILLS_DIR, "orchestrate", "SKILL.md"))
        assert content is not None, "orchestrate/SKILL.md not found"

        # Extract the "Key relay routes" section
        # Pattern: "- **source -> dest**: description"
        route_pattern = re.compile(r"\*\*(\w+)\s*(?:→|->)\s*(\w+)\*\*")
        found_routes = route_pattern.findall(content)

        assert len(found_routes) > 0, (
            "No relay routes found in SKILL.md (expected pattern: **source -> dest**)"
        )

        # Map prose names to schema route names
        for source, dest in found_routes:
            route_name = f"{source}_to_{dest}"
            assert route_name in RELAY_SCHEMAS, (
                f"Relay route '{source} -> {dest}' documented in SKILL.md "
                f"but '{route_name}' not in RELAY_SCHEMAS"
            )


# ===========================================================================
# TestAgentRelayAcknowledgment
# ===========================================================================


class TestAgentRelayAcknowledgment:
    """Verify persistent agents acknowledge relay messages."""

    @pytest.mark.parametrize("agent_name", PERSISTENT_AGENTS)
    def test_persistent_agents_have_relay_ack(self, agent_name):
        """Each persistent agent .md file contains RELAY_ACK somewhere."""
        path = os.path.join(AGENTS_DIR, f"{agent_name}.md")
        content = _read_file(path)
        assert content is not None, f"Agent file not found: {path}"
        assert "RELAY_ACK" in content, (
            f"Persistent agent '{agent_name}' does not contain 'RELAY_ACK'. "
            f"All persistent agents must acknowledge relay context."
        )


# ===========================================================================
# TestPhaseGateReferences
# ===========================================================================


class TestPhaseGateReferences:
    """Verify phase gate protocol is referenced in skill files."""

    def test_orchestrate_skill_mentions_gate_protocol(self):
        """The orchestrate SKILL.md references phase gating for transitions."""
        content = _read_file(os.path.join(SKILLS_DIR, "orchestrate", "SKILL.md"))
        assert content is not None, "orchestrate/SKILL.md not found"

        # Check for any of: "Phase Gate Protocol", "validate_phase_gate",
        # "validate_phase_requirements", "gate" near "phase transition"
        has_gate_ref = (
            "Phase Gate Protocol" in content
            or "validate_phase_gate" in content
            or "validate_phase_requirements" in content
            or "phase-gates.json" in content
        )
        assert has_gate_ref, (
            "orchestrate/SKILL.md does not reference phase gate validation. "
            "Expected 'Phase Gate Protocol', 'validate_phase_gate', "
            "'validate_phase_requirements', or 'phase-gates.json'."
        )

    def test_pipeline_state_has_gate_functions(self):
        """pipeline_state.py exports validate_phase_gate and log_phase_gate."""
        import pipeline_state

        assert hasattr(pipeline_state, "validate_phase_gate"), (
            "pipeline_state.py missing validate_phase_gate function"
        )
        assert hasattr(pipeline_state, "log_phase_gate"), (
            "pipeline_state.py missing log_phase_gate function"
        )
        assert callable(pipeline_state.validate_phase_gate)
        assert callable(pipeline_state.log_phase_gate)

    @pytest.mark.parametrize("phase_num", range(2, 10))
    def test_phase_references_mention_gates(self, phase_num):
        """Each phase reference file (phase-2 through phase-9) mentions gate checking."""
        filename = f"phase-{phase_num}-{['prerequisites', 'baseline', 'checkpoint', 'research', 'implement', 'experiment-loop', 'stacking', 'report'][phase_num - 2]}.md"
        path = os.path.join(REFERENCES_DIR, filename)
        content = _read_file(path)
        if content is None:
            pytest.skip(f"Reference file not found: {filename}")

        has_gate = (
            "gate" in content.lower()
            or "validate_phase" in content
            or "log-gate" in content
            or "phase-gates" in content
        )
        assert has_gate, (
            f"Phase reference '{filename}' does not mention phase gate checking."
        )


# ===========================================================================
# TestDecisionLoggingReferences
# ===========================================================================


class TestDecisionLoggingReferences:
    """Verify decision logging protocol is referenced in relevant skills."""

    def test_hyperagent_skill_mentions_decision_logging(self):
        """hyperagent/SKILL.md mentions decision logging."""
        content = _read_file(os.path.join(SKILLS_DIR, "hyperagent", "SKILL.md"))
        assert content is not None, "hyperagent/SKILL.md not found"

        has_decision_logging = (
            "log-decision" in content
            or "log_decision" in content
            or "decision logging" in content.lower()
            or "Decision Logging Protocol" in content
            or "decision-log.json" in content
        )
        assert has_decision_logging, (
            "hyperagent/SKILL.md does not mention decision logging. "
            "Expected 'log-decision', 'log_decision', 'Decision Logging Protocol', "
            "or 'decision-log.json'."
        )

    def test_pipeline_state_has_log_decision(self):
        """pipeline_state.py exports log_decision function."""
        import pipeline_state

        assert hasattr(pipeline_state, "log_decision"), (
            "pipeline_state.py missing log_decision function"
        )
        assert callable(pipeline_state.log_decision)

    def test_pipeline_state_has_get_decisions(self):
        """pipeline_state.py exports get_decisions function for querying."""
        import pipeline_state

        assert hasattr(pipeline_state, "get_decisions"), (
            "pipeline_state.py missing get_decisions function"
        )
        assert callable(pipeline_state.get_decisions)


# ===========================================================================
# TestMetaPatchReferences
# ===========================================================================


class TestMetaPatchReferences:
    """Verify meta-patch validation is referenced in the hyperagent skill."""

    def test_hyperagent_skill_mentions_meta_patch_validation(self):
        """hyperagent/SKILL.md mentions meta-patch validation."""
        content = _read_file(os.path.join(SKILLS_DIR, "hyperagent", "SKILL.md"))
        assert content is not None, "hyperagent/SKILL.md not found"

        has_validation = (
            "meta-patch validate" in content
            or "validate_meta_patch" in content
            or "meta-patch validation" in content.lower()
            or "meta_patch_validation" in content
        )
        assert has_validation, (
            "hyperagent/SKILL.md does not mention meta-patch validation. "
            "Expected 'meta-patch validate', 'validate_meta_patch', or "
            "'meta-patch validation'."
        )

    def test_hyperagent_skill_mentions_meta_patches(self):
        """hyperagent/SKILL.md at least mentions meta-patches as a concept."""
        content = _read_file(os.path.join(SKILLS_DIR, "hyperagent", "SKILL.md"))
        assert content is not None, "hyperagent/SKILL.md not found"

        assert "meta-patch" in content.lower() or "meta_patch" in content.lower(), (
            "hyperagent/SKILL.md does not mention meta-patches at all"
        )


# ===========================================================================
# TestGoldenDecisions
# ===========================================================================


class TestGoldenDecisions:
    """Validate golden decision fixtures against the log_decision schema."""

    @pytest.fixture
    def decision_exp_root(self, tmp_path):
        """Create a temporary experiments directory for decision logging."""
        return str(tmp_path)

    @pytest.mark.parametrize("fixture_name", [
        "phase7_continue.json",
        "phase7_pivot.json",
    ])
    def test_golden_decision_logs_successfully(self, fixture_name, decision_exp_root):
        """Golden decision fixtures can be logged via log_decision without error."""
        from pipeline_state import log_decision

        fixture_path = os.path.join(
            FIXTURES_DIR, "golden_decisions", fixture_name
        )
        with open(fixture_path) as f:
            decision_data = json.load(f)

        # log_decision should succeed (returns an id string)
        decision_id = log_decision(decision_exp_root, decision_data)
        assert isinstance(decision_id, str)
        assert int(decision_id) > 0

    @pytest.mark.parametrize("fixture_name", [
        "phase7_continue.json",
        "phase7_pivot.json",
    ])
    def test_golden_decision_has_required_fields(self, fixture_name):
        """Golden decision fixtures contain all required fields."""
        fixture_path = os.path.join(
            FIXTURES_DIR, "golden_decisions", fixture_name
        )
        with open(fixture_path) as f:
            data = json.load(f)

        required = ["phase", "agent", "decision_type", "decision"]
        for field in required:
            assert field in data, (
                f"Golden fixture '{fixture_name}' missing required field: {field}"
            )

    def test_golden_decision_round_trip(self, decision_exp_root):
        """Logged decisions can be retrieved via get_decisions."""
        from pipeline_state import log_decision, get_decisions

        fixture_path = os.path.join(
            FIXTURES_DIR, "golden_decisions", "phase7_continue.json"
        )
        with open(fixture_path) as f:
            decision_data = json.load(f)

        log_decision(decision_exp_root, decision_data)

        results = get_decisions(decision_exp_root, phase=7, agent="analysis")
        assert len(results) == 1
        assert results[0]["decision_type"] == "continue"
        assert results[0]["phase"] == 7


# ===========================================================================
# TestSkillStructuralIntegrity
# ===========================================================================


class TestSkillStructuralIntegrity:
    """Verify structural integrity of agent and skill files."""

    def test_all_agent_files_have_frontmatter(self):
        """Every .md file in agents/ starts with --- (YAML frontmatter)."""
        agent_files = _list_agent_files()
        assert len(agent_files) > 0, "No agent .md files found"

        missing_frontmatter = []
        for path in agent_files:
            content = _read_file(path)
            if content is None:
                missing_frontmatter.append(f"{os.path.basename(path)} (not readable)")
            elif not content.startswith("---"):
                missing_frontmatter.append(os.path.basename(path))

        assert not missing_frontmatter, (
            f"Agent files without YAML frontmatter: {missing_frontmatter}"
        )

    def test_all_skill_files_exist(self):
        """For each agent, every ml-optimizer skill in its frontmatter has a directory under skills/."""
        agent_files = _list_agent_files()
        assert len(agent_files) > 0, "No agent .md files found"

        missing_skills = []
        for path in agent_files:
            content = _read_file(path)
            if content is None:
                continue
            frontmatter = _extract_frontmatter(content)
            skills = _extract_skills_from_frontmatter(frontmatter)

            agent_name = os.path.basename(path)
            for skill_ref in skills:
                # Only check ml-optimizer skills (not claude-mem, superpowers, etc.)
                if not skill_ref.startswith("ml-optimizer:"):
                    continue
                skill_name = skill_ref.split(":", 1)[1]
                skill_dir = os.path.join(SKILLS_DIR, skill_name)

                # Accept the directory existing as either a real dir or a symlink
                # (even broken symlinks count -- the directory is *declared*)
                if not os.path.isdir(skill_dir) and not os.path.islink(skill_dir):
                    missing_skills.append(
                        f"{agent_name} references '{skill_ref}' "
                        f"but skills/{skill_name}/ does not exist"
                    )

        assert not missing_skills, (
            f"Agent files reference missing skill directories:\n"
            + "\n".join(f"  - {m}" for m in missing_skills)
        )

    def test_no_orphan_skills(self):
        """Every skill directory with a SKILL.md is referenced by at least one agent."""
        # Collect all ml-optimizer skill references from all agents
        referenced_skills = set()
        for path in _list_agent_files():
            content = _read_file(path)
            if content is None:
                continue
            frontmatter = _extract_frontmatter(content)
            skills = _extract_skills_from_frontmatter(frontmatter)
            for skill_ref in skills:
                if skill_ref.startswith("ml-optimizer:"):
                    referenced_skills.add(skill_ref.split(":", 1)[1])

        # Check each real skill directory (with SKILL.md) is referenced
        orphans = []
        for skill_name in _list_skill_dirs():
            skill_md = os.path.join(SKILLS_DIR, skill_name, "SKILL.md")
            if not os.path.isfile(skill_md):
                continue  # No SKILL.md -> not a real skill, skip
            if skill_name not in referenced_skills:
                orphans.append(skill_name)

        assert not orphans, (
            f"Orphan skills (have SKILL.md but no agent references them): {orphans}"
        )

    def test_agent_files_have_skills_key(self):
        """Every agent .md has a 'skills:' key in its frontmatter."""
        agent_files = _list_agent_files()
        missing = []
        for path in agent_files:
            content = _read_file(path)
            if content is None:
                continue
            frontmatter = _extract_frontmatter(content)
            if frontmatter is None:
                continue  # Already caught by test_all_agent_files_have_frontmatter
            skills = _extract_skills_from_frontmatter(frontmatter)
            if not skills:
                missing.append(os.path.basename(path))

        assert not missing, (
            f"Agent files without skills in frontmatter: {missing}"
        )

    def test_agent_files_have_name_key(self):
        """Every agent .md has a 'name:' key in its frontmatter."""
        agent_files = _list_agent_files()
        missing = []
        for path in agent_files:
            content = _read_file(path)
            if content is None:
                continue
            frontmatter = _extract_frontmatter(content)
            if frontmatter is None:
                continue
            has_name = any(
                line.strip().startswith("name:")
                for line in frontmatter.splitlines()
            )
            if not has_name:
                missing.append(os.path.basename(path))

        assert not missing, (
            f"Agent files without 'name:' in frontmatter: {missing}"
        )
