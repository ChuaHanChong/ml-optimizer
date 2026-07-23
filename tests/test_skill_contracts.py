"""Tests for skill/agent .md prompt contracts — reliability protocols (phases 5-8 run as dynamic workflows)."""

import json
import re
from pathlib import Path

import pytest

from conftest import PLUGIN_ROOT, SKILLS_DIR, AGENTS_DIR, REFERENCES_DIR, FIXTURES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_file(path):
    """Read a file and return its content, or None if not found."""
    try:
        return Path(path).read_text()
    except FileNotFoundError:
        return None


def _list_agent_files():
    """Return sorted list of .md file paths in agents/."""
    if not AGENTS_DIR.is_dir():
        return []
    return sorted(
        AGENTS_DIR / f.name
        for f in AGENTS_DIR.iterdir()
        if f.name.endswith(".md")
    )


def _list_skill_dirs():
    """Return sorted list of skill directory names under skills/.

    Only includes directories that actually exist on disk (resolves symlinks).
    Broken symlinks are excluded.
    """
    if not SKILLS_DIR.is_dir():
        return []
    result = []
    for entry in sorted(SKILLS_DIR.iterdir()):
        # Path.is_dir follows symlinks; broken symlinks return False
        if entry.is_dir():
            result.append(entry.name)
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


# Relay routes defined in the orchestrate SKILL.md (cross-agent file/args handoff
# table). The RELAY_SCHEMAS still validate the file/args payloads handed between
# workflow stages.
DOCUMENTED_RELAY_ROUTES = [
    "analyze_to_tuning",
    "analyze_to_research",
    "monitor_to_tuning",
    "research_to_implement",
    "research_to_tuning",
    "experiments_to_analyze",
]


# ===========================================================================
# TestRelaySchemaCompleteness
# ===========================================================================


class TestRelaySchemaCompleteness:
    """Verify that every relay route mentioned in skill docs is defined in RELAY_SCHEMAS."""

    def test_all_relay_routes_defined(self):
        """Every relay route documented in SKILL.md or phase-7 reference has a schema."""
        from schema_validator import RELAY_SCHEMAS

        missing = []
        for route in DOCUMENTED_RELAY_ROUTES:
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

    def test_documented_handoff_routes_have_schemas(self):
        """Each cross-agent handoff route the SKILL.md documents has a schema.

        Phases 5-8 hand off context via args + files (no message bus). The
        RELAY_SCHEMAS validate those file/args payloads, so every documented
        handoff route must still resolve to a schema entry.
        """
        from schema_validator import RELAY_SCHEMAS

        content = _read_file(SKILLS_DIR / "orchestrate" / "SKILL.md")
        assert content is not None, "orchestrate/SKILL.md not found"

        # Cross-agent handoff routes are documented as table rows like
        # "| analyze → tuning | ... |" (arrow may be → or ->).
        route_pattern = re.compile(r"\|\s*(\w+)\s*(?:→|->)\s*(\w+)\s*\|")
        found_routes = route_pattern.findall(content)

        assert len(found_routes) > 0, (
            "No cross-agent handoff routes found in SKILL.md "
            "(expected table rows: 'source → dest |')"
        )

        for source, dest in found_routes:
            route_name = f"{source}_to_{dest}"
            assert route_name in RELAY_SCHEMAS, (
                f"Handoff route '{source} → {dest}' documented in SKILL.md "
                f"but '{route_name}' not in RELAY_SCHEMAS"
            )


# ===========================================================================
# TestPhaseGateReferences
# ===========================================================================


class TestPhaseGateReferences:
    """Verify phase gate protocol is referenced in skill files."""

    def test_orchestrate_skill_mentions_gate_protocol(self):
        """The orchestrate SKILL.md references phase gating for transitions."""
        content = _read_file(SKILLS_DIR / "orchestrate" / "SKILL.md")
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
        path = REFERENCES_DIR / filename
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

        fixture_path = FIXTURES / "golden_decisions" / fixture_name
        decision_data = json.loads(fixture_path.read_text())

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
        fixture_path = FIXTURES / "golden_decisions" / fixture_name
        data = json.loads(fixture_path.read_text())

        required = ["phase", "agent", "decision_type", "decision"]
        for field in required:
            assert field in data, (
                f"Golden fixture '{fixture_name}' missing required field: {field}"
            )

    def test_golden_decision_round_trip(self, decision_exp_root):
        """Logged decisions can be retrieved via get_decisions."""
        from pipeline_state import log_decision, get_decisions

        fixture_path = FIXTURES / "golden_decisions" / "phase7_continue.json"
        decision_data = json.loads(fixture_path.read_text())

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
                missing_frontmatter.append(f"{path.name} (not readable)")
            elif not content.startswith("---"):
                missing_frontmatter.append(path.name)

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

            agent_name = path.name
            for skill_ref in skills:
                # Only check ml-optimizer skills (not claude-mem, superpowers, etc.)
                if not skill_ref.startswith("ml-optimizer:"):
                    continue
                skill_name = skill_ref.split(":", 1)[1]
                skill_dir = SKILLS_DIR / skill_name

                # Accept the directory existing as either a real dir or a symlink
                # (even broken symlinks count -- the directory is *declared*)
                if not skill_dir.is_dir() and not skill_dir.is_symlink():
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
            skill_md = SKILLS_DIR / skill_name / "SKILL.md"
            if not skill_md.is_file():
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
                missing.append(path.name)

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
                missing.append(path.name)

        assert not missing, (
            f"Agent files without 'name:' in frontmatter: {missing}"
        )


# ===========================================================================
# TestPhase7CleanupContract
# ===========================================================================


class TestPhase7CleanupContract:
    """Phase 7 loop doc calls the real cleanup CLI, not a nonexistent helper."""

    def test_phase7_loop_doc_uses_real_cleanup(self):
        content = _read_file(REFERENCES_DIR / "phase-7-experiment-loop.md")
        assert content is not None
        assert "cleanup_stale_experiments" not in content
        assert "pipeline_state.py <exp_root> cleanup" in content


# ===========================================================================
# TestBaselineTimeoutFieldContracts (Batch B)
# ===========================================================================


class TestBaselineTimeoutFieldContracts:
    """baseline.json records model_category + timeout/duration; fallbacks reconciled."""

    def test_baseline_skill_writes_new_fields(self):
        content = _read_file(SKILLS_DIR / "baseline" / "SKILL.md")
        assert content is not None
        assert '"model_category"' in content
        assert "training_duration_seconds" in content
        # tabular section AND RL section both record the shared timeout field
        assert content.count("estimated_timeout_seconds") >= 3

    def test_experiment_skill_timeout_chain_reconciled(self):
        content = _read_file(SKILLS_DIR / "experiment" / "SKILL.md")
        assert content is not None
        assert "estimated_timeout_seconds` (tabular ML)" not in content
        assert "training_duration_seconds" in content
        assert "21600" in content
        assert "14400" not in content

    def test_phase7_reference_timeout_chain_reconciled(self):
        content = _read_file(REFERENCES_DIR / "phase-7-experiment-loop.md")
        assert content is not None
        assert "estimated_timeout_seconds` (tabular ML)" not in content
        assert "training_duration_seconds" in content


# ===========================================================================
# TestFixedStepBudgetContracts (Batch B)
# ===========================================================================


class TestFixedStepBudgetContracts:
    """RL budget unit: offered at Phase 0, mapped by baseline + experiment."""

    def test_phase0_offers_step_budget(self):
        content = _read_file(REFERENCES_DIR / "phase-0-discovery.md")
        assert content is not None
        assert "fixed_step_budget" in content
        assert "environment timesteps" in content

    def test_baseline_maps_step_budget(self):
        content = _read_file(SKILLS_DIR / "baseline" / "SKILL.md")
        assert content is not None
        assert "fixed_step_budget" in content
        assert "total_timesteps" in content

    def test_experiment_maps_step_budget(self):
        content = _read_file(SKILLS_DIR / "experiment" / "SKILL.md")
        assert content is not None
        assert "fixed_step_budget" in content
        assert '"step_budget"' in content


# ===========================================================================
# TestMonitoringOwnershipContracts (Batch B)
# ===========================================================================


class TestMonitoringOwnershipContracts:
    """Experiment agent owns in-run monitoring; RL eval mirrors baseline."""

    def test_experiment_skill_owns_in_run_monitoring(self):
        content = _read_file(SKILLS_DIR / "experiment" / "SKILL.md")
        assert content is not None
        assert "every 5 minutes" in content
        assert "--scan-records" in content
        assert '"plateaued"' in content  # plateau reported, never kills

    def test_experiment_skill_10x_rule_scoped(self):
        content = _read_file(SKILLS_DIR / "experiment" / "SKILL.md")
        assert "lower-is-better AND its baseline value is positive" in content

    def test_experiment_skill_rl_eval_symmetry(self):
        content = _read_file(SKILLS_DIR / "experiment" / "SKILL.md")
        assert "deterministic policy" in content
        assert "FORBIDDEN" in content
        assert "best-over" in content.lower()

    def test_claude_md_diagram_reflects_folded_monitoring(self):
        content = _read_file(SKILLS_DIR.parent / ".claude" / "CLAUDE.md")
        assert content is not None
        assert "folded into each experiment agent" in content


# ===========================================================================
# TestResearchPriorRoute (Batch C, Task 14)
# ===========================================================================


class TestResearchPriorRoute:
    """Research-derived HP priors flow research -> hp-tune (contract strings)."""

    def test_research_skill_documents_search_space(self):
        content = _read_file(SKILLS_DIR / "research" / "SKILL.md")
        assert content is not None
        assert "{param, range, scale, source}" in content
        assert '"search_space"' in content

    def test_hp_tune_prior_order(self):
        content = _read_file(SKILLS_DIR / "hp-tune" / "SKILL.md")
        assert content is not None
        assert "Research-derived priors (PRIMARY)" in content
        assert "cold-start fallback ONLY" in content
        assert "gamma, clip_range, ent_coef, n_steps" in content
        assert '"research_requested": true' in content

    def test_tuning_strategy_rl_vla_sections(self):
        content = _read_file(SKILLS_DIR / "hp-tune" / "references" / "tuning-strategy.md")
        assert content is not None
        assert content.count("cold-start fallback — prefer research-derived priors") >= 2
        assert "Linear LR-batch scaling does NOT apply" in content
        assert "Epoch budgets do NOT apply" in content
        assert "PPO" in content and "SAC" in content
        assert "Imitation Learning" in content

    def test_analyze_regularization_names_rl_knobs(self):
        content = _read_file(SKILLS_DIR / "analyze" / "SKILL.md")
        assert content is not None
        assert "entropy coefficient / KL penalty" in content

    def test_orchestrate_documents_research_to_tuning_route(self):
        content = _read_file(SKILLS_DIR / "orchestrate" / "SKILL.md")
        assert content is not None
        assert "research → tuning" in content


# ===========================================================================
# TestSeedVarianceContracts (Batch C, Task 15)
# ===========================================================================


class TestSeedVarianceContracts:
    """seeds_per_config / random_seed prose contracts."""

    def test_hp_tune_documents_seed_replicates(self):
        content = _read_file(SKILLS_DIR / "hp-tune" / "SKILL.md")
        assert content is not None
        assert "seeds_per_config" in content
        assert "random_seed" in content
        assert "Seed-replicate exemption" in content
        assert "PYTHONHASHSEED" in content

    def test_phase0_offers_seeds_per_config(self):
        content = _read_file(REFERENCES_DIR / "phase-0-discovery.md")
        assert content is not None
        assert "seeds_per_config" in content

    def test_phase4_preauthorizes_seeds_per_config(self):
        content = _read_file(REFERENCES_DIR / "phase-4-checkpoint.md")
        assert content is not None
        assert "seeds_per_config" in content

    def test_analyze_noise_floor_wording(self):
        content = _read_file(SKILLS_DIR / "analyze" / "SKILL.md")
        assert content is not None
        assert "noise floor" in content
        assert "replicate" in content

    def test_report_template_noise_sentence_measured(self):
        content = _read_file(SKILLS_DIR / "report" / "references" / "report-template.md")
        assert content is not None
        assert "Improvements < 1% relative may be within random variation" not in content
        assert "replicate spread" in content


# ===========================================================================
# TestLowNContract (Batch C, Task 16)
# ===========================================================================


class TestLowNContract:
    """Analyze must treat low-n correlations as preliminary."""

    def test_analyze_low_n_instruction(self):
        content = _read_file(SKILLS_DIR / "analyze" / "SKILL.md")
        assert content is not None
        assert "low_n" in content
        assert "NEVER base a pivot" in content


# ===========================================================================
# TestWarmStartRLCaveats (Batch C, Task 17)
# ===========================================================================


class TestWarmStartRLCaveats:
    """RL warm-start requirements + analyze checkpoint_source segregation."""

    def test_hp_tune_rl_warm_start_requirements(self):
        content = _read_file(SKILLS_DIR / "hp-tune" / "SKILL.md")
        assert content is not None
        assert "RL warm-start requirement" in content
        assert "optimizer state" in content
        assert "normalizer" in content
        assert "replay buffer" in content
        assert "total_timesteps" in content

    def test_analyze_segregates_checkpoint_source(self):
        content = _read_file(SKILLS_DIR / "analyze" / "SKILL.md")
        assert content is not None
        assert "Warm-start segregation" in content
        assert "checkpoint_source" in content


# ===========================================================================
# TestEmbodiedResearchCoverage (Batch E, Task 22)
# ===========================================================================


class TestEmbodiedResearchCoverage:
    """Embodied-AI research/implement coverage: RL query expansion, VLA query set, RL patterns."""

    def test_research_rl_queries_expanded(self):
        """RL search section covers sim2real/domain randomization/curriculum + PPO/SAC tricks."""
        text = _read_file(SKILLS_DIR / "research" / "SKILL.md")
        assert "domain randomization sim-to-real curriculum learning" in text
        assert "PPO SAC implementation tricks" in text

    def test_research_vla_query_set(self):
        """A dedicated VLA/imitation-learning query set exists."""
        text = _read_file(SKILLS_DIR / "research" / "SKILL.md")
        assert "### VLA / Imitation Learning search queries" in text
        assert "action chunking diffusion policy imitation learning" in text
        assert "demonstration augmentation co-training" in text

    def test_implementation_patterns_rl_section(self):
        """implementation-patterns.md has the RL/robot-learning section + detection rows."""
        text = _read_file(
            SKILLS_DIR / "implement" / "references" / "implementation-patterns.md")
        assert "## 10. RL / Robot-Learning Codebases" in text
        assert "Stable-Baselines3" in text
        assert "IsaacLab / rsl_rl" in text
        assert "LeRobot" in text
        assert "robomimic" in text
        assert "library-owned" in text

    def test_implement_test_focus_env_reward_row(self):
        """The implement skill's test-focus table has an env/reward row."""
        text = _read_file(SKILLS_DIR / "implement" / "SKILL.md")
        assert "| Env/reward change |" in text


# ===========================================================================
# TestRLScopeRows (Batch E, Task 23)
# ===========================================================================


class TestRLScopeRows:
    """Both scope tables + implement's goal check carry an RL row per level."""

    def test_research_scope_rl_rows(self):
        text = _read_file(SKILLS_DIR / "research" / "SKILL.md")
        # Appears in BOTH the Inputs bullets and the knowledge-mode table
        assert text.count("entropy/GAE schedules") >= 2
        assert text.count("policy/value network changes") >= 2
        assert text.count("env dynamics, curriculum, domain randomization") >= 2

    def test_implement_goal_check_rl(self):
        text = _read_file(SKILLS_DIR / "implement" / "SKILL.md")
        assert "entropy/GAE schedules" in text
        assert "policy/value network changes" in text
        assert "domain randomization" in text


# ===========================================================================
# TestSecondaryMetricsContract (Batch E, Task 24)
# ===========================================================================


class TestSecondaryMetricsContract:
    """secondary_metrics plumbing: Phase 0 question + analyze guardrail step."""

    def test_phase0_secondary_metrics_question(self):
        text = _read_file(REFERENCES_DIR / "phase-0-discovery.md")
        assert "secondary_metrics" in text
        assert "guardrail" in text

    def test_analyze_reads_secondary_metrics(self):
        text = _read_file(SKILLS_DIR / "analyze" / "SKILL.md")
        assert "secondary_metrics" in text
        assert "Secondary Metrics & Guardrails" in text

    def test_analyze_guardrail_gates_best_and_stacking(self):
        text = _read_file(SKILLS_DIR / "analyze" / "SKILL.md")
        assert "guardrail_regressed" in text
        assert "stacking" in text.split("guardrail_regressed")[1][:600].lower()


class TestSecondaryMetricsHpTune:
    """hp-tune lists secondary_metrics as guardrail-only input."""

    def test_hp_tune_lists_secondary_metrics(self):
        content = _read_file(SKILLS_DIR / "hp-tune" / "SKILL.md")
        assert content is not None
        assert "secondary_metrics" in content
        assert "never replaces `primary_metric`" in content


# ===========================================================================
# TestPolarityAndDocDrift (Batch E, Task 25)
# ===========================================================================


class TestPolarityAndDocDrift:
    """Plot polarity flag, metric-rule rewording, monitor common values, report paths."""

    def test_report_plot_invocations_have_polarity_flag(self):
        text = _read_file(SKILLS_DIR / "report" / "SKILL.md")
        # comparison + timeline + sensitivity + progress all carry the flag
        assert text.count("--higher-is-better") >= 4
        assert "when `lower_is_better` is false" in text

    def test_report_no_hardcoded_experiments_path(self):
        text = _read_file(SKILLS_DIR / "report" / "SKILL.md")
        assert "<project_root>/experiments" not in text

    def test_claude_md_metric_rule_uses_divergence_metric(self):
        text = (PLUGIN_ROOT / ".claude" / "CLAUDE.md").read_text()
        assert 'Always monitor `"loss"`' not in text
        assert "Monitor/divergence always uses loss" not in text
        assert text.count("divergence_lower_is_better=false") >= 2

    def test_monitor_common_values_include_reward(self):
        text = _read_file(SKILLS_DIR / "monitor" / "SKILL.md")
        assert '`"reward"`, `"episode_return"`' in text


# ===========================================================================
# TestScriptBugFixes (Batch E, Task 26)
# ===========================================================================


class TestScriptBugFixes:
    """Script templates + SKILL docs: pid/exit-code fixes, checkpoint pre-flight, find-based artifact copy."""

    def test_templates_background_launch_real_exit_code(self):
        text = _read_file(SKILLS_DIR / "experiment" / "references" / "script-templates.md")
        assert "set -o pipefail" in text
        assert "echo $$" not in text  # wrapper pid removed everywhere
        assert "wait $! || EXIT_CODE=$?" in text
        assert "exit $EXIT_CODE" in text

    def test_worktree_artifact_copy_uses_find(self):
        text = _read_file(SKILLS_DIR / "experiment" / "references" / "script-templates.md")
        assert "find . -maxdepth 4" in text
        assert "cp -r *.pt" not in text

    def test_experiment_skill_timeout_wrapper_background(self):
        text = _read_file(SKILLS_DIR / "experiment" / "SKILL.md")
        assert "wait $! || EXIT_CODE=$?" in text
        assert "PIPESTATUS" not in text

    def test_time_budget_checkpoint_preflight(self):
        baseline = _read_file(SKILLS_DIR / "baseline" / "SKILL.md")
        experiment = _read_file(SKILLS_DIR / "experiment" / "SKILL.md")
        assert "eval_checkpoint_missing" in baseline
        assert "eval_checkpoint_missing" in experiment
        assert "mtime" in experiment
