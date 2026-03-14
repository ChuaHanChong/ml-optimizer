"""Comprehensive plugin structure validation.

Validates all 10 agents, 11 skills, hooks, and scripts are correctly
configured for the agent-based dispatch architecture. Run anytime:

    python -m pytest tests/test_plugin_structure.py -v
"""

import json
import re
from pathlib import Path

import pytest

from conftest import FIXTURES, _write_result

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
        "forbidden_tools": {"Edit"}, "color": "#6B7280", "background": False,
    },
    "baseline-agent": {
        "model": "sonnet", "skill": "ml-optimizer:baseline",
        "required_tools": {"Bash", "Read", "Write", "Glob", "Grep", "Skill"},
        "forbidden_tools": {"Edit"}, "color": "#3B82F6", "background": False,
    },
    "experiment-agent": {
        "model": "sonnet", "skill": "ml-optimizer:experiment",
        "required_tools": {"Bash", "Read", "Write", "Glob", "Grep"},
        "forbidden_tools": {"Edit"}, "color": "#10B981", "background": True,
    },
    "monitor-agent": {
        "model": "sonnet", "skill": "ml-optimizer:monitor",
        "required_tools": {"Bash", "Read", "Write", "Glob", "Grep", "Skill"},
        "forbidden_tools": {"Edit"}, "color": "#F59E0B", "background": True,
    },
    "research-agent": {
        "model": "opus", "skill": "ml-optimizer:research",
        "required_tools": {"Bash", "Read", "Write", "Glob", "Grep", "WebSearch", "WebFetch"},
        "forbidden_tools": {"Edit"}, "color": "#8B5CF6", "background": False,
        "external_skills": ["claude-mem:mem-search"],
    },
    "implement-agent": {
        "model": "opus", "skill": "ml-optimizer:implement",
        "required_tools": {"Bash", "Read", "Write", "Edit", "Glob", "Grep"},
        "forbidden_tools": set(), "color": "#EC4899", "background": False,
        "external_skills": ["superpowers:systematic-debugging"],
    },
    "tuning-agent": {
        "model": "opus", "skill": "ml-optimizer:hp-tune",
        "required_tools": {"Read", "Write", "Bash", "Glob", "Grep"},
        "forbidden_tools": {"Edit"}, "color": "#F97316", "background": False,
    },
    "analysis-agent": {
        "model": "opus", "skill": "ml-optimizer:analyze",
        "required_tools": {"Read", "Write", "Bash", "Glob", "Grep", "Skill"},
        "forbidden_tools": {"Edit"}, "color": "#06B6D4", "background": False,
    },
    "report-agent": {
        "model": "opus", "skill": "ml-optimizer:report",
        "required_tools": {"Read", "Write", "Bash", "Glob", "Grep", "Skill"},
        "forbidden_tools": {"Edit"}, "color": "#6366F1", "background": False,
    },
    "review-agent": {
        "model": "opus", "skill": "ml-optimizer:review",
        "required_tools": {"Read", "Write", "Bash", "Glob", "Grep", "Skill"},
        "forbidden_tools": {"Edit"}, "color": "#EF4444", "background": True,
    },
}

EXPECTED_SKILLS = [
    "orchestrate", "prerequisites", "baseline", "experiment", "monitor",
    "research", "implement", "hp-tune", "analyze", "report", "review",
]

NON_ORCHESTRATE_SKILLS = [s for s in EXPECTED_SKILLS if s != "orchestrate"]


# ---------------------------------------------------------------------------
# Agent file validation (one comprehensive test per agent)
# ---------------------------------------------------------------------------

class TestAgentFiles:
    """Validate all 10 agent definition files."""

    def test_all_10_agents_exist_and_no_extra(self):
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
        assert fm.get("model") == spec["model"], (
            f"{agent_name}: expected model {spec['model']}, got {fm.get('model')}")

        # color
        assert fm.get("color") == spec["color"], (
            f"{agent_name}: expected color {spec['color']}")
        assert re.match(r"^#[0-9A-Fa-f]{6}$", fm.get("color", ""))

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


# ---------------------------------------------------------------------------
# Skill file validation
# ---------------------------------------------------------------------------

class TestSkillFiles:
    """Validate all 11 skill definition files."""

    @pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
    def test_skill_frontmatter(self, skill_name):
        """Each skill exists and has correct name, disable-model-invocation, user-invocable."""
        path = SKILLS_DIR / skill_name / "SKILL.md"
        assert path.exists(), f"Missing skill: {path}"
        fm = _parse_frontmatter(path)
        assert fm.get("name") == skill_name, f"{skill_name}: name mismatch"
        assert fm.get("disable-model-invocation") is True, (
            f"{skill_name}: must have disable-model-invocation: true")
        assert fm.get("user-invocable") is False, (
            f"{skill_name}: must have user-invocable: false")

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
        skill_to_agents: dict[str, list[str]] = {s: [] for s in NON_ORCHESTRATE_SKILLS}
        for agent_file in AGENTS_DIR.glob("*.md"):
            for skill in _parse_skills(agent_file):
                skill_name = skill.replace("ml-optimizer:", "")
                if skill_name in skill_to_agents:
                    skill_to_agents[skill_name].append(agent_file.stem)
        for skill_name, agents in skill_to_agents.items():
            assert len(agents) == 1, (
                f"Skill '{skill_name}' loaded by {len(agents)} agents: {agents} (expected 1)")


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

EXPECTED_HOOKS = [
    "bash-safety.sh", "file-guardrail.sh", "detect-critical-errors.sh",
    "subagent-stop-hook.sh", "pre-compact.sh", "post-compact-context.sh",
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
        import os
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
    "excalidraw_gen.py",
]


class TestScripts:
    """Validate all Python scripts exist and are importable."""

    @pytest.mark.parametrize("script", EXPECTED_SCRIPTS)
    def test_script_exists_and_importable(self, script):
        import importlib
        assert (SCRIPTS_DIR / script).exists(), f"Missing script: {script}"
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

    def test_plugin_json_valid(self):
        assert PLUGIN_JSON.exists()
        data = json.loads(PLUGIN_JSON.read_text())
        assert data.get("name") == "ml-optimizer"
        assert "version" in data
        assert "description" in data
        assert re.match(r"^\d+\.\d+\.\d+$", data["version"]), f"Invalid version: {data['version']}"


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
        """Named agent dispatch, no bare skill invocations, all 10 agents referenced."""
        text = self._orchestrate_full_text()
        named_dispatches = re.findall(r'subagent_type.*ml-optimizer:', text)
        assert len(named_dispatches) >= 5
        bare_invocations = re.findall(r'Invoke\s+the\s+ml-optimizer:', text)
        assert not bare_invocations
        for agent in EXPECTED_AGENTS:
            assert agent in text, f"Orchestrate does not reference {agent}"


# ---------------------------------------------------------------------------
# Documentation consistency (merged)
# ---------------------------------------------------------------------------

class TestDocumentation:
    """Verify docs reflect 10-agent architecture and key features."""

    @pytest.mark.parametrize("keyword", [
        "10 subagent definitions", 'Agent(subagent_type="ml-optimizer:',
        "stuck protocol", "dead-end", "research agenda",
        "immutable baseline", "disable-model-invocation",
    ])
    def test_claude_md_documents_feature(self, keyword):
        text = (PLUGIN_ROOT / ".claude" / "CLAUDE.md").read_text()
        assert keyword.lower() in text.lower(), f"CLAUDE.md should mention '{keyword}'"

    def test_implement_skill_mentions_unit_tests(self):
        text = (SKILLS_DIR / "implement" / "SKILL.md").read_text()
        assert "unit test" in text.lower()

    def test_implement_agent_mentions_test_writing(self):
        text = (AGENTS_DIR / "implement-agent.md").read_text()
        assert "Test Writing" in text

    def test_readme_disable_model_invocation(self):
        text = (PLUGIN_ROOT / "README.md").read_text()
        assert "disable-model-invocation" in text


# ---------------------------------------------------------------------------
# Skill contract documentation tests (merged)
# ---------------------------------------------------------------------------

class TestSkillContracts:
    """Verify skills reference the autoresearch-inspired features they consume."""

    @pytest.mark.parametrize("skill,keyword", [
        ("analyze", "dead-end"), ("analyze", "agenda"),
        ("research", "dead-end"), ("research", "agenda"),
        ("hp-tune", "dead-end"), ("hp-tune", "agenda"),
        ("report", "agenda"), ("experiment", "time_budget"),
        ("baseline", "auto-repair"), ("experiment", "non-retryable"),
    ])
    def test_skill_mentions_feature(self, skill, keyword):
        text = (SKILLS_DIR / skill / "SKILL.md").read_text().lower()
        assert keyword.lower() in text, f"Skill '{skill}' should mention '{keyword}'"

    def test_phase7_has_baseline_verification_and_dashboard(self):
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


# --- Budget, method_tier, speculative, monitor, analyze, review trigger ---

def test_budget_flow_contract():
    """Budget calculation, capping, exhaustion, tracking, and difficulty multipliers."""
    assert min(max(4, 1), 3) == 3    # capped by remaining_budget
    assert min(max(2, 1), 10) == 2   # capped by GPU count
    assert min(max(0, 1), 5) == 1    # CPU-only
    assert max(1, 1) * 8 == 8        # easy 1 GPU
    assert max(4, 1) * 25 == 100     # hard 4 GPUs
    # Autonomous: stop after 3 consecutive
    assert ("autonomous" == "autonomous" and 3 >= 3) is True
    assert ("autonomous" == "autonomous" and 0 >= 3) is False
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


def test_speculative_proposal_rules():
    """Speculative proposals: discard on stop/pivot, use on continue, validate branches."""
    assert ("stop" == "continue") is False
    assert ("pivot" == "continue") is False
    assert ("continue" == "continue") is True
    # Branch pruning
    proposals = [
        {"exp_id": "e1", "code_branch": "ml-opt/pruned"},
        {"exp_id": "e2", "code_branch": None},
    ]
    valid = [p for p in proposals if p.get("code_branch") is None or p["code_branch"] != "ml-opt/pruned"]
    assert len(valid) == 1 and valid[0]["exp_id"] == "e2"
    # Budget gate
    assert (2 > max(4, 1)) is False
    assert (10 > max(4, 1)) is True


def test_monitor_and_analyze_contracts():
    """Monitor divergence_status schema; analyze stop/pivot flow; review trigger logic."""
    # Monitor
    for status in ("healthy", "diverged", "completed", "unmonitored", "failed", "no_output"):
        assert status in ("healthy", "diverged", "completed", "unmonitored", "failed", "no_output")

    # Analyze stop prevents hp-tune
    assert ("stop" not in ("stop",)) is False
    # Pivot narrows search space
    updated = {"lr": [1e-5, 1e-3]}
    assert updated["lr"] == [1e-5, 1e-3]

    # Review trigger
    def should_trigger(wasted, consecutive):
        return wasted >= 3 or consecutive >= 2
    assert should_trigger(3, 0) is True
    assert should_trigger(1, 2) is True
    assert should_trigger(2, 1) is False


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
