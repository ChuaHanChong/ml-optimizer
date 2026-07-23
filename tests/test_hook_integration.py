"""Tests for the 3-checkpoint enforcement hooks — subprocess integration."""

import json
import os
import subprocess

import pytest  # noqa: F401  (enables pytest fixture discovery)

from conftest import HOOKS_DIR, PLUGIN_ROOT, SCRIPTS_DIR

SUBAGENT_START_HOOK = HOOKS_DIR / "subagent-start-inject-goals.sh"
VALIDATE_WRITE_SCRIPT = SCRIPTS_DIR / "validate_experiment_write.py"
VALIDATE_OUTPUT_SCRIPT = SCRIPTS_DIR / "validate_agent_output.py"
DEV_NOTES_SCRIPT = SCRIPTS_DIR / "dev_notes.py"


def _run_hook(hook_path, stdin_json, extra_env=None):
    """Invoke a hook script with stdin JSON, return (exit_code, stdout, stderr)."""
    env = os.environ.copy()
    env["CLAUDE_PLUGIN_ROOT"] = str(PLUGIN_ROOT)
    if extra_env:
        env.update(extra_env)
    if hook_path.suffix == ".sh":
        cmd = ["bash", str(hook_path)]
    else:
        cmd = ["python3", str(hook_path)]
    result = subprocess.run(
        cmd,
        input=json.dumps(stdin_json),
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
    )
    return result.returncode, result.stdout, result.stderr


def _parse_decision(stdout: str) -> dict:
    """Parse the last JSON decision object from stdout.

    Raises `AssertionError` if no decision JSON is present — this gives
    tests a clear failure message instead of a `NoneType` subscript error.
    """
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if "decision" in obj:
                    return obj
            except json.JSONDecodeError:
                continue
    raise AssertionError(f"no decision JSON in hook stdout: {stdout!r}")


@pytest.fixture
def exp_root(tmp_path):
    """Create a minimal experiment directory with breadcrumb for hook scripts.

    Returns the project root (not exp_root) since hooks use cwd to find the
    .claude/ml-optimizer.json breadcrumb that points to the real exp_root.
    """
    project = tmp_path / "project"
    exp = project / "experiments"
    project.mkdir()
    exp.mkdir()
    (exp / "results").mkdir()
    (exp / "logs").mkdir()
    (exp / "scripts").mkdir()
    (exp / "artifacts").mkdir()
    (exp / "reports").mkdir()
    (project / ".claude").mkdir()
    (project / ".claude" / "ml-optimizer.json").write_text(
        json.dumps({"exp_root": str(exp)})
    )
    return project


# ---------------------------------------------------------------------------
# Layer 1: SubagentStart — output contract injection
# ---------------------------------------------------------------------------


class TestLayer1Injection:
    """SubagentStart hook injects the per-agent contract into the prompt."""

    def test_baseline_agent_injection(self, exp_root):
        """Baseline agent receives its required output paths in the injected contract."""
        exit_code, stdout, _ = _run_hook(
            SUBAGENT_START_HOOK,
            {"cwd": str(exp_root), "subagent_type": "ml-optimizer:baseline-agent"},
        )
        assert exit_code == 0
        assert "results/baseline.json" in stdout
        assert "logs/baseline/train.log" in stdout

    def test_analysis_agent_any_of_rendering(self, exp_root):
        """Analysis agent contract renders the any-of choice between batch and review outputs."""
        exit_code, stdout, _ = _run_hook(
            SUBAGENT_START_HOOK,
            {"cwd": str(exp_root), "subagent_type": "ml-optimizer:analysis-agent"},
        )
        assert exit_code == 0
        assert "AT LEAST ONE of" in stdout
        assert "batch-*-analysis.md" in stdout
        assert "session-review.md" in stdout

    def test_prerequisites_agent_required_if_rendering(self, exp_root):
        """Prerequisites agent contract renders the conditional prepared-data requirement."""
        exit_code, stdout, _ = _run_hook(
            SUBAGENT_START_HOOK,
            {"cwd": str(exp_root), "subagent_type": "ml-optimizer:prerequisites-agent"},
        )
        assert exit_code == 0
        assert "conditional" in stdout
        assert "dataset.prepared" in stdout

    def test_experiment_agent_placeholder_preservation(self, exp_root):
        """Unresolved {round_dir}/{exp_id} placeholders are preserved for agent substitution."""
        exit_code, stdout, _ = _run_hook(
            SUBAGENT_START_HOOK,
            {"cwd": str(exp_root), "subagent_type": "ml-optimizer:experiment-agent"},
        )
        assert exit_code == 0
        assert "{round_dir}" in stdout
        assert "{exp_id}" in stdout

    def test_missing_breadcrumb_exits_silently(self, tmp_path):
        """When .claude/ml-optimizer.json is absent, hook exits 0 without output."""
        exit_code, stdout, _ = _run_hook(
            SUBAGENT_START_HOOK,
            {"cwd": str(tmp_path), "subagent_type": "ml-optimizer:baseline-agent"},
        )
        assert exit_code == 0
        # No contract text should appear since exp_root couldn't be found
        assert "REQUIRED OUTPUTS" not in stdout

    def test_unknown_agent_type_exits_silently(self, exp_root):
        """Unknown agent type produces no contract output, but hook still exits 0."""
        exit_code, stdout, _ = _run_hook(
            SUBAGENT_START_HOOK,
            {"cwd": str(exp_root), "subagent_type": "ml-optimizer:fake-agent"},
        )
        assert exit_code == 0
        assert "REQUIRED OUTPUTS for fake-agent" not in stdout


# ---------------------------------------------------------------------------
# Layer 2: PreToolUse Write/Edit — write validation
# ---------------------------------------------------------------------------


def _write_payload(exp_root, rel_path, content):
    """Build a PreToolUse hook_input for a Write tool."""
    full = exp_root / "experiments" / rel_path
    return {
        "cwd": str(exp_root),
        "tool_name": "Write",
        "tool_input": {
            "file_path": str(full),
            "content": content if isinstance(content, str) else json.dumps(content),
        },
    }


class TestLayer2WriteValidation:
    """PreToolUse hook blocks invalid writes and approves valid ones."""

    def test_valid_round_based_result_approved(self, exp_root):
        """A complete result written to a round subdirectory is approved."""
        payload = _write_payload(exp_root, "results/round-1-hp/exp-001.json", {
            "exp_id": "exp-001", "status": "completed",
            "config": {"lr": 0.01}, "metrics": {"loss": 0.4},
            "iteration": 1, "method_tier": "baseline", "duration_seconds": 120.0,
        })
        _, stdout, _ = _run_hook(VALIDATE_WRITE_SCRIPT, payload)
        decision = _parse_decision(stdout)
        assert decision["decision"] == "approve"

    def test_flat_results_path_blocked(self, exp_root):
        """A result written directly under results/ instead of a round subdirectory is blocked."""
        payload = _write_payload(exp_root, "results/exp-002.json", {
            "exp_id": "exp-002", "status": "completed",
            "config": {}, "metrics": {"loss": 0.4},
            "iteration": 1, "method_tier": "baseline", "duration_seconds": 120.0,
        })
        _, stdout, _ = _run_hook(VALIDATE_WRITE_SCRIPT, payload)
        decision = _parse_decision(stdout)
        assert decision["decision"] == "block"
        assert "round subdirectory" in decision["reason"]

    def test_completed_missing_completeness_fields_blocked(self, exp_root):
        """A completed result missing mandatory completeness fields is blocked."""
        payload = _write_payload(exp_root, "results/round-1-hp/exp-003.json", {
            "exp_id": "exp-003", "status": "completed",
            "config": {}, "metrics": {"loss": 0.4},
        })
        _, stdout, _ = _run_hook(VALIDATE_WRITE_SCRIPT, payload)
        decision = _parse_decision(stdout)
        assert decision["decision"] == "block"
        assert "mandatory fields" in decision["reason"]
        for field in ("iteration", "method_tier", "duration_seconds"):
            assert field in decision["reason"]

    def test_stacked_tier_missing_code_branches_blocked(self, exp_root):
        """A stacked-tier result missing code_branches and stacking_order is blocked."""
        payload = _write_payload(exp_root, "results/round-1-stacked/exp-004.json", {
            "exp_id": "exp-004", "status": "completed",
            "config": {}, "metrics": {"loss": 0.3},
            "iteration": 1, "method_tier": "stacked_default_hp", "duration_seconds": 60,
        })
        _, stdout, _ = _run_hook(VALIDATE_WRITE_SCRIPT, payload)
        decision = _parse_decision(stdout)
        assert decision["decision"] == "block"
        assert "code_branches" in decision["reason"]
        assert "stacking_order" in decision["reason"]

    def test_failed_without_notes_blocked(self, exp_root):
        """A failed result without a notes explanation is blocked."""
        payload = _write_payload(exp_root, "results/round-1-hp/exp-005.json", {
            "exp_id": "exp-005", "status": "failed",
            "config": {}, "metrics": {},
        })
        _, stdout, _ = _run_hook(VALIDATE_WRITE_SCRIPT, payload)
        decision = _parse_decision(stdout)
        assert decision["decision"] == "block"
        assert "notes" in decision["reason"]

    def test_diverged_with_notes_approved(self, exp_root):
        """A diverged result that includes notes is approved."""
        payload = _write_payload(exp_root, "results/round-1-hp/exp-006.json", {
            "exp_id": "exp-006", "status": "diverged",
            "config": {}, "metrics": {}, "notes": "NaN at step 50",
        })
        _, stdout, _ = _run_hook(VALIDATE_WRITE_SCRIPT, payload)
        decision = _parse_decision(stdout)
        assert decision["decision"] == "approve"

    def test_placeholder_running_approved(self, exp_root):
        """A running placeholder result is exempt from completeness checks and approved."""
        payload = _write_payload(exp_root, "results/round-1-hp/exp-007.json", {
            "exp_id": "exp-007", "status": "running",
            "config": {}, "metrics": {},
        })
        _, stdout, _ = _run_hook(VALIDATE_WRITE_SCRIPT, payload)
        decision = _parse_decision(stdout)
        assert decision["decision"] == "approve"

    def test_frozen_parameter_violation_blocked(self, exp_root):
        """A result whose config touches a frozen parameter is blocked."""
        goals = exp_root / "experiments" / "optimization-goals.json"
        goals.write_text(json.dumps(
            {"constraints": {"frozen_parameters": ["model_size", "dataset"]}}
        ))
        payload = _write_payload(exp_root, "results/round-1-hp/exp-008.json", {
            "exp_id": "exp-008", "status": "completed",
            "config": {"lr": 0.01, "model_size": "large"},
            "metrics": {"loss": 0.4},
            "iteration": 1, "method_tier": "baseline", "duration_seconds": 60,
        })
        _, stdout, _ = _run_hook(VALIDATE_WRITE_SCRIPT, payload)
        decision = _parse_decision(stdout)
        assert decision["decision"] == "block"
        assert "frozen parameter" in decision["reason"]
        assert "model_size" in decision["reason"]

    def test_oom_batch_size_cap_blocked(self, exp_root):
        """A result whose batch size exceeds the learned OOM cap is blocked."""
        behaviors = exp_root / "experiments" / "learned-behaviors.json"
        behaviors.write_text(json.dumps(
            {"resource_constraints": [{"max_batch_size": 128}]}
        ))
        payload = _write_payload(exp_root, "results/round-1-hp/exp-009.json", {
            "exp_id": "exp-009", "status": "completed",
            "config": {"batch_size": 512}, "metrics": {"loss": 0.4},
            "iteration": 1, "method_tier": "baseline", "duration_seconds": 60,
        })
        _, stdout, _ = _run_hook(VALIDATE_WRITE_SCRIPT, payload)
        decision = _parse_decision(stdout)
        assert decision["decision"] == "block"
        assert "OOM limit" in decision["reason"]
        assert "512" in decision["reason"]

    def test_valid_proposed_config_approved(self, exp_root):
        """A valid proposed-config written to a round subdirectory is approved."""
        payload = _write_payload(exp_root, "proposed-configs/round-1-hp/exp-010.json", {
            "exp_id": "exp-010",
            "config": {"lr": 0.005, "batch_size": 64},
            "method_tier": "method_tuned_hp",
            "iteration": 2, "code_branch": None, "gpu_id": 0,
            "reasoning": "Lower lr based on prior results",
        })
        _, stdout, _ = _run_hook(VALIDATE_WRITE_SCRIPT, payload)
        decision = _parse_decision(stdout)
        assert decision["decision"] == "approve"

    def test_proposed_config_goal_compliance(self, exp_root):
        """Proposed-configs also go through goal compliance check."""
        behaviors = exp_root / "experiments" / "learned-behaviors.json"
        behaviors.write_text(json.dumps(
            {"resource_constraints": [{"max_batch_size": 128}]}
        ))
        payload = _write_payload(exp_root, "proposed-configs/round-1-hp/exp-011.json", {
            "exp_id": "exp-011",
            "config": {"lr": 0.01, "batch_size": 512},
            "method_tier": "method_tuned_hp",
            "iteration": 2, "code_branch": None, "gpu_id": 0, "reasoning": "test",
        })
        _, stdout, _ = _run_hook(VALIDATE_WRITE_SCRIPT, payload)
        decision = _parse_decision(stdout)
        assert decision["decision"] == "block"
        assert "OOM limit" in decision["reason"]


# ---------------------------------------------------------------------------
# Layer 3: SubagentStop — output verification
# ---------------------------------------------------------------------------


class TestLayer3OutputVerification:
    """SubagentStop hook blocks agents that didn't produce required outputs."""

    def _stop_payload(self, exp_root, agent_type, agent_id=""):
        """Build a SubagentStop hook_input for the given agent type and id."""
        return {
            "cwd": str(exp_root),
            "subagent_type": f"ml-optimizer:{agent_type}",
            "agent_id": agent_id,
        }

    def test_baseline_agent_with_both_outputs_approved(self, exp_root):
        """Baseline agent that produced both its result JSON and training log is approved."""
        exp = exp_root / "experiments"
        (exp / "results" / "baseline.json").write_text('{"exp_id":"baseline"}')
        (exp / "logs" / "baseline").mkdir()
        (exp / "logs" / "baseline" / "train.log").write_text("log data")
        _, stdout, _ = _run_hook(
            VALIDATE_OUTPUT_SCRIPT,
            self._stop_payload(exp_root, "baseline-agent"),
        )
        decision = _parse_decision(stdout)
        assert decision["decision"] == "approve"

    def test_baseline_agent_missing_log_blocked(self, exp_root):
        """Baseline agent missing its training log is blocked from finishing."""
        exp = exp_root / "experiments"
        (exp / "results" / "baseline.json").write_text('{"exp_id":"baseline"}')
        # No log file
        _, stdout, _ = _run_hook(
            VALIDATE_OUTPUT_SCRIPT,
            self._stop_payload(exp_root, "baseline-agent"),
        )
        decision = _parse_decision(stdout)
        assert decision["decision"] == "block"
        assert "train.log" in decision["reason"]

    def test_analysis_agent_any_of_batch_report(self, exp_root):
        """Analysis agent that produced a batch analysis report satisfies the any-of requirement."""
        (exp_root / "experiments" / "reports" / "batch-1-analysis.md").write_text("# Batch 1")
        _, stdout, _ = _run_hook(
            VALIDATE_OUTPUT_SCRIPT,
            self._stop_payload(exp_root, "analysis-agent"),
        )
        decision = _parse_decision(stdout)
        assert decision["decision"] == "approve"

    def test_analysis_agent_any_of_session_review(self, exp_root):
        """Analysis agent that produced a session review satisfies the any-of requirement."""
        (exp_root / "experiments" / "reports" / "session-review.md").write_text("# Review")
        _, stdout, _ = _run_hook(
            VALIDATE_OUTPUT_SCRIPT,
            self._stop_payload(exp_root, "analysis-agent"),
        )
        decision = _parse_decision(stdout)
        assert decision["decision"] == "approve"

    def test_analysis_agent_any_of_neither_blocked(self, exp_root):
        """Analysis agent producing neither any-of output is blocked from finishing."""
        _, stdout, _ = _run_hook(
            VALIDATE_OUTPUT_SCRIPT,
            self._stop_payload(exp_root, "analysis-agent"),
        )
        decision = _parse_decision(stdout)
        assert decision["decision"] == "block"

    def test_prerequisites_required_if_prepared_false_approved(self, exp_root):
        """Prerequisites agent is approved without prepared-data when prepared is false."""
        exp = exp_root / "experiments"
        (exp / "results" / "prerequisites.json").write_text(json.dumps(
            {"status": "ready", "dataset": {"prepared": False, "train_path": "/data"}}
        ))
        _, stdout, _ = _run_hook(
            VALIDATE_OUTPUT_SCRIPT,
            self._stop_payload(exp_root, "prerequisites-agent"),
        )
        decision = _parse_decision(stdout)
        assert decision["decision"] == "approve"

    def test_prerequisites_required_if_prepared_true_dir_missing_blocked(self, exp_root):
        """Prerequisites agent is blocked when prepared is true but prepared-data is missing."""
        exp = exp_root / "experiments"
        (exp / "results" / "prerequisites.json").write_text(json.dumps(
            {"status": "ready", "dataset": {"prepared": True, "train_path": "/data"}}
        ))
        _, stdout, _ = _run_hook(
            VALIDATE_OUTPUT_SCRIPT,
            self._stop_payload(exp_root, "prerequisites-agent"),
        )
        decision = _parse_decision(stdout)
        assert decision["decision"] == "block"
        assert "prepared-data" in decision["reason"]

    def test_prerequisites_required_if_prepared_true_dir_exists_approved(self, exp_root):
        """Prerequisites agent is approved when prepared is true and prepared-data exists."""
        exp = exp_root / "experiments"
        (exp / "results" / "prerequisites.json").write_text(json.dumps(
            {"status": "ready", "dataset": {"prepared": True, "train_path": "/data"}}
        ))
        (exp / "prepared-data").mkdir()
        _, stdout, _ = _run_hook(
            VALIDATE_OUTPUT_SCRIPT,
            self._stop_payload(exp_root, "prerequisites-agent"),
        )
        decision = _parse_decision(stdout)
        assert decision["decision"] == "approve"

    def test_experiment_agent_all_outputs_approved(self, exp_root):
        """Experiment agent that produced all four round-scoped outputs is approved."""
        exp = exp_root / "experiments"
        (exp / "results" / "round-1-hp").mkdir()
        (exp / "results" / "round-1-hp" / "exp-001.json").write_text('{"exp_id":"exp-001"}')
        (exp / "logs" / "round-1-hp" / "exp-001").mkdir(parents=True)
        (exp / "logs" / "round-1-hp" / "exp-001" / "train.log").write_text("log")
        (exp / "scripts" / "round-1-hp" / "exp-001").mkdir(parents=True)
        (exp / "artifacts" / "round-1-hp" / "exp-001").mkdir(parents=True)
        _, stdout, _ = _run_hook(
            VALIDATE_OUTPUT_SCRIPT,
            self._stop_payload(exp_root, "experiment-agent"),
        )
        decision = _parse_decision(stdout)
        assert decision["decision"] == "approve"

    def test_monitor_agent_auto_approved(self, exp_root):
        """Agents not in CONTRACTS (monitor) are always approved."""
        _, stdout, _ = _run_hook(
            VALIDATE_OUTPUT_SCRIPT,
            self._stop_payload(exp_root, "monitor-agent"),
        )
        decision = _parse_decision(stdout)
        assert decision["decision"] == "approve"
