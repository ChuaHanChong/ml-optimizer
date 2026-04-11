#!/usr/bin/env python3
"""Tests for output_contract.py — per-agent output contracts."""

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from output_contract import CONTRACTS, check_outputs, format_injection, get_contract


# ---------------------------------------------------------------------------
# get_contract
# ---------------------------------------------------------------------------


class TestGetContract:
    def test_experiment_agent_returns_4_paths(self, tmp_path):
        contract = get_contract(
            "experiment-agent", str(tmp_path), round_dir="round-3-hp", exp_id="exp-007"
        )
        assert len(contract) == 4
        assert any("exp-007.json" in c["path"] for c in contract)
        assert any("train.log" in c["path"] for c in contract)
        assert any("scripts/round-3-hp/exp-007" in c["path"] for c in contract)
        assert any("artifacts" in c["path"] for c in contract)

    def test_baseline_agent_returns_2_paths(self, tmp_path):
        contract = get_contract("baseline-agent", str(tmp_path))
        assert len(contract) == 2

    def test_unknown_agent_returns_empty(self, tmp_path):
        contract = get_contract("unknown-agent", str(tmp_path))
        assert contract == []

    def test_paths_are_absolute(self, tmp_path):
        contract = get_contract("baseline-agent", str(tmp_path))
        for entry in contract:
            assert str(tmp_path) in entry["path"]

    def test_prerequisites_agent_returns_2_paths(self, tmp_path):
        contract = get_contract("prerequisites-agent", str(tmp_path))
        assert len(contract) == 2
        assert "prerequisites.json" in contract[0]["path"]
        # Second entry is conditional prepared-data/
        assert "prepared-data" in contract[1]["path"]
        assert "required_if" in contract[1]

    def test_research_agent_glob_flag(self, tmp_path):
        contract = get_contract("research-agent", str(tmp_path))
        assert len(contract) == 1
        assert contract[0].get("glob") is True

    def test_tuning_agent_dir_flag(self, tmp_path):
        contract = get_contract("tuning-agent", str(tmp_path), round_dir="round-1-hp")
        assert len(contract) == 1
        assert contract[0].get("dir") is True
        assert "round-1-hp" in contract[0]["path"]

    def test_report_agent_returns_2_paths(self, tmp_path):
        contract = get_contract("report-agent", str(tmp_path))
        assert len(contract) == 2
        assert any("final-report.md" in c["path"] for c in contract)
        assert any("progress_chart.png" in c["path"] for c in contract)

    def test_implement_agent_returns_1_path(self, tmp_path):
        contract = get_contract("implement-agent", str(tmp_path))
        assert len(contract) == 1
        assert "implementation-manifest.json" in contract[0]["path"]

    def test_missing_kwargs_leaves_placeholder(self, tmp_path):
        # experiment-agent needs round_dir and exp_id — omitting them
        contract = get_contract("experiment-agent", str(tmp_path))
        # Should still return entries (with unresolved placeholders)
        assert len(contract) == 4


# ---------------------------------------------------------------------------
# format_injection
# ---------------------------------------------------------------------------


class TestFormatInjection:
    def test_experiment_agent_injection_has_all_paths(self, tmp_path):
        text = format_injection(
            "experiment-agent",
            str(tmp_path),
            round_dir="round-1-hp",
            exp_id="exp-001",
        )
        assert "REQUIRED OUTPUTS" in text
        assert "exp-001.json" in text
        assert "train.log" in text
        assert "scripts/round-1-hp/exp-001" in text
        assert "artifacts" in text
        assert "PreToolUse" in text
        assert "SubagentStop" in text

    def test_unknown_agent_returns_empty(self, tmp_path):
        text = format_injection("unknown-agent", str(tmp_path))
        assert text == ""

    def test_schema_example_included(self, tmp_path):
        text = format_injection(
            "experiment-agent", str(tmp_path), round_dir="r", exp_id="e"
        )
        assert "method_tier" in text  # from schema example

    def test_baseline_schema_example_included(self, tmp_path):
        text = format_injection("baseline-agent", str(tmp_path))
        assert "profiling" in text

    def test_agent_without_schema_example(self, tmp_path):
        text = format_injection("prerequisites-agent", str(tmp_path))
        assert "REQUIRED OUTPUTS" in text
        assert "JSON schema example:" not in text

    def test_numbered_list(self, tmp_path):
        text = format_injection("baseline-agent", str(tmp_path))
        assert "  1." in text
        assert "  2." in text

    def test_descriptions_present(self, tmp_path):
        text = format_injection("baseline-agent", str(tmp_path))
        assert "Baseline metrics" in text
        assert "Baseline training log" in text


# ---------------------------------------------------------------------------
# check_outputs
# ---------------------------------------------------------------------------


class TestCheckOutputs:
    def test_missing_all(self, tmp_path):
        result = check_outputs("baseline-agent", str(tmp_path))
        assert result["complete"] is False
        assert len(result["missing"]) == 2

    def test_all_present(self, tmp_path):
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "baseline.json").write_text("{}")
        (tmp_path / "logs" / "baseline").mkdir(parents=True)
        (tmp_path / "logs" / "baseline" / "train.log").write_text("log")
        result = check_outputs("baseline-agent", str(tmp_path))
        assert result["complete"] is True
        assert len(result["found"]) == 2

    def test_partial_present(self, tmp_path):
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "baseline.json").write_text("{}")
        # logs/baseline/train.log is missing
        result = check_outputs("baseline-agent", str(tmp_path))
        assert result["complete"] is False
        assert len(result["found"]) == 1
        assert len(result["missing"]) == 1

    def test_glob_check(self, tmp_path):
        (tmp_path / "reports").mkdir()
        (tmp_path / "reports" / "research-findings.md").write_text("x")
        result = check_outputs("research-agent", str(tmp_path))
        assert result["complete"] is True

    def test_glob_check_variant(self, tmp_path):
        (tmp_path / "reports").mkdir()
        (tmp_path / "reports" / "research-findings-method-proposals.md").write_text("x")
        result = check_outputs("research-agent", str(tmp_path))
        assert result["complete"] is True

    def test_glob_check_missing(self, tmp_path):
        (tmp_path / "reports").mkdir()
        # No research-findings*.md files
        result = check_outputs("research-agent", str(tmp_path))
        assert result["complete"] is False

    def test_dir_check(self, tmp_path):
        (tmp_path / "results" / "round-1-hp").mkdir(parents=True)
        (tmp_path / "results" / "round-1-hp" / "exp-001.json").write_text("{}")
        (tmp_path / "logs" / "round-1-hp" / "exp-001").mkdir(parents=True)
        (tmp_path / "logs" / "round-1-hp" / "exp-001" / "train.log").write_text("log")
        (tmp_path / "scripts" / "round-1-hp" / "exp-001").mkdir(parents=True)
        (tmp_path / "artifacts" / "round-1-hp" / "exp-001").mkdir(parents=True)
        result = check_outputs(
            "experiment-agent",
            str(tmp_path),
            round_dir="round-1-hp",
            exp_id="exp-001",
        )
        assert result["complete"] is True

    def test_dir_check_missing(self, tmp_path):
        result = check_outputs(
            "tuning-agent", str(tmp_path), round_dir="round-1-hp"
        )
        assert result["complete"] is False
        assert len(result["missing"]) == 1

    def test_dir_check_present(self, tmp_path):
        (tmp_path / "proposed-configs" / "round-1-hp").mkdir(parents=True)
        result = check_outputs(
            "tuning-agent", str(tmp_path), round_dir="round-1-hp"
        )
        assert result["complete"] is True

    def test_unknown_agent_always_complete(self, tmp_path):
        result = check_outputs("unknown-agent", str(tmp_path))
        assert result["complete"] is True

    def test_report_agent_missing(self, tmp_path):
        result = check_outputs("report-agent", str(tmp_path))
        assert result["complete"] is False
        assert len(result["missing"]) == 2

    def test_analysis_agent_missing_both(self, tmp_path):
        (tmp_path / "reports").mkdir()
        result = check_outputs("analysis-agent", str(tmp_path))
        assert result["complete"] is False
        assert len(result["missing"]) == 1  # one any_of group

    def test_analysis_agent_batch_satisfies(self, tmp_path):
        (tmp_path / "reports").mkdir()
        (tmp_path / "reports" / "batch-3-analysis.md").write_text("# batch 3")
        result = check_outputs("analysis-agent", str(tmp_path))
        assert result["complete"] is True

    def test_analysis_agent_session_review_satisfies(self, tmp_path):
        (tmp_path / "reports").mkdir()
        (tmp_path / "reports" / "session-review.md").write_text("# review")
        result = check_outputs("analysis-agent", str(tmp_path))
        assert result["complete"] is True

    def test_analysis_agent_either_alone_is_enough(self, tmp_path):
        (tmp_path / "reports").mkdir()
        (tmp_path / "reports" / "batch-1-analysis.md").write_text("# b1")
        (tmp_path / "reports" / "batch-2-analysis.md").write_text("# b2")
        result = check_outputs("analysis-agent", str(tmp_path))
        assert result["complete"] is True
        assert len(result["found"]) == 1

    def test_format_injection_any_of(self):
        text = format_injection("analysis-agent", "/tmp/exp")
        assert "AT LEAST ONE of:" in text
        assert "batch-*-analysis.md" in text
        assert "session-review.md" in text


class TestRequiredIf:
    """required_if entries are enforced only when a referenced JSON predicate is true."""

    def _write_prereq(self, tmp_path, prepared):
        results = tmp_path / "results"
        results.mkdir()
        (results / "prerequisites.json").write_text(json.dumps(
            {"dataset": {"prepared": prepared, "train_path": "/data/train"}}
        ))

    def test_condition_false_skips_entry(self, tmp_path):
        """prepared=false → prepared-data/ is not required → complete."""
        self._write_prereq(tmp_path, prepared=False)
        result = check_outputs("prerequisites-agent", str(tmp_path))
        assert result["complete"] is True
        assert "/prepared-data/" not in " ".join(result["missing"])

    def test_condition_true_and_dir_missing_blocks(self, tmp_path):
        """prepared=true but no prepared-data/ → block."""
        self._write_prereq(tmp_path, prepared=True)
        result = check_outputs("prerequisites-agent", str(tmp_path))
        assert result["complete"] is False
        assert any("prepared-data" in m for m in result["missing"])

    def test_condition_true_and_dir_exists_approves(self, tmp_path):
        """prepared=true AND prepared-data/ exists → complete."""
        self._write_prereq(tmp_path, prepared=True)
        (tmp_path / "prepared-data").mkdir()
        result = check_outputs("prerequisites-agent", str(tmp_path))
        assert result["complete"] is True
        assert any("prepared-data" in f for f in result["found"])

    def test_missing_reference_file_skips_entry(self, tmp_path):
        """No prerequisites.json at all → can't evaluate condition → skip."""
        # Only the unconditional prerequisites.json is missing — fails on that,
        # but the required_if entry must NOT also appear as a failure.
        result = check_outputs("prerequisites-agent", str(tmp_path))
        assert result["complete"] is False
        # Missing list should contain only the unconditional prereq, not prepared-data
        assert not any("prepared-data" in m for m in result["missing"])

    def test_malformed_reference_file_skips_entry(self, tmp_path):
        """Malformed prerequisites.json → condition can't be evaluated → skip gracefully."""
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "prerequisites.json").write_text("not valid json")
        result = check_outputs("prerequisites-agent", str(tmp_path))
        # Unconditional file exists but the conditional is skipped
        assert not any("prepared-data" in m for m in result["missing"])

    def test_jsonpath_navigation_missing_key(self, tmp_path):
        """Reference file exists but jsonpath key is missing → treat as false → skip."""
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "prerequisites.json").write_text(json.dumps(
            {"dataset": {"train_path": "/data"}}  # no 'prepared' key
        ))
        result = check_outputs("prerequisites-agent", str(tmp_path))
        assert not any("prepared-data" in m for m in result["missing"])

    def test_format_injection_shows_conditional(self):
        text = format_injection("prerequisites-agent", "/tmp/exp")
        assert "conditional" in text
        assert "dataset.prepared" in text


# ---------------------------------------------------------------------------
# CONTRACTS integrity
# ---------------------------------------------------------------------------


class TestAllContractsHaveDescriptions:
    def test_all_entries_have_description(self):
        for agent, entries in CONTRACTS.items():
            for entry in entries:
                assert "description" in entry, f"{agent} missing description"
                assert "path" in entry or "any_of" in entry, (
                    f"{agent} entry must have either 'path' or 'any_of'"
                )

    def test_all_paths_start_with_exp_root(self):
        for agent, entries in CONTRACTS.items():
            for entry in entries:
                if "any_of" in entry:
                    for alt in entry["any_of"]:
                        assert alt.startswith("{exp_root}/"), (
                            f"{agent} any_of path does not start with "
                            f"{{exp_root}}/: {alt}"
                        )
                else:
                    assert entry["path"].startswith("{exp_root}/"), (
                        f"{agent} path does not start with "
                        f"{{exp_root}}/: {entry['path']}"
                    )

    def test_known_agents_match_architecture(self):
        """Verify that all agents in CONTRACTS are real plugin agents."""
        expected = {
            "prerequisites-agent",
            "baseline-agent",
            "research-agent",
            "implement-agent",
            "tuning-agent",
            "experiment-agent",
            "analysis-agent",
            "report-agent",
        }
        assert set(CONTRACTS.keys()) == expected
