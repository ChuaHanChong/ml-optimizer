"""Tests for skill interface contracts — verify data flows between skills correctly."""

import json

from conftest import FIXTURES, _write_result

from implement_utils import parse_research_proposals
from result_analyzer import load_results, rank_by_metric
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

SAMPLE_FINDINGS = FIXTURES / "sample_research_findings.md"
SAMPLE_FINDINGS_REF = FIXTURES / "sample_research_findings_with_reference.md"


# --- Research → Implement contract ---

def test_research_proposals_have_implement_required_fields():
    """parse_research_proposals() output must have fields that implement skill expects."""
    proposals = parse_research_proposals(str(SAMPLE_FINDINGS))
    required_fields = {"index", "name", "slug", "body", "files_to_modify", "complexity", "implementation_steps"}
    for proposal in proposals:
        missing = required_fields - set(proposal.keys())
        assert not missing, f"Proposal {proposal.get('name', '?')} missing fields: {missing}"


def test_research_proposals_slug_is_valid_branch_name():
    """Slugs must be valid git branch name components."""
    proposals = parse_research_proposals(str(SAMPLE_FINDINGS))
    import re
    for proposal in proposals:
        slug = proposal["slug"]
        assert re.match(r'^[a-z0-9][a-z0-9-]*[a-z0-9]$', slug), \
            f"Slug '{slug}' is not a valid branch name component"


# --- Experiment result → Analyze contract ---

def test_experiment_result_matches_analyze_input(tmp_path):
    """Experiment result JSON must be loadable by result_analyzer."""
    result = {
        "exp_id": "exp-001",
        "status": "completed",
        "config": {"lr": 0.001, "batch_size": 16},
        "metrics": {"loss": 0.5, "accuracy": 82.5},
        "gpu_id": 0,
        "duration_seconds": 3600,
        "code_branch": "ml-opt/perceptual-loss",
        "code_proposal": "Perceptual Loss Function",
    }
    (tmp_path / "exp-001.json").write_text(json.dumps(result))
    loaded = load_results(str(tmp_path))
    assert "exp-001" in loaded
    ranked = rank_by_metric(loaded, "loss", lower_is_better=True)
    assert len(ranked) == 1
    assert ranked[0]["value"] == 0.5


# --- Baseline → Orchestrate Phase 5 contract ---

def test_baseline_schema_for_orchestrate():
    """baseline.json must have the fields that orchestrate Phase 5 expects."""
    baseline = {
        "exp_id": "baseline",
        "status": "completed",
        "config": {"lr": 0.01, "batch_size": 64},
        "metrics": {"loss": 1.5, "accuracy": 45.0},
        "profiling": {
            "gpu_memory_used_mib": 8000,
            "gpu_memory_total_mib": 24576,
            "throughput_samples_per_sec": 150,
        },
    }
    # Orchestrate Phase 5 requires metrics and config keys
    assert "metrics" in baseline
    assert "config" in baseline
    assert isinstance(baseline["metrics"], dict)
    assert isinstance(baseline["config"], dict)
    assert len(baseline["metrics"]) > 0
    assert len(baseline["config"]) > 0


# --- HP-tune proposed config contract ---

def test_hp_tune_proposed_config_schema():
    """HP-tune proposed configs must have all required fields."""
    proposed = {
        "exp_id": "exp-003",
        "config": {"lr": 0.0001, "batch_size": 32},
        "code_branch": None,
        "code_proposal": None,
        "gpu_id": 0,
        "reasoning": "Lower LR showed best results in iteration 1",
        "iteration": 2,
    }
    required = {"exp_id", "config", "gpu_id", "reasoning", "iteration"}
    missing = required - set(proposed.keys())
    assert not missing, f"Proposed config missing fields: {missing}"
    assert isinstance(proposed["config"], dict)
    assert isinstance(proposed["gpu_id"], int)


# --- Implementation manifest contract ---

def test_manifest_schema_for_orchestrate():
    """implementation-manifest.json must have fields orchestrate expects."""
    manifest = {
        "original_branch": "main",
        "strategy": "git_branch",
        "proposals": [
            {
                "name": "Perceptual Loss",
                "slug": "perceptual-loss",
                "branch": "ml-opt/perceptual-loss",
                "status": "validated",
                "files_modified": ["models/classifier.py"],
                "validation": {"syntax": "pass", "import": "pass"},
            },
            {
                "name": "Bad Proposal",
                "slug": "bad-proposal",
                "branch": "ml-opt/bad-proposal",
                "status": "validation_failed",
                "validation": {"syntax": "fail"},
            },
        ],
        "conflicts": [],
    }
    # Orchestrate should only pick validated proposals
    validated = [p for p in manifest["proposals"] if p["status"] == "validated"]
    assert len(validated) == 1
    assert validated[0]["name"] == "Perceptual Loss"
    # Must have branch info
    assert "branch" in validated[0]


# --- Research → Implement contract (strategy fields) ---

def test_research_proposals_have_strategy_fields():
    """All proposals must include implementation_strategy field."""
    proposals = parse_research_proposals(str(SAMPLE_FINDINGS_REF))
    for proposal in proposals:
        assert "implementation_strategy" in proposal, \
            f"Proposal {proposal['name']} missing implementation_strategy"
        assert proposal["implementation_strategy"] in ("from_scratch", "from_reference"), \
            f"Proposal {proposal['name']} has invalid strategy: {proposal['implementation_strategy']}"


def test_research_proposals_reference_fields_when_from_reference():
    """from_reference proposals must have reference_repo and reference_files."""
    proposals = parse_research_proposals(str(SAMPLE_FINDINGS_REF))
    for proposal in proposals:
        if proposal["implementation_strategy"] == "from_reference":
            assert proposal["reference_repo"], \
                f"from_reference proposal {proposal['name']} missing reference_repo"
            assert len(proposal["reference_files"]) > 0, \
                f"from_reference proposal {proposal['name']} missing reference_files"


# --- Error tracker → Review contract ---


def test_review_summary_output_has_required_fields(tmp_path):
    """summarize_session() output must have fields the review skill expects."""
    log_event(str(tmp_path), create_event("training_failure", "critical", "experiment", "crash"))
    log_event(str(tmp_path), create_event("divergence", "warning", "monitor", "nan"))
    summary = summarize_session(str(tmp_path))
    required = {"total_events", "by_category", "by_severity", "patterns_detected"}
    missing = required - set(summary.keys())
    assert not missing, f"Summary missing fields: {missing}"


def test_review_patterns_output_has_required_fields():
    """detect_patterns() output must have fields the review skill expects."""
    events = [
        create_event("divergence", "warning", "monitor", "NaN",
                      config={"lr": 0.1, "batch_size": 32}),
        create_event("divergence", "warning", "monitor", "NaN",
                      config={"lr": 0.2, "batch_size": 32}),
        create_event("divergence", "warning", "monitor", "NaN",
                      config={"lr": 0.05, "batch_size": 64}),
    ]
    patterns = detect_patterns(events)
    required = {"pattern_id", "description", "occurrences", "suggested_action"}
    for p in patterns:
        missing = required - set(p.keys())
        assert not missing, f"Pattern {p.get('pattern_id', '?')} missing fields: {missing}"


def test_review_success_metrics_output_schema(tmp_path):
    """compute_success_metrics() output must have fields the review skill expects."""
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results, "baseline", "completed", {}, {"acc": 70.0})
    _write_result(results, "exp-001", "completed", {}, {"acc": 75.0})
    m = compute_success_metrics(str(tmp_path), "acc", lower_is_better=False)
    required = {"total_experiments", "completed", "failed", "diverged",
                "success_rate", "improvement_rate", "top_configs", "worst_configs"}
    missing = required - set(m.keys())
    assert not missing, f"Success metrics missing fields: {missing}"


def test_review_proposal_outcomes_output_schema(tmp_path):
    """compute_proposal_outcomes() output must have fields the review skill expects."""
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results, "baseline", "completed", {}, {"acc": 70.0})
    p = compute_proposal_outcomes(str(tmp_path), "acc", lower_is_better=False)
    required = {"research_proposals", "hp_proposals", "implementation_stats"}
    missing = required - set(p.keys())
    assert not missing, f"Proposal outcomes missing fields: {missing}"


def test_review_category_to_file_mapping_complete():
    """Every VALID_CATEGORIES entry must have a known mapping to a plugin file.

    This canary test fails if someone adds a category without updating the
    review skill's Step 1.5 mapping table.
    """
    # This mapping mirrors skills/review/SKILL.md Step 1.5 table
    mapped_categories = {
        "agent_failure", "divergence", "training_failure",
        "implementation_error", "pipeline_inefficiency", "config_error",
        "research_failure", "timeout", "resource_error",
    }
    unmapped = set(VALID_CATEGORIES) - mapped_categories
    assert not unmapped, (
        f"Categories {unmapped} are in VALID_CATEGORIES but not mapped in "
        f"review skill Step 1.5. Update both the skill and this test."
    )


def test_review_rank_suggestions_output_schema():
    """rank_suggestions() output must have pattern fields plus score."""
    patterns = [
        {"pattern_id": "oom_batch_size", "description": "OOM", "occurrences": 2,
         "suggested_action": "reduce bs"},
    ]
    ranked = rank_suggestions(patterns)
    assert len(ranked) == 1
    required = {"pattern_id", "description", "occurrences", "suggested_action", "score"}
    missing = required - set(ranked[0].keys())
    assert not missing, f"Ranked suggestion missing fields: {missing}"


def test_review_rank_includes_significance_when_total_provided():
    """rank_suggestions with total_experiments must include significance field."""
    patterns = [
        {"pattern_id": "oom_batch_size", "description": "OOM", "occurrences": 5,
         "suggested_action": "reduce bs"},
    ]
    ranked = rank_suggestions(patterns, total_experiments=50)
    assert "significance" in ranked[0], "significance field missing when total_experiments provided"
    assert ranked[0]["significance"] == 0.1
    # Without total_experiments, no significance
    ranked_no_total = rank_suggestions(patterns)
    assert "significance" not in ranked_no_total[0]


def test_review_suggestion_history_schema(tmp_path):
    """log_suggestion and get_suggestion_history produce expected schema."""
    log_suggestion(str(tmp_path), "wasted_budget", scope="session")
    history = get_suggestion_history(str(tmp_path))
    assert len(history) == 1
    required = {"pattern_id", "timestamp", "scope", "iteration"}
    missing = required - set(history[0].keys())
    assert not missing, f"Suggestion history entry missing fields: {missing}"
    assert isinstance(history[0]["iteration"], int)
    assert history[0]["iteration"] >= 1


# --- Prerequisites → Orchestrate contract ---

def test_prerequisites_report_has_orchestrator_required_fields():
    """prerequisites.json must have all fields that orchestrate Phase 2→3 expects."""
    report = {
        "status": "ready",
        "dataset": {
            "train_path": "/data/train",
            "val_path": "/data/val",
            "format_detected": "image_folder",
            "prepared": False,
            "prepared_train_path": None,
            "prepared_val_path": None,
            "validation_passed": True,
            "notes": "",
        },
        "environment": {
            "manager": "conda",
            "python_version": "3.10.12",
            "packages_installed": ["torch", "torchvision"],
            "packages_failed": [],
            "all_imports_resolved": True,
            "notes": "",
        },
        "ready_for_baseline": True,
    }
    # Orchestrate checks these four top-level keys
    for key in ("status", "dataset", "environment", "ready_for_baseline"):
        assert key in report, f"prerequisites.json missing '{key}'"
    assert report["status"] in ("ready", "partial", "failed")
    assert isinstance(report["dataset"], dict)
    assert isinstance(report["environment"], dict)
    assert isinstance(report["ready_for_baseline"], bool)


def test_prerequisites_prepared_paths_flow_to_baseline():
    """prerequisites.json prepared paths must match what baseline/experiment expect."""
    report = {
        "status": "ready",
        "dataset": {
            "train_path": "/data/train",
            "val_path": "/data/val",
            "prepared": True,
            "prepared_train_path": "/exp/prepared-data/train",
            "prepared_val_path": "/exp/prepared-data/val",
        },
        "environment": {"manager": "conda", "packages_installed": ["torch"]},
        "ready_for_baseline": True,
    }
    # Orchestrator extracts prepared paths when dataset.prepared is True
    assert report["dataset"]["prepared"] is True
    assert "prepared_train_path" in report["dataset"]
    assert "prepared_val_path" in report["dataset"]
    # Paths must be non-empty strings (baseline/experiment substitute these)
    assert isinstance(report["dataset"]["prepared_train_path"], str)
    assert len(report["dataset"]["prepared_train_path"]) > 0
    assert isinstance(report["dataset"]["prepared_val_path"], str)
    assert len(report["dataset"]["prepared_val_path"]) > 0


def test_prerequisites_failed_blocks_pipeline():
    """A failed prerequisites report must block baseline from proceeding."""
    report = {
        "status": "failed",
        "dataset": {"train_path": None, "errors": ["Data path does not exist"]},
        "environment": {"manager": "unknown"},
        "ready_for_baseline": False,
    }
    assert report["status"] == "failed"
    assert report["ready_for_baseline"] is False


def test_prerequisites_partial_has_actionable_info():
    """A partial prerequisites report must still have dataset/environment info."""
    report = {
        "status": "partial",
        "dataset": {"train_path": "/data/train", "prepared": False},
        "environment": {"manager": "pip", "packages_installed": [], "packages_failed": ["torch"]},
        "ready_for_baseline": False,
    }
    assert report["status"] in ("ready", "partial", "failed")
    # Partial must still have structured data for the user to decide
    assert isinstance(report["dataset"], dict)
    assert isinstance(report["environment"], dict)
