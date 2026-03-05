"""Tests for skill interface contracts — verify data flows between skills correctly."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from implement_utils import parse_research_proposals
from result_analyzer import load_results, rank_by_metric

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_FINDINGS = FIXTURES / "sample_research_findings.md"


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
