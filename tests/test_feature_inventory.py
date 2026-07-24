"""Feature-inventory keep-set gate — every pre-existing feature survives.

Each entry maps a feature (README Key Features row or CLAUDE.md design
pattern) to grep-style probes: (repo-relative path, literal needle). If a
refactor drops a needle, the feature it anchors was likely trimmed — restore
it, or consciously retire the feature by updating the keep-set AND the
README/CLAUDE.md row in the same change.
"""

import re

import pytest

from conftest import PLUGIN_ROOT

# Feature name -> [(repo-relative path, literal needle), ...]
FEATURE_PROBES = {
    # --- README "### Key Features" table rows ---
    "Workflow-Driven Loop": [
        ("skills/orchestrate/workflows/phase-7-experiment.js", "export const meta"),
    ],
    "ShinkaEvolve": [("skills/evolve/SKILL.md", "shinka")],
    "Research-Implement": [("skills/implement/SKILL.md", "ml-opt/")],
    "Stuck Protocol": [("skills/orchestrate/SKILL.md", "stuck protocol")],
    "Dead-End Catalog": [("scripts/error_tracker.py", "dead-end")],
    "Research Agenda": [("scripts/error_tracker.py", "research-agenda.json")],
    "Progress Dashboard": [("scripts/dashboard.py", "--live")],
    "Immutable Baseline": [("scripts/pipeline_state.py", "baseline_checksum")],
    "Goal Anchoring": [("scripts/goal_memory.py", "optimization-goals.json")],
    "Cross-Session Learning": [("agents/research-agent.md", "claude-mem")],
    "Behavioral Memory": [("scripts/goal_memory.py", "learned-behaviors.json")],
    "Workflow-Driven Phases 5–8": [
        ("skills/orchestrate/workflows/phase-5-research.js", "export const meta"),
        ("skills/orchestrate/workflows/phase-6-implement.js", "export const meta"),
        ("skills/orchestrate/workflows/phase-8-stacking.js", "export const meta"),
    ],
    "File/Args Handoff": [("scripts/schema_validator.py", "def validate_relay")],
    "GitNexus Code Graph": [("scripts/gitnexus_utils.py", "--index-only")],
    # --- CLAUDE.md design patterns not named in the README table ---
    "Three-tier result tracking": [("scripts/schema_validator.py", "method_tier")],
    "Round lifecycle": [("scripts/round_manager.py", "rounds-manifest.json")],
    "OOM feedback loop": [("scripts/validate_experiment_write.py", "OOM limit")],
    "Model-category divergence defaults": [
        ("scripts/detect_divergence.py", "MODEL_CATEGORY_DEFAULTS"),
    ],
    "HP interaction detection": [("scripts/result_analyzer.py", "detect_hp_interactions")],
    "Checkpoint warm-starting": [("scripts/experiment_setup.py", "checkpoint_path")],
    "3-checkpoint output contracts": [("scripts/output_contract.py", "any_of")],
    "Dataset format detection": [("scripts/prerequisites_check.py", "_FORMAT_PATTERNS")],
    "HF Trainer log parsing": [("scripts/parse_logs.py", "hf_trainer")],
    "Plot polarity flag": [("scripts/plot_results.py", "--higher-is-better")],
    "Excalidraw diagrams": [("scripts/excalidraw_gen.py", "hp-landscape")],
    "GPU-aware installs": [("scripts/prerequisites_check.py", "gpu_install_command")],
}

_ALL_PROBES = [
    pytest.param(path, needle, id=f"{name}:{path}")
    for name, probes in FEATURE_PROBES.items()
    for path, needle in probes
]


@pytest.mark.parametrize("path,needle", _ALL_PROBES)
def test_feature_probe_survives(path, needle):
    target = PLUGIN_ROOT / path
    assert target.is_file(), f"keep-set file missing: {path}"
    assert needle in target.read_text(), (
        f"feature marker {needle!r} missing from {path} — a pre-existing "
        f"feature was trimmed (all existing features must be preserved)"
    )


def test_readme_key_features_all_probed():
    """Every row of README '### Key Features' has a keep-set probe."""
    readme = (PLUGIN_ROOT / "README.md").read_text()
    section = readme.split("### Key Features", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\| \*\*(.+?)\*\*", section, re.MULTILINE)
    assert rows, "README Key Features table not found"
    missing = [r for r in rows if r not in FEATURE_PROBES]
    assert not missing, (
        f"README features without a keep-set probe: {missing} — "
        f"add (path, needle) entries to FEATURE_PROBES"
    )
