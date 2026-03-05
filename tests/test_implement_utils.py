"""Tests for implement_utils.py."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from implement_utils import (
    _extract_reference_files,
    analyze_reference_structure,
    backup_files,
    cleanup_reference_repo,
    clone_reference_repo,
    detect_conflicts,
    get_current_branch,
    is_git_repo,
    parse_research_proposals,
    slugify,
    validate_imports,
    validate_syntax,
    write_manifest,
)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_FINDINGS = FIXTURES / "sample_research_findings.md"
SAMPLE_FINDINGS_REF = FIXTURES / "sample_research_findings_with_reference.md"


# --- slugify ---

def test_slugify_basic():
    assert slugify("Perceptual Loss Function") == "perceptual-loss-function"


def test_slugify_special_chars():
    assert slugify("CutMix (Data Aug.)") == "cutmix-data-aug"


def test_slugify_uppercase_and_extra_spaces():
    assert slugify("  SWIN   Transformer  ") == "swin-transformer"


# --- parse_research_proposals ---

def test_parse_research_proposals():
    proposals = parse_research_proposals(str(SAMPLE_FINDINGS))
    assert len(proposals) == 3
    assert proposals[0]["name"] == "Perceptual Loss Function"
    assert proposals[0]["slug"] == "perceptual-loss-function"
    assert proposals[0]["complexity"] == "Low"
    assert len(proposals[0]["files_to_modify"]) == 2
    assert "models/classifier.py" in proposals[0]["files_to_modify"]
    assert len(proposals[0]["implementation_steps"]) == 3


def test_parse_selected_subset():
    proposals = parse_research_proposals(str(SAMPLE_FINDINGS), selected_indices=[1, 3])
    assert len(proposals) == 2
    assert proposals[0]["index"] == 1
    assert proposals[1]["index"] == 3
    assert proposals[1]["name"] == "CutMix Data Augmentation"


# --- detect_conflicts ---

def test_detect_conflicts():
    proposals = parse_research_proposals(str(SAMPLE_FINDINGS))
    conflicts = detect_conflicts(proposals)
    # Proposals 1 and 2 both modify models/classifier.py
    assert len(conflicts) == 1
    assert conflicts[0]["file"] == "models/classifier.py"
    assert 1 in conflicts[0]["proposal_indices"]
    assert 2 in conflicts[0]["proposal_indices"]


def test_detect_no_conflicts():
    proposals = parse_research_proposals(str(SAMPLE_FINDINGS), selected_indices=[1, 3])
    conflicts = detect_conflicts(proposals)
    assert len(conflicts) == 0


# --- validate_syntax ---

def test_validate_syntax_pass(tmp_path):
    good_file = tmp_path / "good.py"
    good_file.write_text("x = 1 + 2\n")
    results = validate_syntax([str(good_file)])
    assert len(results) == 1
    assert results[0]["passed"] is True
    assert results[0]["error"] is None


def test_validate_syntax_fail(tmp_path):
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("def foo(\n")
    results = validate_syntax([str(bad_file)])
    assert len(results) == 1
    assert results[0]["passed"] is False
    assert results[0]["error"] is not None


# --- backup_files ---

def test_backup_files(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    file1 = src_dir / "model.py"
    file1.write_text("class Model: pass\n")
    file2 = src_dir / "train.py"
    file2.write_text("import model\n")

    backup_dir = tmp_path / "backups"
    mapping = backup_files([str(file1), str(file2)], str(backup_dir))

    assert len(mapping) == 2
    for original, backup in mapping.items():
        assert Path(backup).exists()
        assert Path(backup).read_text() == Path(original).read_text()


def test_backup_files_same_name_different_dirs(tmp_path):
    project = tmp_path / "project"
    models_dir = project / "models"
    data_dir = project / "data"
    models_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    models_utils = models_dir / "utils.py"
    models_utils.write_text("# models utils\n")
    data_utils = data_dir / "utils.py"
    data_utils.write_text("# data utils\n")

    backup_dir = tmp_path / "backups"
    mapping = backup_files(
        [str(models_utils), str(data_utils)],
        str(backup_dir),
        project_root=str(project),
    )

    assert len(mapping) == 2
    # Both backups must exist with correct content
    for original, backup in mapping.items():
        assert Path(backup).exists()
        assert Path(backup).read_text() == Path(original).read_text()

    # Verify directory structure is preserved (no collision)
    assert Path(mapping[str(models_utils)]) == backup_dir / "models" / "utils.py"
    assert Path(mapping[str(data_utils)]) == backup_dir / "data" / "utils.py"


# --- write_manifest ---

def test_write_manifest(tmp_path):
    manifest_path = tmp_path / "results" / "implementation-manifest.json"
    data = {
        "original_branch": "main",
        "strategy": "git_branch",
        "proposals": [{"name": "Test", "status": "validated"}],
        "conflicts": [],
    }
    result = write_manifest(str(manifest_path), data)
    assert Path(result).exists()
    loaded = json.loads(Path(result).read_text())
    assert loaded["original_branch"] == "main"
    assert len(loaded["proposals"]) == 1


# --- validate_imports with quote in path ---

def test_validate_imports_with_quote_in_path(tmp_path):
    dir_with_quote = tmp_path / "it's a dir"
    dir_with_quote.mkdir()
    mod_file = dir_with_quote / "mod.py"
    mod_file.write_text("x = 1\n")
    result = validate_imports(str(mod_file), str(tmp_path))
    assert result["passed"] is True


# --- is_git_repo ---

def test_is_git_repo_true(tmp_path):
    (tmp_path / ".git").mkdir()
    assert is_git_repo(str(tmp_path)) is True


def test_is_git_repo_false(tmp_path):
    assert is_git_repo(str(tmp_path)) is False


# --- get_current_branch ---

def test_get_current_branch():
    project_root = str(Path(__file__).parent.parent)
    branch = get_current_branch(project_root)
    assert isinstance(branch, str)
    assert len(branch) > 0


# --- _extract_reference_files ---

def test_extract_reference_files_backtick():
    body = "- **Reference files:** `basicsr/models/restormer_arch.py`, `basicsr/losses/loss.py`"
    result = _extract_reference_files(body)
    assert result == ["basicsr/models/restormer_arch.py", "basicsr/losses/loss.py"]


def test_extract_reference_files_comma():
    body = "- **Reference files:** models/arch.py, models/loss.py"
    result = _extract_reference_files(body)
    assert result == ["models/arch.py", "models/loss.py"]


def test_extract_reference_files_missing():
    body = "- **Complexity:** Low\n- **Risk:** None"
    result = _extract_reference_files(body)
    assert result == []


# --- parse_research_proposals with reference fields ---

def test_parse_proposals_with_reference_repo():
    proposals = parse_research_proposals(str(SAMPLE_FINDINGS_REF))
    # Proposal 1 is from_reference
    p1 = proposals[0]
    assert p1["implementation_strategy"] == "from_reference"
    assert "github.com/swz30/Restormer" in p1["reference_repo"]
    assert len(p1["reference_files"]) == 2
    assert "basicsr/models/archs/restormer_arch.py" in p1["reference_files"]
    # Proposal 2 is from_scratch
    p2 = proposals[1]
    assert p2["implementation_strategy"] == "from_scratch"
    assert p2["reference_repo"] == ""
    assert p2["reference_files"] == []


def test_parse_proposals_backward_compat():
    """Old fixture without strategy fields gets safe defaults."""
    proposals = parse_research_proposals(str(SAMPLE_FINDINGS))
    for p in proposals:
        assert p["implementation_strategy"] == "from_scratch"
        assert p["reference_repo"] == ""
        assert p["reference_files"] == []


# --- clone_reference_repo ---

def test_clone_reference_repo_invalid_url():
    result = clone_reference_repo("https://bitbucket.org/user/repo", "/tmp/test-clone")
    assert result["success"] is False
    assert "URL must be" in result["error"]


# --- analyze_reference_structure ---

def test_analyze_reference_structure(tmp_path):
    # Create a mock repo directory
    (tmp_path / "model.py").write_text("import torch\nfrom torch import nn\nclass MyModel(nn.Module):\n    pass\n")
    (tmp_path / "train.py").write_text("import torch\nfor epoch in range(10):\n    pass\n")
    (tmp_path / "utils.py").write_text("import os\ndef helper(): pass\n")
    (tmp_path / "test_model.py").write_text("# should be skipped\n")
    (tmp_path / "requirements.txt").write_text("torch>=2.0\nnumpy\n")
    (tmp_path / "README.md").write_text("# My Model\nA great model for doing things.\n")

    result = analyze_reference_structure(str(tmp_path))
    assert result["framework"] == "pytorch"
    assert "model.py" in result["python_files"]
    assert "train.py" in result["python_files"]
    assert "utils.py" in result["python_files"]
    assert "test_model.py" not in result["python_files"]
    assert "model.py" in result["model_files"]
    assert "train.py" in result["training_files"]
    assert "torch>=2.0" in result["requirements"]
    assert "numpy" in result["requirements"]
    assert "My Model" in result["readme_summary"]


# --- cleanup_reference_repo ---

def test_cleanup_reference_repo(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "file.py").write_text("x = 1\n")
    assert cleanup_reference_repo(str(repo_dir)) is True
    assert not repo_dir.exists()


def test_cleanup_reference_repo_nonexistent(tmp_path):
    assert cleanup_reference_repo(str(tmp_path / "nonexistent")) is False
