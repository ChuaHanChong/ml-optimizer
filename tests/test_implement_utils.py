"""Tests for implement_utils.py."""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from conftest import FIXTURES
from implement_utils import (
    _extract_files,
    _extract_reference_files,
    _extract_steps,
    analyze_reference_structure,
    backup_files,
    cleanup_reference_repo,
    clone_reference_repo,
    create_proposal_branch,
    detect_conflicts,
    extract_branch_diff,
    get_current_branch,
    is_git_repo,
    parse_research_proposals,
    slugify,
    validate_imports,
    validate_syntax,
    write_manifest,
)
SAMPLE_FINDINGS = FIXTURES / "sample_research_findings.md"
SAMPLE_FINDINGS_REF = FIXTURES / "sample_research_findings_with_reference.md"
SAMPLE_FINDINGS_KNOWLEDGE = FIXTURES / "sample_research_findings_knowledge.md"


# --- slugify ---

def test_slugify_basic():
    assert slugify("Perceptual Loss Function") == "perceptual-loss-function"


def test_slugify_special_chars():
    assert slugify("CutMix (Data Aug.)") == "cutmix-data-aug"


def test_slugify_uppercase_and_extra_spaces():
    assert slugify("  SWIN   Transformer  ") == "swin-transformer"


def test_slugify_empty_input():
    """Empty string, all-punctuation, and whitespace-only input produce 'proposal'."""
    assert slugify("") == "proposal"
    assert slugify("!!!") == "proposal"
    assert slugify("   ") == "proposal"


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
    # files_to_modify must NOT include reference files
    assert len(p1["files_to_modify"]) == 2
    assert "basicsr/models/archs/restormer_arch.py" not in p1["files_to_modify"]
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


# --- create_proposal_branch ---


def test_create_proposal_branch(tmp_path):
    """Create a proposal branch in a real git repo."""
    subprocess.run(["git", "init", "-b", "main", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@test.com"], capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], capture_output=True)
    (tmp_path / "file.txt").write_text("hello")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True)
    branch = create_proposal_branch(str(tmp_path), "test-feature", "main")
    assert branch == "ml-opt/test-feature"
    result = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True,
    )
    assert result.stdout.strip() == "ml-opt/test-feature"


# --- validate_imports failure ---


def test_validate_imports_failure(tmp_path):
    """Importing a module that fails should return passed=False."""
    bad_mod = tmp_path / "bad_mod.py"
    bad_mod.write_text("import nonexistent_module_xyz_123\n")
    result = validate_imports(str(bad_mod), str(tmp_path))
    assert result["passed"] is False
    assert result["error"] is not None


def test_validate_imports_syntax_error(tmp_path):
    """Module with syntax error returns passed=False."""
    (tmp_path / "syntax_err.py").write_text("def f(\n")
    result = validate_imports(str(tmp_path / "syntax_err.py"), str(tmp_path))
    assert result["passed"] is False
    assert result["error"] is not None


# --- _extract_files section break ---


def test_extract_files_section_break():
    """_extract_files stops at ### section boundary."""
    body = """**What to change:**
- `models/classifier.py` — modify forward pass
### Next Section
- `should/not/appear.py`
"""
    files = _extract_files(body)
    assert "models/classifier.py" in files
    assert "should/not/appear.py" not in files


def test_extract_files_stops_at_field_header():
    """_extract_files stops at '- **FieldName:**' headers."""
    body = """**What to change:**
- `models/net.py` — modify architecture
- **Expected improvement:** +1 dB
- **Reference files:** `external/model.py`
"""
    files = _extract_files(body)
    assert files == ["models/net.py"]


# --- _extract_steps section break ---


def test_extract_steps_section_break():
    """_extract_steps stops at ** section boundary."""
    body = """**Implementation steps:**
1. First step
2. Second step
**Risk:**
3. Not a step
"""
    steps = _extract_steps(body)
    assert len(steps) == 2
    assert steps[0] == "First step"


# --- clone_reference_repo ---


def test_clone_reference_repo_success():
    """Successful clone returns success."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("implement_utils.subprocess.run", return_value=mock_result):
        result = clone_reference_repo("https://github.com/user/repo", "/tmp/dest")
    assert result["success"] is True
    assert result["path"] == "/tmp/dest"


def test_clone_reference_repo_failure():
    """Clone returning nonzero exit code returns error."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "fatal: not found"
    with patch("implement_utils.subprocess.run", return_value=mock_result):
        result = clone_reference_repo("https://github.com/user/repo", "/tmp/dest")
    assert result["success"] is False
    assert "not found" in result["error"]


def test_clone_reference_repo_timeout():
    """Clone timeout returns error."""
    with patch("implement_utils.subprocess.run", side_effect=subprocess.TimeoutExpired("git", 120)):
        result = clone_reference_repo("https://github.com/user/repo", "/tmp/dest")
    assert result["success"] is False
    assert "timed out" in result["error"].lower()


# --- analyze_reference_structure edge cases ---


def test_analyze_no_requirements(tmp_path):
    """Repo without requirements.txt returns empty requirements."""
    (tmp_path / "model.py").write_text("import torch\nclass M(torch.nn.Module): pass\n")
    result = analyze_reference_structure(str(tmp_path))
    assert result["requirements"] == []


def test_analyze_reference_sklearn_framework(tmp_path):
    """sklearn framework is detected in reference structure analysis."""
    (tmp_path / "model.py").write_text(
        "from sklearn.ensemble import RandomForestClassifier\n"
        "class MyModel(RandomForestClassifier): pass\n"
    )
    result = analyze_reference_structure(str(tmp_path))
    assert result["framework"] == "sklearn"


def test_analyze_reference_xgboost_framework(tmp_path):
    """XGBoost framework is detected in reference structure analysis."""
    (tmp_path / "train.py").write_text(
        "import xgboost as xgb\n"
        "model = xgb.XGBClassifier()\n"
    )
    result = analyze_reference_structure(str(tmp_path))
    assert result["framework"] == "xgboost"


def test_analyze_readme_priority(tmp_path):
    """README.rst is found when README.md doesn't exist."""
    (tmp_path / "model.py").write_text("x = 1\n")
    (tmp_path / "README.rst").write_text("My RST Readme\n==============\n")
    result = analyze_reference_structure(str(tmp_path))
    assert "My RST Readme" in result["readme_summary"]


# --- CLI tests ---


def test_cli_no_args(run_main):
    """CLI with no args prints usage and exits 1."""
    r = run_main("implement_utils.py")
    assert r.returncode == 1
    assert "Usage" in r.stdout


def test_cli_parse_proposals(run_main):
    """CLI parses proposals from fixture file."""
    r = run_main("implement_utils.py", str(SAMPLE_FINDINGS), '[1,3]')
    assert r.returncode == 0
    output = json.loads(r.stdout)
    assert "proposals" in output
    assert len(output["proposals"]) == 2


def test_cli_invalid_selected_json(run_main):
    """CLI with invalid selected JSON exits cleanly."""
    r = run_main("implement_utils.py", "/dev/null", "{bad")
    assert r.returncode == 1
    assert "Error" in r.stdout


def test_cli_analyze(run_main, tmp_path):
    """CLI analyze subcommand works."""
    (tmp_path / "model.py").write_text("import torch\n")
    r = run_main("implement_utils.py", "analyze", str(tmp_path))
    assert r.returncode == 0
    output = json.loads(r.stdout)
    assert "python_files" in output


def test_cli_clone_missing_args(run_main):
    """CLI clone subcommand with missing args prints usage and exits 1."""
    r = run_main("implement_utils.py", "clone")
    assert r.returncode == 1
    assert "Usage" in r.stdout


def test_parse_proposals_double_hash(tmp_path):
    """Proposals with ## headers (instead of ###) should parse correctly."""
    content = '''# Research Findings

## Proposal 1: Test Technique (Priority: High)

**Complexity:** Low
**Implementation strategy:** from_scratch

**What to change:**
- `train.py` — modify training loop

**Implementation steps:**
1. Do the thing
'''
    f = tmp_path / "findings.md"
    f.write_text(content)
    proposals = parse_research_proposals(str(f))
    assert len(proposals) == 1
    assert proposals[0]["name"] == "Test Technique"


# --- proposal_source extraction ---


def test_parse_proposals_knowledge_source():
    """Knowledge-mode proposals have proposal_source='llm_knowledge'."""
    proposals = parse_research_proposals(str(SAMPLE_FINDINGS_KNOWLEDGE))
    assert len(proposals) == 3
    for p in proposals:
        assert p["proposal_source"] == "llm_knowledge"
        assert p["implementation_strategy"] == "from_scratch"


def test_parse_proposals_paper_source_default():
    """Old findings without proposal_source default to 'paper'."""
    proposals = parse_research_proposals(str(SAMPLE_FINDINGS))
    for p in proposals:
        assert p["proposal_source"] == "paper"


def test_cli_parse_knowledge_proposals(run_main):
    """CLI parses knowledge-mode findings and includes proposal_source."""
    r = run_main("implement_utils.py", str(SAMPLE_FINDINGS_KNOWLEDGE), '[1,2]')
    assert r.returncode == 0
    output = json.loads(r.stdout)
    assert len(output["proposals"]) == 2
    for p in output["proposals"]:
        assert p["proposal_source"] == "llm_knowledge"


# --- CLI missing-args edge cases ---


def test_cli_analyze_missing_repo_path(run_main):
    """CLI analyze subcommand with no repo_path prints usage and exits 1."""
    r = run_main("implement_utils.py", "analyze")
    assert r.returncode == 1
    assert "Usage" in r.stdout


def test_cli_default_parse_missing_selected(run_main):
    """CLI default mode with only findings path (no selected_json) exits 1."""
    r = run_main("implement_utils.py", str(SAMPLE_FINDINGS))
    assert r.returncode == 1
    assert "Usage" in r.stdout


def test_cli_clone_invalid_url(run_main, tmp_path):
    """CLI clone subcommand with non-GitHub/GitLab URL exits 1 with error JSON."""
    dest = tmp_path / "dest_repo"
    r = run_main("implement_utils.py", "clone", "https://example.com/repo", str(dest))
    assert r.returncode == 1
    output = json.loads(r.stdout)
    assert output["success"] is False
    assert "github.com" in output["error"]


# --- clone_reference_repo generic exception ---


def test_clone_reference_repo_generic_exception():
    """Generic exception in clone returns error dict (lines 251-252)."""
    with patch("implement_utils.subprocess.run",
               side_effect=OSError("permission denied")):
        result = clone_reference_repo("https://github.com/user/repo", "/tmp/dest")
    assert result["success"] is False
    assert "permission denied" in result["error"]


# --- analyze_reference_structure OS-error edge cases ---


def test_analyze_unreadable_file(tmp_path):
    """analyze_reference_structure skips files that can't be read (lines 303-304)."""
    py_file = tmp_path / "model.py"
    py_file.write_text("import torch")
    os.chmod(str(py_file), 0o000)
    try:
        result = analyze_reference_structure(str(tmp_path))
        assert isinstance(result, dict)
        # File is listed (os.walk finds it) but content can't be analyzed
        assert "model.py" in result["python_files"]
    finally:
        os.chmod(str(py_file), 0o644)


def test_analyze_unreadable_requirements(tmp_path):
    """analyze_reference_structure handles unreadable requirements.txt (lines 335-336)."""
    (tmp_path / "model.py").write_text("x = 1\n")
    req = tmp_path / "requirements.txt"
    req.write_text("torch>=2.0\n")
    os.chmod(str(req), 0o000)
    try:
        result = analyze_reference_structure(str(tmp_path))
        assert result["requirements"] == []
    finally:
        os.chmod(str(req), 0o644)


def test_analyze_unreadable_readme(tmp_path):
    """analyze_reference_structure handles unreadable README.md (lines 345-346)."""
    (tmp_path / "model.py").write_text("x = 1\n")
    readme = tmp_path / "README.md"
    readme.write_text("# My Project\n")
    os.chmod(str(readme), 0o000)
    try:
        result = analyze_reference_structure(str(tmp_path))
        assert result["readme_summary"] == ""
    finally:
        os.chmod(str(readme), 0o644)


def test_analyze_skips_docs_and_test_dirs(tmp_path):
    """analyze_reference_structure skips files in docs/ and test dirs (line 297)."""
    # Create files in various skip directories
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.py").write_text("x = 1\n")

    tests_dir = tmp_path / "src" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "check.py").write_text("x = 1\n")

    test_dir = tmp_path / "src" / "test"
    test_dir.mkdir(parents=True)
    (test_dir / "verify.py").write_text("x = 1\n")

    # Create a normal file that should be included
    (tmp_path / "model.py").write_text("x = 1\n")

    result = analyze_reference_structure(str(tmp_path))
    assert "model.py" in result["python_files"]
    assert "docs/guide.py" not in result["python_files"]
    assert "src/tests/check.py" not in result["python_files"]
    assert "src/test/verify.py" not in result["python_files"]


# ---------------------------------------------------------------------------
# extract_branch_diff
# ---------------------------------------------------------------------------


def test_extract_branch_diff_not_git(tmp_path):
    """extract_branch_diff returns error dict for non-git directory."""
    result = extract_branch_diff(str(tmp_path), "some-branch")
    assert result["error"] == "Not a git repo"
    assert result["files_changed"] == 0
    assert result["diff_summary"] == ""


def test_extract_branch_diff_basic(tmp_path):
    """extract_branch_diff returns structured diff data for a git repo."""
    # Set up a real git repo with a branch
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"],
                   cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"],
                   cwd=str(tmp_path), capture_output=True)

    # Create initial file and commit
    (tmp_path / "train.py").write_text("def train():\n    pass\n")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"],
                   cwd=str(tmp_path), capture_output=True)

    # Create a branch with changes
    subprocess.run(["git", "checkout", "-b", "ml-opt/test"],
                   cwd=str(tmp_path), capture_output=True)
    (tmp_path / "train.py").write_text("def train():\n    print('improved')\n")
    (tmp_path / "loss.py").write_text("def focal_loss():\n    pass\n")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "commit", "-m", "add focal loss"],
                   cwd=str(tmp_path), capture_output=True)

    # Go back to main
    subprocess.run(["git", "checkout", "main"],
                   cwd=str(tmp_path), capture_output=True, check=False)
    # Try master if main doesn't exist
    subprocess.run(["git", "checkout", "master"],
                   cwd=str(tmp_path), capture_output=True, check=False)

    result = extract_branch_diff(str(tmp_path), "ml-opt/test")
    assert result["files_changed"] >= 1
    assert result["lines_added"] >= 1
    assert isinstance(result["changed_functions"], list)
    assert isinstance(result["diff_summary"], str)
