"""Consolidated tests for implement_utils.py and prerequisites_check.py."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
from prerequisites_check import (
    scan_imports,
    check_missing_packages,
    detect_env_manager,
    detect_dataset_format,
    detect_dataset_format_project,
    validate_data_path,
    pip_name,
    gpu_install_command,
    bulk_install_command,
    _best_torch_cuda_tag,
    _detect_cuda_version,
    _wrap_for_conda,
    IMPORT_TO_PACKAGE,
)

SAMPLE_FINDINGS = FIXTURES / "sample_research_findings.md"
SAMPLE_FINDINGS_REF = FIXTURES / "sample_research_findings_with_reference.md"
SAMPLE_FINDINGS_KNOWLEDGE = FIXTURES / "sample_research_findings_knowledge.md"
TINY_RESNET = FIXTURES / "tiny_resnet_cifar10"


# =========================================================================
# TestProposalParsing — parsing research findings into proposals
# =========================================================================

class TestProposalParsing:
    def test_parse_all_proposals(self):
        proposals = parse_research_proposals(str(SAMPLE_FINDINGS))
        assert len(proposals) == 3
        assert proposals[0]["name"] == "Perceptual Loss Function"
        assert proposals[0]["slug"] == "perceptual-loss-function"
        assert proposals[0]["complexity"] == "Low"
        assert len(proposals[0]["files_to_modify"]) == 2
        assert "models/classifier.py" in proposals[0]["files_to_modify"]
        assert len(proposals[0]["implementation_steps"]) == 3

    def test_parse_selected_subset(self):
        proposals = parse_research_proposals(str(SAMPLE_FINDINGS), selected_indices=[1, 3])
        assert len(proposals) == 2
        assert proposals[0]["index"] == 1
        assert proposals[1]["index"] == 3
        assert proposals[1]["name"] == "CutMix Data Augmentation"

    def test_detect_conflicts(self):
        proposals = parse_research_proposals(str(SAMPLE_FINDINGS))
        conflicts = detect_conflicts(proposals)
        assert len(conflicts) == 1
        assert conflicts[0]["file"] == "models/classifier.py"
        assert 1 in conflicts[0]["proposal_indices"]
        assert 2 in conflicts[0]["proposal_indices"]

    def test_detect_no_conflicts(self):
        proposals = parse_research_proposals(str(SAMPLE_FINDINGS), selected_indices=[1, 3])
        assert len(detect_conflicts(proposals)) == 0

    def test_backward_compat_defaults(self):
        """Old fixture without strategy fields gets safe defaults."""
        proposals = parse_research_proposals(str(SAMPLE_FINDINGS))
        for p in proposals:
            assert p["implementation_strategy"] == "from_scratch"
            assert p["reference_repo"] == ""
            assert p["reference_files"] == []
            assert p["proposal_source"] == "paper"

    def test_knowledge_source(self):
        """Knowledge-mode proposals have proposal_source='llm_knowledge'."""
        proposals = parse_research_proposals(str(SAMPLE_FINDINGS_KNOWLEDGE))
        assert len(proposals) == 3
        for p in proposals:
            assert p["proposal_source"] == "llm_knowledge"
            assert p["implementation_strategy"] == "from_scratch"

    def test_double_hash_headers(self, tmp_path):
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

    def test_extract_files_section_break(self):
        """_extract_files stops at ### section boundary."""
        body = """**What to change:**
- `models/classifier.py` — modify forward pass
### Next Section
- `should/not/appear.py`
"""
        files = _extract_files(body)
        assert "models/classifier.py" in files
        assert "should/not/appear.py" not in files

    def test_extract_files_stops_at_field_header(self):
        """_extract_files stops at '- **FieldName:**' headers."""
        body = """**What to change:**
- `models/net.py` — modify architecture
- **Expected improvement:** +1 dB
- **Reference files:** `external/model.py`
"""
        assert _extract_files(body) == ["models/net.py"]

    def test_extract_steps_section_break(self):
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

    def test_write_manifest(self, tmp_path):
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


# =========================================================================
# TestGitStrategy — branch creation, slugify, is_git_repo, get_current_branch
# =========================================================================

class TestGitStrategy:
    @pytest.mark.parametrize("input_name,expected_slug", [
        ("Perceptual Loss Function", "perceptual-loss-function"),
        ("CutMix (Data Aug.)", "cutmix-data-aug"),
        ("  SWIN   Transformer  ", "swin-transformer"),
        ("", "proposal"),
        ("!!!", "proposal"),
        ("   ", "proposal"),
    ])
    def test_slugify(self, input_name, expected_slug):
        assert slugify(input_name) == expected_slug

    def test_is_git_repo_true(self, tmp_path):
        (tmp_path / ".git").mkdir()
        assert is_git_repo(str(tmp_path)) is True

    def test_is_git_repo_false(self, tmp_path):
        assert is_git_repo(str(tmp_path)) is False

    def test_get_current_branch(self):
        project_root = str(Path(__file__).parent.parent)
        branch = get_current_branch(project_root)
        assert isinstance(branch, str)
        assert len(branch) > 0

    def test_create_proposal_branch(self, tmp_path):
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


# =========================================================================
# TestFileStrategy — backup/restore files
# =========================================================================

class TestFileStrategy:
    def test_backup_files(self, tmp_path):
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

    def test_backup_same_name_different_dirs(self, tmp_path):
        project = tmp_path / "project"
        (project / "models").mkdir(parents=True)
        (project / "data").mkdir(parents=True)
        (project / "models" / "utils.py").write_text("# models\n")
        (project / "data" / "utils.py").write_text("# data\n")
        backup_dir = tmp_path / "backups"
        mapping = backup_files(
            [str(project / "models" / "utils.py"), str(project / "data" / "utils.py")],
            str(backup_dir), project_root=str(project),
        )
        assert len(mapping) == 2
        assert Path(mapping[str(project / "models" / "utils.py")]) == backup_dir / "models" / "utils.py"
        assert Path(mapping[str(project / "data" / "utils.py")]) == backup_dir / "data" / "utils.py"


# =========================================================================
# TestValidation — syntax validation, import validation
# =========================================================================

class TestValidation:
    def test_validate_syntax_pass(self, tmp_path):
        good_file = tmp_path / "good.py"
        good_file.write_text("x = 1 + 2\n")
        results = validate_syntax([str(good_file)])
        assert results[0]["passed"] is True
        assert results[0]["error"] is None

    def test_validate_syntax_fail(self, tmp_path):
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def foo(\n")
        results = validate_syntax([str(bad_file)])
        assert results[0]["passed"] is False
        assert results[0]["error"] is not None

    def test_validate_imports_pass_and_fail(self, tmp_path):
        """Test both passing and failing import validation."""
        good = tmp_path / "good.py"
        good.write_text("x = 1\n")
        assert validate_imports(str(good), str(tmp_path))["passed"] is True

        bad = tmp_path / "bad.py"
        bad.write_text("import nonexistent_module_xyz_123\n")
        result = validate_imports(str(bad), str(tmp_path))
        assert result["passed"] is False
        assert result["error"] is not None


# =========================================================================
# TestReferenceRepos — clone, analyze, cleanup reference repos
# =========================================================================

class TestReferenceRepos:
    @pytest.mark.parametrize("body,expected", [
        ("- **Reference files:** `basicsr/models/restormer_arch.py`, `basicsr/losses/loss.py`",
         ["basicsr/models/restormer_arch.py", "basicsr/losses/loss.py"]),
        ("- **Complexity:** Low\n- **Risk:** None", []),
    ])
    def test_extract_reference_files(self, body, expected):
        assert _extract_reference_files(body) == expected

    def test_parse_proposals_with_reference_repo(self):
        proposals = parse_research_proposals(str(SAMPLE_FINDINGS_REF))
        p1 = proposals[0]
        assert p1["implementation_strategy"] == "from_reference"
        assert "github.com/swz30/Restormer" in p1["reference_repo"]
        assert len(p1["reference_files"]) == 2
        p2 = proposals[1]
        assert p2["implementation_strategy"] == "from_scratch"
        assert p2["reference_repo"] == ""

    def test_clone_success_and_failure(self):
        """Test clone with valid URL (mocked success) and invalid URL."""
        # invalid URL
        result = clone_reference_repo("https://bitbucket.org/user/repo", "/tmp/test-clone")
        assert result["success"] is False
        assert "URL must be" in result["error"]
        # mocked success
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("implement_utils.subprocess.run", return_value=mock_result):
            result = clone_reference_repo("https://github.com/user/repo", "/tmp/dest")
        assert result["success"] is True

    def test_clone_error_paths(self):
        """Clone failure, timeout, and generic exception."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "fatal: not found"
        with patch("implement_utils.subprocess.run", return_value=mock_result):
            assert clone_reference_repo("https://github.com/user/repo", "/tmp/dest")["success"] is False
        with patch("implement_utils.subprocess.run", side_effect=subprocess.TimeoutExpired("git", 120)):
            assert clone_reference_repo("https://github.com/user/repo", "/tmp/dest")["success"] is False
        with patch("implement_utils.subprocess.run", side_effect=OSError("denied")):
            assert clone_reference_repo("https://github.com/user/repo", "/tmp/dest")["success"] is False

    def test_analyze_full_structure(self, tmp_path):
        (tmp_path / "model.py").write_text("import torch\nfrom torch import nn\nclass MyModel(nn.Module):\n    pass\n")
        (tmp_path / "train.py").write_text("import torch\nfor epoch in range(10):\n    pass\n")
        (tmp_path / "test_model.py").write_text("# should be skipped\n")
        (tmp_path / "requirements.txt").write_text("torch>=2.0\nnumpy\n")
        (tmp_path / "README.md").write_text("# My Model\nA great model.\n")
        result = analyze_reference_structure(str(tmp_path))
        assert result["framework"] == "pytorch"
        assert "model.py" in result["python_files"]
        assert "test_model.py" not in result["python_files"]
        assert "model.py" in result["model_files"]
        assert "torch>=2.0" in result["requirements"]
        assert "My Model" in result["readme_summary"]

    def test_analyze_unreadable_graceful(self, tmp_path):
        """Unreadable file, requirements, and readme are handled gracefully."""
        py_file = tmp_path / "model.py"
        py_file.write_text("import torch")
        req = tmp_path / "requirements.txt"
        req.write_text("torch>=2.0\n")
        readme = tmp_path / "README.md"
        readme.write_text("# Proj\n")
        os.chmod(str(py_file), 0o000)
        os.chmod(str(req), 0o000)
        os.chmod(str(readme), 0o000)
        try:
            result = analyze_reference_structure(str(tmp_path))
            assert "model.py" in result["python_files"]
            assert result["requirements"] == []
            assert result["readme_summary"] == ""
        finally:
            for f in [py_file, req, readme]:
                os.chmod(str(f), 0o644)

    def test_analyze_skips_docs_and_test_dirs(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "guide.py").write_text("x = 1\n")
        (tmp_path / "model.py").write_text("x = 1\n")
        result = analyze_reference_structure(str(tmp_path))
        assert "model.py" in result["python_files"]
        assert "docs/guide.py" not in result["python_files"]

    def test_cleanup(self, tmp_path):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / "file.py").write_text("x = 1\n")
        assert cleanup_reference_repo(str(repo_dir)) is True
        assert not repo_dir.exists()
        assert cleanup_reference_repo(str(tmp_path / "nonexistent")) is False


# =========================================================================
# TestBranchDiff — extract_branch_diff
# =========================================================================

class TestBranchDiff:
    def test_not_git_repo(self, tmp_path):
        result = extract_branch_diff(str(tmp_path), "some-branch")
        assert result["error"] == "Not a git repo"
        assert result["files_changed"] == 0

    def test_basic_diff(self, tmp_path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"],
                       cwd=str(tmp_path), capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"],
                       cwd=str(tmp_path), capture_output=True)
        (tmp_path / "train.py").write_text("def train():\n    pass\n")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"],
                       cwd=str(tmp_path), capture_output=True)
        subprocess.run(["git", "checkout", "-b", "ml-opt/test"],
                       cwd=str(tmp_path), capture_output=True)
        (tmp_path / "train.py").write_text("def train():\n    print('improved')\n")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(["git", "commit", "-m", "improve"],
                       cwd=str(tmp_path), capture_output=True)
        subprocess.run(["git", "checkout", "main"],
                       cwd=str(tmp_path), capture_output=True, check=False)
        subprocess.run(["git", "checkout", "master"],
                       cwd=str(tmp_path), capture_output=True, check=False)
        result = extract_branch_diff(str(tmp_path), "ml-opt/test")
        assert result["files_changed"] >= 1
        assert result["lines_added"] >= 1


# =========================================================================
# TestPrerequisites — scan imports, check packages, detect env, detect
# format, validate data, bulk install, GPU install commands, CLI
# =========================================================================

class TestPrerequisites:
    # --- scan_imports (2 tests) ---

    def test_scan_imports_tiny_resnet(self):
        result = scan_imports(str(TINY_RESNET))
        assert "torch" in result["third_party"]
        assert "torchvision" in result["third_party"]
        assert "sys" in result["stdlib"]
        assert "model" in result["local"]

    def test_scan_imports_edge_cases(self, tmp_path):
        """Syntax errors skipped, relative imports ignored, __pycache__ excluded,
        conditional imports found, exclude_dirs honored, empty dir."""
        (tmp_path / "bad.py").write_text("def foo(:\n  pass\n")
        (tmp_path / "good.py").write_text("import os\n")
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "sub.py").write_text("from . import utils\n")
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "mod.py").write_text("import secretlib\n")
        (tmp_path / "cond.py").write_text(
            "try:\n    import yaml\nexcept ImportError:\n    yaml = None\n"
        )
        excluded = tmp_path / "vendor"
        excluded.mkdir()
        (excluded / "lib.py").write_text("import numpy\n")

        result = scan_imports(str(tmp_path), exclude_dirs=["vendor"])
        assert "os" in result["stdlib"]
        assert "yaml" in result["third_party"]
        assert "utils" not in result["third_party"]
        assert "secretlib" not in result["third_party"]
        assert "numpy" not in result["third_party"]

        # empty dir
        empty = tmp_path / "empty"
        empty.mkdir()
        r2 = scan_imports(str(empty))
        assert r2["third_party"] == []

    # --- check_missing_packages (2 tests) ---

    def test_check_packages(self):
        result = check_missing_packages(["os", "_nonexistent_abc_999"])
        assert "os" in result["installed"]
        assert "_nonexistent_abc_999" in result["missing"]
        # empty list
        assert check_missing_packages([])["installed"] == []

    def test_check_packages_error_paths(self):
        r = check_missing_packages(["os"], python_executable="/nonexistent/python")
        assert r["missing"] == ["os"]
        with patch("prerequisites_check.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("python3", 30)):
            r2 = check_missing_packages(["slow"])
            assert "slow" in r2["missing"]

    # --- pip_name (5 representative cases) ---

    @pytest.mark.parametrize("import_name,expected_pip", [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("PIL", "Pillow"),
        ("cv2", "opencv-python"),
        ("sklearn", "scikit-learn"),
    ])
    def test_pip_name(self, import_name, expected_pip):
        assert pip_name(import_name) == expected_pip

    # --- detect_env_manager (3 single-file + 1 priority) ---

    @pytest.mark.parametrize("filename,content,expected_manager", [
        ("environment.yml", "name: myenv\n", "conda"),
        ("requirements.txt", "torch\nnumpy\n", "pip"),
        ("uv.lock", "", "uv"),
    ])
    def test_env_detection_single_file(self, tmp_path, filename, content, expected_manager):
        (tmp_path / filename).write_text(content)
        assert detect_env_manager(str(tmp_path))["manager"] == expected_manager

    def test_env_priority_and_empty(self, tmp_path):
        # conda > pip
        d1 = tmp_path / "d1"
        d1.mkdir()
        (d1 / "environment.yml").write_text("name: env\n")
        (d1 / "requirements.txt").write_text("torch\n")
        assert detect_env_manager(str(d1))["manager"] == "conda"
        # empty -> unknown
        d2 = tmp_path / "d2"
        d2.mkdir()
        assert detect_env_manager(str(d2))["manager"] == "unknown"

    # --- detect_dataset_format (3 fixture + 1 dynamic + 1 error) ---

    @pytest.mark.parametrize("fixture_path,expected_format", [
        (TINY_RESNET / "train.py", "cifar"),
        (FIXTURES / "sample_imageloader_script.py", "image_folder"),
        (FIXTURES / "sample_csv_loader_script.py", "csv"),
    ])
    def test_format_detection_fixtures(self, fixture_path, expected_format):
        assert detect_dataset_format(str(fixture_path))["format"] == expected_format

    def test_format_detection_dynamic_and_error(self, tmp_path):
        script = tmp_path / "loader.py"
        script.write_text("import xgboost as xgb\ndtrain = xgb.DMatrix(data, label=labels)\n")
        assert detect_dataset_format(str(script))["format"] == "xgboost"
        assert detect_dataset_format("/nonexistent/script.py")["format"] == "unknown"

    # --- detect_dataset_format_project ---

    def test_project_format_follows_imports(self, tmp_path):
        (tmp_path / "train.py").write_text(
            "import torch\nfrom data import get_loader\nloader = get_loader()\n"
        )
        (tmp_path / "data.py").write_text(
            "from torchvision.datasets import ImageFolder\n"
            "from torch.utils.data import DataLoader\n"
            "def get_loader():\n"
            "    dataset = ImageFolder(root='./data/train')\n"
            "    return DataLoader(dataset)\n"
        )
        result = detect_dataset_format_project(str(tmp_path), str(tmp_path / "train.py"))
        assert result["format"] == "image_folder"

    # --- validate_data_path (3 cases) ---

    def test_validate_nonexistent(self):
        result = validate_data_path("/nonexistent/data", "image_folder")
        assert result["exists"] is False

    def test_validate_image_folder(self, tmp_path):
        root = tmp_path / "dataset"
        root.mkdir()
        for cls_name in ["cat", "dog"]:
            cls_dir = root / cls_name
            cls_dir.mkdir()
            (cls_dir / "img1.jpg").write_bytes(b"\xff\xd8\xff")
        result = validate_data_path(str(root), "image_folder")
        assert result["format_matches"] is True
        assert result["details"]["class_dirs"] == 2

    def test_validate_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3\n")
        assert validate_data_path(str(f), "csv")["format_matches"] is True

    # --- GPU install (2 cases) ---

    def test_gpu_install_with_cuda(self, monkeypatch):
        monkeypatch.setattr("prerequisites_check._detect_cuda_version", lambda: "12.1")
        result = gpu_install_command("torch")
        assert result["gpu_detected"] is True
        assert "cu121" in result["install_command"]

    def test_gpu_install_no_cuda(self, monkeypatch):
        monkeypatch.setattr("prerequisites_check._detect_cuda_version", lambda: None)
        assert gpu_install_command("numpy")["install_command"] == "pip install numpy"

    # --- _best_torch_cuda_tag (3 cases) ---

    @pytest.mark.parametrize("version,expected_tag", [
        ("12.4", "12.4"),
        ("11.7", None),
        (None, None),
    ])
    def test_best_torch_cuda_tag(self, version, expected_tag):
        assert _best_torch_cuda_tag(version) == expected_tag

    # --- _detect_cuda_version (2 cases) ---

    def test_detect_cuda_parses_nvidia_smi(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "| NVIDIA-SMI 535   CUDA Version: 12.4 |\n"
        with patch("prerequisites_check.subprocess.run", return_value=mock_result):
            assert _detect_cuda_version() == "12.4"

    def test_detect_cuda_returns_none_on_error(self):
        with patch("prerequisites_check.subprocess.run", side_effect=FileNotFoundError):
            assert _detect_cuda_version() is None

    # --- _wrap_for_conda (2 cases) ---

    def test_wrap_for_conda(self):
        assert _wrap_for_conda("pip install torch", "conda", "myenv") == \
               "conda run --no-banner -n myenv pip install torch"
        assert _wrap_for_conda("pip install torch", "pip", None) == "pip install torch"

    # --- bulk_install_command (2 cases) ---

    @pytest.mark.parametrize("filename,content,manager,expected_in_cmd", [
        ("requirements.txt", "torch\nnumpy\n", "pip", "pip install -r"),
        ("environment.yml", "name: myenv\ndependencies:\n  - numpy\n", "conda", "conda"),
    ])
    def test_bulk_install(self, tmp_path, filename, content, manager, expected_in_cmd):
        (tmp_path / filename).write_text(content)
        result = bulk_install_command(str(tmp_path), manager)
        assert result["has_deps_file"] is True
        assert expected_in_cmd in result["install_command"]

    def test_bulk_install_no_deps(self, tmp_path):
        assert bulk_install_command(str(tmp_path), "pip")["install_command"] is None

    # --- CLI tests (3 error + 3 success) ---

    @pytest.mark.parametrize("args", [[], ["bogus"], ["scan-imports"]])
    def test_cli_error_cases(self, run_main, args):
        assert run_main("prerequisites_check.py", *args).returncode == 1

    def test_cli_scan_imports(self, run_main):
        r = run_main("prerequisites_check.py", "scan-imports", str(TINY_RESNET))
        assert r.returncode == 0
        assert "torch" in json.loads(r.stdout)["third_party"]

    def test_cli_detect_format(self, run_main):
        r = run_main("prerequisites_check.py", "detect-format", str(TINY_RESNET / "train.py"))
        assert json.loads(r.stdout)["format"] == "cifar"

    def test_cli_gpu_install_cmd(self, run_main):
        r = run_main("prerequisites_check.py", "gpu-install-cmd", "numpy")
        assert "pip install numpy" in json.loads(r.stdout)["install_command"]

    # --- implement_utils CLI tests (3 error + 2 success) ---

    @pytest.mark.parametrize("args", [[], ["analyze"], ["clone"]])
    def test_implement_cli_error_cases(self, run_main, args):
        r = run_main("implement_utils.py", *args)
        assert r.returncode == 1
        assert "Usage" in r.stdout

    def test_implement_cli_parse_proposals(self, run_main):
        r = run_main("implement_utils.py", str(SAMPLE_FINDINGS), '[1,3]')
        assert len(json.loads(r.stdout)["proposals"]) == 2

    def test_implement_cli_analyze(self, run_main, tmp_path):
        (tmp_path / "model.py").write_text("import torch\n")
        r = run_main("implement_utils.py", "analyze", str(tmp_path))
        assert "python_files" in json.loads(r.stdout)
