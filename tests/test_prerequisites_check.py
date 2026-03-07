"""Tests for scripts/prerequisites_check.py."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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

FIXTURES = Path(__file__).parent / "fixtures"
TINY_RESNET = FIXTURES / "tiny_resnet_cifar10"


# =========================================================================
# scan_imports
# =========================================================================

class TestScanImports:
    def test_tiny_resnet_detects_torch(self):
        result = scan_imports(str(TINY_RESNET))
        assert "torch" in result["third_party"]

    def test_tiny_resnet_detects_torchvision(self):
        result = scan_imports(str(TINY_RESNET))
        assert "torchvision" in result["third_party"]

    def test_tiny_resnet_detects_stdlib(self):
        result = scan_imports(str(TINY_RESNET))
        assert "sys" in result["stdlib"]
        assert "json" in result["stdlib"]

    def test_tiny_resnet_detects_local_model(self):
        result = scan_imports(str(TINY_RESNET))
        assert "model" in result["local"]

    def test_empty_dir(self, tmp_path):
        result = scan_imports(str(tmp_path))
        assert result["stdlib"] == []
        assert result["third_party"] == []
        assert result["local"] == []

    def test_syntax_error_file_skipped(self, tmp_path):
        bad = tmp_path / "bad.py"
        bad.write_text("def foo(:\n  pass\n")
        good = tmp_path / "good.py"
        good.write_text("import os\n")
        result = scan_imports(str(tmp_path))
        assert "os" in result["stdlib"]

    def test_relative_imports_ignored(self, tmp_path):
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "sub.py").write_text("from . import utils\n")
        result = scan_imports(str(tmp_path))
        # Relative imports (level > 0) should not appear
        assert "utils" not in result["third_party"]

    def test_exclude_dirs(self, tmp_path):
        excluded = tmp_path / "vendor"
        excluded.mkdir()
        (excluded / "lib.py").write_text("import numpy\n")
        (tmp_path / "main.py").write_text("import os\n")
        result = scan_imports(str(tmp_path), exclude_dirs=["vendor"])
        assert "numpy" not in result["third_party"]
        assert "os" in result["stdlib"]

    def test_conditional_imports(self, tmp_path):
        script = tmp_path / "cond.py"
        script.write_text(
            "try:\n"
            "    import yaml\n"
            "except ImportError:\n"
            "    yaml = None\n"
        )
        result = scan_imports(str(tmp_path))
        assert "yaml" in result["third_party"]

    def test_pycache_excluded_by_default(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "mod.py").write_text("import secretlib\n")
        (tmp_path / "app.py").write_text("import os\n")
        result = scan_imports(str(tmp_path))
        assert "secretlib" not in result["third_party"]


# =========================================================================
# check_missing_packages
# =========================================================================

class TestCheckMissingPackages:
    def test_installed_stdlib_packages(self):
        result = check_missing_packages(["os", "sys", "json"])
        assert result["installed"] == ["os", "sys", "json"]
        assert result["missing"] == []

    def test_missing_fabricated_package(self):
        result = check_missing_packages(["_nonexistent_package_xyz_12345"])
        assert result["missing"] == ["_nonexistent_package_xyz_12345"]
        assert "_nonexistent_package_xyz_12345" in result["errors"]

    def test_empty_list(self):
        result = check_missing_packages([])
        assert result["installed"] == []
        assert result["missing"] == []

    def test_mixed_installed_and_missing(self):
        result = check_missing_packages(["os", "_nonexistent_abc_999"])
        assert "os" in result["installed"]
        assert "_nonexistent_abc_999" in result["missing"]

    def test_invalid_python_executable(self):
        result = check_missing_packages(
            ["os"], python_executable="/nonexistent/python"
        )
        assert result["missing"] == ["os"]
        assert "os" in result["errors"]


# =========================================================================
# pip_name
# =========================================================================

class TestPipName:
    def test_known_mappings(self):
        assert pip_name("cv2") == "opencv-python"
        assert pip_name("sklearn") == "scikit-learn"
        assert pip_name("PIL") == "Pillow"
        assert pip_name("yaml") == "PyYAML"
        assert pip_name("Bio") == "biopython"
        assert pip_name("magic") == "python-magic"
        assert pip_name("socks") == "PySocks"
        assert pip_name("dotenv") == "python-dotenv"
        assert pip_name("comet_ml") == "comet-ml"
        assert pip_name("lightning") == "pytorch-lightning"

    def test_unknown_returns_same(self):
        assert pip_name("torch") == "torch"
        assert pip_name("numpy") == "numpy"


# =========================================================================
# detect_env_manager
# =========================================================================

class TestDetectEnvManager:
    @pytest.mark.parametrize("filename,content,expected_manager", [
        ("environment.yml", "name: myenv\n", "conda"),
        ("environment.yaml", "name: myenv\n", "conda"),
        ("uv.lock", "", "uv"),
        ("pyproject.toml", "[tool.poetry]\nname = 'myproject'\n", "poetry"),
        ("requirements.txt", "torch\nnumpy\n", "pip"),
        ("setup.py", "from setuptools import setup\n", "pip"),
        ("pyproject.toml", "[build-system]\n", "pip"),
    ])
    def test_single_file_detection(self, tmp_path, filename, content, expected_manager):
        (tmp_path / filename).write_text(content)
        result = detect_env_manager(str(tmp_path))
        assert result["manager"] == expected_manager

    def test_conda_takes_priority_over_pip(self, tmp_path):
        (tmp_path / "environment.yml").write_text("name: env\n")
        (tmp_path / "requirements.txt").write_text("torch\n")
        result = detect_env_manager(str(tmp_path))
        assert result["manager"] == "conda"

    def test_venv_dot_venv(self, tmp_path):
        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()
        (venv_dir / "pyvenv.cfg").write_text("home = /usr/bin\n")
        result = detect_env_manager(str(tmp_path))
        assert result["manager"] == "venv"
        assert "pyvenv.cfg" in result["config_file"]

    def test_venv_venv_dir(self, tmp_path):
        venv_dir = tmp_path / "venv"
        venv_dir.mkdir()
        (venv_dir / "pyvenv.cfg").write_text("home = /usr/bin\n")
        result = detect_env_manager(str(tmp_path))
        assert result["manager"] == "venv"

    def test_venv_priority_below_conda(self, tmp_path):
        """conda should take priority over venv."""
        (tmp_path / "environment.yml").write_text("name: env\n")
        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()
        (venv_dir / "pyvenv.cfg").write_text("home = /usr/bin\n")
        result = detect_env_manager(str(tmp_path))
        assert result["manager"] == "conda"

    def test_empty_dir_unknown(self, tmp_path):
        result = detect_env_manager(str(tmp_path))
        assert result["manager"] == "unknown"
        assert result["config_file"] is None


# =========================================================================
# detect_dataset_format
# =========================================================================

class TestDetectDatasetFormat:
    def test_tiny_resnet_cifar(self):
        result = detect_dataset_format(str(TINY_RESNET / "train.py"))
        assert result["format"] == "cifar"
        assert "CIFAR10" in result["patterns_found"]

    def test_imagefolder_script(self):
        result = detect_dataset_format(str(FIXTURES / "sample_imageloader_script.py"))
        assert result["format"] == "image_folder"
        assert "ImageFolder" in result["patterns_found"]

    def test_csv_loader_script(self):
        result = detect_dataset_format(str(FIXTURES / "sample_csv_loader_script.py"))
        assert result["format"] == "csv"
        assert "read_csv" in result["patterns_found"]

    def test_csv_data_args_detected(self):
        result = detect_dataset_format(str(FIXTURES / "sample_csv_loader_script.py"))
        assert "--train_path" in result["data_args"]
        assert "--val_path" in result["data_args"]

    def test_huggingface_script(self):
        result = detect_dataset_format(str(FIXTURES / "sample_huggingface_script.py"))
        assert result["format"] == "huggingface"
        assert "load_dataset" in result["patterns_found"]

    def test_expanded_data_args_keywords(self, tmp_path):
        """_extract_data_args picks up --source_dir, --output_location, --file."""
        script = tmp_path / "args.py"
        script.write_text(
            "import argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--source_dir')\n"
            "parser.add_argument('--output_location')\n"
            "parser.add_argument('--model_file')\n"
            "parser.add_argument('--lr', type=float)\n"
        )
        result = detect_dataset_format(str(script))
        assert "--source_dir" in result["data_args"]
        assert "--output_location" in result["data_args"]
        assert "--model_file" in result["data_args"]
        # --lr has no data-related keyword
        assert "--lr" not in result["data_args"]

    def test_nonexistent_file(self):
        result = detect_dataset_format("/nonexistent/script.py")
        assert result["format"] == "unknown"
        assert result["confidence"] == "low"

    def test_syntax_error_script(self, tmp_path):
        bad = tmp_path / "bad.py"
        bad.write_text("def foo(:\n")
        result = detect_dataset_format(str(bad))
        assert result["format"] == "unknown"
        assert result["confidence"] == "low"

    def test_no_data_patterns(self, tmp_path):
        script = tmp_path / "noop.py"
        script.write_text("import os\nprint('hello')\n")
        result = detect_dataset_format(str(script))
        assert result["format"] == "unknown"

    def test_hdf5_detection(self, tmp_path):
        script = tmp_path / "hdf5_loader.py"
        script.write_text("import h5py\nf = h5py.File('data.h5', 'r')\n")
        result = detect_dataset_format(str(script))
        assert result["format"] == "hdf5"

    def test_sklearn_format_detection(self, tmp_path):
        """sklearn train_test_split pattern is detected."""
        script = tmp_path / "train_sklearn.py"
        script.write_text(
            "from sklearn.model_selection import train_test_split\n"
            "import pandas as pd\n"
            "df = pd.read_csv('data.csv')\n"
            "X_train, X_test = train_test_split(df)\n"
        )
        result = detect_dataset_format(str(script))
        assert result["format"] in ("sklearn", "csv")
        assert len(result["patterns_found"]) > 0

    def test_xgboost_dmatrix_detection(self, tmp_path):
        """XGBoost DMatrix pattern is detected."""
        script = tmp_path / "train_xgb.py"
        script.write_text(
            "import xgboost as xgb\n"
            "dtrain = xgb.DMatrix(data, label=labels)\n"
        )
        result = detect_dataset_format(str(script))
        assert result["format"] == "xgboost"
        assert "DMatrix" in result["patterns_found"]

    def test_lightgbm_dataset_detection(self, tmp_path):
        """LightGBM lgb.Dataset pattern is detected."""
        script = tmp_path / "train_lgb.py"
        script.write_text(
            "import lightgbm as lgb\n"
            "dtrain = lgb.Dataset(data, label=labels)\n"
        )
        result = detect_dataset_format(str(script))
        assert result["format"] == "lightgbm"

    def test_tfrecord_detection(self, tmp_path):
        """TFRecordDataset pattern is detected."""
        script = tmp_path / "train_tf.py"
        script.write_text(
            "import tensorflow as tf\n"
            "dataset = tf.data.TFRecordDataset('train.tfrecord')\n"
        )
        result = detect_dataset_format(str(script))
        assert result["format"] == "tfrecord"
        assert "TFRecordDataset" in result["patterns_found"]

    def test_parquet_detection(self, tmp_path):
        """read_parquet pattern is detected."""
        script = tmp_path / "data_loader.py"
        script.write_text(
            "import pandas as pd\n"
            "df = pd.read_parquet('data.parquet')\n"
        )
        result = detect_dataset_format(str(script))
        assert result["format"] == "parquet"
        assert "read_parquet" in result["patterns_found"]

    def test_auto_download_detection(self, tmp_path):
        """download=True pattern is detected as auto_download."""
        script = tmp_path / "train.py"
        script.write_text(
            "import torchvision\n"
            "dataset = torchvision.datasets.STL10(root='./data', download=True)\n"
        )
        result = detect_dataset_format(str(script))
        assert result["format"] == "auto_download"
        assert "download=True" in result["patterns_found"]

    def test_multiple_formats_detected(self, tmp_path):
        """When multiple format patterns found, all are captured in patterns_found."""
        script = tmp_path / "multi_data.py"
        script.write_text(
            "import h5py\n"
            "import pandas as pd\n"
            "df = pd.read_csv('meta.csv')\n"
            "f = h5py.File('data.h5', 'r')\n"
        )
        result = detect_dataset_format(str(script))
        assert result["format"] in ("csv", "hdf5")
        assert len(result["patterns_found"]) >= 2


# =========================================================================
# validate_data_path
# =========================================================================

class TestValidateDataPath:
    def test_nonexistent_path(self):
        result = validate_data_path("/nonexistent/data", "image_folder")
        assert result["exists"] is False
        assert len(result["errors"]) > 0

    def test_empty_directory(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        result = validate_data_path(str(d), "image_folder")
        assert result["exists"] is True
        assert result["non_empty"] is False

    def test_valid_image_folder(self, tmp_path):
        root = tmp_path / "dataset"
        root.mkdir()
        cls_a = root / "cat"
        cls_a.mkdir()
        (cls_a / "img1.jpg").write_bytes(b"\xff\xd8\xff")
        (cls_a / "img2.png").write_bytes(b"\x89PNG")
        cls_b = root / "dog"
        cls_b.mkdir()
        (cls_b / "img1.jpg").write_bytes(b"\xff\xd8\xff")
        result = validate_data_path(str(root), "image_folder")
        assert result["exists"] is True
        assert result["non_empty"] is True
        assert result["format_matches"] is True
        assert result["details"]["class_dirs"] == 2

    def test_image_folder_no_subdirs(self, tmp_path):
        root = tmp_path / "flat"
        root.mkdir()
        (root / "img1.jpg").write_bytes(b"\xff\xd8\xff")
        result = validate_data_path(str(root), "image_folder")
        assert result["format_matches"] is False

    def test_image_folder_empty_subdirs(self, tmp_path):
        root = tmp_path / "noimg"
        root.mkdir()
        (root / "classA").mkdir()
        (root / "classB").mkdir()
        result = validate_data_path(str(root), "image_folder")
        assert result["format_matches"] is False

    def test_csv_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3\n")
        result = validate_data_path(str(f), "csv")
        assert result["format_matches"] is True

    def test_csv_dir_with_csv_files(self, tmp_path):
        d = tmp_path / "csvdir"
        d.mkdir()
        (d / "train.csv").write_text("a,b\n1,2\n")
        result = validate_data_path(str(d), "csv")
        assert result["format_matches"] is True

    def test_csv_dir_without_csv_files(self, tmp_path):
        d = tmp_path / "wrongdir"
        d.mkdir()
        (d / "data.json").write_text("{}")
        result = validate_data_path(str(d), "csv")
        assert result["format_matches"] is False

    def test_file_format_mismatch(self, tmp_path):
        f = tmp_path / "data.h5"
        f.write_bytes(b"\x89HDF")
        result = validate_data_path(str(f), "csv")
        assert result["format_matches"] is False
        assert len(result["errors"]) > 0

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("")
        result = validate_data_path(str(f), "csv")
        assert result["non_empty"] is False

    def test_unknown_format_passes(self, tmp_path):
        d = tmp_path / "generic"
        d.mkdir()
        (d / "stuff.bin").write_bytes(b"\x00\x01")
        result = validate_data_path(str(d), "unknown")
        assert result["exists"] is True
        assert result["non_empty"] is True
        assert result["format_matches"] is None


# =========================================================================
# CLI tests
# =========================================================================

class TestCLI:
    def test_no_args(self, run_main):
        r = run_main("prerequisites_check.py")
        assert r.returncode == 1
        assert "Usage" in r.stdout

    def test_unknown_action(self, run_main):
        r = run_main("prerequisites_check.py", "bogus")
        assert r.returncode == 1
        assert "Unknown action" in r.stdout

    def test_scan_imports_cli(self, run_main):
        r = run_main("prerequisites_check.py", "scan-imports", str(TINY_RESNET))
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "third_party" in data
        assert "torch" in data["third_party"]

    def test_check_packages_cli(self, run_main):
        r = run_main("prerequisites_check.py", "check-packages", '["os", "sys"]')
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["installed"] == ["os", "sys"]

    def test_check_packages_cli_bad_json(self, run_main):
        r = run_main("prerequisites_check.py", "check-packages", "not-json")
        assert r.returncode == 1

    def test_detect_env_cli(self, run_main):
        r = run_main("prerequisites_check.py", "detect-env", str(TINY_RESNET))
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "manager" in data

    def test_detect_format_cli(self, run_main):
        r = run_main(
            "prerequisites_check.py", "detect-format",
            str(TINY_RESNET / "train.py"),
        )
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["format"] == "cifar"

    def test_validate_data_cli(self, run_main, tmp_path):
        d = tmp_path / "testdata"
        d.mkdir()
        (d / "a.csv").write_text("x\n1\n")
        r = run_main(
            "prerequisites_check.py", "validate-data", str(d), "csv",
        )
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["exists"] is True

    def test_scan_imports_missing_arg(self, run_main):
        r = run_main("prerequisites_check.py", "scan-imports")
        assert r.returncode == 1

    def test_detect_format_missing_arg(self, run_main):
        r = run_main("prerequisites_check.py", "detect-format")
        assert r.returncode == 1

    def test_validate_data_missing_arg(self, run_main):
        r = run_main("prerequisites_check.py", "validate-data", "/some/path")
        assert r.returncode == 1

    def test_gpu_install_cmd_cli(self, run_main):
        r = run_main("prerequisites_check.py", "gpu-install-cmd", "numpy")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["package"] == "numpy"
        assert "pip install numpy" in data["install_command"]

    def test_gpu_install_cmd_missing_arg(self, run_main):
        r = run_main("prerequisites_check.py", "gpu-install-cmd")
        assert r.returncode == 1


# =========================================================================
# GPU install command tests
# =========================================================================

class TestGpuInstallCommand:
    def test_unknown_package_returns_plain_pip(self):
        result = gpu_install_command("numpy")
        assert result["package"] == "numpy"
        assert result["install_command"] == "pip install numpy"

    def test_torch_package_returns_install_command(self):
        result = gpu_install_command("torch")
        assert result["package"] == "torch"
        assert "pip install torch" in result["install_command"]

    def test_tensorflow_without_gpu(self, monkeypatch):
        monkeypatch.setattr(
            "prerequisites_check._detect_cuda_version", lambda: None
        )
        result = gpu_install_command("tensorflow")
        assert result["gpu_detected"] is False
        assert result["install_command"] == "pip install tensorflow"

    def test_tensorflow_with_gpu(self, monkeypatch):
        monkeypatch.setattr(
            "prerequisites_check._detect_cuda_version", lambda: "12.1"
        )
        result = gpu_install_command("tensorflow")
        assert result["gpu_detected"] is True
        assert "[and-cuda]" in result["install_command"]

    def test_torch_with_cuda_121(self, monkeypatch):
        monkeypatch.setattr(
            "prerequisites_check._detect_cuda_version", lambda: "12.1"
        )
        result = gpu_install_command("torch")
        assert result["gpu_detected"] is True
        assert "cu121" in result["install_command"]
        assert "--index-url" in result["install_command"]

    def test_torch_with_cuda_118(self, monkeypatch):
        monkeypatch.setattr(
            "prerequisites_check._detect_cuda_version", lambda: "11.8"
        )
        result = gpu_install_command("torchvision")
        assert "cu118" in result["install_command"]

    def test_best_torch_cuda_tag_picks_highest_compatible(self):
        assert _best_torch_cuda_tag("12.8") == "12.8"
        assert _best_torch_cuda_tag("12.4") == "12.4"
        assert _best_torch_cuda_tag("12.3") == "12.1"
        assert _best_torch_cuda_tag("11.8") == "11.8"
        assert _best_torch_cuda_tag("11.7") is None

    def test_best_torch_cuda_tag_invalid_version(self):
        assert _best_torch_cuda_tag("abc") is None


# =========================================================================
# check-packages CLI with custom python executable
# =========================================================================

class TestCheckPackagesCustomPython:
    def test_cli_with_python_arg(self, run_main):
        """CLI check-packages accepts optional 3rd arg for python executable."""
        r = run_main(
            "prerequisites_check.py", "check-packages",
            '["os", "sys"]', sys.executable,
        )
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["installed"] == ["os", "sys"]

    def test_cli_without_python_arg_defaults_to_python3(self, run_main):
        """CLI check-packages without 3rd arg still works (uses default)."""
        r = run_main(
            "prerequisites_check.py", "check-packages", '["os"]',
        )
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "os" in data["installed"]


# =========================================================================
# bulk_install_command
# =========================================================================

class TestBulkInstallCommand:
    def test_pip_requirements(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("torch\nnumpy\n")
        result = bulk_install_command(str(tmp_path), "pip")
        assert result["has_deps_file"] is True
        assert "requirements.txt" in result["deps_file"]
        assert "pip install -r" in result["install_command"]

    def test_conda_env_yml(self, tmp_path):
        (tmp_path / "environment.yml").write_text("name: myenv\ndependencies:\n  - numpy\n")
        result = bulk_install_command(str(tmp_path), "conda")
        assert result["has_deps_file"] is True
        assert "environment.yml" in result["deps_file"]
        assert "conda" in result["install_command"]

    def test_poetry(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.poetry]\nname = 'proj'\n")
        result = bulk_install_command(str(tmp_path), "poetry")
        assert result["has_deps_file"] is True
        assert "pyproject.toml" in result["deps_file"]
        assert "poetry install" in result["install_command"]

    def test_uv(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("torch\n")
        result = bulk_install_command(str(tmp_path), "uv")
        assert result["has_deps_file"] is True
        assert "uv pip install" in result["install_command"]

    def test_no_deps_file(self, tmp_path):
        result = bulk_install_command(str(tmp_path), "pip")
        assert result["has_deps_file"] is False
        assert result["deps_file"] is None
        assert result["install_command"] is None

    def test_cli(self, run_main, tmp_path):
        (tmp_path / "requirements.txt").write_text("numpy\n")
        r = run_main(
            "prerequisites_check.py", "bulk-install-cmd",
            str(tmp_path), "pip",
        )
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["has_deps_file"] is True

    def test_cli_missing_args(self, run_main):
        r = run_main("prerequisites_check.py", "bulk-install-cmd")
        assert r.returncode == 1


# =========================================================================
# detect_dataset_format_project (follows imports)
# =========================================================================

class TestDetectDatasetFormatProject:
    def test_follows_imports(self, tmp_path):
        """Format detected in imported data module, not the training script."""
        # train.py imports from data module
        (tmp_path / "train.py").write_text(
            "import torch\n"
            "from data import get_loader\n"
            "\n"
            "loader = get_loader()\n"
        )
        # data.py has the ImageFolder pattern
        (tmp_path / "data.py").write_text(
            "from torchvision.datasets import ImageFolder\n"
            "from torch.utils.data import DataLoader\n"
            "\n"
            "def get_loader():\n"
            "    dataset = ImageFolder(root='./data/train')\n"
            "    return DataLoader(dataset)\n"
        )
        result = detect_dataset_format_project(str(tmp_path), str(tmp_path / "train.py"))
        assert result["format"] == "image_folder"
        assert "ImageFolder" in result["patterns_found"]
        assert len(result["scanned_files"]) >= 2

    def test_direct_match(self, tmp_path):
        """Format detected directly in the training script."""
        (tmp_path / "train.py").write_text(
            "from torchvision.datasets import CIFAR10\n"
            "dataset = CIFAR10(root='./data', download=True)\n"
        )
        result = detect_dataset_format_project(str(tmp_path), str(tmp_path / "train.py"))
        assert result["format"] == "cifar"
        assert "scanned_files" in result

    def test_no_local_imports(self, tmp_path):
        """Falls back to single-file behavior when no local imports found."""
        (tmp_path / "train.py").write_text(
            "import torch\n"
            "import numpy as np\n"
            "print('no data loading here')\n"
        )
        result = detect_dataset_format_project(str(tmp_path), str(tmp_path / "train.py"))
        assert result["format"] == "unknown"
        assert "scanned_files" in result

    def test_cli(self, run_main, tmp_path):
        (tmp_path / "train.py").write_text(
            "from torchvision.datasets import CIFAR10\n"
            "dataset = CIFAR10(root='./data')\n"
        )
        r = run_main(
            "prerequisites_check.py", "detect-format-project",
            str(tmp_path), str(tmp_path / "train.py"),
        )
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["format"] == "cifar"

    def test_cli_missing_args(self, run_main):
        r = run_main("prerequisites_check.py", "detect-format-project")
        assert r.returncode == 1


# =========================================================================
# Format priority fix: specific formats beat generic ones
# =========================================================================

class TestFormatPriority:
    def test_cifar_beats_image_folder(self, tmp_path):
        """CIFAR10 (specific) should rank above ImageFolder (generic)."""
        script = tmp_path / "multi.py"
        script.write_text(
            "from torchvision.datasets import ImageFolder, CIFAR10\n"
            "dataset_a = ImageFolder(root='./data')\n"
            "dataset_b = CIFAR10(root='./data', download=True)\n"
        )
        result = detect_dataset_format(str(script))
        assert result["format"] == "cifar"

    def test_mnist_beats_image_folder(self, tmp_path):
        """MNIST (specific) should rank above ImageFolder (generic)."""
        script = tmp_path / "multi.py"
        script.write_text(
            "from torchvision.datasets import ImageFolder, MNIST\n"
            "x = ImageFolder(root='./data')\n"
            "y = MNIST(root='./data')\n"
        )
        result = detect_dataset_format(str(script))
        assert result["format"] == "mnist"


# =========================================================================
# JAX GPU install support (Task A)
# =========================================================================

class TestJaxGpuInstall:
    def test_jax_with_cuda12(self, monkeypatch):
        monkeypatch.setattr(
            "prerequisites_check._detect_cuda_version", lambda: "12.4"
        )
        result = gpu_install_command("jax")
        assert result["gpu_detected"] is True
        assert result["install_command"] == "pip install jax[cuda12]"

    def test_jax_with_cuda11(self, monkeypatch):
        monkeypatch.setattr(
            "prerequisites_check._detect_cuda_version", lambda: "11.8"
        )
        result = gpu_install_command("jax")
        assert result["gpu_detected"] is True
        assert result["install_command"] == "pip install jax[cuda11_pip]"

    def test_jax_without_gpu(self, monkeypatch):
        monkeypatch.setattr(
            "prerequisites_check._detect_cuda_version", lambda: None
        )
        result = gpu_install_command("jax")
        assert result["gpu_detected"] is False
        assert result["install_command"] == "pip install jax"

    def test_jaxlib_with_gpu(self, monkeypatch):
        """jaxlib standalone install is plain pip (jax[cuda] pulls it in)."""
        monkeypatch.setattr(
            "prerequisites_check._detect_cuda_version", lambda: "12.1"
        )
        result = gpu_install_command("jaxlib")
        assert result["gpu_detected"] is True
        assert result["install_command"] == "pip install jaxlib"


# =========================================================================
# [tool.uv] detection (Task B)
# =========================================================================

class TestUvPyprojectDetection:
    def test_uv_in_pyproject_toml(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname = 'myproj'\n\n[tool.uv]\n"
        )
        result = detect_env_manager(str(tmp_path))
        assert result["manager"] == "uv"
        assert "pyproject.toml" in result["config_file"]

    def test_poetry_takes_priority_over_uv_in_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            "[tool.poetry]\nname = 'myproj'\n\n[tool.uv]\n"
        )
        result = detect_env_manager(str(tmp_path))
        assert result["manager"] == "poetry"


# =========================================================================
# iterdir PermissionError handling (Task C)
# =========================================================================

class TestIterdirPermissionError:
    def test_validate_data_path_iterdir_error(self, tmp_path):
        d = tmp_path / "restricted"
        d.mkdir()
        (d / "data.csv").write_text("a,b\n1,2\n")
        with patch("prerequisites_check.itertools.islice", side_effect=PermissionError("denied")):
            result = validate_data_path(str(d), "csv")
        assert result["exists"] is True
        assert result["readable"] is True
        assert any("Error reading" in e for e in result["errors"])

    def test_image_folder_iterdir_error(self, tmp_path):
        root = tmp_path / "dataset"
        root.mkdir()
        (root / "placeholder").write_bytes(b"\x00")
        # The outer iterdir for entries works, but _validate_image_folder's iterdir fails
        with patch.object(Path, "iterdir", side_effect=PermissionError("denied")):
            result = validate_data_path(str(root), "image_folder")
        assert any("Error reading" in e for e in result["errors"])


# =========================================================================
# Broken symlink detection (Task D)
# =========================================================================

class TestBrokenSymlink:
    def test_broken_symlink_reported(self, tmp_path):
        target = tmp_path / "missing_target"
        link = tmp_path / "broken_link"
        link.symlink_to(target)
        result = validate_data_path(str(link), "csv")
        assert result["exists"] is False
        assert any("Broken symlink" in e for e in result["errors"])
        assert "missing_target" in result["errors"][0]


# =========================================================================
# _best_torch_cuda_tag edge cases (Task E/I)
# =========================================================================

class TestBestTorchCudaTagEdgeCases:
    def test_empty_string(self):
        assert _best_torch_cuda_tag("") is None

    def test_single_segment(self):
        assert _best_torch_cuda_tag("12") is None

    def test_three_segments(self):
        assert _best_torch_cuda_tag("12.3.4") is None

    def test_none_input(self):
        assert _best_torch_cuda_tag(None) is None

    @pytest.mark.parametrize("malformed", [
        "12.x", "x.4", ".", "12.", ".4", "cuda12.1", "12.1.2",
    ])
    def test_malformed_cuda_versions_return_none(self, malformed):
        """Various malformed CUDA version strings all return None."""
        assert _best_torch_cuda_tag(malformed) is None


# =========================================================================
# _detect_cuda_version nvidia-smi parsing (Task F)
# =========================================================================

class TestDetectCudaVersion:
    def test_parses_realistic_nvidia_smi_output(self):
        nvidia_output = (
            "+-------------------------------------------------------------------------+\n"
            "| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.4 |\n"
            "+-------------------------------------------------------------------------+\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = nvidia_output
        with patch("prerequisites_check.subprocess.run", return_value=mock_result):
            version = _detect_cuda_version()
        assert version == "12.4"

    def test_returns_none_when_no_gpu(self):
        with patch("prerequisites_check.subprocess.run", side_effect=FileNotFoundError):
            version = _detect_cuda_version()
        assert version is None

    def test_returns_none_on_nonzero_exit(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("prerequisites_check.subprocess.run", return_value=mock_result):
            version = _detect_cuda_version()
        assert version is None

    def test_returns_none_on_timeout(self):
        with patch(
            "prerequisites_check.subprocess.run",
            side_effect=subprocess.TimeoutExpired("nvidia-smi", 10),
        ):
            version = _detect_cuda_version()
        assert version is None


# =========================================================================
# Subprocess timeout in check_missing_packages (Task G)
# =========================================================================

class TestCheckPackagesTimeout:
    def test_timeout_handled_gracefully(self):
        with patch(
            "prerequisites_check.subprocess.run",
            side_effect=subprocess.TimeoutExpired("python3", 30),
        ):
            result = check_missing_packages(["slow_pkg"])
        assert "slow_pkg" in result["missing"]
        assert "timed out" in result["errors"]["slow_pkg"].lower()


# =========================================================================
# Permission denied in validate_data_path (Task H)
# =========================================================================

class TestValidateDataPathPermission:
    def test_unreadable_directory(self, tmp_path):
        d = tmp_path / "restricted"
        d.mkdir()
        (d / "data.csv").write_text("a,b\n1,2\n")
        with patch("prerequisites_check.os.access", return_value=False):
            result = validate_data_path(str(d), "csv")
        assert result["exists"] is True
        assert result["readable"] is False
        assert any("not readable" in e for e in result["errors"])


# =========================================================================
# Large directory sampling (Task J)
# =========================================================================

class TestLargeDirectorySampling:
    def test_samples_at_most_1000_entries(self, tmp_path):
        d = tmp_path / "bigdir"
        d.mkdir()
        for i in range(1100):
            (d / f"file_{i:04d}.bin").write_bytes(b"\x00")
        result = validate_data_path(str(d), "unknown")
        assert result["exists"] is True
        assert result["non_empty"] is True
        assert result["details"]["entry_count_sampled"] == 1000


# =========================================================================
# _wrap_for_conda helper
# =========================================================================

class TestWrapForConda:
    def test_wraps_when_conda_with_env_name(self):
        result = _wrap_for_conda("pip install torch", "conda", "myenv")
        assert result == "conda run --no-banner -n myenv pip install torch"

    def test_no_wrap_when_pip(self):
        result = _wrap_for_conda("pip install torch", "pip", None)
        assert result == "pip install torch"

    def test_no_wrap_when_conda_without_env_name(self):
        result = _wrap_for_conda("pip install torch", "conda", None)
        assert result == "pip install torch"

    def test_no_wrap_when_no_manager(self):
        result = _wrap_for_conda("pip install torch", None, None)
        assert result == "pip install torch"


# =========================================================================
# gpu_install_command with conda env_manager (P2)
# =========================================================================

class TestGpuInstallCommandConda:
    def test_torch_conda_wraps_command(self, monkeypatch):
        monkeypatch.setattr(
            "prerequisites_check._detect_cuda_version", lambda: "12.1"
        )
        result = gpu_install_command("torch", env_manager="conda", env_name="myenv")
        assert result["install_command"].startswith("conda run --no-banner -n myenv ")
        assert "pip install torch --index-url" in result["install_command"]
        assert "cu121" in result["install_command"]

    def test_tensorflow_conda_wraps_command(self, monkeypatch):
        monkeypatch.setattr(
            "prerequisites_check._detect_cuda_version", lambda: "12.1"
        )
        result = gpu_install_command("tensorflow", env_manager="conda", env_name="ml")
        assert result["install_command"].startswith("conda run --no-banner -n ml ")
        assert "[and-cuda]" in result["install_command"]

    def test_jax_conda_wraps_command(self, monkeypatch):
        monkeypatch.setattr(
            "prerequisites_check._detect_cuda_version", lambda: "12.4"
        )
        result = gpu_install_command("jax", env_manager="conda", env_name="jaxenv")
        assert result["install_command"] == "conda run --no-banner -n jaxenv pip install jax[cuda12]"

    def test_plain_package_conda_wraps(self):
        result = gpu_install_command("numpy", env_manager="conda", env_name="base")
        assert result["install_command"] == "conda run --no-banner -n base pip install numpy"

    def test_no_env_manager_no_wrap(self, monkeypatch):
        monkeypatch.setattr(
            "prerequisites_check._detect_cuda_version", lambda: "12.1"
        )
        result = gpu_install_command("torch")
        assert not result["install_command"].startswith("conda run")

    def test_cli_with_env_args(self, run_main):
        r = run_main(
            "prerequisites_check.py", "gpu-install-cmd", "numpy", "conda", "testenv",
        )
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "conda run --no-banner -n testenv" in data["install_command"]

    def test_cli_without_env_args_backward_compatible(self, run_main):
        r = run_main("prerequisites_check.py", "gpu-install-cmd", "numpy")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["install_command"] == "pip install numpy"


# =========================================================================
# bulk_install_command with env_name (P3)
# =========================================================================

class TestBulkInstallCommandCondaEnvName:
    def test_conda_requirements_includes_env_name(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("torch\nnumpy\n")
        result = bulk_install_command(str(tmp_path), "conda", env_name="myenv")
        assert result["has_deps_file"] is True
        assert "-n myenv" in result["install_command"]
        assert "conda install" in result["install_command"]

    def test_conda_env_yml_includes_env_name(self, tmp_path):
        (tmp_path / "environment.yml").write_text("name: default\ndependencies:\n  - numpy\n")
        result = bulk_install_command(str(tmp_path), "conda", env_name="myenv")
        assert result["has_deps_file"] is True
        assert "-n myenv" in result["install_command"]
        assert "conda env update" in result["install_command"]

    def test_conda_without_env_name_no_flag(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("torch\n")
        result = bulk_install_command(str(tmp_path), "conda")
        assert result["has_deps_file"] is True
        assert "-n " not in result["install_command"]

    def test_conda_pyproject_fallback_wraps(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[build-system]\n")
        result = bulk_install_command(str(tmp_path), "conda", env_name="myenv")
        assert result["has_deps_file"] is True
        assert "conda run --no-banner -n myenv" in result["install_command"]

    def test_pip_ignores_env_name(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("torch\n")
        result = bulk_install_command(str(tmp_path), "pip", env_name="myenv")
        assert "-n myenv" not in result["install_command"]
        assert "pip install -r" in result["install_command"]

    def test_cli_with_env_name(self, run_main, tmp_path):
        (tmp_path / "requirements.txt").write_text("numpy\n")
        r = run_main(
            "prerequisites_check.py", "bulk-install-cmd",
            str(tmp_path), "conda", "myenv",
        )
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "-n myenv" in data["install_command"]

    def test_cli_without_env_name_backward_compatible(self, run_main, tmp_path):
        (tmp_path / "requirements.txt").write_text("numpy\n")
        r = run_main(
            "prerequisites_check.py", "bulk-install-cmd",
            str(tmp_path), "pip",
        )
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "pip install -r" in data["install_command"]
