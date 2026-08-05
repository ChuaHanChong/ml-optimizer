#!/usr/bin/env python3
"""Prerequisites validation: import scanning, package checking, environment detection, dataset format detection, and data path validation.

Also generates GPU-aware (CUDA-tagged) install commands for PyTorch/TensorFlow/JAX
and bulk install commands from dependency files (requirements.txt, environment.yml,
pyproject.toml), env-manager-aware (pip/conda/uv/poetry). "venv" has no dedicated
skip branch — unlike "pip", it does NOT skip conda-only files, so a venv-managed
project with only an environment.yml present will get a `conda env update` install
command.

Usage:
    python3 prerequisites_check.py scan-imports <project_root>                              # Classify imports as stdlib/third_party/local
    python3 prerequisites_check.py check-packages '<json_list>' [python_executable] [host]  # Report which packages are installed/missing (probe over ssh when host is given)
    python3 prerequisites_check.py detect-env <project_root>                                # Detect the env manager (conda/uv/poetry/pip/venv)
    python3 prerequisites_check.py detect-format <training_script>                          # Detect dataset format from a training script
    python3 prerequisites_check.py detect-format-project <project_root> <training_script>   # Detect dataset format across script + local imports
    python3 prerequisites_check.py validate-data <path> <format>                           # Validate a data path matches an expected format
    python3 prerequisites_check.py gpu-install-cmd <package> [env_manager] [env_name]       # Build a GPU-aware install command
    python3 prerequisites_check.py bulk-install-cmd <project_root> <env_manager> [env_name] # Build a bulk install command from dependency files
"""

import ast
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# Import-name → pip-package mapping for common ML aliases
IMPORT_TO_PACKAGE: dict[str, str] = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "attr": "attrs",
    "bs4": "beautifulsoup4",
    "skimage": "scikit-image",
    "dateutil": "python-dateutil",
    "gi": "PyGObject",
    "serial": "pyserial",
    "usb": "pyusb",
    "wx": "wxPython",
    "Crypto": "pycryptodome",
    "magic": "python-magic",
    "Bio": "biopython",
    "lxml": "lxml",
    "socks": "PySocks",
    "dotenv": "python-dotenv",
    "comet_ml": "comet-ml",
    "lightning": "pytorch-lightning",
    "stable_baselines3": "stable-baselines3",
    "sample_factory": "sample-factory",
    "rsl_rl": "rsl-rl-lib",
    "tensorflow_datasets": "tensorflow-datasets",
}

# Imports that must NEVER be auto-installed from PyPI: they ship with a
# simulator/robotics runtime (Isaac Sim, ROS, habitat conda packages) or a bare
# `pip install <name>` grabs the wrong package. Surfaced as
# "manual_install_required" with guidance instead of installing.
NEVER_AUTO_INSTALL: dict[str, str] = {
    "omni": "Ships with NVIDIA Isaac Sim / Omniverse — install via the Isaac Sim installer, then run training inside that environment.",
    "carb": "Ships with NVIDIA Isaac Sim / Omniverse — install via the Isaac Sim installer.",
    "pxr": "USD Python bindings — ship with Isaac Sim/Omniverse (or usd-core); do not pip install blindly.",
    "isaacsim": "Install via the NVIDIA Isaac Sim installer, not PyPI.",
    "isaaclab": "Install via the Isaac Lab setup script (isaaclab.sh) inside an Isaac Sim environment, not PyPI.",
    "habitat": "Install habitat-lab from source (github.com/facebookresearch/habitat-lab); the pip name collides with an unrelated package.",
    "habitat_sim": "Install via conda: conda install habitat-sim -c conda-forge -c aihabitat; no pip wheels are published.",
    "rclpy": "ROS 2 Python bindings — installed with a ROS 2 distribution; source your ROS setup script instead of pip.",
    "rospy": "ROS 1 Python bindings — installed with a ROS distribution; source your ROS setup script instead of pip.",
    "warp": "Usually NVIDIA Warp (pip install warp-lang); bare 'pip install warp' installs a different package.",
}

# Directories to skip when scanning imports
_DEFAULT_EXCLUDE_DIRS: set[str] = {
    ".git", "__pycache__", "node_modules", ".eggs", "build", "dist",
    ".tox", ".mypy_cache", ".pytest_cache", "venv", ".venv",
    "env", ".env", ".worktrees", "worktrees",
}

# Image extensions for ImageFolder validation
_IMAGE_EXTENSIONS: set[str] = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp",
}


def scan_imports(
    project_root: str,
    exclude_dirs: list[str] | None = None,
) -> dict:
    """Scan all `.py` files under *project_root* for import statements.

    Uses `ast.parse` to classify each top-level module name as *stdlib*, *local*
    (file exists in project), or *third_party*. Returns
    {"stdlib": [...], "third_party": [...], "local": [...]}.
    """
    root = Path(project_root).resolve()
    excludes = _DEFAULT_EXCLUDE_DIRS | set(exclude_dirs or [])

    # Dynamically skip the user's actual exp_root so the scanner doesn't
    # walk a potentially large output directory full of checkpoints/logs.
    breadcrumb = root / ".claude" / "ml-optimizer.json"
    if breadcrumb.is_file():
        try:
            _bc = json.loads(breadcrumb.read_text())
            _exp = _bc.get("active") or _bc.get("exp_root") or ""
            if _exp:
                excludes.add(Path(_exp).name)
        except (json.JSONDecodeError, OSError):
            pass

    # Collect local module names (any .py file or package dir in the project)
    local_names: set[str] = set()
    all_py_files: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories (mutate in-place for os.walk)
        dirnames[:] = [d for d in dirnames if d not in excludes]
        for fname in filenames:
            if fname.endswith(".py"):
                all_py_files.append(Path(dirpath) / fname)
                # Register the stem as a local module name
                local_names.add(Path(fname).stem)
        # Package directories (contain __init__.py)
        for dname in dirnames:
            if (Path(dirpath) / dname / "__init__.py").is_file():
                local_names.add(dname)

    # Parse imports from each file
    raw_imports: set[str] = set()
    for py_file in all_py_files:
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    raw_imports.add(top)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    top = node.module.split(".")[0]
                    raw_imports.add(top)

    # Classify
    stdlib_modules: set[str] = getattr(sys, "stdlib_module_names", set())
    stdlib: list[str] = sorted(i for i in raw_imports if i in stdlib_modules)
    local: list[str] = sorted(
        i for i in raw_imports if i not in stdlib_modules and i in local_names
    )
    third_party: list[str] = sorted(
        i for i in raw_imports if i not in stdlib_modules and i not in local_names
    )

    return {"stdlib": stdlib, "third_party": third_party, "local": local}


def check_missing_packages(
    packages: list[str],
    python_executable: str = "python3",
    host: str = "",
) -> dict:
    """Check which *packages* cannot be imported by *python_executable*.

    With *host* set (or ML_OPTIMIZER_GPU_HOST in the environment), the probes run there
    over ssh — for remote training it is that environment which must satisfy the imports.

    Imports listed in NEVER_AUTO_INSTALL are routed to "manual_install_required"
    (import name → guidance) instead of "missing" when absent. Returns
    {"installed": [...], "missing": [...], "manual_install_required": {...},
    "errors": {...}}.
    """
    installed: list[str] = []
    missing: list[str] = []
    manual: dict[str, str] = {}
    errors: dict[str, str] = {}
    host = host or os.environ.get("ML_OPTIMIZER_GPU_HOST", "")

    for pkg in packages:
        # shlex.quote so a package name can never break out of the remote shell word.
        probe = [python_executable, "-c", f"import {pkg}"]
        cmd = (
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
             " ".join(shlex.quote(part) for part in probe)]
            if host
            else probe
        )
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60 if host else 30,
            )
            if result.returncode == 0:
                installed.append(pkg)
            else:
                if pkg in NEVER_AUTO_INSTALL:
                    manual[pkg] = NEVER_AUTO_INSTALL[pkg]
                else:
                    missing.append(pkg)
                stderr = result.stderr.strip()
                if stderr:
                    errors[pkg] = stderr.splitlines()[-1]
        except FileNotFoundError:
            missing.append(pkg)
            # With host set it is ssh that is missing locally, not the remote interpreter.
            errors[pkg] = "ssh not found" if host else f"Python executable not found: {python_executable}"
        except subprocess.TimeoutExpired:
            missing.append(pkg)
            errors[pkg] = "Import timed out"

    return {
        "installed": installed,
        "missing": missing,
        "manual_install_required": manual,
        "errors": errors,
    }


def pip_name(import_name: str) -> str:
    """Map an import name to its pip package name."""
    return IMPORT_TO_PACKAGE.get(import_name, import_name)


def detect_env_manager(project_root: str) -> dict:
    """Detect the project's environment manager from config files (priority order).

    Returns {"manager": "conda|uv|poetry|pip|venv|unknown", "config_file": <path>|None}.
    """
    root = Path(project_root).resolve()

    # conda
    for name in ("environment.yml", "environment.yaml"):
        cfg = root / name
        if cfg.is_file():
            return {"manager": "conda", "config_file": str(cfg)}

    # uv
    uv_lock = root / "uv.lock"
    if uv_lock.is_file():
        return {"manager": "uv", "config_file": str(uv_lock)}

    # poetry (pyproject.toml with [tool.poetry])
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            text = pyproject.read_text(encoding="utf-8", errors="replace")
            if "[tool.poetry]" in text:
                return {"manager": "poetry", "config_file": str(pyproject)}
            if "[tool.uv]" in text:
                return {"manager": "uv", "config_file": str(pyproject)}
        except OSError:
            pass

    # venv / virtualenv (check common directories for pyvenv.cfg)
    for venv_dir in (".venv", "venv", "env"):
        cfg = root / venv_dir / "pyvenv.cfg"
        if cfg.is_file():
            return {"manager": "venv", "config_file": str(cfg)}
    # Also check VIRTUAL_ENV environment variable
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env and Path(virtual_env, "pyvenv.cfg").is_file():
        return {"manager": "venv", "config_file": str(Path(virtual_env, "pyvenv.cfg"))}

    # pip (requirements.txt, setup.py, setup.cfg, or generic pyproject.toml)
    for name in ("requirements.txt", "setup.py", "setup.cfg"):
        cfg = root / name
        if cfg.is_file():
            return {"manager": "pip", "config_file": str(cfg)}
    if pyproject.is_file():
        return {"manager": "pip", "config_file": str(pyproject)}

    return {"manager": "unknown", "config_file": None}


# AST patterns to look for in training scripts
_FORMAT_PATTERNS: dict[str, list[str]] = {
    "image_folder": [
        "ImageFolder",
        "image_dataset_from_directory",
    ],
    "cifar": [
        "CIFAR10", "CIFAR100",
    ],
    "mnist": [
        "MNIST", "FashionMNIST", "EMNIST",
    ],
    "csv": [
        "read_csv", "CsvDataset", "csv.reader", "csv.DictReader",
    ],
    "hdf5": [
        "h5py",
    ],
    "tfrecord": [
        "TFRecordDataset",
    ],
    "parquet": [
        "read_parquet",
    ],
    "numpy": [
        "numpy.load", "np.load",
    ],
    "torch_tensor": [
        "torch.load",
    ],
    "huggingface": [
        "load_dataset",
        "datasets.load_dataset",
        "datasets.Dataset.from_",
    ],
    "lerobot": [
        "LeRobotDataset", "lerobot",
    ],
    "rlds": [
        "rlds", "builder_from_directory",
    ],
    "zarr": [
        "zarr.open", "zarr",
    ],
    "robomimic": [
        "robomimic", "SequenceDataset",
    ],
    "auto_download": [
        "download=True",
    ],
    "sklearn": [
        "train_test_split", "cross_val_score", "load_iris", "load_digits",
        "load_wine", "load_breast_cancer", "fetch_openml",
    ],
    "xgboost": [
        "DMatrix", "xgb.DMatrix", "xgboost.DMatrix",
    ],
    "lightgbm": [
        "lgb.Dataset", "lightgbm.Dataset",
    ],
}


def _collect_names_and_attrs(tree: ast.AST) -> set[str]:
    """Walk the AST and collect all Name ids, Attribute attrs, and string constants."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
            # Also collect dotted chains like torch.load
            parts = []
            cur = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                parts.reverse()
                names.add(".".join(parts))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    return names


def _extract_data_args(tree: ast.AST) -> list[str]:
    """Find argparse arguments that look data-related."""
    data_args: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Look for add_argument("--data_dir", ...) style calls
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                    and node.args):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        val = arg.value
                        if any(k in val.lower() for k in ("data", "dataset", "train", "val", "test", "input", "path", "dir", "file", "source", "output", "location")):
                            data_args.append(val)
    return data_args


def detect_dataset_format(training_script: str) -> dict:
    """Analyze a training script to detect expected dataset format.

    Returns {"format": "image_folder|csv|cifar|...|unknown", "patterns_found": [...],
    "data_args": [...], "confidence": "high|medium|low"}.
    """
    path = Path(training_script)
    if not path.is_file():
        return {
            "format": "unknown",
            "patterns_found": [],
            "data_args": [],
            "confidence": "low",
        }

    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError):
        return {
            "format": "unknown",
            "patterns_found": [],
            "data_args": [],
            "confidence": "low",
        }

    names = _collect_names_and_attrs(tree)
    data_args = _extract_data_args(tree)

    # Match patterns
    found: dict[str, list[str]] = {}
    for fmt, patterns in _FORMAT_PATTERNS.items():
        matches = [p for p in patterns if p in names]
        if matches:
            found[fmt] = matches

    if not found:
        # Fallback: check raw source for patterns we might have missed
        for fmt, patterns in _FORMAT_PATTERNS.items():
            matches = [p for p in patterns if p in source]
            if matches:
                found[fmt] = matches

    if not found:
        return {
            "format": "unknown",
            "patterns_found": [],
            "data_args": data_args,
            "confidence": "low",
        }

    # Determine primary format (specific formats first, then generic)
    priority = [
        "cifar", "mnist", "huggingface",
        "lerobot", "rlds", "robomimic", "zarr",
        "image_folder", "csv", "hdf5", "tfrecord", "parquet",
        "sklearn", "xgboost", "lightgbm",
        "numpy", "torch_tensor",
        "auto_download",
    ]
    all_patterns: list[str] = []
    primary = "unknown"
    for fmt in priority:
        if fmt in found:
            primary = fmt
            all_patterns.extend(found[fmt])
            break

    # Gather remaining patterns
    for fmt, pats in found.items():
        if fmt != primary:
            all_patterns.extend(pats)

    # Confidence: multiple pattern matches or specific format → high
    total_matches = sum(len(v) for v in found.values())
    if total_matches >= 2:
        confidence = "high"
    elif primary in ("image_folder", "csv", "hdf5", "tfrecord", "parquet",
                      "sklearn", "xgboost", "lightgbm",
                      "lerobot", "rlds", "robomimic", "zarr"):
        confidence = "high"
    elif primary in ("cifar", "mnist"):
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "format": primary,
        "patterns_found": all_patterns,
        "data_args": data_args,
        "confidence": confidence,
    }


def detect_dataset_format_project(
    project_root: str,
    training_script: str,
) -> dict:
    """Detect dataset format by scanning the training script AND its local imports.

    Parses the script's imports, scans matching local `.py` files for data-loading
    patterns too, and returns the most confident detection. Same schema as
    :func:`detect_dataset_format`, plus "scanned_files": [...].
    """
    root = Path(project_root).resolve()
    script_path = Path(training_script).resolve()
    scanned: list[str] = [str(script_path)]

    # Start with the training script's own detection
    best = detect_dataset_format(training_script)

    # Parse training script to find local imports
    if script_path.is_file():
        try:
            source = script_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(script_path))
        except (SyntaxError, ValueError):
            tree = None

        if tree is not None:
            local_imports: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        local_imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.level == 0:
                        local_imports.add(node.module.split(".")[0])

            # Find matching .py files in the project
            for mod_name in local_imports:
                candidates = [
                    root / f"{mod_name}.py",
                    root / mod_name / "__init__.py",
                ]
                for candidate in candidates:
                    if candidate.is_file() and str(candidate) not in scanned:
                        scanned.append(str(candidate))
                        result = detect_dataset_format(str(candidate))
                        # Keep the result with higher confidence
                        if _confidence_rank(result["confidence"]) > _confidence_rank(best["confidence"]):
                            best = result
                        elif (
                            _confidence_rank(result["confidence"]) == _confidence_rank(best["confidence"])
                            and best["format"] == "unknown"
                            and result["format"] != "unknown"
                        ):
                            best = result

    best["scanned_files"] = scanned
    return best


def _confidence_rank(confidence: str) -> int:
    """Map confidence string to a sortable integer."""
    return {"high": 3, "medium": 2, "low": 1}.get(confidence, 0)


def validate_data_path(path: str, expected_format: str) -> dict:
    """Validate that *path* exists and matches *expected_format*.

    Samples only the first N files/entries on large datasets. Returns
    {"exists": bool, "readable": bool, "non_empty": bool,
    "format_matches": bool|None, "details": {...}, "errors": [...]}.
    """
    p = Path(path)
    result: dict = {
        "exists": False,
        "readable": False,
        "non_empty": False,
        "format_matches": None,
        "details": {},
        "errors": [],
    }

    if not p.exists():
        if p.is_symlink():
            target = os.readlink(path)
            result["errors"].append(f"Broken symlink: {path} -> {target}")
        else:
            result["errors"].append(f"Path does not exist: {path}")
        return result
    result["exists"] = True

    if not os.access(path, os.R_OK):
        result["errors"].append(f"Path is not readable: {path}")
        return result
    result["readable"] = True

    if p.is_dir():
        # Sample up to 1000 entries to avoid OOM on huge directories
        try:
            entries = list(itertools.islice(p.iterdir(), 1000))
        except (PermissionError, OSError) as exc:
            result["errors"].append(f"Error reading directory: {exc}")
            return result
        if not entries:
            result["errors"].append("Directory is empty")
            return result
        result["non_empty"] = True
        result["details"]["entry_count_sampled"] = len(entries)
        result["details"]["is_directory"] = True

        if expected_format == "image_folder":
            result["format_matches"] = _validate_image_folder(p, result)
        elif expected_format == "zarr":
            result["format_matches"] = _validate_zarr_store(entries, result)
        elif expected_format in ("csv", "hdf5", "parquet", "tfrecord",
                                 "robomimic", "rlds"):
            # For these formats, we expect files, not a directory of class dirs
            # Check if sampled entries contain matching files
            ext_map = {
                "csv": {".csv", ".tsv"},
                "hdf5": {".h5", ".hdf5", ".hdf"},
                "parquet": {".parquet"},
                "tfrecord": {".tfrecord", ".tfrecords"},
                "robomimic": {".h5", ".hdf5", ".hdf"},
                "rlds": {".tfrecord", ".tfrecords"},
            }
            exts = ext_map.get(expected_format, set())
            matching = [f for f in entries if f.is_file() and f.suffix.lower() in exts]
            result["format_matches"] = len(matching) > 0
            result["details"]["matching_files"] = len(matching)
            if not matching:
                result["errors"].append(
                    f"No files with extensions {exts} found in directory"
                )
        else:
            # Generic: just confirm it's non-empty
            result["format_matches"] = None

    elif p.is_file():
        if p.stat().st_size == 0:
            result["errors"].append("File is empty")
            return result
        result["non_empty"] = True
        result["details"]["is_directory"] = False
        result["details"]["size_bytes"] = p.stat().st_size

        ext = p.suffix.lower()
        ext_format_map = {
            ".csv": "csv", ".tsv": "csv",
            ".json": "json", ".jsonl": "json",
            ".h5": "hdf5", ".hdf5": "hdf5", ".hdf": "hdf5",
            ".parquet": "parquet",
            ".tfrecord": "tfrecord", ".tfrecords": "tfrecord",
            ".npy": "numpy", ".npz": "numpy",
            ".pt": "torch_tensor", ".pth": "torch_tensor",
        }
        detected = ext_format_map.get(ext)
        # Formats stored in a container format: accept the container's extension
        equivalents = {"robomimic": "hdf5", "rlds": "tfrecord"}
        expected_effective = equivalents.get(expected_format, expected_format)
        if detected and expected_format != "unknown":
            result["format_matches"] = (detected == expected_effective)
            if not result["format_matches"]:
                result["errors"].append(
                    f"File extension '{ext}' suggests '{detected}', "
                    f"expected '{expected_format}'"
                )
        else:
            result["format_matches"] = None

    return result


def _validate_image_folder(p: Path, result: dict) -> bool:
    """Check ImageFolder structure: subdirectories with image files."""
    try:
        subdirs = [d for d in p.iterdir() if d.is_dir()]
    except (PermissionError, OSError) as exc:
        result["errors"].append(f"Error reading directory: {exc}")
        return False
    if not subdirs:
        result["errors"].append(
            "ImageFolder expects subdirectories (one per class), found none"
        )
        return False

    # Sample up to 5 subdirs, check for images
    classes_with_images = 0
    sample = subdirs[:5]
    for subdir in sample:
        try:
            images = [
                f for f in subdir.iterdir()
                if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
            ]
        except (PermissionError, OSError):
            continue
        if images:
            classes_with_images += 1

    result["details"]["class_dirs"] = len(subdirs)
    result["details"]["sampled_dirs_with_images"] = classes_with_images

    if classes_with_images == 0:
        result["errors"].append(
            "No image files found in any sampled class subdirectory"
        )
        return False
    return True


def _validate_zarr_store(entries: list[Path], result: dict) -> bool:
    """Check zarr store structure: .zgroup/.zarray markers at top level or one level down."""
    markers = {".zgroup", ".zarray"}
    has_marker = any(f.name in markers for f in entries)
    if not has_marker:
        # Nested stores: markers may live one level down (e.g., data/.zarray)
        for sub in [e for e in entries if e.is_dir()][:20]:
            try:
                if any(f.name in markers
                       for f in itertools.islice(sub.iterdir(), 100)):
                    has_marker = True
                    break
            except (PermissionError, OSError):
                continue
    result["details"]["zarr_markers_found"] = has_marker
    if not has_marker:
        result["errors"].append(
            "No zarr markers (.zgroup/.zarray) found at top level or one level down"
        )
    return has_marker


# Dependency file → (manager, install_command_template) mapping
_DEPS_FILES: list[tuple[str, str, str]] = [
    # (filename, manager, command_template)
    ("environment.yml", "conda", "conda env update -f {deps_file}"),
    ("environment.yaml", "conda", "conda env update -f {deps_file}"),
    ("requirements.txt", "pip", "pip install -r {deps_file}"),
    ("setup.py", "pip", "pip install -e {project_root}"),
    ("setup.cfg", "pip", "pip install -e {project_root}"),
]


def bulk_install_command(
    project_root: str,
    env_manager: str,
    env_name: str | None = None,
) -> dict:
    """Generate the recommended bulk install command for a project.

    Checks common dependency files (`requirements.txt`, `environment.yml`,
    `pyproject.toml`) and builds the install command for *env_manager*. When
    *env_manager* is `"conda"` and *env_name* is given, conda commands include
    `-n <env_name>`. Returns {"has_deps_file": bool, "deps_file": <path>|None,
    "install_command": <command>|None, "manager": <env_manager>}.
    """
    root = Path(project_root).resolve()
    conda_n = f" -n {env_name}" if env_manager == "conda" and env_name else ""
    no_deps = {
        "has_deps_file": False,
        "deps_file": None,
        "install_command": None,
        "manager": env_manager,
    }

    # Manager-specific overrides
    if env_manager == "poetry":
        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            return {
                "has_deps_file": True,
                "deps_file": str(pyproject),
                "install_command": "poetry install",
                "manager": "poetry",
            }
        return no_deps

    if env_manager == "uv":
        # uv uses requirements.txt or pyproject.toml
        for name in ("requirements.txt", "pyproject.toml"):
            candidate = root / name
            if candidate.is_file():
                return {
                    "has_deps_file": True,
                    "deps_file": str(candidate),
                    "install_command": f"uv pip install -r {candidate}",
                    "manager": "uv",
                }
        return no_deps

    # For conda and pip, scan in priority order
    for filename, file_manager, cmd_template in _DEPS_FILES:
        candidate = root / filename
        if candidate.is_file():
            # If user says conda but only requirements.txt exists, use conda
            if env_manager == "conda" and file_manager == "pip":
                cmd = f"conda install -y{conda_n} --file {candidate}"
            elif env_manager == "pip" and file_manager == "conda":
                # pip user but environment.yml → skip, look for requirements.txt
                continue
            else:
                cmd = cmd_template.format(
                    deps_file=str(candidate),
                    project_root=str(root),
                )
                # Inject -n <env_name> for conda commands
                if env_manager == "conda" and conda_n and "conda " in cmd:
                    # Insert after "conda env update" or "conda install"
                    cmd = cmd.replace("conda env update", f"conda env update{conda_n}", 1)
                    cmd = cmd.replace("conda install", f"conda install{conda_n}", 1)
            return {
                "has_deps_file": True,
                "deps_file": str(candidate),
                "install_command": cmd,
                "manager": env_manager,
            }

    # Check pyproject.toml as last resort
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        cmd = f"pip install -e {root}"
        if env_manager == "conda" and env_name:
            cmd = f"conda run --no-banner -n {env_name} {cmd}"
        return {
            "has_deps_file": True,
            "deps_file": str(pyproject),
            "install_command": cmd,
            "manager": env_manager,
        }

    return no_deps


# PyTorch CUDA wheel index URLs keyed by CUDA major.minor
_TORCH_CUDA_URLS: dict[str, str] = {
    "11.8": "https://download.pytorch.org/whl/cu118",
    "12.1": "https://download.pytorch.org/whl/cu121",
    "12.4": "https://download.pytorch.org/whl/cu124",
    "12.6": "https://download.pytorch.org/whl/cu126",
    "12.8": "https://download.pytorch.org/whl/cu128",
}

_TORCH_PACKAGES = {"torch", "torchvision", "torchaudio"}
_TENSORFLOW_PACKAGES = {"tensorflow", "tf-nightly"}
_JAX_PACKAGES = {"jax", "jaxlib"}
_JAX_CUDA_EXTRAS: dict[str, str] = {
    "11": "cuda11_pip",
    "12": "cuda12",
}


def _detect_cuda_version() -> str | None:
    """Run `nvidia-smi` and parse the CUDA driver version.

    Returns a version string like `"12.1"` or `None` if no GPU found.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    # nvidia-smi output contains "CUDA Version: XX.Y"
    match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
    return match.group(1) if match else None


def _best_torch_cuda_tag(cuda_version: str) -> str | None:
    """Pick the highest PyTorch CUDA tag that doesn't exceed *cuda_version*."""
    if not cuda_version:
        return None
    try:
        driver_major, driver_minor = (int(x) for x in cuda_version.split("."))
    except (ValueError, AttributeError):
        return None

    best: str | None = None
    best_tuple: tuple[int, int] = (0, 0)
    for tag_ver in _TORCH_CUDA_URLS:
        tag_major, tag_minor = (int(x) for x in tag_ver.split("."))
        if (tag_major, tag_minor) <= (driver_major, driver_minor):
            if (tag_major, tag_minor) > best_tuple:
                best = tag_ver
                best_tuple = (tag_major, tag_minor)
    return best


def gpu_install_command(
    package: str,
    env_manager: str | None = None,
    env_name: str | None = None,
) -> dict:
    """Return the recommended install command for *package*, GPU-aware.

    torch/torchvision/torchaudio → CUDA-detected `--index-url` command;
    tensorflow → `tensorflow[and-cuda]` when a GPU is present; jax →
    `jax[cuda12]` or similar (jaxlib always installs plain — it's pulled in
    transitively by `jax[cudaXX]`, so a standalone jaxlib install is CPU-only);
    otherwise plain `pip install <package>`. When
    *env_manager* is `"conda"` and *env_name* is given, the pip command is
    wrapped with `conda run --no-banner -n <env_name>`. Returns {"package": ...,
    "gpu_detected": bool, "cuda_version": ...|None, "install_command": ...}.
    """
    cuda_version = _detect_cuda_version()
    gpu_detected = cuda_version is not None

    if package in _TORCH_PACKAGES:
        if gpu_detected:
            tag = _best_torch_cuda_tag(cuda_version)
            if tag:
                url = _TORCH_CUDA_URLS[tag]
                cmd = f"pip install {package} --index-url {url}"
            else:
                cmd = f"pip install {package}"
        else:
            cmd = f"pip install {package}"
        return {
            "package": package,
            "gpu_detected": gpu_detected,
            "cuda_version": cuda_version,
            "install_command": _wrap_for_conda(cmd, env_manager, env_name),
        }

    if package in _TENSORFLOW_PACKAGES:
        if gpu_detected:
            cmd = f"pip install {package}[and-cuda]"
        else:
            cmd = f"pip install {package}"
        return {
            "package": package,
            "gpu_detected": gpu_detected,
            "cuda_version": cuda_version,
            "install_command": _wrap_for_conda(cmd, env_manager, env_name),
        }

    if package in _JAX_PACKAGES:
        if gpu_detected:
            major = cuda_version.split(".")[0]
            extra = _JAX_CUDA_EXTRAS.get(major)
            if extra and package == "jax":
                cmd = f"pip install jax[{extra}]"
            else:
                # jaxlib is pulled in by jax[cudaXX]; standalone = CPU
                cmd = f"pip install {package}"
        else:
            cmd = f"pip install {package}"
        return {
            "package": package,
            "gpu_detected": gpu_detected,
            "cuda_version": cuda_version,
            "install_command": _wrap_for_conda(cmd, env_manager, env_name),
        }

    # Default: plain pip install
    cmd = f"pip install {package}"
    return {
        "package": package,
        "gpu_detected": gpu_detected,
        "cuda_version": cuda_version,
        "install_command": _wrap_for_conda(cmd, env_manager, env_name),
    }


def _wrap_for_conda(
    pip_cmd: str,
    env_manager: str | None,
    env_name: str | None,
) -> str:
    """Wrap a pip command for conda if *env_manager* is `"conda"`."""
    if env_manager == "conda" and env_name:
        return f"conda run --no-banner -n {env_name} {pip_cmd}"
    return pip_cmd


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  prerequisites_check.py scan-imports <project_root>\n"
            "  prerequisites_check.py check-packages '<json_list>' [python_executable] [host]\n"
            "  prerequisites_check.py detect-env <project_root>\n"
            "  prerequisites_check.py detect-format <training_script>\n"
            "  prerequisites_check.py detect-format-project <project_root> <training_script>\n"
            "  prerequisites_check.py validate-data <path> <format>\n"
            "  prerequisites_check.py gpu-install-cmd <package> [env_manager] [env_name]\n"
            "  prerequisites_check.py bulk-install-cmd <project_root> <env_manager> [env_name]"
        )
        sys.exit(1)

    action = sys.argv[1]

    if action == "scan-imports":
        if len(sys.argv) < 3:
            print("Usage: prerequisites_check.py scan-imports <project_root>")
            sys.exit(1)
        _print_json(scan_imports(sys.argv[2]))

    elif action == "check-packages":
        if len(sys.argv) < 3:
            print("Usage: prerequisites_check.py check-packages '<json_list>'")
            sys.exit(1)
        try:
            packages = json.loads(sys.argv[2])
        except json.JSONDecodeError:
            print(f"Error: invalid JSON '{sys.argv[2]}'")
            sys.exit(1)
        python_exec = sys.argv[3] if len(sys.argv) > 3 else "python3"
        remote_host = sys.argv[4] if len(sys.argv) > 4 else ""
        _print_json(
            check_missing_packages(packages, python_executable=python_exec, host=remote_host)
        )

    elif action == "detect-env":
        if len(sys.argv) < 3:
            print("Usage: prerequisites_check.py detect-env <project_root>")
            sys.exit(1)
        _print_json(detect_env_manager(sys.argv[2]))

    elif action == "detect-format":
        if len(sys.argv) < 3:
            print("Usage: prerequisites_check.py detect-format <training_script>")
            sys.exit(1)
        _print_json(detect_dataset_format(sys.argv[2]))

    elif action == "validate-data":
        if len(sys.argv) < 4:
            print("Usage: prerequisites_check.py validate-data <path> <format>")
            sys.exit(1)
        _print_json(validate_data_path(sys.argv[2], sys.argv[3]))

    elif action == "detect-format-project":
        if len(sys.argv) < 4:
            print("Usage: prerequisites_check.py detect-format-project <project_root> <training_script>")
            sys.exit(1)
        _print_json(detect_dataset_format_project(sys.argv[2], sys.argv[3]))

    elif action == "gpu-install-cmd":
        if len(sys.argv) < 3:
            print("Usage: prerequisites_check.py gpu-install-cmd <package> [env_manager] [env_name]")
            sys.exit(1)
        em = sys.argv[3] if len(sys.argv) > 3 else None
        en = sys.argv[4] if len(sys.argv) > 4 else None
        _print_json(gpu_install_command(sys.argv[2], env_manager=em, env_name=en))

    elif action == "bulk-install-cmd":
        if len(sys.argv) < 4:
            print("Usage: prerequisites_check.py bulk-install-cmd <project_root> <env_manager> [env_name]")
            sys.exit(1)
        en = sys.argv[4] if len(sys.argv) > 4 else None
        _print_json(bulk_install_command(sys.argv[2], sys.argv[3], env_name=en))

    else:
        print(f"Unknown action: {action}")
        sys.exit(1)
