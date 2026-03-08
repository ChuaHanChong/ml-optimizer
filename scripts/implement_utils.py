#!/usr/bin/env python3
"""Helpers for implementing research proposals: branch management, validation, and manifest writing."""

import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
from pathlib import Path


def slugify(name: str) -> str:
    """Convert a proposal name to a URL-safe slug.

    "Perceptual Loss Function" -> "perceptual-loss-function"
    """
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug or "proposal"


def is_git_repo(project_root: str) -> bool:
    """Check if the project root contains a .git directory."""
    return (Path(project_root) / ".git").is_dir()


def get_current_branch(project_root: str) -> str:
    """Get the current git branch name."""
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def create_proposal_branch(project_root: str, slug: str, base_branch: str, prefix: str = "ml-opt") -> str:
    """Create a new branch for a proposal and check it out.

    Returns the branch name (<prefix>/<slug>).
    """
    branch_name = f"{prefix}/{slug}"
    subprocess.run(
        ["git", "checkout", base_branch],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "-b", branch_name],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return branch_name


def backup_files(files: list[str], backup_dir: str, project_root: str = "") -> dict:
    """Non-git fallback: copy original files to a backup directory.

    When *project_root* is provided and a file resides inside it, the
    relative directory structure is preserved under *backup_dir* so that
    files with identical names in different subdirectories do not collide.

    Returns a mapping of {original_path: backup_path}.
    """
    Path(backup_dir).mkdir(parents=True, exist_ok=True)
    mapping = {}
    for filepath in files:
        src = Path(filepath)
        if src.exists():
            if project_root and src.is_relative_to(Path(project_root)):
                dest = Path(backup_dir) / src.relative_to(project_root)
            else:
                dest = Path(backup_dir) / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dest))
            mapping[str(src)] = str(dest)
    return mapping


def validate_syntax(files: list[str]) -> list[dict]:
    """Run py_compile on each file. Returns list of {file, passed, error}."""
    results = []
    for filepath in files:
        try:
            py_compile.compile(filepath, doraise=True)
            results.append({"file": filepath, "passed": True, "error": None})
        except py_compile.PyCompileError as e:
            results.append({"file": filepath, "passed": False, "error": str(e)})
    return results


def validate_imports(module_path: str, project_root: str) -> dict:
    """Attempt to import a module by running a subprocess.

    Returns {passed: bool, error: str|None}.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "import importlib.util, sys; spec = importlib.util.spec_from_file_location('mod', sys.argv[1]); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)",
         module_path],
        cwd=project_root,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if result.returncode == 0:
        return {"passed": True, "error": None}
    return {"passed": False, "error": result.stderr.strip()}


def parse_research_proposals(findings_path: str, selected_indices: list[int] | None = None) -> list[dict]:
    """Parse research-findings.md and extract structured proposals.

    Each proposal is extracted from a '### Proposal N: ...' block.
    If selected_indices is provided, only return those proposals (1-based).
    """
    text = Path(findings_path).read_text()

    # Split on proposal headers
    proposal_pattern = re.compile(
        r"^#{2,3}\s+Proposal\s+(\d+):\s*(.+?)(?:\s*\(Priority:.*?\))?\s*$",
        re.MULTILINE,
    )
    matches = list(proposal_pattern.finditer(text))

    proposals = []
    for i, match in enumerate(matches):
        index = int(match.group(1))
        name = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        proposal = {
            "index": index,
            "name": name,
            "slug": slugify(name),
            "body": body,
            "files_to_modify": _extract_files(body),
            "complexity": _extract_field(body, "Complexity"),
            "implementation_steps": _extract_steps(body),
            "implementation_strategy": _extract_field(body, "Implementation strategy") or "from_scratch",
            "reference_repo": _extract_field(body, "Reference repo") or "",
            "reference_files": _extract_reference_files(body),
        }
        proposals.append(proposal)

    if selected_indices is not None:
        proposals = [p for p in proposals if p["index"] in selected_indices]

    return proposals


def _extract_files(body: str) -> list[str]:
    """Extract file paths from 'What to change' section."""
    files = []
    in_section = False
    for line in body.splitlines():
        if "what to change" in line.lower():
            in_section = True
            continue
        if in_section:
            stripped = line.strip()
            if stripped.startswith("**") or stripped.startswith("###"):
                break
            if re.match(r"^-\s+\*\*", stripped):
                break  # New field header like "- **Expected improvement:**"
            if stripped.startswith("- ") or stripped.startswith("* "):
                path_match = re.search(r"`([^`]+\.\w+)`", line)
                if path_match:
                    files.append(path_match.group(1))
    return files


def _extract_field(body: str, field_name: str) -> str:
    """Extract a **Field:** value from the proposal body."""
    pattern = re.compile(rf"\*\*{field_name}:\*\*\s*(.+)", re.IGNORECASE)
    match = pattern.search(body)
    return match.group(1).strip() if match else ""


def _extract_steps(body: str) -> list[str]:
    """Extract implementation steps from numbered list."""
    steps = []
    in_steps = False
    for line in body.splitlines():
        if "implementation steps" in line.lower():
            in_steps = True
            continue
        if in_steps:
            step_match = re.match(r"\s*\d+\.\s+(.+)", line)
            if step_match:
                steps.append(step_match.group(1).strip())
            elif line.strip().startswith("**") or line.strip().startswith("###"):
                break
    return steps


def _extract_reference_files(body: str) -> list[str]:
    """Extract reference file paths from '**Reference files:**' field.

    Handles backtick-delimited and comma-separated paths:
      **Reference files:** `path/to/model.py`, `path/to/loss.py`
      **Reference files:** path/to/model.py, path/to/loss.py
    """
    pattern = re.compile(r"\*\*Reference files:\*\*\s*(.+)", re.IGNORECASE)
    match = pattern.search(body)
    if not match:
        return []
    raw = match.group(1).strip()
    # Extract backtick-delimited paths first
    backtick_paths = re.findall(r"`([^`]+)`", raw)
    if backtick_paths:
        return [p.strip() for p in backtick_paths if p.strip()]
    # Fall back to comma-separated
    return [p.strip() for p in raw.split(",") if p.strip()]


def clone_reference_repo(repo_url: str, dest_dir: str, shallow: bool = True) -> dict:
    """Clone a reference repository for code adaptation.

    Only allows GitHub and GitLab URLs for safety.
    Returns {"success": True, "path": dest_dir} or {"success": False, "error": "..."}.
    """
    if not re.match(r"https://(github\.com|gitlab\.com)/", repo_url):
        return {"success": False, "error": f"URL must be https://github.com/ or https://gitlab.com/: {repo_url}"}

    cmd = ["git", "clone"]
    if shallow:
        cmd.extend(["--depth", "1"])
    cmd.extend([repo_url, dest_dir])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return {"success": False, "error": result.stderr.strip()}
        return {"success": True, "path": dest_dir}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Clone timed out after 120 seconds"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def analyze_reference_structure(repo_path: str) -> dict:
    """Analyze a cloned reference repository's structure.

    Returns framework detection, file categorization, requirements, and README summary.
    """
    repo = Path(repo_path)
    skip_dirs = {".git", "__pycache__", "node_modules", ".eggs", "dist", "build"}
    skip_files = {"setup.py", "setup.cfg", "conftest.py"}

    python_files = []
    model_files = []
    training_files = []
    framework_hints: dict[str, int] = {}

    framework_patterns = {
        "pytorch": [r"import torch", r"from torch"],
        "tensorflow": [r"import tensorflow", r"from tensorflow", r"from keras"],
        "jax": [r"import jax", r"from jax", r"from flax"],
        "lightning": [r"import lightning", r"import pytorch_lightning"],
        "transformers": [r"from transformers"],
        "sklearn": [r"import sklearn", r"from sklearn"],
        "xgboost": [r"import xgboost", r"from xgboost"],
        "lightgbm": [r"import lightgbm", r"from lightgbm"],
    }
    model_patterns = [
        r"class\s+\w+\(.*(?:nn\.Module|Model|LightningModule)",
        r"class\s+\w+Model",
        r"class\s+\w+\(.*(?:BaseEstimator|ClassifierMixin|RegressorMixin|TransformerMixin)",
    ]
    training_patterns = ["train", "main", "run"]

    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            if fname in skip_files or fname.startswith("test_"):
                continue

            rel_path = os.path.relpath(os.path.join(dirpath, fname), repo_path)
            # Skip files in docs/ or test directories
            if rel_path.startswith("docs/") or "/tests/" in rel_path or "/test/" in rel_path:
                continue

            python_files.append(rel_path)

            try:
                content = Path(os.path.join(dirpath, fname)).read_text(errors="ignore")
            except OSError:
                continue

            # Detect framework
            for fw, patterns in framework_patterns.items():
                for pat in patterns:
                    if re.search(pat, content):
                        framework_hints[fw] = framework_hints.get(fw, 0) + 1

            # Detect model files
            for pat in model_patterns:
                if re.search(pat, content):
                    model_files.append(rel_path)
                    break

            # Detect training files
            base = fname.lower().replace(".py", "")
            if any(tp in base for tp in training_patterns):
                training_files.append(rel_path)

    # Determine primary framework
    framework = max(framework_hints, key=framework_hints.get) if framework_hints else "unknown"

    # Read requirements.txt
    requirements: list[str] = []
    req_path = repo / "requirements.txt"
    if req_path.exists():
        try:
            requirements = [
                line.strip() for line in req_path.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        except OSError:
            pass

    # Read README summary
    readme_summary = ""
    for readme_name in ["README.md", "README.rst", "README.txt", "README"]:
        readme_path = repo / readme_name
        if readme_path.exists():
            try:
                readme_summary = readme_path.read_text(errors="ignore")[:500]
            except OSError:
                pass
            break

    return {
        "framework": framework,
        "python_files": sorted(python_files),
        "model_files": sorted(model_files),
        "training_files": sorted(training_files),
        "requirements": requirements,
        "readme_summary": readme_summary,
    }


def cleanup_reference_repo(repo_path: str) -> bool:
    """Remove a cloned reference repository.

    Returns True if cleanup succeeded, False otherwise.
    """
    try:
        shutil.rmtree(repo_path)
        return True
    except Exception:
        return False


def detect_conflicts(proposals: list[dict]) -> list[dict]:
    """Check for overlapping files_to_modify across proposals.

    Returns list of {file, proposal_indices} for files touched by multiple proposals.
    """
    file_map: dict[str, list[int]] = {}
    for p in proposals:
        for f in p.get("files_to_modify", []):
            file_map.setdefault(f, []).append(p["index"])

    conflicts = []
    for filepath, indices in file_map.items():
        if len(indices) > 1:
            conflicts.append({"file": filepath, "proposal_indices": indices})
    return conflicts


def write_manifest(path: str, data: dict) -> str:
    """Write implementation-manifest.json."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))
    return str(p)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  implement_utils.py <findings_path> <selected_json>  — parse proposals")
        print("  implement_utils.py clone <repo_url> <dest_dir>      — clone reference repo")
        print("  implement_utils.py analyze <repo_path>              — analyze repo structure")
        sys.exit(1)

    if sys.argv[1] == "clone":
        if len(sys.argv) < 4:
            print("Usage: implement_utils.py clone <repo_url> <dest_dir>")
            sys.exit(1)
        result = clone_reference_repo(sys.argv[2], sys.argv[3])
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["success"] else 1)
    elif sys.argv[1] == "analyze":
        if len(sys.argv) < 3:
            print("Usage: implement_utils.py analyze <repo_path>")
            sys.exit(1)
        result = analyze_reference_structure(sys.argv[2])
        print(json.dumps(result, indent=2))
    else:
        # Original behavior: parse proposals
        if len(sys.argv) < 3:
            print("Usage: implement_utils.py <findings_path> <selected_json>")
            print('  selected_json: JSON array of 1-based indices, e.g. "[1,3]"')
            sys.exit(1)
        findings_path = sys.argv[1]
        try:
            selected = json.loads(sys.argv[2])
        except json.JSONDecodeError:
            print(f"Error: invalid JSON '{sys.argv[2]}'")
            print('Usage: implement_utils.py <findings_path> <selected_json>')
            print('  selected_json: JSON array of 1-based indices, e.g. "[1,3]"')
            sys.exit(1)
        proposals = parse_research_proposals(findings_path, selected)
        conflicts = detect_conflicts(proposals)
        print(json.dumps({"proposals": proposals, "conflicts": conflicts}, indent=2))
