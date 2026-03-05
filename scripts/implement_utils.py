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
    return slug.strip("-")


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
        r"^### Proposal (\d+):\s*(.+?)(?:\s*\(Priority:.*?\))?\s*$",
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
            if line.strip().startswith("- ") or line.strip().startswith("* "):
                # Look for file paths (patterns like path/to/file.py)
                path_match = re.search(r"`([^`]+\.\w+)`", line)
                if path_match:
                    files.append(path_match.group(1))
            elif line.strip().startswith("**") or line.strip().startswith("###"):
                break
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
    if len(sys.argv) < 3:
        print("Usage: implement_utils.py <findings_path> <selected_json>")
        print('  selected_json: JSON array of 1-based indices, e.g. "[1,3]"')
        sys.exit(1)
    findings_path = sys.argv[1]
    selected = json.loads(sys.argv[2])
    proposals = parse_research_proposals(findings_path, selected)
    conflicts = detect_conflicts(proposals)
    print(json.dumps({"proposals": proposals, "conflicts": conflicts}, indent=2))
