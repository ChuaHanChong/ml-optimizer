#!/usr/bin/env python3
"""Round lifecycle management and completeness checking.

Manages the round directory lifecycle (rounds-manifest.json) and runs completeness
checks over baseline, prerequisites, manifest, round results, and proposed configs.

Usage:
    python3 round_manager.py <exp_root> create-round <type> [--branch <name>]  # Create a round dir + manifest entry
    python3 round_manager.py <exp_root> current-round                          # Show the latest round
    python3 round_manager.py <exp_root> register-experiment <round_dir> <exp_id>  # Register an exp-id in a round
    python3 round_manager.py <exp_root> close-round [--summary <text>]         # Close the current round
    python3 round_manager.py <exp_root> next-id                               # Globally unique next exp-id
    python3 round_manager.py <exp_root> check-baseline                        # Validate baseline.json completeness
    python3 round_manager.py <exp_root> check-prerequisites                   # Validate prerequisites.json
    python3 round_manager.py <exp_root> check-manifest                        # Validate implementation-manifest.json
    python3 round_manager.py <exp_root> check-round <round_dir>               # Validate a round's results + logs
    python3 round_manager.py <exp_root> check-proposals <round_dir>           # Validate proposed configs in a round

Valid round types: hp, evolved, research, stacked. Exit codes: 0 = success,
1 = error, 2 = check failure (check-* command found incomplete output).

Examples:
    python3 round_manager.py <exp_root> create-round hp
    python3 round_manager.py <exp_root> register-experiment round-1-hp exp-001
    python3 round_manager.py <exp_root> check-round round-1-hp
"""

import fcntl
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Cross-import from same directory
sys.path.insert(0, str(Path(__file__).parent))
from schema_validator import (
    validate_baseline,
    validate_hp_proposal,
    validate_manifest,
    validate_prerequisites,
    validate_result,
)

VALID_ROUND_TYPES = ["hp", "evolved", "research", "stacked"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, data) -> None:
    """Write JSON atomically via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _manifest_path(exp_root: str) -> Path:
    return Path(exp_root) / "results" / "rounds-manifest.json"


def _load_manifest(exp_root: str) -> dict:
    """Load rounds-manifest.json, return default if not found."""
    path = _manifest_path(exp_root)
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"rounds": [], "current_round": 0, "total_experiments": 0}


def _scan_exp_ids(results_dir: Path) -> list[int]:
    """Scan for all exp-NNN IDs across flat and round directories."""
    exp_pattern = re.compile(r"^exp-(\d+)\.json$")
    nums: list[int] = []

    # Flat results
    if results_dir.is_dir():
        for f in results_dir.iterdir():
            if f.is_file():
                m = exp_pattern.match(f.name)
                if m:
                    nums.append(int(m.group(1)))

    # Round directories
    if results_dir.is_dir():
        for rdir in results_dir.iterdir():
            if rdir.is_dir() and rdir.name.startswith("round-"):
                for f in rdir.iterdir():
                    if f.is_file():
                        m = exp_pattern.match(f.name)
                        if m:
                            nums.append(int(m.group(1)))

    return nums


# ---------------------------------------------------------------------------
# Round lifecycle
# ---------------------------------------------------------------------------

def create_round(exp_root: str, round_type: str, branch: str | None = None) -> dict:
    """Create a new round directory and update manifest.

    Creates <exp_root>/results/round-N-<type>/ and the matching
    proposed-configs/round-N-<type>/ directory.  Updates
    rounds-manifest.json atomically with file locking.
    """
    if round_type not in VALID_ROUND_TYPES:
        return {"error": f"Invalid round type '{round_type}': must be one of {VALID_ROUND_TYPES}"}

    path = _manifest_path(exp_root)
    lock_path = path.with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            manifest = _load_manifest(exp_root)
            round_id = manifest["current_round"] + 1
            dir_name = f"round-{round_id}-{round_type}"

            # Create round directories (results, proposed-configs, logs, scripts, artifacts)
            root = Path(exp_root)
            (root / "results" / dir_name).mkdir(parents=True, exist_ok=True)
            (root / "proposed-configs" / dir_name).mkdir(parents=True, exist_ok=True)
            (root / "logs" / dir_name).mkdir(parents=True, exist_ok=True)
            (root / "scripts" / dir_name).mkdir(parents=True, exist_ok=True)
            (root / "artifacts" / dir_name).mkdir(parents=True, exist_ok=True)
            round_path = root / "results" / dir_name

            entry = {
                "id": round_id,
                "dir": dir_name,
                "type": round_type,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "experiments": [],
                "code_branch": branch,
                "summary": None,
                "closed_at": None,
            }

            manifest["rounds"].append(entry)
            manifest["current_round"] = round_id
            _atomic_write_json(path, manifest)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)

    return {
        "id": round_id,
        "dir": dir_name,
        "type": round_type,
        "path": str(round_path),
    }


def current_round(exp_root: str) -> dict | None:
    """Get the current (latest) round info from manifest.

    Returns None if no rounds exist.
    """
    manifest = _load_manifest(exp_root)
    if not manifest["rounds"]:
        return None
    return manifest["rounds"][-1]


def register_experiment(exp_root: str, round_dir: str, exp_id: str) -> dict:
    """Register an experiment ID in the specified round.

    Updates the round's experiments list in the manifest.
    """
    path = _manifest_path(exp_root)
    lock_path = path.with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            manifest = _load_manifest(exp_root)
            found = False
            for rnd in manifest["rounds"]:
                if rnd["dir"] == round_dir:
                    if exp_id not in rnd["experiments"]:
                        rnd["experiments"].append(exp_id)
                        manifest["total_experiments"] = sum(
                            len(r["experiments"]) for r in manifest["rounds"]
                        )
                    found = True
                    break

            if not found:
                return {"error": f"Round directory '{round_dir}' not found in manifest"}

            _atomic_write_json(path, manifest)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)

    return {"registered": True, "round_dir": round_dir, "exp_id": exp_id}


def close_round(exp_root: str, summary: str | None = None) -> dict:
    """Mark the current round as closed with optional summary."""
    path = _manifest_path(exp_root)
    lock_path = path.with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            manifest = _load_manifest(exp_root)
            if not manifest["rounds"]:
                return {"error": "No rounds to close"}

            last = manifest["rounds"][-1]
            last["closed_at"] = datetime.now(timezone.utc).isoformat()
            if summary:
                last["summary"] = summary

            _atomic_write_json(path, manifest)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)

    return {"closed": True, "round_id": last["id"], "dir": last["dir"]}


def next_experiment_id(exp_root: str) -> str:
    """Generate globally unique next exp-id by scanning all round directories."""
    results_dir = Path(exp_root) / "results"
    nums = _scan_exp_ids(results_dir)
    if not nums:
        return "exp-001"
    return f"exp-{max(nums) + 1:03d}"


# ---------------------------------------------------------------------------
# Completeness checks
# ---------------------------------------------------------------------------

def _check_file(exp_root: str, filename: str, validator) -> dict:
    """Check a file exists and passes schema validation."""
    path = Path(exp_root) / "results" / filename
    if not path.is_file():
        return {"complete": False, "errors": [f"{filename} does not exist"]}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return {"complete": False, "errors": [f"{filename}: {exc}"]}
    result = validator(data)
    return {
        "complete": result["valid"],
        "errors": result["errors"],
    }


def check_baseline(exp_root: str) -> dict:
    """Check baseline.json exists and is schema-valid."""
    return _check_file(exp_root, "baseline.json", validate_baseline)


def check_prerequisites(exp_root: str) -> dict:
    """Check prerequisites.json exists and is schema-valid."""
    return _check_file(exp_root, "prerequisites.json", validate_prerequisites)


def check_manifest(exp_root: str) -> dict:
    """Check implementation-manifest.json exists and is schema-valid."""
    return _check_file(exp_root, "implementation-manifest.json", validate_manifest)


def check_round(exp_root: str, round_dir: str) -> dict:
    """Check completeness of a round directory.

    For each exp-*.json in the round directory, validates the result
    schema, checks for terminal status, and verifies the log file exists.
    """
    root = Path(exp_root)
    rpath = root / "results" / round_dir
    if not rpath.is_dir():
        return {
            "complete": False, "total": 0, "valid": 0, "terminal": 0,
            "missing_logs": [], "invalid": [], "non_terminal": [],
            "errors": [f"Round directory '{round_dir}' does not exist"],
        }

    exp_pattern = re.compile(r"^exp-\d+\.json$")
    terminal_statuses = {"completed", "failed", "diverged", "timeout"}

    total = 0
    valid_count = 0
    terminal_count = 0
    missing_logs: list[str] = []
    invalid: list[dict] = []
    non_terminal: list[str] = []

    for f in sorted(rpath.iterdir()):
        if not f.is_file() or not exp_pattern.match(f.name):
            continue
        total += 1
        exp_id = f.stem

        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            invalid.append({"exp_id": exp_id, "errors": [str(exc)]})
            continue

        result = validate_result(data)
        if not result["valid"]:
            invalid.append({"exp_id": exp_id, "errors": result["errors"]})
        else:
            valid_count += 1

        status = data.get("status", "")
        if status in terminal_statuses:
            terminal_count += 1
        else:
            non_terminal.append(exp_id)

        # Check log file exists (round-based path)
        log_path = root / "logs" / round_dir / exp_id / "train.log"
        if not log_path.is_file():
            missing_logs.append(exp_id)

    complete = (total > 0 and not invalid and not non_terminal)
    return {
        "complete": complete,
        "total": total,
        "valid": valid_count,
        "terminal": terminal_count,
        "missing_logs": missing_logs,
        "invalid": invalid,
        "non_terminal": non_terminal,
    }


def check_proposals(exp_root: str, round_dir: str) -> dict:
    """Check proposed configs in proposed-configs/<round_dir>/.

    For each exp-*.json, checks that exp_id and config fields exist
    and that exp_id matches the filename.
    """
    root = Path(exp_root)
    ppath = root / "proposed-configs" / round_dir
    if not ppath.is_dir():
        return {
            "complete": False, "total": 0, "valid": 0, "invalid": [],
            "errors": [f"Proposals directory '{round_dir}' does not exist"],
        }

    exp_pattern = re.compile(r"^exp-\d+\.json$")
    total = 0
    valid_count = 0
    invalid: list[dict] = []

    for f in sorted(ppath.iterdir()):
        if not f.is_file() or not exp_pattern.match(f.name):
            continue
        total += 1
        expected_id = f.stem

        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            invalid.append({"exp_id": expected_id, "errors": [str(exc)]})
            continue

        result = validate_hp_proposal(data)
        errors = list(result["errors"])

        # Check exp_id matches filename
        actual_id = data.get("exp_id", "")
        if actual_id != expected_id:
            errors.append(
                f"exp_id '{actual_id}' does not match filename '{expected_id}'"
            )

        if errors:
            invalid.append({"exp_id": expected_id, "errors": errors})
        else:
            valid_count += 1

    complete = (total > 0 and not invalid)
    return {
        "complete": complete,
        "total": total,
        "valid": valid_count,
        "invalid": invalid,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: round_manager.py <exp_root> <command> [args...]")
        print("Commands:")
        print("  create-round <type> [--branch <name>]")
        print("  current-round")
        print("  register-experiment <round_dir> <exp_id>")
        print("  close-round [--summary <text>]")
        print("  next-id")
        print("  check-baseline")
        print("  check-prerequisites")
        print("  check-manifest")
        print("  check-round <round_dir>")
        print("  check-proposals <round_dir>")
        sys.exit(1)

    exp_root = sys.argv[1]
    command = sys.argv[2]

    if command == "create-round":
        if len(sys.argv) < 4:
            print("Usage: round_manager.py <exp_root> create-round <type> [--branch <name>]")
            sys.exit(1)
        rtype = sys.argv[3]
        branch = None
        i = 4
        while i < len(sys.argv):
            if sys.argv[i] == "--branch" and i + 1 < len(sys.argv):
                branch = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        output = create_round(exp_root, rtype, branch=branch)

    elif command == "current-round":
        output = current_round(exp_root)
        if output is None:
            output = {"error": "No rounds exist"}

    elif command == "register-experiment":
        if len(sys.argv) < 5:
            print("Usage: round_manager.py <exp_root> register-experiment <round_dir> <exp_id>")
            sys.exit(1)
        output = register_experiment(exp_root, sys.argv[3], sys.argv[4])

    elif command == "close-round":
        summary = None
        if "--summary" in sys.argv:
            idx = sys.argv.index("--summary")
            if idx + 1 < len(sys.argv):
                summary = sys.argv[idx + 1]
        output = close_round(exp_root, summary=summary)

    elif command == "next-id":
        output = {"exp_id": next_experiment_id(exp_root)}

    elif command == "check-baseline":
        output = check_baseline(exp_root)

    elif command == "check-prerequisites":
        output = check_prerequisites(exp_root)

    elif command == "check-manifest":
        output = check_manifest(exp_root)

    elif command == "check-round":
        if len(sys.argv) < 4:
            print("Usage: round_manager.py <exp_root> check-round <round_dir>")
            sys.exit(1)
        output = check_round(exp_root, sys.argv[3])

    elif command == "check-proposals":
        if len(sys.argv) < 4:
            print("Usage: round_manager.py <exp_root> check-proposals <round_dir>")
            sys.exit(1)
        output = check_proposals(exp_root, sys.argv[3])

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

    print(json.dumps(output, indent=2))

    # Exit codes: 0=success, 1=error, 2=check failure
    if "error" in output:
        sys.exit(1)
    elif command.startswith("check-") and not output.get("complete", True):
        sys.exit(2)
    sys.exit(0)
