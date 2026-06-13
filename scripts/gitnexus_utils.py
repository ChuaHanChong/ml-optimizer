#!/usr/bin/env python3
"""Helpers for GitNexus code-graph indexing: availability, MCP registration, and indexing.

Wraps the `gitnexus` CLI to build/check the `.gitnexus` code knowledge graph
that agents query — via the gitnexus MCP server — instead of grep. GitNexus is a
hard prerequisite: `index()` never raises, reporting failure via its return dict
so callers can halt with install guidance rather than fall back.

Indexing uses `gitnexus analyze <path> --index-only`. `--index-only` keeps the
index pure: it does NOT inject a "GitNexus — Code Intelligence" section into the
repo's CLAUDE.md / AGENTS.md, nor install skill files under `.claude/` — so the
indexed project (and any worktree) is never contaminated. On success `index()`
also adds `.gitnexus/` to the repo's git exclude so the artifact never shows up
as untracked. `gitnexus analyze` is incremental; `index()` skips re-indexing an
already-indexed path unless `force=True` (which adds `--force`).

Querying the graph is MCP-only (the agents use the `mcp__gitnexus__*` tools), so
`mcp_registered()` lets Phase 2 warn early when the CLI is installed but its MCP
server was never registered with Claude Code.

Usage:
    python3 gitnexus_utils.py available            # Check if the gitnexus CLI is on PATH
    python3 gitnexus_utils.py mcp-registered       # Check if the gitnexus MCP server is registered with Claude Code
    python3 gitnexus_utils.py require              # Hard prerequisite check (exits nonzero if CLI absent)
    python3 gitnexus_utils.py index <path>         # Index a repo (gitnexus analyze <path> --index-only)
    python3 gitnexus_utils.py index <path> --force # Force a full re-index
    python3 gitnexus_utils.py is-indexed <path>    # Check for an existing .gitnexus index

`require` and `index` exit nonzero on failure; `available`, `mcp-registered`,
and `is-indexed` always exit 0 and report status in their JSON output.

Examples:
    python3 gitnexus_utils.py available
    python3 gitnexus_utils.py mcp-registered
    python3 gitnexus_utils.py require
    python3 gitnexus_utils.py index /path/to/repo
    python3 gitnexus_utils.py index /path/to/repo --force
    python3 gitnexus_utils.py is-indexed /path/to/repo
"""

import json
import os
import shutil
import subprocess
import sys


# Canonical install/repair guidance. `gitnexus setup` auto-registers the MCP
# server for Claude Code (the official path); the manual `claude mcp add` form is
# the fallback when `setup` cannot be used.
INSTALL_GUIDANCE = [
    "npm install -g gitnexus",
    "gitnexus setup",
]

_USAGE_LINES = [
    "Usage:",
    "  gitnexus_utils.py available            — check if gitnexus CLI is installed",
    "  gitnexus_utils.py mcp-registered       — check if the gitnexus MCP server is registered with Claude Code",
    "  gitnexus_utils.py require              — hard prerequisite check (exits nonzero if CLI absent)",
    "  gitnexus_utils.py index <path> [--force]  — index a repo (gitnexus analyze --index-only)",
    "  gitnexus_utils.py is-indexed <path>    — check for an existing .gitnexus index",
]


def _print_usage() -> None:
    """Print the CLI usage block."""
    print("\n".join(_USAGE_LINES))


def available() -> bool:
    """Return True if the `gitnexus` CLI is on PATH."""
    return shutil.which("gitnexus") is not None


def mcp_registered():
    """Return whether the gitnexus MCP server is registered with Claude Code.

    Best-effort: runs `claude mcp get gitnexus`. Returns True if the server is
    registered, False if `claude` runs but the server is not registered, and
    None if the check cannot run (the `claude` CLI is absent, errors, or times
    out). Never raises. Querying the code graph is MCP-only, so this lets Phase 2
    warn early when the CLI is present but its MCP server was never registered.
    """
    if shutil.which("claude") is None:
        return None
    try:
        proc = subprocess.run(
            ["claude", "mcp", "get", "gitnexus"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return None
    if proc.returncode == 0 and "gitnexus" in (proc.stdout or ""):
        return True
    return False


def is_indexed(path: str) -> bool:
    """Return True if *path* already has a GitNexus index (a `.gitnexus` directory)."""
    return os.path.isdir(os.path.join(path, ".gitnexus"))


def _exclude_gitnexus_artifact(path: str) -> None:
    """Best-effort: add `.gitnexus/` to the repo's git exclude file.

    Keeps the index artifact out of `git status` so it is never accidentally
    committed. Resolves the correct exclude file via `git rev-parse --git-path`
    (which handles worktrees and the shared git dir). Never raises and silently
    no-ops for non-git paths or when the entry already exists.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", path, "rev-parse", "--git-path", "info/exclude"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return
        rel = (proc.stdout or "").strip()
        if not rel:
            return
        exclude_path = rel if os.path.isabs(rel) else os.path.join(path, rel)
        os.makedirs(os.path.dirname(exclude_path), exist_ok=True)
        existing = ""
        if os.path.exists(exclude_path):
            with open(exclude_path, "r", encoding="utf-8") as fh:
                existing = fh.read()
        if ".gitnexus/" in existing:
            return
        prefix = "" if existing == "" or existing.endswith("\n") else "\n"
        with open(exclude_path, "a", encoding="utf-8") as fh:
            fh.write(f"{prefix}.gitnexus/\n")
    except Exception:
        return


def index(path: str, timeout: int = 600, force: bool = False) -> dict:
    """Index a repository with `gitnexus analyze <path> --index-only`.

    Writes the code knowledge graph to ``<path>/.gitnexus``. ``--index-only``
    keeps the index pure — it does NOT inject a GitNexus section into the repo's
    CLAUDE.md / AGENTS.md or install skill files under ``.claude/``. On success,
    ``.gitnexus/`` is added to the repo's git exclude so it never appears as
    untracked.

    Skips re-indexing when *path* already has a ``.gitnexus`` index unless
    *force* is True. (``gitnexus analyze`` is itself incremental, but skipping
    avoids even spawning the subprocess.) ``force=True`` forces a full re-index
    (``--force``).

    Never raises: any failure (CLI missing, timeout, non-zero exit) is reported
    via the returned dict so callers can surface a clean HARD ERROR (with
    install/repair guidance) rather than crashing. ``success: False`` is NOT a
    signal to fall back — it is a signal to halt.

    Returns {"success": bool, "graph_path": str, "already_indexed": bool,
    "output": str|None, "error": str|None}.
    """
    graph_path = os.path.join(path, ".gitnexus")

    if not available():
        return {
            "success": False,
            "graph_path": graph_path,
            "already_indexed": False,
            "output": None,
            "error": "gitnexus CLI not found on PATH",
        }

    if not force and is_indexed(path):
        return {
            "success": True,
            "graph_path": graph_path,
            "already_indexed": True,
            "output": None,
            "error": None,
        }

    cmd = ["gitnexus", "analyze", path, "--index-only"]
    if force:
        cmd.append("--force")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except FileNotFoundError:
        return {
            "success": False,
            "graph_path": graph_path,
            "already_indexed": False,
            "output": None,
            "error": "gitnexus CLI not found on PATH",
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "graph_path": graph_path,
            "already_indexed": False,
            "output": None,
            "error": f"gitnexus analyze timed out after {timeout} seconds",
        }
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "graph_path": graph_path,
            "already_indexed": False,
            "output": None,
            "error": (e.stderr or e.stdout or str(e)).strip(),
        }

    _exclude_gitnexus_artifact(path)

    return {
        "success": True,
        "graph_path": graph_path,
        "already_indexed": False,
        "output": (result.stdout or "").strip(),
        "error": None,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "available":
        print(json.dumps({"available": available()}, indent=2))
        sys.exit(0)
    elif mode == "mcp-registered":
        print(json.dumps({"registered": mcp_registered()}, indent=2))
        sys.exit(0)
    elif mode == "require":
        ok = available()
        payload: dict[str, object] = {
            "available": ok,
            "required": True,
            "mcp_registered": mcp_registered(),
        }
        if not ok:
            payload["error"] = "GitNexus is a HARD PREREQUISITE and is not installed."
            payload["install"] = list(INSTALL_GUIDANCE)
        print(json.dumps(payload, indent=2))
        sys.exit(0 if ok else 1)
    elif mode == "index":
        positional = [a for a in sys.argv[2:] if not a.startswith("--")]
        if not positional:
            print("Usage: gitnexus_utils.py index <path> [--force]")
            sys.exit(1)
        force = "--force" in sys.argv[2:]
        result = index(positional[0], force=force)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["success"] else 1)
    elif mode == "is-indexed":
        if len(sys.argv) < 3:
            print("Usage: gitnexus_utils.py is-indexed <path>")
            sys.exit(1)
        print(json.dumps({"indexed": is_indexed(sys.argv[2])}, indent=2))
        sys.exit(0)
    else:
        print(f"Error: unknown mode '{mode}'")
        _print_usage()
        sys.exit(1)
