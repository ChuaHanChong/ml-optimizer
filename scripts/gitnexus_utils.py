#!/usr/bin/env python3
"""GitNexus code-graph helpers: CLI availability, MCP registration, and indexing.

Wraps the `gitnexus` CLI to build/check the `.gitnexus` code knowledge graph that
agents query via the gitnexus MCP server (never grep). GitNexus is a hard
prerequisite, so `index()` never raises — it reports failure via its return dict
so callers halt with install guidance rather than fall back.

Indexing runs `gitnexus analyze <path> --index-only`: `--index-only` keeps the
index pure (no GitNexus section injected into the repo's CLAUDE.md/AGENTS.md, no
`.claude/` skill files), so the indexed repo/worktree is never contaminated. On
success `index()` adds `.gitnexus/` to the repo's git exclude; re-indexing an
already-indexed path is skipped unless force=True (adds `--force`).

Querying the graph is MCP-only, so `mcp_registered()` lets Phase 2 warn when the
CLI is installed but its MCP server was never registered with Claude Code.

CLI: available | mcp-registered | require | index <path> [--force] | is-indexed <path>
require/index exit nonzero on failure; available/mcp-registered/is-indexed always exit 0.
"""

import json
import os
import shutil
import subprocess
import sys


# `gitnexus setup` auto-registers the MCP server (official path); manual
# `claude mcp add` is the fallback when setup can't be used.
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
    """Whether the gitnexus MCP server is registered with Claude Code.

    Best-effort `claude mcp get gitnexus`: True if registered, False if `claude`
    runs but the server isn't, None if the check can't run (claude absent/errors/
    times out). Never raises.
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
    """Best-effort: add `.gitnexus/` to the repo's git exclude so the index never
    shows in `git status`. Resolves the exclude file via `git rev-parse
    --git-path` (handles worktrees/shared git dir). Never raises; no-ops for
    non-git paths or when the entry already exists.
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
    """Index a repo with `gitnexus analyze <path> --index-only` (graph at
    `<path>/.gitnexus`). `--index-only` keeps the index pure (no CLAUDE.md/
    AGENTS.md injection, no `.claude/` skills). On success adds `.gitnexus/` to the
    git exclude; skips re-indexing an already-indexed path unless *force* (adds
    `--force`).

    Never raises: any failure (CLI missing, timeout, non-zero exit) is reported via
    the return dict so callers halt with install guidance — success:False is a halt
    signal, NOT a fallback signal.

    Returns {success, graph_path, already_indexed, output, error}.
    """
    graph_path = os.path.join(path, ".gitnexus")

    def result(success, already_indexed=False, output=None, error=None):
        return {
            "success": success,
            "graph_path": graph_path,
            "already_indexed": already_indexed,
            "output": output,
            "error": error,
        }

    if not available():
        return result(False, error="gitnexus CLI not found on PATH")

    if not force and is_indexed(path):
        return result(True, already_indexed=True)

    cmd = ["gitnexus", "analyze", path, "--index-only"]
    if force:
        cmd.append("--force")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=True
        )
    except FileNotFoundError:
        return result(False, error="gitnexus CLI not found on PATH")
    except subprocess.TimeoutExpired:
        return result(False, error=f"gitnexus analyze timed out after {timeout} seconds")
    except subprocess.CalledProcessError as e:
        return result(False, error=(e.stderr or e.stdout or str(e)).strip())

    _exclude_gitnexus_artifact(path)
    return result(True, output=(proc.stdout or "").strip())


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
        payload = {"available": ok, "required": True, "mcp_registered": mcp_registered()}
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
