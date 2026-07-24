#!/usr/bin/env python3
"""Managed dev_notes.md writer — dated, agent-attributed session log entries.

Appends to <exp_root>/dev_notes.md with file locking for concurrent-safe writes.
The --agent-id is embedded as an HTML comment so SubagentStop can verify the
specific agent invocation (not just the agent type) wrote the entry.

Usage:
    python3 dev_notes.py <exp_root> init
    python3 dev_notes.py <exp_root> append <agent_name> '<message>' [--agent-id <id>]
    python3 dev_notes.py <exp_root> last-agent
"""

import fcntl
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _dev_notes_path(exp_root: str) -> Path:
    return Path(exp_root) / "dev_notes.md"


def _atomic_append(path: Path, text: str) -> None:
    """Append text to file with file locking for concurrent safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            with open(path, "a") as f:
                f.write(text)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def init(exp_root: str) -> str:
    """Initialize dev_notes.md with header."""
    path = _dev_notes_path(exp_root)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Dev Notes\n\nSession task log.\n\n")
    return str(path)


def append(exp_root: str, agent_name: str, message: str, agent_id: str = "") -> dict:
    """Append a dated entry to dev_notes.md.

    Format: optional `<!-- agent_id: <id> -->` comment, then
    `## <timestamp> — <first-line description> (<agent_name>)`, then message body.
    Returns {"appended": True, "path": str, "timestamp": str, ...}.
    """
    path = _dev_notes_path(exp_root)
    if not path.exists():
        init(exp_root)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Brief description from first line of message (truncated to 80 chars)
    first_line = message.strip().split("\n")[0]
    if len(first_line) > 80:
        first_line = first_line[:77] + "..."

    lines = ["\n"]
    if agent_id:
        lines.append(f"<!-- agent_id: {agent_id} -->\n")
    lines.append(f"## {timestamp} — {first_line} ({agent_name})\n")
    for line in message.strip().split("\n"):
        lines.append(f"- {line}\n")
    lines.append("\n")

    entry = "".join(lines)
    _atomic_append(path, entry)

    return {
        "appended": True,
        "path": str(path),
        "timestamp": timestamp,
        "agent": agent_name,
        "agent_id": agent_id,
    }


def last_agent(exp_root: str) -> dict:
    """Get the agent name and agent_id from the last entry in dev_notes.md.

    Returns {"agent": str, "agent_id": str, "timestamp": str} or
    {"agent": None} if no entries.
    """
    path = _dev_notes_path(exp_root)
    if not path.is_file():
        return {"agent": None}

    content = path.read_text()
    import re
    # Optional "<!-- agent_id: ID -->\n" then "## ts — desc (name)"
    entry_pattern = re.compile(
        r"(?:<!-- agent_id: ([^ ]+) -->\n)?^## (.+?) — .+ \(([^)]+)\)$",
        re.MULTILINE,
    )
    matches = list(entry_pattern.finditer(content))
    if not matches:
        return {"agent": None}

    last = matches[-1]
    return {
        "agent": last.group(3),
        "agent_id": last.group(1) or "",
        "timestamp": last.group(2),
    }


# CLI
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: dev_notes.py <exp_root> append|last-agent|init [args...]")
        sys.exit(1)

    exp_root = sys.argv[1]
    cmd = sys.argv[2]

    if cmd == "init":
        path = init(exp_root)
        print(json.dumps({"initialized": True, "path": path}))

    elif cmd == "append":
        if len(sys.argv) < 5:
            print("Usage: dev_notes.py <exp_root> append <agent_name> '<message>' [--agent-id <id>]")
            sys.exit(1)
        agent_name = sys.argv[3]
        message = sys.argv[4]
        agent_id = ""
        if "--agent-id" in sys.argv:
            idx = sys.argv.index("--agent-id")
            if idx + 1 < len(sys.argv):
                agent_id = sys.argv[idx + 1]
        result = append(exp_root, agent_name, message, agent_id=agent_id)
        print(json.dumps(result))

    elif cmd == "last-agent":
        result = last_agent(exp_root)
        print(json.dumps(result))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
