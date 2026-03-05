"""Tests for agent tool lists — verify each agent has required tools."""

import re
from pathlib import Path

AGENTS_DIR = Path(__file__).parent.parent / "agents"


def _parse_tools(agent_file: Path) -> set[str]:
    """Extract the tools list from an agent's YAML frontmatter."""
    text = agent_file.read_text()
    match = re.search(r'^tools:\s*"(.+?)"', text, re.MULTILINE)
    if match:
        return {t.strip() for t in match.group(1).split(",")}
    return set()


def test_tuning_agent_has_write():
    tools = _parse_tools(AGENTS_DIR / "tuning-agent.md")
    assert "Write" in tools, f"tuning-agent missing Write tool. Has: {tools}"


def test_research_agent_has_bash():
    tools = _parse_tools(AGENTS_DIR / "research-agent.md")
    assert "Bash" in tools, f"research-agent missing Bash tool. Has: {tools}"


def test_experiment_agent_has_grep():
    tools = _parse_tools(AGENTS_DIR / "experiment-agent.md")
    assert "Grep" in tools, f"experiment-agent missing Grep tool. Has: {tools}"


def test_implement_agent_has_all_essential_tools():
    tools = _parse_tools(AGENTS_DIR / "implement-agent.md")
    required = {"Bash", "Read", "Write", "Edit", "Glob", "Grep"}
    missing = required - tools
    assert not missing, f"implement-agent missing tools: {missing}"


def test_all_agents_have_read():
    """Every agent should at minimum have Read access."""
    for agent_file in AGENTS_DIR.glob("*.md"):
        tools = _parse_tools(agent_file)
        assert "Read" in tools, f"{agent_file.stem} missing Read tool"
