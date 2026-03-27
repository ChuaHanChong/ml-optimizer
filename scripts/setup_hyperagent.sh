#!/usr/bin/env bash
# Setup script for the Hyperagent integration.
# Initializes the git submodule and creates symlinks for skill auto-discovery.
# Same pattern as setup_evolve.sh — skills live in the submodule, symlinks at top level.
#
# Usage:
#   bash scripts/setup_hyperagent.sh
#
# Or from plugin root:
#   bash ${CLAUDE_PLUGIN_ROOT}/scripts/setup_hyperagent.sh

set -euo pipefail

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SKILLS_DIR="$PLUGIN_ROOT/skills"
HYPERAGENT_DIR="$SKILLS_DIR/hyperagent/Hyperagents"

# Step 1: Initialize submodule if not already done
if [ ! -f "$HYPERAGENT_DIR/utils/gl_utils.py" ]; then
    echo "Initializing Hyperagents submodule..."
    cd "$PLUGIN_ROOT"
    git submodule update --init skills/hyperagent/Hyperagents
    echo "Done: Hyperagents submodule initialized"
else
    echo "Done: Hyperagents submodule already initialized"
fi

# Step 2: Create symlinks for Claude Code skill auto-discovery
# Claude Code discovers skills at skills/*/SKILL.md — the hyperagent skills
# are nested inside the submodule and need symlinks at the top level.
for skill in hyperagent-init hyperagent-select hyperagent-generate hyperagent-eval hyperagent-archive hyperagent-inspect; do
    target="$SKILLS_DIR/$skill"
    source="hyperagent/Hyperagents/skills/$skill"
    if [ -L "$target" ]; then
        echo "Done: Symlink exists: $skill"
    elif [ -d "$target" ]; then
        echo "— Skipping $skill: directory already exists (not a symlink)"
    else
        ln -s "$source" "$target"
        echo "Done: Created symlink: $skill → $source"
    fi
done

# Step 3: Install required dependencies (numpy for gl_utils.py)
_pip_install() {
    if command -v pip >/dev/null 2>&1; then
        pip install "$@"
    elif python3 -m pip --version >/dev/null 2>&1; then
        python3 -m pip install "$@"
    elif command -v conda >/dev/null 2>&1; then
        conda install -y "$@"
    else
        echo "ERROR: No pip or conda found. Install manually: pip install $*" >&2
        return 1
    fi
}

if python3 -c "import numpy" 2>/dev/null; then
    echo "Done: numpy already installed"
else
    echo "Installing numpy (required by Hyperagents gl_utils.py)..."
    _pip_install numpy 2>&1 | tail -3
    echo "Done: numpy installed"
fi

echo ""
echo "Hyperagents setup complete."
echo "The hyperagent-agent now has access to: hyperagent, hyperagent-init, hyperagent-select, hyperagent-generate, hyperagent-eval, hyperagent-archive, hyperagent-inspect"
