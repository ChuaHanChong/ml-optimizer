#!/usr/bin/env bash
# Setup script for the ShinkaEvolve integration.
# Initializes the git submodule and creates symlinks for skill auto-discovery.
#
# Usage:
#   bash scripts/setup_evolve.sh
#
# Or from plugin root:
#   bash ${CLAUDE_PLUGIN_ROOT}/scripts/setup_evolve.sh

set -euo pipefail

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SKILLS_DIR="$PLUGIN_ROOT/skills"
SHINKA_DIR="$SKILLS_DIR/evolve/ShinkaEvolve"

# Step 1: Initialize submodule if not already done
if [ ! -f "$SHINKA_DIR/shinka/__init__.py" ]; then
    echo "Initializing ShinkaEvolve submodule..."
    cd "$PLUGIN_ROOT"
    git submodule update --init skills/evolve/ShinkaEvolve
    echo "Done: ShinkaEvolve submodule initialized"
else
    echo "Done: ShinkaEvolve submodule already initialized"
fi

# Step 2: Create symlinks for Claude Code skill auto-discovery
# Claude Code discovers skills at skills/*/SKILL.md — the shinka skills
# are nested inside the submodule and need symlinks at the top level.
for skill in shinka-setup shinka-convert shinka-run shinka-inspect; do
    target="$SKILLS_DIR/$skill"
    source="evolve/ShinkaEvolve/skills/$skill"
    if [ -L "$target" ]; then
        echo "Done: Symlink exists: $skill"
    elif [ -d "$target" ]; then
        echo "— Skipping $skill: directory already exists (not a symlink)"
    else
        ln -s "$source" "$target"
        echo "Done: Created symlink: $skill → $source"
    fi
done

# Step 3: Install ShinkaEvolve package and dependencies
_pip_install() {
    if command -v pip >/dev/null 2>&1; then
        pip install "$@"
    elif python3 -m pip --version >/dev/null 2>&1; then
        python3 -m pip install "$@"
    elif command -v conda >/dev/null 2>&1; then
        conda run pip install "$@"
    else
        echo "ERROR: No pip or conda found. Install manually: pip install $*" >&2
        return 1
    fi
}

if python3 -c "import shinka" 2>/dev/null; then
    echo "Done: ShinkaEvolve package already installed"
else
    echo "Installing ShinkaEvolve from submodule..."
    _pip_install -e "$SHINKA_DIR" 2>&1 | tail -3
    echo "Done: ShinkaEvolve package installed"
fi

echo ""
echo "ShinkaEvolve setup complete."
echo "The implement-agent now has access to: evolve, shinka-setup, shinka-convert, shinka-run, shinka-inspect"
