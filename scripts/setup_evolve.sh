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
    echo "✓ ShinkaEvolve submodule initialized"
else
    echo "✓ ShinkaEvolve submodule already initialized"
fi

# Step 2: Create symlinks for Claude Code skill auto-discovery
# Claude Code discovers skills at skills/*/SKILL.md — the shinka skills
# are nested inside the submodule and need symlinks at the top level.
for skill in shinka-setup shinka-convert shinka-run shinka-inspect; do
    target="$SKILLS_DIR/$skill"
    source="evolve/ShinkaEvolve/skills/$skill"
    if [ -L "$target" ]; then
        echo "✓ Symlink exists: $skill"
    elif [ -d "$target" ]; then
        echo "— Skipping $skill: directory already exists (not a symlink)"
    else
        ln -s "$source" "$target"
        echo "✓ Created symlink: $skill → $source"
    fi
done

echo ""
echo "ShinkaEvolve setup complete."
echo "The implement-agent now has access to: evolve, shinka-setup, shinka-convert, shinka-run, shinka-inspect"
