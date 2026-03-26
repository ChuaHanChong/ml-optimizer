#!/usr/bin/env bash
# Setup script for the Hyperagent integration.
# Initializes the git submodule and creates symlinks for skill auto-discovery.
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
# are nested inside skills/hyperagent/skills/ and need symlinks at the top level.
for skill in hyperagent-setup hyperagent-init hyperagent-select hyperagent-generate hyperagent-eval hyperagent-archive; do
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

# Step 3: Verify Python can import from the submodule
echo ""
echo "Verifying Python imports..."
if python3 -c "
import sys
sys.path.insert(0, '$HYPERAGENT_DIR')
from utils.gl_utils import select_parent, load_archive_data, update_and_save_archive
print('Done: gl_utils imports successful')
" 2>/dev/null; then
    :
else
    echo "Warning: Could not import gl_utils.py — numpy may be missing"
    echo "  Try: pip install numpy"
    echo "  The adapter script has a stdlib fallback, so this is not blocking."
fi

# Step 4: Verify adapter script exists and runs
ADAPTER="$PLUGIN_ROOT/scripts/hyperagent_adapter.py"
if [ -f "$ADAPTER" ]; then
    if python3 "$ADAPTER" --help 2>/dev/null | head -1 | grep -q "hyperagent_adapter"; then
        echo "Done: Adapter script works"
    else
        echo "Done: Adapter script exists (run 'python3 $ADAPTER --help' to verify)"
    fi
else
    echo "Warning: Adapter script not found at $ADAPTER"
fi

echo ""
echo "Hyperagent setup complete."
echo "The meta-agent now has access to: hyperagent-setup, hyperagent-init, hyperagent-select, hyperagent-generate, hyperagent-eval, hyperagent-archive"
