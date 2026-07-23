#!/usr/bin/env bash
# Setup script for the ShinkaEvolve integration.
# Initializes the git submodule (SSH with HTTPS fallback), creates skill symlinks,
# and installs the package. ShinkaEvolve is a hard requirement — this script
# fails loudly if the submodule cannot be fetched, rather than degrading silently.
#
# Usage:
#   bash scripts/setup_evolve.sh
#   bash ${CLAUDE_PLUGIN_ROOT}/scripts/setup_evolve.sh

set -euo pipefail

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SKILLS_DIR="$PLUGIN_ROOT/skills"
SHINKA_DIR="$SKILLS_DIR/evolve/ShinkaEvolve"
SUBMODULE_PATH="skills/evolve/ShinkaEvolve"
HTTPS_URL="https://github.com/ChuaHanChong/ShinkaEvolve.git"

# Files test_evolve.py + skill discovery require (stdlib-only imports — no pip deps).
REQUIRED_FILES=(
    "shinka/__init__.py"
    "shinka/llm/query.py"
    "shinka/llm/file_handoff_provider.py"
)

# Step 1: Initialize the submodule. Try the configured (SSH) URL, then fall back to
# HTTPS so a clone without SSH keys still works. No silent skip.
cd "$PLUGIN_ROOT"
if [ -f "$SHINKA_DIR/shinka/__init__.py" ]; then
    echo "Done: ShinkaEvolve submodule already initialized"
else
    echo "Initializing ShinkaEvolve submodule..."
    if git submodule update --init --recursive "$SUBMODULE_PATH" 2>/dev/null; then
        echo "Done: ShinkaEvolve submodule initialized"
    else
        echo "— SSH fetch failed; retrying over HTTPS ($HTTPS_URL)..."
        git config "submodule.$SUBMODULE_PATH.url" "$HTTPS_URL"
        git submodule sync "$SUBMODULE_PATH"
        git submodule update --init --recursive "$SUBMODULE_PATH"
        echo "Done: ShinkaEvolve submodule initialized (HTTPS)"
    fi
fi

# Verify the submodule is complete — hard requirement, no fallback.
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$SHINKA_DIR/$f" ]; then
        echo "ERROR: ShinkaEvolve submodule incomplete — missing $f" >&2
        echo "  Fix: cd $PLUGIN_ROOT && git submodule update --init --recursive $SUBMODULE_PATH" >&2
        echo "  If SSH fails: git config submodule.$SUBMODULE_PATH.url $HTTPS_URL && git submodule sync $SUBMODULE_PATH, then retry" >&2
        exit 1
    fi
done
echo "Done: ShinkaEvolve submodule verified"

# Step 2: Create symlinks for Claude Code skill auto-discovery.
# Claude Code discovers skills at skills/*/SKILL.md — the shinka skills are nested
# inside the submodule and need symlinks at the top level.
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

# Step 3: Install the ShinkaEvolve package (needed at optimize-time for the evolve
# pipeline; test_evolve.py only needs the submodule files verified above).
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
    # Upgrade build tools first so the editable (PEP 660) install succeeds on the
    # first try — old setuptools (<64) lacks the build_editable hook.
    _pip_install -U pip setuptools wheel >/dev/null 2>&1 || true
    if _pip_install -e "$SHINKA_DIR" 2>&1 | tail -3; then
        echo "Done: ShinkaEvolve package installed (editable)"
    elif _pip_install -e "$SHINKA_DIR" --no-build-isolation 2>&1 | tail -3; then
        echo "Done: ShinkaEvolve package installed (editable, --no-build-isolation)"
    else
        echo "WARNING: editable install failed. ShinkaEvolve still runs via PYTHONPATH:"
        echo "  export PYTHONPATH=\"$SHINKA_DIR:\$PYTHONPATH\"   # prepend before shinka_run / SHINKA_PROVIDER=claude_code"
    fi
fi

echo ""
echo "ShinkaEvolve setup complete."
echo "The implement-agent now has access to: evolve, shinka-setup, shinka-convert, shinka-run, shinka-inspect"
