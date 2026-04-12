#!/usr/bin/env bash
# Shared environment for ml-optimizer hooks.
# Source this at the top of every hook that needs PLUGIN_ROOT or EXP_DIR.
#
# Provides:
#   PLUGIN_ROOT  — absolute path to the ml-optimizer plugin directory.
#                  Uses CLAUDE_PLUGIN_ROOT (set by the harness at runtime)
#                  or falls back to this file's own grandparent directory
#                  (so hooks work when exercised directly by tests).

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
