#!/usr/bin/env bash
# Bash command safety hook for ml-optimizer autonomous mode.
# Blocks dangerous commands that could damage the system or project.
# Exit 0 = allow, Exit 2 = block (reason fed back to Claude via stderr).

INPUT=$(cat)
CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null || true)

if [ -z "$CMD" ]; then
  exit 0
fi

block() {
  echo "{\"decision\":\"block\",\"reason\":\"$1\"}" >&2
  exit 2
}

# --- Blocked patterns ---

# Catastrophic deletion: rm -rf / or rm -rf ~ or rm -rf $HOME
# Match rm with any flags containing r, followed by / or ~ or $HOME as a standalone path
if echo "$CMD" | grep -qP 'rm\s+-\w*r\w*\s+(/\s|/$|/\)|~\s|~$|~/|~\)|\$HOME)'; then
  block "Blocked: rm -rf on root or home directory"
fi

# Remote history destruction
if echo "$CMD" | grep -qP 'git\s+push\s+.*(-f\b|--force\b)'; then
  block "Blocked: git push --force can destroy remote history"
fi

# Local work destruction
if echo "$CMD" | grep -qE 'git\s+reset\s+--hard'; then
  block "Blocked: git reset --hard destroys uncommitted work"
fi

# Arbitrary code execution from network
if echo "$CMD" | grep -qE 'curl\s.*\|\s*(ba)?sh' || echo "$CMD" | grep -qE 'wget\s.*\|\s*(ba)?sh'; then
  block "Blocked: piping network content to shell executes arbitrary code"
fi

# Insecure permissions
if echo "$CMD" | grep -qE 'chmod\s+777'; then
  block "Blocked: chmod 777 sets insecure permissions"
fi

# Disk destruction
if echo "$CMD" | grep -qE '>\s*/dev/sd[a-z]' || echo "$CMD" | grep -qE 'mkfs\.'; then
  block "Blocked: writing to raw disk device or formatting partition"
fi

# Fork bomb
if echo "$CMD" | grep -qF '(){' && echo "$CMD" | grep -qF '|:&'; then
  block "Blocked: fork bomb detected"
fi

# All checks passed
exit 0
