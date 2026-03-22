---
name: statusline-setup
description: Configure the ml-optimizer status line for real-time pipeline progress
user-invocable: true
---

# Status Line Setup

Configure a real-time status line that shows ml-optimizer pipeline progress in the terminal footer.

## What it shows

```
[ml-opt] P7 I3 | 8/20 exp (2 running) | best: 0.42 (18.3%) | GPU0: 45% 8G/40G
```

Fields: phase, iteration, completed/budget experiments, best metric with improvement %, GPU utilization/memory.

## Setup

Use the `statusline-setup` agent to configure the status line in the user's Claude Code settings:

```
Agent(
  subagent_type: "statusline-setup",
  description: "Configure ml-optimizer status line",
  prompt: "Set the statusLine in the user's Claude Code settings to: {\"type\": \"command\", \"command\": \"${CLAUDE_PLUGIN_ROOT}/hooks/statusline.sh\"}"
)
```

The status line only appears when an ml-optimizer pipeline is active (experiments/pipeline-state.json exists). It exits silently in non-optimizer sessions.
