---
description: "Optimize an ML model with agent orchestration"
argument-hint: "[describe your model optimization goal]"
allowed-tools: "Bash, Read, Write, Edit, Glob, Grep, Agent, Skill, WebSearch, WebFetch, AskUserQuestion, EnterPlanMode, ExitPlanMode"
---

You MUST invoke the `ml-optimizer:orchestrate` skill using the Skill tool BEFORE generating any response. Do NOT attempt to handle optimization without loading this skill first.

User request: $ARGUMENTS
