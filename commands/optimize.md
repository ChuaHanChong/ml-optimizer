---
description: "Optimize an ML model with agent orchestration"
argument-hint: "[describe your model optimization goal]"
allowed-tools: "Bash, Read, Write, Edit, Glob, Grep, Agent, Skill, WebSearch, WebFetch, AskUserQuestion, EnterPlanMode, ExitPlanMode"
---

You MUST invoke the `ml-optimizer:orchestrate` skill using the Skill tool BEFORE generating any response.

CRITICAL RULES:
- Do NOT propose hyperparameters yourself
- Do NOT run training commands yourself
- Do NOT dispatch ml-optimizer agents directly — the orchestrate skill manages all agent coordination
- The orchestrate skill handles EVERYTHING — your only job is to invoke it

User request: $ARGUMENTS
