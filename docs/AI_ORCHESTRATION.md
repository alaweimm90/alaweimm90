# AI Orchestration Guide

> Single source of truth for AI tool configuration, routing, and orchestration.

## Overview

This document consolidates AI orchestration information from:
- .config/ai/context.yaml - AI context configuration
- .config/ai/superprompts/ - YAML superprompts for prompt engine
- utomation/prompts/ - Project and task-specific prompts
- CLAUDE.md - AI instructions for this repository

## AI Tool Routing

| Task Type | Primary Tool | Superprompt | Fallback |
|-----------|--------------|-------------|----------|
| Complex features | Claude Code | architect | - |
| Refactoring | Kilo Code | codebase-sentinel | Claude |
| Quick fixes | Copilot | - | - |
| Code review | Claude Code | codebase-sentinel + security-auditor | - |
| Security audit | Claude Code | security-auditor | - |
| Architecture | Claude Code | architect | - |

## Superprompt Library

### Meta-Orchestration (.config/ai/superprompts/)
- `codebase-sentinel.yaml` - Systematic codebase quality maintenance
- `security-auditor.yaml` - Security-first code analysis
- `architect.yaml` - High-level system design
- `meta-orchestration-master.yaml` - Supreme AI governance controller

### Project-Specific (`automation/prompts/project/`)
- 25+ project-specific superprompts
- See `automation/prompts/CATALOG.md` for full list

## Configuration Files

| File | Purpose |
|------|---------|
| `.config/ai/context.yaml` | Master AI context configuration |
| `.config/ai/settings.yaml` | Global AI settings |
| `.config/ai/task-history.json` | Task tracking history |
| `CLAUDE.md` | Repository-specific AI instructions |

## MCP Integration

Model Context Protocol servers registered in:
- `.config/ai/mcp/server-registry.yaml`

## Quick Start

### Select a Superprompt
`python .config/ai/prompt-engine/engine.py select --task "refactor auth module"`

### View Prompt Stats
`python .config/ai/prompt-engine/engine.py report`

## Related Documentation

- [Architecture](ARCHITECTURE.md) - System architecture
- [Codemap](CODEMAP.md) - Codebase navigation
- [Framework](FRAMEWORK.md) - Development framework
