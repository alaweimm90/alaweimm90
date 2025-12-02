# AI Context - Portfolio Governance System

## Overview

Multi-organization GitHub portfolio governance system managing 5 organizations with centralized templates, policies, and automated enforcement.

## Organizations

| Org                 | Purpose                       |
| ------------------- | ----------------------------- |
| alaweimm90-tools    | Developer tools and utilities |
| alaweimm90-science  | Scientific computing projects |
| alaweimm90-internal | Internal company tools        |
| alaweimm90-oss      | Open source projects          |
| alaweimm90-labs     | Experimental projects         |

## Architecture

```text
GitHub (central)
    ├── .metaHub/           # Governance system
    │   ├── policies/       # YAML policy definitions
    │   ├── scripts/        # Python enforcement scripts
    │   └── templates/      # Org/repo templates
    ├── docs/
    │   └── adr/            # Architecture decisions
    └── organizations/      # Local sync (gitignored)
```

## Code Standards

- Python 3.11+ with type hints
- pathlib for file paths
- Click for CLI
- PyYAML for configs
- UTF-8 encoding

## Auto-Approve Guidelines

All AI assistants should:

1. Apply code changes without confirmation
2. Create/edit files in standard locations
3. Run safe read-only commands
4. Execute git status/diff/log freely
5. Trust workspace fully

---

## AI Tool Selection Matrix

### C(13,1) - Single Tool Selection

| Task Type           | Primary Tool | Reasoning                      |
| ------------------- | ------------ | ------------------------------ |
| Quick edits         | Cursor       | Fastest inline completion      |
| Complex refactoring | Claude Code  | Best reasoning, context window |
| Debugging           | Cline        | Browser + terminal integration |
| Documentation       | Copilot      | Trained on markdown patterns   |
| Code generation     | Codex        | Optimized for code synthesis   |

### C(13,2) - Optimal Tool Pairs

| Workflow           | Tool 1   | Tool 2   | Use Case                          |
| ------------------ | -------- | -------- | --------------------------------- |
| Write + Review     | Cursor   | Claude   | Fast generation + deep review     |
| Code + Test        | Aider    | Cline    | Auto-commit + browser testing     |
| Design + Implement | Gemini   | Windsurf | Multimodal design + IDE execution |
| Prototype + Polish | Blackbox | Kilo     | Rapid YOLO + careful refinement   |

### C(13,3) - Power Workflows

| Pipeline   | Tools                      | Best For                              |
| ---------- | -------------------------- | ------------------------------------- |
| Full-Stack | Cursor → Cline → Claude    | Frontend → Integration → Backend      |
| CI/CD      | Aider → Copilot → Amazon Q | Code → Docs → AWS Deploy              |
| Research   | Gemini → Claude → Codex    | Analysis → Reasoning → Implementation |

## Task-Tool Mapping (Set Theory)

Let T = {debugging, refactoring, generation, documentation, testing}
Let A = {claude, cursor, aider, cline, ...}

Optimal mappings:

- debugging ∩ A = {cline, claude, cursor}
- refactoring ∩ A = {claude, aider, windsurf}
- generation ∩ A = {cursor, codex, blackbox}
- documentation ∩ A = {copilot, gemini, claude}
- testing ∩ A = {cline, aider, amazonq}

## Workflow Efficiency Formula

Efficiency = (Task Completion Rate × Quality) / (Context Switches × Time)

Minimize context switches by:

1. Using IDE-native tools for small changes (Cursor, Windsurf)
2. Using CLI tools for batch operations (Aider, Claude Code)
3. Using browser-capable tools for full-stack (Cline)
