# Master AI Specification

> Single source of truth for all AI tool configurations, prompts, and orchestration rules.

## Quick Reference

| Config Type | Location | Format |
|-------------|----------|--------|
| AI Context | `.config/ai/context.yaml` | YAML |
| Superprompts | `.config/ai/superprompts/` | YAML |
| Project Prompts | `automation/prompts/` | Markdown |
| Task History | `.config/ai/task-history.json` | JSON |
| Repository Instructions | `CLAUDE.md` | Markdown |

---

## AI Tool Routing Matrix

| Task Type | Primary | Superprompt | Notes |
|-----------|---------|-------------|-------|
| Complex features | Claude Code | `architect.yaml` | Full context awareness |
| Refactoring | Kilo Code | `codebase-sentinel.yaml` | Systematic changes |
| Quick fixes | Copilot | - | Inline completions |
| Code review | Claude Code | `security-auditor.yaml` | Security-focused |
| Documentation | Claude Code | `meta-orchestration-master.yaml` | Consistency |
| Architecture | Claude Code | `architect.yaml` | System design |

---

## Superprompt Library

### Meta-Orchestration (`.config/ai/superprompts/`)

| File | Purpose | Priority |
|------|---------|----------|
| `meta-orchestration-master.yaml` | Supreme AI governance controller | P0 |
| `codebase-sentinel.yaml` | Systematic codebase quality | P1 |
| `security-auditor.yaml` | Security-first analysis | P1 |
| `architect.yaml` | High-level system design | P1 |
| `refactor-agent.yaml` | Safe refactoring patterns | P2 |
| `test-generator.yaml` | Test case generation | P2 |
| `doc-writer.yaml` | Documentation generation | P2 |

### Project-Specific (`automation/prompts/`)

```
automation/prompts/
├── meta/                 # Cross-project prompts
│   ├── SUPERPROMPT.md
│   └── GOVERNANCE.md
├── project/              # Project scaffolding
│   ├── LOVABLE_*.md
│   └── TEMPLATE_*.md
└── task/                 # Task-specific prompts
    ├── REFACTOR.md
    └── REVIEW.md
```

---

## Configuration Schema

### context.yaml Structure

```yaml
version: "1.0"
ai:
  default_agent: "claude-code"
  routing:
    feature: { agent: "claude-code", superprompt: "architect" }
    refactor: { agent: "kilo-code", superprompt: "codebase-sentinel" }
    security: { agent: "claude-code", superprompt: "security-auditor" }
  
paths:
  superprompts: ".config/ai/superprompts/"
  prompts: "automation/prompts/"
  history: ".config/ai/task-history.json"

llcs:
  - name: "Alawein Technologies LLC"
    projects: ["librex", "talai", "mezan", "helios", "simcore"]
  - name: "REPZ LLC"
    projects: ["repz"]
  - name: "Live It Iconic LLC"
    projects: ["liveiticonic"]
```

---

## Task Tracking Commands

```bash
# Start a task
npm run ai:start <type> <tags> "<description>"
# Example: npm run ai:start feature auth,api "Add OAuth"

# Get context for task
npm run ai:context <type> <tag>

# Complete and log task
npm run ai:complete <success> "<files>" <tokens> <minutes> <iterations> "<notes>"

# View metrics
npm run ai:metrics
```

---

## Protected Files (Never Auto-Modify)

| File | Protection Level |
|------|------------------|
| `README.md` | Strict |
| `LICENSE` | Strict |
| `CODEOWNERS` | Strict |
| `.github/workflows/*.yml` | Strict |
| `.metaHub/policies/*.yaml` | Strict |
| `.env*`, `*.key`, `*.pem` | Forbidden |

---

## MCP Integration

Model Context Protocol servers in `.config/ai/mcp/`:

| Server | Purpose |
|--------|---------|
| `filesystem` | File operations |
| `git` | Version control |
| `supabase` | Database access |

---

## Related Documentation

- [AI Orchestration](AI_ORCHESTRATION.md) - Detailed routing guide
- [LLC Registry](LLC_PROJECT_REGISTRY.md) - Project ownership
- [Architecture](ARCHITECTURE.md) - System design
- [Codemap](CODEMAP.md) - Codebase navigation

