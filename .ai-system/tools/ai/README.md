# AI Tools

Unified AI orchestration, monitoring, and compliance tooling for the meta-governance repository.

## Structure

```
tools/ai/
├── api/            # REST API server
├── cli/            # CLI commands (compliance, security, issues)
├── docs/           # Documentation generator
├── integrations/   # External integrations (ORCHEX, etc.)
├── mcp/            # Model Context Protocol server
├── scripts/        # Shell and Python automation scripts
├── utils/          # Utility functions
├── vscode/         # VS Code integration
└── *.ts            # Core modules
```

## Core Modules

| Module            | Purpose                        |
| ----------------- | ------------------------------ |
| `cache.ts`        | Response caching and storage   |
| `compliance.ts`   | Governance compliance checking |
| `dashboard.ts`    | Monitoring dashboard           |
| `errors.ts`       | Error handling and reporting   |
| `issues.ts`       | Issue tracking and management  |
| `monitor.ts`      | System monitoring              |
| `orchestrator.ts` | Task orchestration             |
| `security.ts`     | Security scanning              |
| `sync.ts`         | Configuration synchronization  |
| `telemetry.ts`    | Usage telemetry                |

## Usage

```bash
# Run compliance check
npm run ai:compliance:score

# Start monitoring
npm run ai:monitor

# Run security scan
npm run ai:security
```

## Integration

This module integrates with:

- ORCHEX agent framework (`tools/orchex/`)
- MCP servers (`.ai/mcp/`)
- Governance policies (`.metaHub/policies/`)
