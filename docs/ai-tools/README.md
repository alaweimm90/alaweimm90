# AI Tools Documentation

> Enterprise-grade AI orchestration, monitoring, and compliance tools

## Overview

The AI Tools suite provides comprehensive capabilities for AI-assisted development:

- **Task Orchestration**: Track and manage AI-assisted tasks
- **Multi-layer Caching**: Semantic similarity-based caching
- **Circuit Breakers**: Fault tolerance for AI operations
- **Compliance Scoring**: Policy-based validation
- **Security Scanning**: Secrets, vulnerabilities, licenses
- **Issue Management**: Automated issue tracking
- **Full Observability**: Telemetry and alerting

## Quick Start

```bash
# View all available tools
npm run ai

# View metrics dashboard
npm run ai:dashboard

# Run compliance check
npm run ai:compliance:score

# Run security scan
npm run ai:security:scan

# Sync context
npm run ai:sync
```

## Tools by Category

### Core Tools

| Tool                              | Description                                                       |
| --------------------------------- | ----------------------------------------------------------------- |
| [Orchestrator](./orchestrator.md) | Task management and context injection for AI-assisted development |
| [Sync](./sync.md)                 | Context synchronization from git and other sources                |
| [Dashboard](./dashboard.md)       | ASCII metrics dashboard for AI effectiveness visualization        |

### Infrastructure Tools

| Tool                        | Description                                                     |
| --------------------------- | --------------------------------------------------------------- |
| [Cache](./cache.md)         | Multi-layer caching with semantic similarity for AI operations  |
| [Monitor](./monitor.md)     | Continuous monitoring with circuit breakers for fault tolerance |
| [Telemetry](./telemetry.md) | Observability and alerting for AI operations                    |
| [Errors](./errors.md)       | Structured error handling with automatic recovery strategies    |

### Governance Tools

| Tool                          | Description                                                            |
| ----------------------------- | ---------------------------------------------------------------------- |
| [Compliance](./compliance.md) | Policy-based validation with quantitative scoring and recommendations  |
| [Security](./security.md)     | Security scanning for secrets, vulnerabilities, and license compliance |
| [Issues](./issues.md)         | Automated issue management and tracking                                |

## Integration

### MCP Server

Start the MCP server for AI assistant integration:

```bash
npm run ai:mcp:start
```

### REST API

Start the REST API server:

```bash
npm run ai:api:start
```

### VS Code Extension

Import the VS Code integration module in your extension:

```typescript
import { commands, getStatusBarItems, getTreeViewData } from 'tools/ai/vscode/integration';
```
