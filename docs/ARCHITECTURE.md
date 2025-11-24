# Architecture
This document describes the abstract design patterns enabling MCPs, Agents, and Orchestration to be configuration-driven and plugin-based.
## Core Principles
### 1. Abstraction Over Implementation
Abstract interfaces allow implementations to be swapped without changing definitions.
### 2. Configuration-Driven Design
JSON configurations in `.claude/` define behavior. No code changes needed for new MCPs or workflows.
### 3. Layered Architecture
```
User Interface (Claude Code)
    ↓
Orchestration (Rules, Workflows)
    ↓
Agents (CodeAgent, AnalysisAgent)
    ↓
MCPs (Foundation: filesystem, git, etc)
```
### 4. Registry Pattern
Components register themselves rather than being hardcoded.
### 5. Environment Abstraction
Same code works in devcontainer, local, and CI/CD.
## Package Organization
| Package | Purpose |
|---------|---------|
| mcp-core | MCP registry and management |
| agent-core | Agent framework and orchestrator |
| context-provider | Shared context management |
| issue-library | Issue templates |
| workflow-templates | Workflow definitions |
## Extension Points
1. **New MCP**: Update `.claude/mcp-config.json`
2. **New Agent**: Extend BaseAgent
3. **New Workflow**: Create `.claude/workflows/my-workflow.json`
4. **New Rule**: Update `.claude/orchestration.json`
## Design Patterns
- **Singleton**: ContextProvider
- **Registry**: Orchestrator
- **Factory**: Agent creation
- **Strategy**: Different agent implementations
- **Decorator**: Context wrapping
- **Observer**: Orchestration rules
## Benefits
✅ Configuration-driven
✅ Plugin architecture
✅ Environment agnostic
✅ Testable and mockable
✅ Easily scalable
✅ Maintainable
See [MCP_AGENTS_ORCHESTRATION.md](./MCP_AGENTS_ORCHESTRATION.md) for complete details.
