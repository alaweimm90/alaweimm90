# Repository Codemap

> Simplified architecture after consolidation (v3.2 - Dec 2025)

## Directory Structure

```text
meta-governance/
├── automation/          # Python automation system (unified)
│   ├── agents/          # AI-powered automation agents
│   ├── cli/             # Python CLI commands
│   ├── core/            # Core automation utilities
│   ├── deployment/      # Deployment configurations
│   ├── orchestration/   # Multi-agent orchestration
│   ├── prompts/         # AI prompt templates
│   ├── services/        # Service integrations
│   ├── types/           # Type definitions
│   └── workflows/       # Automation workflow definitions
├── demo/                # Demos, examples, test scenarios
├── docs/                # Documentation (MkDocs source)
├── organizations/       # Organization monorepo templates
├── src/                 # Service implementations
├── tests/               # Unit tests (Vitest + Pytest)
├── tools/               # TypeScript toolkit
│   ├── ai/              # AI orchestration & MCP integration
│   ├── ORCHEX/           # Code analysis & refactoring engine
│   ├── bin/             # Binary entry points
│   ├── cli/             # Main CLI entry points
│   ├── devops/          # DevOps agents & templates
│   ├── lib/             # Shared TypeScript libraries
│   ├── scripts/         # Build & utility scripts
│   └── security/        # Security scanning tools
├── .ai/                 # AI assistant configurations
├── .allstar/            # GitHub Allstar security config
├── .orchex/              # ORCHEX runtime state & reports
├── .claude/             # Claude Code configuration
├── .github/             # GitHub Actions & workflows
├── .husky/              # Git hooks (pre-commit)
├── .metaHub/            # Governance policies & catalogs
├── .vscode/             # VS Code workspace settings
└── .archive/            # Archived code (historical)
```

## System Architecture

```mermaid
flowchart TB
    subgraph User["User Interface"]
        CLI[ORCHEX CLI]
        VSCode[VS Code Extension]
        MCP[MCP Servers]
    end

    subgraph Orchestration["Orchestration Layer"]
        Router[Task Router]
        DevOpsAgents[DevOps Agents<br/>20 specialized]
        Workflow[Workflow Engine]
    end

    subgraph Core["Core Systems"]
        ORCHEX[ORCHEX Engine]
        AI[AI Integration]
        Governance[Governance]
    end

    subgraph Output["Outputs"]
        Reports[Analysis Reports]
        Metrics[Dashboards]
        Artifacts[Build Artifacts]
    end

    CLI --> Router
    VSCode --> Router
    MCP --> Router

    Router --> DevOpsAgents
    Router --> Workflow
    DevOpsAgents --> ORCHEX
    Workflow --> ORCHEX

    ORCHEX --> AI
    ORCHEX --> Governance

    AI --> Reports
    Governance --> Metrics
    ORCHEX --> Artifacts

    style Router fill:#6366F1,color:#fff
    style DevOpsAgents fill:#10B981,color:#fff
    style ORCHEX fill:#F59E0B,color:#fff
```

## DevOps Agent System

20 specialized agents organized into 4 categories:

```mermaid
flowchart LR
    subgraph Pipeline["Pipeline Agents"]
        BuildAgent[build-agent]
        TestAgent[test-agent]
        LintAgent[lint-agent]
        DependencyAgent[dependency-agent]
    end

    subgraph Security["Security Agents"]
        SecScan[secret-scanner]
        SAST[sast-agent]
        VulnAgent[vuln-agent]
        ComplianceAgent[compliance-agent]
    end

    subgraph Observability["Observability"]
        LogAgent[log-agent]
        MetricsAgent[metrics-agent]
        TracingAgent[tracing-agent]
        AlertAgent[alert-agent]
    end

    subgraph Release["Release Agents"]
        VersionAgent[version-agent]
        ChangelogAgent[changelog-agent]
        DeployAgent[deploy-agent]
        RollbackAgent[rollback-agent]
    end

    style Pipeline fill:#3B82F6,color:#fff
    style Security fill:#EF4444,color:#fff
    style Observability fill:#10B981,color:#fff
    style Release fill:#8B5CF6,color:#fff
```

## Pre-built Workflows

```mermaid
flowchart TD
    subgraph CICD["CI/CD Pipeline"]
        lint[lint-agent] --> build[build-agent]
        build --> test[test-agent]
        test --> security[sast-agent]
    end

    subgraph SecureRelease["Secure Release"]
        scan[secret-scanner] --> vuln[vuln-agent]
        vuln --> compliance[compliance-agent]
        compliance --> deploy[deploy-agent]
    end

    subgraph Incident["Incident Response"]
        detect[alert-agent] --> analyze[log-agent]
        analyze --> trace[tracing-agent]
        trace --> recover[rollback-agent]
    end

    style CICD fill:#3B82F6,color:#fff
    style SecureRelease fill:#10B981,color:#fff
    style Incident fill:#EF4444,color:#fff
```

## ORCHEX Analysis Flow

```mermaid
flowchart LR
    subgraph Input["Input"]
        Source[Source Code]
        Config[ORCHEX.config.yaml]
    end

    subgraph Analysis["Analysis"]
        Router[Task Router]
        Analyzer[Code Analyzer]
        Optimizer[Continuous Optimizer]
    end

    subgraph Validation["Validation"]
        Governance[Governance Check]
        CircuitBreaker[Circuit Breaker]
        Fallback[Fallback Manager]
    end

    subgraph Output["Output"]
        Report[Analysis Report]
        Dashboard[Metrics Dashboard]
        Fix[Auto-Fix Suggestions]
    end

    Source --> Router
    Config --> Router
    Router --> Analyzer
    Analyzer --> Optimizer

    Optimizer --> Governance
    Governance --> CircuitBreaker
    CircuitBreaker --> Fallback

    Fallback --> Report
    Fallback --> Dashboard
    Fallback --> Fix

    style Router fill:#6366F1,color:#fff
    style Optimizer fill:#10B981,color:#fff
    style Governance fill:#F59E0B,color:#fff
```

## Governance Layer

```mermaid
flowchart TB
    subgraph Policies["Policy Enforcement"]
        RootStructure[root-structure.yaml]
        ProtectedFiles[protected-files.yaml]
        NamingConvention[naming-convention.yaml]
    end

    subgraph Validation["Validation"]
        PreTask[Pre-Task Check]
        PostTask[Post-Task Check]
        FileCheck[File Path Check]
    end

    subgraph Actions["GitHub Actions"]
        CI[ci.yml]
        Enforce[enforce.yml]
        Catalog[catalog.yml]
    end

    Policies --> Validation
    Validation --> Actions

    RootStructure --> PreTask
    ProtectedFiles --> FileCheck
    PreTask --> CI
    PostTask --> Enforce
    FileCheck --> Catalog

    style Policies fill:#EF4444,color:#fff
    style Validation fill:#F59E0B,color:#fff
    style Actions fill:#3B82F6,color:#fff
```

## Quick Reference

| Component      | Path                                                                                          | Purpose                  |
| -------------- | --------------------------------------------------------------------------------------------- | ------------------------ |
| ORCHEX CLI     | [tools/orchex/cli/](../tools/orchex/cli/)                                                     | Main command interface   |
| DevOps Agents  | [tools/orchex/orchestration/devops-agents.ts](../tools/orchex/orchestration/devops-agents.ts) | 20 specialized agents    |
| AI Integration | [tools/ai/](../tools/ai/)                                                                     | MCP servers & AI routing |
| Governance     | [.metaHub/policies/](../.metaHub/policies/)                                                   | Policy definitions       |
| Workflows      | [.github/workflows/](../.github/workflows/)                                                   | CI/CD automation         |
| Tests          | [tests/](../tests/)                                                                           | Unit & integration tests |

## Key Files

```text
CLAUDE.md                    # AI assistant instructions
package.json                 # npm scripts & dependencies
tsconfig.json                # TypeScript configuration
eslint.config.js             # ESLint v9 flat config
vitest.config.ts             # Test runner config
.metaHub/policies/*.yaml     # Governance policies
tools/orchex/cli/commands.ts  # CLI command registry
```

## CLI Commands

```bash
# ORCHEX Commands
npm run ORCHEX -- agents       # List DevOps agents
npm run ORCHEX -- workflows    # List available workflows
npm run ORCHEX -- run <name>   # Execute a workflow
npm run ORCHEX -- devops ci    # Run CI/CD pipeline

# Development
npm run lint                  # Run ESLint
npm test                      # Run Vitest tests
npm run build                 # Build TypeScript
```

---

Auto-generated: 2025-12-02 | Structure v3.2
