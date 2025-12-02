# Repository Codemap

> Simplified architecture after consolidation (v3.1 - Dec 2025)

## Directory Structure

```text
meta-governance/
├── automation/          # Python automation system (agents, workflows)
│   ├── agents/          # AI-powered automation agents
│   ├── cli/             # Python CLI commands
│   ├── orchestration/   # Multi-agent orchestration
│   └── workflows/       # Automation workflow definitions
├── demo/                # Demos, examples, test scenarios
├── docs/                # Documentation (MkDocs source)
├── organizations/       # Organization monorepo templates
├── src/                 # Service implementations
├── tests/               # Unit tests (Vitest + Pytest)
├── tools/               # TypeScript toolkit
│   ├── ai/              # AI orchestration & MCP integration
│   ├── atlas/           # Code analysis & refactoring engine
│   ├── cli/             # Main CLI entry points
│   ├── devops/          # DevOps agents & templates
│   └── scripts/         # Build & utility scripts
├── .ai/                 # AI assistant configurations
├── .atlas/              # ATLAS runtime state & reports
├── .github/             # GitHub Actions & workflows
├── .metaHub/            # Governance policies & catalogs
└── .archive/            # Archived code (historical)
```

## System Architecture

```mermaid
flowchart TB
    subgraph User["User Interface"]
        CLI[ATLAS CLI]
        VSCode[VS Code Extension]
        MCP[MCP Servers]
    end

    subgraph Orchestration["Orchestration Layer"]
        Router[Task Router]
        DevOpsAgents[DevOps Agents<br/>20 specialized]
        Workflow[Workflow Engine]
    end

    subgraph Core["Core Systems"]
        Atlas[ATLAS Engine]
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
    DevOpsAgents --> Atlas
    Workflow --> Atlas

    Atlas --> AI
    Atlas --> Governance

    AI --> Reports
    Governance --> Metrics
    Atlas --> Artifacts

    style Router fill:#6366F1,color:#fff
    style DevOpsAgents fill:#10B981,color:#fff
    style Atlas fill:#F59E0B,color:#fff
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

## ATLAS Analysis Flow

```mermaid
flowchart LR
    subgraph Input["Input"]
        Source[Source Code]
        Config[atlas.config.yaml]
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

| Component      | Path                                                                                        | Purpose                  |
| -------------- | ------------------------------------------------------------------------------------------- | ------------------------ |
| ATLAS CLI      | [tools/atlas/cli/](../tools/atlas/cli/)                                                     | Main command interface   |
| DevOps Agents  | [tools/atlas/orchestration/devops-agents.ts](../tools/atlas/orchestration/devops-agents.ts) | 20 specialized agents    |
| AI Integration | [tools/ai/](../tools/ai/)                                                                   | MCP servers & AI routing |
| Governance     | [.metaHub/policies/](../.metaHub/policies/)                                                 | Policy definitions       |
| Workflows      | [.github/workflows/](../.github/workflows/)                                                 | CI/CD automation         |
| Tests          | [tests/](../tests/)                                                                         | Unit & integration tests |

## Key Files

```text
CLAUDE.md                    # AI assistant instructions
package.json                 # npm scripts & dependencies
tsconfig.json                # TypeScript configuration
eslint.config.js             # ESLint v9 flat config
vitest.config.ts             # Test runner config
.metaHub/policies/*.yaml     # Governance policies
tools/atlas/cli/commands.ts  # CLI command registry
```

## CLI Commands

```bash
# ATLAS Commands
npm run atlas -- agents       # List DevOps agents
npm run atlas -- workflows    # List available workflows
npm run atlas -- run <name>   # Execute a workflow
npm run atlas -- devops ci    # Run CI/CD pipeline

# Development
npm run lint                  # Run ESLint
npm test                      # Run Vitest tests
npm run build                 # Build TypeScript
```

---

Auto-generated: 2025-12-02 | Structure v3.0
