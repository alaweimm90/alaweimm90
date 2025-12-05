# Meta-Governance Architecture

## Overview

This repository implements a **Golden Path** / **Internal Developer Platform (IDP)** pattern - a centralized source of truth for DevOps templates, governance policies, and shared tooling.

```
┌─────────────────────────────────────────────────────────────────┐
│                     META-GOVERNANCE REPO                        │
│                    (alawein/alawein)                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Templates  │  │  Policies   │  │   Tools     │              │
│  │  (Golden    │  │  (Policy    │  │  (CLI &     │              │
│  │   Path)     │  │   as Code)  │  │   Automation)│             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│         └────────────────┼────────────────┘                      │
│                          │                                       │
│                          ▼                                       │
│              ┌───────────────────────┐                           │
│              │   Reusable Workflows  │                           │
│              │   (.github/workflows) │                           │
│              └───────────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │ Project A  │  │ Project B  │  │ Project C  │
    │ (Consumer) │  │ (Consumer) │  │ (Consumer) │
    └────────────┘  └────────────┘  └────────────┘
```

## Industry Terms

| Term                     | Definition                          | Our Implementation   |
| ------------------------ | ----------------------------------- | -------------------- |
| **Golden Path**          | Opinionated, supported way to build | `templates/devops/`  |
| **Paved Road**           | Pre-built infrastructure patterns   | Reusable workflows   |
| **Policy as Code**       | Codified governance rules           | `.metaHub/policies/` |
| **Platform Engineering** | Building internal dev platforms     | This entire repo     |
| **Inner Source**         | Open source within org              | Shared across orgs   |

## Directory Structure

```
alawein/
├── .github/
│   └── workflows/           # Reusable workflows (called by other repos)
│       ├── reusable-ci.yml
│       ├── reusable-cd.yml
│       └── reusable-release.yml
│
├── .metaHub/
│   └── policies/            # Policy as Code
│       ├── protected-files.yaml
│       └── governance.yaml
│
├── templates/
│   └── devops/              # Golden Path templates
│       ├── ci-cd/
│       ├── kubernetes/
│       ├── terraform/
│       └── monitoring/
│
├── tools/
│   ├── ORCHEX/               # AI-powered refactoring
│   ├── devops/              # Template CLI tools
│   └── governance/          # Policy validators
│
├── docs/
│   ├── ARCHITECTURE.md      # This file
│   ├── FRAMEWORK.md         # User guide
│   └── ai-coding-tools/     # AI tool documentation
│
├── CLAUDE.md                # AI tool instructions
└── README.md                # Personal profile (protected)
```

## How Consumer Repos Use This

### 1. Reusable Workflows

Consumer repos reference workflows from this repo:

```yaml
# In consumer repo: .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  build:
    uses: alawein/alawein/.github/workflows/reusable-ci.yml@main
    with:
      node-version: '20'
    secrets: inherit
```

### 2. Template Bootstrapping

Use the DevOps CLI to scaffold new projects:

```bash
# From consumer repo
npx @alawein/devops-cli init --template=node-service
```

### 3. Policy Inheritance

Consumer repos can reference policies:

```yaml
# In consumer repo: .metaHub/config.yaml
extends: alawein/alawein/.metaHub/policies/governance.yaml
```

## Enforcement Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                     ENFORCEMENT STACK                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 4: GitHub Branch Protection                              │
│  ├── Required status checks                                     │
│  ├── Required reviews                                           │
│  └── Restrict who can push                                      │
│                                                                 │
│  Layer 3: GitHub Actions (CI/CD)                                │
│  ├── Reusable workflows enforce standards                       │
│  ├── Security scanning (npm audit, secret detection)            │
│  └── Code quality gates (lint, test, type-check)                │
│                                                                 │
│  Layer 2: Pre-commit Hooks (Local)                              │
│  ├── Protected files check                                      │
│  ├── Lint-staged (format, lint)                                 │
│  ├── YAML validation                                            │
│  └── File size limits                                           │
│                                                                 │
│  Layer 1: AI Tool Instructions                                  │
│  ├── CLAUDE.md (Claude Code)                                    │
│  ├── .cursorrules (Cursor)                                      │
│  └── .kilorc (Kilo Code)                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## GitHub Organization Pattern

For multi-repo organizations, create a `.github` repository:

```
org-name/.github/
├── profile/
│   └── README.md            # Org profile displayed on GitHub
├── ISSUE_TEMPLATE/
│   ├── bug_report.md
│   └── feature_request.md
├── PULL_REQUEST_TEMPLATE.md
├── CODEOWNERS                # Default code owners
├── CONTRIBUTING.md           # Default contributing guide
├── SECURITY.md               # Security policy
└── workflow-templates/       # Starter workflows for new repos
    ├── ci.yml
    └── ci.properties.json
```

## Sync Strategy

```
Meta-Governance Repo          Consumer Repos
       │                            │
       │  1. Manual (use template)  │
       ├───────────────────────────►│
       │                            │
       │  2. Auto (reusable wf)     │
       ├───────────────────────────►│
       │                            │
       │  3. CLI (scaffold/update)  │
       ├───────────────────────────►│
       │                            │
       │  4. Renovate/Dependabot    │
       ◄────────────────────────────┤
       │  (version bumps)           │
```

## Protected Files

See `.metaHub/policies/protected-files.yaml` for the complete policy.

| Category    | Files                         | Enforcement                   |
| ----------- | ----------------------------- | ----------------------------- |
| Strict      | README.md, LICENSE, workflows | Pre-commit warning, CLAUDE.md |
| Conditional | package.json, tsconfig.json   | Can modify if task requires   |
| Forbidden   | .env\*, secrets, keys         | Never modify, blocked         |

## Adding New Templates

1. Create template in `templates/devops/<category>/`
2. Add manifest: `templates/devops/<category>/manifest.json`
3. Register in CLI: `tools/devops/templates.ts`
4. Document in `docs/templates/<category>.md`

## Version Strategy

- **Workflows**: Semantic versioning via git tags (`@v1`, `@v2`)
- **Templates**: Version in manifest.json
- **CLI**: npm package versioning
- **Policies**: Date-based versioning in YAML header

## Related Documentation

- [Framework Overview](FRAMEWORK.md)
- [Quick Start](../KILO-QUICK-START.md)
- [Contributing](../CONTRIBUTING.md)
