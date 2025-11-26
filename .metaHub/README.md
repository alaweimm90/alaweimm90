# GitHub Governance System

**Enterprise-grade governance framework** for 55+ repositories across 5 organizations.

[![CI](https://github.com/alaweimm90/GitHub/actions/workflows/ci.yml/badge.svg)](https://github.com/alaweimm90/GitHub/actions/workflows/ci.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/alaweimm90/GitHub/badge)](https://securityscorecards.dev/viewer/?uri=github.com/alaweimm90/GitHub)

## Overview

This repository is the **single source of truth** for governance across the alaweimm90 portfolio:

| Component | Purpose |
|-----------|---------|
| **Policies** | OPA/Rego rules for structure, Docker, Kubernetes, dependencies |
| **Schemas** | JSON Schema for `.meta/repo.yaml` validation |
| **Reusable Workflows** | CI/CD templates for Python, TypeScript, Go, Rust |
| **Templates** | Dockerfiles, pre-commit configs, README templates |
| **Scripts** | Enforcement, catalog generation, meta auditing |

## Quick Start

### For Consumer Repositories

1. Create `.meta/repo.yaml` in your repository:

```yaml
type: library          # library, tool, demo, research, adapter
language: python       # python, typescript, go, rust
tier: 2                # 1=critical, 2=important, 3=experimental, 4=unknown
owner: your-org
description: Brief description
status: active
```

2. Add CI workflow calling reusable workflows:

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  python:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main
    with:
      python-version: '3.11'

  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
```

3. Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

### For Governance Development

```bash
# Clone and setup
git clone https://github.com/alaweimm90/GitHub.git
cd GitHub
pip install -r .metaHub/scripts/requirements.txt

# Run enforcement
python .metaHub/scripts/enforce.py ./organizations/my-org/

# Generate catalog
python .metaHub/scripts/catalog.py

# Run meta audit
python .metaHub/scripts/meta.py scan-projects
```

## Architecture

### Three-Layer Enforcement

```
Layer 1: Local (Pre-commit)
    - Linting, formatting, schema validation
    - Runs before every commit

Layer 2: CI/CD (GitHub Actions)
    - Reusable workflows for all languages
    - Security scanning, testing, coverage

Layer 3: Portfolio (Governance)
    - OPA policy enforcement
    - Catalog generation
    - Drift detection
```

### Repository Structure

```
.
├── .github/
│   ├── workflows/           # CI/CD workflows
│   │   ├── ci.yml           # Governance repo CI
│   │   ├── reusable-python-ci.yml
│   │   ├── reusable-ts-ci.yml
│   │   ├── reusable-policy.yml
│   │   └── ...
│   ├── CODEOWNERS
│   └── CI_ENFORCEMENT_RULES.md
├── .metaHub/
│   ├── policies/            # OPA/Rego policies
│   │   ├── repo-structure.rego
│   │   ├── docker-security.rego
│   │   ├── dependency-security.rego
│   │   └── ...
│   ├── schemas/             # JSON Schema
│   │   └── repo-schema.json
│   ├── scripts/             # CLI tools
│   │   ├── enforce.py       # Policy enforcement
│   │   ├── catalog.py       # Catalog generation
│   │   ├── meta.py          # Meta auditor
│   │   └── requirements.txt
│   └── templates/           # File templates
│       ├── docker/          # Dockerfiles
│       ├── pre-commit/      # Pre-commit configs
│       └── README.md.template
├── organizations/           # 55+ repositories
├── scripts/
│   ├── govern.sh            # Local governance hook
│   └── verify_and_enforce_golden_path.py
├── tests/                   # Test suite
├── CONTRIBUTING.md
├── GOVERNANCE.md
├── SECURITY.md
└── README.md
```

## CLI Tools

### enforce.py - Policy Enforcement

```bash
# Enforce on single repo
python .metaHub/scripts/enforce.py ./my-repo

# Enforce on organization
python .metaHub/scripts/enforce.py ./organizations/my-org/

# JSON output
python .metaHub/scripts/enforce.py ./my-repo --report json --output results.json

# Strict mode (warnings as errors)
python .metaHub/scripts/enforce.py ./my-repo --strict --fail-on-warnings
```

### catalog.py - Service Catalog

```bash
# Generate JSON catalog
python .metaHub/scripts/catalog.py

# Generate Markdown
python .metaHub/scripts/catalog.py --format markdown --output catalog.md

# Generate HTML
python .metaHub/scripts/catalog.py --format html --output catalog.html
```

### meta.py - Portfolio Auditor

```bash
# Scan all projects
python .metaHub/scripts/meta.py scan-projects

# Filter by organization
python .metaHub/scripts/meta.py scan-projects --org alaweimm90-tools

# Promote project to full repo status
python .metaHub/scripts/meta.py promote-project my-project --org my-org

# Generate audit report
python .metaHub/scripts/meta.py audit --output audit-report.md
```

## Policies

| Policy | Purpose |
|--------|---------|
| `repo-structure.rego` | Repository structure validation |
| `docker-security.rego` | Dockerfile security best practices |
| `dependency-security.rego` | Dependency management security |
| `k8s-governance.rego` | Kubernetes manifest validation |
| `service-slo.rego` | Service level objectives |
| `adr-policy.rego` | Architecture decision records |

## Supported Languages

| Language | CI Workflow | Pre-commit | Dockerfile |
|----------|-------------|------------|------------|
| Python | `reusable-python-ci.yml` | `python.yaml` | `python.Dockerfile` |
| TypeScript | `reusable-ts-ci.yml` | `typescript.yaml` | `typescript.Dockerfile` |
| Go | - | `go.yaml` | `go.Dockerfile` |
| Rust | - | `rust.yaml` | `rust.Dockerfile` |

## Security

- **OpenSSF Scorecard**: Weekly security analysis
- **SLSA Level 3**: Provenance for governance artifacts
- **Trivy**: Container and filesystem scanning
- **Renovate**: Automated dependency updates

See [SECURITY.md](SECURITY.md) for vulnerability reporting.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Maintainer**: [@alaweimm90](https://github.com/alaweimm90)
