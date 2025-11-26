<div align="center">

# Portfolio Governance System

<br>

<img src="https://img.shields.io/badge/Repositories-81+-A855F7?style=for-the-badge&labelColor=1a1b27" alt="Repos"/>
<img src="https://img.shields.io/badge/Organizations-5-EC4899?style=for-the-badge&labelColor=1a1b27" alt="Orgs"/>
<img src="https://img.shields.io/badge/Compliance-86%25-4CC9F0?style=for-the-badge&labelColor=1a1b27" alt="Compliance"/>

<br><br>

[![CI](https://img.shields.io/github/actions/workflow/status/alaweimm90/alaweimm90/ci.yml?style=flat-square&label=CI&labelColor=1a1b27&color=10B981)](https://github.com/alaweimm90/alaweimm90/actions)
[![OpenSSF Scorecard](https://img.shields.io/ossf-scorecard/github.com/alaweimm90/alaweimm90?style=flat-square&label=OpenSSF&labelColor=1a1b27&color=A855F7)](https://securityscorecards.dev/viewer/?uri=github.com/alaweimm90/alaweimm90)

</div>

---

## Overview

> **Enterprise-grade governance framework** for the alaweimm90 portfolio.

This is the **single source of truth** for policies, schemas, and automation across all repositories.

| Component | Purpose |
|-----------|---------|
| **Policies** | OPA/Rego rules for structure, Docker, Kubernetes, dependencies |
| **Schemas** | JSON Schema for `.meta/repo.yaml` validation |
| **Reusable Workflows** | CI/CD templates for Python, TypeScript, Go, Rust |
| **Templates** | Dockerfiles, pre-commit configs, README templates |
| **Scripts** | Enforcement, catalog generation, meta auditing |

---

## Quick Start

### For Consumer Repositories

**1. Create `.meta/repo.yaml`:**

```yaml
type: library          # library, tool, demo, research, adapter
language: python       # python, typescript, go, rust
tier: 2                # 1=critical, 2=important, 3=experimental, 4=unknown
owner: your-org
description: Brief description
status: active
```

**2. Add CI workflow:**

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  python:
    uses: alaweimm90/alaweimm90/.github/workflows/reusable-python-ci.yml@main
    with:
      python-version: '3.11'

  policy:
    uses: alaweimm90/alaweimm90/.github/workflows/reusable-policy.yml@main
```

**3. Install pre-commit:**

```bash
pip install pre-commit
pre-commit install
```

### For Governance Development

```bash
# Setup
git clone https://github.com/alaweimm90/alaweimm90.git
cd alaweimm90
pip install -r .metaHub/scripts/requirements.txt

# Enforce
python .metaHub/scripts/enforce.py ./organizations/my-org/

# Catalog
python .metaHub/scripts/catalog.py

# Audit
python .metaHub/scripts/meta.py scan-projects
```

---

## Architecture

### Three-Layer Enforcement

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Local (Pre-commit)                                │
│  ├── Linting, formatting, schema validation                 │
│  └── Runs before every commit                               │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: CI/CD (GitHub Actions)                            │
│  ├── Reusable workflows for all languages                   │
│  └── Security scanning, testing, coverage                   │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Portfolio (Governance)                            │
│  ├── OPA policy enforcement                                 │
│  ├── Catalog generation                                     │
│  └── Drift detection                                        │
└─────────────────────────────────────────────────────────────┘
```

### Repository Structure

```
.
├── .github/
│   ├── workflows/              # CI/CD workflows
│   │   ├── ci.yml              # Governance repo CI
│   │   ├── reusable-python-ci.yml
│   │   ├── reusable-ts-ci.yml
│   │   └── reusable-policy.yml
│   └── CODEOWNERS
├── .metaHub/
│   ├── policies/               # OPA/Rego policies
│   ├── schemas/                # JSON Schema
│   ├── scripts/                # CLI tools
│   └── templates/              # File templates
├── organizations/              # 81+ repositories
├── tests/                      # Test suite
├── CONTRIBUTING.md
├── GOVERNANCE.md
└── SECURITY.md
```

---

## CLI Tools

### `enforce.py` — Policy Enforcement

```bash
python .metaHub/scripts/enforce.py ./my-repo                    # Single repo
python .metaHub/scripts/enforce.py ./organizations/my-org/      # Organization
python .metaHub/scripts/enforce.py ./my-repo --report json      # JSON output
python .metaHub/scripts/enforce.py ./my-repo --strict           # Strict mode
```

### `catalog.py` — Service Catalog

```bash
python .metaHub/scripts/catalog.py                              # JSON catalog
python .metaHub/scripts/catalog.py --format markdown            # Markdown
python .metaHub/scripts/catalog.py --format html                # HTML
```

### `checkpoint.py` — Drift Detection

```bash
python .metaHub/scripts/checkpoint.py --baseline                # Create baseline
python .metaHub/scripts/checkpoint.py                           # Detect drift
```

### `meta.py` — Portfolio Auditor

```bash
python .metaHub/scripts/meta.py scan-projects                   # Scan all
python .metaHub/scripts/meta.py scan-projects --org my-org      # Filter by org
python .metaHub/scripts/meta.py audit --output report.md        # Audit report
```

---

## Policies

| Policy | Purpose |
|--------|---------|
| `repo-structure.rego` | Repository structure validation |
| `docker-security.rego` | Dockerfile security best practices |
| `dependency-security.rego` | Dependency management security |
| `k8s-governance.rego` | Kubernetes manifest validation |
| `service-slo.rego` | Service level objectives |

See [Policies README](./policies/README.md) for details.

---

## Supported Languages

| Language | CI Workflow | Pre-commit | Dockerfile |
|----------|-------------|------------|------------|
| Python | `reusable-python-ci.yml` | `python.yaml` | `python.Dockerfile` |
| TypeScript | `reusable-ts-ci.yml` | `typescript.yaml` | `typescript.Dockerfile` |
| Go | — | `go.yaml` | `go.Dockerfile` |
| Rust | — | `rust.yaml` | `rust.Dockerfile` |

---

## Security

<img src="https://img.shields.io/badge/OpenSSF-Scorecard-A855F7?style=flat-square" />
<img src="https://img.shields.io/badge/SLSA-Level_3-EC4899?style=flat-square" />
<img src="https://img.shields.io/badge/Trivy-Scanning-4CC9F0?style=flat-square" />

- **OpenSSF Scorecard**: Weekly security analysis
- **SLSA Level 3**: Provenance for governance artifacts
- **Trivy**: Container and filesystem scanning

See [SECURITY.md](../SECURITY.md) for vulnerability reporting.

---

## Documentation

| Document | Purpose |
|----------|---------|
| [Quick Start](./docs/QUICK_START.md) | Get started in 5 minutes |
| [Consumer Guide](./guides/consumer-guide.md) | Implementing governance in your repos |
| [Operations Runbook](../docs/OPERATIONS_RUNBOOK.md) | Production operations |
| [Deployment Guide](./guides/DEPLOYMENT_GUIDE.md) | Step-by-step deployment |

---

<div align="center">

**Maintainer**: [@alaweimm90](https://github.com/alaweimm90)

<br>

<img src="https://img.shields.io/badge/License-MIT-A855F7?style=flat-square&labelColor=1a1b27" />

</div>
