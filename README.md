# Governance Contract — Meta-Repository

**Governance contract repository** that defines policies, schemas, and reference examples for portfolio repositories.

**STATUS:** Work in Progress — 60% complete. Core governance (policies, schemas, examples, reusable workflows) ready. Optimization in progress.

## 🎯 Purpose

This repository provides the **governance contract** that other repos consume:
- **Policies** (`.metaHub/policies/`) — OPA/Rego rules for structure, Docker security, Kubernetes, SLOs (COMPLETE)
- **Schemas** (`.metaHub/schemas/`) — `.meta/repo.yaml` format definition (COMPLETE)
- **Reusable Workflows** (`.github/workflows/`) — Callable CI/CD templates for Python, TypeScript, releases (COMPLETE)
- **Infrastructure Examples** (`.metaHub/infra/examples/`) — Reference Dockerfile and docker-compose (COMPLETE)

**This repo is the governance contract** — consumer repos will implement what's defined here.

---

## 📁 Structure

```
alaweimm90/alaweimm90 (governance contract — WIP)

ROOT (7 files — actual)
├── README.md                   # This file
├── LICENSE                     # MIT license
├── .github/                    # GitHub Actions workflows
├── .metaHub/                   # Governance infrastructure
├── .gitattributes              # Git line ending rules
├── .gitignore                  # Git ignore rules
└── SECURITY.md                 # Security policy

.allstar/ (IN PROGRESS)
└── alstar.yaml                 # Allstar security policies [PENDING]

.github/workflows/ (9 total — Governance + Reusable)
├── opa-conftest.yml            # Policy validation on changed files
├── renovate.yml                # Dependency update automation
├── scorecard.yml               # OpenSSF security scoring
├── slsa-provenance.yml         # SLSA supply chain security
├── super-linter.yml            # Code quality linting
├── reusable-python-ci.yml      # Callable: Python CI/testing
├── reusable-ts-ci.yml          # Callable: TypeScript CI/testing
├── reusable-policy.yml         # Callable: OPA policy gate
└── reusable-release.yml        # Callable: Release automation

.metaHub/
├── policies/                   # OPA/Rego governance policies [COMPLETE]
│   ├── repo-structure.rego     # Repository structure (warning-only)
│   ├── docker-security.rego    # Docker security checks
│   ├── k8s-governance.rego     # Kubernetes manifests
│   ├── service-slo.rego        # Service-level objectives
│   ├── adr-policy.rego         # Architecture decision records
│   └── README.md               # Policy documentation
├── schemas/                    # Repository metadata schema [COMPLETE]
│   ├── repo-schema.json        # .meta/repo.yaml schema definition
│   └── README.md               # Schema documentation
└── infra/examples/             # Infrastructure reference examples [COMPLETE]
    ├── Dockerfile.example      # Multi-stage Python Dockerfile
    └── docker-compose.example.yml  # Dev environment reference
```

---

## 🚀 How Consumer Repos Use This

Consumer repositories reference this governance contract via:

1. **Reference policies** from this repo's OPA bundle
   ```bash
   opa eval -d https://github.com/alaweimm90/alaweimm90/policies \
     -i <(./scripts/repo-snapshot.sh) 'data.repo.deny'
   ```

2. **Call reusable workflows** from `.github/workflows/`
   ```yaml
   - uses: alaweimm90/alaweimm90/.github/workflows/reusable-python-ci.yml@main
   - uses: alaweimm90/alaweimm90/.github/workflows/reusable-policy.yml@main
   ```

3. **Implement `.meta/repo.yaml`** per schema in `.metaHub/schemas/`
   ```bash
   ajv validate -s <this-repo>/schemas/repo-schema.json -d .meta/repo.yaml
   ```

4. **Copy examples** from `.metaHub/infra/examples/` as starter code
   ```bash
   cp <this-repo>/.metaHub/infra/examples/Dockerfile.example ./Dockerfile
   ```

---

## 📚 Documentation

Policy documentation:
- **`.metaHub/policies/README.md`** — Policy descriptions and usage
- **`.metaHub/schemas/README.md`** — Schema documentation

---

## 🔗 For Portfolio Operations

This repository **is the governance contract only**. Related operations live in separate repos:

- **Census/Audit:** [`portfolio-census`](https://github.com/alaweimm90/portfolio-census) repo
- **Repo Templates:** [`governance-templates`](https://github.com/alaweimm90/governance-templates) repo
- **Infrastructure Examples:** [`governance-infra`](https://github.com/alaweimm90/governance-infra) repo

---

## 🛡️ Policies

All policies are **warning-only (non-blocking)** — teams learn before enforcement tightens.

See `.metaHub/policies/README.md` for complete documentation.

---

## 📊 Status

**Repository State**: Governance contract — Work in Progress (60% Complete)

**Core Governance (COMPLETE):**
- **Policies**: ✅ 5 OPA/Rego policies, warning-only mode
- **Schemas**: ✅ JSON Schema with complete documentation
- **Examples**: ✅ Dockerfile and docker-compose references
- **Reusable Workflows**: ✅ Python, TypeScript, Policy Gate, Release

**Optimization (IN PROGRESS):**
- Tracked files: 44 (target: reduce to ~30 through consolidation)
- Workflows: 9 all legitimate and necessary (governance + reusable)
- Remaining work: Complete .allstar/ configuration, optional file consolidation

**Last Updated**: 2025-11-26
**Maintainer**: @alaweimm90
