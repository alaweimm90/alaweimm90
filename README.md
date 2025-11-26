# Governance Contract — Meta-Repository

**Pure governance contract** that defines the rules, schemas, and reusable workflows for all repositories in the portfolio.

## 🎯 Purpose

This repository provides the **governance contract** that other repos consume:
- **Policies** (`.metaHub/policies/`) — OPA/Rego rules for structure, Docker security, Kubernetes, SLOs
- **Schemas** (`.metaHub/schemas/`) — `.meta/repo.yaml` format definition
- **Reusable Workflows** (`.github/workflows/`) — Callable CI/CD templates (Python, TypeScript, release)
- **Infrastructure Examples** (`.metaHub/infra/examples/`) — Reference configurations (Dockerfiles, docker-compose)

**This repo is the governance contract** — consumer repos implement what's defined here.

---

## 📁 Structure

```
alaweimm90/alaweimm90 (pure governance contract)

ROOT (7 files — MINIMAL)
├── README.md                   # This file
├── LICENSE                     # License
├── .github/                    # GitHub Actions workflows
├── .metaHub/                   # Pure governance infrastructure
├── .allstar/                   # Allstar security config
├── .gitignore                  # Git ignore rules
└── .gitattributes              # Git attributes

.github/workflows/
├── reusable-python-ci.yml      # Callable: Python CI template
├── reusable-ts-ci.yml          # Callable: TypeScript CI template
├── reusable-policy.yml         # Callable: OPA policy gate
├── reusable-release.yml        # Callable: Release workflow
└── opa-conftest.yml            # Run OPA on changed files (warning-only)

.metaHub/
├── policies/                   # OPA/Rego governance policies
│   ├── repo-structure.rego     # Repository structure validation
│   ├── docker-security.rego    # Docker security checks
│   ├── k8s-governance.rego     # Kubernetes manifests
│   ├── service-slo.rego        # Service-level objectives
│   ├── adr-policy.rego         # Architecture decision records
│   └── README.md               # Policy documentation
├── schemas/                    # Repository metadata schema
│   ├── repo-schema.json        # .meta/repo.yaml schema definition
│   └── README.md               # Schema documentation
└── infra/examples/             # Infrastructure reference examples
    ├── Dockerfile.example      # Multi-stage Python Dockerfile
    └── docker-compose.example.yml
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

**Repository State**: Pure governance contract ✨
- **Tracked files**: ~15 (policies, schemas, workflows, examples)
- **Purpose**: Crystal-clear (governance only)
- **Reusability**: High (other repos consume this contract)
- **Policy mode**: Warning-only (non-blocking)

**Last Updated**: 2025-11-26
**Maintainer**: @alaweimm90
