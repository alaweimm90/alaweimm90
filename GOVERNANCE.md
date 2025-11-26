# Governance

<img src="https://img.shields.io/badge/Strategy-Org_Monorepos-A855F7?style=flat-square&labelColor=1a1b27" alt="Strategy"/>
<img src="https://img.shields.io/badge/Status-Active-10B981?style=flat-square&labelColor=1a1b27" alt="Status"/>

---

## Repository Strategy

> **Decision:** Organization-Level Monorepos

Each organization directory is a logical unit containing related projects as subdirectories.

### Why This Approach

| Benefit | Description |
|---------|-------------|
| **Logical Grouping** | Projects within orgs are related (science, tools, etc.) |
| **Shared Governance** | Consistent CI/CD, CODEOWNERS, policies |
| **Atomic Commits** | Cross-project changes in single commits |
| **Simplified Management** | Fewer repos to manage vs. 80+ separate repos |

### Structure

```
organizations/
├── alaweimm90-business/    # Business projects
├── alaweimm90-science/     # Scientific computing
├── alaweimm90-tools/       # Developer tools
├── AlaweinOS/              # OS & infrastructure
└── MeatheadPhysicist/      # Physics education
```

---

## Project Qualification

A subdirectory qualifies as a "project" if it has:

| Requirement | File |
|-------------|------|
| Build config | `pyproject.toml` or `package.json` |
| Deployment config | `Dockerfile` |
| Independent release | Semantic versioning capability |

---

## Compliance Requirements

### All Repositories Must Have

- `.meta/repo.yaml` — Metadata conforming to schema
- `README.md` — Documentation
- `.github/CODEOWNERS` — Ownership definition
- CI workflow — Using reusable workflows

### Tier-Based Requirements

| Tier | Requirements |
|------|--------------|
| **1** (Critical) | Full test coverage, SLO monitoring, incident runbooks |
| **2** (Important) | 80%+ coverage, basic monitoring |
| **3** (Experimental) | Metadata only |

---

## Decision Records

Major architectural decisions are documented as ADRs in `docs/adr/`.

| ADR | Decision |
|-----|----------|
| ADR-001 | Organization-level monorepos |
| ADR-002 | OPA/Rego for policy enforcement |
| ADR-003 | JSON Schema for metadata validation |

---

## Maintainers

| Role | Contact |
|------|---------|
| Lead | [@alaweimm90](https://github.com/alaweimm90) |

---

**See also:** [Contributing](./CONTRIBUTING.md) · [Security](./SECURITY.md) · [Governance Docs](./.metaHub/README.md)
