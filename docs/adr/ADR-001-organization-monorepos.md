# ADR-001: Organization-Level Monorepos

<img src="https://img.shields.io/badge/Status-Accepted-10B981?style=flat-square&labelColor=1a1b27" alt="Accepted"/>
<img src="https://img.shields.io/badge/Date-2025--11--26-A855F7?style=flat-square&labelColor=1a1b27" alt="Date"/>

---

## Context

Managing 80+ repositories across 5 organizations creates significant overhead:

- Scattered CI/CD configurations
- Inconsistent governance policies
- Complex dependency management
- Difficult cross-project changes

## Decision

**Adopt organization-level monorepos** where each organization directory contains related projects as subdirectories.

```
organizations/
├── alaweimm90-business/    # Business projects
├── alaweimm90-science/     # Scientific computing
├── alaweimm90-tools/       # Developer tools
├── AlaweinOS/              # OS & infrastructure
└── MeatheadPhysicist/      # Physics education
```

## Rationale

| Factor                | Monorepo                      | Multi-repo            |
| --------------------- | ----------------------------- | --------------------- |
| **Atomic changes**    | Single commit across projects | Multiple PRs required |
| **Shared governance** | One policy set                | Duplicated configs    |
| **CI/CD complexity**  | Centralized                   | Scattered             |
| **Discoverability**   | High (one place)              | Low (search needed)   |
| **Dependencies**      | Direct imports                | Version coordination  |

## Consequences

### Positive

- Consistent CI/CD and governance across all projects
- Simplified cross-project refactoring
- Single source of truth for policies
- Easier onboarding (one repo to clone)

### Negative

- Larger clone size
- More complex CI (path-based triggers needed)
- Requires clear ownership boundaries (CODEOWNERS)

### Mitigations

- Use sparse checkout for large clones
- Implement path-based CI triggers
- Enforce CODEOWNERS for all subdirectories

## Status

**Accepted** - Implemented across all 5 organizations.

---

**See also:** [GOVERNANCE.md](../../GOVERNANCE.md) · [ADR-002](./ADR-002-opa-policy-enforcement.md)
