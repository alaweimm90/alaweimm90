# Architecture Decision Records

<img src="https://img.shields.io/badge/ADRs-3-A855F7?style=flat-square&labelColor=1a1b27" alt="ADRs"/>
<img src="https://img.shields.io/badge/Status-Active-10B981?style=flat-square&labelColor=1a1b27" alt="Active"/>

---

## What are ADRs?

Architecture Decision Records capture significant architectural decisions along with their context and consequences.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-001](./ADR-001-organization-monorepos.md) | Organization-Level Monorepos | Accepted |
| [ADR-002](./ADR-002-opa-policy-enforcement.md) | OPA/Rego for Policy Enforcement | Accepted |
| [ADR-003](./ADR-003-json-schema-metadata.md) | JSON Schema for Metadata Validation | Accepted |

## Template

When adding new ADRs, use this structure:

```markdown
# ADR-NNN: Title

## Context
What is the issue that we're seeing that motivates this decision?

## Decision
What is the change that we're proposing and/or doing?

## Rationale
Why is this the best choice among alternatives?

## Consequences
What becomes easier or harder as a result?

## Status
Proposed | Accepted | Deprecated | Superseded
```

---

**See also:** [GOVERNANCE.md](../../GOVERNANCE.md)
