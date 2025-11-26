# ADR-002: OPA/Rego for Policy Enforcement

<img src="https://img.shields.io/badge/Status-Accepted-10B981?style=flat-square&labelColor=1a1b27" alt="Accepted"/>
<img src="https://img.shields.io/badge/Date-2025--11--26-A855F7?style=flat-square&labelColor=1a1b27" alt="Date"/>

---

## Context

Governance policies need to be:

- Machine-readable and enforceable
- Version-controlled alongside code
- Testable and auditable
- Flexible enough for diverse project types

Options considered:

| Option | Pros | Cons |
|--------|------|------|
| Custom scripts | Simple, flexible | Hard to maintain, not portable |
| JSON Schema only | Standard, well-tooled | Limited to structure validation |
| **OPA/Rego** | Powerful, portable, testable | Learning curve |
| CUE | Strong typing | Smaller ecosystem |

## Decision

**Use Open Policy Agent (OPA) with Rego** for policy enforcement.

```rego
# Example: Dockerfile security policy
package dockerfile.security

deny[msg] {
    input.user == "root"
    msg := "Container must not run as root"
}

deny[msg] {
    not input.healthcheck
    msg := "Dockerfile must include HEALTHCHECK"
}
```

## Rationale

1. **Industry standard** - OPA is CNCF graduated, widely adopted
2. **Declarative** - Policies are data, not imperative code
3. **Testable** - Built-in testing framework
4. **Portable** - Same policies work in CI, admission control, CLI
5. **Composable** - Policies can import and extend each other

## Consequences

### Positive

- Policies are auditable and version-controlled
- Same engine for Dockerfile, Kubernetes, repo structure
- Integration with Conftest for CI pipelines
- Clear separation of policy from enforcement

### Negative

- Rego has a learning curve
- Requires OPA/Conftest tooling in CI
- Policy authoring needs domain expertise

### Mitigations

- Provide policy templates for common patterns
- Document Rego basics in contributor guide
- Use Conftest for familiar CLI experience

## Implementation

Policies live in `.metaHub/policies/`:

```
.metaHub/policies/
├── dockerfile.rego      # Container security
├── kubernetes.rego      # K8s best practices
├── structure.rego       # Repo structure rules
└── dependency-security.rego
```

## Status

**Accepted** - Policies implemented and enforced via `enforce.py`.

---

**See also:** [ADR-001](./ADR-001-organization-monorepos.md) · [ADR-003](./ADR-003-json-schema-metadata.md)
