# ADR-003: JSON Schema for Metadata Validation

<img src="https://img.shields.io/badge/Status-Accepted-10B981?style=flat-square&labelColor=1a1b27" alt="Accepted"/>
<img src="https://img.shields.io/badge/Date-2025--11--26-A855F7?style=flat-square&labelColor=1a1b27" alt="Date"/>

---

## Context

Every repository needs standardized metadata for:

- Service catalog generation
- Ownership tracking
- Tier classification
- Dependency mapping
- Compliance reporting

This metadata must be:

- Human-readable and editable
- Machine-parseable
- Validated at commit time
- Consistent across all repos

## Decision

**Use JSON Schema to validate `.meta/repo.yaml` files** in every repository.

```yaml
# .meta/repo.yaml
name: my-service
tier: 2
owner: '@alaweimm90'
language: python
status: active
visibility: public
dependencies:
  - core-lib
  - shared-utils
```

## Rationale

| Factor            | JSON Schema                  | Custom Validation |
| ----------------- | ---------------------------- | ----------------- |
| **Tooling**       | Excellent (IDE, CI, editors) | Must build        |
| **Standards**     | IETF RFC 8927                | Proprietary       |
| **Documentation** | Self-documenting             | Separate docs     |
| **Extensibility** | $ref, allOf, oneOf           | Custom code       |

JSON Schema provides:

1. **IDE integration** - Autocomplete, inline validation
2. **CI validation** - `jsonschema` CLI or library
3. **Self-documentation** - Schema describes valid values
4. **Ecosystem** - Generators, validators, converters

## Schema Design

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["name", "tier", "owner"],
  "properties": {
    "name": { "type": "string", "minLength": 1 },
    "tier": { "enum": [1, 2, 3] },
    "owner": { "type": "string", "pattern": "^@" },
    "language": { "enum": ["python", "typescript", "go", "rust"] },
    "status": { "enum": ["active", "maintenance", "deprecated"] }
  }
}
```

## Consequences

### Positive

- Consistent metadata across 81+ repositories
- Automatic catalog generation
- IDE support for editing metadata
- Clear contract between repos and governance

### Negative

- Schema evolution requires migration
- YAML parsing adds dependency
- Must keep schema and docs in sync

### Mitigations

- Use draft-07 for stability
- Provide migration scripts for schema changes
- Generate docs from schema

## Implementation

- Schema: `.metaHub/schemas/repo-schema.json`
- Validation: `enforce.py` and pre-commit hooks
- Catalog: `catalog.py` reads validated metadata

## Status

**Accepted** - Schema enforced across all repositories.

---

**See also:** [ADR-002](./ADR-002-opa-policy-enforcement.md) · [Schema README](../../.metaHub/schemas/README.md)
