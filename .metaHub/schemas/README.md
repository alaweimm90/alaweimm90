# Repository Metadata Schema

Defines the structure of `.meta/repo.yaml` in consumer repositories.

## Schema File

`repo-schema.json` — JSON Schema (draft-07) definition of the `.meta/repo.yaml` format.

## Required Fields

- **`type`** — Repository type classification
  - `lib` — Reusable library (library code)
  - `tool` — Command-line or utility tool
  - `core` — Core infrastructure (orchestrator, control plane)
  - `research` — Research/experimental work
  - `demo` — Demo/example application
  - `workspace` — Monorepo workspace

- **`language`** — Primary programming language
  - `python` — Python project
  - `typescript` — TypeScript/JavaScript project
  - `mixed` — Multiple languages

## Optional Fields

- **`tier`** (integer: 1-3) — Criticality tier
  - `1` — Mission-critical (always monitored, strict SLA)
  - `2` — Important (monitored, standard SLA)
  - `3` — Experimental (best-effort)

- **`coverage`** (object) — Test coverage thresholds
  - `lines` (integer: 0-100) — Minimum line coverage percentage

- **`docs`** (object) — Documentation profile
  - `profile` — `standard` (full docs/) or `minimal` (README only)

- **`interfaces`** (object) — Public interfaces
  - `cli` (boolean) — Provides CLI interface
  - `api` (boolean) — Provides API interface

- **`owner`** (string) — Owner or team responsible

## Example `.meta/repo.yaml`

```yaml
type: lib
tier: 2
language: python
coverage:
  lines: 85
docs:
  profile: standard
interfaces:
  cli: false
  api: true
owner: "@my-team"
```

## Validation

Consumer repos can validate their `.meta/repo.yaml` against this schema:

```bash
# Using ajv CLI
ajv validate -s .metaHub/schemas/repo-schema.json -d .meta/repo.yaml

# Or in GitHub Actions
- uses: schemathesis/ajv-cli-action@v1
  with:
    schema: .metaHub/schemas/repo-schema.json
    instance: .meta/repo.yaml
```

## References

- **JSON Schema Spec:** https://json-schema.org/
- **Validator Tools:** https://www.jsonschemavalidator.net/
- **Root README:** `../../README.md` for consumption guide
