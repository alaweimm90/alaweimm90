# Repository Metadata Schema

Defines the structure of `.meta/repo.yaml` — repository metadata used for governance, cataloging, and compliance.

## Schema File

**Location:** `repo-schema.json` (JSON Schema draft-07)

All consumer repositories MUST implement a `.meta/repo.yaml` file that conforms to this schema.

## Required Fields

### `type` (required)
Repository classification. Determines governance expectations and deployment patterns.

**Allowed values:**
- `lib` — Reusable library (SDK, framework, utility)
- `tool` — Standalone tool or CLI
- `core` — Core service (mission-critical infrastructure)
- `research` — Experimental or research project
- `demo` — Example or demonstration code
- `workspace` — Monorepo or workspace aggregating multiple services

### `language` (required)
Primary programming language. Used for CI workflow selection and linting rules.

**Allowed values:**
- `python` — Python application
- `typescript` — TypeScript/Node.js application
- `mixed` — Multiple languages or polyglot

## Optional Fields

### `tier` (optional)
Criticality tier. Determines deployment frequency, SLO stringency, and disaster recovery requirements.

**Values:**
- `1` — Mission-critical (24/7 support, 99.99% availability)
- `2` — Important production service (business hours support, 99.5% availability)
- `3` — Experimental or non-critical (best-effort support)

### `coverage` (optional)
Test coverage thresholds and targets.

**Structure:**
```yaml
coverage:
  target: 85          # Target coverage percentage
  current: 82         # Current coverage (informational)
  lines: 85           # Line coverage minimum
```

### `docs` (optional)
Documentation profile and requirements.

**Structure:**
```yaml
docs:
  profile: standard   # standard or minimal
  required: true      # Docs are required
  location: docs/     # Where docs are located
```

### `interfaces` (optional)
API endpoints and their specifications.

**Structure:**
```yaml
interfaces:
  - type: rest
    port: 8000
    docs_url: /docs
  - type: grpc
    port: 9000
  - type: websocket
    port: 8080
    path: /ws
```

### `owner` (optional)
Team or person responsible for the repository.

**Structure:**
```yaml
owner: engineering-team      # Team name
contact_email: team@company.com
slack_channel: "#team-name"
```

## Example `.meta/repo.yaml`

```yaml
---
metadata:
  name: my-service
  description: Brief description of what this service does
  owner: platform-team

type: lib
language: python
tier: 2

coverage:
  target: 85
  current: 82

docs:
  profile: standard
  required: true

interfaces:
  - type: rest
    port: 8000
    docs_url: /docs

owner: platform-team
contact_email: platform-team@company.com
slack_channel: "#platform-team"
```

## Validation

All `.meta/repo.yaml` files are validated against this schema in CI.

### Local Validation

```bash
# Install AJV CLI
npm install -g ajv-cli

# Validate your repo.yaml
ajv validate -s .metaHub/schemas/repo-schema.json -d .meta/repo.yaml

# Validate all repos in portfolio
for repo in organizations/*/*/; do
  ajv validate -s .metaHub/schemas/repo-schema.json -d "$repo/.meta/repo.yaml"
done
```

### GitHub Actions Validation

Consumer repositories can validate in CI:
```yaml
- name: Validate metadata
  run: ajv validate -s .metaHub/schemas/repo-schema.json -d .meta/repo.yaml
```

## Why Metadata Matters

The metadata in `.meta/repo.yaml` is used for:

- **Governance:** Determines which policies apply
- **Cataloging:** Inventory of all repositories and their purpose
- **SLO Tracking:** Monitors service-level objectives
- **Compliance:** Maps repos to compliance requirements
- **Automation:** Determines deployment and testing strategies
- **Documentation:** Generates architecture and inventory reports

---

**For more information:** See [parent README](../../README.md) and [Policies](../policies/README.md)
