# Repository Metadata Schema

<img src="https://img.shields.io/badge/Schema-JSON_Draft_07-A855F7?style=flat-square&labelColor=1a1b27" alt="Schema"/>
<img src="https://img.shields.io/badge/Status-Stable-10B981?style=flat-square&labelColor=1a1b27" alt="Status"/>

---

> Defines the structure of `.meta/repo.yaml` — repository metadata for governance, cataloging, and compliance.

**Schema File:** [`repo-schema.json`](./repo-schema.json)

All consumer repositories **must** implement a `.meta/repo.yaml` that conforms to this schema.

---

## Required Fields

### `type`

Repository classification. Determines governance expectations and deployment patterns.

| Value | Description |
|-------|-------------|
| `lib` | Reusable library (SDK, framework, utility) |
| `tool` | Standalone tool or CLI |
| `core` | Core service (mission-critical infrastructure) |
| `research` | Experimental or research project |
| `demo` | Example or demonstration code |
| `workspace` | Monorepo aggregating multiple services |

### `language`

Primary programming language. Used for CI workflow selection and linting rules.

| Value | Description |
|-------|-------------|
| `python` | Python application |
| `typescript` | TypeScript/Node.js application |
| `mixed` | Multiple languages or polyglot |

---

## Optional Fields

### `tier`

Criticality tier. Determines SLO stringency and support requirements.

| Value | SLA | Availability |
|-------|-----|--------------|
| `1` | 24/7 support | 99.99% |
| `2` | Business hours | 99.5% |
| `3` | Best-effort | — |

### `coverage`

Test coverage thresholds:

```yaml
coverage:
  target: 85          # Target percentage
  current: 82         # Current (informational)
  lines: 85           # Line coverage minimum
```

### `docs`

Documentation requirements:

```yaml
docs:
  profile: standard   # standard | minimal
  required: true
  location: docs/
```

### `interfaces`

API endpoints:

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

### `owner`

Team ownership:

```yaml
owner: engineering-team
contact_email: team@company.com
slack_channel: "#team-name"
```

---

## Complete Example

```yaml
---
metadata:
  name: my-service
  description: Brief description of the service
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

---

## Validation

### Local

```bash
# Install AJV CLI
npm install -g ajv-cli

# Validate your repo.yaml
ajv validate -s .metaHub/schemas/repo-schema.json -d .meta/repo.yaml

# Validate all repos
for repo in organizations/*/*/; do
  ajv validate -s .metaHub/schemas/repo-schema.json -d "$repo/.meta/repo.yaml"
done
```

### GitHub Actions

```yaml
- name: Validate metadata
  run: ajv validate -s .metaHub/schemas/repo-schema.json -d .meta/repo.yaml
```

---

## Why Metadata Matters

| Use Case | Description |
|----------|-------------|
| **Governance** | Determines which policies apply |
| **Cataloging** | Inventory of all repositories |
| **SLO Tracking** | Monitors service-level objectives |
| **Compliance** | Maps repos to requirements |
| **Automation** | Determines deployment strategies |

---

**See also:** [Governance README](../README.md) · [Policies](../policies/README.md)
