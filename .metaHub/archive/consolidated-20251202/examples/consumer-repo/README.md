# Example Consumer Repository

This directory demonstrates how a **consumer repository** uses the governance contract from the parent repository (`alawein/alawein`).

## Structure

```
consumer-repo/
├── .meta/
│   └── repo.yaml                    # Repository metadata (per governance contract schema)
├── .github/workflows/
│   └── ci.yml                       # CI workflow using reusable workflows from governance contract
├── src/
│   ├── main.py
│   └── api/
├── tests/
│   └── test_*.py
├── Dockerfile                       # Multi-stage Dockerfile (per governance examples)
├── docker-compose.yml               # Dev environment (per governance examples)
├── requirements.txt                 # Python dependencies
├── pytest.ini                       # Test configuration
└── README.md                        # This file
```

## How This Repo Uses the Governance Contract

### 1. **Repository Metadata** (`.meta/repo.yaml`)

Implements the schema defined in the governance contract:

```bash
# Validate this repo's metadata against the governance contract schema
ajv validate \
  -s https://github.com/alawein/alawein/raw/main/.metaHub/schemas/repo-schema.json \
  -d .meta/repo.yaml
```

**Key fields:**
- `type`: Repository classification (lib, tool, core, research, demo, workspace)
- `language`: Primary language
- `tier`: Criticality level with SLOs
- `interfaces`: API endpoints (REST, gRPC, etc.)
- `dependencies`: Governance contract reference, required services, versions

### 2. **Reusable Workflows** (`.github/workflows/ci.yml`)

References callable workflows from the governance contract using `workflow_call`:

```yaml
jobs:
  python-ci:
    uses: alawein/alawein/.github/workflows/reusable-python-ci.yml@main
    with:
      python-version: '3.11'
      test-command: 'pytest tests/ -v --cov=src'

  policy-validation:
    uses: alawein/alawein/.github/workflows/reusable-policy.yml@main
    with:
      policy-path: .metaHub/policies
```

**Available reusable workflows:**
- `reusable-python-ci.yml` — Python testing, linting, type checking, coverage
- `reusable-ts-ci.yml` — TypeScript/Node CI
- `reusable-policy.yml` — OPA policy validation
- `reusable-release.yml` — Semantic versioning and GitHub releases

### 3. **Infrastructure Examples**

Uses infrastructure patterns from the governance contract:

**Dockerfile:**
- Multi-stage build (builder + runtime)
- Non-root user security
- Health checks
- Minimal base images

**docker-compose.yml:**
- Local development environment
- Service dependencies
- Volume management
- Health checks and startup ordering

### 4. **OPA Policy Validation**

Validates the repository structure and configuration against governance policies:

```bash
# Run OPA policies locally
opa eval -d https://github.com/alawein/alawein/.metaHub/policies \
  -i <(./scripts/repo-snapshot.sh) 'data.repo.warn'
```

**Policies validated:**
- Repository structure (files, directories, organization)
- Docker security (image scanning, base images)
- Kubernetes manifests (resource requests, liveness checks)
- Service SLOs (availability, latency, error rates)
- Architecture Decision Records (ADRs)

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Git

### Setup

```bash
# Clone and navigate
git clone <this-repo-url>
cd consumer-repo

# Install Python dependencies
pip install -r requirements.txt

# Run tests locally
pytest tests/ -v --cov=src

# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f app
```

### Validate Against Governance Contract

```bash
# Validate metadata schema
ajv validate -s ../.metaHub/schemas/repo-schema.json -d .meta/repo.yaml

# Run OPA policies (warning-only mode)
opa eval -d ../.metaHub/policies \
  -i <(cat .meta/repo.yaml) 'data.repo.warn'

# Check Docker image security (per governance)
docker build -t example-microservice:test .
trivy image example-microservice:test
```

## CI/CD Pipeline

This repo's CI pipeline (`.github/workflows/ci.yml`) demonstrates:

1. **Python CI** — Runs linting, type checking, unit tests, coverage
2. **Policy Validation** — Validates against OPA governance policies
3. **Release** — Creates semantic version tags and GitHub releases

All workflows are **reusable workflows** from the governance contract, ensuring:
- **Consistency** — All consumer repos use the same templates
- **Maintainability** — Updates to workflows propagate to all consumers
- **Compliance** — Enforcement of governance policies across the portfolio

## Governance Contract Reference

This repository implements the governance contract defined in:

📍 **Governance Contract Repository:** [`alawein/alawein`](https://github.com/alawein/alawein)

**Core Components:**
- **Policies:** `.metaHub/policies/` — OPA/Rego governance rules
- **Schemas:** `.metaHub/schemas/` — Repository metadata format
- **Examples:** `.metaHub/infra/examples/` — Infrastructure patterns
- **Workflows:** `.github/workflows/reusable-*.yml` — Callable CI/CD templates

## Key Takeaways

This example demonstrates:

✅ **Metadata Compliance** — Repository metadata follows governance schema
✅ **Workflow Reuse** — CI pipeline uses callable workflows from governance contract
✅ **Infrastructure Patterns** — Dockerfile and docker-compose follow governance examples
✅ **Policy Validation** — Repository validates against OPA governance policies
✅ **Documentation** — Clear reference to governance contract and implementation patterns

## For Portfolio Maintainers

If you're looking to adopt this governance contract across your portfolio:

1. **Create a new repo** from this example
2. **Customize `.meta/repo.yaml`** with your repository's details
3. **Update `.github/workflows/ci.yml`** with your specific CI needs
4. **Copy Dockerfile and docker-compose.yml** as starting points for your infrastructure
5. **Reference governance contract** in your README for team alignment

## Status

- ✅ Governance contract reference (complete)
- ✅ Metadata implementation (complete)
- ✅ Reusable workflow integration (complete)
- ✅ Infrastructure examples (complete)
- 🔄 Consumer testing (in progress)
- 📋 Optional: Add Kubernetes manifests per `k8s-governance.rego` policy

---

**Last Updated:** 2025-11-26
**Maintainer:** Platform Team
**Governance Contract Version:** [See parent repo](https://github.com/alawein/alawein)
