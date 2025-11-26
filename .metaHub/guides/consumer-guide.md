# Consumer Guide — Implementing the Governance Contract

This guide shows how to adopt the governance contract in your own repositories.

## Quick Start

### 1. Reference the Example Consumer Repository

Start by reviewing the complete example:

📍 [Example Consumer Repository](./.metaHub/examples/consumer-repo/)

This example demonstrates:
- ✅ Repository metadata schema implementation (`.meta/repo.yaml`)
- ✅ Reusable workflow integration (`.github/workflows/ci.yml`)
- ✅ Infrastructure patterns (Dockerfile, docker-compose.yml)
- ✅ Complete working application (FastAPI microservice)
- ✅ Test suite with coverage targets

### 2. Implement Repository Metadata

Create `.meta/repo.yaml` in your repository:

```bash
mkdir -p .meta
cp <governance-contract-repo>/.metaHub/examples/consumer-repo/.meta/repo.yaml .meta/repo.yaml
# Edit .meta/repo.yaml with your repository details
```

**Validate your metadata:**

```bash
# Download and install AJV CLI
npm install -g ajv-cli

# Validate against governance contract schema
ajv validate \
  -s https://github.com/alaweimm90/alaweimm90/raw/main/.metaHub/schemas/repo-schema.json \
  -d .meta/repo.yaml
```

### 3. Integrate Reusable Workflows

Create or update `.github/workflows/ci.yml`:

```yaml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  # For Python repositories
  python-ci:
    uses: alaweimm90/alaweimm90/.github/workflows/reusable-python-ci.yml@main
    with:
      python-version: '3.11'
      test-command: 'pytest tests/'

  # For TypeScript repositories
  typescript-ci:
    uses: alaweimm90/alaweimm90/.github/workflows/reusable-ts-ci.yml@main
    with:
      node-version: '18'
      package-manager: 'npm'

  # For all repositories - OPA policy validation
  policy-validation:
    uses: alaweimm90/alaweimm90/.github/workflows/reusable-policy.yml@main
    with:
      policy-path: .metaHub/policies

  # For releases
  release:
    if: startsWith(github.ref, 'refs/tags/v')
    uses: alaweimm90/alaweimm90/.github/workflows/reusable-release.yml@main
    with:
      version: ${{ github.ref_name }}
      prerelease: false
```

### 4. Copy Infrastructure Examples

Use the governance contract examples as starting points:

```bash
# Copy Dockerfile
cp <governance-contract-repo>/.metaHub/infra/examples/Dockerfile.example ./Dockerfile
# Customize for your application

# Copy docker-compose.yml
cp <governance-contract-repo>/.metaHub/infra/examples/docker-compose.example.yml ./docker-compose.yml
# Customize services for your application
```

### 5. Validate Against Governance Policies

The governance contract includes OPA/Rego policies for validation:

```bash
# Install OPA
curl -L -o opa https://openpolicyagent.org/downloads/latest/opa_linux_x86_64
chmod +x opa
sudo mv opa /usr/local/bin/

# Validate your repository
opa eval -d https://github.com/alaweimm90/alaweimm90/.metaHub/policies \
  -i <(cat .meta/repo.yaml) 'data.repo.warn'
```

**Available policy checks:**
- `repo-structure.rego` — Repository structure and files
- `docker-security.rego` — Docker image and container security
- `k8s-governance.rego` — Kubernetes manifest validation
- `service-slo.rego` — Service-level objective compliance
- `adr-policy.rego` — Architecture Decision Records

## Implementation Patterns

### Repository Types

The governance contract defines these repository types in `.meta/repo.yaml`:

| Type | Purpose | Example |
|------|---------|---------|
| `lib` | Reusable libraries | SDK, utilities |
| `tool` | Standalone tools | CLI, scripts |
| `core` | Core services | API gateway, auth |
| `research` | Experimental work | Proofs of concept |
| `demo` | Educational examples | Sample applications |
| `workspace` | Monorepo workspace | Aggregates multiple services |

### Language Support

The governance contract supports:
- **Python** — FastAPI, Django, async frameworks (via `reusable-python-ci.yml`)
- **TypeScript/Node.js** — Express, Next.js, npm packages (via `reusable-ts-ci.yml`)
- **Go, Rust, etc.** — Custom workflows with OPA policy validation

### Tier Classification

Classify your repository by criticality:

```yaml
tier:
  level: 1  # 1=critical, 2=production, 3=experimental
  criticality: critical|high|medium|low
  slo:
    availability: 99.9%
    latency_p99_ms: 100
    error_rate_target: 0.01%
```

### Interface Declaration

Document your service interfaces:

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

## File Structure

Here's the recommended structure for consumer repositories:

```
my-service/
├── .meta/
│   └── repo.yaml                    # Governance contract metadata
├── .github/
│   └── workflows/
│       └── ci.yml                   # CI using reusable workflows
├── src/
│   ├── main.py (or main.ts)
│   └── ...
├── tests/
│   ├── test_*.py (or *.spec.ts)
│   └── ...
├── docs/
│   ├── runbooks/
│   │   ├── deployment.md
│   │   ├── incident-response.md
│   │   └── ...
│   └── architecture/
│       └── decisions.md
├── Dockerfile                       # Multi-stage per governance examples
├── docker-compose.yml               # Dev environment per governance examples
├── requirements.txt (or package.json)
├── pytest.ini (or jest.config.js)
├── .gitignore
├── .gitattributes
├── README.md
├── SECURITY.md
└── LICENSE
```

## Governance Policies

The governance contract includes **5 OPA/Rego policies** that validate your repository:

### 1. Repository Structure Policy

**File:** `.metaHub/policies/repo-structure.rego`

Enforces:
- ✅ Root files are in the allowed list (README.md, LICENSE, Dockerfile, etc.)
- ✅ `.metaHub/` directory structure
- ✅ `.github/` workflows are present
- ⚠️ Large files (>10MB) should use Git LFS

**Mode:** Warning-only (non-blocking)

### 2. Docker Security Policy

**File:** `.metaHub/policies/docker-security.rego`

Enforces:
- ✅ Non-root user in containers
- ✅ Health checks configured
- ✅ Resource limits specified
- ✅ No hardcoded secrets
- ⚠️ Base image best practices

### 3. Kubernetes Governance Policy

**File:** `.metaHub/policies/k8s-governance.rego`

Enforces:
- ✅ Resource requests/limits
- ✅ Liveness/readiness probes
- ✅ Security context configuration
- ✅ Network policies
- ⚠️ Pod disruption budgets

### 4. Service SLO Policy

**File:** `.metaHub/policies/service-slo.rego`

Enforces:
- ✅ SLO definitions in `.meta/repo.yaml`
- ✅ Monitoring and alerting
- ✅ Error budget tracking
- ⚠️ Compliance with portfolio targets

### 5. Architecture Decision Records Policy

**File:** `.metaHub/policies/adr-policy.rego`

Enforces:
- ✅ ADRs for significant decisions
- ✅ Proper ADR format and status
- ✅ Decision tracking and history
- ⚠️ Team review and approval

## Common Tasks

### Add a New Workflow

Don't create a custom workflow for standard tasks. Instead, use reusable workflows:

```yaml
# ❌ DON'T: Create custom CI workflow
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: pytest

# ✅ DO: Use reusable workflow from governance contract
jobs:
  test:
    uses: alaweimm90/alaweimm90/.github/workflows/reusable-python-ci.yml@main
    with:
      python-version: '3.11'
```

### Update a Policy

Governance policies are maintained in the central contract repository. To request a policy change:

1. Open an issue in `alaweimm90/alaweimm90`
2. Describe the policy change needed
3. Submit a pull request with tests
4. All consumer repositories automatically benefit from the update

### Add OPA Policies to Your Repo

If your repository needs additional policies:

```bash
# Create your repo-specific policies
mkdir -p .metaHub/policies
cat > .metaHub/policies/my-policy.rego << 'EOF'
package my_repo

warn[msg] {
    input.some_field == "invalid"
    msg := "Your custom warning here"
}
EOF

# Run locally
opa eval -d .metaHub/policies -i <(cat .meta/repo.yaml) 'data.my_repo.warn'
```

### Deploy with GitHub Actions

Use the release workflow for semantic versioning:

```bash
# Push a tag to trigger release
git tag -a v1.2.3 -m "Release v1.2.3"
git push origin v1.2.3

# Automatic actions:
# 1. Reusable release workflow triggers
# 2. GitHub release is created
# 3. Docker image is built and published
# 4. Changelog is generated
```

## Troubleshooting

### "Validation failed: .meta/repo.yaml not found"

**Solution:** Create `.meta/repo.yaml` in your repository root:
```bash
mkdir -p .meta
# Copy and customize example metadata
```

### "OPA policy evaluation failed"

**Solution:** Check policy warnings and fix issues:
```bash
# Run policy locally with verbose output
opa eval -d .metaHub/policies -i <(cat .meta/repo.yaml) 'data.repo.warn' -f pretty
```

### "Reusable workflow not found"

**Solution:** Verify the governance contract repository path and branch:
```yaml
uses: alaweimm90/alaweimm90/.github/workflows/reusable-python-ci.yml@main
#     └─ owner ─┘└─ repo ─┘ └─ path to workflow ─┘           └─ branch
```

### "Schema validation failed"

**Solution:** Validate your `.meta/repo.yaml` structure:
```bash
ajv validate \
  -s https://github.com/alaweimm90/alaweimm90/raw/main/.metaHub/schemas/repo-schema.json \
  -d .meta/repo.yaml
```

## Migration Path

If you have existing repositories:

### Phase 1: Metadata (Week 1)
- [ ] Create `.meta/repo.yaml` with your repository details
- [ ] Validate against schema
- [ ] Add CODEOWNERS and SECURITY.md

### Phase 2: Workflows (Week 2)
- [ ] Replace custom CI workflows with reusable workflows
- [ ] Test CI pipeline
- [ ] Add OPA policy validation job

### Phase 3: Infrastructure (Week 3)
- [ ] Update Dockerfile per governance examples
- [ ] Update docker-compose.yml per governance examples
- [ ] Test local development environment

### Phase 4: Policies (Week 4)
- [ ] Review policy violations
- [ ] Fix structural issues
- [ ] Document policy waivers if needed

### Phase 5: Documentation (Week 5)
- [ ] Update README to reference governance contract
- [ ] Create architecture decision records
- [ ] Document runbooks and procedures

## Support and Questions

For questions about the governance contract:

1. **Review Examples:** Start with `.metaHub/examples/consumer-repo/`
2. **Check Documentation:** See `.metaHub/policies/README.md` and `.metaHub/schemas/README.md`
3. **Open an Issue:** Report problems in `alaweimm90/alaweimm90`
4. **Submit PR:** Contribute improvements to the contract

## Next Steps

1. ✅ Clone the example consumer repository as a template
2. ✅ Customize `.meta/repo.yaml` for your service
3. ✅ Integrate reusable workflows into your CI/CD
4. ✅ Validate against governance policies
5. ✅ Update infrastructure for your application
6. ✅ Document your service and operational procedures

---

**Governance Contract:** [alaweimm90/alaweimm90](https://github.com/alaweimm90/alaweimm90)
**Last Updated:** 2025-11-26
**Version:** 1.0
