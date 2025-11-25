# .metaHub - Governance & Infrastructure

Central governance hub containing ALL tools, automation, agents, AI infrastructure, and governance for the monorepo.

## Structure

```
.metaHub/
├── backstage/          # Developer portal (Backstage)
├── policies/           # OPA policy enforcement
├── security/           # Security scan results
├── renovate.json       # Dependency automation
└── service-catalog.json # Service registry
```

## What Goes Here

**.metaHub contains:**
- Governance tools (Backstage, OPA)
- Automation & CI/CD infrastructure
- AI agents and orchestration
- Security & compliance frameworks
- Shared tooling
- Developer productivity tools
- Monitoring & observability
- Documentation generators

## What Doesn't Go Here

- **alaweimm90/**: Personal GitHub profile repository (README, badges, portfolio)
- **organizations/**: Actual business/org repositories (empty for now, will add later)
- **Root**: Only essential monorepo config files

## Canonical Structure

```
├── .metaHub/          # THIS - All governance, tools, AI, automation
├── alaweimm90/        # Profile repo only (README.md, .github/)
├── organizations/     # Real orgs (empty for now)
├── .github/           # Root workflows
├── .husky/            # Git hooks
└── [config files]     # package.json, docker-compose.yml, etc.
```

This is the ONLY correct structure. Everything else was hallucinated.

## Governance Implementation Status

### Tier 1: Core Enforcement (1-Day Setup) ✅

| Tool | Status | Location | Description |
|------|--------|----------|-------------|
| **GitHub Rulesets** | 🟡 Pending | GitHub UI | Native branch protection - requires manual setup |
| **Renovate** | ✅ Active | [.metaHub/renovate.json](./renovate.json) | Automated dependency updates every 3 hours |
| **OpenSSF Scorecard** | ✅ Active | [.github/workflows/scorecard.yml](../.github/workflows/scorecard.yml) | Weekly security health checks (18 tests) |
| **Super-Linter** | ✅ Active | [.github/workflows/super-linter.yml](../.github/workflows/super-linter.yml) | Multi-language code quality gates |
| **CODEOWNERS** | ✅ Active | [.github/CODEOWNERS](../.github/CODEOWNERS) | Required reviews for sensitive paths |

### Tier 2: Policy Hardening (1-Week) 🔄

| Tool | Status | Description |
|------|--------|-------------|
| **Policy-Bot** | 📋 Planned | Advanced PR approval policies |
| **OPA/Conftest** | 📋 Planned | Policy-as-code validation (already have .rego files) |

### Tier 3: Strategic Deployment (1-Month) 🔄

| Tool | Status | Description |
|------|--------|-------------|
| **Backstage Portal** | 🏗️ In Progress | Developer portal + service catalog |
| **SLSA Provenance** | 📋 Planned | Supply chain security attestations |
| **OpenSSF Allstar** | 📋 Planned | Continuous security enforcement |

## Key Features

### Bypass-Proof Enforcement

- All policies enforced at GitHub server level (cannot bypass with `--no-verify`)
- Required status checks block merges until passing
- CODEOWNERS enforces mandatory reviews
- Branch protection via Rulesets (pending GitHub UI setup)

### Automated Maintenance

- **Renovate**: Auto-updates dependencies with intelligent grouping
- **Scorecard**: Weekly security health monitoring
- **Super-Linter**: Automated code quality checks on every PR

### Security-First

- SARIF integration with GitHub Security tab
- Vulnerability alerts with priority labels
- Historical security metrics in `.metaHub/security/scorecard/history/`
- Docker image scanning, dependency auditing, workflow validation
