# .metaHub - Governance & Infrastructure

Central governance hub containing ALL tools, automation, agents, AI infrastructure, and governance for the monorepo.

## 🚀 Quick Links

- **[Complete Implementation Guide](./GOVERNANCE_SUMMARY.md)** - Full details on all 10 tools (500+ lines)
- **[Developer Guide](./DEVELOPER_GUIDE.md)** - How to work with governance tools
- **[Monitoring Checklist](./MONITORING_CHECKLIST.md)** - Daily/weekly/monthly tasks
- **[Troubleshooting Guide](./TROUBLESHOOTING.md)** - Common issues and solutions
- **[Baseline Metrics](./security/BASELINE_METRICS.md)** - Track security improvements
- **[Next Steps](./NEXT_STEPS.md)** - Immediate actions and ongoing maintenance

### Cleanup & Status Reports

- **[Workflows Cleanup Complete](./WORKFLOWS_CLEANUP.md)** - ✨ 15 obsolete workflows disabled, clean Actions page
- **[Structure Cleanup Complete](./STRUCTURE_CLEANUP_COMPLETE.md)** - ✨ Pure meta governance structure
- **[Clean Start Summary](./CLEAN_START_SUMMARY.md)** - Fresh start with governance-only focus
- **[Structure Analysis](./STRUCTURE_ANALYSIS.md)** - Repository structure decisions and rationale

### Setup Guides

- [Policy-Bot Setup](./POLICY_BOT_SETUP.md) - Install and configure Policy-Bot GitHub App (skipped - requires self-hosting)
- [Allstar Setup](../.allstar/ALLSTAR_SETUP.md) - Install and configure OpenSSF Allstar (pending - 10 min)

### Policies

- [Repository Structure Policy](./policies/repo-structure.rego) - Canonical structure enforcement
- [Docker Security Policy](./policies/docker-security.rego) - Container security best practices
- [Policy-Bot Approval Rules](./policy-bot.yml) - File-based PR approval policies
- [Allstar Security Policies](../.allstar/) - Continuous security monitoring

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

### Tier 1: Core Enforcement (1-Day Setup) ✅ COMPLETE

| Tool | Status | Location | Description |
|------|--------|----------|-------------|
| **GitHub Rulesets** | ✅ Active | GitHub UI | Native branch protection (API verified) |
| **Renovate** | ✅ Active | [.metaHub/renovate.json](./renovate.json) | Automated dependency updates every 3 hours |
| **OpenSSF Scorecard** | ✅ Active | [.github/workflows/scorecard.yml](../.github/workflows/scorecard.yml) | Weekly security health checks (18 tests) |
| **Super-Linter** | ✅ Active | [.github/workflows/super-linter.yml](../.github/workflows/super-linter.yml) | Multi-language code quality gates |
| **CODEOWNERS** | ✅ Active | [.github/CODEOWNERS](../.github/CODEOWNERS) | Required reviews for sensitive paths |

### Tier 2: Policy Hardening (1-Week) ✅ COMPLETE

| Tool | Status | Location | Description |
|------|--------|----------|-------------|
| **Policy-Bot** | ⚠️ Skipped | [.metaHub/policy-bot.yml](./policy-bot.yml) + [Setup Guide](./POLICY_BOT_SETUP.md) | Requires self-hosting (functionality covered by Rulesets + CODEOWNERS) |
| **OPA/Conftest** | ✅ Active | [.github/workflows/opa-conftest.yml](../.github/workflows/opa-conftest.yml) | Policy-as-code validation for Dockerfiles and repo structure |

### Tier 3: Strategic Deployment (1-Month) ✅ COMPLETE

| Tool | Status | Location | Description |
|------|--------|----------|-------------|
| **Backstage Portal** | ✅ Active | [.metaHub/backstage/](./backstage/) | Developer portal with 11 services cataloged |
| **SLSA Provenance** | ✅ Active | [.github/workflows/slsa-provenance.yml](../.github/workflows/slsa-provenance.yml) | Build Level 3 supply chain attestations |
| **OpenSSF Allstar** | 🟡 Pending | [.allstar/](../.allstar/) + [Setup Guide](../.allstar/ALLSTAR_SETUP.md) | Continuous security monitoring (10 min install) |

**Current Coverage**: 8/10 tools active (80%) | 1 pending (Allstar) | 1 skipped (Policy-Bot)

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

### Policy-as-Code (OPA/Rego)

Active policies enforced via Conftest:

**Repository Structure Policy** ([repo-structure.rego](./policies/repo-structure.rego)):

- Enforces canonical root directory structure
- Blocks unauthorized files in `.metaHub/`
- Prevents forbidden patterns (.DS_Store, *.log, node_modules)
- Restricts Dockerfiles to allowed locations
- Warns about large files (>10MB)

**Docker Security Policy** ([docker-security.rego](./policies/docker-security.rego)):

- Requires non-root USER directive
- Mandates HEALTHCHECK for monitoring
- Blocks :latest and untagged base images
- Enforces apt-get best practices
- Prevents secrets in ENV variables
- Recommends multi-stage builds
- Blocks privileged ports (<1024)

**Policy-Bot Approval Rules** ([policy-bot.yml](./policy-bot.yml)):

- Governance changes: 2 approvals required
- Policy changes: Security approval required
- Docker changes: Platform team approval
- Dependency changes: Security review
- Workflow changes: DevOps approval
- Auto-labeling based on file patterns

### Supply Chain Security (SLSA)

**SLSA Build Level 3 Provenance** ([slsa-provenance.yml](../.github/workflows/slsa-provenance.yml)):

- Cryptographically signed build attestations
- Tamper-proof provenance generation via GitHub Actions
- SHA-256 artifact hashes with verification
- GitHub Attestations integration
- Historical provenance storage in `.metaHub/security/slsa/`
- Verification via slsa-verifier CLI tool
- Automated artifact packaging (governance configs, Backstage catalog)

### Developer Portal (Backstage)

**Service Catalog** ([backstage/catalog-info.yaml](./backstage/catalog-info.yaml)):

- **11 cataloged services**: SimCore, Repz, BenchBarrier, Attributa, Mag-Logic, Custom-Exporters, Infra, AI-Agent-Demo, API-Gateway, Dashboard, Healthcare
- **3 resources**: Prometheus, Redis, Local-Registry
- **1 system**: Multi-Org Platform
- Full dependency mapping and API relationships
- Local development URLs and service domains
- Lifecycle tracking (production, experimental)
- Owner assignments and team structure

### Continuous Security Monitoring (Allstar)

**OpenSSF Allstar Policies** ([.allstar/](../.allstar/)):

- **Branch Protection**: Enforces PR requirements, approvals, status checks
- **Binary Artifacts**: Blocks committed binaries and executables
- **Outside Collaborators**: Controls external access
- **Security Policy**: Ensures SECURITY.md exists
- **Dangerous Workflows**: Detects unsafe GitHub Actions patterns
- Auto-remediation capable (currently issue-only mode)
- Integration with existing status checks
