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
