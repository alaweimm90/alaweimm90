# Multi-Organization Monorepo

Enterprise-grade multi-organization monorepo with governance infrastructure.

## Structure

```
.metaHub/           # Governance infrastructure (Backstage, OPA policies, security)
alaweimm90/         # Primary organization workspace
organizations/      # Multi-organization workspace
```

## Quick Start

```bash
# Install dependencies
pnpm install

# Start development environment
docker compose up -d

# Access Backstage Developer Portal
http://localhost:3030
```

## Governance

This repository uses:
- **Backstage** for service catalog and developer portal
- **OPA (Open Policy Agent)** for policy enforcement
- **Renovate** for automated dependency updates
- **OpenSSF Scorecard** for security health checks

## Documentation

- [Security Policy](SECURITY.md)
- [License](LICENSE)
