# Meta-Governance Framework

**Enterprise-grade governance, DevOps templates, and automation tools**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)
[![Node: >=20](https://img.shields.io/badge/node-%3E%3D20-brightgreen.svg)](../package.json)

---

## Overview

Meta-governance repository providing:

- **DevOps Templates** - Production-ready templates for CI/CD, Kubernetes, Terraform, and more
- **Governance Tools** - Automated validation and enforcement of repository standards
- **CLI Tools** - Command-line utilities for template management and code generation
- **MCP Integration** - Model Context Protocol server orchestration

---

## Quick Start

```bash
# Install dependencies
npm install

# List available templates
npm run devops:list

# Generate code from template
npm run devops:coder -- --action=node-service

# Run governance validation
npm run lint
npm test
```

---

## Project Structure

```
/
├── src/                    # Source code (planned consolidation)
├── tools/                  # CLI and automation tools
│   ├── devops/            # DevOps template tools (TypeScript)
│   ├── governance/        # Governance validators (Python)
│   └── orchestration/     # Workflow orchestration (Python)
├── templates/             # DevOps templates library
│   └── devops/           # CI/CD, K8s, Terraform, etc.
├── tests/                 # Test suites
├── docs/                  # Documentation
└── scripts/               # Build and deployment scripts
```

---

## Key Features

### DevOps Templates

- **CI/CD**: GitHub Actions, CircleCI, Jenkins
- **Kubernetes**: Helm charts, Kustomize, raw manifests
- **Infrastructure**: Terraform modules for AWS, Azure, GCP
- **Databases**: MongoDB, PostgreSQL, Prisma
- **Monitoring**: Prometheus, Grafana

### Governance Tools

- Structure validation
- Policy enforcement
- Compliance checking
- Automated synchronization

### CLI Tools

- Template discovery and listing
- Code generation from templates
- Project bootstrapping
- Dependency installation

---

## Development

```bash
# Type checking
npm run type-check

# Linting
npm run lint
npm run lint:fix

# Formatting
npm run format
npm run format:check

# Testing
npm test
npm run test:coverage
```

---

## Documentation

- [Quick Start Guide](../KILO-QUICK-START.md)
- [Action Plan](../KILO-ACTION-PLAN.md)
- [Audit Report](../KILO-AUDIT-REPORT.md)
- [Contributing](../CONTRIBUTING.md)
- [License](../LICENSE)
