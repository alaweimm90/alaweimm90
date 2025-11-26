# Governance Contract

**Pure governance contract repository** — defines policies, schemas, and reusable workflows for portfolio repositories.

## What This Is

This repository is a **single source of truth for governance**. Consumer repositories consume:

- **Policies** (`.metaHub/policies/`) — OPA/Rego rules for structure, Docker security, Kubernetes, SLOs
- **Schemas** (`.metaHub/schemas/`) — `.meta/repo.yaml` format definition
- **Reusable Workflows** (`.github/workflows/`) — Callable CI/CD templates
- **Infrastructure Examples** (`.metaHub/infra/examples/`) — Dockerfile and docker-compose templates

## Start Here

**New to this governance contract?**
- 📖 Read: [Consumer Guide](./.metaHub/guides/consumer-guide.md)
- 📋 See: [Example Consumer Repository](./.metaHub/examples/consumer-repo/)

**Want to understand the governance system?**
- 🏗️ Read: `.metaHub/README.md` (governance index)
- 📋 See: [Policy Documentation](./.metaHub/policies/README.md)
- 📋 See: [Schema Documentation](./.metaHub/schemas/README.md)

## Quick Links

| Need | Location |
|------|----------|
| Policies | [`.metaHub/policies/`](./.metaHub/policies/) |
| Schemas | [`.metaHub/schemas/`](./.metaHub/schemas/) |
| Infrastructure Examples | [`.metaHub/infra/examples/`](./.metaHub/infra/examples/) |
| Consumer Guide | [`.metaHub/guides/consumer-guide.md`](./.metaHub/guides/consumer-guide.md) |
| Example Repo | [`.metaHub/examples/consumer-repo/`](./.metaHub/examples/consumer-repo/) |
| Security Policy | [`.metaHub/SECURITY.md`](./.metaHub/SECURITY.md) |

## All Documentation Lives in `.metaHub/`

Everything you need is in the `.metaHub/` directory:

```
.metaHub/
├── policies/          # OPA/Rego governance rules
├── schemas/           # Repository metadata format
├── guides/            # How-to guides and documentation
├── examples/          # Example consumer repository
├── infra/examples/    # Infrastructure templates
└── README.md          # Navigation hub
```

---

**Status:** Pure governance contract (production-ready)
**License:** MIT
**Maintainer:** [@alaweimm90](https://github.com/alaweimm90)
