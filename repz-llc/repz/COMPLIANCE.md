# Compliance & Governance Overview

## Contacts
- Security incidents: security@repzcoach.com
- Privacy inquiries: privacy@repzcoach.com
- Maintainer: engineering@repzcoach.com

## Core Policies
- [Code of Conduct](CODE_OF_CONDUCT.md) *(pending update)*
- [Security Policy](SECURITY.md)
- [License](LICENSE)
- [Privacy Notice](docs/privacy.md) *(pending)*
- [Production Configuration](PRODUCTION-CONFIG.md)

## Obligations
| Area | Requirement | Enforcement |
|------|-------------|-------------|
| Source quality | `npm run lint`, `npm run format` | `.pre-commit-config.yaml`, CI pipelines |
| Tests | Unit + integration suites (`npm run test`, `npm run test:e2e`) | GitHub Actions (`.github/workflows/`) |
| Secrets | `.env` files excluded from git; use vault or deployment secrets | `.gitignore`, environment provisioning |
| Billing & pricing | Stripe product IDs validated per release | `docs/STRIPE_PRODUCTS_REPZ.md`, release checklist |
| Data retention | Logs 90 days, analytics 24 months, customer data 7 years | Ops runbooks |

## Checklist
- [ ] SECURITY.md documents disclosure process and response timeline
- [ ] CODE_OF_CONDUCT.md finalized and published
- [ ] No secrets committed (`trufflehog` / `git secrets` clean)
- [ ] CI workflows enforce lint, tests, type-check, and bundle thresholds
- [ ] Incident response playbooks stored in `docs/operations/`

## Change Control
1. Pull request with at least one approval.
2. Successful CI (lint, tests, security checks, preview deploy).
3. Tag release and record in `CHANGELOG.md`.
4. Post-release verification logged in `docs/operations/RELEASE_LOG.md`.

## Data Handling
- Supabase credentials stored in secret manager (e.g., Vercel/Netlify/GitHub).
- Stripe live keys restricted to billing microservice; rotate quarterly.
- Export / erasure requests tracked in `docs/privacy-requests.md` with 30-day SLA.

## Auditing Cadence
- Monthly dependency & vulnerability scan.
- Quarterly Stripe reconciliation and AI feature review.
- Annual penetration test with findings logged under `docs/security/`.

Keep this document current with every release to remain audit-ready.
