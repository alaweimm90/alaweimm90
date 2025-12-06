# Security Policy

## Supported Versions
- Production branch (`main`)
- Latest tag from the release pipeline

## Reporting a Vulnerability
Email security@repzcoach.com with:
1. Summary of the issue
2. Steps to reproduce and impacted endpoints
3. Impact assessment (if known) and suggested mitigation

We acknowledge reports within **48 hours** and provide remediation ETA within **5 business days**. Please encrypt sensitive details using our PGP key (see `docs/security/PGP.txt`).

## Disclosure Expectations
- Do not publicly disclose the issue until we confirm a fix or coordinate a timeline.
- Coordinate test timelines with our team to avoid production disruption.
- We credit reporters who request acknowledgement.

## Handling Secrets & Credentials
- Never commit secrets. `.env`, `.env.*`, and credential dumps are ignored by git.
- Use approved secret managers (1Password, Doppler, Vercel secrets) for runtime configuration.
- Rotate live credentials immediately after suspected exposure and log the incident in `docs/operations/INCIDENT_LOG.md`.

## Security Hardening Checklist
- [ ] Dependency scan (`npm audit`, `pnpm audit`, or `snyk test`) monthly
- [ ] Static analysis (ESLint, TypeScript, security plugins) in CI
- [ ] Secrets scan (`trufflehog` | `gitleaks`) before each release
- [ ] HTTPS enforced across environments; HSTS enabled
- [ ] Access logging retained for minimum 90 days
- [ ] Supabase rules verified quarterly to ensure least privilege

## Incident Response
1. **Detect** – Automated alert or report received.
2. **Triage** – Assign severity, assemble response team (security@repzcoach.com + engineering lead).
3. **Contain** – Disable affected keys, isolate compromised services, alert stakeholders.
4. **Eradicate** – Patch vulnerability, rotate credentials, verify fix via tests.
5. **Recover** – Deploy clean build, monitor logs.
6. **Postmortem** – Document lessons learned in `docs/operations/POSTMORTEMS/` within 7 days.

## Responsible Use of AI Features
- AI features require explicit consent. Training data is anonymized.
- Logs from AI providers (OpenAI, etc.) are stored ≤ 30 days.
- All AI interactions are covered by the privacy notice.

## Contacts
- Primary: security@repzcoach.com
- Backup: engineering@repzcoach.com
