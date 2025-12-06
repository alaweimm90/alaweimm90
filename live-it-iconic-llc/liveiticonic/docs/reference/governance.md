# Repository Governance

This document outlines governance policies applied to LiveItIconic, aligned with templates from the Github* source repositories.

## Branch Protection
- Require PRs for changes to `main` and `develop`.
- Require at least 1 approving review.
- Require status checks: CI, CodeQL, Commitlint, SBOM.
- Enforce linear history and signed commits (recommended).

## Code Reviews
- At least one reviewer from CODEOWNERS for changes in owned areas.
- Use PR template and link related issues.

## Commit Messages
- Follow Conventional Commits.
- Use `feat:`, `fix:`, etc. Include scope when helpful.

## Templates
- Issue templates for bug, feature, and design requests.
- Standard PR template.

## CI/CD
- Node CI for root and platform packages (lint, typecheck, test).
- Security: CodeQL, npm audit, SBOM.

## Documentation
- Keep README and docs updated with changes.
- Use `.vale` for prose style in app/assets/profile subfolders.

## Security Scanning
- Automated scans and weekly scheduled runs.

## Dependency Management
- Dependabot for npm and actions, weekly cadence.

## Exceptions
Document exceptions in PR description and tag `governance-exception` if required.

