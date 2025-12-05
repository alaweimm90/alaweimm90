# OpenSSF Allstar Setup Guide

This governance contract uses **OpenSSF Allstar** for continuous security enforcement.

## What is Allstar?

Allstar is an automated security policy enforcement tool that:

- Monitors repository security posture continuously
- Detects policy violations (branch protection, dangerous workflows, etc.)
- Creates GitHub issues when violations are found
- Non-blocking: violations are reported but don't prevent work
- Can transition to blocking enforcement once policies are established

## Configuration

**File:** `allstar.yaml`

**Current Policy Mode:** Issue-only (non-blocking)

- Violations are reported as GitHub issues
- Teams can learn security best practices before enforcement tightens
- Planned transition to blocking enforcement after initial adoption period

## Policies Enabled

1. **Branch Protection** — Ensures PR requirements are configured
2. **Binary Artifacts** — Prevents executable files from being committed
3. **Outside Collaborators** — Controls external access
4. **Security Policy** — Requires SECURITY.md documentation
5. **Dangerous Workflows** — Detects unsafe GitHub Actions patterns

## Installation

To activate Allstar for this repository:

1. Visit: https://github.com/apps/allstar-app
2. Click "Install"
3. Select this repository (alawein/alawein)
4. Authorize the app
5. Allstar will begin monitoring within 24 hours

## Transition to Blocking

To enable blocking enforcement (violations will fail checks):

1. Update `allstar.yaml` policies from `action: issue` to `action: enforce`
2. Create a commit and push
3. Allstar will enforce policies on new PRs

## References

- **Allstar Docs:** https://github.com/ossf/allstar
- **Policy Details:** https://github.com/ossf/allstar/wiki
- **GitHub App:** https://github.com/apps/allstar-app
