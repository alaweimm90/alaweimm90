# Developer Guide: Working with Governance Tools

## Quick Start

1. **Clone the repository**
2. **Create a feature branch** (never commit to master)
3. **Make your changes**
4. **Create a Pull Request**
5. **Wait for checks and approvals**
6. **Merge when green**

## What to Expect

### Automatic Checks on Every PR

- **Super-Linter**: Validates code quality (JavaScript, TypeScript, Python, YAML, JSON, Markdown, Bash, Dockerfile)
- **OPA/Conftest**: Enforces repository structure and Docker security policies
- **OpenSSF Scorecard**: Weekly security health check (not per-PR)

### Approval Requirements

- **CODEOWNERS**: @alaweimm90 must approve changes to governance files
- **Policy-Bot**: Different files require different numbers of approvals:
  - Governance files (.metaHub, .github/workflows): 2 approvals
  - Docker files: 1 approval
  - Dependencies: 1 approval
  - Regular code: 1 approval

### Automated Tools

- **Renovate**: Creates PRs for dependency updates every 3 hours
  - Auto-merges dev dependencies and minor updates after 3 days
  - Major updates require manual review
  - Security updates are labeled priority-high

- **Allstar**: Monitors security continuously
  - Creates issues for policy violations
  - Check for issues with label "allstar"

## Common Workflows

### Making a Code Change

```bash
git checkout -b feature/my-feature
# Make changes
git add .
git commit -m "feat: add my feature"
git push origin feature/my-feature
gh pr create
```

### Handling a Renovate PR

1. Review dependency update
2. Check if tests pass
3. If minor/patch: Approve and merge (or wait for auto-merge)
4. If major: Review changelog, test manually, then merge

### Fixing a Policy Violation

If OPA/Conftest fails:

1. Read the error message
2. Check `.metaHub/policies/` for the rule
3. Fix the violation (e.g., add USER to Dockerfile)
4. Push fix, checks will re-run

### Working with Docker

When creating or modifying Dockerfiles, ensure:

- ✅ Use specific version tags (not `:latest`)
- ✅ Include `USER` directive (run as non-root)
- ✅ Include `HEALTHCHECK` directive
- ✅ Use `-y` flag with `apt-get install`
- ✅ Clean up after `apt-get` (add `apt-get clean && rm -rf /var/lib/apt/lists/*`)
- ✅ Use `COPY` with `--chown` when using USER
- ❌ Don't expose privileged ports (<1024)
- ❌ Don't put secrets in `ENV` variables

Example compliant Dockerfile:

```dockerfile
FROM node:20-alpine

RUN apk add --no-cache curl

WORKDIR /app

COPY --chown=node:node package*.json ./

RUN npm ci --only=production

USER node

HEALTHCHECK --interval=30s CMD curl -f http://localhost:3000/health || exit 1

EXPOSE 3000

CMD ["node", "server.js"]
```

## Understanding Policy Violations

### Repository Structure Policy

The repo structure policy enforces:

- Only these top-level directories: `.github`, `.metaHub`, `alaweimm90`, `organizations`
- Only these subdirectories in `.metaHub`: `backstage/`, `policies/`, `security/`
- No forbidden patterns: `.DS_Store`, `*.log`, `node_modules/`, `.env`
- Dockerfiles only in service directories
- Files under 10MB (warning for larger)

### Docker Security Policy

Common violations and fixes:

**Violation**: `FROM uses :latest tag`
**Fix**: Use specific version: `FROM node:20-alpine`

**Violation**: `Dockerfile must include USER directive`
**Fix**: Add before CMD: `USER node`

**Violation**: `Dockerfile must include HEALTHCHECK directive`
**Fix**: Add healthcheck:

```dockerfile
HEALTHCHECK --interval=30s CMD curl -f http://localhost:3000/health || exit 1
```

**Violation**: `apt-get install must use -y flag`
**Fix**: `RUN apt-get update && apt-get install -y curl`

**Violation**: `ENV may contain secrets`
**Fix**: Use runtime environment variables, not hardcoded in Dockerfile

## Tools Reference

- **Full documentation**: See `.metaHub/GOVERNANCE_SUMMARY.md`
- **Policy-Bot rules**: See `.metaHub/policy-bot.yml`
- **OPA policies**: See `.metaHub/policies/*.rego`
- **Allstar setup**: See `.allstar/ALLSTAR_SETUP.md`

## Getting Help

- Check governance documentation in `.metaHub/`
- Review Allstar issues for security guidance
- Contact @alaweimm90 for policy exceptions

## Backstage Service Catalog

The repository includes a Backstage developer portal with:

- **11 services**: SimCore, Repz, BenchBarrier, Attributa, Mag-Logic, Custom-Exporters, Infra, AI-Agent-Demo, API-Gateway, Dashboard, Healthcare
- **3 resources**: Prometheus, Redis, Local-Registry
- **1 system**: Multi-Org Platform

To browse the catalog locally:

```bash
cd .metaHub/backstage
node server.js
# Visit http://localhost:3030
```

## Viewing Security Results

### OpenSSF Scorecard

Weekly security health check results:

- GitHub Security tab: https://github.com/alaweimm90/alaweimm90/security
- Artifacts in Actions: `.metaHub/security/scorecard/history/`

### SLSA Provenance

Build attestations for supply chain security:

- GitHub Attestations: https://github.com/alaweimm90/alaweimm90/attestations
- Historical provenances: `.metaHub/security/slsa/`

To verify an artifact:

```bash
# Install slsa-verifier
wget https://github.com/slsa-framework/slsa-verifier/releases/download/v2.5.1/slsa-verifier-linux-amd64
chmod +x slsa-verifier-linux-amd64

# Verify artifact
./slsa-verifier-linux-amd64 verify-artifact \
  --provenance-path .metaHub/security/slsa/provenance-*.intoto.jsonl \
  --source-uri github.com/alaweimm90/alaweimm90 \
  path/to/artifact.tar.gz
```

## Checking Renovate Status

View all dependency update PRs:

```bash
gh pr list --label dependencies
```

Check Renovate workflow runs:

```bash
gh run list --workflow=renovate.yml --limit 5
```

## Monitoring Allstar Issues

View all security monitoring issues:

```bash
gh issue list --label allstar
```

Allstar creates issues for:

- Branch protection misconfigurations
- Committed binary artifacts
- Unauthorized outside collaborators
- Missing SECURITY.md file
- Dangerous workflow patterns

## Emergency Procedures

### Bypassing Checks for Hotfix

If production is down and you need an emergency fix:

1. You may need admin override if configured in GitHub Rulesets
2. Create hotfix branch with minimal fix
3. Use admin bypass to merge (if available)
4. **Immediately after**:
   - Create follow-up PR with proper process
   - Document bypass in commit message
   - Notify team of governance bypass

### Temporarily Disabling a Tool

If a tool is creating false positives:

**Disable Super-Linter validator**:

```yaml
# .github/workflows/super-linter.yml
env:
  VALIDATE_<LANGUAGE>: false
```

**Disable OPA policy**:

```rego
# .metaHub/policies/policy-name.rego
# Comment out problematic rule or add exception
```

**Pause Renovate**:

```json
// .metaHub/renovate.json
{
  "enabled": false
}
```

**Disable Allstar policy**:

```yaml
# .allstar/<policy>.yaml
enabled: false
```

Commit the change, then re-enable after fixing the root issue.

## Best Practices

1. **Keep PRs small**: Easier to review, faster to merge
2. **Write clear commit messages**: Follow conventional commits format
3. **Test locally first**: Run linters/tests before pushing
4. **Review Renovate PRs promptly**: Don't let them accumulate
5. **Read policy violation messages**: They explain exactly what's wrong
6. **Update documentation**: When adding new services, update Backstage catalog
7. **Monitor security**: Check Scorecard results and Allstar issues regularly

## Resources

- [Complete Governance Summary](./GOVERNANCE_SUMMARY.md)
- [Monitoring Checklist](./MONITORING_CHECKLIST.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)
- [Baseline Metrics](./security/BASELINE_METRICS.md)
