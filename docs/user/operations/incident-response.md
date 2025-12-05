# Incident Response Playbook

**Purpose:** Structured approach to handling production incidents
**Last Updated:** 2025-12-03

---

## Incident Severity Matrix

| Severity          | Impact            | Response Time | Examples                                |
| ----------------- | ----------------- | ------------- | --------------------------------------- |
| **P0 - Critical** | Complete outage   | Immediate     | CI/CD down, security breach, data loss  |
| **P1 - High**     | Major degradation | < 1 hour      | Build failures, broken main branch      |
| **P2 - Medium**   | Partial impact    | < 4 hours     | Flaky tests, slow builds                |
| **P3 - Low**      | Minor issues      | < 24 hours    | Documentation errors, non-critical bugs |

---

## Incident Response Process

### 1. DETECT

**Automated Alerts:**

- CI/CD pipeline failures
- Pre-commit hook failures
- Security scan alerts
- Technical debt gate failures

**Manual Detection:**

- User reports
- Code review findings
- Monitoring alerts

### 2. ASSESS

Quickly determine:

- **Severity** (P0-P3)
- **Scope** (What's affected?)
- **Impact** (Who's affected?)
- **Cause** (Known/Unknown?)

### 3. RESPOND

#### For P0/P1 (Critical/High):

```bash
# 1. Create incident branch
git checkout -b incident/$(date +%Y%m%d-%H%M)-description
git push -u origin HEAD

# 2. Communicate
# - Post in team channel
# - Update status page
# - Notify stakeholders

# 3. Investigate
git log --oneline -20  # Recent changes
git diff main..HEAD    # What changed
npm test              # Run tests
npm run lint          # Check linting

# 4. Fix or Rollback
```

**Rollback Decision Matrix:**

- Unknown root cause → **ROLLBACK**
- Fix time > 30 min → **ROLLBACK**
- Multiple attempts failed → **ROLLBACK**
- Clear fix available → **FIX FORWARD**

#### For P2/P3 (Medium/Low):

Follow standard development workflow:

1. Create issue
2. Prioritize in backlog
3. Fix in next sprint

---

## Security Incidents

### Immediate Actions

```bash
# 1. Assess if secrets exposed
git log -p | grep -i "password\|token\|key\|secret"

# 2. If secrets found, ROTATE IMMEDIATELY
# - GitHub tokens
# - API keys
# - Database credentials

# 3. Force push to remove secrets (if in recent commits)
git reset --hard HEAD~1
git push --force

# 4. Use BFG Repo-Cleaner for historical commits
# See: SECURITY.md
```

### Post-Incident

- [ ] Update `.gitignore` to prevent recurrence
- [ ] Review pre-commit hook configuration
- [ ] Document in postmortem
- [ ] Update security training

---

## Common Incident Types

### CI/CD Pipeline Failure

**Symptoms:** Build/deploy fails

**Quick Check:**

```bash
# Check recent commits
git log --oneline -5

# Check CI logs
gh run list --limit 5
gh run view <run-id> --log

# Check for dependency issues
npm ci
npm test
```

**Common Causes:**

- Dependency conflicts
- Test flakiness
- Configuration errors
- Quota/rate limits

### Main Branch Broken

**Symptoms:** Tests fail on main

**Immediate Actions:**

```bash
# Identify breaking commit
git bisect start
git bisect bad HEAD
git bisect good HEAD~10

# Revert breaking commit
git revert <commit-hash>
git push

# Or rollback to last good commit
git reset --hard <last-good-commit>
git push --force  # ONLY for emergency on main
```

### Technical Debt Gate Failure

**Symptoms:** Debt scan exceeds thresholds

**Response:**

```bash
# Run debt scan
python automation/debt_cli.py scan --path .

# Check high-priority items
cat automation/debt_scan.md

# Apply policy
python automation/debt_gate.py --scan automation/debt_scan.json --env ci
```

**Mitigation:**

- Address critical items first
- Update threshold if false positives
- Document exceptions

---

## Postmortem Template

After resolving P0/P1 incidents:

```markdown
# Postmortem: [Incident Title]

**Date:** YYYY-MM-DD
**Severity:** P0/P1
**Duration:** X hours
**Status:** Resolved

## Summary

[One-paragraph summary of what happened]

## Timeline

- HH:MM - Incident detected
- HH:MM - Investigation began
- HH:MM - Root cause identified
- HH:MM - Fix deployed
- HH:MM - Incident resolved

## Root Cause

[Technical explanation of what caused the incident]

## Impact

- [Who/what was affected]
- [Duration of impact]
- [Scope of impact]

## Resolution

[How the incident was resolved]

## Action Items

- [ ] Immediate fixes
- [ ] Preventive measures
- [ ] Process improvements
- [ ] Documentation updates

## Lessons Learned

- [What went well]
- [What could be improved]
- [Preventive measures]
```

---

## Escalation

When to escalate:

- Incident persists > 2 hours
- Multiple failed resolution attempts
- Security implications unclear
- External dependencies involved

**Escalation Path:**

1. Team lead (@alawein)
2. External security team (if security-related)
3. Service providers (GitHub, npm, etc.)

---

## Tools & Resources

### Diagnostic Commands

```bash
# System status
git status
git log --oneline -10
npm test
npm run lint
npm audit

# CI status
gh run list
gh workflow list

# Debt status
python automation/debt_gate.py --scan automation/debt_scan.json
```

### Reference Links

- [SECURITY.md](../../SECURITY.md)
- [GOVERNANCE.md](../../GOVERNANCE.md)
- [Technical Debt Policies](../../automation/governance/policies/technical_debt.yaml)
- [GitHub Status](https://www.githubstatus.com/)

---

## Prevention

### Pre-commit Checks

- Pre-commit hooks enabled
- Secret scanning active
- Lint checks enforced
- Type checking enabled

### Continuous Monitoring

- CI/CD pipeline alerts
- Dependency security scans
- Technical debt tracking
- Code review process

### Regular Maintenance

- Weekly dependency updates
- Monthly security audits
- Quarterly policy reviews
- Continuous documentation updates
