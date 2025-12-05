# Runbook: Rollback Procedure

**Purpose:** Safely rollback a problematic release to restore stability
**Trigger:** Critical issues discovered in production after deployment
**Duration:** 15-30 minutes

---

## When to Rollback

Use this runbook when:

- ✅ Critical bug discovered in released version (P0/P1 severity)
- ✅ Security vulnerability introduced
- ✅ Data corruption or loss risk
- ✅ Breaking change affecting users
- ✅ Unable to fix forward within 2 hours

**Do NOT rollback for:**

- ❌ Minor bugs (P3/P4) - Fix forward instead
- ❌ Issues affecting < 5% of users - Consider feature flag
- ❌ Cosmetic issues - Fix forward in next release
- ❌ Already in production > 1 week - Fix forward instead

---

## Rollback Decision Matrix

| Issue Severity | User Impact | Time Since Release | Action                   |
| -------------- | ----------- | ------------------ | ------------------------ |
| P0 Critical    | > 50% users | Any time           | **ROLLBACK IMMEDIATELY** |
| P0 Critical    | < 50% users | < 24 hours         | Rollback                 |
| P0 Critical    | < 50% users | > 24 hours         | Hotfix forward           |
| P1 High        | > 80% users | < 48 hours         | Rollback                 |
| P1 High        | < 80% users | Any time           | Hotfix forward           |
| P2 Medium      | Any         | Any time           | Fix forward              |

---

## Prerequisites

- [ ] Incident documented with severity assessment
- [ ] Rollback decision approved by @alawein
- [ ] Team notified of rollback in progress
- [ ] Known good version identified
- [ ] Backup of current state taken (if applicable)

---

## Rollback Procedure

### Step 1: Assess the Situation

```bash
# Document current state
CURRENT_VERSION=$(git describe --tags --abbrev=0)
echo "Current version: $CURRENT_VERSION"

# Identify last known good version
LAST_GOOD_VERSION=$(git describe --tags --abbrev=0 $CURRENT_VERSION^)
echo "Rolling back to: $LAST_GOOD_VERSION"

# Review changes between versions
git log $LAST_GOOD_VERSION..$CURRENT_VERSION --oneline
git diff $LAST_GOOD_VERSION..$CURRENT_VERSION --stat
```

**Document:**

- Current version: `v___.___.___`
- Target version: `v___.___.___`
- Reason for rollback: `___________`
- Approver: `@___________`

### Step 2: Notify Stakeholders

**Internal Notification:**

```bash
# Post in team channel immediately
🚨 ROLLBACK IN PROGRESS 🚨

Version: v1.2.3 → v1.2.2
Reason: [Critical bug description]
ETA: 15-30 minutes
Incident: [Link to incident doc]

DO NOT DEPLOY until rollback complete.
```

**External Notification (if applicable):**

```markdown
⚠️ Service Notice: Temporary rollback in progress

We've identified an issue with v1.2.3 and are rolling back to v1.2.2.
Expected completion: [time]
Updates: [status page link]
```

### Step 3: Rollback Git State

**Option A: Revert Commits (Recommended)**

Preserves history and is safer:

```bash
# Checkout main branch
git checkout main
git pull origin main

# Revert the problematic release
git revert --no-commit $LAST_GOOD_VERSION..HEAD
git commit -m "revert: rollback to $LAST_GOOD_VERSION due to critical bug

Reason: [Brief description]
Original version: $CURRENT_VERSION
Rolling back to: $LAST_GOOD_VERSION

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push revert
git push origin main
```

**Option B: Reset to Previous Version (Use with caution)**

Only use if revert is not feasible:

```bash
# ⚠️ DANGEROUS - Creates force push
git checkout main
git reset --hard $LAST_GOOD_VERSION

# Force push (only for emergencies)
git push --force origin main
```

⚠️ **Force push requires consensus from team lead!**

### Step 4: Rollback GitHub Release

```bash
# Mark current release as broken
gh release edit $CURRENT_VERSION \
  --prerelease \
  --notes "⚠️ This release has been rolled back due to critical issues.
Use $LAST_GOOD_VERSION instead.

Issues:
- [Brief description]

Rollback date: $(date +%Y-%m-%d)"

# Or delete the release entirely
gh release delete $CURRENT_VERSION --yes

# Delete git tag
git tag -d $CURRENT_VERSION
git push origin :refs/tags/$CURRENT_VERSION
```

### Step 5: Rollback npm Package (If Published)

**Within 72 hours of publish:**

```bash
# Unpublish specific version (only works within 72 hours)
npm unpublish meta-governance@$CURRENT_VERSION

# Verify unpublished
npm view meta-governance versions
```

**After 72 hours:**

```bash
# Deprecate the version instead
npm deprecate meta-governance@$CURRENT_VERSION "Critical bug - use $LAST_GOOD_VERSION instead"

# Verify deprecated
npm view meta-governance
```

### Step 6: Verify Rollback

```bash
# Verify git state
git describe --tags --abbrev=0
# Should show: $LAST_GOOD_VERSION or a new version

# Verify GitHub release
gh release list | head -5

# Verify npm (if applicable)
npm view meta-governance version
npm view meta-governance dist-tags.latest
```

### Step 7: Monitor for Stability

**Immediate checks (first 30 minutes):**

```bash
# Monitor CI status
gh run list --limit 5

# Check for incoming issues
gh issue list --label "bug" --state open --json number,title,createdAt

# Review logs (if applicable)
# [Your monitoring system]
```

**Extended monitoring (next 2 hours):**

- [ ] No new critical issues reported
- [ ] CI/CD pipeline stable
- [ ] User reports indicate stability
- [ ] Metrics returned to normal

---

## Rollback Scenarios

### Scenario 1: Rollback Due to Critical Bug

**Example:** v1.2.3 has a bug that breaks all CLI commands

```bash
# 1. Revert commits
git revert --no-commit v1.2.2..HEAD
git commit -m "revert: rollback to v1.2.2 - CLI commands broken"

# 2. Push revert
git push origin main

# 3. Deprecate npm version
npm deprecate meta-governance@1.2.3 "Critical bug - CLI commands broken"

# 4. Update GitHub release
gh release edit v1.2.3 --prerelease \
  --notes "⚠️ Rolled back: CLI commands broken. Use v1.2.2."
```

### Scenario 2: Rollback Due to Security Issue

**Example:** v1.2.3 exposes sensitive data

```bash
# 1. URGENT: Remove release immediately
gh release delete v1.2.3 --yes
git tag -d v1.2.3
git push origin :refs/tags/v1.2.3

# 2. Unpublish from npm (if within 72 hours)
npm unpublish meta-governance@1.2.3

# 3. Revert code
git revert --no-commit v1.2.2..HEAD
git commit -m "revert: SECURITY - rollback v1.2.3 due to data exposure"

# 4. Push immediately
git push origin main

# 5. Notify security team
# Create private security advisory on GitHub
```

### Scenario 3: Partial Rollback (Feature Flag)

**Example:** One feature is broken, but rest of release is good

```bash
# Instead of full rollback, disable problematic feature

# 1. Create hotfix branch
git checkout -b hotfix/disable-broken-feature

# 2. Disable feature via configuration
# Edit config or feature flag file

# 3. Fast-track deploy
npm version patch  # v1.2.3 → v1.2.4
git push origin hotfix/disable-broken-feature
gh pr create --title "hotfix: disable broken feature"

# 4. Merge and deploy immediately
```

---

## Post-Rollback Actions

### Immediate (Within 1 hour)

1. **Create postmortem document:**

   ```bash
   # Use incident response template
   cp docs/operations/incident-response.md POSTMORTEM-$(date +%Y%m%d).md
   ```

2. **Document rollback details:**
   - What went wrong
   - Why rollback was necessary
   - What was rolled back
   - Current system state

3. **Update team:**

   ```
   ✅ ROLLBACK COMPLETE

   Version: Restored to v1.2.2
   Status: System stable
   Next steps: [Link to postmortem]

   Safe to resume normal operations.
   ```

### Short-term (Within 24 hours)

1. **Root cause analysis:**
   - Identify what caused the issue
   - Why wasn't it caught in testing?
   - What tests were missing?

2. **Create GitHub issue:**

   ```bash
   gh issue create \
     --title "Postmortem: v1.2.3 rollback" \
     --label "postmortem,critical" \
     --body "[Link to postmortem document]"
   ```

3. **Fix the issue:**
   - Develop proper fix
   - Add regression tests
   - Enhanced CI checks

### Long-term (Within 1 week)

1. **Complete postmortem:** Include action items to prevent recurrence

2. **Update documentation:**
   - Update runbooks if process issues found
   - Enhance deployment checklist
   - Add new testing requirements

3. **Deploy fix:**
   - Release patched version (e.g., v1.2.4)
   - Include fix for original issue
   - Reference postmortem in release notes

4. **Review deployment process:**
   - Add automated checks that would have caught issue
   - Update rollback procedures based on learnings
   - Conduct blameless retrospective

---

## Preventing Future Rollbacks

### Enhanced Pre-deployment Testing

```bash
# Add to deployment checklist
- [ ] Manual smoke test of critical paths
- [ ] Test in production-like environment
- [ ] Canary deployment (if applicable)
- [ ] Feature flags for risky changes
```

### Automated Checks

```yaml
# .github/workflows/pre-release.yml
name: Pre-release Validation

on:
  push:
    tags:
      - 'v*'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Run smoke tests
        run: npm run test:smoke

      - name: Validate critical paths
        run: npm run test:critical

      - name: Security scan
        run: npm audit --audit-level=high
```

### Deployment Gates

```bash
# Require all checks before tagging
npm run validate:release

# Includes:
# - All tests passing
# - No high-severity vulnerabilities
# - Technical debt within limits
# - Changelog updated
# - Version bumped correctly
```

---

## Communication Templates

### Internal Rollback Notification

```markdown
🚨 ROLLBACK INITIATED 🚨

**Version:** v1.2.3 → v1.2.2
**Reason:** [Critical bug description]
**Impact:** [Who/what is affected]
**ETA:** 15-30 minutes
**Incident:** [Link]
**Owner:** @username

**Actions:**

- ❌ DO NOT deploy new changes
- ❌ DO NOT merge PRs
- ✅ Monitor #incidents channel
- ✅ Report any related issues immediately

Will update when complete.
```

### External Status Update

```markdown
⚠️ System Notice

We've identified an issue with our latest release (v1.2.3) and are rolling back to the previous stable version (v1.2.2).

**Status:** In progress (started HH:MM UTC)
**Expected Resolution:** HH:MM UTC
**Impact:** [Description of user impact]

We'll post updates as the rollback progresses.

Thank you for your patience.
```

### Rollback Complete Notification

```markdown
✅ ROLLBACK COMPLETE

**Restored Version:** v1.2.2
**Status:** System stable and operational
**Duration:** XX minutes

**Next Steps:**

- Postmortem scheduled for [date/time]
- Fix in development
- Expected fix release: [date]

**For users:**

- If you upgraded to v1.2.3, please downgrade:
  `npm install meta-governance@1.2.2`
- All features in v1.2.2 working as expected

Thank you for your patience.
```

---

## Rollback Metrics

Track these metrics to improve rollback process:

| Metric                     | Target     | Current      |
| -------------------------- | ---------- | ------------ |
| Time to decision           | < 30 min   | \_\_\_ min   |
| Time to rollback           | < 30 min   | \_\_\_ min   |
| Rollback success rate      | 100%       | \_\_\_%      |
| Rollbacks per quarter      | < 1        | \_\_\_       |
| Time to fix after rollback | < 48 hours | \_\_\_ hours |

---

## Emergency Contacts

| Role               | Contact  | Availability |
| ------------------ | -------- | ------------ |
| Incident Commander | @alawein | 24/7 for P0  |
| Technical Lead     | @alawein | 24/7 for P0  |
| npm Admin          | @alawein | As needed    |

---

## References

- [Deployment Procedure](./deploy-production.md)
- [Incident Response](../incident-response.md)
- [Postmortem Template](../incident-response.md#postmortem-template)
- [GitHub Release Docs](https://docs.github.com/en/repositories/releasing-projects-on-github)
- [npm unpublish policy](https://docs.npmjs.com/policies/unpublish)
