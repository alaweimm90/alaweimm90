# Runbook: Production Deployment

**Purpose:** Deploy changes to production (GitHub releases, published packages)
**Frequency:** As needed (typically weekly or per milestone)
**Duration:** 30-60 minutes

---

## Prerequisites

- [ ] All tests passing on main branch
- [ ] No open P0/P1 technical debt items
- [ ] Security scan clean (0 vulnerabilities)
- [ ] Technical debt gate passing
- [ ] Code review approved
- [ ] Changelog updated
- [ ] Version bumped appropriately

---

## Pre-deployment Checklist

### 1. Verify Code Quality

```bash
# Ensure on main branch and up to date
git checkout main
git pull origin main

# Run full test suite
npm test
npm run type-check
npm run lint

# Run technical debt scan
python automation/debt_cli.py scan --path .
python automation/debt_gate.py --scan automation/debt_scan.json --env prod

# Security audit
npm audit
```

All checks must pass before proceeding.

### 2. Review Changes Since Last Release

```bash
# View commits since last release
LAST_TAG=$(git describe --tags --abbrev=0)
echo "Changes since $LAST_TAG:"
git log $LAST_TAG..HEAD --oneline --no-merges

# View detailed diff
git diff $LAST_TAG..HEAD --stat
```

### 3. Verify CI/CD Status

```bash
# Check recent workflow runs
gh run list --limit 5 --json workflowName,conclusion,startedAt

# Ensure all passing
gh run list --limit 1 --json conclusion --jq '.[] | select(.conclusion != "success")'
# Should return empty
```

---

## Deployment Procedure

### Step 1: Version Bump

**Semantic Versioning:**

- **Patch** (1.0.0 → 1.0.1): Bug fixes, minor changes
- **Minor** (1.0.0 → 1.1.0): New features, backward compatible
- **Major** (1.0.0 → 2.0.0): Breaking changes

```bash
# Choose appropriate version bump
npm version patch  # or minor, or major

# This creates a version commit and git tag
# Example: v1.2.3
```

**Manual version bump (if needed):**

```bash
# Edit package.json manually
# Then commit and tag
git add package.json package-lock.json
git commit -m "chore(release): bump version to 1.2.3"
git tag -a v1.2.3 -m "Release v1.2.3"
```

### Step 2: Generate Release Notes

```bash
# Use GitHub CLI to generate release notes
gh release create v1.2.3 --generate-notes --draft

# Or create manually with changelog
cat > release-notes.md << 'EOF'
# Release v1.2.3

## 🚀 Features

- Feature 1: Description
- Feature 2: Description

## 🐛 Bug Fixes

- Fix 1: Description
- Fix 2: Description

## 📚 Documentation

- Docs update 1
- Docs update 2

## 🔒 Security

- Security improvement 1

## Contributors

Thanks to @contributor1, @contributor2

EOF
```

### Step 3: Create GitHub Release

**Option A: Using GitHub CLI (Recommended)**

```bash
# Create draft release
gh release create v1.2.3 \
  --title "Release v1.2.3" \
  --notes-file release-notes.md \
  --draft

# Review draft at: https://github.com/YOUR_ORG/YOUR_REPO/releases

# Publish when ready
gh release edit v1.2.3 --draft=false
```

**Option B: Using GitHub Web Interface**

1. Go to `https://github.com/YOUR_ORG/YOUR_REPO/releases/new`
2. Choose tag: `v1.2.3`
3. Set title: `Release v1.2.3`
4. Paste release notes
5. Click "Save draft"
6. Review and publish

### Step 4: Publish to npm (If Applicable)

**⚠️ Only if CLI tools are intended for npm distribution**

```bash
# Ensure logged in to npm
npm whoami

# Dry run first
npm publish --dry-run

# Review what will be published
npm pack
tar -tzf meta-governance-1.2.3.tgz

# Publish to npm
npm publish

# Verify published
npm view meta-governance version
```

**Skip this step if CLI tools are repository-only.**

### Step 5: Deploy Documentation

```bash
# If using GitHub Pages for docs
npm run docs:build  # If applicable
git checkout gh-pages
git merge main --strategy-option theirs
git push origin gh-pages

# Or deploy to external docs site
# (depends on your docs hosting)
```

### Step 6: Push Tags

```bash
# Push version tag to GitHub
git push origin v1.2.3

# Or push all tags
git push --tags

# Verify tag on GitHub
gh release list
```

### Step 7: Update Related Repositories (If Any)

```bash
# If other repos depend on this one
# Create PRs to update dependency versions

# Example:
cd ../dependent-repo
npm install meta-governance@1.2.3
git commit -am "chore(deps): update meta-governance to v1.2.3"
gh pr create --title "chore(deps): update meta-governance"
```

---

## Post-Deployment Verification

### 1. Verify GitHub Release

```bash
# Check release is public
gh release view v1.2.3

# Verify release URL works
open https://github.com/YOUR_ORG/YOUR_REPO/releases/tag/v1.2.3
```

### 2. Verify npm Package (If Published)

```bash
# Install from npm in fresh directory
mkdir /tmp/test-install
cd /tmp/test-install
npm install meta-governance@1.2.3

# Test CLI works
npx meta-governance --version
# Should output: 1.2.3
```

### 3. Verify Documentation

```bash
# Check docs site is updated
curl -I https://your-docs-site.com
# Verify 200 OK

# Check changelog is accessible
open https://github.com/YOUR_ORG/YOUR_REPO/blob/main/CHANGELOG.md
```

### 4. Monitor for Issues

**First 24 Hours:**

- [ ] Monitor GitHub issues for bug reports
- [ ] Check CI/CD status on main branch
- [ ] Review download/install metrics (if applicable)
- [ ] Monitor npm download stats: `npm stats meta-governance`

**First Week:**

- [ ] Gather user feedback
- [ ] Address any critical bugs immediately
- [ ] Plan hotfix release if needed

---

## Deployment Environments

This repository doesn't have traditional "staging" and "production" environments. Instead:

| Environment | Description       | Branch           | Validation                |
| ----------- | ----------------- | ---------------- | ------------------------- |
| Development | Local testing     | feature branches | Pre-commit hooks          |
| Integration | PR testing        | PR branches      | CI/CD full suite          |
| Production  | Published release | main + tags      | All gates + manual review |

---

## Deployment Frequency

**Recommended Cadence:**

- **Patch releases:** As needed for critical bugs (within 24-48 hours)
- **Minor releases:** Weekly or bi-weekly for features
- **Major releases:** Monthly or quarterly for breaking changes

**Avoid:**

- Deploying on Fridays (unless critical hotfix)
- Deploying during holidays
- Deploying without CI passing

---

## Communication

### Internal Notification

```bash
# Post in team channel
Team: Release v1.2.3 is live! 🚀

Changes:
- Feature 1: [description]
- Bug fix: [description]

Release notes: https://github.com/ORG/REPO/releases/tag/v1.2.3
```

### External Announcement (If Applicable)

```markdown
# Example Twitter/LinkedIn post

🚀 Released meta-governance v1.2.3

New features:
✅ [Feature 1]
✅ [Feature 2]

Upgrade: npm install meta-governance@latest

Docs: https://link-to-docs
```

---

## Rollback Procedure

If critical issues are discovered post-deployment:

**See:** [rollback.md](./rollback.md)

**Quick rollback:**

```bash
# Unpublish from npm (within 72 hours)
npm unpublish meta-governance@1.2.3

# Delete GitHub release
gh release delete v1.2.3

# Remove git tag
git tag -d v1.2.3
git push origin :refs/tags/v1.2.3
```

**Better approach:** Deploy a hotfix release instead of rollback.

---

## Hotfix Procedure

For critical bugs in production:

1. **Create hotfix branch:**

   ```bash
   git checkout main
   git checkout -b hotfix/critical-bug-fix
   ```

2. **Apply minimal fix:**
   - Fix only the critical issue
   - Add regression test
   - Update changelog

3. **Fast-track testing:**

   ```bash
   npm test
   npm run lint
   npm run type-check
   ```

4. **Emergency release:**

   ```bash
   npm version patch  # e.g., 1.2.3 → 1.2.4
   git push origin hotfix/critical-bug-fix
   gh pr create --title "hotfix: [description]" --base main
   ```

5. **Fast-track review:** Request immediate review from maintainer

6. **Deploy immediately after merge:**
   Follow standard deployment procedure with expedited timeline

---

## Deployment Metrics

Track these metrics to improve deployment process:

| Metric                  | Target    | Current      |
| ----------------------- | --------- | ------------ |
| Time to deploy          | < 30 min  | \_\_\_ min   |
| Deployment frequency    | 1-2x/week | \_\_\_x/week |
| Deployment success rate | > 95%     | \_\_\_%      |
| Time to rollback        | < 15 min  | \_\_\_ min   |
| Hotfix frequency        | < 1/month | \_\_\_/month |

**Review quarterly and optimize bottlenecks.**

---

## Automation Opportunities

### Future Improvements

1. **Automated changelog generation:**

   ```bash
   # Use conventional-changelog
   npx conventional-changelog-cli -p angular -i CHANGELOG.md -s
   ```

2. **Automated release creation:**

   ```yaml
   # .github/workflows/release.yml
   # Trigger on tag push, auto-create release
   ```

3. **Automated npm publish:**

   ```yaml
   # CD workflow to publish on release
   # Uses NPM_TOKEN secret
   ```

4. **Deployment notifications:**
   - Slack webhook on release
   - Email to subscribers
   - GitHub Discussions post

---

## Emergency Contacts

| Role            | Contact  | Availability       |
| --------------- | -------- | ------------------ |
| Release Manager | @alawein | 24/7 for P0 issues |
| CI/CD Support   | @alawein | Business hours     |
| npm Access      | @alawein | As needed          |

---

## Related Documentation

- [Rollback Procedure](./rollback.md)
- [Incident Response](../incident-response.md)
- [Dependency Updates](./dependency-updates.md)
- [GitHub Releases Documentation](https://docs.github.com/en/repositories/releasing-projects-on-github)
- [npm Publishing](https://docs.npmjs.com/cli/v9/commands/npm-publish)
