# Runbook: Dependency Updates

**Purpose:** Safely update project dependencies
**Frequency:** Weekly (automated via Dependabot)
**Duration:** 15-30 minutes

---

## Prerequisites

- [ ] Node.js v22.20.0 installed (check `.nvmrc`)
- [ ] Write access to repository
- [ ] All tests passing on main branch

## Procedure

### 1. Review Dependabot PRs

```bash
# List open Dependabot PRs
gh pr list --label "dependencies"
```

**For each PR:**

- Review changelog/release notes
- Check for breaking changes
- Review automated test results

### 2. Manual Dependency Check

```bash
# Check for outdated packages
npm outdated

# Check for security vulnerabilities
npm audit

# Fix auto-fixable vulnerabilities
npm audit fix
```

### 3. Update Dependencies

#### Patch/Minor Updates (Low Risk)

```bash
# Update all patch and minor versions
npm update

# Verify no breaking changes
npm test
npm run lint
npm run type-check
```

#### Major Updates (High Risk)

```bash
# Update specific package
npm install <package>@latest

# Run full test suite
npm test

# Check for type errors
npm run type-check

# Manual regression testing
npm run dev
# Test critical workflows manually
```

### 4. Validation

- [ ] All tests pass: `npm test`
- [ ] No lint errors: `npm run lint`
- [ ] No type errors: `npm run type-check`
- [ ] Technical debt gate passes: `python automation/debt_gate.py --scan automation/debt_scan.json`
- [ ] Build succeeds: `npm run build` (if applicable)

### 5. Commit Changes

```bash
git add package.json package-lock.json
git commit -m "chore(deps): update dependencies

- Update <package> from vX.Y to vX.Z
- Fix security vulnerabilities
- All tests passing

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

git push
```

---

## Rollback Procedure

If issues arise after update:

```bash
# Revert commit
git revert HEAD

# Or restore package files
git checkout HEAD~1 -- package.json package-lock.json
npm ci

# Push revert
git push
```

---

## Common Issues

### Issue: Tests fail after update

**Solution:**

1. Check package changelog for breaking changes
2. Update test code to match new API
3. If unfixable, pin to previous version

### Issue: Type errors appear

**Solution:**

1. Update `@types/*` packages
2. Adjust tsconfig.json if needed
3. Update code to match new types

### Issue: Build size increases significantly

**Solution:**

1. Check if new dependencies are tree-shakeable
2. Use bundlesize to track growth
3. Consider alternative packages

---

## Post-Update Monitoring

Monitor for 24 hours after deployment:

- CI/CD pipeline stability
- Test pass rates
- Build times
- Bundle sizes

---

## References

- [npm update docs](https://docs.npmjs.com/cli/v8/commands/npm-update)
- [Dependabot configuration](../../.github/dependabot.yml)
- [Security policy](../../SECURITY.md)
