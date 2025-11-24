# Git Workflow & Branching Strategy for Multi-Org Monorepo

**Date**: November 24, 2025
**Purpose**: Scalable, multi-team Git workflow supporting parallel development
**Scope**: 14+ organizations, 50+ teams, daily releases

---

## 🌳 BRANCHING STRATEGY: Modified Git Flow

### Branch Types

```
main                 (Production-ready)
├─ hotfix/*         (Emergency fixes)
│
develop             (Integration branch)
├─ feature/*        (New features)
├─ bugfix/*         (Bug fixes)
├─ perf/*           (Performance improvements)
├─ refactor/*       (Code refactoring)
├─ chore/*          (Non-functional changes)
└─ test/*           (Experimental/test branches)

release/*           (Release candidates)
│
staging             (Staging environment)
```

### Branch Hierarchy & Flow

```
                           Production
                                ▲
                                │
        ┌─ hotfix/issue-123 ────┤
        │                        │
main ◄──┴─────────────────────────┤
 │                                │
 │      release/v1.2.0 ─────────┬─┤
 │       │                       │ │
develop │◄──────┬────────────────┤ │
 │       │      │                │ │
 │ feature/org-123  ─────┐       │ │
 │       │      │        │       │ │
 │       │      │ hotfix/perf ──┬┤
 │       │      │   │           ││
 │   bugfix/123 ◄──┤            ││
 │    (from any org)         Staging
 │
 └──alaweimm90/* (org-specific branches)
    └──alaweimm90-science/* (org-specific branches)
```

---

## 📋 BRANCH NAMING CONVENTIONS

### Format: `{type}/{org-or-team}/{description}`

#### Type Prefixes

| Type        | Purpose            | Lifecycle              | Merge Target          |
| ----------- | ------------------ | ---------------------- | --------------------- |
| `feature/`  | New features       | Long-lived (1-2 weeks) | `develop`             |
| `bugfix/`   | Bug fixes          | Short-lived (1-3 days) | `develop` then `main` |
| `hotfix/`   | Production fixes   | Urgent (same day)      | `main` + `develop`    |
| `perf/`     | Performance work   | Medium (3-5 days)      | `develop`             |
| `refactor/` | Code cleanup       | Short (2-4 days)       | `develop`             |
| `chore/`    | Config, deps, docs | Short (1 day)          | `develop`             |
| `release/`  | Release candidates | Temporary (1 week)     | `main` then `develop` |
| `test/`     | Experiments        | Very short (ad-hoc)    | Deleted               |

#### Naming Examples

```
# Good examples
feature/alaweimm90/api-gateway-rate-limiting
feature/science/ml-pipeline-optimization
bugfix/security/jwt-validation-check
hotfix/alaweimm90/database-connection-leak
perf/mobile/reduce-bundle-size
refactor/core/agent-orchestrator-cleanup
chore/dependencies/upgrade-typescript-5.3

# BAD examples (don't do this)
feature/new-stuff                  ❌ Too vague, no team/org
fix/issue                         ❌ Too vague, no context
feature/alaweimm90-api-gateway    ❌ Using underscore instead of slash
FEATURE/ALAWEIMM90/TEST           ❌ Uppercase
feature/alaweimm90/api_gateway    ❌ Mix of separators
```

---

## 🔄 WORKFLOW BY SCENARIO

### Scenario 1: New Feature in Organization

```bash
# 1. Create feature branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/alaweimm90/user-authentication

# 2. Make changes, commit frequently
git add .
git commit -m "feat(auth): add JWT token validation

- Implement token validation middleware
- Add unit tests for token parsing
- Update error handling

Closes #123"

# 3. Keep updated with develop
git fetch origin
git rebase origin/develop  # Prefer rebase for clean history

# 4. Push and create Pull Request
git push -u origin feature/alaweimm90/user-authentication
# Then create PR on GitHub with description

# 5. After PR approval, merge to develop
# (GitHub UI merge)

# 6. Delete branch
git branch -d feature/alaweimm90/user-authentication
git push origin --delete feature/alaweimm90/user-authentication
```

### Scenario 2: Urgent Production Fix

```bash
# 1. Create hotfix branch from main (CRITICAL)
git checkout main
git pull origin main
git checkout -b hotfix/database-leak

# 2. Make minimal fix
git add .
git commit -m "fix(db): prevent connection pool exhaustion"

# 3. Push and create PR targeting main
git push -u origin hotfix/database-leak
# Create PR targeting 'main' branch

# 4. After approval, merge to BOTH main and develop
git checkout main
git merge hotfix/database-leak
git push origin main

# Then update develop
git checkout develop
git pull origin develop
git merge main
git push origin develop

# 5. Tag the release
git tag -a v1.0.1 -m "Hotfix: database connection leak"
git push origin v1.0.1

# 6. Delete hotfix branch
git branch -d hotfix/database-leak
git push origin --delete hotfix/database-leak
```

### Scenario 3: Shared Code Used by Multiple Organizations

```bash
# 1. Feature in core packages affects multiple orgs
git checkout develop
git checkout -b feature/core/agent-orchestrator-v2

# 2. Make changes in /packages/
# (Changes apply to all orgs using this package)

# 3. Update both org-specific tests
cd alaweimm90 && pnpm test
cd ../alaweimm90-science && pnpm test

# 4. Create PR with multiple org test verification
git push -u origin feature/core/agent-orchestrator-v2

# PR Description includes:
# "Affects: alaweimm90, alaweimm90-science, [others]
#  Tests passing in: [list all]"

# 5. Merge after all org tests pass
```

### Scenario 4: Release Cycle

```bash
# 1. When ready to release, create release branch
git checkout develop
git checkout -b release/v1.2.0

# 2. Update version numbers
pnpm version v1.2.0
git add .
git commit -m "chore: release v1.2.0"

# 3. Update CHANGELOG and generate release notes
node scripts/generate-changelog.js --version=v1.2.0
git add CHANGELOG.md
git commit -m "docs: add changelog for v1.2.0"

# 4. Create PR to main
git push -u origin release/v1.2.0
# Create PR: release/v1.2.0 → main

# 5. After main merge, merge back to develop
git checkout develop
git pull origin develop
git merge main
git push origin develop

# 6. Tag and create GitHub Release
git tag -a v1.2.0 -m "Release v1.2.0"
git push origin v1.2.0
# Also create on GitHub UI with release notes

# 7. Delete release branch
git branch -d release/v1.2.0
git push origin --delete release/v1.2.0
```

---

## 📝 COMMIT MESSAGE STANDARDS

### Format: Conventional Commits

```
{type}({scope}): {subject}

{body}

{footer}
```

#### Type

- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `refactor`: Code refactoring
- `test`: Test addition/modification
- `docs`: Documentation
- `style`: Code style (no logic change)
- `chore`: Build, dependencies, config

#### Scope (Optional but Recommended)

- Package name: `agent-core`, `mcp-core`, etc.
- Organization: `alaweimm90`, `science`
- Feature area: `auth`, `database`, `api`

#### Subject (Imperative Mood)

```
✅ add user authentication
✅ fix database connection leak
✅ improve build performance

❌ added user authentication
❌ fixes database connection leak
❌ improved build performance
```

#### Body (Optional)

Explain WHAT and WHY, not HOW:

```
feat(auth): add JWT token validation

JWT tokens are validated on every API request to ensure
only authenticated users can access protected resources.

This implements RS256 verification against the JWKS endpoint
to ensure tokens are properly signed by the auth service.
```

#### Footer (Optional)

Reference issues and breaking changes:

```
fix(api): handle concurrent requests

Fixes #123
Refs #124, #125
Breaking-Change: API now requires Authorization header
```

### Full Example

```
feat(alaweimm90/api): add rate limiting to user endpoints

- Implement token bucket rate limiting (100 requests/min)
- Add X-RateLimit-* response headers
- Log rate limit violations for monitoring
- Add integration tests for rate limiting

Fixes #456
Closes #457
Breaking-Change: Requests exceeding rate limit return 429 status
```

---

## 🔀 MERGE STRATEGIES

### Strategy by Branch Type

| Branch Type  | Strategy     | Reason                             |
| ------------ | ------------ | ---------------------------------- |
| `feature/*`  | Squash       | Clean commit history               |
| `bugfix/*`   | Squash       | Single logical fix                 |
| `hotfix/*`   | Merge commit | Preserve hotfix history            |
| `release/*`  | Merge commit | Preserve release point             |
| `refactor/*` | Squash       | Single refactor operation          |
| `perf/*`     | Merge commit | Preserve perf optimization history |

### Squash Merge Example

```bash
# Automatic in GitHub UI:
# 1. Open PR
# 2. Click "Squash and merge"
# 3. Edit commit message to match standards
# 4. Confirm

# Manual CLI:
git checkout develop
git merge --squash feature/alaweimm90/new-feature
git commit -m "feat(alaweimm90): add new feature"
git push origin develop
```

---

## 🚀 RELEASE & VERSIONING

### Semantic Versioning (semver)

```
MAJOR.MINOR.PATCH
│      │      └── Patch: bug fixes (backward compatible)
│      └────────── Minor: new features (backward compatible)
└───────────────── Major: breaking changes
```

#### Examples

```
v0.1.0 → v0.1.1   # Bug fix
v0.1.1 → v0.2.0   # New feature
v0.2.0 → v1.0.0   # Breaking change
v1.0.0 → v1.0.1   # Hotfix
```

### Version Management with Changesets

```bash
# When you make a change that needs to be released:
pnpm exec changeset

# Prompted with:
# - Which packages changed? (select via checkbox)
# - Semver bump? (patch, minor, major)
# - Change description (for changelog)

# Creates .changeset/{id}.md:
---
"@monorepo/agent-core": minor
"@monorepo/context-provider": patch
---

Add support for custom agent middleware
```

### Pre-Release Versions

For beta/RC releases:

```
v1.0.0-alpha.1    # Alpha (development)
v1.0.0-beta.1     # Beta (feature complete)
v1.0.0-rc.1       # Release candidate

# Create with:
git tag -a v1.0.0-beta.1 -m "Beta release"
git push origin v1.0.0-beta.1

# npm install from pre-release:
pnpm add @monorepo/agent-core@beta
```

---

## 🔐 BRANCH PROTECTION RULES

### Configure in GitHub Settings

**Main Branch** (`main`):

```
✓ Require pull request reviews (2 approvals)
✓ Dismiss stale pull request approvals
✓ Require status checks to pass (all CI/CD)
✓ Require branches to be up to date before merging
✓ Require code review from code owners
✓ Restrict who can push to matching branches
✓ Allow auto-merge (if tests pass)
✓ Automatically delete head branches
```

**Develop Branch** (`develop`):

```
✓ Require pull request reviews (1 approval)
✓ Require status checks to pass (build + tests)
✓ Require branches to be up to date
✓ Allow auto-merge
✓ Automatically delete head branches
```

**Release Branches** (`release/*`):

```
✓ Require pull request reviews (1 approval)
✓ Require status checks to pass
✓ Require branches to be up to date
✓ Automatically delete head branches
```

---

## 🏷️ TAG STRATEGY

### Tag Format

```
v{MAJOR}.{MINOR}.{PATCH}
v{MAJOR}.{MINOR}.{PATCH}-{prerelease}

Examples:
v1.0.0
v1.2.3-alpha.1
v1.2.3-beta.2
v1.2.3-rc.1
```

### Create Tags

```bash
# Annotated tags (recommended)
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Lightweight tags (don't use for releases)
git tag v1.0.0  # Don't do this for releases
```

### Tag for Each Release

Every version pushed to npm should have corresponding tag:

```
pnpm version v1.2.0  # Updates package.json + creates tag
git push origin --follow-tags  # Push commits and tags
```

---

## 🔍 CODE REVIEW PROCESS

### PR Guidelines

**Before Creating PR**:

1. Pull latest develop
2. Ensure all tests pass locally
3. Run linter and format code
4. Add tests for new functionality

**PR Description Template**:

```markdown
## Description

Brief description of the change.

## Motivation & Context

Why is this change needed? What issue does it solve?

## Related Issues

Fixes #123
Relates to #456

## Type of Change

- [ ] New feature
- [ ] Bug fix
- [ ] Breaking change
- [ ] Documentation update

## Testing

How was this tested?

- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual testing

## Checklist

- [ ] Code follows style guidelines
- [ ] No new warnings generated
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Affected Organizations

- [ ] alaweimm90
- [ ] alaweimm90-science
- [ ] [others]
```

### Review Checklist

**Reviewer Checks**:

- [ ] Code is readable and maintainable
- [ ] Tests are comprehensive
- [ ] No obvious bugs or security issues
- [ ] Follows project conventions
- [ ] Documentation is accurate
- [ ] Performance implications considered
- [ ] No hardcoded values/credentials

### Approval Requirements

```
main branch:   2 approvals minimum
develop:       1 approval minimum
hotfix/*:      1 approval + on-call engineer
release/*:     2 approvals + release manager
```

---

## 🔄 HANDLING MERGE CONFLICTS

### Strategy: Keep Feature Branch Updated

```bash
# Regularly rebase on develop to avoid conflicts
git fetch origin
git rebase origin/develop

# If conflicts occur during rebase:
# 1. Manually resolve conflicts in files
# 2. Mark as resolved
git add conflicted-file.ts
# 3. Continue rebase
git rebase --continue
# 4. Force push (safe since you're just rebasing your work)
git push -f origin feature/your-feature
```

### Conflict Resolution Rules

```typescript
// For package.json version conflicts:
// → Keep the newer version (from develop)

// For code conflicts:
// → Prefer develop version initially
// → Discuss with feature author
// → Make intentional decision

// For lock files (pnpm-lock.yaml):
// → Always accept develop version
// → Rerun pnpm install to regenerate if needed
```

---

## 📊 GITHUB WORKFLOW TEMPLATES

### `.github/PULL_REQUEST_TEMPLATE.md`

```markdown
## Type of Change

- [ ] ✨ New feature
- [ ] 🐛 Bug fix
- [ ] 📚 Documentation
- [ ] 🔧 Configuration
- [ ] ♻️ Refactoring
- [ ] 🚀 Performance
- [ ] 🧪 Testing

## Description

<!-- Brief description of changes -->

## Related Issues

<!-- Fix #123 or Ref #456 -->

## Testing

<!-- How was this tested? -->

## Checklist

- [ ] Tests pass locally
- [ ] Code style verified
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tested in affected organizations
```

---

## ⚠️ ANTI-PATTERNS & DON'Ts

### ❌ Things to Avoid

```bash
# Don't commit directly to main or develop
git push origin feature -f  # Then push to main directly
# Instead → Create PR and merge via GitHub

# Don't use force push to shared branches
git push -f origin develop
# It breaks other developers' repos
# Only force push to your feature branch

# Don't commit secrets
git add .env  # DON'T
# Use .env.example and git-secrets pre-commit hook

# Don't merge without tests passing
# GitHub will prevent this if rules configured properly

# Don't use vague commit messages
git commit -m "fix stuff"  # ❌
# Instead: "fix(auth): prevent double login on refresh token expiry"

# Don't let PRs get too large
# Review becomes difficult, testing harder
# Break into multiple smaller PRs

# Don't ignore failing tests to merge
# They'll fail in production too
```

---

## 🔄 CONTINUOUS INTEGRATION GATES

### Automated Checks Before Merge

```yaml
# GitHub checks that must pass:
- ✅ All PR checks pass (lint, test, build)
- ✅ No merge conflicts
- ✅ Branch is up to date with base
- ✅ Required approvals obtained
- ✅ No secret patterns detected (git-secrets)
- ✅ Code coverage threshold met
- ✅ Performance regression tests pass
```

---

## 📈 METRICS & MONITORING

### Track These Git Metrics

```
- Branch creation frequency (how often new features start)
- PR review time (how long reviews take)
- Merge frequency (deployment rate)
- Hotfix frequency (production issues)
- Conflict frequency (integration problems)
- Time to merge (PR cycle time)
```

### Example Metrics Dashboard

```
Weekly Metrics:
- Average PR review time: 2h 15m
- Median merge time: 1h 30m
- Total merges to main: 8
- Hotfixes: 1 (emergency db fix)
- Conflicts: 2 (expected for fast-moving codebase)

PR Statistics:
- Open PRs: 3
- Waiting for review: 2
- Waiting for updates: 1
- Average PR size: 180 lines changed
```

---

## ✅ IMPLEMENTATION CHECKLIST

- [ ] Document branch naming convention
- [ ] Set up branch protection rules
- [ ] Configure GitHub CODEOWNERS file
- [ ] Set PR template
- [ ] Establish code review guidelines
- [ ] Set up git hooks (husky, pre-commit)
- [ ] Configure Changesets for versioning
- [ ] Document release process
- [ ] Create team workflows documentation
- [ ] Run migration for existing branches

---

## 🎊 SUMMARY

### Key Principles

1. **Branch per feature**: Isolate work
2. **Frequent integration**: Keep branches short-lived
3. **Always from develop**: Maintain single source of truth
4. **Squash for cleanliness**: Keep history readable
5. **Tag every release**: Know what's in production
6. **Protect main**: Require reviews and tests
7. **Clear commits**: Use conventional commit messages

### Typical Cycle (Feature to Production)

```
1. Create feature branch (develop → feature/...)
2. Make changes + commit (conventional messages)
3. Keep updated with git rebase origin/develop
4. Create PR with description + checklist
5. Team reviews + approves
6. Squash merge to develop
7. Create release branch when ready
8. Merge release → main (with tag)
9. Merge main → develop (for bookkeeping)
```

**Expected Duration**: Feature to production = 3-5 days

---

**Status**: ✅ GIT WORKFLOW DOCUMENTED
**Next**: Common Pitfalls & Security Guide
