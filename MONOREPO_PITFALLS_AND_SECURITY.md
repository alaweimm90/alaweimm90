# Common Pitfalls & Security Considerations for Multi-Organization Monorepo

**Date**: November 24, 2025
**Purpose**: Prevent common mistakes and ensure secure operations
**Scope**: Development, deployment, and operational security

---

## ⚠️ PART 1: COMMON PITFALLS

### Pitfall 1: Dependency Version Hell

**Problem**: Different organizations require different versions of the same dependency

```
alaweimm90 needs: express@4.18.0 (stable)
alaweimm90-science needs: express@5.0.0-beta (for new features)
Result: pnpm can't resolve → build fails for both
```

**Why It Happens**:

- Teams evolve at different speeds
- Different risk tolerances
- Lack of coordination on upgrades

**Impact**: 🔴 CRITICAL

- Build completely breaks
- Can't install dependencies
- Blocks all developers

**Solutions**:

```json
{
  "pnpm": {
    "overrides": {
      "express": "^4.18.0"
    }
  }
}
```

**Prevention**:

- ✅ Use pnpm workspace overrides for harmony
- ✅ Establish "core version" for shared dependencies
- ✅ Quarterly dependency upgrade reviews
- ✅ Communicate breaking changes early

**Detection**:

```bash
# Automated check in CI
pnpm list --depth=0 | grep "duplicate"
# Warn if duplicates detected
```

---

### Pitfall 2: Circular Dependencies

**Problem**: Package A imports from B, which imports from A

```
packages/agent-core/src/agent.ts:
  import { ContextProvider } from '@monorepo/context-provider'

packages/context-provider/src/context.ts:
  import { BaseAgent } from '@monorepo/agent-core'  ← CIRCULAR!
```

**Why It Happens**:

- Not paying attention to dependency direction
- Coupling grows over time
- No automated checks in place

**Impact**: 🔴 CRITICAL

- Module resolution fails at build time
- TypeScript compiler errors
- Runtime errors: "Cannot find module"

**Detection Tools**:

```bash
# Install dependency checker
pnpm add -D madge

# Add to package.json scripts
"check:cycles": "madge --circular packages/*/src/index.ts"

# Run in CI
pnpm check:cycles || exit 1
```

**Solutions**:

1. **Extract Common Types**:

```typescript
// packages/shared-types/src/index.ts
export interface IAgent { ... }
export interface IContext { ... }

// packages/agent-core/src/index.ts
import { IContext } from '@monorepo/shared-types'

// packages/context-provider/src/index.ts
import { IAgent } from '@monorepo/shared-types'
// ✅ No circular import!
```

2. **Invert Dependencies**:

```typescript
// Instead of A → B → A
// Make it:
// A → C (both import from C)
```

---

### Pitfall 3: Build Time Explosion

**Problem**: Monorepo takes 30+ minutes to build

**Why It Happens**:

```
❌ No caching
❌ Duplicate builds (not incremental)
❌ No parallel builds
❌ Large node_modules (dependencies duplicated per package)
❌ Heavy linting/testing on every file
```

**Impact**: 🔴 HIGH

- 2 hour developer wait time per day
- Slow CI/CD feedback loop
- Developer frustration → context switching

**Before/After**:

```
BEFORE: 45 minutes (sequential, no caching)
└─ Install: 8 min
└─ Lint: 12 min
└─ Type check: 15 min
└─ Test: 20 min
└─ Build: 10 min

AFTER: 6 minutes (parallel, cached, incremental)
└─ Install: 1 min (cached)
└─ Lint: 4 min (parallel with Turbo)
└─ Type check: 2 min (parallel)
└─ Test: 3 min (parallel)
└─ Build: 1 min (cached)
```

**Solutions**:

1. **Enable Turbo Caching**:

```json
{
  "turbo": {
    "pipeline": {
      "build": {
        "dependsOn": ["^build"],
        "outputs": ["dist/**"],
        "cache": true
      }
    }
  }
}
```

2. **Parallel Job Execution**:

```yaml
# GitHub Actions matrix
strategy:
  matrix:
    package: [mcp-core, agent-core, context-provider]
  max-parallel: 10 # Run 10 jobs simultaneously
```

3. **Optimize pnpm**:

```bash
# Use shamefully-hoist for fewer node_modules
echo "shamefully-hoist=true" >> .npmrc
```

**Prevention**:

- ✅ Monitor build times in CI (chart weekly)
- ✅ Set alerts if build time increases 20%+
- ✅ Profile slow steps: `time pnpm test`
- ✅ Use Turbo dashboard: https://vercel.com/docs/concepts/monorepos/turbo

---

### Pitfall 4: Flaky Tests

**Problem**: Tests pass locally but fail in CI 30% of the time

**Symptoms**:

```
✅ Test passed (local)
❌ Test failed (CI)
❌ Test passed (CI re-run)
✅ Test passed (local again)
```

**Why It Happens**:

- Race conditions (async timing)
- Shared database state between tests
- Environment differences (timezone, locale)
- Randomized test order
- Timing-dependent assertions

**Impact**: 🔴 HIGH

- Blocks deployments
- Developers "just re-run" instead of fixing
- False confidence in test suite

**Common Causes**:

```typescript
// ❌ BAD: Timing-dependent
test('API returns data', async () => {
  setTimeout(() => {
    expect(data).toBeDefined(); // Flaky!
  }, 100);
});

// ✅ GOOD: Wait for actual condition
test('API returns data', async () => {
  await waitFor(() => {
    expect(data).toBeDefined();
  });
});

// ❌ BAD: Shared state
let counter = 0;
test('increment counter', () => {
  counter++;
  expect(counter).toBe(1); // Flaky if run with other tests!
});

// ✅ GOOD: Isolated state
test('increment counter', () => {
  const counter = 0;
  // ... test with local counter
});
```

**Detection & Prevention**:

```bash
# Run tests 100 times to find flakiness
for i in {1..100}; do
  pnpm test --seed=$RANDOM || break
done

# Use Jest's seed for reproducibility
pnpm test --seed=12345
```

**Solutions**:

- ✅ Use `waitFor()` for async assertions
- ✅ Mock external dependencies (databases, APIs)
- ✅ Run tests in random order: `--randomOrder`
- ✅ Use isolated test data per test
- ✅ Set explicit timeouts (not arbitrary)
- ✅ Use test containers for databases

---

### Pitfall 5: Test Cache Invalidation Issues

**Problem**: CI passes, but test results are wrong (cached results)

**Why It Happens**:

```
# Test changes, but cache still returns old results
Test: "user authentication works"
(Test code changed, but cache key is same)
Old Result: ✅ passed
New Result: ❌ failed
But cache returns: ✅ passed (wrong!)
```

**Impact**: 🔴 HIGH

- False positive test results
- Bug gets merged to main
- Discovered in production

**Solutions**:

```json
{
  "turbo": {
    "globalDependencies": [
      "pnpm-lock.yaml",      // Include lock file in hash
      "tsconfig.json",
      "jest.config.js"       // Include test config!
    ],
    "pipeline": {
      "test": {
        "cache": false       // Disable test caching
        // Or enable with strict dependencies:
        "inputs": [
          "src/**/*.ts",
          "tests/**/*.ts",
          "jest.config.js",
          "package.json"
        ]
      }
    }
  }
}
```

**Prevention**:

- ✅ Disable caching for test jobs
- ✅ Include all test inputs in cache key
- ✅ Run tests without cache on main branch

---

### Pitfall 6: Forgotten Database Migrations

**Problem**: Deploy code that depends on new database column, but column doesn't exist

```
Code: SELECT * FROM users WHERE role = $1
DB Schema: users table has no 'role' column
Result: Runtime error in production
```

**Why It Happens**:

- Migration not created
- Migration created but not deployed
- Different orgs have different schemas
- Org-specific migrations in shared code

**Impact**: 🔴 CRITICAL

- 500 errors in production
- Data corruption if writes attempted
- Requires emergency rollback

**Solutions**:

1. **Database Schema Versioning**:

```
migrations/
├── 001_create_users_table.sql
├── 002_add_role_to_users.sql
├── 003_create_organizations_table.sql
```

2. **Migration Tracking Per Org**:

```typescript
// src/db/migrations.ts
const migrations = {
  'alaweimm90': [001, 002, 003],
  'alaweimm90-science': [001, 002]
  // Different orgs might be at different versions
};
```

3. **Pre-Deployment Checks**:

```bash
# Before deploying code changes:
# 1. Check schema version matches code expectations
node scripts/validate-schema.js
# 2. Run pending migrations
pnpm migrate:latest
# 3. Run smoke tests against new schema
pnpm test:smoke
```

**Prevention**:

- ✅ Code review: check for schema changes
- ✅ Require migration for schema changes
- ✅ Test against real database in CI
- ✅ Validate schema on app startup

---

### Pitfall 7: Monorepo Size Explosion

**Problem**: Repository becomes too large to work with

**Symptoms**:

```
.git directory: 5+ GB
Clone time: 30+ minutes
Checkout slow
IDE performance degraded
```

**Why It Happens**:

- Large binary files committed (node_modules, build artifacts)
- Backups and archives committed instead of .gitignored
- Large test fixtures
- Old logs and caches
- Media files (images, videos)

**Impact**: 🟡 HIGH

- Slow operations for everyone
- Hard drive space issues
- CI/CD caching inefficient

**Your Monorepo Status**:

```
Current issues found:
❌ /.backup_20251123_143839/ (1+ GB) - not in .gitignore
❌ /.cache/backups-* (1+ GB) - not in .gitignore
❌ /.config/organizations/alaweimm90-business-duplicate (274 MB) - exact duplicate
❌ node_modules/ potentially committed?
```

**Solutions**:

1. **Update .gitignore**:

```gitignore
# Node
node_modules/
dist/
build/
coverage/
.turbo/

# Environment
.env
.env.*.local
.env.local

# Build caches
.cache/
*.tsbuildinfo

# Backups (should not be in git!)
.backup_*
.archives/
*.backup

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

2. **Clean Up Existing Commits**:

```bash
# Remove large files from history
git filter-branch --tree-filter 'rm -rf .backup_*' HEAD

# Or use BFG Repo-Cleaner (faster)
bfg --delete-folders '.backup_*' --no-blob-protection

# Force push (be careful!)
git push origin --force-all
```

3. **Use Git LFS for Large Files**:

```bash
git lfs install
git lfs track "*.iso"
git lfs track "*.mp4"
git add .gitattributes
```

**Prevention**:

- ✅ `.gitignore` all build artifacts
- ✅ Store backups outside repo
- ✅ Use external storage for large media
- ✅ Run `git clean -fd` before commits
- ✅ Monitor repo size: `git count-objects -v`

---

### Pitfall 8: Inconsistent Development Environments

**Problem**: "Works on my machine" but fails in CI/prod

**Why It Happens**:

- Different Node/npm/pnpm versions
- Different OS (Windows vs Mac vs Linux)
- Global packages installed locally but not in package.json
- Environment variables set locally but not documented
- Cached modules from old versions

**Impact**: 🟡 MEDIUM

- Hard to debug
- Wastes developer time
- False confidence in CI

**Solutions**:

1. **Document Requirements**:

```
# .nvmrc (Node version)
20.10.0

# .tool-versions (for asdf)
nodejs 20.10.0
pnpm 9.0.0

# .dev-tools/versions.json
{
  "node": "20.10.0",
  "pnpm": "9.0.0",
  "docker": "24.0.0"
}
```

2. **Use DevContainers** (in VSCode):

```json
{
  ".devcontainer/devcontainer.json": {
    "image": "mcr.microsoft.com/devcontainers/typescript-node:20",
    "features": {
      "ghcr.io/devcontainers/features/git:1": {},
      "ghcr.io/devcontainers/features/docker-in-docker:2": {}
    }
  }
}
```

3. **Validate on Startup**:

```bash
# scripts/setup.sh
#!/bin/bash
set -e

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" != "20" ]; then
  echo "Error: Node 20 required, found $NODE_VERSION"
  echo "Run: nvm use"
  exit 1
fi

PNPM_VERSION=$(pnpm --version | cut -d'.' -f1)
if [ "$PNPM_VERSION" != "9" ]; then
  echo "Installing pnpm 9..."
  npm install -g pnpm@9
fi

echo "✅ Environment OK"
pnpm install
```

**Prevention**:

- ✅ Document in README
- ✅ Use Docker for consistency
- ✅ Use nvm/.nvmrc
- ✅ Use Volta for pnpm version

---

### Pitfall 9: Slow Package Installation

**Problem**: `pnpm install` takes 8+ minutes

**Current Status** (from your monorepo):

```
Issues found:
- Multiple pnpm instances (separate node_modules)
- No pnpm workspace hoisting
- Duplicate packages across orgs
```

**Solutions**:

1. **Optimize pnpm**:

```bash
# .npmrc
shamefully-hoist=true
strict-peer-dependencies=false
resolution-mode=highest

# Use pnpm 9+ for speed improvements
pnpm self-update
```

2. **Use Dependency Caching in CI**:

```yaml
- uses: actions/setup-node@v4
  with:
    node-version: '20'
    cache: 'pnpm' # Automatic cache key generation

- run: pnpm install --frozen-lockfile
```

3. **Remove Duplication**:

```bash
# Analyze what's duplicated
pnpm list --depth=0

# Remove workspace packages that should be shared
# Move common dependencies to root
```

**Prevention**:

- ✅ Use pnpm workspace efficiency
- ✅ Keep lock file in git
- ✅ Monitor install time trends
- ✅ Use CI caching

---

### Pitfall 10: Visibility & Discoverability

**Problem**: Teams don't know what code exists or how to use it

**Why It Happens**:

- 14+ organizations with no central index
- Documentation scattered/outdated
- No API documentation
- No architecture overview

**Impact**: 🟡 MEDIUM

- Code duplication across orgs
- Teams build similar features twice
- Long onboarding
- Reinventing the wheel

**Solutions**:

1. **Central Index**:
   Create `/docs/INDEX.md` documenting all packages/orgs

2. **API Documentation**:

```bash
pnpm add typedoc
pnpm docs:api  # Auto-generates HTML docs
```

3. **Architecture Diagrams**:

```
docs/
├── ARCHITECTURE.md
└── diagrams/
    ├── system-architecture.svg
    ├── data-flow.svg
    └── dependency-graph.svg
```

---

## 🔐 PART 2: SECURITY CONSIDERATIONS

### Security 1: Secret Management

**Problem**: Secrets hardcoded or insecurely stored

```typescript
// ❌ NEVER DO THIS
const API_KEY = 'sk-12345abcde';
const DB_PASSWORD = 'mypassword123';

// ❌ EVEN WORSE
const config = {
  aws: {
    accessKeyId: 'AKIA2EXAMPLE',
    secretAccessKey: 'wJalrXUtnFEMI/K7MDENG+...',
  },
};
```

**Solutions**:

1. **Environment Variables**:

```bash
# .env.example (checked in)
DATABASE_URL=postgresql://localhost/dev
API_KEY=your-key-here
STRIPE_SECRET=your-stripe-secret

# .env (NOT checked in, .gitignore)
DATABASE_URL=postgresql://user:pass@prod.db.com/prod
API_KEY=sk-prod-12345
STRIPE_SECRET=sk_live_...
```

2. **GitHub Secrets**:

```yaml
# .github/workflows/deploy.yml
env:
  API_KEY: ${{ secrets.API_KEY }}
  DATABASE_URL: ${{ secrets.DATABASE_URL }}
```

3. **Vault for Production**:

```typescript
// Use HashiCorp Vault or AWS Secrets Manager
const secret = await vault.get('database/credentials');
```

**Prevention**:

- ✅ Use `.gitignore` for `.env`
- ✅ Add `git-secrets` pre-commit hook
- ✅ Scan for secrets in CI/CD
- ✅ Rotate secrets regularly
- ✅ Use short-lived tokens

**Detection**:

```bash
# Add to CI
pnpm add git-secrets
git secrets --install
git secrets --scan-all  # Check commit history

# Or use TruffleHog for secret scanning
pnpm add trufflehog
trufflehog filesystem .
```

---

### Security 2: Dependency Vulnerability Management

**Problem**: Using packages with known security vulnerabilities

**Solutions**:

1. **Regular Audits**:

```bash
# Weekly
pnpm audit --audit-level=moderate

# In CI
name: Dependency Audit
on:
  schedule:
    - cron: '0 9 * * 1'  # Weekly Monday
```

2. **Automated Updates**:

```yaml
# Enable Dependabot in GitHub
name: dependabot
version: 2
updates:
  - package-ecosystem: npm
    directory: "/"
    schedule:
      interval: weekly
    reviewers:
      - @security-team
    assignees:
      - @dev-lead
```

3. **Snyk for Real-time Monitoring**:

```bash
pnpm add -D snyk
snyk test --all-projects
```

**Prevention**:

- ✅ Regular dependency audits
- ✅ Dependabot or Renovate
- ✅ Automatic testing of updates
- ✅ Security review before major upgrades

---

### Security 3: Access Control

**Problem**: Anyone can deploy to production, modify sensitive code

**Solutions**:

1. **GitHub Branch Protection**:

```
Settings → Branches → main:
✓ Require pull request reviews (2)
✓ Require status checks to pass
✓ Restrict who can push (admins only)
✓ Require dismissal of stale reviews
```

2. **Code Owners for Sensitive Code**:

```
# CODEOWNERS
* @dev-team
/packages/mcp-core/ @security-team
/packages/security-plugin/ @security-team
/.github/workflows/ @devops-team
/database/migrations/ @dba-team
```

3. **Environment Protection for Production**:

```yaml
# GitHub environment:
Settings → Environments → production:
✓ Require approval before deployment
✓ Restrict deployment to main branch
✓ Add environment secrets (API keys, etc.)
```

4. **Time-based Access**:

```yaml
# Only allow deployments 9am-5pm EST
# Require second approval outside business hours
```

**Prevention**:

- ✅ Enable branch protection
- ✅ Use CODEOWNERS
- ✅ Require PR approvals
- ✅ Audit deployment logs

---

### Security 4: Dependency Pinning

**Problem**: Using ^ and ~ allows breaking changes

```json
{
  "dependencies": {
    "lodash": "^4.17.0" // ^4.17.0 ≤ v < 5.0.0 (major change allowed!)
  }
}
```

**Solutions**:

1. **Pin Exact Versions for Security**:

```json
{
  "dependencies": {
    "lodash": "4.17.21" // Exact version only
  }
}
```

2. **Use Package Lock File**:

```bash
# pnpm-lock.yaml (always commit)
git add pnpm-lock.yaml

# Install from lock file
pnpm install --frozen-lockfile  # Error if lock out of sync
```

**Prevention**:

- ✅ Use --frozen-lockfile in CI
- ✅ Commit lock file to repo
- ✅ Update only intentionally
- ✅ Review all dependency changes

---

### Security 5: Multi-Organization Isolation

**Problem**: One org's code/data leaks to another org

**Solutions**:

1. **Data Isolation**:

```typescript
// Always filter by organization
const users = db.query(
  'SELECT * FROM users WHERE organization_id = $1',
  [currentOrg.id] // Must verify authorization
);
```

2. **Environment Separation**:

```
alaweimm90:
  DB: alaweimm90_prod
  API Key: alaweimm90_sk_live_...
  Credentials: Isolated in Vault under /alaweimm90/

alaweimm90-science:
  DB: science_prod
  API Key: science_sk_live_...
  Credentials: Isolated in Vault under /alaweimm90-science/
```

3. **Code-Level Enforcement**:

```typescript
// Middleware to enforce org isolation
export const requireOrganization = (req, res, next) => {
  const orgId = req.headers['x-organization-id'];
  if (!orgId || !isAuthorized(req.user, orgId)) {
    return res.status(403).json({ error: 'Forbidden' });
  }
  req.organization = orgId;
  next();
};

// All APIs require organization context
app.get('/api/users', requireOrganization, (req, res) => {
  const users = getUsers(req.organization);
  res.json(users);
});
```

**Prevention**:

- ✅ Default deny access
- ✅ Verify org on every API call
- ✅ Separate databases per org (if sensitive)
- ✅ Audit access logs

---

### Security 6: Secure CI/CD

**Problem**: CI/CD pipeline is attack vector

**Solutions**:

1. **Protect Workflow Files**:

```
CODEOWNERS:
/.github/workflows/ @devops-team @security-team
```

2. **Least Privilege Tokens**:

```yaml
# Instead of broad GITHUB_TOKEN:
jobs:
  deploy:
    # Only needs read packages, not write to repo
    permissions:
      contents: read
      packages: read
      deployments: write
```

3. **Audit CI/CD Changes**:

```yaml
name: Audit CI/CD Changes
on:
  pull_request:
    paths:
      - '.github/workflows/**'
jobs:
  audit:
    # Require approval for any workflow changes
    if: github.event.pull_request.draft == false
    runs-on: ubuntu-latest
    steps:
      - name: Comment
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              body: '⚠️ CI/CD changes require security review'
            })
```

**Prevention**:

- ✅ Restrict workflow file permissions
- ✅ Use minimal token permissions
- ✅ Audit all CI/CD changes
- ✅ Sign workflow triggers

---

### Security 7: Threat: Typosquatting & Malicious Packages

**Problem**: Install typosquatted package (e.g., `lodash` vs `lodahs`)

**Solutions**:

1. **Use npm Audit**:

```bash
pnpm audit  # Detects malicious packages
```

2. **Npm Registry Verification**:

```json
{
  "engines": {
    "node": ">=18.0.0",
    "npm": ">=9.0.0"
  },
  "publishConfig": {
    "registry": "https://registry.npmjs.org/"
  }
}
```

3. **Lock File Validation**:
   Always review `.pnpm-lock.yaml` changes for unexpected packages

**Prevention**:

- ✅ Review all new dependencies
- ✅ Use npm audit
- ✅ Use trusted registries
- ✅ Keep dependencies current

---

### Security 8: Compliance & Audit Trails

**Problem**: Can't prove who changed what and when

**Solutions**:

1. **Git Commit Signing**:

```bash
git config --global user.signingkey ABCDEF123456
git commit -S -m "Important change"  # Sign commits
```

2. **Enforce Signed Commits**:

```
GitHub Settings → Branch Protection:
✓ Require signed commits
```

3. **Audit Logging**:

```typescript
// Log all sensitive operations
logger.info('Deployment initiated', {
  timestamp: new Date(),
  user: req.user.email,
  organization: req.organization,
  version: process.env.VERSION,
  environment: process.env.NODE_ENV,
});
```

**Prevention**:

- ✅ Sign commits
- ✅ Audit all sensitive actions
- ✅ Keep immutable audit logs
- ✅ Regular compliance reviews

---

### Security Checklist

- [ ] All secrets in environment variables or Vault
- [ ] No secrets committed to repo
- [ ] Dependency audits run weekly
- [ ] Branch protection enabled
- [ ] Code owners assigned to sensitive files
- [ ] Environment protections in place
- [ ] Deployments require approval
- [ ] Access logs maintained
- [ ] Commits signed
- [ ] Compliance reviewed quarterly

---

## 🚀 IMPLEMENTATION PRIORITY

### Week 1 (Immediate)

- [ ] Fix dependency version incompatibilities
- [ ] Add secret scanning to CI
- [ ] Enable branch protection on main
- [ ] Set up CODEOWNERS file

### Week 2 (High Priority)

- [ ] Implement Turbo caching for speed
- [ ] Add database migration validation
- [ ] Enable Dependabot
- [ ] Clean up large files from history

### Week 3 (Important)

- [ ] Set up environment protections
- [ ] Implement test cache validation
- [ ] Document environment setup
- [ ] Add flaky test detection

### Week 4 (Enhancement)

- [ ] Implement Snyk monitoring
- [ ] Set up audit logging
- [ ] Enforce commit signing
- [ ] Quarterly security review process

---

## 📊 SUMMARY

### Common Pitfalls (Severity)

| Pitfall                      | Severity | Frequency | Fix Time |
| ---------------------------- | -------- | --------- | -------- |
| Dependency version conflicts | Critical | Common    | 2 hours  |
| Circular dependencies        | Critical | Rare      | 4 hours  |
| Build time explosion         | High     | Common    | 8 hours  |
| Flaky tests                  | High     | Common    | 4 hours  |
| Cache invalidation           | High     | Rare      | 2 hours  |
| DB migration issues          | Critical | Rare      | 2 hours  |
| Repo size explosion          | High     | Common    | 4 hours  |
| Environment inconsistency    | Medium   | Common    | 2 hours  |
| Slow installation            | Medium   | Common    | 3 hours  |
| Poor discoverability         | Medium   | Always    | Ongoing  |

### Security Priorities

1. **Immediate**: Secret management, dependency audits
2. **Short-term**: Branch protection, access control
3. **Ongoing**: Compliance, audit trails, monitoring

---

**Status**: ✅ PITFALLS & SECURITY DOCUMENTED
**Comprehensive Monorepo Analysis**: COMPLETE ✅
