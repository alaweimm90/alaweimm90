# Runbook: Build Optimization

**Purpose:** Improve build performance and reduce CI/CD execution times
**Frequency:** Quarterly review or when budgets exceeded
**Duration:** 2-4 hours

---

## Prerequisites

- [ ] Node.js v22.20.0 installed (check `.nvmrc`)
- [ ] Python 3.11+ installed
- [ ] Write access to repository
- [ ] Baseline performance metrics documented

## Current Optimization Status

**Implemented Optimizations:**

- ✅ npm dependency caching in CI (via `actions/setup-node@v4`)
- ✅ pip dependency caching in CI (via `actions/setup-python@v5`)
- ✅ Docker layer caching for Python CI
- ✅ Parallel job execution in GitHub Actions
- ✅ TypeScript incremental compilation
- ✅ Minimal production dependencies (6 packages)

---

## Optimization Procedures

### 1. Benchmark Current Performance

Before making changes, establish baseline metrics:

```bash
# CI/CD baseline
gh run list --limit 5 --json workflowName,startedAt,createdAt,conclusion

# Local build baseline
time npm ci
time npm run type-check
time npm run lint
time npm run test:run

# Python baseline
time pip install -r .metaHub/scripts/requirements.txt
time python -m pytest tests/

# Measure CLI startup times
time npm run devops -- --help
time npm run ORCHEX -- --help
```

**Document Results:**

```
CI Duration:        ___ minutes
Type Check:         ___ seconds
Lint:               ___ seconds
Tests:              ___ seconds
CLI Startup:        ___ milliseconds
```

### 2. Analyze Bottlenecks

#### CI/CD Analysis

```bash
# Find slowest workflow jobs
gh run view <run-id> --log | grep "took"

# Check for jobs that could be parallelized
# Review .github/workflows/ci.yml for sequential dependencies
```

#### TypeScript Analysis

```bash
# Show compilation diagnostics
npx tsc --diagnostics --noEmit

# List all files being compiled
npx tsc --listFiles --noEmit | wc -l

# Find largest type definitions
find node_modules/@types -name "*.d.ts" -exec wc -l {} + | sort -rn | head -20
```

#### Dependency Analysis

```bash
# Find unused dependencies
npx depcheck

# Analyze bundle size
npm ls --depth=0
du -sh node_modules/*/ | sort -rh | head -20

# Check for duplicate dependencies
npm dedupe --dry-run
```

### 3. Apply Optimizations

#### TypeScript Optimizations

**Option A: Enable skipLibCheck (if not already enabled)**

```bash
# Edit tsconfig.json
{
  "compilerOptions": {
    "skipLibCheck": true  // Skip type checking of .d.ts files
  }
}
```

**Benefit:** 20-40% faster type checking
**Risk:** Low - only skips checking library definitions

**Option B: Use Project References**

For large monorepo structures:

```json
// tsconfig.base.json
{
  "compilerOptions": {
    "composite": true,
    "declaration": true
  }
}

// Individual project tsconfig.json
{
  "extends": "./tsconfig.base.json",
  "references": [
    { "path": "../shared" }
  ]
}
```

**Benefit:** Incremental builds across projects
**Risk:** Medium - requires restructuring

#### Dependency Optimizations

**Remove Unused Dependencies:**

```bash
# Detect unused packages
npx depcheck

# Remove identified packages
npm uninstall <unused-package>

# Verify builds still work
npm run type-check && npm run test:run
```

**Deduplicate Dependencies:**

```bash
# Remove duplicate packages
npm dedupe

# Verify no issues
npm run test:run
```

**Replace Heavy Dependencies:**

Common replacements:

- `moment` → `date-fns` or native `Intl.DateTimeFormat`
- `lodash` → ES6 built-ins or `lodash-es` (tree-shakeable)
- `request` → `node-fetch` or native `fetch`

#### CI/CD Optimizations

**Option A: Add Path Filters**

Only run workflows when relevant files change:

```yaml
# .github/workflows/ci.yml
on:
  push:
    paths:
      - 'tools/**'
      - 'src/**'
      - 'package.json'
      - 'tsconfig.json'
  pull_request:
    paths:
      - 'tools/**'
      - 'src/**'
```

**Option B: Matrix Strategy for Tests**

Parallelize test execution:

```yaml
jobs:
  test:
    strategy:
      matrix:
        test-suite:
          - unit
          - integration
          - e2e
    steps:
      - run: npm run test:${{ matrix.test-suite }}
```

**Option C: Early Termination**

Fail fast on critical errors:

```yaml
jobs:
  quick-checks:
    runs-on: ubuntu-latest
    steps:
      - name: Lint
        run: npm run lint
      - name: Type check
        run: npm run type-check

  full-tests:
    needs: quick-checks # Only run if quick checks pass
    runs-on: ubuntu-latest
    steps:
      - run: npm run test:run
```

#### Pre-commit Hook Optimizations

**Option A: Scope to Staged Files**

```json
// package.json
{
  "lint-staged": {
    "*.{ts,tsx}": ["eslint --fix --max-warnings 0", "prettier --write"]
  }
}
```

Already configured ✅

**Option B: Parallel Execution**

```bash
# .husky/pre-commit
npx lint-staged &
npm run type-check &
wait
```

⚠️ Be cautious - parallel execution can consume resources

### 4. Validate Changes

After applying optimizations:

```bash
# Re-run benchmarks
time npm ci
time npm run type-check
time npm run lint
time npm run test:run

# Compare to baseline
# Calculate improvement percentage
```

**Expected Improvements:**

- Type check: 10-30% faster
- Lint: 5-15% faster
- CI total: 10-20% faster
- npm install: 30-50% faster (with cache)

### 5. Monitor for Regressions

**Set up performance tracking:**

```bash
# Create performance log
cat > performance-log.md << 'EOF'
# Performance Tracking

## YYYY-MM-DD - Baseline
- CI Duration: X min
- Type Check: Y sec
- Tests: Z sec

## YYYY-MM-DD - After Optimization
- CI Duration: X min (-N%)
- Type Check: Y sec (-N%)
- Tests: Z sec (-N%)
EOF
```

---

## Advanced Optimizations

### TypeScript Incremental Builds

Enable incremental compilation for local development:

```json
// tsconfig.json
{
  "compilerOptions": {
    "incremental": true,
    "tsBuildInfoFile": "./.tsbuildinfo"
  }
}
```

Add to `.gitignore`:

```
.tsbuildinfo
*.tsbuildinfo
```

### ESLint Caching

Enable ESLint caching for faster linting:

```json
// package.json
{
  "scripts": {
    "lint": "eslint . --cache --cache-location .eslintcache"
  }
}
```

Add to `.gitignore`:

```
.eslintcache
```

### Vitest Watch Mode

For local development:

```bash
# Run tests in watch mode (only re-runs changed tests)
npm run test

# Or specific file watch
npm run test -- src/utils/config.test.ts
```

---

## Common Issues

### Issue: Type checking is slow

**Diagnosis:**

```bash
# Check what's being type-checked
npx tsc --listFiles --noEmit | wc -l

# Expected: < 500 files
# If more, investigate what's included
```

**Solutions:**

1. Add exclusions to `tsconfig.json`:

   ```json
   {
     "exclude": ["node_modules", "dist", "build", "**/*.test.ts", "tests/**/*"]
   }
   ```

2. Use separate configs for build vs. type-check

### Issue: npm ci is slow

**Diagnosis:**

```bash
# Check node_modules size
du -sh node_modules

# Expected: < 100 MB
# If larger, audit dependencies
```

**Solutions:**

1. Use npm v8+ with improved caching
2. Verify CI cache is working:

   ```yaml
   - uses: actions/setup-node@v4
     with:
       cache: 'npm' # Must be present
   ```

3. Consider `pnpm` for faster installs (advanced)

### Issue: Tests are slow

**Diagnosis:**

```bash
# Run tests with timing reporter
npm run test:run -- --reporter=verbose

# Identify slow tests
```

**Solutions:**

1. Parallelize tests (Vitest does this by default)
2. Use `test.concurrent()` for independent tests
3. Mock slow external dependencies
4. Use `beforeAll` instead of `beforeEach` when possible

---

## Rollback Procedure

If optimizations cause issues:

```bash
# Revert changes
git diff HEAD~1 tsconfig.json package.json
git checkout HEAD~1 -- tsconfig.json package.json

# Reinstall dependencies
npm ci

# Verify everything works
npm run type-check
npm run lint
npm run test:run

# Push revert if needed
git commit -m "revert: rollback build optimizations due to issues"
git push
```

---

## Quarterly Review Checklist

- [ ] Run full benchmark suite
- [ ] Compare to previous quarter's metrics
- [ ] Review new dependency additions
- [ ] Audit CI/CD workflow durations
- [ ] Update performance budgets if needed
- [ ] Document any optimizations applied

---

## Performance Budget Compliance

After optimization, verify compliance with budgets:

```bash
# Check against budgets in performance-budgets.md
# - TypeScript CI: < 2 min
# - Python CI: < 90 sec
# - Total CI: < 5 min
# - Production deps: < 20 packages
# - CLI startup: < 500 ms
```

---

## References

- [Performance Budgets](../performance-budgets.md)
- [TypeScript Performance](https://github.com/microsoft/TypeScript/wiki/Performance)
- [npm ci documentation](https://docs.npmjs.com/cli/v9/commands/npm-ci)
- [GitHub Actions caching](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
- [ESLint performance](https://eslint.org/docs/latest/use/configure/configuration-files#performance)
