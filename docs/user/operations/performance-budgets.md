# Performance Budgets

**Purpose:** Define and monitor performance targets for CI/CD, builds, and operations
**Last Updated:** 2025-12-04

---

## Overview

Performance budgets ensure the repository maintains fast feedback cycles and efficient resource usage. This document defines acceptable thresholds for various performance metrics.

## CI/CD Pipeline Budgets

### Build Times

| Job                   | Target   | Warning | Critical | Notes                           |
| --------------------- | -------- | ------- | -------- | ------------------------------- |
| TypeScript CI         | < 2 min  | 3 min   | 5 min    | Includes lint, type-check, test |
| Python CI             | < 90 sec | 2 min   | 3 min    | Includes lint, mypy, pytest     |
| Governance Validation | < 60 sec | 90 sec  | 2 min    | CLI tests and schema validation |
| Security Scan         | < 2 min  | 3 min   | 5 min    | Trivy vulnerability scanning    |
| Technical Debt Gate   | < 90 sec | 2 min   | 3 min    | Debt scanning and gate checks   |

**Total CI Duration Target:** < 5 minutes for full pipeline

### Test Execution

| Test Suite      | Target   | Warning | Critical |
| --------------- | -------- | ------- | -------- |
| Unit Tests      | < 30 sec | 45 sec  | 60 sec   |
| Integration     | < 60 sec | 90 sec  | 2 min    |
| Full Test Suite | < 90 sec | 2 min   | 3 min    |

## Dependency Size Budgets

### Node.js Dependencies

**Target:** Keep production dependencies minimal for CLI performance

| Metric                  | Target    | Warning | Critical |
| ----------------------- | --------- | ------- | -------- |
| Production Dependencies | < 20 pkgs | 25 pkgs | 30 pkgs  |
| Total node_modules size | < 100 MB  | 150 MB  | 200 MB   |
| CLI startup time        | < 500 ms  | 1 sec   | 2 sec    |

**Current Production Dependencies:** 6 packages (chalk, commander, ora, ws, yaml, better-sqlite3)

### Python Dependencies

| Metric                  | Target   | Warning | Critical |
| ----------------------- | -------- | ------- | -------- |
| Production pip packages | < 15     | 20      | 25       |
| Script startup time     | < 300 ms | 500 ms  | 1 sec    |

## Build Artifacts

### TypeScript Compilation

| Metric               | Target   | Warning | Critical |
| -------------------- | -------- | ------- | -------- |
| tsc compilation time | < 10 sec | 15 sec  | 30 sec   |
| Type check time      | < 15 sec | 20 sec  | 30 sec   |

### Tool Bundle Sizes (if bundling CLIs)

| Tool         | Target | Warning | Critical |
| ------------ | ------ | ------- | -------- |
| DevOps CLI   | < 5 MB | 10 MB   | 20 MB    |
| ORCHEX CLI   | < 5 MB | 10 MB   | 20 MB    |
| AI Tools CLI | < 5 MB | 10 MB   | 20 MB    |

## Resource Usage Budgets

### CI/CD Resource Consumption

| Resource    | Target       | Warning    | Critical   |
| ----------- | ------------ | ---------- | ---------- |
| CPU minutes | < 10 min/run | 15 min/run | 20 min/run |
| Storage     | < 500 MB/run | 1 GB/run   | 2 GB/run   |
| Network     | < 100 MB/run | 200 MB/run | 500 MB/run |

### Local Development

| Operation        | Target   | Warning | Critical |
| ---------------- | -------- | ------- | -------- |
| npm install      | < 30 sec | 45 sec  | 60 sec   |
| npm ci           | < 20 sec | 30 sec  | 45 sec   |
| pip install      | < 10 sec | 15 sec  | 30 sec   |
| Pre-commit hooks | < 10 sec | 15 sec  | 30 sec   |

---

## Monitoring

### Automated Checks

Performance metrics are monitored automatically:

```yaml
# GitHub Actions workflow times
# - Available in Actions dashboard
# - Use GitHub API for historical data

# Local benchmarking
npm run test:run -- --reporter=verbose  # Shows test timing
time npm run type-check                 # Measure TypeScript performance
time npm run lint                       # Measure linting performance
```

### Manual Benchmarking

```bash
# Measure CLI startup time
time npm run devops -- --help

# Measure build time
time npm run type-check

# Measure test execution
time npm run test:run

# Check node_modules size
du -sh node_modules

# Count production dependencies
cat package.json | jq '.dependencies | length'
```

### GitHub Actions Performance

```bash
# View recent workflow runs with durations
gh run list --limit 10 --json workflowName,conclusion,startedAt,createdAt

# View detailed timing for a specific run
gh run view <run-id> --log
```

---

## Optimization Strategies

### When CI Times Exceed Budgets

1. **Analyze bottlenecks:**

   ```bash
   # Review workflow logs for slow steps
   gh run view <run-id> --log | grep "took"
   ```

2. **Enable caching (already configured):**
   - npm dependencies cached via `actions/setup-node@v4`
   - pip dependencies cached via `actions/setup-python@v5`
   - Docker layers cached via GitHub Actions cache

3. **Parallelize jobs:**
   - CI already runs jobs in parallel
   - Consider splitting large test suites

4. **Skip redundant steps:**
   - Use path filters to skip workflows when irrelevant files change
   - Example: Skip Python CI if only TypeScript files changed

### When Dependencies Grow Too Large

1. **Audit dependencies:**

   ```bash
   npm ls --depth=0              # List direct dependencies
   npx depcheck                  # Find unused dependencies
   npm audit                     # Check for vulnerabilities
   ```

2. **Remove unused dependencies:**

   ```bash
   npm uninstall <package>
   pip uninstall <package>
   ```

3. **Consider alternatives:**
   - Replace heavy packages with lightweight alternatives
   - Use built-in Node.js modules where possible
   - Lazy-load non-critical dependencies

### When Build Times Increase

1. **Profile TypeScript compilation:**

   ```bash
   npx tsc --diagnostics         # Show compilation stats
   npx tsc --listFiles           # See what's being compiled
   ```

2. **Optimize tsconfig.json:**
   - Use `incremental: true` for faster rebuilds
   - Exclude unnecessary directories
   - Use `skipLibCheck: true` for faster type checking

3. **Split large files:**
   - Break monolithic files into smaller modules
   - Improve tree-shaking and compilation speed

---

## Performance Regression Prevention

### Pre-merge Checks

All PRs must pass these performance gates:

- [ ] CI pipeline completes in < 5 minutes
- [ ] No new dependencies added without justification
- [ ] Type check completes in < 30 seconds
- [ ] Tests complete in < 2 minutes

### Dependency Updates

When updating dependencies:

```bash
# Before update - benchmark current state
time npm ci && time npm run test:run

# After update - compare performance
time npm ci && time npm run test:run

# If performance degrades > 20%, investigate or revert
```

### Continuous Monitoring

Track performance trends over time:

- **Weekly:** Review CI dashboard for average build times
- **Monthly:** Audit dependency sizes and startup times
- **Quarterly:** Review and update performance budgets

---

## Alert Thresholds

### When to Take Action

| Threshold | Action                           |
| --------- | -------------------------------- |
| Warning   | Investigate during next sprint   |
| Critical  | Immediate investigation required |
| Exceeded  | Block merges until resolved      |

### Escalation

If performance budgets are consistently exceeded:

1. Create GitHub issue with performance metrics
2. Assign to team lead (@alawein)
3. Schedule optimization sprint
4. Update budgets if infrastructure changed

---

## Historical Baseline

**Initial Measurements (2025-12-04):**

```
TypeScript CI:           ~2 min
Python CI:               ~1.5 min
Total Pipeline:          ~4-5 min
node_modules size:       ~80 MB
Production deps:         6 packages
```

**Latest Measurements (2025-12-04):**

| Metric            | Value  | Target  | Status       |
| ----------------- | ------ | ------- | ------------ |
| Type check        | 3.54s  | < 15s   | ✅ Excellent |
| Test suite        | 15.11s | < 90s   | ✅ Excellent |
| Production deps   | 7 pkgs | < 20    | ✅ Excellent |
| node_modules size | 111 MB | < 100MB | ⚠️ Warning   |
| ESLint errors     | 0      | 0       | ✅ Pass      |
| TypeScript errors | 0      | 0       | ✅ Pass      |

**Unused Dependencies Identified:**

Production (consider removal):

- `better-sqlite3` - not imported in production code
- `chokidar` - not imported in production code
- `ora` - not imported in production code
- `ws` - not imported in production code

Dev (consider removal):

- `@rollup/rollup-win32-x64-msvc` - platform-specific, may be auto-installed
- `@types/better-sqlite3` - unused if better-sqlite3 removed
- `@types/js-yaml` - unused
- `@types/pg` - unused
- `@types/ws` - unused if ws removed

**Performance Goals:**

- Maintain CI times under 5 minutes as codebase grows
- Keep production dependencies under 10 packages
- Ensure CLI startup time stays under 500ms

---

## References

- [GitHub Actions billing](https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions)
- [CI/CD workflows](../../.github/workflows/)
- [Dependency updates runbook](./runbooks/dependency-updates.md)
- [Node.js Performance](https://nodejs.org/en/docs/guides/simple-profiling/)
