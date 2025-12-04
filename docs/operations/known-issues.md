# Known Issues

**Last Updated:** 2025-12-04

---

## Active Issues

### 1. Vitest Test Discovery Failure

**Status:** 🔴 **Blocking** - All tests fail to run
**Severity:** High
**Affects:** Test suite (18 test files)

**Description:**
Vitest 4.0.15 fails to discover test suites with error: "No test suite found in file". All test files (`.test.ts`) contain valid test suites with `describe` and `it` blocks from Vitest, but the test runner cannot discover them.

**Error Message:**

```
Error: No test suite found in file c:/Users/mesha/Desktop/GitHub/tests/[file].test.ts
```

**Impact:**

- Cannot run test suite
- CI/CD tests are failing
- No test coverage reporting available

**Affected Files:**

- `tests/ai/*.test.ts` (7 files)
- `tests/atlas/**/*.test.ts` (2 files)
- `tests/*.test.ts` (4 files)
- All 18 test files affected

**Investigation:**

- ✅ Test files are properly structured with valid Vitest syntax
- ✅ TypeScript compilation succeeds (with unrelated warnings)
- ✅ vitest.config.ts is properly configured
- ❌ Even minimal test files fail with same error
- ❌ Vitest 4.0.15 may have regression or compatibility issue

**Potential Causes:**

1. Vitest 4.x breaking changes with TypeScript `moduleResolution: "bundler"`
2. Windows path resolution issues in Vitest 4.x
3. Module transform configuration incompatibility

**Workarounds:**
None available - test suite is non-functional.

**Action Items:**

- [ ] Try downgrading to Vitest 3.x
- [ ] Check Vitest 4.x migration guide for breaking changes
- [ ] Report issue to Vitest repository if confirmed bug
- [ ] Consider alternative test runner (e.g., Jest, Mocha) if Vitest issue persists

**Related:**

- [Vitest v4 Release Notes](https://github.com/vitest-dev/vitest/releases/tag/v4.0.0)
- Package.json: `"vitest": "4.0.15"`

---

## Resolved Issues

_(None yet)_

---

## Monitoring

To check if this issue is resolved:

```bash
# Run minimal test
npm run test:run -- tests/meta-cli.test.ts

# Expected: Tests should execute successfully
# Current: "No test suite found" error
```

---

## Issue Reporting

When reporting issues:

1. **Include context:** OS, Node version, package versions
2. **Minimal reproduction:** Simplest case that demonstrates the issue
3. **Expected vs actual:** What should happen vs what actually happens
4. **Investigation:** Steps already taken to diagnose

**Repository:** https://github.com/alaweimm90/alaweimm90
**Maintainer:** @alaweimm90
