# Known Issues

**Last Updated:** 2025-12-04

---

## Active Issues

_(No active issues)_

---

## Resolved Issues

### 1. Vitest Test Discovery Failure

**Status:** ✅ **Resolved** - 2025-12-04
**Resolution:** Downgraded from Vitest 4.0.15 to 3.2.4

**Original Issue:**
Vitest 4.0.15 failed to discover test suites with error: "No test suite found in file". All 18 test files were affected, making the entire test suite non-functional.

**Root Cause:**
Vitest 4.x has a regression or compatibility issue with Windows path resolution and/or TypeScript `moduleResolution: "bundler"` configuration.

**Solution:**

```bash
npm install -D vitest@3.2.4
```

**Result:**

- ✅ All 241 tests passing across 18 test files
- ✅ Test execution time: ~14 seconds
- ✅ CI/CD tests restored to working state

**Recommendation:**
Do not upgrade to Vitest 4.x until the Windows compatibility issue is resolved upstream. Monitor [Vitest releases](https://github.com/vitest-dev/vitest/releases) for a fix.

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
