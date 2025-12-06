# Backward Compatibility Policy

## Overview

This document defines the backward compatibility policy for the Librex optimization framework, ensuring smooth upgrades and migrations for users.

## Semantic Versioning

Librex follows [Semantic Versioning 2.0.0](https://semver.org/):

**Version Format:** MAJOR.MINOR.PATCH

- **MAJOR:** Incompatible API changes
- **MINOR:** Backward-compatible functionality additions
- **PATCH:** Backward-compatible bug fixes

**Current Version:** 1.0.0 (Beta)

## Compatibility Guarantees

### Version 1.x.x Series

**API Stability Guarantees:**

✅ **Guaranteed Stable (No breaking changes):**
- `optimize()` function signature
- `StandardizedProblem`, `StandardizedSolution`, `ValidationResult` dataclasses
- Domain adapter interfaces (`QAPAdapter`, `TSPAdapter`)
- Baseline method names and basic config parameters
- Return value structures from `optimize()`

⚠️ **May Change (With deprecation warnings):**
- Internal method implementations
- Optional configuration parameters
- Utility functions not in public API
- Novel/experimental methods

❌ **No Guarantees (Subject to change):**
- Private methods (prefixed with `_`)
- Research-grade methods marked experimental
- Internal data structures
- Undocumented features

## Deprecation Policy

### 4-Phase Deprecation Process

1. **Phase 1: Deprecation Warning** (1 minor version)
2. **Phase 2: Loud Warning** (1 minor version)
3. **Phase 3: Error on Use** (Next major version)
4. **Phase 4: Complete Removal** (After 1 major version)

### Example: FFT-Laplace

**Status:** Phase 3 - Raises NotImplementedError

**Migration:** Use `simulated_annealing` or other baseline methods

## Python Version Support

**Current:** Python 3.9-3.12
**Policy:** Support latest 4 minor versions
**Deprecation:** 1 year notice before dropping support

## Contact

**Issues:** https://github.com/AlaweinOS/AlaweinOS/issues

---

**Version:** 1.0 | **Updated:** 2025-11-18
