# QAPLIB Large Instances Download Report

**Date:** 2025-11-06
**Purpose:** Download larger QAPLIB instances for scalability and FFT-Laplace threshold testing
**Source:** GitHub - mhahsler/qap repository
**Repository URL:** https://github.com/mhahsler/qap/tree/master/inst/qaplib

---

## Download Summary

### Overall Status: ✓ SUCCESS

- **Total files downloaded:** 10 instances
- **Total size:** ~53 KB
- **All files verified:** YES
- **All files loadable:** YES
- **Ready for benchmarking:** YES

---

## Files Downloaded (by Priority)

### HIGH PRIORITY - n=20 Instances (FFT Threshold Testing)

| Instance   | Size  | n  | Optimal Solution | Status | Purpose                  |
|------------|-------|----|------------------|--------|--------------------------|
| had20.dat  | 2.4KB | 20 | 6,922           | ✓      | Hadamard problem         |
| nug20.dat  | 2.0KB | 20 | 2,570           | ✓      | Nugent problem           |
| tai20a.dat | 2.4KB | 20 | 703,482         | ✓      | Taillard uniform         |
| chr20a.dat | 2.0KB | 20 | 2,192           | ✓      | Christofides real-life   |
| chr20b.dat | 2.0KB | 20 | 2,298           | ✓      | Christofides variant     |
| rou20.dat  | 2.4KB | 20 | 725,522         | ✓      | Roucairol real-life      |

**Total: 6 instances, ~13 KB**

### MEDIUM PRIORITY - n=30 Instances

| Instance   | Size  | n  | Optimal Solution | Status | Purpose                  |
|------------|-------|----|------------------|--------|--------------------------|
| tai30a.dat | 5.4KB | 30 | 1,818,146       | ✓      | Taillard uniform         |
| ste36a.dat | 13KB  | 36 | 9,526           | ✓      | Steinberg wiring         |

**Total: 2 instances, ~18 KB**

### SCALABILITY TESTING - n=40-50 Instances

| Instance   | Size  | n  | Optimal Solution | Status | Purpose                  |
|------------|-------|----|------------------|--------|--------------------------|
| tai40a.dat | 9.5KB | 40 | 3,139,370       | ✓      | Taillard uniform         |
| tai50a.dat | 15KB  | 50 | 4,938,796       | ✓      | Taillard uniform         |

**Total: 2 instances, ~24 KB**

---

## Validation Results

### File Format Validation

All 10 files successfully validated:

```
Testing file loading...
============================================================
had20        - n=20 - Flow: (20, 20), Dist: (20, 20) ✓
nug20        - n=20 - Flow: (20, 20), Dist: (20, 20) ✓
tai20a       - n=20 - Flow: (20, 20), Dist: (20, 20) ✓
chr20a       - n=20 - Flow: (20, 20), Dist: (20, 20) ✓
chr20b       - n=20 - Flow: (20, 20), Dist: (20, 20) ✓
rou20        - n=20 - Flow: (20, 20), Dist: (20, 20) ✓
tai30a       - n=30 - Flow: (30, 30), Dist: (30, 30) ✓
ste36a       - n=36 - Flow: (36, 36), Dist: (36, 36) ✓
tai40a       - n=40 - Flow: (40, 40), Dist: (40, 40) ✓
tai50a       - n=50 - Flow: (50, 50), Dist: (50, 50) ✓
============================================================
```

### Key Validation Points

✓ **Download Integrity:** All files downloaded without errors
✓ **File Sizes:** All file sizes reasonable for problem dimensions
✓ **Format Compatibility:** All files successfully parsed by `load_qaplib_instance()`
✓ **Matrix Dimensions:** All n×n matrices correctly loaded
✓ **Multi-line Format:** Correctly handles both single-line and wrapped matrix rows

---

## Problem Size Distribution

| Problem Size (n) | Count | Files                                               |
|------------------|-------|-----------------------------------------------------|
| 20               | 6     | had20, nug20, tai20a, chr20a, chr20b, rou20        |
| 30               | 1     | tai30a                                              |
| 36               | 1     | ste36a                                              |
| 40               | 1     | tai40a                                              |
| 50               | 1     | tai50a                                              |

**Total:** 10 instances across 5 different problem sizes

---

## Problem Type Distribution

| Type        | Count | Description                              |
|-------------|-------|------------------------------------------|
| Taillard    | 4     | Randomly generated uniform instances     |
| Christofides| 2     | Real-life inspired problems              |
| Hadamard    | 1     | Based on Hadamard matrices              |
| Nugent      | 1     | Facility location problem               |
| Roucairol   | 1     | Real-life problem                       |
| Steinberg   | 1     | Wiring problem                          |

**Total:** 6 different problem types providing diverse test coverage

---

## Download Failures

**None** - All requested files downloaded successfully.

---

## Next Steps for Benchmarking

### 1. FFT Threshold Analysis (n=20)
- Use the 6 n=20 instances to determine FFT-Laplace crossover point
- Compare direct convolution vs. FFT-based for n=20
- Establish threshold decision rules for UltraThink

### 2. Scalability Testing (n=30-50)
- Profile performance scaling from n=20 to n=50
- Measure FFT advantage as problem size increases
- Validate O(n² log n) vs O(n³) complexity difference

### 3. Diverse Problem Coverage
- Test across 6 different problem types
- Verify FFT benefits are consistent across problem structures
- Identify any problem-specific patterns

### 4. Validation
- Compare results against known optimal solutions
- Ensure correctness before performance optimization
- Document any deviations from optimal

---

## File Locations

All instances saved to: `/home/user/QAP-CLAUDE-CODE/data/qaplib/`

### Directory Contents (14 total instances):

**n=12 instances (previously downloaded):**
- chr12c.dat, had12.dat, nug12.dat, tai12a.dat

**n=20 instances (newly downloaded):**
- chr20a.dat, chr20b.dat, had20.dat, nug20.dat, rou20.dat, tai20a.dat

**n=30+ instances (newly downloaded):**
- tai30a.dat, ste36a.dat, tai40a.dat, tai50a.dat

---

## Conclusion

**Download Status:** ✅ COMPLETE
**Verification Status:** ✅ PASSED
**Ready for Benchmarking:** ✅ YES

All 10 requested large instances successfully downloaded and verified. The dataset now includes comprehensive coverage from n=12 to n=50, providing excellent test coverage for:

1. FFT threshold determination (n=20 critical range)
2. Scalability analysis (n=30-50 progression)
3. Algorithm validation (known optimal solutions)
4. Diverse problem types (6 different families)

The instances are immediately ready for use in performance benchmarking and FFT-Laplace threshold testing.
