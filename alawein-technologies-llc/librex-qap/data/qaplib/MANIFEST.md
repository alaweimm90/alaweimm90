# QAPLIB Benchmark Data Manifest

## Download Information

**Date Downloaded:** 2025-11-06
**Source:** GitHub - mhahsler/qap repository
**Repository URL:** https://github.com/mhahsler/qap/tree/master/inst/qaplib
**Original Source:** QAPLIB (Quadratic Assignment Problem Library)

## Files Downloaded

### 1. had12.dat (Hadamard Problem)
- **Size:** 901 bytes (0.9 KB)
- **Problem Size:** n=12
- **Known Optimal Solution:** 1652
- **MD5:** 649ec9b97f46482bdf80f75126b8d760
- **SHA256:** f449f4e5af533a23ae5170c8781cdd92ddfec6fdf8f648115e2608349c078be7
- **Description:** Hadamard matrices-based QAP instance

### 2. nug12.dat (Nugent Problem)
- **Size:** 713 bytes (0.7 KB)
- **Problem Size:** n=12
- **Known Optimal Solution:** 578
- **MD5:** 009801a839f6dd6e8724f38e1316e5b8
- **SHA256:** 15ad1047d3d39e2820466349dc75ce8cc560443e125c4f67ff06223d735f47b6
- **Description:** Nugent facility location problem

### 3. chr12c.dat (Christofides Problem)
- **Size:** 1709 bytes (1.7 KB)
- **Problem Size:** n=12
- **Known Optimal Solution:** 11156
- **MD5:** 978ce575cee8780c13b0606800974227
- **SHA256:** 3baebb29e783e752bc43d3a92045666e69d5775ad54468d4ef8d4ab1cb8f9fe6
- **Description:** Christofides real-life problem instance

### 4. tai12a.dat (Taillard Problem)
- **Size:** 895 bytes (0.9 KB)
- **Problem Size:** n=12
- **Known Optimal Solution:** 224416
- **MD5:** 47b808637422f08d3c301f0ba4d04d36
- **SHA256:** 3a9fa8069e82f9fd7806c37175e9c568a66ffcf9798935bf1f7336df81938318
- **Description:** Taillard randomly generated uniform instance

---

## LARGER INSTANCES (Downloaded: 2025-11-06)

### n=20 Instances (HIGH PRIORITY - FFT Threshold Testing)

### 5. had20.dat (Hadamard Problem)
- **Size:** 2.4 KB
- **Problem Size:** n=20
- **Known Optimal Solution:** 6922
- **Description:** Hadamard matrices-based QAP instance (larger)

### 6. nug20.dat (Nugent Problem)
- **Size:** 2.0 KB
- **Problem Size:** n=20
- **Known Optimal Solution:** 2570
- **Description:** Nugent facility location problem (larger)

### 7. tai20a.dat (Taillard Problem)
- **Size:** 2.4 KB
- **Problem Size:** n=20
- **Known Optimal Solution:** 703482
- **Description:** Taillard randomly generated uniform instance (larger)

### 8. chr20a.dat (Christofides Problem)
- **Size:** 2.0 KB
- **Problem Size:** n=20
- **Known Optimal Solution:** 2192
- **Description:** Christofides real-life problem instance (larger)

### 9. chr20b.dat (Christofides Problem)
- **Size:** 2.0 KB
- **Problem Size:** n=20
- **Known Optimal Solution:** 2298
- **Description:** Christofides real-life problem instance variant

### 10. rou20.dat (Roucairol Problem)
- **Size:** 2.4 KB
- **Problem Size:** n=20
- **Known Optimal Solution:** 725522
- **Description:** Roucairol real-life problem instance

### n=30 Instances (MEDIUM PRIORITY)

### 11. tai30a.dat (Taillard Problem)
- **Size:** 5.4 KB
- **Problem Size:** n=30
- **Known Optimal Solution:** 1818146
- **Description:** Taillard randomly generated uniform instance

### 12. ste36a.dat (Steinberg Problem)
- **Size:** 13 KB
- **Problem Size:** n=36
- **Known Optimal Solution:** 9526
- **Description:** Steinberg wiring problem

### n=40-50 Instances (SCALABILITY TESTING)

### 13. tai40a.dat (Taillard Problem)
- **Size:** 9.5 KB
- **Problem Size:** n=40
- **Known Optimal Solution:** 3139370
- **Description:** Taillard randomly generated uniform instance

### 14. tai50a.dat (Taillard Problem)
- **Size:** 15 KB
- **Problem Size:** n=50
- **Known Optimal Solution:** 4938796
- **Description:** Taillard randomly generated uniform instance

## File Format

All files follow the standard QAPLIB format:

```
n                           # problem size
flow_matrix                 # n lines of n integers (flow/weight matrix)
                            # blank line
distance_matrix             # n lines of n integers (distance/cost matrix)
```

## Validation Status

### Initial n=12 Instances
- File downloads: SUCCESS (4 files)
- File sizes: VERIFIED (all files within expected range of 0.7-1.7 KB)
- File format: VERIFIED (all files contain correct n×n matrix structure)
- Readable content: VERIFIED (all files are plain text with integer matrices)

### Larger Instances (n=20-50)
- File downloads: SUCCESS (10 files)
- File sizes: VERIFIED (2.0 KB - 15 KB, appropriate for problem sizes)
- File format: VERIFIED (all files successfully parsed by load_qaplib_instance)
- Matrix dimensions: VERIFIED (all n×n matrices correctly loaded)
- Total instances: 14 files (n=12: 4, n=20: 6, n=30: 1, n=36: 1, n=40: 1, n=50: 1)

## Usage Notes

### Small Instances (n=12)
These files are suitable for:
- Initial validation testing of QAP solvers
- Algorithm benchmarking and comparison
- Small-scale performance testing
- Verifying correctness of implementations
- Quick iteration during development

### Larger Instances (n=20-50)
These files are suitable for:
- **FFT Threshold Testing (n=20):** Critical for determining when FFT-Laplace convolution becomes beneficial
- **Scalability Analysis (n=30-50):** Testing algorithm performance as problem size increases
- **Performance Benchmarking:** Comparing direct vs. FFT-based methods
- **UltraThink Framework:** Testing adaptive threshold decisions
- **Production Validation:** Realistic problem sizes for many applications

The optimal solutions listed above can be used to verify that your QAP solver is producing correct results.

## References

1. Burkard, R.E., Karisch, S.E., and Rendl, F. (1997). QAPLIB - A Quadratic Assignment Problem Library. Journal of Global Optimization, 10, 391-403.
2. QAPLIB Website: https://coral.ise.lehigh.edu/data-sets/qaplib/
3. GitHub Mirror: https://github.com/mhahsler/qap
