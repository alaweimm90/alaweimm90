# Week 5 Summary: Real ASlib Data Integration

**Status**: ✅ COMPLETE
**Date**: Week 5 of 12-week timeline (March 31, 2025 deadline)
**Target**: AutoML Conference 2025 submission

---

## Achievements

### 1. ARFF Parser Implementation (Week 5.1) ✅

**File Created**: `benchmark/aslib_parser.py` (~380 lines)

**Classes Implemented**:
1. `ARFFParser` - Parses .arff files into pandas DataFrames
2. `ASLibScenarioLoader` - Loads complete ASlib scenarios with features and performance data

**Key Features**:
- Case-insensitive ARFF directive parsing (@DATA, @ATTRIBUTE)
- Automatic type conversion with validation (>50% values must be numeric)
- Handles missing values and NaN gracefully
- Supports all ASlib file formats:
  - `feature_values.arff` - Problem instance features
  - `algorithm_runs.arff` - Algorithm performance data
  - `cv.arff` - Cross-validation splits
  - `description.txt` - Scenario metadata

**Parser Fixes Applied**:
- Made @DATA and @ATTRIBUTE parsing case-insensitive (handles both uppercase and lowercase)
- Smart numeric conversion: uses `errors='coerce'` with 50% validity threshold
- Skips known string columns (instance_id, algorithm, runstatus, type)

---

### 2. Evaluation Pipeline Integration (Week 5.2) ✅

**File Modified**: `benchmark/run_evaluation.py` (+~200 lines)

**New Methods Added**:
- `_run_real_scenario()` - Evaluates methods on real ASlib data
- `_evaluate_method_on_real_instances()` - Computes metrics on real instances

**Integration Features**:
- Seamless switching between mock and real data (`use_mock_data` parameter)
- Creates algorithm wrappers for real ASlib solvers
- Uses 80/20 train/test split (cv splits support to be added later)
- Passes features directly to avoid re-extraction issues

**Test Created**: `benchmark/test_real_data_integration.py`
**Status**: ✅ All integration tests passing

---

### 3. Real Data Validation (Week 5.3) ✅

**Test Created**: `benchmark/test_optimal_config.py`

**Scenarios Tested**:
- **SAT11-HAND**: 296 instances, 15 algorithms, 105 features ✅
- **CSP-2010**: 2,024 instances, 2 algorithms, 86 features ✅
- **GRAPHS-2015**: (pending)

---

## Key Findings

### Finding 1: Hyperparameters from Mock Data Don't Transfer to Real Data 🔥

**Critical Discovery**: The "optimal" configuration from Week 4 ablation studies (on mock data) **performed worse** on real SAT11-HAND data.

**SAT11-HAND Results (Real Data)**:

| Configuration | Avg Regret | Top-1 Acc | Top-3 Acc |
|---------------|------------|-----------|-----------|
| **Default** (n_clusters=5, elo_k=32, ucb_c=1.414, n_rounds=5) | **0.112** ✅ | **16.7%** ✅ | **41.7%** ✅ |
| **"Optimal"** (n_clusters=20, elo_k=16, ucb_c=0.5, n_rounds=10) | **0.135** ❌ | 15.0% ❌ | 40.0% ❌ |

**Performance Change**:
- **Regret: +21.1%** (WORSE, not better!)
- **Top-1 Accuracy: -10.0%** (WORSE)
- **Top-3 Accuracy: -4.0%** (WORSE)

**Implications**:
1. Mock data has **different characteristics** than real ASlib data
2. Hyperparameter tuning must be done on **real data**
3. Week 4 ablation studies need to be **re-run with real data**
4. Generalization across scenarios is crucial

---

### Finding 2: Baseline Comparison on Real Data

**SAT11-HAND Scenario (Default Librex.Meta config)**:

| Method | Avg Regret | Top-1 Acc | Top-3 Acc | Sel Time (ms) |
|--------|------------|-----------|-----------|---------------|
| **AutoFolio** | **0.059** 🥇 | **56.7%** 🥇 | **75.0%** 🥇 | 28.6 |
| **SMAC** | **0.042** 🥈 | 38.3% 🥈 | 56.7% | 37.6 |
| **SATzilla** | 0.045 | 31.7% | 46.7% | 518.5 |
| **Librex.Meta (default)** | 0.112 | 16.7% | 41.7% | **0.24** 🥇 |
| **Hyperband** | 0.126 | 13.3% | 28.3% | **0.16** 🥈 |
| **BOHB** | 0.126 | 13.3% | 28.3% | 1.22 |

**Librex.Meta Performance (Default Config)**:
- **Rank**: 4th out of 6 methods
- **Speed**: 2,162x faster than SATzilla, 119x faster than AutoFolio
- **Regret**: 2.7x worse than SMAC (best)
- **Top-1 Accuracy**: 3.4x worse than AutoFolio (best)

**Status**: Librex.Meta is currently **not competitive** with state-of-the-art on real data

---

### Finding 3: Scenario-Specific Characteristics

**SAT11-HAND** (Boolean Satisfiability):
- 296 instances, 15 algorithms, 105 features
- High feature dimensionality
- Many algorithms to choose from
- **AutoFolio performs best** (feature selection + portfolio)

**CSP-2010** (Constraint Satisfaction):
- 2,024 instances, 2 algorithms, 86 features
- Smaller algorithm portfolio
- Binary choice problem
- Simpler selection task

**GRAPHS-2015** (Graph Problems):
- (To be evaluated)

---

## Technical Challenges & Solutions

### Challenge 1: Case-Sensitive ARFF Parsing

**Problem**: SAT11-HAND cv.arff uses lowercase `@data` but parser expected `@DATA`

**Solution**: Made parsing case-insensitive
```python
if line.strip().upper().startswith('@DATA'):
```

**Files Affected**: `benchmark/aslib_parser.py:37, 47`

---

### Challenge 2: Feature Type Conversion Failure

**Problem**: CSP-2010 features were all parsed as 'object' dtype instead of numeric
**Root Cause**: `pd.to_numeric(errors='ignore')` doesn't convert anything

**Solution**: Use `errors='coerce'` with validation
```python
converted = pd.to_numeric(df[col], errors='coerce')
# Only use if >50% values are valid numbers
if converted.notna().sum() > len(converted) * 0.5:
    df[col] = converted
```

**Result**: CSP-2010 now correctly loads 86 numeric features

---

### Challenge 3: Feature Dimension Mismatch

**Problem**: KMeans trained with 105 features, but feature_extractor returned 10 features
**Root Cause**: Feature extraction was called on wrapped instance instead of using pre-extracted features

**Solution**: Pass features directly to select_solver
```python
selected = method.select_solver(instance, features=features)
```

**Files Affected**: `benchmark/run_evaluation.py:379-382`

---

## Code Statistics

### New Code (Week 5)

| Component | Lines | Purpose |
|-----------|-------|---------|
| `aslib_parser.py` | 380 | ARFF parsing + ASlib loading |
| `run_evaluation.py` (additions) | ~200 | Real data integration |
| `test_real_data_integration.py` | 40 | Integration test |
| `test_optimal_config.py` | 120 | Config validation test |
| `validate_config_phase1.py` | 80 | Phase 1 validation script |
| **Total New Code** | **~820** | |

### Cumulative Progress

| Component | Files | Lines | Tests | Status |
|-----------|-------|-------|-------|--------|
| Core Librex.Meta | 2 | 410 | 11 | ✅ |
| Baselines | 6 | 1,306 | 8 | ✅ |
| Evaluation Pipeline | 4 | 1,000 | 5 | ✅ |
| Statistical Analysis | 1 | 400 | 0 | ✅ |
| Ablation Studies | 1 | 600 | 0 | ✅ |
| **ASlib Integration** | **3** | **820** | **0** | ✅ |
| **Total Production** | **17** | **~4,536** | **24** | ✅ |

---

## Next Steps: Week 6+

### Priority 1: Re-Run Ablation Studies on Real Data (Week 6)

**Problem**: Mock data hyperparameters don't generalize
**Solution**: Re-run Week 4 ablation studies on real ASlib scenarios

**Tasks**:
1. Modify `ablation_studies.py` to use real data
2. Test on SAT11-HAND, CSP-2010, GRAPHS-2015
3. Find hyperparameters that work across multiple real scenarios
4. Compare real-data optimal vs. mock-data optimal

**Expected Outcome**: New optimal configuration for real data

---

### Priority 2: Improve Librex.Meta Baseline Performance

**Current Issue**: Librex.Meta ranks 4th/6 on SAT11-HAND

**Possible Improvements**:
1. Better feature extraction (currently uses mock feature_extractor)
2. Proper cross-validation instead of simple 80/20 split
3. Ensemble methods (combine Librex.Meta with baselines)
4. Algorithm-specific Elo initialization
5. Online learning during test phase

---

### Priority 3: Phase 2 Evaluation (8-10 Scenarios)

Expand evaluation to more scenarios:
1. SAT11-HAND ✅
2. CSP-2010 ✅
3. GRAPHS-2015 (pending)
4. QBF-2011
5. ASP-POTASSCO
6. MAXSAT12-PMS
7. SAT12-ALL
8. SAT12-HAND

---

## Research Insights

### Insight 1: Mock vs. Real Data Gap

**Observation**: Hyperparameters tuned on mock data **fail to transfer** to real data.

**Why This Matters**:
- Mock data is useful for **pipeline development**
- Mock data is **not sufficient** for hyperparameter tuning
- Real ASlib data must be used for **final validation**

**Impact on Timeline**:
- Week 4 ablation studies results are **not usable** for the paper
- Must re-run ablation studies on real data (adds ~1 week)

---

### Insight 2: AutoFolio Dominance on SAT11-HAND

**Observation**: AutoFolio significantly outperforms all other methods (including Librex.Meta).

**Hypothesis**:
- Feature selection is crucial for SAT problems (105 features)
- Portfolio-based approaches work well with many algorithms (15 solvers)
- Librex.Meta's clustering may not capture SAT problem structure

**Potential Improvements**:
- Add feature selection to Librex.Meta
- Use domain-specific features for clustering
- Investigate what makes AutoFolio successful

---

### Insight 3: Speed vs. Accuracy Trade-off

**Observation**: Librex.Meta is **very fast** (0.24ms) but **less accurate** than baselines.

**Trade-off Analysis**:
- **Fastest**: Hyperband (0.16ms), Librex.Meta (0.24ms)
- **Most Accurate**: AutoFolio (56.7% top-1), SMAC (38.3% top-1)
- **Best Balance**: SMAC (0.042 regret, 37.6ms selection time)

**Implication**: Librex.Meta's speed advantage may not justify accuracy loss

---

## Revised Timeline Assessment

### Original Plan (from Week 4)

| Week | Milestone | Original Status |
|------|-----------|-----------------|
| 5 | Real Data Integration | ✅ COMPLETE |
| 6 | Optimal Config Validation | ⚠️ Needs Re-work |
| 7-8 | Full Benchmarking | On track |
| 9 | Visualization | On track |
| 10-11 | Paper Writing | On track |
| 12 | Submission (March 31) | At risk |

### Revised Plan

| Week | Milestone | Revised Status |
|------|-----------|----------------|
| **5** | **Real Data Integration** | ✅ **COMPLETE** |
| **6** | **Re-Run Ablation Studies on Real Data** | 🔄 **CRITICAL** |
| **7** | **Phase 2 Evaluation (8 scenarios)** | High priority |
| **8** | **Improve Librex.Meta Performance** | Medium priority |
| **9** | **Full Benchmarking + Analysis** | Medium priority |
| **10-11** | **Paper Writing** | High priority |
| **12** | **Submission (March 31)** | 🎯 **ON TRACK (tight)** |

---

## Risk Assessment

### Risk 1: Librex.Meta Not Competitive ⚠️ HIGH

**Issue**: Librex.Meta ranks 4th/6 on SAT11-HAND with default config
**Impact**: Paper contribution may be weak if performance doesn't improve
**Mitigation**:
- Re-tune hyperparameters on real data
- Investigate AutoFolio's success factors
- Consider ensemble approaches
- Focus on specific problem domains where Librex.Meta excels

---

### Risk 2: Hyperparameter Generalization 🔴 CRITICAL

**Issue**: Mock data hyperparameters don't transfer to real data
**Impact**: Need to re-run all ablation studies (adds ~1 week)
**Mitigation**:
- Prioritize ablation studies on real data (Week 6)
- Test across multiple scenarios for generalization
- Use conservative parameter settings if no clear optimum found

---

### Risk 3: Time Constraint 🔴 CRITICAL

**Issue**: 7 weeks remaining, need competitive results for paper
**Impact**: May not have time for thorough evaluation + writing
**Mitigation**:
- Focus on 8-10 scenarios (not 15)
- Start paper writing in parallel (Week 10)
- Prepare supplementary materials early
- Have fallback: "competitive in speed, moderate in accuracy" story

---

## Success Metrics (Updated)

### Minimum Viable Paper

- ✅ Core Librex.Meta implementation
- ✅ 5 baseline comparisons
- ✅ Real ASlib parser and integration
- ⏳ Real data ablation study (Week 6)
- ⏳ Evaluation on 5-8 scenarios (Week 7-9)
- ⏳ Statistical significance testing
- ⏳ Paper draft (Week 10-11)

### Strong Paper

- ⏳ Librex.Meta competitive with baselines (top 3)
- ⏳ Novel hyperparameter insights (on real data)
- ⏳ Evaluation on 10-12 scenarios
- ⏳ Comprehensive statistical analysis
- ⏳ High-quality visualizations

### Exceptional Paper (Stretch)

- ⏳ Librex.Meta outperforms baselines on some scenarios
- ⏳ UCB + Elo synergy validated on real data
- ⏳ 15+ scenarios evaluated
- ⏳ Open-source release

---

## Conclusion

**Week 5 Status**: ✅ **COMPLETE**

Successfully completed:
- ✅ ARFF parser implementation (380 lines)
- ✅ Real ASlib data integration (200 lines)
- ✅ Evaluation pipeline modification
- ✅ SAT11-HAND validation (AutoFolio best, Librex.Meta 4th)
- ✅ CSP-2010 parser fixes (86 features extracted)

**Major Discovery**: 🔥 **Hyperparameters from mock data don't transfer to real data**

**Critical Finding**: Librex.Meta (default config) is **not yet competitive** with state-of-the-art on real ASlib data.

**Next**: Week 6 - **Re-run ablation studies on real data** to find truly optimal hyperparameters

**Timeline**: 🎯 **ON TRACK** (but tight - 7 weeks remaining)

---

**Generated**: Week 5 Completion
**Author**: Itqān Libria Suite Development Team
**Target**: AutoML Conference 2025 (September 8-11, Vancouver)
