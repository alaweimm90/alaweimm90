# Research Topic [003]: Formal Survival Score Specification & Calibration

## Question/Goal
Establish a formal, reproducible Survival Score specification with proper calibration methods to ensure consistency across Nightmare Mode evaluations.

## Key Findings
1. **Score Components** (Based on ORCHEX priorities):
   - Statistical robustness (25%): Tests for p-hacking, multiple comparisons, power analysis
   - Methodological integrity (25%): Study design, confounders, selection bias
   - Logical consistency (20%): Circular reasoning, false equivalence, hasty generalization
   - Ethical compliance (15%): IRB requirements, dual-use concerns, consent
   - Economic feasibility (15%): Cost-benefit, resource requirements, scalability

2. **Calibration Requirements**:
   - Bootstrap confidence intervals for score stability
   - Cross-validation across different hypothesis types
   - Expected Calibration Error (ECE) target: < 0.05
   - Inter-rater reliability for human baselines

## MCP Research Opportunities
- [ ] Literature search: "adversarial evaluation calibration methods"
- [ ] Statistical frameworks: "bootstrap confidence intervals hypothesis testing"
- [ ] Benchmarks: "red team evaluation metrics"

## Proposed Improvement
Enhance Nightmare_Mode.md with:
1. Explicit score component weights configurable via parameters
2. Calibration plot generation requirement
3. Confidence interval reporting
4. Versioned scoring rubric with change tracking

## Validation Plan
1. Create test inputs spanning difficulty tiers
2. Generate manifests with score_breakdown field populated
3. Verify ECE < 0.05 across 100 test hypotheses
4. Compare against human expert baselines

## Implementation Status
- [ ] Score specification formalized
- [ ] Calibration code implemented
- [ ] Test suite created
- [ ] Documentation updated

## Citations
- Priority [003] from ATLAS_DEEP_ANALYSIS_500.md
- Input schema: NEW2/.meta/schemas/input.schema.json
- Artifact schema: NEW2/.meta/schemas/artifact.schema.json