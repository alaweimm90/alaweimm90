# Research Topic [007]: Red-Team Adversarial Suite + Regression Guards

## Question/Goal
Build a comprehensive, versioned red-team attack catalog with regression testing to ensure consistent adversarial coverage across hypothesis evaluations.

## Key Findings
1. **Attack Taxonomy Structure**:
   - Statistical Attacks (50+ vectors): p-hacking, cherry-picking, Simpson's paradox, base rate fallacy
   - Methodological Attacks (40+ vectors): selection bias, confounding, measurement error, attrition
   - Logical Attacks (30+ vectors): ad hominem, strawman, false dichotomy, circular reasoning
   - Domain-Specific Attacks (50+ vectors): field-specific vulnerabilities
   - Emergent Attacks (30+ vectors): LLM-specific biases, prompt injection, hallucination triggers

2. **Regression Guard Requirements**:
   - Maintain golden set of 100 test hypotheses
   - Track attack coverage metrics over time
   - Alert on coverage drops > 5%
   - Version control attack definitions

## MCP Research Opportunities
- [ ] "Red team evaluation frameworks" - Microsoft, Google, OpenAI practices
- [ ] "Adversarial robustness benchmarks" - academic literature
- [ ] "Statistical hypothesis testing failures" - meta-research studies
- [ ] "LLM vulnerability taxonomies" - recent security research

## Proposed Improvement
Enhance attack catalog with:
1. Unique attack IDs for tracking (e.g., STAT-001, METH-012)
2. Severity levels (Low/Medium/High/Critical)
3. Evidence requirements per attack type
4. Mitigation templates
5. Regression test suite with CI integration

## Validation Plan
1. Catalog completeness: ≥200 unique attacks documented
2. Coverage testing: 95% of attacks triggered on test set
3. Regression stability: No coverage drops across versions
4. Performance: Attack generation < 100ms per vector

## Implementation Status
- [ ] Attack catalog created with IDs
- [ ] Regression test suite implemented
- [ ] CI integration configured
- [ ] Documentation updated

## Citations
- Priority [007] from ATLAS_DEEP_ANALYSIS_500.md
- Related to [003] for scoring integration
- Feeds into [060] shadow evaluations