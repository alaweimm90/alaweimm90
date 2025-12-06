# Research Topic [019]: Hierarchical Seed Strategy + Run Manifests

## Question/Goal
Implement a hierarchical seed management system that ensures reproducibility across all ORCHEX runs while maintaining proper experiment tracking through comprehensive run manifests.

## Key Findings
1. **Hierarchical Seed Structure**:
   - Global seed: Controls entire experiment
   - Feature seed: Per-superprompt reproducibility
   - Component seed: For specific operations (attack generation, evolution mutations)
   - Instance seed: For individual evaluations

2. **Run Manifest Requirements**:
   - Unique run ID with timestamp
   - Complete seed hierarchy
   - Input fingerprint
   - Environment snapshot
   - Dependency versions
   - Result checksums

3. **Reproducibility Guarantees**:
   - Deterministic random number generation
   - Seed inheritance rules
   - State checkpoint/restore
   - Version pinning

## MCP Research Opportunities
- [ ] "Reproducible ML experiments" - best practices from MLflow, W&B
- [ ] "Hierarchical random seeds" - distributed systems approaches
- [ ] "Experiment tracking systems" - Neptune, Comet, TensorBoard
- [ ] "Deterministic evaluation" - addressing non-determinism in LLMs

## Proposed Improvement
1. Create seed management system with inheritance
2. Implement run manifest generator
3. Add checkpoint/restore capabilities
4. Build reproducibility validator

## Validation Plan
1. Test exact reproducibility with same seeds (100% match)
2. Verify different seeds produce different results
3. Validate checkpoint/restore functionality
4. Test seed inheritance across components

## Implementation Status
- [ ] Seed hierarchy defined
- [ ] Run manifest schema created
- [ ] Reproducibility tests implemented
- [ ] Documentation updated

## Citations
- Priority [019] from ATLAS_DEEP_ANALYSIS_500.md
- Supports [060] shadow evaluations
- Enables [149] comparative analysis