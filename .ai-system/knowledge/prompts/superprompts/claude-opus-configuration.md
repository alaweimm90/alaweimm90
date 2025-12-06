# Instructions for Claude Opus: Repository Consolidation

## Your Mission

Execute a comprehensive repository consolidation to transform 47 root folders into 10 streamlined directories while maintaining zero data loss and full functionality.

## Available Prompts

You have access to these detailed execution guides:

1. **Master Prompt**: `repository-consolidation-master.md` - Overall strategy
2. **Phase 1**: `phase-1-infrastructure.md` - Infrastructure consolidation
3. **Phase 2**: `phase-2-tooling.md` - Tooling consolidation
4. **Phase 3**: `phase-3-ai-integration.md` - AI system integration

## Execution Protocol

### Step 1: Read and Understand

```
Read all prompt files in this order:
1. repository-consolidation-master.md
2. phase-1-infrastructure.md
3. phase-2-tooling.md
4. phase-3-ai-integration.md
```

### Step 2: Pre-Flight Safety

Before ANY changes:

```bash
# Create backup branch
git checkout -b backup-pre-consolidation

# Document current state
tree -L 2 > structure-before-consolidation.txt

# Verify clean state
git status

# Run tests
npm test
pytest
```

### Step 3: Execute Phases in Order

#### Phase 6 First (Safest)

Cache cleanup - no dependencies:

```bash
# Add to .gitignore
echo ".mypy_cache/" >> .gitignore
echo ".ruff_cache/" >> .gitignore

# Remove from git
git rm -r --cached .mypy_cache .ruff_cache

# Delete folders
rm -rf .mypy_cache .ruff_cache

# Commit
git add .gitignore
git commit -m "chore: remove cache folders from repository"
```

#### Phase 5 (Low Risk)

Documentation consolidation:

```bash
git mv examples/ORCHEX/ docs/examples/ORCHEX/
rmdir examples/
git commit -m "docs: consolidate examples into docs/"
```

#### Phase 7 (Low Risk)

Empty folder removal - verify each is truly empty first:

```bash
# Check if empty
ls -la maintenance/ troubleshooting/ optimization/

# If empty, remove
rmdir maintenance/ troubleshooting/ optimization/
git commit -m "chore: remove empty placeholder folders"
```

#### Phase 4 (Moderate)

Kubernetes cleanup - check content first:

```bash
# For each K8s folder, check if it has content
ls -la api-gateway/ configmaps/ elasticsearch/ grafana/ ingress/ kibana/ storage/

# If has content, move to .metaHub/k8s/
git mv api-gateway/ .metaHub/k8s/api-gateway/

# If empty, just delete
rmdir api-gateway/

# Commit after all K8s moves
git commit -m "refactor: consolidate Kubernetes resources into .metaHub/k8s/"
```

#### Phase 1 (High Impact)

Infrastructure consolidation - follow phase-1-infrastructure.md exactly:

```bash
# Execute each step from the prompt
# Test after each major move
# Commit frequently
```

#### Phase 2 (High Impact)

Tooling consolidation - follow phase-2-tooling.md exactly:

```bash
# Execute each step from the prompt
# Update import paths immediately
# Test CLI commands after each move
```

#### Phase 3 (Critical)

AI integration - follow phase-3-ai-integration.md exactly:

```bash
# Execute each step from the prompt
# Test AI orchestration thoroughly
# Verify all integrations work
```

### Step 4: Post-Consolidation Validation

#### Run Full Test Suite

```bash
npm test
pytest
npm run type-check
npm run lint
```

#### Verify CLI Commands

```bash
npm run devops -- --help
npm run ORCHEX -- --help
npm run ai:start
```

#### Check CI/CD

```bash
# Push to test branch
git push origin consolidation-phase-1

# Monitor GitHub Actions
# Verify all workflows pass
```

#### Update Memory Bank

```bash
# Regenerate memory bank with new structure
# Update structure.md with new folder organization
```

### Step 5: Final Documentation

Create consolidation report:

```markdown
# Repository Consolidation Report

## Summary

- Reduced from 47 to 10 root folders
- Zero data loss
- All tests passing
- CI/CD functional

## Changes Made

[List all moves and updates]

## Validation Results

[Test results and metrics]

## Breaking Changes

[Document any breaking changes]

## Migration Guide

[Instructions for developers]
```

## Critical Rules

### DO

✅ Use `git mv` to preserve history
✅ Commit after each phase
✅ Test after each major change
✅ Update import paths immediately
✅ Document all changes
✅ Create backup branches
✅ Verify before deleting

### DON'T

❌ Copy files (use move)
❌ Delete without verification
❌ Skip testing
❌ Make multiple phases in one commit
❌ Ignore broken imports
❌ Rush through validation
❌ Forget to update documentation

## Error Handling

### If Tests Fail

```bash
# Rollback to last good state
git checkout backup-pre-consolidation

# Investigate issue
# Fix and retry
```

### If Import Paths Break

```bash
# Use global search/replace
# Pattern: old-path → new-path
# Verify with type-check
```

### If CI/CD Breaks

```bash
# Check workflow files
# Update path references
# Test locally first
```

## Communication Protocol

### After Each Phase

Report:

1. What was moved
2. What was updated
3. Test results
4. Any issues encountered
5. Next steps

### If Blocked

Ask for:

1. Clarification on ambiguous folders
2. Permission for destructive operations
3. Guidance on complex dependencies

## Success Criteria Checklist

- [ ] All 7 phases completed
- [ ] 47 folders reduced to ~10
- [ ] All tests passing
- [ ] CI/CD pipelines functional
- [ ] Import paths updated
- [ ] Documentation updated
- [ ] Memory Bank regenerated
- [ ] Git history preserved
- [ ] Zero data loss
- [ ] Consolidation report created

## Final Command

After all phases complete:

```bash
# Merge to main
git checkout main
git merge consolidation-phase-3
git tag v2.0.0-consolidated
git push origin main --tags

# Delete backup branches
git branch -D backup-pre-consolidation
```

## Notes

- Take your time - accuracy over speed
- When in doubt, ask before deleting
- Document everything
- Test thoroughly
- Preserve git history
- Communicate progress clearly
