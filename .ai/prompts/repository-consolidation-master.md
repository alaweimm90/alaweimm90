# Repository Consolidation Master Prompt

## Mission

Transform the GitHub workspace from 47 root folders into a streamlined 10-folder architecture while preserving all functionality, improving governance, and maintaining zero data loss.

## Current State Analysis

- **47 root-level folders** with significant overlap
- **Scattered infrastructure** across deploy/, .metaHub/, templates/
- **Fragmented tooling** in tools/, scripts/, ai-tools/
- **Empty/sparse K8s folders** (api-gateway/, configmaps/, etc.)
- **Cache pollution** (.mypy_cache/, .ruff_cache/)

## Target Architecture

```
GitHub/
├── .ai/                    # Unified AI orchestration (merge ai-tools/)
├── .amazonq/               # Memory bank (keep as-is)
├── .metaHub/               # Consolidated DevOps (merge deploy/, templates/, tools/)
├── organizations/          # Multi-tenant business logic (keep as-is)
├── docs/                   # Documentation hub (merge examples/)
├── tests/                  # Quality assurance (keep as-is)
├── bin/                    # CLI executables (keep as-is)
├── enterprise/             # Advanced features (keep as-is)
├── ecosystem/              # SDK integrations (keep as-is)
└── [root configs]          # Essential configs only
```

## Consolidation Strategy

### Phase 1: Infrastructure Unification

**Merge**: `deploy/` + `.metaHub/infra/` + `templates/` → `.metaHub/`

**Actions**:

1. Move `deploy/docker/` → `.metaHub/docker/`
2. Move `deploy/kubernetes/` → `.metaHub/k8s/`
3. Move `deploy/terraform/` → `.metaHub/infra/terraform/`
4. Move `deploy/monitoring/` → `.metaHub/monitoring/`
5. Move `templates/devops/` → `.metaHub/templates/devops/`
6. Move `templates/org-github/` → `.metaHub/templates/org-github/`
7. Delete empty `deploy/` folder

**Validation**: Verify all CI/CD references updated

### Phase 2: Tooling Consolidation

**Merge**: `tools/` + `scripts/` → `.metaHub/tools/`

**Actions**:

1. Move `tools/ai/` → `.metaHub/tools/ai/`
2. Move `tools/atlas/` → `.metaHub/tools/atlas/`
3. Move `tools/cli/` → `.metaHub/tools/cli/`
4. Move `tools/devops/` → `.metaHub/tools/devops/`
5. Move `tools/security/` → `.metaHub/tools/security/`
6. Move `scripts/*.{ts,py,sh,ps1}` → `.metaHub/scripts/`
7. Archive `tools/legacy/` → `.archive/tools-legacy/`
8. Delete empty `tools/` and `scripts/` folders

**Validation**: Test all CLI commands still work

### Phase 3: AI System Integration

**Merge**: `ai-tools/` → `.ai/tools/`

**Actions**:

1. Move `ai-tools/src/` → `.ai/tools/src/`
2. Move `ai-tools/integrations/` → `.ai/integrations/`
3. Move `ai-tools/package.json` → `.ai/tools/package.json`
4. Update import paths in all files
5. Delete empty `ai-tools/` folder

**Validation**: Test AI orchestration workflows

### Phase 4: Kubernetes Cleanup

**Consolidate**: Sparse K8s folders → `.metaHub/k8s/`

**Actions**:

1. Move `api-gateway/` → `.metaHub/k8s/api-gateway/` (if has content)
2. Move `configmaps/` → `.metaHub/k8s/configmaps/` (if has content)
3. Move `elasticsearch/` → `.metaHub/k8s/elasticsearch/` (if has content)
4. Move `grafana/` → `.metaHub/k8s/grafana/` (if has content)
5. Move `ingress/` → `.metaHub/k8s/ingress/` (if has content)
6. Move `kibana/` → `.metaHub/k8s/kibana/` (if has content)
7. Move `storage/` → `.metaHub/k8s/storage/` (if has content)
8. Move `orchestration/` → `.metaHub/orchestration/` (if has content)
9. Delete empty folders with `}/` suffix (backup-recovery}, base}, etc.)

**Validation**: Check K8s manifests reference correct paths

### Phase 5: Documentation Consolidation

**Merge**: `examples/` → `docs/examples/`

**Actions**:

1. Move `examples/atlas/` → `docs/examples/atlas/`
2. Delete empty `examples/` folder

**Validation**: Update documentation links

### Phase 6: Cache & Artifact Cleanup

**Remove**: Build artifacts and cache folders

**Actions**:

1. Add to `.gitignore`:
   ```
   .mypy_cache/
   .ruff_cache/
   __pycache__/
   *.pyc
   ```
2. Delete `.mypy_cache/` folder
3. Delete `.ruff_cache/` folder
4. Run `git rm -r --cached .mypy_cache .ruff_cache`

**Validation**: Verify clean git status

### Phase 7: Empty Folder Removal

**Delete**: Truly empty or placeholder folders

**Actions**:

1. Check and remove: `maintenance/`, `troubleshooting/`, `optimization/`
2. Check and remove: `execution/`, `scaling/` (if empty)
3. Check and remove: `demo/` (if obsolete)

**Validation**: Confirm no references in code

## Critical Safeguards

### Pre-Consolidation Checklist

- [ ] Create full backup: `git branch backup-pre-consolidation`
- [ ] Document current folder structure: `tree -L 2 > structure-before.txt`
- [ ] Run full test suite: `npm test && pytest`
- [ ] Verify CI/CD pipelines passing
- [ ] Check no uncommitted changes: `git status`

### During Consolidation

- [ ] Move files, don't copy (preserve git history)
- [ ] Update all import paths immediately
- [ ] Update CI/CD configuration files
- [ ] Update documentation references
- [ ] Test after each phase

### Post-Consolidation Validation

- [ ] Run full test suite again
- [ ] Verify all CLI commands work
- [ ] Check CI/CD pipelines still pass
- [ ] Validate AI orchestration workflows
- [ ] Review git diff for unintended changes
- [ ] Update Memory Bank documentation

## File Path Updates Required

### Package.json Scripts

```json
{
  "devops": "tsx .metaHub/tools/cli/devops.ts",
  "atlas": "tsx .metaHub/tools/atlas/cli/index.ts",
  "ai": "tsx .ai/tools/orchestrator.ts"
}
```

### Import Path Patterns

```typescript
// Before
import { tool } from '../tools/ai/orchestrator';
import { deploy } from '../deploy/docker/config';

// After
import { tool } from '../.ai/tools/orchestrator';
import { deploy } from '../.metaHub/docker/config';
```

### CI/CD Workflow Updates

```yaml
# Update all .github/workflows/*.yml
- uses: ./.metaHub/templates/devops/
- run: tsx .metaHub/tools/cli/devops.ts
```

## Success Metrics

- ✅ Reduced from 47 to ~10 root folders
- ✅ All tests passing
- ✅ CI/CD pipelines functional
- ✅ Zero data loss
- ✅ Improved governance scores
- ✅ Faster navigation and discovery
- ✅ Clearer separation of concerns

## Rollback Plan

If issues arise:

```bash
git checkout backup-pre-consolidation
git branch -D consolidation-attempt
```

## Execution Order

1. Phase 6 (Cache cleanup) - safest, no dependencies
2. Phase 5 (Documentation) - minimal impact
3. Phase 7 (Empty folders) - low risk
4. Phase 4 (K8s cleanup) - moderate complexity
5. Phase 1 (Infrastructure) - high impact, test thoroughly
6. Phase 2 (Tooling) - high impact, test thoroughly
7. Phase 3 (AI integration) - critical, test extensively

## Notes

- Use `git mv` to preserve history
- Commit after each phase
- Tag major milestones: `git tag consolidation-phase-1`
- Update Memory Bank after completion
