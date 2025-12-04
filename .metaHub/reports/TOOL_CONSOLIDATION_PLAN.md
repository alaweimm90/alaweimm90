# Tool Consolidation Plan

## Summary

**Total Savings: 121 duplicate configs across 64 projects**

## Global Tools Available (126)

metaHub provides centralized tools for:
- **CI/CD**: Workflows, testing, deployment
- **Security**: Secret scanning, SAST, dependency checks
- **Compliance**: Policy enforcement, validation
- **AI**: Orchestration, caching, monitoring
- **DevOps**: Atlas, governance, automation

## Consolidation Opportunities

### alaweimm90-business (9 projects)
- **Savings**: 21 configs
- **Actions**:
  - CI: 9 projects → 1 global workflow
  - Lint: 5 projects → 1 eslint.config.js
  - Build: 4 projects → 1 vite.config.ts
  - Format: 3 projects → 1 .prettierrc
  - Docker: 3 projects → 1 Dockerfile template

### alaweimm90-science (6 projects)
- **Savings**: 11 configs
- **Actions**:
  - CI: 6 projects → 1 global workflow
  - Lint: 5 projects → 1 ruff.toml
  - Docker: 3 projects → 1 Dockerfile template

### AlaweinOS (19 projects)
- **Savings**: 47 configs
- **Actions**:
  - CI: 19 projects → 1 global workflow
  - Lint: 9 projects → 1 config
  - Test: 9 projects → 1 vitest.config.ts
  - Docker: 7 projects → 1 template
  - Build: 5 projects → 1 config

### MeatheadPhysicist (30 projects)
- **Savings**: 42 configs
- **Actions**:
  - CI: 30 projects → 1 global workflow
  - Lint: 9 projects → 1 config
  - Docker: 6 projects → 1 template

## Implementation

### Phase 1: Create Global Configs
```bash
cd .metaHub/scripts
python consolidate_tools.py
```

Creates:
- `tools/config/eslint.config.js`
- `tools/config/vitest.config.ts`
- `tools/config/playwright.config.ts`
- `tools/config/ruff.toml`

### Phase 2: Replace Project Configs
Replaces project-specific configs with symlinks to global:
- Backs up originals to `.backup_*`
- Creates relative symlinks
- Maintains project-specific overrides where needed

### Phase 3: Update CI Workflows
All projects use `.github/workflows/reusable-*.yml`:
- `reusable-python-ci.yml`
- `reusable-ts-ci.yml`
- `reusable-release.yml`

## Benefits

1. **Consistency**: All projects use same standards
2. **Maintenance**: Update once, apply everywhere
3. **Speed**: Shared configs = faster setup
4. **Quality**: Centralized best practices

## Next Steps

1. Run `dedupe_tools.py` to analyze current state
2. Review recommendations in `tool-deduplication.json`
3. Execute `consolidate_tools.py` to apply changes
4. Test one project per org before full rollout
5. Update project READMEs to reference global tools

## Rollback

All original configs backed up as `.backup_*` files.
To rollback: `find . -name '.backup_*' -exec sh -c 'mv "$1" "${1#.backup_}"' _ {} \;`
