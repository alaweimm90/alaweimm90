# Repository Consolidation - Phase 1: AI System Unification

**Status**: Ready to Execute
**Estimated Time**: 30-45 minutes
**Risk Level**: Low (no breaking changes)

---

## Objective

Merge three separate AI systems into a unified `.ai/` directory structure.

**Current State**:

- `.ai/` - Superprompts, tool configs (14 tools)
- `automation/` - Python automation, agents, workflows
- `tools/ai/` - TypeScript AI utilities, CLI tools

**Target State**:

- `.ai/` - Master AI orchestration hub
- `automation/` - Keep only Python execution scripts
- `tools/ai/` - Keep only TypeScript APIs

---

## Pre-Flight Checklist

- [ ] Commit all current changes
- [ ] Create backup branch: `git checkout -b backup/pre-consolidation`
- [ ] Return to main branch: `git checkout optimization/phase-1-foundation`
- [ ] Verify no uncommitted changes: `git status`

---

## Step 1: Move Automation Assets to .ai/

### 1.1 Move Agents

```bash
# Create target directory
mkdir -p .ai/agents

# Move agent configurations
mv automation/agents/config .ai/agents/
mv automation/agents/templates .ai/agents/

# Keep governance agents separate (Phase 3)
# automation/agents/governance/ stays for now
```

### 1.2 Move Workflows

```bash
# Create target directory
mkdir -p .ai/workflows

# Move workflow configurations
mv automation/workflows/config .ai/workflows/
mv automation/workflows/templates .ai/workflows/
```

### 1.3 Move Orchestration

```bash
# Create target directory
mkdir -p .ai/orchestration

# Move orchestration configs
mv automation/orchestration/config .ai/orchestration/
mv automation/orchestration/patterns .ai/orchestration/
```

### 1.4 Consolidate Prompts

```bash
# Merge automation prompts into .ai/prompts
cp -r automation/prompts/* .ai/prompts/

# Create catalog
cat > .ai/prompts/CATALOG.md << 'EOF'
# AI Prompts Catalog

Consolidated from automation/prompts/ and .ai/prompts/

## Categories
- system/ - System orchestrator prompts
- project/ - Project-specific superprompts
- tasks/ - Reusable task prompts

See automation/prompts/CATALOG.md for detailed index.
EOF
```

---

## Step 2: Consolidate Tools/AI Scripts

### 2.1 Move Scripts to .ai/

```bash
# Move AI scripts
mkdir -p .ai/scripts
mv tools/ai/scripts/* .ai/scripts/

# Keep TypeScript APIs in tools/ai/
# tools/ai/*.ts stays (orchestrator.ts, cache.ts, etc.)
```

### 2.2 Update Import Paths

```bash
# Update references in tools/ai/ TypeScript files
# Change: ../scripts/ → ../../.ai/scripts/
```

---

## Step 3: Update Documentation

### 3.1 Update .ai/README.md

```bash
# Add new sections:
# - agents/ - Agent definitions and registry
# - workflows/ - Workflow configurations
# - orchestration/ - Orchestration patterns
```

### 3.2 Update automation/README.md

```bash
# Update to reflect new structure
# Note: Core assets moved to .ai/
# Execution scripts remain here
```

---

## Step 4: Update Configuration Files

### 4.1 Update .ai/context.yaml

```yaml
# Add new paths
agents:
  registry: .ai/agents/config/agents.yaml
  templates: .ai/agents/templates/

workflows:
  registry: .ai/workflows/config/workflows.yaml
  templates: .ai/workflows/templates/

orchestration:
  config: .ai/orchestration/config/orchestration.yaml
  patterns: .ai/orchestration/patterns/
```

### 4.2 Update package.json scripts

```json
{
  "scripts": {
    "ai:list": "python .ai/prompt-engine/engine.py list",
    "ai:select": "python .ai/prompt-engine/engine.py select",
    "ai:agents": "cat .ai/agents/config/agents.yaml",
    "ai:workflows": "cat .ai/workflows/config/workflows.yaml"
  }
}
```

---

## Step 5: Cleanup

### 5.1 Remove Empty Directories

```bash
# Remove now-empty directories
rmdir automation/agents/config automation/agents/templates
rmdir automation/workflows/config automation/workflows/templates
rmdir automation/orchestration/config automation/orchestration/patterns
rmdir tools/ai/scripts
```

### 5.2 Update .gitignore

```bash
# Add to .gitignore if needed
.ai/learning/data/
.ai/task-history.json
```

---

## Step 6: Validation

### 6.1 Verify Structure

```bash
# Check .ai/ structure
tree .ai/ -L 2

# Expected output:
# .ai/
# ├── agents/
# │   ├── config/
# │   └── templates/
# ├── workflows/
# │   ├── config/
# │   └── templates/
# ├── orchestration/
# │   ├── config/
# │   └── patterns/
# ├── prompts/
# ├── scripts/
# └── superprompts/
```

### 6.2 Test Functionality

```bash
# Test prompt engine
python .ai/prompt-engine/engine.py list

# Test agent registry
cat .ai/agents/config/agents.yaml

# Test workflow registry
cat .ai/workflows/config/workflows.yaml
```

### 6.3 Run Tests

```bash
# Run existing tests
npm test

# Check for broken imports
npm run type-check
```

---

## Step 7: Commit Changes

```bash
# Stage changes
git add .ai/ automation/ tools/ai/ package.json

# Commit
git commit -m "refactor: Phase 1 - Consolidate AI systems into .ai/

- Move agents/ from automation/ to .ai/agents/
- Move workflows/ from automation/ to .ai/workflows/
- Move orchestration/ from automation/ to .ai/orchestration/
- Consolidate prompts into .ai/prompts/
- Move AI scripts from tools/ai/scripts/ to .ai/scripts/
- Update .ai/context.yaml with new paths
- Update package.json scripts for new structure
- Update documentation to reflect consolidation

Phase 1 of 4-phase repository consolidation.
Reduces AI system fragmentation from 3 locations to 1.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Rollback Plan

If issues arise:

```bash
# Restore from backup branch
git checkout backup/pre-consolidation

# Or revert commit
git revert HEAD
```

---

## Success Criteria

- [ ] All agents accessible via `.ai/agents/`
- [ ] All workflows accessible via `.ai/workflows/`
- [ ] All orchestration configs in `.ai/orchestration/`
- [ ] Prompt engine still functional
- [ ] No broken imports in TypeScript
- [ ] All tests passing
- [ ] Documentation updated

---

## Next Steps

After Phase 1 completion:

- **Phase 2**: ATLAS Consolidation
- **Phase 3**: Governance Merge
- **Phase 4**: Documentation Restructure

---

## Notes

- Keep `automation/agents/governance/` for Phase 3
- Keep Python execution scripts in `automation/`
- Keep TypeScript APIs in `tools/ai/`
- This phase focuses on configuration/asset consolidation only
