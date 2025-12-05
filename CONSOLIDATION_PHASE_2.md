# Repository Consolidation - Phase 2: ATLAS Consolidation

**Status**: Ready to Execute
**Estimated Time**: 20-30 minutes
**Risk Level**: Low

---

## Objective

Clarify separation between ATLAS implementation and runtime data.

**Current State**:

- `tools/atlas/` - Core implementation (analysis, orchestration, CLI)
- `.atlas/` - Mixed: some config, some runtime data

**Target State**:

- `tools/atlas/` - All implementation code (keep as-is)
- `.atlas/` - Runtime data ONLY (reports, metrics, evidence)

---

## Analysis

### What's in tools/atlas/ (Implementation)

```
tools/atlas/
├── adapters/        # LLM adapters (Anthropic, OpenAI, Google)
├── agents/          # Agent registry and teams
├── analysis/        # Code analysis, AST parsing, complexity
├── api/             # REST API server
├── cli/             # Command-line interface
├── config/          # Configuration loader
├── core/            # Kilo bridge integration
├── integrations/    # AI integrations
├── orchestration/   # Workflow orchestration
├── refactoring/     # Refactoring engine
├── services/        # Background services
├── storage/         # Database backends
├── types/           # TypeScript types
└── utils/           # Utilities
```

### What's in .atlas/ (Runtime Data)

```
.atlas/
├── demo-web/        # Demo artifacts (DELETE or move to demo/)
├── demo-web2/       # Demo artifacts (DELETE or move to demo/)
├── evidence/        # Runtime evidence (KEEP)
├── reports/         # Generated reports (KEEP)
├── agent-registry.json   # Runtime registry (KEEP)
├── circuit.json     # Circuit breaker state (KEEP)
└── metrics.json     # Runtime metrics (KEEP)
```

---

## Actions

### Step 1: Clean Demo Artifacts

```bash
# Option A: Delete demos (if not needed)
rm -rf .atlas/demo-web .atlas/demo-web2

# Option B: Move to demo/ (if needed for reference)
mv .atlas/demo-web demo/atlas-demo-web
mv .atlas/demo-web2 demo/atlas-demo-web2
```

### Step 2: Update .gitignore

```bash
# Add to .gitignore
echo "" >> .gitignore
echo "# ATLAS runtime data" >> .gitignore
echo ".atlas/*.json" >> .gitignore
echo ".atlas/reports/" >> .gitignore
echo ".atlas/evidence/" >> .gitignore
echo "!.atlas/.gitkeep" >> .gitignore
```

### Step 3: Add .gitkeep to .atlas/

```bash
# Ensure .atlas/ directory is tracked but contents are ignored
touch .atlas/.gitkeep
```

### Step 4: Update Documentation

```bash
# Update tools/atlas/README.md to clarify:
# - tools/atlas/ = implementation
# - .atlas/ = runtime data (gitignored)
```

---

## Validation

### Verify Structure

```bash
# Check tools/atlas/ has all implementation
ls tools/atlas/

# Check .atlas/ has only runtime data
ls .atlas/
```

### Test ATLAS CLI

```bash
# Verify ATLAS still works
npm run atlas -- --help
```

---

## Commit

```bash
git add .atlas/ .gitignore tools/atlas/README.md
git commit -m "refactor: Phase 2 - Clarify ATLAS implementation vs runtime data

- Remove/move demo artifacts from .atlas/
- Update .gitignore to exclude .atlas/ runtime data
- Add .gitkeep to preserve .atlas/ directory
- Update documentation to clarify separation

Phase 2 of 4-phase repository consolidation."
```

---

## Success Criteria

- [ ] `.atlas/` contains only runtime data
- [ ] Demo artifacts removed or moved to `demo/`
- [ ] `.gitignore` updated to exclude `.atlas/*.json`
- [ ] `.atlas/.gitkeep` added
- [ ] Documentation updated
- [ ] ATLAS CLI still functional

---

## Next: Phase 3

After Phase 2:

- **Phase 3**: Governance Merge (`.metaHub/` + `automation/governance/`)
- **Phase 4**: Documentation Restructure
