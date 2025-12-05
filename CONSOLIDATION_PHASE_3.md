# Repository Consolidation - Phase 3: Governance Merge

**Status**: Ready to Execute
**Estimated Time**: 15-20 minutes
**Risk Level**: Low

---

## Objective

Merge governance automation into `.metaHub/` for unified governance system.

**Current State**:

- `.metaHub/` - Policies, schemas, enforcement scripts
- `automation/governance/` - Policy automation scripts

**Target State**:

- `.metaHub/` - All governance (policies + automation)
- `automation/` - Keep only execution scripts (no governance)

---

## Analysis

### What's in .metaHub/

```
.metaHub/
├── policies/          # Policy definitions
├── schemas/           # JSON schemas
├── scripts/           # Enforcement scripts
├── templates/         # Templates
└── tools/             # Governance tools
```

### What's in automation/governance/

```
automation/governance/
├── policies/          # DUPLICATE - technical debt policies
└── (scripts)          # Policy automation
```

---

## Actions

### Step 1: Move Governance Scripts

```bash
# Move governance automation to .metaHub/
mkdir -p .metaHub/automation
mv automation/agents/governance .metaHub/automation/agents
```

### Step 2: Consolidate Policies

```bash
# Check for duplicate policies
# Keep .metaHub/policies/ as source of truth
# Remove automation/governance/policies/ if duplicate
```

### Step 3: Update References

```bash
# Update any scripts referencing automation/governance/
# Point to .metaHub/ instead
```

---

## Validation

### Verify Structure

```bash
dir .metaHub\automation
dir automation\governance
```

### Test Governance

```bash
python .metaHub/scripts/enforce.py
```

---

## Commit

```bash
git add .metaHub/ automation/
git commit -m "refactor: Phase 3 - Consolidate governance into .metaHub/

- Move governance agents to .metaHub/automation/agents/
- Consolidate policy definitions in .metaHub/policies/
- Update references to point to .metaHub/

Phase 3 of 4-phase repository consolidation."
```

---

## Success Criteria

- [ ] Governance agents in `.metaHub/automation/agents/`
- [ ] No duplicate policies
- [ ] All references updated
- [ ] Governance scripts functional

---

## Next: Phase 4

After Phase 3:

- **Phase 4**: Documentation Restructure
