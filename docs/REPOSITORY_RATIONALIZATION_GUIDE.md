# Repository Rationalization Guide

## What is Repository Rationalization?

The systematic process of reducing repository complexity by:
1. **Consolidating** scattered functionality into cohesive modules
2. **Compiling** frequently-used code into distributable CLIs
3. **Archiving** source code with full traceability
4. **Minimizing** active file count while preserving all work

## Common Problems This Solves

### Repository Sprawl Symptoms
- ❌ 1000s of small, scattered files
- ❌ Duplicated code across projects
- ❌ Hard to find anything
- ❌ Unclear module boundaries
- ❌ "Dead" experiments cluttering active work
- ❌ Import/dependency chaos

### After Rationalization
- ✅ Clean, navigable structure
- ✅ Modular, reusable components
- ✅ Fast CLI tools for common tasks
- ✅ Archived experiments with full provenance
- ✅ Clear ownership and boundaries
- ✅ Maintainable codebase

## The CLI + Archive Pattern

### Concept
Convert complex source trees into:
1. **Compiled CLIs** - Fast, self-contained executables
2. **Source Archive** - Original code preserved with metadata
3. **Decompilation Index** - Map from CLI → source location

### Example Structure

```
organizations/AlaweinOS/
├── bin/                          # Active CLIs
│   ├── mezan-optimizer           # Compiled from MEZAN/
│   ├── atlas-research            # Compiled from ATLAS/
│   └── qap-solver                # Compiled from Libria/
│
├── .archive/                     # Source preservation
│   ├── mezan-optimizer-src/
│   │   ├── MANIFEST.json         # Build metadata
│   │   ├── COMPILE_INFO.md       # How to rebuild
│   │   └── src/                  # Original source
│   ├── atlas-research-src/
│   └── qap-solver-src/
│
└── docs/
    └── archive-index.md          # CLI → source mapping
```

## Rationalization Process

### Phase 1: Analysis

**Goal:** Identify consolidation opportunities

```bash
# Analyze codebase
python .metaHub/scripts/analyze_repository.py \
  --detect-duplication \
  --find-unused-files \
  --identify-modules

# Output:
# - Duplication report
# - Modularization suggestions
# - Archive candidates
```

**Key Questions:**
- Which code is actively used?
- What can be packaged as a CLI?
- What experiments are complete (archivable)?
- Where's the duplication?

### Phase 2: Modularization

**Goal:** Create clean, reusable modules

**Checklist:**
- [ ] Extract core logic into library modules
- [ ] Remove duplication (DRY principle)
- [ ] Define clear interfaces
- [ ] Write comprehensive tests
- [ ] Document APIs

**Example:**
```python
# Before: Scattered across 20 files
feature-a/util1.py
feature-a/helper.py
feature-b/util1.py (duplicate!)
experiments/old-util.py (abandoned)

# After: Consolidated module
src/core/utilities.py  # Single source of truth
```

### Phase 3: CLI Creation

**Goal:** Package modules as distributable tools

**Tools:**
- **Python:** `PyInstaller`, `cx_Freeze`, `Nuitka`
- **Node.js:** `pkg`, `nexe`, `esbuild`
- **Go:** Native compilation
- **Rust:** Native compilation

**Example (Python CLI):**
```bash
# Build standalone executable
pyinstaller --onefile \
  --name mezan-optimizer \
  --add-data "config:config" \
  src/mezan/cli.py

# Result: Single executable
dist/mezan-optimizer
```

### Phase 4: Source Archival

**Goal:** Preserve all source code with full traceability

**Archive Manifest Template:**
```json
{
  "cli_name": "mezan-optimizer",
  "version": "1.0.0",
  "compiled_date": "2025-01-15",
  "source_location": ".archive/mezan-optimizer-src/",
  "git_commit": "abc123def456",
  "build_command": "pyinstaller --onefile src/mezan/cli.py",
  "dependencies": ["requirements.txt"],
  "decompile_instructions": "See COMPILE_INFO.md",
  "original_path": "organizations/AlaweinOS/MEZAN/",
  "maintainer": "alaweimm90",
  "documentation": "docs/mezan/README.md"
}
```

### Phase 5: Documentation

**Goal:** Ensure nothing is lost

**Required Docs:**
1. **Archive Index** - Master map of all archived code
2. **Build Instructions** - How to rebuild from source
3. **CLI Usage** - How to use the compiled tools
4. **Migration Guide** - What changed and where things moved

## MCP-Orchestrated Workflow

### Using MCP to Coordinate Researchers & Engineers

```yaml
# .metaHub/workflows/rationalization.yaml
name: Repository Rationalization
orchestrator: mcp

phases:
  - name: research_analysis
    owner: researchers
    tasks:
      - Identify completed experiments
      - Extract key learnings
      - Document findings
      - Mark for archival

  - name: engineering_consolidation
    owner: engineers
    tasks:
      - Refactor into modules
      - Remove duplication
      - Build CLIs
      - Set up archive structure

  - name: validation
    owner: both
    tasks:
      - Verify no functionality lost
      - Test CLIs work correctly
      - Confirm archive completeness
      - Update documentation

coordination:
  - Researchers flag what's archivable
  - Engineers build the infrastructure
  - Joint validation ensures quality
```

### MCP Commands for Orchestration

```bash
# Start rationalization initiative
mcp-orchestrator start-project \
  --project repository-rationalization \
  --assign-researchers 3 \
  --assign-engineers 2

# Researchers: Identify archive candidates
mcp task create \
  --phase research_analysis \
  --title "Identify completed experiments in AlaweinOS" \
  --assignee researcher-1

# Engineers: Build consolidation infrastructure
mcp task create \
  --phase engineering_consolidation \
  --title "Create archive structure and CLI build pipeline" \
  --assignee engineer-1

# Track progress
mcp dashboard show \
  --project repository-rationalization

# Coordinate handoffs
mcp handoff create \
  --from researcher-1 \
  --to engineer-1 \
  --artifact "experiments-archive-list.json"
```

## Best Practices

### DO:
✅ **Preserve Git History** - Keep full version control in archive
✅ **Document Everything** - Future you will thank present you
✅ **Test Before Archiving** - Ensure CLIs work correctly
✅ **Maintain Traceability** - Always know where code came from
✅ **Version CLIs** - Semantic versioning for compiled tools
✅ **Automate Builds** - CI/CD pipeline for CLI generation

### DON'T:
❌ **Delete Without Archiving** - You'll need it later
❌ **Lose Build Instructions** - Must be able to rebuild
❌ **Break Existing Workflows** - Gradual migration
❌ **Skip Documentation** - Undocumented = lost
❌ **Forget Dependencies** - Archive requirements.txt too
❌ **Rush the Process** - Careful planning prevents issues

## Metrics for Success

### Before vs After

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Active Files | 5,000 | 500 | ↓ 90% |
| Duplicated Code | 35% | 5% | ↓ 86% |
| Navigation Time | 15 min | 2 min | ↓ 87% |
| Build Time | 10 min | 1 min | ↓ 90% |
| Onboarding Time | 2 weeks | 2 days | ↓ 86% |

### Quality Gates

- [ ] Zero functionality lost
- [ ] All source archived with manifest
- [ ] CLIs tested and documented
- [ ] Archive index complete
- [ ] Team can rebuild from source
- [ ] Documentation updated

## Tools & Scripts

### Repository Analysis
```bash
# Find duplication
python .metaHub/scripts/find_duplication.py

# Identify unused files
python .metaHub/scripts/find_unused.py

# Module dependency graph
python .metaHub/scripts/dependency_graph.py
```

### CLI Building
```bash
# Build all CLIs
python .metaHub/scripts/build_all_clis.py

# Test CLIs
python .metaHub/scripts/test_clis.py

# Generate manifests
python .metaHub/scripts/generate_manifests.py
```

### Archive Management
```bash
# Create archive structure
python .metaHub/scripts/create_archive.py

# Validate archive completeness
python .metaHub/scripts/validate_archive.py

# Generate archive index
python .metaHub/scripts/generate_archive_index.py
```

## Example: Consolidating MEZAN

### Current State (Bloated)
```
MEZAN/
├── atlas-core/ (200 files)
├── libria-qap/ (150 files)
├── libria-flow/ (120 files)
├── experiments/ (500 files)
└── prototypes/ (300 files)
Total: 1,270 files
```

### Target State (Rationalized)
```
bin/
├── atlas-cli           # Compiled from atlas-core
├── qap-solver          # Compiled from libria-qap
└── flow-optimizer      # Compiled from libria-flow

.archive/
├── atlas-src/          # Original 200 files
├── libria-qap-src/     # Original 150 files
├── libria-flow-src/    # Original 120 files
├── experiments/        # All 500 preserved
└── prototypes/         # All 300 preserved

Total Active: 3 CLIs + docs (fast, clean)
Total Archived: All original work (preserved)
```

### Impact
- **Active files:** 1,270 → ~50 (↓ 96%)
- **Functionality:** 100% preserved
- **Speed:** 10x faster navigation
- **Maintainability:** ↑ 90%

## References

- [Root Governance](../GOVERNANCE.md)
- [Archive Index](./archive-index.md)
- [CLI Build Pipeline](./.github/workflows/build-clis.yml)
- [MCP Orchestration Guide](./MCP_ORCHESTRATION.md)

---

**When in doubt, archive—don't delete.**
