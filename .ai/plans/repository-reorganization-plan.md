# Repository Reorganization & Governance Enforcement Plan

**Created:** 2025-11-30
**Status:** Ready for Execution
**Mode:** Agentic Multi-Phase Implementation

---

## Executive Summary

This plan addresses critical organizational drift in the meta-governance repository through a systematic, multi-phase approach leveraging the existing ORCHEX orchestration framework and automation systems.

### Key Problems Identified

1. **Root Directory Clutter:** 20+ scattered files including temp files, malformed directories
2. **Duplicate Code:** Config/FS utilities duplicated across tools/
3. **Inconsistent Module Systems:** ESModules vs CommonJS mismatch
4. **Python Script Sprawl:** 23 unorganized scripts in .metaHub/scripts/
5. **Missing Documentation:** 40% of tool directories lack README files
6. **AI Tool Architecture Confusion:** Three overlapping systems (ai/, ai-suite/, ai-orchestration/)
7. **Governance Drift:** Policies exist but enforcement is incomplete

---

## Phase 1: Immediate Cleanup (Critical)

### 1.1 Remove Stale/Malformed Files

```powershell
# Files to delete
- temp_config.js
- temp_config_complete.js
- nul (garbage file)
- -p/ (malformed directory)
```

### 1.2 Fix Malformed Directory Names

```powershell
# Rename malformed directory
backup-recovery}/ → backup-recovery/
```

### 1.3 Archive Legacy Reports

Move to `.archive/reports/2025-11/`:

- KILO-ACTION-PLAN.md
- KILO-AUDIT-REPORT.md
- KILO-EXECUTION-SUMMARY.md
- KILO-FINAL-REPORT.md
- KILO-CHANGES.md
- KILO-QUICK-START.md
- AUDIT_SUMMARY.md
- COMMIT_PLAN.md
- TODO-REPORT.txt

---

## Phase 2: Code Consolidation

### 2.1 Deduplicate Utility Files

**Action:** Consolidate duplicate utilities into `tools/lib/`

| Source                   | Destination           | Action                |
| ------------------------ | --------------------- | --------------------- |
| `tools/devops/config.ts` | `tools/lib/config.ts` | Merge & Delete source |
| `tools/devops/fs.ts`     | `tools/lib/fs.ts`     | Merge & Delete source |

### 2.2 Unify AI Tool Architecture

**Current State:**

```
tools/
├── ai/              # Primary CLI tools
├── ai-suite/        # Test integration
├── ai-orchestration/# MCP orchestration
├── ai-docs/         # Documentation tools
```

**Target State:**

```
tools/ai/
├── cli/             # CLI commands (from ai/)
├── orchestration/   # Merged from ai-orchestration/
├── mcp/             # MCP server (from ai-orchestration/mcp/)
├── suite/           # Test integration (from ai-suite/)
├── docs/            # Documentation tools (from ai-docs/)
└── README.md        # Unified documentation
```

### 2.3 Standardize Module Systems

**Action:** Convert `automation-ts` from CommonJS to ESModules

```json
// automation-ts/tsconfig.json changes
{
  "module": "ESNext", // Was: commonjs
  "moduleResolution": "bundler",
  "type": "module"
}
```

**Action:** Unify YAML library (use `yaml` not `js-yaml`)

---

## Phase 3: Python Reorganization

### 3.1 Restructure .metaHub/scripts/

**Current:** 23 flat Python files

**Target Structure:**

```
.metaHub/scripts/
├── __init__.py
├── orchestration/
│   ├── __init__.py
│   ├── checkpoint.py        # from orchestration_checkpoint.py
│   ├── telemetry.py         # from orchestration_telemetry.py
│   └── validator.py         # from compliance_validator.py (orchestration parts)
├── ai/
│   ├── __init__.py
│   ├── audit.py             # from ai_audit.py
│   ├── hallucination.py     # from hallucination_verifier.py
│   └── integrator.py        # from agent_mcp_integrator.py
├── compliance/
│   ├── __init__.py
│   ├── validator.py         # from compliance_validator.py
│   └── enforcement.py       # from enforcement.py
├── setup/
│   ├── __init__.py
│   ├── org.py               # from setup_org.py
│   └── repo_ci.py           # from setup_repo_ci.py
├── integration/
│   ├── __init__.py
│   ├── mcp_wrapper.py       # from mcp_cli_wrapper.py
│   └── mcp_tester.py        # from mcp_server_tester.py
└── utils/
    ├── __init__.py
    ├── catalog.py           # from catalog.py
    └── meta.py              # from meta.py
```

---

## Phase 4: Documentation Compliance

### 4.1 Add Missing README Files

Create README.md files for:

| Directory        | Purpose                      |
| ---------------- | ---------------------------- |
| `tools/ai/`      | AI orchestration CLI tools   |
| `tools/devops/`  | DevOps utilities             |
| `tools/lib/`     | Shared utility library       |
| `tools/scripts/` | Build and automation scripts |
| `tests/`         | Test suite organization      |
| `src/`           | Service implementations      |

### 4.2 Create Architecture Documentation

Create `docs/TOOL_ARCHITECTURE.md`:

- Tool dependency graph
- Module system documentation
- Integration points
- Configuration reference

---

## Phase 5: Governance Enforcement

### 5.1 Update Root Structure Policy

Update `.metaHub/policies/root-structure.yaml`:

```yaml
allowed_root_items:
  directories:
    - .ai
    - .orchex
    - .github
    - .husky
    - .metaHub
    - .vscode
    - automation
    - automation-ts
    - docs
    - enterprise
    - examples
    - infrastructure
    - k8s
    - organizations
    - src
    - tests
    - tools
    - node_modules # auto-generated
    - dist # build output

  files:
    - package.json
    - package-lock.json
    - tsconfig.json
    - vitest.config.ts
    - eslint.config.js
    - mkdocs.yaml
    - README.md
    - CLAUDE.md
    - LICENSE
    - SECURITY.md
    - GOVERNANCE.md
    - CONTRIBUTING.md
    - .gitignore
    - .gitattributes
    - .dockerignore
    - .prettierrc
    - .prettierignore
    - .eslintignore
    - .pre-commit-config.yaml
    - .yamllint.yaml
    - .env.example

forbidden_patterns:
  - 'temp_*.js'
  - '*.tmp'
  - 'nul'
  - '*}/' # malformed directories
```

### 5.2 Create Automated Enforcement Workflow

Create `.github/workflows/structure-validation.yml`:

- Runs on PR and push to main
- Validates root structure against policy
- Checks for README in tool directories
- Validates module system consistency

### 5.3 Implement Pre-commit Hooks

Add to `.pre-commit-config.yaml`:

- Root structure validation
- README presence check
- Module system consistency check

---

## Phase 6: Workflow Integration

### 6.1 Connect ORCHEX to Governance

Update `tools/orchex/orchestration/` to:

- Check governance compliance before task execution
- Route refactoring tasks through compliance validator
- Log governance violations to `.orchex/evidence/`

### 6.2 Create Reorganization Workflow

Add to `automation/workflows/config/workflows.yaml`:

```yaml
repository_reorganization:
  name: 'Repository Reorganization'
  description: 'Systematic repository cleanup and reorganization'
  pattern: prompt_chaining
  steps:
    - cleanup_stale_files
    - consolidate_duplicates
    - reorganize_python
    - add_documentation
    - enforce_governance
  agents:
    - devops_agent
    - compliance_agent
    - technical_writer_agent
```

---

## Execution Strategy

### Multi-Agent Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (Claude Code)                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Cleanup Agent                                      │
│  ├── Delete stale files                                      │
│  ├── Fix directory names                                     │
│  └── Archive legacy reports                                  │
│                                                              │
│  Phase 2: Consolidation Agent                                │
│  ├── Merge duplicate utilities                               │
│  ├── Unify AI tool architecture                              │
│  └── Standardize module systems                              │
│                                                              │
│  Phase 3: Python Reorganization Agent                        │
│  ├── Create module structure                                 │
│  ├── Move and rename files                                   │
│  └── Update imports                                          │
│                                                              │
│  Phase 4: Documentation Agent                                │
│  ├── Generate README files                                   │
│  └── Create architecture docs                                │
│                                                              │
│  Phase 5: Governance Agent                                   │
│  ├── Update policies                                         │
│  ├── Create enforcement workflows                            │
│  └── Add pre-commit hooks                                    │
│                                                              │
│  Phase 6: Integration Agent                                  │
│  ├── Connect ORCHEX to governance                             │
│  └── Create reorganization workflow                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Parallel Execution Opportunities

- Phase 1 tasks can run in parallel
- Phase 4 README generation can run in parallel
- Phase 5.1 and 5.2 can run in parallel

### Checkpoints

After each phase:

1. Run `npm run lint` to validate TypeScript
2. Run `npm test` to verify no regressions
3. Commit changes with descriptive message
4. Update `.ai/task-history.json`

---

## Success Metrics

| Metric                     | Before  | Target          |
| -------------------------- | ------- | --------------- |
| Root directory items       | 50+     | 35              |
| Duplicate utility files    | 4+      | 0               |
| Directories without README | 8       | 0               |
| Malformed directories      | 3       | 0               |
| Stale temp files           | 3       | 0               |
| Python scripts (flat)      | 23      | 0 (modularized) |
| Governance enforcement     | Partial | Automated       |

---

## Risk Mitigation

1. **Breaking Changes:** All file moves preserve git history with `git mv`
2. **Import Breaks:** Update all imports after file consolidation
3. **Test Failures:** Run full test suite after each phase
4. **Rollback:** Create git tag before each phase for easy rollback

---

## Estimated Scope

- **Files to delete:** 6
- **Files to move/archive:** 15+
- **Files to consolidate:** 4
- **README files to create:** 6
- **Python modules to create:** 6
- **Workflow files to create/update:** 3
- **Policy files to update:** 2
