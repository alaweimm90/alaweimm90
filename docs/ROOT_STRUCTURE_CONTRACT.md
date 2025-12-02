# Root Directory Structure Contract

> **Version:** 2.1.0
> **Status:** Active
> **Last Updated:** 2025-11-27
> **Migration Status:** Complete
> **Supersedes:** ADR-002-root-structure-policy.md (to be updated)

---

## 1. Overview and Goals

This document defines the **authoritative contract** for the root directory structure of the `alaweimm90` portfolio governance repository. It establishes:

1. **What MUST exist** at root (required items)
2. **What MAY exist** at root (optional items with governance)
3. **What MUST NOT exist** at root (forbidden items)
4. **How to change** the root structure (change-control process)
5. **How violations are detected** and remediated (enforcement integration)

### Design Principles

| Principle            | Description                                         |
| -------------------- | --------------------------------------------------- |
| **Schema-First**     | All rules encoded in machine-readable YAML          |
| **Idempotent**       | Enforcement produces same result on repeated runs   |
| **Adapter-Agnostic** | Works with any CI system, not just GitHub Actions   |
| **Enforced via CI**  | Automated checks block non-compliant changes        |
| **AI-Auditable**     | Structure supports AI governance tools (KILO, etc.) |

### Governance Philosophy

The root directory is the **first impression** of the repository. It MUST:

- Be navigable by humans in < 10 seconds
- Contain only governance-critical items
- Hide implementation details in subdirectories
- Follow the same standards we enforce on downstream repos

---

## 2. Standardized Root Hierarchy

### 2.1 Canonical Root Layout

```text
/
├── .git/                       # [IMPLICIT] VCS metadata
│
├── .github/                    # [REQUIRED] GitHub CI, workflows, policies
├── .metaHub/                   # [REQUIRED] Governance engine
├── docs/                       # [REQUIRED] Documentation (MkDocs)
│
├── tests/                      # [RECOMMENDED] Governance tests
├── .vscode/                    # [OPTIONAL] Editor config (team-wide)
│
├── .allstar/                   # [TOOLING] Security policy config
├── .husky/                     # [TOOLING] Git hooks
│
├── .ai/                        # [AI-TOOLING] Consolidated AI configs
│   ├── claude/                 # Claude Code config
│   ├── cursor/                 # Cursor IDE config
│   ├── cline/                  # Cline config
│   ├── aider/                  # Aider config
│   ├── ... (other tools)
│   └── settings.yaml           # Unified settings
│
├── README.md                   # [REQUIRED] Repository overview
├── LICENSE                     # [REQUIRED] MIT License
├── SECURITY.md                 # [REQUIRED] Security policy
├── GOVERNANCE.md               # [REQUIRED] Governance overview
├── CONTRIBUTING.md             # [RECOMMENDED] Contribution guide
│
├── .gitignore                  # [REQUIRED] Git ignore rules
├── .gitattributes              # [RECOMMENDED] Git attributes
├── .pre-commit-config.yaml     # [RECOMMENDED] Pre-commit hooks
├── mkdocs.yml                  # [RECOMMENDED] Docs site config
│
├── .ai-settings.yaml           # [AI] Unified AI config (legacy, migrate to .ai/)
├── .ai-context.md              # [AI] Shared AI context
│
└── organizations/              # [GITIGNORED] Local org templates
```

### 2.2 Root Entry Classification Table

| Entry                     | Type | Status          | Owner    | Rationale                      |
| ------------------------- | ---- | --------------- | -------- | ------------------------------ |
| `.github/`                | dir  | **REQUIRED**    | platform | CI/CD, workflows, CODEOWNERS   |
| `.metaHub/`               | dir  | **REQUIRED**    | platform | Core governance engine         |
| `docs/`                   | dir  | **REQUIRED**    | docs     | MkDocs documentation           |
| `README.md`               | file | **REQUIRED**    | platform | Repository overview            |
| `LICENSE`                 | file | **REQUIRED**    | platform | Open source license            |
| `SECURITY.md`             | file | **REQUIRED**    | security | Vulnerability reporting        |
| `GOVERNANCE.md`           | file | **REQUIRED**    | platform | Governance overview            |
| `.gitignore`              | file | **REQUIRED**    | platform | Git ignore rules               |
| `tests/`                  | dir  | **RECOMMENDED** | platform | Governance tests               |
| `CONTRIBUTING.md`         | file | **RECOMMENDED** | platform | Contribution guidelines        |
| `.pre-commit-config.yaml` | file | **RECOMMENDED** | platform | Pre-commit hooks               |
| `mkdocs.yml`              | file | **RECOMMENDED** | docs     | MkDocs configuration           |
| `.gitattributes`          | file | **RECOMMENDED** | platform | Line endings, LFS              |
| `.vscode/`                | dir  | **OPTIONAL**    | platform | Team editor settings           |
| `.allstar/`               | dir  | **OPTIONAL**    | security | GitHub Allstar config          |
| `.husky/`                 | dir  | **OPTIONAL**    | platform | Git hooks manager              |
| `.ai/`                    | dir  | **OPTIONAL**    | platform | Consolidated AI configs        |
| `.ai-settings.yaml`       | file | **DEPRECATED**  | platform | Migrate to `.ai/settings.yaml` |
| `organizations/`          | dir  | **GITIGNORED**  | platform | Local working copies           |
| `*.pdf`                   | file | **FORBIDDEN**   | -        | Move to `docs/reports/`        |
| `*-report.json`           | file | **FORBIDDEN**   | -        | Add to `.gitignore`            |
| `temp-*/`                 | dir  | **FORBIDDEN**   | -        | Temporary, gitignore           |

### 2.3 AI Tooling Consolidation Strategy

**Current State (14 tools scattered at root):**

```text
.claude/, .cursor/, .cline/, .amazonq/, .continue/,
.codex/, .gemini/, .trae/, .blackbox/, .augment/,
.kilocode/, .aider/, .windsurf/, .copilot/
+ .cursorrules, .windsurfrules, .clinerules, .augmentrules,
  .aider.conf.yml, .ai-context.md, .ai-settings.yaml
```

**Target State (consolidated under `.ai/`):**

```text
.ai/
├── settings.yaml           # Unified configuration (master)
├── context.md              # Shared AI context
├── rules/                  # Tool-specific rules files
│   ├── cursor.rules
│   ├── windsurf.rules
│   ├── cline.rules
│   └── augment.rules
├── claude/                 # Claude Code config
├── cursor/                 # Cursor IDE config
├── cline/                  # Cline config
├── aider/                  # Aider config
├── continue/               # Continue.dev config
├── kilocode/               # Kilo Code config
├── amazonq/                # Amazon Q config
├── trae/                   # Trae config
├── blackbox/               # Blackbox config
├── gemini/                 # Gemini config
├── codex/                  # Codex config
├── augment/                # Augment config
├── windsurf/               # Windsurf config
└── copilot/                # Copilot config (instructions only)
```

**Migration Policy:**

| Phase | Action                                         | Timeline  |
| ----- | ---------------------------------------------- | --------- |
| 1     | Create `.ai/` structure                        | Immediate |
| 2     | Symlink legacy locations to `.ai/`             | Week 1    |
| 3     | Update tooling to read from `.ai/`             | Week 2    |
| 4     | Remove legacy locations                        | Week 4    |
| 5     | Update enforcement to forbid scattered configs | Week 5    |

**First-Class Supported Tools:**

| Tool        | Config Committed   | Auto-Approve      | Status       |
| ----------- | ------------------ | ----------------- | ------------ |
| Aider       | Yes                | `yes-always`      | Supported    |
| Cursor      | Yes                | `autoApply`       | Supported    |
| Cline       | Yes                | `bypassApprovals` | Supported    |
| Claude Code | Yes                | CLI flag          | Supported    |
| Windsurf    | Yes                | In-rules          | Supported    |
| Kilo Code   | Yes                | `auto_approve`    | Supported    |
| Continue    | Yes                | Experimental      | Supported    |
| Amazon Q    | Yes                | Limited           | Supported    |
| Blackbox    | Yes                | `yoloMode`        | Supported    |
| Augment     | Yes                | `autoApprove`     | Supported    |
| Gemini      | Yes                | `autoApprove`     | Supported    |
| Codex       | Yes                | `autoApprove`     | Supported    |
| Trae        | Yes                | `autoApprove`     | Supported    |
| Copilot     | Yes (instructions) | N/A               | Context only |

---

## 3. Root-Level Rules and Justifications

### 3.1 Normative Rules

#### MUST (Required)

1. **MUST** have `README.md` at root providing repository overview
2. **MUST** have `LICENSE` file with MIT license text
3. **MUST** have `SECURITY.md` with vulnerability reporting instructions
4. **MUST** have `GOVERNANCE.md` linking to governance documentation
5. **MUST** have `.github/` directory with workflows and policies
6. **MUST** have `.metaHub/` directory containing governance engine
7. **MUST** have `docs/` directory for MkDocs documentation
8. **MUST** have `.gitignore` with patterns for generated files

#### SHOULD (Recommended)

1. **SHOULD** have `CONTRIBUTING.md` with contribution guidelines
2. **SHOULD** have `tests/` directory with governance tests
3. **SHOULD** have `.pre-commit-config.yaml` for automated hooks
4. **SHOULD** have `mkdocs.yml` for documentation site configuration
5. **SHOULD** have `.gitattributes` for line ending normalization
6. **SHOULD** consolidate AI configs under `.ai/` directory

#### MAY (Optional)

1. **MAY** have `.vscode/` for team-wide editor settings
2. **MAY** have `.allstar/` for GitHub Allstar security config
3. **MAY** have `.husky/` for git hooks management
4. **MAY** have individual AI tool configs at root during migration period
5. **MAY** have `pyproject.toml` if governance tools need Python packaging
6. **MAY** have `package.json` if governance tools need Node.js

#### MUST NOT (Forbidden)

1. **MUST NOT** have application source code (`src/`, `app/`, `lib/`)
2. **MUST NOT** have organization monorepos at root (use `organizations/` gitignored)
3. **MUST NOT** have binary files (`.pdf`, `.zip`, `.tar.gz`)
4. **MUST NOT** have generated reports (`*-report.json`, `*-results.json`)
5. **MUST NOT** have temporary directories (`temp-*`, `new-*`, `backup*`)
6. **MUST NOT** have environment files (`.env`, `.env.*`)
7. **MUST NOT** have IDE-specific cache (`.DS_Store`, `Thumbs.db`)
8. **MUST NOT** have workspace files at root (`*.code-workspace`)

### 3.2 File Category Placement

| Category          | Root Location                | Rationale                 |
| ----------------- | ---------------------------- | ------------------------- |
| Governance docs   | Root (`GOVERNANCE.md`, etc.) | First-class visibility    |
| Technical docs    | `docs/`                      | Organized, MkDocs-managed |
| Reports/artifacts | `docs/reports/`              | Keeps root clean          |
| Governance engine | `.metaHub/`                  | Clear ownership           |
| CI/CD configs     | `.github/`                   | GitHub convention         |
| AI tool configs   | `.ai/`                       | Consolidated, governed    |
| IDE configs       | `.vscode/`                   | Standard location         |
| Security configs  | `.allstar/`                  | Security-specific         |

### 3.3 Handling Edge Cases

#### Generated Reports

```yaml
# In .gitignore
ai-audit-report.json
enforcement-report.json
audit-results.json
*-report.json
*-results.json
```

Reports MUST be gitignored. If needed for CI artifacts, use `docs/reports/` with explicit commits.

#### PDF/Binary Deliverables

```text
# WRONG - at root
/reGitHub Governance System Audit.pdf

# CORRECT - in docs
/docs/reports/governance-audit-2025-11.pdf
```

#### Multiple AI Assistants

All AI configs SHOULD migrate to `.ai/` within 4 weeks. During migration:

- Legacy locations are symlinked to `.ai/`
- Tools read from both locations
- Enforcement warns but does not fail

---

## 4. Change-Control Process

### 4.1 Who Can Propose Changes

| Role                 | Permissions                        |
| -------------------- | ---------------------------------- |
| Platform Maintainers | Full access to propose and approve |
| Contributors         | Propose via PR with ADR            |
| Bots/CI              | Cannot propose root changes        |

### 4.2 Where Changes Are Recorded

1. **ADR Update:** Modify [`docs/adr/ADR-002-root-structure-policy.md`](adr/ADR-002-root-structure-policy.md)
2. **Policy Update:** Modify [`.metaHub/policies/root-structure.yaml`](../.metaHub/policies/root-structure.yaml)
3. **Template Update:** Modify [`.metaHub/templates/structures/portfolio-structure.yaml`](../.metaHub/templates/structures/portfolio-structure.yaml)
4. **Contract Update:** Modify this document

### 4.3 Step-by-Step Change Procedure

#### Step 1: Discovery and Motivation

- [ ] Document the concrete use case requiring root structure change
- [ ] Identify which items need to be added/renamed/removed
- [ ] Assess impact on existing tooling and workflows

#### Step 2: Proposal Creation

- [ ] Create branch: `root-structure/<change-name>`
- [ ] Update [`ADR-002-root-structure-policy.md`](adr/ADR-002-root-structure-policy.md) with decision
- [ ] Update [`.metaHub/policies/root-structure.yaml`](../.metaHub/policies/root-structure.yaml):
  - Add to `allowed_files` or `allowed_directories` if adding
  - Add to `forbidden` if removing
  - Update `migration` section if relocating
- [ ] Update [`portfolio-structure.yaml`](../.metaHub/templates/structures/portfolio-structure.yaml) `required_structure` section
- [ ] Update this contract document

#### Step 3: Enforcement Updates

- [ ] Update [`structure_validator.py`](../.metaHub/scripts/structure_validator.py):
  - Modify category lists in `validate_root_structure()`
  - Add/remove patterns as needed
- [ ] Update [`enforce.py`](../.metaHub/scripts/enforce.py) if root-level checks needed
- [ ] Verify changes don't break existing validation

#### Step 4: Test Updates

- [ ] Add/update tests in [`tests/`](../tests/) for new rules
- [ ] Run local validation: `python .metaHub/scripts/structure_validator.py --root`
- [ ] Verify CI passes with changes

#### Step 5: Workflow Updates

- [ ] Update [`weekly-governance-check.yml`](../.github/workflows/weekly-governance-check.yml) if needed
- [ ] Ensure root structure check is included in validation jobs
- [ ] Verify actionable error messages for failures

#### Step 6: Review and Approval

- [ ] Request review from platform maintainers (per [`GOVERNANCE.md`](../GOVERNANCE.md))
- [ ] Address review feedback
- [ ] Optionally request KILO/AI audit confirmation
- [ ] Merge after approval

#### Step 7: Rollout

- [ ] Announce change to contributors
- [ ] Update downstream repos if patterns changed
- [ ] Monitor for enforcement failures

### 4.4 Emergency Changes

For critical security or compliance issues:

1. Platform maintainer may make direct changes to `main`
2. Must create retrospective ADR within 24 hours
3. Must notify other maintainers immediately

---

## 5. Tooling and Enforcement Integration

### 5.1 Enforcement Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    Enforcement Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Pre-commit   │───▶│ PR Validation │───▶│ Weekly Check │  │
│  │ Hook         │    │ (CI)          │    │ (Scheduled)  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            structure_validator.py --root              │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
│                            ▼                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          root-structure.yaml (Policy)                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Enforcement Points Mapping

| Rule                      | Enforcement File                                   | Check Type | Failure Action |
| ------------------------- | -------------------------------------------------- | ---------- | -------------- |
| Required files exist      | `structure_validator.py:validate_root_structure()` | Error      | Block merge    |
| Required dirs exist       | `structure_validator.py:validate_root_structure()` | Error      | Block merge    |
| Forbidden patterns absent | `structure_validator.py:validate_root_structure()` | Warning    | Warn only      |
| AI configs consolidated   | `structure_validator.py:validate_root_structure()` | Warning    | Warn only      |
| Gitignored items absent   | `.gitignore` + CI check                            | Info       | Log only       |

### 5.3 Structure Validator Integration

The [`structure_validator.py`](../.metaHub/scripts/structure_validator.py) `validate_root_structure()` method:

```python
# Current implementation reads from root-structure.yaml
# Categories to parse:
FILE_CATEGORIES = ["required", "recommended", "configuration", "ai_rules"]
DIR_CATEGORIES = ["required", "recommended", "optional", "tooling", "cache", "ai_assistants"]
```

**Proposed Enhancement:**

Add explicit `portfolio_root` section to [`portfolio-structure.yaml`](../.metaHub/templates/structures/portfolio-structure.yaml):

```yaml
portfolio_root:
  description: 'Root directory contract for governance repo'
  required:
    files: ['README.md', 'LICENSE', 'SECURITY.md', 'GOVERNANCE.md', '.gitignore']
    dirs: ['.github/', '.metaHub/', 'docs/']
  recommended:
    files: ['CONTRIBUTING.md', '.pre-commit-config.yaml', 'mkdocs.yml']
    dirs: ['tests/']
  forbidden:
    patterns: ['*.pdf', '*-report.json', 'temp-*']
```

### 5.4 Workflow Integration

**Weekly Governance Check ([`weekly-governance-check.yml`](../.github/workflows/weekly-governance-check.yml)):**

```yaml
jobs:
  root-structure-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Validate Root Structure
        run: |
          python .metaHub/scripts/structure_validator.py --root --report json \
            > root-validation.json
      - name: Check for violations
        run: |
          violations=$(jq '.violations | length' root-validation.json)
          if [ "$violations" -gt 0 ]; then
            echo "::error::Root structure violations found"
            exit 1
          fi
```

### 5.5 KILO / .kilocode Alignment

The [`.kilocode/`](../.kilocode/) configuration MUST:

1. Recognize `.ai/` as the canonical AI config location
2. Not assume AI configs are scattered at root
3. Include root structure rules in governance checks

**Required `.kilocode/config.yaml` update:**

```yaml
governance:
  root_structure:
    policy_file: '.metaHub/policies/root-structure.yaml'
    enforce: true
    severity: warning
```

---

## 6. Examples and Scenarios

### 6.1 Current Repo State Analysis

**Compliant and Expected:**

| Item            | Status       | Notes                  |
| --------------- | ------------ | ---------------------- |
| `README.md`     | ✅ Compliant | Present and documented |
| `LICENSE`       | ✅ Compliant | MIT license            |
| `SECURITY.md`   | ✅ Compliant | Security policy        |
| `GOVERNANCE.md` | ✅ Compliant | Links to governance    |
| `.github/`      | ✅ Compliant | Workflows present      |
| `.metaHub/`     | ✅ Compliant | Governance engine      |
| `docs/`         | ✅ Compliant | MkDocs documentation   |
| `tests/`        | ✅ Compliant | Governance tests       |
| `.gitignore`    | ✅ Compliant | Patterns defined       |

**Deprecated (MIGRATION COMPLETE):**

| Item                | Previous     | Migrated To                | Status  |
| ------------------- | ------------ | -------------------------- | ------- |
| `.claude/`          | Root symlink | `.ai/claude/`              | Removed |
| `.cursor/`          | Root symlink | `.ai/cursor/`              | Removed |
| `.cline/`           | Root symlink | `.ai/cline/`               | Removed |
| (12 more AI dirs)   | Root symlink | `.ai/<tool>/`              | Removed |
| `.ai-settings.yaml` | Root file    | `.ai/settings.yaml`        | Removed |
| `.cursorrules`      | Root file    | `.ai/rules/cursor.rules`   | Removed |
| `.windsurfrules`    | Root file    | `.ai/rules/windsurf.rules` | Removed |
| `.clinerules`       | Root file    | `.ai/rules/cline.rules`    | Removed |
| `.augmentrules`     | Root file    | `.ai/rules/augment.rules`  | Removed |
| `.ai-context.md`    | Root file    | `.ai/context.md`           | Removed |
| `.aider.conf.yml`   | Root file    | `.ai/aider/config.yml`     | Removed |

**Forbidden (gitignored, local-only):**

| Item               | Issue                  | Resolution |
| ------------------ | ---------------------- | ---------- |
| `*-report.json`    | Generated file at root | Gitignored |
| `*.code-workspace` | Workspace at root      | Gitignored |
| `alaweimm90/`      | Local clone directory  | Gitignored |
| `new-repos/`       | Staging directory      | Gitignored |
| `.env`             | Environment file       | Gitignored |

### 6.2 Scenario: Adding a New AI Tool

**Use Case:** Add support for "SuperCoder AI" assistant

**Step 1: Create config structure**

```text
.ai/
└── supercoder/
    └── config.json
```

**Step 2: Update policy files**

In [`.metaHub/policies/root-structure.yaml`](../.metaHub/policies/root-structure.yaml):

```yaml
allowed_directories:
  ai_assistants:
    # ... existing tools ...
    - name: '.ai/supercoder'
      purpose: 'SuperCoder AI configuration'
```

**Step 3: Update unified settings**

In `.ai/settings.yaml`:

```yaml
tools:
  supercoder:
    enabled: true
    config_dir: '.ai/supercoder/'
    settings:
      auto_approve: true
    best_for:
      - 'New use case'
    auto_approve_flag: 'autoApprove: true'
    status: 'supported'
```

**Step 4: Update documentation**

- Add to [`AI-AUTO-APPROVE-GUIDE.md`](AI-AUTO-APPROVE-GUIDE.md)
- Add to [`AI-TOOL-PROFILES.md`](AI-TOOL-PROFILES.md)
- Update tool count

**Step 5: Create PR with ADR reference**

### 6.3 Scenario: Adding a New Top-Level Directory

**Use Case:** Add `scripts/` directory for governance helper scripts

**Step 1: Evaluate placement**

- Q: Does it belong at root or inside `.metaHub/`?
- A: Helper scripts should go in `.metaHub/scripts/` (already exists)
- Decision: Do not create `scripts/` at root

**Alternative Use Case:** Add `infra/` for Terraform governance of cloud resources

**Step 1: Create ADR**

```markdown
# ADR-00X: Infrastructure Directory at Root

## Decision

Add `infra/` directory at root for Terraform configurations
that govern cloud resources for the portfolio.

## Rationale

- Terraform configs are infrastructure-as-code, not governance code
- Separating from `.metaHub/` maintains clear boundaries
- Standard location for IaC in modern repos
```

**Step 2: Update policies**

In [`root-structure.yaml`](../.metaHub/policies/root-structure.yaml):

```yaml
allowed_directories:
  optional:
    - name: 'infra'
      purpose: 'Infrastructure-as-code (Terraform) for portfolio cloud resources'
```

In [`portfolio-structure.yaml`](../.metaHub/templates/structures/portfolio-structure.yaml):

```yaml
portfolio:
  required_structure:
    # ... existing ...
    - 'infra/' # Infrastructure configs (optional)
```

**Step 3: Update validator**

In [`structure_validator.py`](../.metaHub/scripts/structure_validator.py):

```python
# No code change needed if policy is correctly parsed
# Validator reads categories from YAML
```

**Step 4: Add tests**

```python
def test_infra_directory_allowed():
    """Test that infra/ is allowed at root."""
    validator = StructureValidator()
    result = validator.validate_root_structure()
    assert "infra" in [d["name"] for d in result["allowed_dirs"]]
```

---

## 7. Next Steps / Implementation Checklist

### Immediate (Week 1) - COMPLETE

- [x] Create `.ai/` directory structure
- [x] Create symlinks from legacy AI config locations to `.ai/`
- [x] Update `.ai-settings.yaml` to reference new locations
- [x] Add deprecation warnings to legacy config usage

### Short-term (Weeks 2-3) - COMPLETE

- [x] Update all AI tools to read from `.ai/`
- [x] Update [`structure_validator.py`](../.metaHub/scripts/structure_validator.py) with consolidated AI check
- [x] Update [`portfolio-structure.yaml`](../.metaHub/templates/structures/portfolio-structure.yaml) with `portfolio_root` section
- [x] Add root structure validation to weekly workflow

### Medium-term (Week 4) - COMPLETE

- [x] Remove legacy AI config directories (symlinks removed from root)
- [x] Update enforcement to error on scattered AI configs
- [x] Clean up any remaining forbidden items at root
- [x] Update downstream documentation

### Long-term (Ongoing)

- [ ] Monitor for new AI tools requiring integration
- [ ] Review root structure contract quarterly
- [ ] Ensure all changes follow change-control process

---

## Appendix A: Root Structure Diagram

```text
┌─────────────────────────────────────────────────────────────────┐
│                     ROOT DIRECTORY STRUCTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  REQUIRED                    RECOMMENDED         OPTIONAL       │
│  ────────                    ───────────         ────────       │
│  .github/                    tests/              .vscode/       │
│  .metaHub/                   CONTRIBUTING.md     .allstar/      │
│  docs/                       .pre-commit-...     .husky/        │
│  README.md                   mkdocs.yml          .ai/           │
│  LICENSE                     .gitattributes                     │
│  SECURITY.md                                                    │
│  GOVERNANCE.md                                                  │
│  .gitignore                                                     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FORBIDDEN (Must not exist at root)                             │
│  ─────────────────────────────────                              │
│  *.pdf, *.doc*, *.xls*, *.ppt*     → Move to docs/reports/      │
│  *-report.json, *-results.json     → Add to .gitignore          │
│  temp-*, new-*, backup*, old*      → Delete or gitignore        │
│  .env, .env.*                      → Never commit               │
│  src/, app/, lib/                  → Wrong repo type            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GITIGNORED (May exist locally, never committed)                │
│  ────────────────────────────────────────────────               │
│  organizations/          Local org working copies               │
│  .mypy_cache/           Type checker cache                      │
│  .pytest_cache/         Test cache                              │
│  .ruff_cache/           Linter cache                            │
│  __pycache__/           Python bytecode                         │
│  node_modules/          Node.js deps                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Appendix B: Quick Reference Card

### Validation Command

```bash
python .metaHub/scripts/structure_validator.py --root
```

### Expected Output (Compliant)

```text
Root Structure: COMPLIANT | Violations: 0 | Warnings: 0
```

### Policy File Location

```text
.metaHub/policies/root-structure.yaml
```

### Change-Control Checklist

1. Update ADR-002
2. Update root-structure.yaml
3. Update portfolio-structure.yaml
4. Update structure_validator.py (if needed)
5. Add tests
6. Update workflows (if needed)
7. Get platform maintainer approval
8. Merge and announce

---

**Document Maintainers:** Platform Team
**Review Cycle:** Quarterly
**Related Documents:**

- [`ADR-002-root-structure-policy.md`](adr/ADR-002-root-structure-policy.md)
- [`.metaHub/policies/root-structure.yaml`](../.metaHub/policies/root-structure.yaml)
- [`portfolio-structure.yaml`](../.metaHub/templates/structures/portfolio-structure.yaml)
- [`AI-AUTO-APPROVE-GUIDE.md`](AI-AUTO-APPROVE-GUIDE.md)
