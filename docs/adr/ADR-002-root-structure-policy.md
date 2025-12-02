# ADR-002: Root Directory Structure Policy

## Status

Accepted

## Date

2025-11-27

## Context

The root directory of the governance repository had become cluttered with:

- Temporary JSON report files (audit results, enforcement reports)
- Misplaced files (PDFs, workspace files)
- Unclear directories (`alaweimm90/`, `new-repos/`)
- No formal policy for what belongs at root level

This made the repository harder to navigate and violated the principle of clear organization that we enforce on downstream repositories.

## Decision

We will implement a **Root Directory Structure Policy** that:

1. **Defines allowed items** - Explicit list of files/directories permitted at root
2. **Forbids clutter** - Specific patterns that must not exist at root
3. **Enforces via automation** - Integrated with existing governance tools
4. **Provides migration path** - Clear rules for relocating misplaced items

### Allowed at Root

#### Required Files

- `README.md` - Repository overview
- `LICENSE` - Open source license
- `SECURITY.md` - Security policy

#### Required Directories

- `.metaHub/` - Central governance system
- `organizations/` - Org monorepo templates (gitignored)
- `.github/` - GitHub workflows and configs

#### Optional (Recommended)

- `docs/` - MkDocs documentation
- `tests/` - Governance tests
- Standard config files (`.gitignore`, `.pre-commit-config.yaml`, etc.)

### Forbidden at Root

- PDF/Office documents (move to `docs/reports/`)
- Generated JSON reports (add to `.gitignore`)
- Workspace files (move to `.vscode/` or gitignore)
- Temporary directories (`temp-*`, `new-*`)

## Consequences

### Positive

- Clear, navigable root directory
- Consistent with governance principles we enforce
- Automated enforcement prevents future clutter
- Easier onboarding for new contributors

### Negative

- Requires migration of existing misplaced files
- May need exceptions for CI/CD generated files

## Implementation

1. Create `root-structure.yaml` policy in `.metaHub/policies/`
2. Update `.gitignore` to exclude generated reports
3. Migrate misplaced files to appropriate locations
4. Add enforcement to CI/CD workflow (optional warning)

## References

- `.metaHub/policies/root-structure.yaml` - Full policy definition
- ADR-001: Organization Monorepos - Related architecture decision
