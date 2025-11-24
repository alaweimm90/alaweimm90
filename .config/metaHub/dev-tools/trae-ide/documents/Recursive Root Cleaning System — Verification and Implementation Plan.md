## Scope & Objectives

- Implement a safe, recursive cleaning tool that runs from repository root and applies to all folders.
- Detect and quarantine temp/cache/build artifacts; standardize naming/structure; normalize permissions; log and verify changes.
- Provide unit tests, CI schedule, and documented exceptions.

## Detection Rules (Recursive)

- Temp: `*.tmp`, `*.temp`, `*.bak`, `~*`, `*.swp`, `.DS_Store`, `Thumbs.db`
- Cache: `.cache/`, `.next/`, `.vite/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.coverage/`, `coverage/`
- Build artifacts: `dist/`, `build/`, `out/`, `*.map`, `*.min.*`
- Python: `__pycache__/`, `.venv/`, `venv/`
- Node: `node_modules/` (quarantine only if not referenced by workspace config)
- Logs: `*.log` older than N days (configurable)
- Duplicate files: identical content MD5, preserve one, quarantine extras

## Exceptions & Safe Zones

- Protected: `.git/`, `.github/`, `archives/`, `.docs-templates/`, configs under `config/`, workflow files, lockfiles
- Allowlist file: `.cleanignore` (glob patterns per folder) to skip paths
- Quarantine instead of delete by default; direct deletion available only in explicit `--hard-delete` mode (never default)

## Naming & Structure Standardization

- Files: enforce kebab-case for docs (`README.md`/canonical exceptions), remove illegal/special chars
- Folders: ensure lower-case with hyphens for generated dirs; merge duplicate case variants
- Rewrite names via mapping log (JSONL) and generate a changeset report

## Permissions & Ownership (Cross-Platform)

- Normalize file mode on POSIX: scripts `0755`, regular files `0644`
- Skip ownership changes on Windows; on POSIX, optional chown via config (disabled by default)
- Dry-run prints intended permission corrections; apply only with `--fix-perms`

## Tooling & Implementation

- Script: `scripts/clean/root-clean.js`
  - Options: `--dry-run`, `--quarantine-dir=archives/quarantine/<ts>`, `--fix-perms`, `--hard-delete=false`
  - Reads `config/cleaning.json` and `config/cleaning.schema.json`
  - Walks filesystem recursively; applies detection rules; moves to quarantine; standardizes names; writes audit JSONL to `archives/clean-ops.log`
  - Error handling: per-file try/catch, continue-on-error; produce `errors.log`
- Verification script: `scripts/clean/verify-clean.js`
  - Computes before/after counts per category; validates no protected files were modified; outputs `clean-verify.json`
- Unit tests: `scripts/clean/__tests__/*.test.js` using `node:test`
  - Mock FS via temp directories; assert detection/move/rename/perm normalization

## Configuration

- `config/cleaning.json` keys:
  - `age_days_for_logs`, `quarantine_path`, `hard_delete` (default false), `fix_permissions` (default false)
  - `detect_patterns`: arrays for temp/cache/build globs
  - `protected_paths`: array of root-relative paths
  - `allowlist_patterns`: global patterns, plus folder `.cleanignore`
- Schema: `config/cleaning.schema.json` (validate in CI via `ajv-cli`)

## Logging & Documentation

- Audit JSONL: ISO timestamp, action (`quarantine`, `rename`, `perm-fix`, `duplicate-remove`), original path, new path, size, reason
- Error log: `archives/clean-errors.log` with stack traces
- README: `docs/CLEANING.md` describing rules, usage, safety, rollback

## CI & Scheduling

- Workflow: `.github/workflows/repo-clean.yml`
  - Steps: validate config → run `root-clean.js --dry-run` → run verification → upload reports → create PR labeled `repo-clean`
  - Scheduled daily; manual `workflow_dispatch`
- Approvals Gate: require `files-approved` label for PRs moving/deleting large files

## Verification & Success Criteria

- Automated checks: `clean-verify.json` shows reduced counts for temp/cache/build; no protected files touched
- Space reclaimed report and dashboards; artifacts uploaded
- Unit tests coverage ≥ 80% for cleaning core; CI green

## Rollback & Safety

- All changes via PR; quarantine preserves original files; rollback by restoring from `archives/quarantine/<ts>/`
- Dry-run mode default; hard delete only with explicit config and approvals label

## Implementation Steps

1. Add config & schema
2. Implement `root-clean.js` (walk, detect, quarantine, rename, perm fix, audit)
3. Implement `verify-clean.js`
4. Write unit tests
5. Add CI workflow and schedule
6. Document in `docs/CLEANING.md`

Approve to proceed; I will add the scripts, configs, tests, CI workflow and documentation, run dry-run verification, and deliver artifacts and a PR for review.
