## Diagnosis

- Current compliance/maintenance workflows scan only directories containing `package.json` or `pyproject.toml`. Root appears to lack either, so root-level docs (e.g., many `SECURITY_*` files) are not in scope and remain untouched.
- Cleaning scripts quarantine temp/log/build artifacts, but do not archive unauthorized Markdown files at the repository root unless the folder is scanned.

## Fix Plan

1. Expand scope detection

- Update `get_scope()` to include `.` (root) and any directory that contains Markdown docs even if no `package.json`/`pyproject.toml`.
- Add a root allowlist: `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `SECURITY.md`, `ARCHITECTURE.md`. Everything else at root is considered unauthorized and will be archived.

2. Root-specific cleaning rules

- Extend docs-compliance and docs-maintenance workflows to:
  - Scan root and archive unauthorized docs at root into `archives/docs/root/<timestamp>/`.
  - Generate missing required root docs from `.docs-templates` if absent.

3. Immediate execution paths

- Add `repo-root-clean.yml` workflow to run a dry-run + archive at root and open a PR labeled `repo-clean`.
- Provide commands to run cleaning locally (`node scripts/docs/clean.js` and root clean script) and generate verification artifacts.

4. Verification & artifacts

- Produce `docs-compliance-report.json` including root entries, `docs-dashboard.json`, and PR diff showing archived files.
- Ensure audit log entries in `archives/docs-ops.log` for root actions.

5. Tests & safety

- Add unit tests for the updated scope logic and root archiving behavior.
- Quarantine-only; no hard deletion. All changes via PR for rollback.

## Steps to Execute

- Implement scope changes in workflows and cleaning scripts.
- Add `repo-root-clean.yml` and run it (manual dispatch) to produce PR that archives unauthorized root docs.
- Review artifacts and merge PR if acceptable.

Approve to proceed; I’ll update scope detection, root allowlist, workflows, and tests, then run the root cleanup to resolve the current mess while preserving rollback safety.
