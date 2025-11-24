## Scope

Implement additional governance and quality automation: CODEOWNERS, dependency updates, dead code detection, TODO/FIXME tracking, mutation testing hooks, type coverage metrics, bundle budgets, release dry run, and stronger security gates.

## Steps

1. CODEOWNERS: Add `.github/CODEOWNERS` for sensitive paths; owners placeholder teams.
2. Dependabot: Add `.github/dependabot.yml` for npm/pnpm/pip/docker/actions daily updates.
3. CI Matrix Enhancements:

- Run `ts-prune` with report; optional fail if errors.
- Run `type-coverage --detail` and upload report.
- Run `size-limit` where configured; upload output.

4. Release Dry Run:

- Add `semantic-release --dry-run` workflow to verify Conventional Commits and changelog generation.

5. Security Gates:

- Update security workflow to fail on critical/high audit vulnerabilities; fail on any SAST findings.

6. TODO/FIXME Tracking:

- Add workflow to count and upload TODO/FIXME occurrences for tech debt monitoring.

## Verification

- CI artifacts for each step; workflows pass and produce reports; security gates fail on critical/high.

Approve to proceed with adding these files and updating workflows.
