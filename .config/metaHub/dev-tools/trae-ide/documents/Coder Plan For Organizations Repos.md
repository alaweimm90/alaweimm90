## Scope

- Make the Coder plan concrete for `c:\Users\mesha\Desktop\GitHub\organizations`, mapping repo-specific tasks, files, CI jobs, and validation.

## Repository Landscape

- Mixed JS/TS and Python across `AlaweinOS/*`, `LLMWorks`, `SimCore`, `TalAI`, `MEZAN`, `qmlab`, `MeatheadPhysicist`, `alaweimm90-business/*`, `alaweimm90-tools/*`.
- Shared configs: `c:\Users\mesha\Desktop\GitHub\organizations\turbo.json`, `pnpm-workspace.yaml`, `.github/workflows/*`, Python `pyproject.toml`, `mypy.ini`, `ruff.toml` spread across repos.

## Org-Wide Tasks (Coder)

- Align Node versions: add `.nvmrc` where missing and pin Node across JS workspaces.
- Enforce TS strictness: enable `"strict": true`, wire `type-check` into PR CI and block on errors.
- Standardize testing: `vitest` or `jest` for TS apps with coverage thresholds; `pytest` for Python with coverage gates.
- Pre-commit hooks: add `ruff`, `mypy`, `eslint/prettier`, `commitlint` hooks; wire in CI.
- Secrets & security: ensure CodeQL, dependency audit, and secret scanning run per repo; add environment validation scripts.

## Repo-Specific Tasks (Coder)

- `AlaweinOS/HELIOS`
  - Add `pytest` suite and wire `mypy`/`ruff` gates; integrate into CI.
  - Files: `c:\Users\mesha\Desktop\GitHub\organizations\AlaweinOS\HELIOS\pyproject.toml`, `mypy.ini`, `ruff.toml`.
- `AlaweinOS/MEZAN`
  - Add CI job for `pytest` with coverage thresholds; tighten `mypy --strict` and fix findings.
  - Files: `c:\Users\mesha\Desktop\GitHub\organizations\AlaweinOS\MEZAN\pyproject.toml`, `pytest.ini`, `mypy.ini`, `ruff.toml`.
- `AlaweinOS/LLMWorks`
  - Enforce `vitest` coverage thresholds; add accessibility checks (axe/lighthouse) to CI; smoke E2E on PR.
  - Files: `.nvmrc`, `package.json`, `vite.config.ts`, `tsconfig.json`.
- `AlaweinOS/SimCore`
  - Integrate Python package build/test with strict `mypy`/`ruff`; unify TS+Py coverage reporting.
  - Files: `package.json`, `pyproject.toml`, `pytest.ini`, `tsconfig.json`, `vite.config.ts`.
- `TalAI`
  - Add CI gates for typing (`mypy`), linting (`ruff`), and tests; redact PII in logs.
  - Files: `CLAUDE.md`, `LICENSE`, `README.md`, `mypy.ini`, `ruff.toml`.
- `qmlab`
  - Wire `vitest` to CI; component unit tests; enable `type-check` and coverage gates on PR.
  - Files: `package.json`, `vitest.config.ts`, `tsconfig*.json`.
- `MeatheadPhysicist`
  - Add `tox`/`nox` matrix; enforce `mypy --strict`; consolidate `pytest.ini`; frontend add integration tests for API client and coverage gates.
  - Files: multiple `pyproject.toml`, `pytest.ini`, `mypy.ini`, `ruff.toml`, `frontend/package.json`.
- `alaweimm90-business/*`
  - Add `pnpm` workspace lint/type-check to CI; ensure service health checks and mocks; performance test thresholds.
  - Files: `turbo.json`, service `package.json`.
- `alaweimm90-tools/admin-dashboard`
  - Add route-level integration tests; mock network I/O; enforce coverage.
  - Files: `package.json`, `tsconfig.json`, `vite.config.ts`.
- `shared/security-middleware`
  - Add unit tests (TS and Python variants); publish reusable packages; enforce API stability via tests.
  - Files: `security-middleware/package.json`, `pyproject.toml`.
- `.personal`
  - Add basic `lint/test/type-check` and CI for portfolio site.
  - Files: `.personal/*/package.json`.

## CI & Workflows (Coder)

- Map per-repo scripts to dedicated jobs in `.github/workflows/*`; add Node/Python version matrices.
- Cache dependencies (pnpm/pip); upload coverage artifacts; block merges on thresholds.
- Run secrets scanning and SBOM generation for dependency auditing.

## Standards & Guards

- TypeScript: strict, no `any`; functions ≤20 lines, single nesting; unit-test critical paths with mocked I/O.
- Python: `mypy --strict`, `ruff` formatting and lint; test coverage gates and isolated tests.
- Security: never commit secrets; redact PII; principle of least privilege in workflows.
- Git: linear history with rebase; auto-accept safe refactors; flag breaking changes.

## Validation Plan

- Per-repo: run type checks, lints, and tests locally; ensure CI passes with added gates.
- Coverage: TS via `vitest`/`jest`, Python via `pytest` + `coverage.py`; set thresholds in CI.
- Accessibility/performance: integrate axe/Lighthouse checks for frontend repos.

## Next Steps

- Confirm the Coder plan and priority ordering; then implement per-repo changes starting with high-impact repos (`LLMWorks`, `SimCore`, `MEZAN`, `HELIOS`).
