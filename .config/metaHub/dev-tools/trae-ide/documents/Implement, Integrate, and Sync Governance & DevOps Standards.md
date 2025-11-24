## Approach

Create a single source of truth for standards and configurations, enforce them via policy-as-code in CI, and continuously sync them across projects with automated checks, drift detection, and auto-PR updates. Documentation, rules, and workflows stay versioned and verified.

## Single Source of Truth

- Central standards: `organizations/REPO_STANDARDS.md` and `.github/workflows/*` as the canonical reference
- Versioned policy packages: publish shared configs (ESLint/Prettier/commitlint/tsconfig) as internal packages (npm/pip) to consume across projects
- Registry: maintain a manifest listing required files, versions, and checks

## Policy-as-Code Enforcement

- Required CI gates (org-level): lint, typecheck, coverage ≥80%, audit (fail on critical/high), SAST/CodeQL, docs lint, README compliance, quality issues auto-creation
- Branch protections: require checks + 2 approvals + CODEOWNERS review for sensitive paths
- Pre-commit: formatters/linters; commitlint; secret scanning

## Automated Sync & Drift Detection

- Drift checker workflow: compares project files against source-of-truth
- Auto-PR updates: opens PRs to align configs/workflows/templates when drift detected
- Exceptions: documented in `CLAUDE.md` with expiration and owner; CI allows justified exceptions via labels

## Documentation Integration

- README template enforcement: compliance CI checks sections (Overview, Setup, API, Quick Start, Testing, Usage)
- ADRs: mandatory for architecture/security changes; templates and CI lint
- API docs: OpenAPI 3.0 under `docs/api`; Redocly lint in CI

## Rollout & Change Management

- Phased rollout: Audit → Apply non-blocking → Enforce gates → Monitor
- Change proposals: ADR + PR checklist with impact analysis; semantic versioning for policy packages
- Dep updates: Dependabot for npm/pip/docker/actions; require CI pass and owner approval

## Verification & Evidence

- CI artifacts: ESLint JSON, coverage summaries, SARIF, audit JSON, docs lint outputs, quality summary
- Dashboards: aggregate metrics per repo (errors, coverage %, vulnerabilities, compliance %) and trend lines
- Commands to verify locally: `pnpm -w -r exec eslint . --format json --output-file eslint-report.json`, `tsc --noEmit`, `vitest --run --coverage`, `pnpm -w audit --json`, `markdownlint`, `redocly lint`

## Ops & Developer Experience

- Quick DevOps workflow: one-click lint, typecheck, test, coverage, build, audit, docs (yolo mode)
- Ops Sandbox: safe/yolo command execution with logs; use safe mode for isolation
- Templates & scaffolding: generators set up standard files and CI automatically

## Governance & Training

- CODEOWNERS for sensitive paths; security sign-offs
- Training: workshops + runbooks; quarterly standards review; KPIs (coverage, vulnerabilities, SLO adherence)
- Audits: periodic compliance scans with board-level reports

## Milestones & Success Criteria

- M1: Standards & CI enforced; branch protections enabled
- M2: All repos synced to source-of-truth; drift auto-PRs active
- M3: Documentation compliance ≥95%; OpenAPI validated
- M4: Security posture: 0 critical/high vulns; SAST/CodeQL clean
- M5: Performance & reliability gates adopted across services

Approve to:

1. Establish the standards registry and shared config packages
2. Add drift detection + auto-PR sync workflow
3. Enable org-level branch protections and dashboards
4. Document exceptions and change management in CLAUDE.md
5. Kick off audit → apply → enforce rollout across repos
