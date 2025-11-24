## Objectives

* Drive the repo to “error free” in iterative autopilot passes: zero lint errors in automation code, green tests, stable CLI/workflows, and clean CI runs, while preserving strict TS standards and minimizing risky changes.

## Error Sources Snapshot

* Linting: ESLint errors in `.automation` (legacy JS patterns: multiple classes per file, `no-console`, loops, `no-return-await`, case declarations, import ordering) and missing plugins/configs across monorepo packages.
* Runtime: Port conflicts on status server, plugin logging level mismatch, approval override path correctness.
* Tests: Automation Jest suite green; monorepo tests may need coverage.

## Autopilot Phases

### Phase 1: Configuration Stabilization
* Normalize root ESLint config with per-file overrides (TS vs JS) to focus errors on real risks; keep warnings for console/log-heavy diagnostic modules.
* Ensure required plugins/configs are installed per package (react/eslint configs for UI packages, TypeScript plugins for TS packages).
* Verify and fix `prettier`/format pipeline compatibility.

### Phase 2: Automation Code Hardening
* Agent orchestrator:
  * Move dynamic `require` usages to top-level imports; replace `return await` with `return` to satisfy `no-return-await`.
  * Split large files if needed to quiet `max-classes-per-file` (or selectively disable rule for JS-only modules where split is not valuable).
  * Remove unused vars; add small refactors for `radix`, continue/loops to array methods where safe.
* Notification plugin: map custom levels to Winston (`warn`, `info`, `error`) and keep console logs as warnings.
* Self optimizer/monitor/repository-scanner: reduce “error” rules to warnings for intentional logging/loops, add minimal fixes (e.g., radix).

### Phase 3: Monorepo Lint Readiness
* For React packages, add `eslint-plugin-react` and appropriate configs; run lint in each workspace and resolve missing plugin errors.
* Scope strict TS rules to `.ts/.tsx` files; avoid applying TS rules to `.js` modules.

### Phase 4: Tests & Workflows
* Run automation tests; keep green.
* Add/adjust tests for failover/override approval to ensure behavior stays correct.
* Execute `code-quality-check` and other workflows programmatically (using distinct `STATUS_PORT`) and fix any runtime errors surfaced.

### Phase 5: CI Pipeline Clean Run
* Run root `npm run lint` and `turbo run test`; fix remaining errors or tune overrides per package to remove spurious errors.
* Ensure zero “error” level lint issues in `.automation`; allow warnings for diagnostic logs.

### Phase 6: Runtime Validation
* Keep main server on `7070`; use `STATUS_PORT` for ad-hoc runs to avoid `EADDRINUSE`.
* Exercise assign/override tasks, workflows and monitor; confirm no uncaught exceptions.

### Phase 7: Documentation Touch‑ups (Minimal)
* Update `AUTOMATION_AGENT_ORCHESTRATION.md` only if behavior changes (why-only comments).

### Phase 8: Housekeeping
* Terminal management (avoid killing running dev server), ensure commands run in new terminals; close when done.

## Deliverables

* Zero ESLint errors in `.automation`; root lint passes with only acceptable warnings.
* Green automation tests; workflows execute without runtime exceptions.
* Commands to reproduce: lint/test runs per package, workflow demos with `STATUS_PORT`.

## Guardrails

* Preserve strict TS settings; no secrets; keep changes minimal and localized; comment only “why”.

## Rollback Plan

* Changes applied per-file with clear diffs; can revert individual edits without impacting overall functionality.

## Next Step

* Proceed to implement phases 1–6 iteratively until error counts reach zero in `.automation` and CI is green; report concise progress after each pass.