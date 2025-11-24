## Goals

- Accelerate CI/DevOps execution with yolo-friendly triggers while preserving core quality gates.
- Provide one-click and comment-driven batch runs (lint, typecheck, coverage, audit, docs, build).
- Maximize cache reuse, parallelism, and auto-merge of safe changes.

## Additions

1. Quick DevOps Batch (Parallel)

- File: organizations/.github/workflows/quick-devops-batch.yml
- Matrix executes selected tasks concurrently: lint, typecheck, coverage, audit, docs, build
- Concurrency: cancel in-progress for same branch; cache pnpm; upload aggregated artifacts

2. Comment-Triggered Runs

- File: organizations/.github/workflows/comment-trigger.yml
- Triggers on issue/PR comments (e.g., `/devops lint typecheck coverage`)
- Uses actions/github-script to dispatch Quick DevOps Batch with chosen tasks and cwd

3. Fast Cache & Concurrency

- Enhance existing workflows with pnpm store cache, `concurrency: group` + `cancel-in-progress: true`
- Use `actions/cache` for `node_modules/.pnpm` and `.vitest` when applicable

4. Safe Auto-Merge for Standards Sync

- File: organizations/.github/workflows/auto-merge.yml
- Auto-merges PRs labeled `repo-standards` when all required checks pass (commitlint, lint, typecheck, coverage, security)

5. Ops Sandbox Presets

- Add presets documentation in Ops Sandbox workflow description
- (Optional) Add `preset` input to load canned command bundles (lint+typecheck, coverage+audit)

6. Aggregated Summary Artifact

- Extend Quick DevOps Batch to produce `fast-summary.json` with counts: ESLint errors/warnings, coverage %, typecheck pass, audit severity

## Verification

- Run comment `/devops lint typecheck coverage audit docs` on a PR → parallel jobs complete, artifacts uploaded, quality gates enforced
- Drift-sync PRs with label `repo-standards` → auto-merge when green
- Concurrency prevents duplicate runs; caches speed up successive executions

## Files to Add/Update

- organizations/.github/workflows/quick-devops-batch.yml (new)
- organizations/.github/workflows/comment-trigger.yml (new)
- organizations/.github/workflows/auto-merge.yml (new)
- Update: organizations/.github/workflows/quick-devops.yml, ci-matrix.yml → add pnpm cache, concurrency group

## Guardrails

- No bypass of security gates; yolo-friendly triggers only change how runs start and parallelize
- Auto-merge limited to `repo-standards` PRs that meet all required checks

Approve to implement these workflows and updates; I will create the files, wire caching/concurrency, and verify by posting a comment to trigger batch runs.
