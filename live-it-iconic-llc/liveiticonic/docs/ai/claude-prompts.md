# AI Assistant Guide for Live It Iconic

This file provides high-level context for AI assistants and contributors working in the Live It Iconic repository.

- Tech stack: React 18 + TypeScript + Vite + TailwindCSS + Supabase + Playwright + Vitest
- Domains: lifestyle coaching, content, subscriptions, analytics

For in-depth feature and architecture docs, use the `docs/` directory.

## Repository Layout (High Level)

- `src/`  main application code (components, pages, hooks, services)
- `tests/`  unit/integration/e2e helpers
- `docs/`  architecture, API, deployment, SEO, and testing documentation
- `scripts/`  automation scripts (performance, SEO, backups, etc.)
- `supabase/`  backend configuration and migrations
- `public/`  static assets, sitemap, manifest, etc.

## How to Work in This Repo (Quick Start)

1. Install dependencies (workspace-level conventions apply; typically `npm install` or `pnpm install`).
2. Run tests and lint before committing:
   - `npm test` / `npm run test`
   - `npm run lint`
3. For new features:
   - Add or update docs under `docs/`.
   - Prefer co-locating tests near the feature or in `tests/`.
4. Keep environment-specific configuration in `.env*` files (never commit real secrets).

When in doubt, prefer clear, explicit documentation and small, well-scoped PRs.

