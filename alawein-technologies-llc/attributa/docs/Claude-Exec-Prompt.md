Comprehensive Claude Code Execution Prompt for Attributa.dev

You are Claude Code operating in a repository with the Attributa.dev project (React + Vite + TypeScript, shadcn UI, Tailwind, optional Supabase). Your objective is to read the provided documents, generate the audit outputs, and implement a minimal, high-impact set of improvements aligned with those documents.

Context documents (read fully before doing anything):
- docs/User-Manual.md
- docs/Audit-Plan.md
- docs/Visual-Audit.md
- docs/Architecture-Audit.md

Operating constraints
- Privacy-first: do not add telemetry; external APIs must remain opt-in.
- Do not change business logic; focus on UX, accessibility, security hardening, performance, and DX.
- Use design tokens from index.css and tailwind.config.ts; avoid raw colors.
- Keep changes minimal, atomic, and well-documented with tests when feasible.

Tasks
1) Comprehension and Plan
- Summarize each document in 5–8 bullets.
- Propose a minimal prioritized plan (max 8 tasks) delivering the highest impact per effort.

2) Visual Audit Execution
- Audit routes: /, /scan, /results, /workspace, /settings, /documentation, 404.
- Verify single H1 per page and SEO tags via src/hooks/useSEO.ts usage; add or fix where missing.
- Ensure images use loading="lazy" and descriptive alt.
- Confirm components use semantic tokens; eliminate ad-hoc colors in UI code.
- Output: docs/visual_audit_findings.md with page-by-page issues and recommendations.

3) Architecture Audit Execution
- Review src/lib/* analyzers, workers, and services for caching, worker offload, and rate limiting.
- Identify bottlenecks and SPOFs; propose concrete improvements.
- Output: docs/architecture_audit_findings.md with risks and recommended changes.

4) Implement High-Impact Improvements (minimal set)
- SEO: Ensure useSEO is invoked on all pages (src/pages/*.tsx) with title/meta/description and canonical. Add missing canonical link tags if needed.
- Performance: Lazy-load heavy components and images; add React.lazy where practical without altering UX.
- Accessibility: Ensure visible focus rings and keyboard navigation on key flows; add ARIA labels where missing.
- Security: Add or document CSP where applicable; ensure inputs have length limits.

5) Testing and Reporting
- Run npm test; fix or update tests minimally as needed.
- Generate artifacts:
  - artifacts/visual_audit_summary.md
  - artifacts/architecture_audit_summary.md
  - artifacts/change_log_diff.md (summarize code changes)
- Ensure CI passes locally (if configured).

6) Git Hygiene
- Use conventional commits (feat:, fix:, docs:, refactor:, perf:, test:, chore:).
- Small PR(s) grouped by concern; include before/after screenshots where UI changes apply.

Acceptance Criteria
- All pages have a single H1 and proper SEO meta via useSEO; canonical links present.
- No direct color classes remain where tokens exist; shadcn variants respected.
- Lazy loading for non-critical images; heavy components code-split where safe.
- Audit findings files created with clear, actionable items.
- All unit tests pass: npm test.

Commands (if needed)
- npm install
- npm run dev
- npm test

Deliverables
- Updated code with minimal, high-impact changes.
- docs/visual_audit_findings.md and docs/architecture_audit_findings.md.
- artifacts/* summaries and change log.
- Brief PR description referencing this prompt and the four docs.
