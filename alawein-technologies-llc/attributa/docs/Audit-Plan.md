# Attributa.dev General Audit Plan and Report Template

Audience: Internal stakeholders and management
Format: Numbered phases with bullet sub-steps; end with findings and actionable recommendations.

1) Planning
- Define scope: frontend (React/Vite/TS), analyzers in src/lib, workers, services, Supabase edge functions.
- Objectives: Identify security vulnerabilities, compliance gaps, and inefficiencies.
- Stakeholders: Product lead, security lead, frontend maintainer, infra owner.
- Artifacts: Codebase, README*, README_ANALYZERS.md, env config, CI logs.

2) Fieldwork
- Code/config review: src/lib/**, src/pages/**, src/components/ui/**, supabase/functions/**.
- Interviews: developers, security, support.
- Test runs: typical scans (text, LaTeX, code), exports, settings toggles.
- Evidence collection: screenshots, console logs, network traces, test results.

3) Reporting
- Findings: Rank by impact × likelihood; include evidence and affected areas.
- Recommendations: Clear owners, effort estimate, ETA, acceptance criteria.
- Compliance mapping: Privacy policy alignment, data handling disclosures.

4) Follow-up
- Acceptance by owners; remediation plan; retest and close.

Scope Checklists
Security
- Supabase: RLS enabled on all user data tables; no permissive policies; storage bucket policies correct; secrets only in edge function env.
- Frontend: Content Security Policy, no inline secrets, no dynamic code execution; sanitize inputs; size/timeouts for worker tasks.
- Dependencies: Audit for known CVEs; lockfile hygiene.

Compliance
- Privacy policy matches behavior; external API usage is opt-in with clear consent.
- Data retention: Define and implement; ability to purge user data.

Operational Efficiency
- Bundle size targets; code splitting heavy components; lazy image loading.
- Model caching and worker offloading; rate limiting for citations; retries with backoff.
- Error boundaries and user-visible toasts for failures.

Deliverables
- Executive Summary
- Detailed Findings per area (Security, Compliance, Efficiency)
- Risk Register (with severity, likelihood, owner)
- Remediation Plan (tasks, ETA)

Templates
Finding
- Title
- Area: Security | Compliance | Efficiency
- Impact | Likelihood
- Evidence
- Recommendation
- Owner | ETA | Acceptance Criteria
