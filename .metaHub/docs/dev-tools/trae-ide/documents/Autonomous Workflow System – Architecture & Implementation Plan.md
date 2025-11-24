## Goals

Build an agentic DevOps system that autonomously enforces quality, security and reasoning standards end‑to‑end: inputs → sanitized prompts → routed models → controlled state → auditable outputs, with CI/CD, observability, and repeatable operations.

## Repository & Folder Standards

- Structure:
  - `/services/*` microservices (API, routing, hygiene, history)
  - `/packages/*` shared libs (prompt engine, validators, model wrappers)
  - `/cli` IDE/CLI tools (preprocessors, linters)
  - `/configs` JSON/YAML (routing tables, selection configs, env overrides)
  - `/tmp` ephemeral workspace (auto-clean: 24h TTL, size quotas)
  - `/archives` immutable artifacts (retention: 90d default, configurable per environment)
  - `/tests` unit/integration/regression/hallucination audits
  - `/docs` ADRs, OpenAPI, SOPs
- Enforcement:
  - CI jobs validate presence and policies (tmp cleanup, archive retention) and fail if non‑compliant

## Core Architecture (Agentic DevOps)

- Automated approval workflows: environment‑gated deploys; policy-as-code gates (lint, typecheck, coverage, security, docs compliance); auto‑PR creation & merge when all gates pass
- Context hygiene service: cleans inputs, strips PII, normalizes tokens, enforces length limits; exposes `/sanitize` API
- Model wrappers: sanitize in/out state, redact sensitive tokens, add trace IDs, enforce deterministic retry/backoff
- Prompt structuring/auto‑fixing: template engine with schema validation; auto‑repair rules for missing sections, inconsistent variables
- Audit trails: append‑only event store (JSON lines); correlation IDs across services; signed logs

## Template Management

- System‑wide template prompts:
  - Template registry (`/configs/templates/*.json`) with schema: `name`, `intent`, `sections`, `variables`, `constraints`
  - Auto‑recall: selection logic matches `intent` and context features with scoring
- Repository tests:
  - Unit tests for each template (≥80% coverage): structure, variable binding, rendering determinism
  - Golden files under `/tests/golden/templates/*`
- Pre‑prompt validators & auto‑cleaners:
  - Validators: required sections (Analysis, Inputs, Constraints, Output), variable resolution, length bounds
  - Cleaners: whitespace normalization, Unicode cleanup, token budget fitting
- Input normalization lint:
  - CLI script (`cli normalize`) runs normalizers and emits diff; CI fails on dirty inputs

## Model Routing System

- Routing tables & selection config (`/configs/routing.json`):
  - Fields: `model`, `capabilities`, `cost`, `latencySLO`, `maxTokens`, `safety`, `preferredUseCases`
- Pre‑task model registries:
  - Per task type (summarize, generate, classify, plan), default routing and fallbacks
- Environment‑specific overrides:
  - `/configs/env/{dev,staging,prod}/routing-overrides.json` to change models/limits per environment
- History management:
  - Stateful store (`/services/history`) with compaction rules (memory compression), TTLs, and GDPR delete support

## Processing Rules

- Context hygiene protocols: PII redaction, profanity filtering, unsafe command removal, max context window caps
- Tokenization normalization: standard tokenizer selection, pre‑token estimation, budget margin (e.g., 5%)
- Prompt template standards: required sections, naming conventions, variable syntax, constraint blocks
- Chain‑of‑thought requirements: enforce structured reasoning sections without exposing internal CoT; external outputs remain concise
- Self‑refutation mechanisms: second‑pass critique with rubric; blocklist and risk heuristics
- Multi‑pass MCSP loop: plan → compose → simulate → critique → patch → finalize; configurable `passes` and stop criteria

## IDE/CLI Tooling

- Pre‑processors for input cleaning: `cli sanitize <file|stdin>`
- Prompt format validators: `cli validate-prompt <template>` (JSON Schema)
- Self‑refutation chain definitions: YAML DSL (`/configs/refutation/*.yaml`) with steps and rubrics
- Logic check passes: lints for contradictions, unmet constraints, missing variables
- Memory compression rules: heuristics to summarize history segments under token budget;
- Agent‑level consistency checks: diff across passes; assert invariants and constraints are met

## Testing Framework

- Unit tests for templates: structure, rendering, normalization, validator passing (≥80% coverage)
- Integration tests for MCSP loops: multi‑step pipelines with mocks; assert stop criteria and patch correctness
- Regression tests for state handling: history compaction, archival, deletion, replay with deterministic outputs
- Hallucination audits: adversarial prompts; measure unsupported claims; ensure refutation catches violations

## DevOps Principles & CI/CD

- Predictability & automation: pipelines run sanitized steps; reproducible builds with lockfiles
- Repeatability & traceability: artifacts contain inputs, normalized forms, template IDs, routing decisions
- Debuggability: per‑step logs, correlation IDs, `--debug` mode to dump intermediary states
- Enforced gates: validated inputs, controlled state transitions, reliable outputs (schema‑validated), strict reasoning patterns, comprehensive docs
- CI jobs:
  - sanitize+lint: input cleaning and prompt validation
  - unit/integration/regression: coverage ≥80%
  - security: Semgrep, CodeQL; audit fail on critical/high
  - docs: README compliance, ADR/OpenAPI lint
  - drift‑sync: align configs; auto‑PR
  - dashboard: aggregate quality metrics

## Logging & Audit Trails

- Structured JSON logs (traceId, spanId, step, service, timing, tokens, decision)
- Append‑only audit files in `/archives` with retention policies and signed digests
- OpenTelemetry spans across services; log correlation; redaction filters applied before persistence

## Versioning & Governance

- Semantic versioning for services/packages; commitlint / Conventional Commits
- CODEOWNERS for sensitive paths
- ADRs required for architectural changes; templates enforced by CI
- Exceptions documented in `CLAUDE.md` with expiry and owner

## Implementation Phases & Milestones

1. Scaffolding & standards: repo layout, configs, CI gates (sanitize, validate, test, docs, security)
2. Core services: hygiene API, wrappers, template engine, routing tables
3. MCSP loop engine & refutation: YAML DSL + orchestrator with pluggable steps
4. History & memory: state store, compression, retention, GDPR deletes
5. Observability & audits: OTel integration, dashboard, append‑only archives
6. Test suite: unit/integration/regression/adversarial audits; coverage targets
7. Hardening & rollout: environment overrides, approvals, auto‑PR sync; SLO monitoring

## Deliverables

- `/services/*` microservices (hygiene, routing, history, orchestrator)
- `/packages/*` shared libs (prompt templates, validators, wrappers)
- `/cli/*` tooling (sanitize, validate, normalize, refute)
- `/configs/*` routing, templates, refutation, env overrides
- `/tests/*` comprehensive suite + coverage reports
- `/docs/*` ADRs, OpenAPI, SOPs; README templates
- CI artifacts: quality dashboard, audit logs, coverage, SARIF; auto‑issues when gates fail

## Success Criteria & Verification

- Coverage: ≥80% overall; MCSP integration tests pass
- Security: 0 critical/high vulnerabilities; SAST/CodeQL clean
- Docs: README compliance ≥95%; ADR/OpenAPI lint pass
- Hygiene: 100% prompt/template validation; inputs normalized; audit logs present for all runs
- Routing: selection decisions logged; env overrides respected

## Risks & Mitigations

- Model drift/latency variance → routing overrides, dynamic scoring
- Over‑sanitization loses intent → whitelist patterns, unit tests
- Audit privacy → redaction and access controls
- CI flakiness → deterministic mocks and retry/tuning

Approve to proceed. I will scaffold services/packages/configs, add validators and CI gates, implement routing and MCSP orchestration, and deliver tests, logs, and dashboards with automated approvals and sync across repositories.
