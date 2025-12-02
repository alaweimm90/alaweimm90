# CLI Tool Extraction Report

**Date:** November 30, 2025  
**Scope:** Portfolio-wide analysis across alaweimm90-business projects, automation, tools, and .metaHub

---

## 1. Executive Summary

This analysis identified **22 candidate CLI tools** extractable from existing code patterns across the portfolio. The patterns fall into 6 major categories:

| Category | Tools Identified | Quick Wins | High Impact |
|----------|-----------------|------------|-------------|
| Common Utilities | 5 | 3 | 2 |
| Authentication & Database | 3 | 1 | 2 |
| Analytics & AI/ML | 4 | 1 | 3 |
| Infrastructure & DevOps | 5 | 3 | 2 |
| Governance & Compliance | 3 | 2 | 3 |
| Cross-Project Synergies | 2 | 1 | 2 |

**Key Findings:**
- Supabase client patterns are duplicated across LiveItIconic, Repz, and other projects
- Analytics tracking code exists in at least 3 different implementations
- Governance/validation scripts are scattered across `tools/legacy/`, `.metaHub/scripts/`, and `tools/cli/`
- E-commerce patterns (cart, payment, email) are duplicated in BenchBarrier and LiveItIconic

---

## 2. Proposed CLI Tools

### Category A: Common Utilities

| Tool Name | Projects | Description | Commands | Complexity |
|-----------|----------|-------------|----------|------------|
| `mh-config` | All projects | Unified config loading (YAML, JSON, env vars) with validation | `load`, `validate`, `merge`, `template` | Low |
| `mh-log` | All projects | Structured logging with JSON/console output, context injection | `setup`, `rotate`, `export` | Low |
| `mh-file` | All projects | File operations (read, write, glob, template rendering) | `read`, `write`, `glob`, `render`, `validate` | Low |
| `mh-http` | LiveItIconic, BenchBarrier, MarketingAutomation | HTTP client wrapper with retry, timeout, auth headers | `get`, `post`, `batch`, `health-check` | Medium |
| `mh-cache` | LiveItIconic, Repz | Local/Redis caching utilities | `get`, `set`, `invalidate`, `stats` | Medium |

### Category B: Authentication & Database

| Tool Name | Projects | Description | Commands | Complexity |
|-----------|----------|-------------|----------|------------|
| `mh-supabase` | LiveItIconic, Repz, CallaLilyCouture | Supabase client generator & migrations | `init`, `migrate`, `seed`, `gen-types`, `backup` | Medium |
| `mh-auth` | LiveItIconic, Repz | Authentication service scaffolding (Supabase Auth, JWT) | `init`, `gen-middleware`, `test-tokens` | Medium |
| `mh-stripe` | Repz, LiveItIconic | Stripe webhook handlers & payment flow scaffolding | `init-webhook`, `gen-checkout`, `test-events` | High |

### Category C: Analytics & AI/ML

| Tool Name | Projects | Description | Commands | Complexity |
|-----------|----------|-------------|----------|------------|
| `mh-analytics` | LiveItIconic, BenchBarrier, MarketingAutomation | Analytics tracking setup (GA4, custom events) | `init`, `track`, `export`, `dashboard` | Medium |
| `mh-predict` | enterprise/predictive, tools/atlas | ML quality predictor for code metrics | `train`, `predict`, `explain`, `report` | High |
| `mh-recommend` | LiveItIconic | Product recommendation engine | `train`, `suggest`, `evaluate` | High |
| `mh-perf` | LiveItIconic, tools/atlas | Performance analysis (complexity, bundle size) | `analyze`, `report`, `compare`, `ci-check` | Medium |

### Category D: Infrastructure & DevOps

| Tool Name | Projects | Description | Commands | Complexity |
|-----------|----------|-------------|----------|------------|
| `mh-docker` | LiveItIconic, Repz, MarketingAutomation | Docker/Compose scaffold & security scanning | `init`, `build`, `scan`, `compose-up` | Low |
| `mh-k8s` | infrastructure/, tools/infrastructure | Kubernetes manifest generation & validation | `gen`, `validate`, `apply`, `diff` | Medium |
| `mh-ci` | All .github/workflows | CI/CD workflow generator & validator | `gen`, `validate`, `lint`, `test-matrix` | Low |
| `mh-security` | tools/security | Security scanning orchestrator (SAST, secrets, deps) | `scan-all`, `scan-secrets`, `scan-deps`, `report` | Low |
| `mh-template` | tools/cli/devops, templates/ | Template discovery, application, and customization | `list`, `apply`, `create`, `update` | Medium |

### Category E: Governance & Compliance

| Tool Name | Projects | Description | Commands | Complexity |
|-----------|----------|-------------|----------|------------|
| `mh-govern` | tools/cli/governance.py (existing) | Unified governance CLI (consolidates 8 scripts) | `enforce`, `checkpoint`, `catalog`, `meta`, `audit`, `sync` | Low (exists) |
| `mh-validate` | tools/lib/validation.py | Schema & structure validation | `schema`, `structure`, `metadata`, `docker`, `tier` | Low |
| `mh-telemetry` | .metaHub/scripts/telemetry_dashboard.py | Workflow telemetry & dashboard | `collect`, `dashboard`, `export`, `alert` | Medium |

### Category F: Cross-Project Synergies

| Tool Name | Projects | Description | Commands | Complexity |
|-----------|----------|-------------|----------|------------|
| `mh-ecommerce` | BenchBarrier, LiveItIconic | E-commerce patterns (cart, checkout, orders) | `gen-cart`, `gen-checkout`, `gen-order`, `gen-email` | High |
| `mh-hooks` | All React projects | React hook generator with TypeScript types | `gen`, `test`, `document` | Medium |

---

## 3. Recommended Implementation Priority

### Phase 1: Quick Wins (Week 1-2)
| Priority | Tool | Effort | Impact | Notes |
|----------|------|--------|--------|-------|
| 1 | `mh-govern` | Exists | High | Already consolidated in tools/cli/governance.py |
| 2 | `mh-validate` | Low | High | Extract from tools/lib/validation.py |
| 3 | `mh-config` | Low | High | Consolidate from automation/utils.py |
| 4 | `mh-log` | Low | Medium | Extract from demo/scripts/logger.js |
| 5 | `mh-ci` | Low | High | Wrap existing reusable workflows |
| 6 | `mh-security` | Low | High | Combine tools/security/*.sh scripts |

### Phase 2: Medium Effort (Week 3-4)
| Priority | Tool | Effort | Impact | Notes |
|----------|------|--------|--------|-------|
| 7 | `mh-supabase` | Medium | High | Standardize across LiveItIconic, Repz |
| 8 | `mh-template` | Medium | High | Already scaffolded in tools/cli/devops.ts |
| 9 | `mh-analytics` | Medium | Medium | Consolidate from LiveItIconic/src/config/analytics.ts |
| 10 | `mh-perf` | Medium | Medium | Based on LiveItIconic/scripts/performance-analysis.ts |
| 11 | `mh-telemetry` | Medium | Medium | Wrap .metaHub/scripts/telemetry_dashboard.py |

### Phase 3: High Effort (Week 5-8)
| Priority | Tool | Effort | Impact | Notes |
|----------|------|--------|--------|-------|
| 12 | `mh-stripe` | High | High | Complex payment flows |
| 13 | `mh-ecommerce` | High | High | Cross-project standardization |
| 14 | `mh-predict` | High | Medium | ML model packaging |
| 15 | `mh-recommend` | High | Medium | Recommendation engine |

---

## 4. Quick Wins Analysis

### 4.1 `mh-govern` (Already Exists)
**Location:** `tools/cli/governance.py`  
**Status:** Consolidated, production-ready  
**Commands:** `enforce`, `checkpoint`, `catalog`, `meta`, `audit`, `sync`

```bash
# Usage
python tools/cli/governance.py enforce ./organizations/my-org/
python tools/cli/governance.py catalog --format json
```

### 4.2 `mh-validate`
**Source Files:**
- `tools/lib/validation.py` (303 lines)
- `tools/legacy/governance/compliance_validator.py`

**Proposed Structure:**
```
.metaHub/tools/mh-validate/
├── __init__.py
├── cli.py              # Click CLI
├── schema.py           # JSON Schema validation
├── structure.py        # Directory structure validation
├── docker.py           # Dockerfile security validation
└── tier.py             # Tier compliance checking
```

### 4.3 `mh-config`
**Source Files:**
- `automation/utils.py` (`load_yaml_file`, `get_automation_path`)
- `.metaHub/scripts/*.py` (common path resolution pattern)

**Proposed Commands:**
```bash
mh-config load ./config.yaml          # Load and validate config
mh-config merge ./base.yaml ./env.yaml # Merge configs
mh-config template ./template.yaml --vars key=value
```

### 4.4 `mh-security`
**Source Files:**
- `tools/security/dependency-scan.sh`
- `tools/security/sast-scan.sh`
- `tools/security/secret-scan.sh`
- `tools/security/trivy-scan.sh`
- `tools/security/security-scan-all.sh`

**Proposed Commands:**
```bash
mh-security scan-all ./                # Run all scans
mh-security scan-secrets ./            # Secrets only
mh-security scan-deps ./               # Dependencies only
mh-security report --format json       # Generate report
```

---

## 5. Existing Abstractions to Consolidate

### 5.1 Legacy Scripts → Unified CLIs
| Legacy Location | Target Tool | Files to Consolidate |
|-----------------|-------------|---------------------|
| `tools/legacy/orchestration/` | `mh-orchestrate` | 5 Python scripts (self_healing, checkpoint, telemetry, validator, hallucination) |
| `tools/legacy/governance/` | `mh-govern` | 8 Python scripts (already done) |
| `tools/legacy/devops/` | `mh-devops` | builder.ts, coder.ts, bootstrap.ts |

### 5.2 Duplicate Service Patterns
| Pattern | Locations | Consolidation Target |
|---------|-----------|---------------------|
| Supabase Client | LiveItIconic, Repz | `mh-supabase gen-client` |
| Auth Service | LiveItIconic/services/authService.ts | `mh-auth gen-service` |
| Payment Service | LiveItIconic, Repz | `mh-stripe gen-service` |
| Email Service | LiveItIconic, MarketingAutomation | `mh-email gen-service` |
| Analytics Utils | LiveItIconic (3 files) | `mh-analytics init` |

### 5.3 CLI Pattern Standardization
Current CLIs use different patterns:
- **Python:** Click (governance.py) ✅ Preferred
- **TypeScript:** Commander (atlas/cli) ✅ Preferred
- **Shell:** Plain bash (security/) ⚠️ Should wrap

**Recommendation:** Standardize on:
- Python CLIs: Click + Rich for output
- TypeScript CLIs: Commander + Ora for spinners

---

## 6. Implementation Architecture

### 6.1 Proposed Directory Structure
```
.metaHub/tools/
├── README.md
├── setup.py              # Package installer
├── pyproject.toml        # Python config
├── package.json          # Node config
│
├── core/                 # Shared utilities
│   ├── config.py         # Config loading
│   ├── logging.py        # Logging setup
│   ├── validation.py     # Schema validation
│   └── telemetry.py      # Metrics collection
│
├── cli/                  # CLI entry points
│   ├── mh.py             # Main Python CLI (mh <command>)
│   └── mh.ts             # Main TS CLI
│
├── commands/             # Command implementations
│   ├── govern/
│   ├── validate/
│   ├── config/
│   ├── security/
│   └── ...
│
└── templates/            # Code templates
    ├── supabase/
    ├── auth/
    ├── stripe/
    └── analytics/
```

### 6.2 Unified CLI Entry Point
```bash
# Single entry point for all tools
mh govern enforce ./path
mh validate schema ./config.yaml
mh security scan-all ./
mh supabase migrate
mh analytics init
```

---

## 7. Next Steps

1. **Immediate:** Symlink `tools/cli/governance.py` → `.metaHub/tools/cli/govern.py`
2. **This Week:** Extract `mh-validate` from `tools/lib/validation.py`
3. **This Week:** Create `mh-security` wrapper for shell scripts
4. **Next Week:** Scaffold `mh-supabase` from existing client patterns
5. **Ongoing:** Track usage metrics to prioritize remaining tools

---

## Appendix A: Source File Inventory

### Governance Scripts (8 files consolidated)
| File | Lines | Status |
|------|-------|--------|
| `.metaHub/scripts/catalog.py` | 517 | ✅ In governance.py |
| `.metaHub/scripts/enforce.py` | 597 | ✅ In governance.py |
| `.metaHub/scripts/checkpoint.py` | 328 | ✅ In lib/checkpoint.py |
| `.metaHub/scripts/sync_governance.py` | 237 | ✅ In governance.py |
| `.metaHub/scripts/ai_audit.py` | ~400 | ✅ In governance.py |
| `.metaHub/scripts/structure_validator.py` | ~200 | ✅ In lib/validation.py |
| `.metaHub/scripts/telemetry_dashboard.py` | 328 | Separate tool |
| `.metaHub/scripts/push_monorepos.py` | 219 | Separate tool |

### Service Patterns (LiveItIconic)
| Service | Lines | Reuse Potential |
|---------|-------|-----------------|
| authService.ts | 314 | High |
| paymentService.ts | 80 | High |
| emailService.ts | 103 | High |
| recommendationService.ts | 65 | Medium |
| analyticsService.ts | 133 | High |

### Infrastructure Patterns
| Pattern | Location | Files |
|---------|----------|-------|
| Docker | Repz, LiveItIconic | 2 Dockerfiles, 2 compose files |
| Kubernetes | infrastructure/, tools/infrastructure | 20+ manifests |
| Terraform | LiveItIconic, Repz (empty) | Placeholder only |
| CI/CD | .github/workflows | 15+ reusable workflows |
