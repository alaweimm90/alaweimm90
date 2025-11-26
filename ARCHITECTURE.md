# Portfolio Governance System Architecture

## System Overview

The portfolio governance system is a three-layer enforcement and visibility system:

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: ENFORCEMENT (enforce.py)                       │
├─────────────────────────────────────────────────────────┤
│ Validates governance compliance at multiple points:     │
│ - Local pre-commit checks                               │
│ - Consumer repo CI/CD validation                        │
│ - Governance repo centralized enforcement               │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 2: VISIBILITY (catalog.py)                        │
├─────────────────────────────────────────────────────────┤
│ Auto-discovers and catalogs portfolio:                  │
│ - Scans organizations/ for all repositories             │
│ - Validates .meta/repo.yaml in each repo               │
│ - Generates catalog.json (authoritative inventory)      │
│ - Tracks compliance status across portfolio             │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 3: DRIFT DETECTION (checkpoint.py)                │
├─────────────────────────────────────────────────────────┤
│ Detects unintentional changes:                          │
│ - Weekly snapshots of portfolio state                   │
│ - Comparison with previous week                         │
│ - Drift report showing intentional vs unintentional     │
│ - Audit trail for compliance                            │
└─────────────────────────────────────────────────────────┘
```

## Script Architecture

### enforce.py - Idempotent Validation

**Purpose:** Single source of truth for enforcement logic

**Checks:**
- Repository structure (allowed files, .metaHub paths)
- Metadata schema validation (.meta/repo.yaml)
- Docker security (non-root USER, HEALTHCHECK, no :latest)
- Kubernetes manifests (resource limits, probes)
- Service SLOs (availability, latency targets)
- Architecture Decision Records (ADR presence)

**Features:**
- Deterministic (same input → same output)
- Composable (each check independent)
- Cacheable (skip unchanged files)
- Multi-format reports (JSON, Markdown, GitHub)

**Integration Points:**
```
Consumer Repo CI → enforce.py → Report
         ↓
Governance Repo CI → enforce.py (portfolio-wide) → Reports
         ↓
Pre-commit Hook → enforce.py --strict → Block bad commits
```

### catalog.py - Portfolio Cataloging

**Purpose:** Authoritative inventory of all repositories

**Process:**
1. Scan organizations/ recursively
2. Find all .meta/repo.yaml files
3. Validate against schema
4. Generate catalog.json
5. Compare with previous version
6. Report changes (new, deleted, status)

**Output:** catalog.json
```json
{
  "version": "1.0",
  "generated": "2025-11-26T00:00:00Z",
  "organizations": [
    {
      "name": "alaweimm90-business",
      "repos": [
        {
          "name": "live-it-iconic",
          "type": "app",
          "language": "typescript",
          "compliance": "compliant"
        }
      ]
    }
  ]
}
```

### checkpoint.py - Drift Detection

**Purpose:** Detect unintentional changes in portfolio

**Weekly Process:**
1. Load previous week's catalog.json
2. Generate current catalog.json
3. Compare (new repos, deletions, status changes)
4. Generate drift-report.md
5. Archive report for audit trail

**Drift Report Example:**
```markdown
# Weekly Drift Report — Week of 2025-11-20

## Summary
- 2 new repositories added
- 1 repository archived  
- 3 repositories with status changes
- 1 new policy violation

## Changes
- organizations/alaweimm90-business/new-product (NEW)
- organizations/repz: compliant → violation (Docker security)
```

## Workflow Integration

### Consumer Repository Flow

```
Developer Pushes Code
       ↓
.github/workflows/policy.yml (calls .github/workflows/policy.yml from governance repo)
       ↓
enforce.py runs locally
       ↓
Report generated and uploaded
       ↓
GitHub shows compliance status
```

### Governance Repository Flow

```
Scheduled: Push to organizations/
       ↓
.github/workflows/enforce.yml (governance repo)
       ↓
enforce.py + catalog.py (portfolio-wide validation)
       ↓
Reports generated and committed
       ↓
.github/workflows/checkpoint.yml (weekly)
       ↓
checkpoint.py generates drift report
       ↓
Weekly summary posted to GitHub
```

## Data Flows

### Push → Enforcement → Catalog

```
Push Event
    ↓
CI Triggers enforce.py
    ↓
Enforcement Report Generated
    ↓
If .meta/repo.yaml exists:
    Catalog.py updates catalog.json
    ↓
    Compliance Status Updated
```

### Weekly Checkpoint Flow

```
Monday 00:00 UTC
    ↓
checkpoint.py starts
    ↓
Load previous catalog.json
    ↓
Run catalog.py for current state
    ↓
Compare catalogs
    ↓
Generate drift-report.md
    ↓
Archive for audit
    ↓
Post summary to GitHub Issues
```

## Philosophy

**Three Principles:**

1. **Idempotency**
   - Same input always produces same output
   - No hidden state, no race conditions
   - Safe to run multiple times

2. **Determinism**
   - Exact same checks locally and in CI
   - Developers catch issues before push
   - No CI-only surprises

3. **Composability**
   - Each check can run independently
   - Easy to add/remove checks
   - Clear responsibility per check

## Integration with OPA

**Python (enforce.py) validates:**
- Structure and schema (fast, Python native)
- Metadata completeness
- File patterns
- Cross-repo consistency

**OPA (policies) provides:**
- Governance principles (as code)
- Complex rules (Rego logic)
- Policy versioning
- Central reference point

**Both run together for complete coverage:**
```
enforce.py (fast, structural)
    ↓
OPA policies (comprehensive, principled)
    ↓
Combined report (full governance picture)
```

## Scaling Considerations

**Current Portfolio:**
- 5 organizations
- ~15 repositories
- ~45% with .meta/repo.yaml

**Future Scaling:**
- Parallel enforcement (by organization)
- Caching layer for large portfolios
- GraphQL API for catalog queries
- Real-time dashboard
- Dependency graph analysis

## Monitoring & Alerts

**Tracked Metrics:**
- Compliance percentage (repos meeting governance)
- Violations by type (structure, docker, k8s, etc.)
- Drift rate (changes per week)
- Mean time to compliance (MTTC)

**Alerts:**
- Critical violations (block deployment)
- Compliance regression (more violations than previous week)
- Drift exceeds threshold
- New organization missing governance

## File Locations

| Component | Location |
|-----------|----------|
| enforce.py | `.metaHub/scripts/enforce.py` |
| catalog.py | `.metaHub/scripts/catalog.py` |
| checkpoint.py | `.metaHub/scripts/checkpoint.py` |
| requirements.txt | `.metaHub/scripts/requirements.txt` |
| catalog.json | `.metaHub/catalog/catalog.json` |
| drift reports | `.metaHub/catalog/weekly-reports/` |
| CI workflows | `.github/workflows/enforce.yml`, etc. |

## Decision Rationale

**Why Python (not just OPA)?**
- Speed: Python faster for local execution
- Flexibility: Complex checks easier in Python
- Integration: Better with GitHub APIs, JSON Schema
- Debugging: Better error messages than Rego

**Why Weekly Checkpoints?**
- Reduces noise vs real-time alerts
- Allows human review and intentional marking
- Audit trail for compliance
- Detects trends over time

**Why Separate enforce/catalog/checkpoint?**
- Single Responsibility Principle
- Easy to test independently
- Can be called in different contexts
- Allows composition and reuse

---

**Generated:** 2025-11-26
**Status:** Phase 2 Architecture Definition
