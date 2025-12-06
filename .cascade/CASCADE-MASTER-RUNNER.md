# 🚀 CASCADE MASTER RUNNER - PHASES 6-10

## Overview

Execute all 5 phases in sequence. Total AI-accelerated time: **2-3 hours**.

| Phase | Focus | Time | Handoff File |
|-------|-------|------|--------------|
| 6 | Security Hardening | 30-60 min | `PHASE-6-SECURITY-HANDOFF.md` |
| 7 | Performance Optimization | 20-40 min | `PHASE-7-PERFORMANCE-HANDOFF.md` |
| 8 | Monitoring & Observability | 30-45 min | `PHASE-8-MONITORING-HANDOFF.md` |
| 9 | Backup & DR | 15-25 min | `PHASE-9-BACKUP-HANDOFF.md` |
| 10 | Accessibility | 20-30 min | `PHASE-10-ACCESSIBILITY-HANDOFF.md` |

---

## Cascade Prompts (Copy-Paste Sequence)

### Phase 6
```
@workspace Execute .cascade/PHASE-6-SECURITY-HANDOFF.md

Complete all 7 security tasks. Commit: feat(security): Complete Phase 6 security hardening
```

### Phase 7
```
@workspace Execute .cascade/PHASE-7-PERFORMANCE-HANDOFF.md

Complete all 5 performance tasks. Commit: feat(perf): Complete Phase 7 performance optimization
```

### Phase 8
```
@workspace Execute .cascade/PHASE-8-MONITORING-HANDOFF.md

Complete all 5 monitoring tasks. Commit: feat(monitoring): Complete Phase 8 monitoring & observability
```

### Phase 9
```
@workspace Execute .cascade/PHASE-9-BACKUP-HANDOFF.md

Complete all 5 backup tasks. Commit: feat(backup): Complete Phase 9 backup & disaster recovery
```

### Phase 10
```
@workspace Execute .cascade/PHASE-10-ACCESSIBILITY-HANDOFF.md

Complete all 5 accessibility tasks. Commit: feat(a11y): Complete Phase 10 accessibility compliance
```

---

## One-Shot Full Execution Prompt

If Cascade supports long context, use this single prompt:

```
@workspace Execute Phases 6-10 from the .cascade/ directory:

1. Read and execute PHASE-6-SECURITY-HANDOFF.md
2. Read and execute PHASE-7-PERFORMANCE-HANDOFF.md  
3. Read and execute PHASE-8-MONITORING-HANDOFF.md
4. Read and execute PHASE-9-BACKUP-HANDOFF.md
5. Read and execute PHASE-10-ACCESSIBILITY-HANDOFF.md

For each phase:
- Complete ALL tasks in order
- Create ALL specified files
- Commit after each phase with the specified message

Expected total time: 2-3 hours with AI acceleration.
```

---

## Verification Checklist (After All Phases)

```bash
# Phase 6 - Security
cat .github/dependabot.yml
cat .pre-commit-config.yaml | grep detect-secrets

# Phase 7 - Performance
cat turbo.json
npm run perf:bundle

# Phase 8 - Monitoring
npm run telemetry dashboard
npm run health

# Phase 9 - Backup
npm run backup configs
npm run backup list

# Phase 10 - Accessibility
npm run a11y:audit
cat .config/accessibility/wcag.yaml
```

---

## Files Created Summary

| Phase | Files Created |
|-------|---------------|
| 6 | dependabot.yml, secret_scanning.yml, tools/security/audit.ts |
| 7 | turbo.json, tools/performance/bundle-analyzer.ts, python-benchmark.py |
| 8 | .config/telemetry/config.yaml, tools/telemetry/index.ts, tools/health/check.ts |
| 9 | .config/backup/config.yaml, tools/backup/backup.ts, docs/DR-RUNBOOK.md |
| 10 | .config/accessibility/wcag.yaml, tools/accessibility/audit.ts |

---

## After Completion

Report back to Augment for:
1. Phase 11-20 handoff creation (Governance & Compliance)
2. QAPLibria packaging and PyPI publish
3. REPZ/LiveItIconic launch preparation

---

## Emergency Rollback

If something breaks:
```bash
git log --oneline -10  # Find last good commit
git reset --hard <commit>
```

