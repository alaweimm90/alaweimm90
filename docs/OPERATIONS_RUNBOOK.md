# Operations Runbook - Golden Path Governance System

This document provides operational procedures for deploying, monitoring, and maintaining the governance system.

## Overview

The Golden Path Governance System consists of three core components:

1. **enforce.py** - Policy enforcement engine
2. **catalog.py** - Service catalog generator
3. **checkpoint.py** - Drift detection system

## Production Deployment

### Prerequisites

```bash
# Python 3.11+
python --version  # Should be 3.11+

# Required packages
pip install -r .metaHub/scripts/requirements.txt
```

### Environment Configuration

Set the following environment variables:

| Variable           | Description                        | Default       |
| ------------------ | ---------------------------------- | ------------- |
| `GOLDEN_PATH_ROOT` | Path to central governance repo    | Auto-detected |
| `GOVERN_STRICT`    | Enable strict mode (warnings fail) | `false`       |

### GitHub Actions Deployment

The system runs automatically via GitHub Actions:

| Workflow         | Schedule                  | Trigger                        |
| ---------------- | ------------------------- | ------------------------------ |
| `enforce.yml`    | Every 6 hours             | Push to organizations/, manual |
| `catalog.yml`    | Daily at 8 AM UTC         | Push to organizations/, manual |
| `checkpoint.yml` | Weekly on Monday 8 AM UTC | Manual                         |

### Manual Deployment

```bash
# Run enforcement across all organizations
for org in organizations/*/; do
    python .metaHub/scripts/enforce.py "$org" --report json --output "enforcement-$(basename $org).json"
done

# Generate catalog
python .metaHub/scripts/catalog.py --format json

# Create checkpoint
python .metaHub/scripts/checkpoint.py --baseline
```

## Monitoring

### Key Metrics

Monitor these metrics from enforcement reports:

| Metric                | Description                        | Alert Threshold     |
| --------------------- | ---------------------------------- | ------------------- |
| `total_violations`    | Policy violations across portfolio | > 10 new violations |
| `total_warnings`      | Policy warnings                    | > 50 new warnings   |
| `compliance_degraded` | Repos losing compliance            | Any                 |
| `new_repos`           | Newly discovered repositories      | Informational       |

### Dashboard Queries

For JSON catalog analysis:

```bash
# Total repositories by tier
python3 -c "
import json
d = json.load(open('.metaHub/catalog/catalog.json'))
for tier, count in d['summary']['by_tier'].items():
    print(f'Tier {tier}: {count} repos')
"

# Non-compliant repositories
python3 -c "
import json
d = json.load(open('.metaHub/checkpoints/checkpoint-latest.json'))
for repo, data in d['repositories'].items():
    if not data.get('compliant'):
        print(f'{repo}: non-compliant')
"
```

### Alerting

The system automatically creates GitHub Issues when:

- Compliance degrades (repos lose compliance)
- New policy violations are detected

To customize alerting, modify `.github/workflows/checkpoint.yml`:

```yaml
- name: Create issue if drift degraded compliance
  if: steps.checkpoint.outputs.has_drift == 'true'
  uses: actions/github-script@v7
  with:
    script: |
      // Customize notification logic here
```

## Operational Procedures

### Daily Operations

1. **Review enforcement reports**
   - Check GitHub Actions for failed runs
   - Review violations in enforcement-\*.json artifacts

2. **Monitor catalog updates**
   - Verify catalog.json is current
   - Check for unexpected repository changes

### Weekly Operations

1. **Review drift report**
   - Check `.metaHub/catalog/weekly-reports/latest-drift.md`
   - Investigate any compliance degradation
   - Follow up on new repositories

2. **Update policies**
   - Review policy violation trends
   - Update policies if needed
   - Communicate changes to teams

### Monthly Operations

1. **Audit compliance**
   - Review tier 1-2 repository compliance
   - Escalate non-compliant critical repos
   - Plan remediation for persistent violations

2. **Clean up checkpoints**
   - Archive old checkpoint files
   - Review checkpoint storage usage

## Troubleshooting

### Workflow Failures

**Symptom:** Enforcement workflow fails
**Diagnosis:**

```bash
# Check workflow logs in GitHub Actions
# Look for Python errors or missing dependencies
```

**Resolution:**

- Verify Python version in workflow
- Check requirements.txt is up to date
- Review repository path existence

### Schema Validation Errors

**Symptom:** Repositories failing schema validation
**Diagnosis:**

```bash
python3 -c "
import json, yaml, jsonschema
schema = json.load(open('.metaHub/schemas/repo-schema.json'))
metadata = yaml.safe_load(open('path/to/.meta/repo.yaml'))
jsonschema.validate(metadata, schema)
"
```

**Resolution:**

- Update .meta/repo.yaml to match schema
- Check for typos in field names
- Verify required fields are present

### Missing Repositories

**Symptom:** Repositories not appearing in catalog
**Diagnosis:**

```bash
# Check organizations directory structure
ls -la organizations/*/

# Verify .meta/repo.yaml exists
find organizations -name "repo.yaml" -path "*/.meta/*"
```

**Resolution:**

- Ensure repository is in organizations/ directory
- Create .meta/repo.yaml if missing
- Re-run catalog generation

### Drift Detection Issues

**Symptom:** Drift not detected or false positives
**Diagnosis:**

```bash
# Compare checkpoints
diff <(jq -S '.repositories' checkpoint-prev.json) \
     <(jq -S '.repositories' checkpoint-latest.json)
```

**Resolution:**

- Verify previous checkpoint exists
- Check file hash calculation
- Review compliance check logic

## Recovery Procedures

### Restore from Checkpoint

If catalog becomes corrupted:

```bash
# Restore catalog from checkpoint
cp .metaHub/checkpoints/checkpoint-latest.json .metaHub/catalog/catalog.json

# Regenerate full catalog
python .metaHub/scripts/catalog.py
```

### Reset Baseline

To start fresh:

```bash
# Archive old checkpoints
mkdir -p .metaHub/checkpoints/archive
mv .metaHub/checkpoints/checkpoint-*.json .metaHub/checkpoints/archive/

# Create new baseline
python .metaHub/scripts/checkpoint.py --baseline
```

### Rollback Policy Changes

If policy changes cause issues:

```bash
# Revert policy file
git checkout HEAD~1 -- .metaHub/policies/problematic-policy.rego

# Re-run enforcement
python .metaHub/scripts/enforce.py organizations/ --report text
```

## Performance Tuning

### Large Portfolios (100+ repos)

For large portfolios, consider:

1. **Parallel execution** in workflows
2. **Incremental scanning** for changed repos only
3. **Caching** dependency installation

Example parallel workflow:

```yaml
jobs:
  enforce:
    strategy:
      matrix:
        org: [org1, org2, org3, org4, org5]
    steps:
      - run: python .metaHub/scripts/enforce.py organizations/${{ matrix.org }}
```

### Memory Optimization

For memory-constrained environments:

```bash
# Process one organization at a time
for org in organizations/*/; do
    python .metaHub/scripts/enforce.py "$org" --report json
    # Clear Python cache
done
```

## Support Escalation

| Level | Contact          | Response Time        |
| ----- | ---------------- | -------------------- |
| L1    | Repository owner | Same day             |
| L2    | Platform team    | 24 hours             |
| L3    | Security team    | Critical issues only |

## Appendix

### File Locations

| File                                | Purpose            |
| ----------------------------------- | ------------------ |
| `.metaHub/scripts/enforce.py`       | Policy enforcement |
| `.metaHub/scripts/catalog.py`       | Catalog generation |
| `.metaHub/scripts/checkpoint.py`    | Drift detection    |
| `.metaHub/catalog/catalog.json`     | Current catalog    |
| `.metaHub/checkpoints/`             | Checkpoint history |
| `.metaHub/schemas/repo-schema.json` | Metadata schema    |
| `.metaHub/policies/`                | OPA/Rego policies  |

### Exit Codes

| Code | Meaning                           |
| ---- | --------------------------------- |
| 0    | Success                           |
| 1    | Violations found (enforce.py)     |
| 1    | Error (catalog.py, checkpoint.py) |

---

**Last Updated:** 2025-11-26
**Version:** 1.0
