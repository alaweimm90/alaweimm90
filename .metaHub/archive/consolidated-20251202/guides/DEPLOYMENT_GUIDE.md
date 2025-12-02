# Production Deployment Guide

## Overview

This guide covers the phased rollout of the governance system across the portfolio.

## Prerequisites

Before deployment:

- [ ] All governance scripts tested locally
- [ ] CI workflows validated
- [ ] OPA policies reviewed
- [ ] Templates verified for all languages
- [ ] Rollback procedures documented

## Deployment Phases

### Phase 1: Foundation (Day 1-2)

**Goal**: Deploy governance infrastructure

1. **Verify central repo**

   ```bash
   # Run all tests
   pytest tests/ -v

   # Validate scripts
   python .metaHub/scripts/enforce.py --help
   python .metaHub/scripts/catalog.py --help
   python .metaHub/scripts/meta.py --help
   ```

2. **Enable CI workflows**

   - Push to main branch
   - Verify all workflows pass
   - Check OpenSSF Scorecard results

3. **Generate initial catalog**

   ```bash
   python .metaHub/scripts/catalog.py --format json
   python .metaHub/scripts/catalog.py --format html --output catalog.html
   ```

### Phase 2: Pilot Organizations (Day 3-5)

**Goal**: Roll out to 2 pilot organizations

1. **Select pilot orgs**

   - Choose 1 small org (< 5 repos)
   - Choose 1 medium org (5-15 repos)

2. **Run enforcement in dry-run mode**

   ```bash
   python scripts/verify_and_enforce_golden_path.py --dry-run
   ```

3. **Review changes**

   - Check generated `.meta/repo.yaml` files
   - Verify CODEOWNERS content
   - Validate CI workflow references

4. **Apply changes**

   ```bash
   python scripts/verify_and_enforce_golden_path.py
   ```

5. **Monitor CI runs**

   - Watch for workflow failures
   - Address any policy violations
   - Document issues for other orgs

### Phase 3: Full Rollout (Day 6-8)

**Goal**: Deploy to all remaining organizations

1. **Run meta audit**

   ```bash
   python .metaHub/scripts/meta.py audit --output pre-rollout-audit.md
   ```

2. **Enforce on remaining orgs**

   ```bash
   for org in alaweimm90-business alaweimm90-science alaweimm90-tools AlaweinOS MeatheadPhysicist; do
     python .metaHub/scripts/enforce.py ./organizations/$org/ --report json --output "enforcement-$org.json"
   done
   ```

3. **Commit and push changes**

   ```bash
   git add organizations/
   git commit -m "chore: apply Golden Path governance to all organizations"
   git push
   ```

### Phase 4: Monitoring & Stabilization (Day 9-10)

**Goal**: Ensure system stability

1. **Generate post-rollout audit**

   ```bash
   python .metaHub/scripts/meta.py audit --output post-rollout-audit.md
   ```

2. **Compare audits**

   - Verify compliance score improvements
   - Identify remaining gaps
   - Prioritize fixes

3. **Enable scheduled enforcement**

   - Verify `enforce.yml` workflow runs every 6 hours
   - Check `catalog.yml` runs daily
   - Monitor `scorecard.yml` weekly runs

## Rollback Procedures

### Immediate Rollback

If critical issues are discovered:

```bash
# Revert last commit
git revert HEAD

# Or reset to specific commit
git reset --hard <commit-sha>
git push --force-with-lease
```

### Partial Rollback

To rollback specific organization:

```bash
# Checkout previous version of org
git checkout HEAD~1 -- organizations/affected-org/

# Commit the revert
git commit -m "revert: rollback governance for affected-org"
git push
```

### Workflow Rollback

To disable a problematic workflow:

1. Rename workflow file: `ci.yml` -> `ci.yml.disabled`
2. Commit and push
3. Investigate and fix
4. Re-enable by renaming back

## Monitoring

### Key Metrics

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| CI Success Rate | > 95% | < 90% |
| Enforcement Duration | < 30 min | > 45 min |
| Policy Violations | Decreasing | Increasing trend |
| Catalog Freshness | < 24 hours | > 48 hours |

### Dashboards

- **GitHub Actions**: Monitor workflow runs
- **Security Tab**: View Scorecard results
- **Catalog HTML**: Portfolio overview

### Alerts

Configure notifications for:

- Workflow failures
- Security vulnerabilities
- Policy enforcement failures
- Scheduled job misses

## Maintenance

### Weekly Tasks

- [ ] Review enforcement reports
- [ ] Check for new security advisories
- [ ] Update dependencies (Renovate PRs)
- [ ] Review Scorecard results

### Monthly Tasks

- [ ] Full portfolio audit
- [ ] Policy effectiveness review
- [ ] Template updates
- [ ] Documentation refresh

### Quarterly Tasks

- [ ] Architecture review
- [ ] Performance optimization
- [ ] Stakeholder feedback
- [ ] Roadmap planning

## Troubleshooting

### Common Issues

**Workflow fails with "schema validation error"**

```bash
# Validate schema locally
python -c "import json; json.load(open('.metaHub/schemas/repo-schema.json'))"

# Check repo.yaml
python -c "import yaml; yaml.safe_load(open('.meta/repo.yaml'))"
```

**Enforcement script hangs**

```bash
# Run with verbose logging
python .metaHub/scripts/enforce.py ./path --verbose

# Check for large files
find ./path -size +10M
```

**Catalog generation fails**

```bash
# Check organizations path
ls -la organizations/

# Run with debug output
python .metaHub/scripts/catalog.py --quiet 2>&1 | head -50
```

## Support

- **Issues**: Create GitHub issue with `governance` label
- **Urgent**: Tag @alaweimm90
- **Documentation**: See README.md and CONTRIBUTING.md
