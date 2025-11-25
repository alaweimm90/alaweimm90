# 1. Governance Quick Reference Card

**Repository**: https://github.com/alaweimm90/alaweimm90
**Status**: 7/10 tools active | 3 manual setups pending (30 min)
**Last Updated**: 2025-11-25

---

## 1.1. 🚀 30-Minute Activation Checklist

### 1.1.1. ☐ Step 1: GitHub Rulesets (5 min)
```
URL: https://github.com/alaweimm90/alaweimm90/settings/rules

Actions:
1. New ruleset → Branch ruleset
2. Name: "Master Branch Protection"
3. Target: master
4. Enable:
   ☑ Require PR (1 approval)
   ☑ Require code owner reviews
   ☑ Dismiss stale reviews
   ☑ Require status checks:
      • Super-Linter
      • OpenSSF Scorecard
      • OPA Policy Enforcement
      • policy-bot (add after installing)
   ☑ Block force pushes
   ☑ Block deletions
5. Bypass: None (or admin only)
6. Create
```

### 1.1.2. ☐ Step 2: Policy-Bot (10 min)
```
URL: https://github.com/apps/policy-bot

Actions:
1. Install/Configure
2. Select: alaweimm90/alaweimm90
3. Grant permissions
4. Install
5. Verify: https://github.com/settings/installations
6. Go back to Rulesets, add "policy-bot" to required checks
```

### 1.1.3. ☐ Step 3: OpenSSF Allstar (10 min)
```
URL: https://github.com/apps/allstar-app

Actions:
1. Install/Configure
2. Select: alaweimm90/alaweimm90
3. Grant permissions
4. Install
5. Check for issues: gh issue list --label allstar
```

### 1.1.4. ☐ Step 4: Verify (5 min)
```bash
# Create test PR
git checkout -b test-activation
echo "# Test" >> README.md
git commit -am "test: verify governance"
git push origin test-activation
gh pr create --title "Test Activation"

# Verify all checks run:
# - Super-Linter ✓
# - OPA/Conftest ✓
# - policy-bot ✓
# - CODEOWNERS ✓
# - Rulesets ✓
```

---

## 1.2. 📊 Tool Status Dashboard

| Tool | Status | Trigger | Location |
|------|--------|---------|----------|
| Super-Linter | ✅ Active | Every PR | `.github/workflows/super-linter.yml` |
| Scorecard | ✅ Active | Weekly Sat 1:30 AM | `.github/workflows/scorecard.yml` |
| Renovate | ✅ Active | Every 3 hours | `.github/workflows/renovate.yml` |
| CODEOWNERS | ✅ Active | Every PR | `.github/CODEOWNERS` |
| OPA/Conftest | ✅ Active | Every PR | `.github/workflows/opa-conftest.yml` |
| Backstage | ✅ Active | On-demand | `.metaHub/backstage/` |
| SLSA | ✅ Active | Push to master | `.github/workflows/slsa-provenance.yml` |
| Rulesets | 🟡 Pending | Always | GitHub UI Settings |
| Policy-Bot | 🟡 Pending | Every PR | `.metaHub/policy-bot.yml` |
| Allstar | 🟡 Pending | Continuous | `.allstar/allstar.yaml` |

---

## 1.3. 🔍 Daily Commands (5 min)

```bash
# Check Renovate PRs
gh pr list --label dependencies

# Check Allstar issues
gh issue list --label allstar

# Check recent workflow runs
gh run list --limit 5

# Check for failed workflows
gh run list --status failure --limit 5
```

---

## 1.4. 📈 Weekly Commands (15 min)

```bash
# View latest Scorecard results
gh run list --workflow=scorecard.yml --limit 1

# Count Renovate PRs merged this week
gh pr list --state merged --search "merged:>=$(date -d '7 days ago' +%Y-%m-%d)" --label dependencies

# Check SLSA provenances
ls -la .metaHub/security/slsa/

# Review Policy-Bot activity (after installed)
gh pr list --search "is:pr" --limit 10
```

---

## 1.5. 🆘 Emergency Commands

```bash
# Bypass pre-commit hook for governance changes
git commit --no-verify -m "your message"

# Force push (if admin and emergency)
# WARNING: Only for emergencies!
git push --force origin branch-name

# Disable Renovate temporarily
# Edit .metaHub/renovate.json: "enabled": false

# Disable Super-Linter validator
# Edit .github/workflows/super-linter.yml: VALIDATE_<LANG>: false
```

---

## 1.6. 📚 Documentation Quick Links

| Need | Document | Location |
|------|----------|----------|
| How do I...? | Developer Guide | `.metaHub/DEVELOPER_GUIDE.md` |
| Something broke | Troubleshooting | `.metaHub/TROUBLESHOOTING.md` |
| What to monitor | Monitoring Checklist | `.metaHub/MONITORING_CHECKLIST.md` |
| Track progress | Baseline Metrics | `.metaHub/security/BASELINE_METRICS.md` |
| What changed | Changelog | `.metaHub/CHANGELOG.md` |
| What's next | Next Steps | `.metaHub/NEXT_STEPS.md` |
| Complete guide | Governance Summary | `.metaHub/GOVERNANCE_SUMMARY.md` |
| Setup Policy-Bot | Policy-Bot Setup | `.metaHub/POLICY_BOT_SETUP.md` |
| Setup Allstar | Allstar Setup | `.allstar/ALLSTAR_SETUP.md` |

---

## 1.7. 🔧 Common Fixes

### 1.7.1. Super-Linter failing on valid code
```yaml
# .github/workflows/super-linter.yml
env:
  VALIDATE_JSCPD: false  # Disable copy-paste detection
```

### 1.7.2. OPA rejecting valid Dockerfile
```rego
# .metaHub/policies/docker-security.rego
# Add to allowed exceptions
allow_latest_for_dev := {
  "node:latest"
}
```

### 1.7.3. Too many Renovate PRs
```json
// .metaHub/renovate.json
{
  "prConcurrentLimit": 2,
  "prHourlyLimit": 1
}
```

### 1.7.4. Policy-Bot not working
1. Check: https://github.com/settings/installations
2. Verify webhook deliveries
3. Ensure in required status checks

---

## 1.8. 📞 Getting Help

**Documentation**: `.metaHub/` folder (4000+ lines)

**Tool Docs**:
- Super-Linter: https://github.com/super-linter/super-linter
- Scorecard: https://github.com/ossf/scorecard
- Renovate: https://docs.renovatebot.com/
- Policy-Bot: https://github.com/palantir/policy-bot
- OPA: https://www.conftest.dev/
- Backstage: https://backstage.io/docs/
- SLSA: https://slsa.dev/
- Allstar: https://github.com/ossf/allstar

**Community**:
- GitHub: https://github.community/
- OpenSSF: https://openssf.slack.com/
- Backstage: https://discord.gg/backstage

**Contact**: @alaweimm90

---

## 1.9. 📋 First Week Checklist

### 1.9.1. Day 1 (Today)
- [ ] Complete 3 manual setups (30 min)
- [ ] Create test PR
- [ ] Verify all checks pass
- [ ] Merge test PR

### 1.9.2. Day 2-3
- [ ] Monitor Renovate PRs
- [ ] Review any Allstar issues
- [ ] Check workflow runs

### 1.9.3. Day 4-5
- [ ] Test creating Dockerfile (OPA validation)
- [ ] Test governance change (Policy-Bot routing)
- [ ] Review SLSA provenance generation

### 1.9.4. Day 6-7 (Weekend)
- [ ] Wait for first Scorecard run (Saturday)
- [ ] Collect baseline metrics
- [ ] Update BASELINE_METRICS.md

---

## 1.10. 🎯 Success Metrics

**Week 1 Targets**:
- [ ] All 10 tools active
- [ ] First test PR enforced correctly
- [ ] First Renovate PR merged
- [ ] Baseline metrics collected
- [ ] Zero false positives

**Month 1 Targets**:
- [ ] Scorecard score: 8+/10
- [ ] Renovate auto-merge rate: >70%
- [ ] Zero Allstar open issues
- [ ] PR merge time: <24 hours
- [ ] Monitoring routine established

---

## 1.11. 💡 Pro Tips

1. **Read the violation message** - It tells you exactly what's wrong
2. **Test locally first** - Run `conftest test` before pushing
3. **Small PRs are better** - Easier to review and merge
4. **Let Renovate auto-merge** - Trust the process for minor updates
5. **Check Scorecard weekly** - Track security improvements
6. **Review SLSA provenances** - Verify supply chain security
7. **Use Backstage catalog** - Understand service dependencies
8. **Monitor Allstar issues** - Address security violations promptly

---

## 1.12. 🚦 Status Indicators

**🟢 Green (Active & Working)**:
- Super-Linter, Scorecard, Renovate, CODEOWNERS, OPA, Backstage, SLSA

**🟡 Yellow (Ready, Manual Setup Needed)**:
- GitHub Rulesets, Policy-Bot, Allstar

**🔴 Red (Needs Attention)**:
- None currently

---

**Print this card** or bookmark this file for quick reference during daily operations!

**Repository**: https://github.com/alaweimm90/alaweimm90
**Updated**: 2025-11-25
**Version**: 1.0
