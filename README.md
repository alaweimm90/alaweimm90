# Meta Governance Repository

**Enterprise-grade meta governance framework** for enforcing security policies, code quality standards, and compliance controls across all repositories.

## 🎯 Purpose

This repository serves as the **central governance layer** that:
- Defines and enforces security policies (OPA/Conftest)
- Provides reusable CI/CD workflows (GitHub Actions)
- Monitors security health (OpenSSF Scorecard, Allstar)
- Catalogs all services (Backstage portal)
- Tracks supply chain security (SLSA provenance)
- Automates dependency updates (Renovate)

**This repo does not contain application code** - it contains policies and configurations that govern other repositories.

---

## 📁 Structure

```
alaweimm90/alaweimm90/          # Meta Governance Repository
├── .github/                    # GitHub-level governance
│   ├── workflows/              # 5 governance workflows
│   │   ├── super-linter.yml              # Code quality gates
│   │   ├── opa-conftest.yml              # Policy enforcement
│   │   ├── slsa-provenance.yml           # Supply chain security
│   │   ├── scorecard.yml                 # Security monitoring
│   │   └── renovate.yml                  # Dependency updates
│   └── CODEOWNERS              # Ownership & approval requirements
│
├── .metaHub/                   # Governance core
│   ├── backstage/              # Developer portal & service catalog
│   ├── policies/               # OPA policies (repo structure, Docker security)
│   ├── security/               # SLSA provenance, Scorecard results, metrics
│   └── [documentation]/        # 11 comprehensive governance guides
│
├── .allstar/                   # Continuous security monitoring
├── .husky/                     # Git hooks
├── SECURITY.md                 # Security policy
├── README.md                   # This file
└── LICENSE                     # License
```

---

## 🛡️ Governance Tools (8/10 Active)

| Tool | Status | Purpose |
|------|--------|---------|
| **Super-Linter** | ✅ Active | Multi-language code quality (40+ validators) |
| **OPA/Conftest** | ✅ Active | Policy-as-code enforcement (15+ rules) |
| **SLSA Provenance** | ✅ Active | Supply chain attestations (Build Level 3) |
| **OpenSSF Scorecard** | ✅ Active | Security health monitoring (18 checks) |
| **Renovate** | ✅ Active | Automated dependency updates |
| **GitHub Rulesets** | ✅ Active | Branch protection (bypass-proof) |
| **CODEOWNERS** | ✅ Active | Mandatory code reviews (21 paths) |
| **Backstage** | ✅ Active | Developer portal (11 services cataloged) |
| **OpenSSF Allstar** | 🟡 Pending | Continuous security monitoring (5 policies) |
| **Policy-Bot** | ⚠️ Skipped | Advanced approval routing (requires self-hosting) |

**Coverage**: 80% (8/10 tools active)

---

## 🚀 Quick Start

### For Governance Administrators

1. **Review governance configuration**:
   ```bash
   # Explore governance policies
   cat .metaHub/policies/*.rego

   # Check workflow configurations
   ls -la .github/workflows/

   # View service catalog
   cat .metaHub/backstage/catalog-info.yaml
   ```

2. **Install remaining tool (Allstar)**:
   - Visit: <https://github.com/apps/allstar-app>
   - Install to this repository
   - Verify: `gh issue list --label allstar`

3. **Monitor governance**:
   ```bash
   # Daily (5 min)
   gh pr list --label dependencies      # Renovate PRs
   gh issue list --label allstar        # Security issues
   gh run list --limit 5                # Recent runs

   # Weekly (15 min)
   gh run list --workflow=scorecard.yml --limit 1  # Security score
   ```

### For Developers (Governed Repositories)

Repositories governed by this meta repo should:

1. **Reference reusable workflows**:
   ```yaml
   # .github/workflows/governance.yml in your repo
   name: Governance
   on: [push, pull_request]
   jobs:
     lint:
       uses: alaweimm90/alaweimm90/.github/workflows/super-linter.yml@master
     policies:
       uses: alaweimm90/alaweimm90/.github/workflows/opa-conftest.yml@master
   ```

2. **Register in Backstage catalog**:
   - Add service to `.metaHub/backstage/catalog-info.yaml`
   - Include API specs, dependencies, ownership

3. **Follow enforced policies**:
   - OPA repository structure policy
   - Docker security policy (no :latest, require USER, HEALTHCHECK)
   - CODEOWNERS approval requirements

---

## 📚 Documentation

Complete governance documentation in `.metaHub/`:

| Document | Purpose |
|----------|---------|
| [GOVERNANCE_SUMMARY.md](.metaHub/GOVERNANCE_SUMMARY.md) | Complete implementation guide (500+ lines) |
| [DEVELOPER_GUIDE.md](.metaHub/DEVELOPER_GUIDE.md) | How to work with governance tools |
| [MONITORING_CHECKLIST.md](.metaHub/MONITORING_CHECKLIST.md) | Daily/weekly/monthly monitoring tasks |
| [TROUBLESHOOTING.md](.metaHub/TROUBLESHOOTING.md) | Common issues and solutions |
| [BASELINE_METRICS.md](.metaHub/security/BASELINE_METRICS.md) | KPI tracking template |
| [CHANGELOG.md](.metaHub/CHANGELOG.md) | Implementation history |
| [QUICK_REFERENCE.md](.metaHub/QUICK_REFERENCE.md) | Printable quick reference card |
| [STRUCTURE_ANALYSIS.md](.metaHub/STRUCTURE_ANALYSIS.md) | Repository structure rationale |
| [CLEAN_START_SUMMARY.md](.metaHub/CLEAN_START_SUMMARY.md) | Cleanup report |

**Total documentation**: 11 files, 4000+ lines

---

## 🔐 Security & Compliance

### Frameworks Supported

- ✅ **NIST SSDF** (Secure Software Development Framework)
- ✅ **EO 14028** (Executive Order on Cybersecurity) - SBOM + SLSA attestations
- ✅ **SOC 2 Type II** - Control mappings documented
- ✅ **OWASP Top 10** - Full coverage

### Defense-in-Depth

1. **GitHub Platform Level** (Bypass-proof):
   - GitHub Rulesets (branch protection, PR requirements)
   - CODEOWNERS (mandatory code reviews)

2. **Workflow Level** (CI/CD):
   - Super-Linter (code quality)
   - OPA/Conftest (policy validation)
   - SLSA Provenance (supply chain attestations)
   - OpenSSF Scorecard (security health)

3. **Continuous Monitoring**:
   - Renovate (dependency updates every 3 hours)
   - Allstar (5 security policies, issue creation)

---

## 🎯 Governed Services

**11 services cataloged in Backstage**:

1. **SimCore** - React TypeScript frontend
2. **Repz** - Node.js backend
3. **BenchBarrier** - Performance monitoring
4. **Attributa** - Attribution system
5. **Mag-Logic** - Python logic engine
6. **Custom-Exporters** - Prometheus exporters
7. **Infra** - Core platform infrastructure
8. **AI-Agent-Demo** - Express API demonstration
9. **API-Gateway** - Advanced gateway with authentication
10. **Dashboard** - React TypeScript UI
11. **Healthcare** - HIPAA-compliant system

View complete service catalog: `.metaHub/backstage/catalog-info.yaml`

---

## 📈 Key Metrics

### Current Targets

- **OpenSSF Scorecard**: 8+/10 (baseline pending first run)
- **Renovate auto-merge rate**: >70%
- **Allstar open issues**: 0
- **PR merge time**: <24 hours
- **Policy violations**: 0/month

### Monitoring

```bash
# Security score (weekly)
gh run list --workflow=scorecard.yml --limit 1

# Dependency updates (daily)
gh pr list --label dependencies

# Security violations (daily)
gh issue list --label allstar

# Workflow health (daily)
gh run list --status failure --limit 5
```

---

## 🆘 Getting Help

### Common Commands

```bash
# Check governance status
gh run list --limit 5

# Test OPA policies locally
conftest test --policy .metaHub/policies/ <file>

# Run Super-Linter locally
docker run -e RUN_LOCAL=true -v $(pwd):/tmp/lint github/super-linter:latest

# View Scorecard results
gh run list --workflow=scorecard.yml --limit 1
```

### Resources

- **Documentation**: `.metaHub/` directory (11 guides)
- **Troubleshooting**: `.metaHub/TROUBLESHOOTING.md`
- **Quick Reference**: `.metaHub/QUICK_REFERENCE.md` (printable card)

### Community

- GitHub Community: <https://github.community/>
- OpenSSF: <https://openssf.org/>
- Backstage: <https://backstage.io/>

---

## 📋 Related Links

- [Security Policy](SECURITY.md) - Vulnerability reporting
- [License](LICENSE) - Repository license
- [GitHub Rulesets](https://github.com/alaweimm90/alaweimm90/settings/rules) - Branch protection config
- [GitHub Actions](https://github.com/alaweimm90/alaweimm90/actions) - Workflow runs

---

## 🤝 Contributing

This is a meta governance repository. Contributions should focus on:

- Improving governance policies (`.metaHub/policies/`)
- Enhancing workflows (`.github/workflows/`)
- Updating documentation (`.metaHub/`)
- Adding security checks
- Refining Backstage catalog

**All changes require**:
- CODEOWNERS approval (@alaweimm90)
- All status checks passing (Super-Linter, OPA, Scorecard)
- GitHub Rulesets enforcement (1 approval minimum)

---

## 📊 Status

**Repository State**: Clean slate ✨
- 8/10 governance tools active (80%)
- 1 manual setup remaining (Allstar - 10 min)
- 5 workflow files (governance only)
- 11 comprehensive documentation guides
- 0 open issues
- 0 open PRs

**Last Updated**: 2025-11-25
**Maintainer**: @alaweimm90
