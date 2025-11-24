# metaHub

> Central coordination hub for monorepo ecosystems

[![Organizations](https://img.shields.io/badge/Organizations-Multiple-blue?style=flat-square)](#organizations)
[![Projects](https://img.shields.io/badge/Projects-Many-green?style=flat-square)](#projects)
[![Compliance](https://img.shields.io/badge/Compliance-Tracked-yellow?style=flat-square)](./docs/standards/compliance.md)

**Purpose:** Coordination hub and documentation center for all organizations
**Owner:** Your Organization
**Repository:** Part of your monorepo

---

## 🎯 What is metaHub?

metaHub is the **central coordination point** for monorepo ecosystems. It provides:

- **📚 Unified Documentation** - All documentation in one place
- **🔧 Shared Tools & Scripts** - Automation across organizations
- **📋 Standards & Templates** - Consistent quality everywhere
- **🤝 Coordination** - How organizations work together
- **📊 Monitoring** - Compliance tracking and reporting

This directory does NOT contain code. It's purely for **coordination, documentation, and tooling**.

---

## 🏢 Organizations

Your monorepo can contain multiple organizations:

| Organization            | Purpose                   | Compliance | Projects | Docs                                                |
| ----------------------- | ------------------------- | ---------- | -------- | --------------------------------------------------- |
| **org-alpha**           | Example organization      | 70%        | 6        | [View](./docs/organizations/org-alpha.md)           |
| **org-beta**            | Example organization      | 65%        | 8        | [View](./docs/organizations/org-beta.md)            |
| **org-gamma**           | Example organization      | 75%        | 10       | [View](./docs/organizations/org-gamma.md)           |

**Total:** Customize based on your organizations and projects

---

## 🚀 Quick Start

### For Developers

```bash
# 1. Start with the documentation
cat metaHub/docs/getting-started/overview.md

# 2. Choose an organization
ls -la org-alpha/ org-beta/ org-gamma/

# 3. Setup your environment
./metaHub/scripts/setup/install-dependencies.sh

# 4. Verify setup
./metaHub/scripts/setup/verify-environment.sh
```

### For AI Assistants (Claude Code, etc.)

**Quick Start:**

1. **First**: Read [CLAUDE_CODE_GUIDE.md](./CLAUDE_CODE_GUIDE.md) - Quick orientation
2. **Then**: Read [GITHUB_STRUCTURE.md](./GITHUB_STRUCTURE.md) - Full repository structure

**Context Loading Order for Projects:**

1. Read this metaHub/README.md
2. Read [GITHUB_STRUCTURE.md](./GITHUB_STRUCTURE.md) - Understand monorepo structure
3. Read the organization README (e.g., AlaweinOS/README.md)
4. Read the organization SUPERPROMPT (e.g., AlaweinOS/ALAWEIN_OS_SUPERPROMPT.md)
5. Read project CLAUDE.md (e.g., AlaweinOS/MEZAN/CLAUDE.md)
6. Read compliance reports for specific issues

**Example for AlaweinOS/MEZAN:**

```bash
# Full context stack:
metaHub/CLAUDE_CODE_GUIDE.md                # Quick orientation (start here!)
metaHub/GITHUB_STRUCTURE.md                 # Understand the structure
metaHub/README.md                           # Monorepo overview
AlaweinOS/README.md                         # Organization overview
AlaweinOS/ALAWEIN_OS_SUPERPROMPT.md         # Comprehensive org context
AlaweinOS/MEZAN/CLAUDE.md                   # Project-specific context
AlaweinOS/MEZAN/README.md                   # Project overview
```

---

## 📚 Documentation

### For Claude Code Users 🤖

- **[CLAUDE_CODE_GUIDE.md](./CLAUDE_CODE_GUIDE.md)** - Quick orientation for Claude Code
- **[GITHUB_STRUCTURE.md](./GITHUB_STRUCTURE.md)** - Complete repository structure with diagrams
- **[TWO_VERSION_STRATEGY.md](./TWO_VERSION_STRATEGY.md)** - Managing public SaaS and private dev versions

### Getting Started

- [Overview](./docs/getting-started/overview.md) - What is this monorepo?
- [Setup Guide](./docs/getting-started/setup.md) - Development environment setup
- [First Contribution](./docs/getting-started/first-contribution.md) - Make your first PR

### Organizations

- [AlaweinOS](./docs/organizations/AlaweinOS.md) - Optimization & quantum ML
- [org-alpha](./docs/organizations/org-alpha.md) - Example organization
- [org-beta](./docs/organizations/org-beta.md) - Example organization
- [org-gamma](./docs/organizations/org-gamma.md) - Example organization
- [Project Index](./docs/organizations/project-index.md) - All projects
- **[GitHub Profile Repository](./docs/organizations/GITHUB_PROFILE_REPO.md)** - Your profile repository

### Standards & Guidelines

- [Repository Standards](./docs/standards/repo-standards.md) - Coding standards (links to ../REPO_STANDARDS.md)
- [Compliance Guide](./docs/standards/compliance.md) - How to achieve 95%+
- [Testing Standards](./docs/standards/testing.md) - Testing requirements
- [Security Guidelines](./docs/standards/security.md) - Security practices

### Architecture

- [Monorepo Design](./docs/architecture/monorepo-design.md) - Why monorepo?
- [Project Relationships](./docs/architecture/project-relationships.md) - How projects connect
- [Tech Stack](./docs/architecture/tech-stack.md) - Technologies used

---

## 🛠️ Tools & Scripts

### Setup Scripts

```bash
./scripts/setup/install-dependencies.sh    # Install all dependencies
./scripts/setup/setup-pre-commit.sh        # Setup pre-commit hooks
./scripts/setup/verify-environment.sh      # Verify environment
```

### Compliance Tools

```bash
./scripts/compliance/check-all-orgs.sh     # Check compliance across all orgs
./scripts/compliance/generate-report.sh    # Generate compliance reports
./scripts/compliance/fix-common-issues.sh  # Auto-fix simple issues
```

### Testing Tools

```bash
./scripts/testing/run-all-tests.sh         # Run all tests
./scripts/testing/check-coverage.sh        # Generate coverage reports
```

### Utility Scripts

```bash
./scripts/utils/update-readmes.sh          # Update all READMEs
./scripts/utils/sync-standards.sh          # Sync REPO_STANDARDS across orgs
```

---

## 📋 Templates

Ready-to-use templates for creating new projects:

- **[Python Project](./templates/python-project/)** - Fully configured Python project with pyproject.toml, tests, CI/CD
- **[TypeScript Project](./templates/typescript-project/)** - Fully configured TS project with tsconfig, ESLint, Prettier
- **[Organization Template](./templates/organization/)** - Create a new organization with all mandatory files

**Usage:**

```bash
# Create new Python project
cp -r metaHub/templates/python-project/ AlaweinOS/my-new-project/
cd AlaweinOS/my-new-project/
# Customize and start developing
```

---

## 📊 Compliance Tracking

**Current Status (November 2025):**

- **Overall Compliance:** 60.6% (Target: 95%+)
- **Critical Violations:** 51 across all organizations
- **Estimated Remediation:** 120-165 hours total

**Top Issues:**

1. 18 projects missing LICENSE files (43%) - LEGAL BLOCKER
2. 23 projects missing SECURITY.md (55%) - SECURITY RISK
3. 15 projects with missing/broken .gitignore (36%) - CREDENTIAL RISK

[View Detailed Compliance Report](../MONOREPO_COMPLIANCE_REPORT_2025-11-20.md)

**Compliance by Organization:**

- org-alpha: 70% (C)
- org-beta: 65% (D+)
- org-gamma: 75% (C+)

---

## 🏗️ Directory Structure

```
metaHub/
├── README.md                    # This file
├── CLAUDE.md                    # AI assistant context
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore rules
│
├── docs/                        # Documentation hub
│   ├── getting-started/         # Onboarding guides
│   ├── organizations/           # Organization overviews
│   ├── standards/               # Standards and guidelines
│   ├── architecture/            # Architecture documentation
│   └── archive/                 # Historical docs (read-only)
│
├── scripts/                     # Automation scripts
│   ├── setup/                   # Environment setup
│   ├── compliance/              # Compliance checking
│   ├── testing/                 # Test automation
│   └── utils/                   # Utility scripts
│
├── config/                      # Shared configurations
│   ├── .pre-commit-config.yaml  # Pre-commit template
│   ├── .eslintrc.base.js        # ESLint base config
│   ├── .prettierrc.base.json    # Prettier base config
│   ├── pyproject.base.toml      # Python base config
│   └── tsconfig.base.json       # TypeScript base config
│
├── templates/                   # Project templates
│   ├── python-project/          # Python project template
│   ├── typescript-project/      # TypeScript template
│   └── organization/            # Organization template
│
├── tools/                       # Development tools
│   ├── compliance-checker/      # Compliance automation
│   ├── readme-generator/        # README generation
│   └── superprompt-builder/     # Superprompt tools
│
└── .github/                     # GitHub configuration
    ├── ISSUE_TEMPLATE/          # Issue templates
    ├── PULL_REQUEST_TEMPLATE.md # PR template
    └── workflows/               # GitHub Actions
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for contribution guidelines.

**Quick Workflow:**

1. Read the organization README for context
2. Read the project SUPERPROMPT for detailed guidance
3. Check compliance reports for known issues
4. Make your changes following REPO_STANDARDS.md
5. Run tests and linting
6. Submit PR with conventional commit messages

---

## 📞 Contact

**Owner:** Your Name
**Email:** contact@example.com
**Website:** <https://example.com>
**GitHub:** [@your-username](https://github.com/your-username)
**Location:** Your Location

**For Issues:** Open an issue in the organizations repository
**For Questions:** See organization-specific documentation

---

## 📄 License

MIT License - See [LICENSE](./LICENSE) for details

Copyright © 2024-2025 Your Organization

---

## 🗺️ Navigation

**From Here:**

- **Browse Organizations:** See [docs/organizations/](./docs/organizations/)
- **Check Compliance:** See [docs/standards/compliance.md](./docs/standards/compliance.md)
- **Use Templates:** See [templates/](./templates/)
- **Run Scripts:** See [scripts/](./scripts/)

**To Organizations:**

- [../AlaweinOS/](../AlaweinOS/) - Optimization & quantum ML
- [../org-alpha/](../org-alpha/) - Example organization
- [../org-beta/](../org-beta/) - Example organization
- [../org-gamma/](../org-gamma/) - Example organization

**To Standards:**

- [../REPO_STANDARDS.md](../REPO_STANDARDS.md) - Repository standards
- [../MONOREPO_COMPLIANCE_REPORT_2025-11-20.md](../MONOREPO_COMPLIANCE_REPORT_2025-11-20.md) - Compliance report
- [../CRITICAL_VIOLATIONS_PRIORITY_ACTION_PLAN.md](../CRITICAL_VIOLATIONS_PRIORITY_ACTION_PLAN.md) - Priority fixes

---

<div align="center">

**Part of your monorepo**

metaHub © 2024-2025 • Coordinating excellence across 5 organizations

</div>
