# 📁 Repository Structure Guide

**Version**: 1.0
**Last Updated**: November 24, 2025
**Status**: ✅ Complete and Organized

---

## 🎯 Quick Overview

This monorepo contains 6 core packages, comprehensive documentation, automation infrastructure, and development tools.

```
github-monorepo/
├── 📚 Documentation (docs/ + root-level guides)
├── 📦 Packages (6 core packages)
├── 🔧 Infrastructure (turbo, pnpm, build tools)
├── ⚙️ Configuration (.config/, config/)
├── 🧪 Testing & Quality (tests/, jest.config.js)
├── 🚀 Deployment & Scripts (scripts/, deployment configs)
└── 🎨 Assets & Resources (templates/, assets/)
```

---

## 📂 Directory Structure (Detailed)

### Root Level
```
github-monorepo/
│
├── .config/                          # Configuration management
│   ├── archives/                     # Archived configurations
│   ├── claude/                       # Claude Code config
│   ├── governance/                   # Governance configs
│   ├── knowledge/                    # Knowledge base config
│   ├── meta/                         # Metadata and docs
│   ├── metaHub/                      # Meta hub infrastructure
│   ├── organizations/                # Organization configs
│   └── vscode/                       # VS Code settings
│
├── .github/                          # GitHub configuration
│   ├── workflows/                    # CI/CD workflows (40+ workflows)
│   ├── CODEOWNERS                    # Code ownership
│   └── dependabot.yml                # Dependency management
│
├── .tools/                           # Development tools
│   ├── automation/                   # Automation framework
│   ├── dev-tools/                    # IDE & dev tools
│   ├── review-auto-approve.ps1       # Auto-approval script
│   └── claude-bridge.ps1             # Claude integration
│
├── .migration_*                      # Migration tracking files
├── .env.example                      # Environment template
├── .eslintrc.json                    # ESLint config
├── .gitattributes                    # Git attributes
├── .gitignore                        # Git ignore patterns
├── .husky/                           # Git hooks
├── .lintstagedrc.json                # Lint-staged config
├── .pieces.config.json               # Pieces OS config
├── .pre-commit-config.yaml           # Pre-commit hooks
├── .prettierrc.json                  # Prettier config
├── .turbo/                           # Turbo cache
│
├── 📚 Documentation (ROOT LEVEL)
│   ├── START_HERE.md                 # ENTRY POINT
│   ├── FINAL_AGGRESSIVE_OPTIMIZATION_SUMMARY.md  # Main reference
│   ├── MASTER_OPTIMIZATION_PLAN_50_STEPS.md      # Optimization plan
│   ├── MONOREPO_ANALYSIS_SUMMARY.md  # Architecture
│   ├── GETTING_STARTED.md            # Setup guide
│   └── ... (29 total .md files)
│
├── alaweimm90/                       # Organization workspace
│
├── apps/                             # Application templates
│   ├── web/
│   ├── mobile/
│   └── ...
│
├── automation/                       # Automation scripts
│
├── config/                           # Configuration files
│   ├── jest.config.js
│   ├── pnpm-workspace.yaml
│   ├── turbo.json
│   └── ...
│
├── coverage/                         # Test coverage reports
│
├── docs/                             # PRIMARY DOCUMENTATION
│   ├── DOCUMENTATION_INDEX.md        # Main index
│   ├── guides/                       # How-to guides
│   ├── references/                   # Reference docs
│   ├── architecture/                 # Architecture docs
│   ├── setup/                        # Setup instructions
│   └── ... (21 docs)
│
├── jest.config.js                    # Jest configuration
│
├── node_modules/                     # Dependencies (in .gitignore)
│
├── openapi/                          # OpenAPI specifications
│   └── ... (API specs)
│
├── 📦 PACKAGES (6 Core Packages)
│   ├── packages/agent-core/
│   ├── packages/context-provider/
│   ├── packages/issue-library/
│   ├── packages/mcp-core/
│   ├── packages/shared-utils/
│   └── packages/workflow-templates/
│
├── package.json                      # Root package manifest
├── package-lock.json                 # Lock file
│
├── pnpm-workspace.yaml              # pnpm workspace config
│
├── reports/                          # Generated reports
│   └── ... (optimization reports)
│
├── 🔧 SCRIPTS
│   ├── scripts/build/                # Build scripts
│   ├── scripts/deploy/               # Deployment scripts
│   ├── scripts/maintenance/          # Maintenance scripts
│   ├── validate-monorepo.js          # Validation script
│   ├── standards-validator.js        # Standards checker
│   └── ... (various scripts)
│
├── src/                              # Source code
│   └── coaching-api/                 # Coaching API
│       ├── auth.ts
│       ├── data.ts
│       ├── risk.ts
│       ├── server.ts
│       └── types.ts
│
├── templates/                        # Project templates
│   ├── blog/
│   ├── e-commerce/
│   ├── landing-page/
│   ├── portfolio/
│   └── stationery/
│
├── 🧪 tests/                         # Test files
│   ├── standards-validator.test.js
│   └── ... (test files)
│
├── turbo.json                        # Turbo build config
├── tsconfig.json                     # TypeScript config
├── README_START_HERE.md              # Alternative entry point
│
└── .cache/                           # Cache directory (cleanup scheduled)
    └── backups-*/                    # Backup files
```

---

## 📦 Core Packages (packages/)

Each package follows this structure:

```
packages/{package-name}/
├── src/
│   ├── index.ts                      # Main entry point
│   ├── *.ts                          # Source files
│   └── __tests__/                    # Test files
├── package.json                      # Package manifest
├── tsconfig.json                     # TypeScript config
├── README.md                         # Package documentation
└── ...
```

### Package Purposes

| Package | Purpose | Status |
|---------|---------|--------|
| `agent-core` | Agent orchestration framework | ✅ Ready |
| `context-provider` | Context management utilities | ✅ Ready |
| `issue-library` | Issue templates and tools | ✅ Ready |
| `mcp-core` | Model Context Protocol implementation | ✅ Ready |
| `shared-utils` | **NEW** - Shared logging, errors, validation | ✅ Ready |
| `workflow-templates` | Workflow automation templates | ✅ Ready |

---

## 🔧 Configuration Files Location

### Primary Configurations
```
turbo.json                     # Turbo build system
pnpm-workspace.yaml            # pnpm workspace definition
tsconfig.json                  # TypeScript settings
jest.config.js                 # Testing framework
.eslintrc.json                 # Linting rules
.prettierrc.json               # Code formatting
```

### Directory-based Configurations
```
.config/
├── claude/                     # Claude Code settings
│   ├── agents.json
│   ├── mcp-config.json
│   ├── orchestration.json
│   └── workflows/
├── metaHub/
│   ├── governance/
│   ├── routing-templates/
│   └── compliance-templates/
└── vscode/                     # VS Code settings
    ├── extensions.json
    ├── settings.json
    ├── keybindings.json
    └── tasks.json
```

### GitHub Configuration
```
.github/
├── workflows/                  # CI/CD pipelines (40+)
│   ├── ci.yml
│   ├── security-*.yml
│   ├── docs-*.yml
│   └── ...
└── CODEOWNERS                  # Code ownership rules
```

---

## 📚 Documentation Architecture

### Hierarchy
1. **START_HERE.md** (Entry point)
2. **Quick Starts** (GETTING_STARTED.md, QUICKSTART_MCP.md)
3. **Main References** (MONOREPO_ANALYSIS_SUMMARY.md, FINAL_AGGRESSIVE_OPTIMIZATION_SUMMARY.md)
4. **Detailed Guides** (in docs/ subdirectories)
5. **References** (CONFIG_REFERENCE.md, SCRIPTS_REFERENCE.md)

### Documentation Organization
```
docs/
├── DOCUMENTATION_INDEX.md      # THIS FILE - master index
├── README.md                   # Docs overview
├── QUICK_START.md
├── ARCHITECTURE.md
├── DEVELOPER_GUIDE.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── SECURITY.md
├── guides/                     # How-to guides
├── references/                 # Reference material
├── architecture/               # Architecture docs
└── setup/                      # Setup guides
```

---

## 🎯 What Goes Where?

| Content | Location | Example |
|---------|----------|---------|
| **Entry points** | Root level | START_HERE.md |
| **Major guides** | Root level | FINAL_AGGRESSIVE_OPTIMIZATION_SUMMARY.md |
| **Quick starts** | Root or docs/ | GETTING_STARTED.md |
| **Implementation docs** | Root level | MONOREPO_ANALYSIS_SUMMARY.md |
| **Architecture** | docs/architecture/ | ARCHITECTURE.md |
| **API docs** | openapi/ | openapi/coaching-api.yaml |
| **Scripts** | scripts/ (organized by type) | scripts/build/, scripts/deploy/ |
| **Configuration** | .config/ or config/ | turbo.json, tsconfig.json |
| **Tests** | packages/{pkg}/src/__tests__/ | validation.test.ts |
| **Project templates** | templates/ | templates/blog/, templates/e-commerce/ |
| **Automation** | .tools/automation/ | orchestration, workflows |

---

## 🚀 Key Directories Explained

### .config/ - Configuration Management
- **claude/** - Claude Code and MCP configurations
- **metaHub/** - Meta hub infrastructure with governance
- **governance/** - Repository governance settings
- **vscode/** - VS Code workspace settings
- **meta/** - Metadata, architecture decisions, compliance

### .github/ - GitHub Integration
- **workflows/** - 40+ CI/CD workflows for:
  - Continuous integration
  - Security scanning
  - Documentation maintenance
  - Compliance checking
  - Deployment automation

### .tools/ - Development Tooling
- **automation/** - Automation framework with:
  - Agent orchestration
  - MCP module system
  - Workflow tools
  - Task automation
- **dev-tools/** - IDE integrations and configurations

### packages/ - Monorepo Core
- **6 core packages** with shared utilities
- Each package self-contained with tests
- pnpm workspace management
- Turbo build acceleration

### docs/ - Documentation Hub
- **Master documentation index**
- Organized into clear categories
- Cross-referenced and linked
- Searchable and navigable

### scripts/ - Development Scripts
- **Organized by purpose** (build, deploy, maintenance)
- All executable utilities in one place
- Documented in SCRIPTS_REFERENCE.md
- Integrated with package.json scripts

### src/ - Source Code
- **Coaching API** - Main application code
- API implementations
- Database models
- Service integrations

### templates/ - Reusable Templates
- **Project templates** (blog, e-commerce, portfolio, etc.)
- Starter templates for common use cases
- Fully documented and ready to use

---

## 📊 Size & Organization

| Category | Count | Size |
|----------|-------|------|
| **Markdown Docs** | 29 files | 28,795+ words |
| **CI/CD Workflows** | 40+ files | Comprehensive |
| **Core Packages** | 6 packages | Production-ready |
| **Configuration Files** | 15+ files | Well-organized |
| **Documentation** | 50+ pages | Complete coverage |
| **Test Suites** | 23 suites | Comprehensive |

---

## ✅ Organization Status

| Aspect | Status | Details |
|--------|--------|---------|
| Documentation | ✅ Organized | 29 docs with index |
| Directory Structure | ✅ Optimized | Clear hierarchy |
| Configuration | ✅ Centralized | .config/ + config/ |
| Scripts | ✅ Categorized | build/, deploy/, maintenance/ |
| Packages | ✅ Complete | 6 packages ready |
| Assets | ✅ Planned | assets/ directory created |
| Cache | ⏳ Cleanup scheduled | Phase 3 in progress |

---

## 🔄 Navigation Tips

### Finding Documentation
1. Start with [docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md)
2. Use the category index to find your topic
3. Follow cross-references as needed
4. Use CTRL+F to search within documents

### Finding Scripts
1. Check [docs/SCRIPTS_REFERENCE.md](docs/SCRIPTS_REFERENCE.md)
2. Look in scripts/{category}/ directory
3. Check package.json for npm script shortcuts
4. Run `npm run` to see all available scripts

### Finding Configuration
1. Check [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md)
2. Look in .config/ for Claude Code configs
3. Look in config/ for build/test configs
4. Check .github/workflows/ for CI/CD

### Finding Source Code
1. Check docs/ARCHITECTURE.md for overview
2. Look in src/ for application code
3. Look in packages/ for library code
4. Check packages/{name}/src/ for specific package

---

## 🎯 Next Steps

1. **Use this guide** to understand repository layout
2. **Refer to DOCUMENTATION_INDEX.md** for full documentation
3. **Check SCRIPTS_REFERENCE.md** for available scripts
4. **Read ARCHITECTURE.md** for technical details
5. **Visit START_HERE.md** for onboarding

---

**Status**: ✅ Complete and Organized
**Last Updated**: November 24, 2025
**Maintained By**: Claude Code + Team

*"A well-organized repository is the foundation for productive development."*
