# Project Structure Guide

Complete documentation of the ORCHEX project directory layout and organization.

## Overview

```
Foundry/
├── Root Documentation
├── docs/          (Complete strategic documentation)
├── products/      (4 production-ready code templates)
├── templates/     (Marketing & metrics templates)
├── scripts/       (Automation & scaffolding)
└── .archive/      (Outdated documentation)
```

---

## Root Level (7 files)

**Core guides that every user should read:**

| File | Purpose | Read Time | Audience |
|------|---------|-----------|----------|
| `START_HERE.md` | Master navigation & decision path | 10 min | Everyone |
| `README.md` | Original 10 product ideas | 20 min | Idea researchers |
| `100_STEP_ROADMAP.md` | Complete validation→build→scale plan | 30 min | Strategic planners |
| `QUICK_DECISIONS.md` | Fast decision framework | 10 min | Decision makers |
| `IMPLEMENTATION_GUIDE.md` | How to build after validation | 30 min | Builders |
| `FAQ.md` | 40+ common questions | Variable | Troubleshooters |
| `CHANGELOG.md` | Version history & changes | 10 min | Contributors |
| `PROJECT.md` | This project's overview | 5 min | New users |
| `STRUCTURE.md` | This file (directory guide) | 5 min | Navigation |
| `CONTRIBUTING.md` | Development guidelines | 10 min | Contributors |

### 📍 Location
```
/home/user/Foundry/
├── START_HERE.md ⭐
├── README.md
├── 100_STEP_ROADMAP.md
├── QUICK_DECISIONS.md
├── IMPLEMENTATION_GUIDE.md
├── FAQ.md
├── CHANGELOG.md
├── PROJECT.md
├── STRUCTURE.md
└── CONTRIBUTING.md (to create)
```

### 🎯 Usage
- **New users:** Start with `START_HERE.md`
- **Fast decision:** Read `QUICK_DECISIONS.md`
- **Questions:** Search `FAQ.md`
- **Strategic overview:** Read `PROJECT.md`

---

## /docs/ Directory (15 files)

**All strategic documentation, analysis, and planning materials.**

### Structure

```
docs/
├── guides/            # Execution guides (8 files)
├── analysis/          # Financial & market analysis (2 files)
├── audits/            # Strategic assessment (1 file)
├── legal/             # Compliance & legal (1 file)
├── roadmaps/          # Implementation plans (1 file)
├── visuals/           # Diagrams & flowcharts (1 file)
└── README.md          # This directory guide
```

### /docs/guides/ (8 files - EXECUTION GUIDES)

**How-to guides for executing each part of the framework:**

| File | Purpose | Timeline | Who |
|------|---------|----------|-----|
| `VALIDATION_FIRST.md` | 4-week validation sprint framework | 4 weeks | Everyone |
| `CUSTOMER_DEVELOPMENT.md` | User interview scripts & methodology | Ongoing | Validators |
| `WEEK_1_CHECKLIST.md` | Daily action items for Week 1 | Week 1 | Executors |
| `MARKETING_PLAYBOOK.md` | 20+ copy-paste marketing templates | Pre-launch | Marketers |
| `METRICS_DASHBOARD.md` | KPI tracking & analytics setup | Ongoing | Metrics owner |
| `TROUBLESHOOTING.md` | 12+ common problems & solutions | As needed | Troubleshooters |
| `PROMPT_OPTIMIZER.md` | Improve AI model outputs | Ongoing | AI operators |
| `GETTING_STARTED.md` | Developer environment setup | Setup only | Developers |

**📍 Location:** `/home/user/Foundry/docs/guides/`

### /docs/analysis/ (2 files - ANALYSIS & RESEARCH)

**Market analysis, financial projections, and priority ranking:**

| File | Purpose | Use Case |
|------|---------|----------|
| `PRIORITY_RANKING.md` | All 10 ideas scored 67-95/100 | Choose which product to build |
| `FINANCIAL_MODEL.md` | 24-month revenue projections | Understand economics |

**📍 Location:** `/home/user/Foundry/docs/analysis/`

### /docs/audits/ (1 file - STRATEGIC ASSESSMENT)

| File | Purpose |
|------|---------|
| `COMPREHENSIVE_AUDIT.md` | 85/100 grade assessment of original plan + recommendations |

**📍 Location:** `/home/user/Foundry/docs/audits/`

### /docs/legal/ (1 file - COMPLIANCE)

| File | Purpose |
|------|---------|
| `LEGAL_COMPLIANCE.md` | GDPR, Privacy Policy, Terms of Service, DPA, Security |

**📍 Location:** `/home/user/Foundry/docs/legal/`

### /docs/roadmaps/ (1 file - IMPLEMENTATION PLANS)

| File | Purpose | Timeline |
|------|---------|----------|
| `PHASE_1_QUICK_WINS.md` | Weeks 1-16 build plan for 3 products | 16 weeks |

**📍 Location:** `/home/user/Foundry/docs/roadmaps/`

### /docs/visuals/ (1 file - DIAGRAMS)

| File | Purpose |
|------|---------|
| `PRINTABLE_FLOWCHARTS.md` | Validation & build process diagrams |

**📍 Location:** `/home/user/Foundry/docs/visuals/`

### /docs/README.md (TO CREATE)

Navigation guide for the docs/ directory.

---

## /products/ Directory (12 files across 4 products)

**4 production-ready code templates for AI research tools.**

### Structure

```
products/
├── nightmare-mode/        # AI attacks papers (6 files)
├── chaos-engine/          # Hypothesis generator (2 files)
├── research-prison/       # Interrogation system (2 files)
├── ORCHEX-core/            # Shared infrastructure (1 file)
└── README.md              # This directory guide
```

### Product Template Structure

Each product follows this pattern:

```
product-name/
├── README.md                 # Product overview & quickstart
├── backend/                  # Python FastAPI application
│   ├── app/
│   ├── models/
│   └── tests/
├── frontend/                 # Next.js React application (if applicable)
├── alembic/                  # Database migrations
├── k8s/                      # Kubernetes deployment configs
├── .github/workflows/        # GitHub Actions CI/CD
├── docker-compose.yml        # Local development
├── requirements.txt          # Python dependencies
└── [other config files]
```

### Products Available

| Product | Files | Use Case | Timeline |
|---------|-------|----------|----------|
| **NIGHTMARE MODE** | 6 | AI attacks research papers from 6 angles | 12 weeks |
| **CHAOS ENGINE** | 2 | Cross-domain hypothesis generator | 10 weeks |
| **RESEARCH PRISON** | 2 | 200-question interrogation system | 8 weeks |
| **ORCHEX CORE** | 1 | Shared infrastructure & AI models | Foundational |

### /products/README.md (TO CREATE)

Quick guide for:
- Choosing which product to start with
- Understanding directory structure
- How to scaffold new products

---

## /templates/ Directory (2 files)

**Marketing & metrics templates for launch & tracking.**

### Structure

```
templates/
├── email/           # Email marketing templates (1 file)
├── metrics/         # Dashboard & KPI templates (1 file)
└── README.md        # This directory guide
```

### /templates/email/ (1 file - MARKETING COPY)

| File | Contents |
|------|----------|
| `ALL_EMAIL_TEMPLATES.md` | 20+ copy-paste email templates |

### /templates/metrics/ (1 file - ANALYTICS)

| File | Contents |
|------|----------|
| `METRICS_SPREADSHEETS.md` | Dashboard templates & KPI tracking |

### /templates/README.md (TO CREATE)

Guide for:
- Which templates to use when
- How to customize templates
- Where to add new templates

---

## /scripts/ Directory (2 files)

**Automation scripts for scaffolding new repositories.**

### Files

| File | Purpose |
|------|---------|
| `create-repo.sh` | Scaffolds a single product repository |
| `create-all-repos.sh` | Creates all 10 product repositories |

### Usage

```bash
# Create single repository
cd scripts/
./create-repo.sh nightmare-mode ORCHEX-nightmare-validator

# Create all repositories
./create-all-repos.sh
```

### /scripts/README.md (TO CREATE)

Documentation for:
- How to use scaffolding scripts
- What gets created
- Customizing scaffold output

---

## /.archive/ Directory (4 files - INTENTIONALLY ARCHIVED)

**Outdated documentation from previous versions.**

```
.archive/
├── 50_STEP_PLAN.md       # Unrealistic 50-step timeline (outdated)
├── EXECUTIVE_SUMMARY.md  # Consolidated into START_HERE.md
├── QUICK_START.md        # Outdated build-first approach
└── README.md             # Why these files were archived
```

### Purpose
- Reference for historical context
- Example of how NOT to plan
- Educational (what changed and why)

---

## File Organization Summary

| Directory | Files | Purpose | Type |
|-----------|-------|---------|------|
| `/` (root) | 10 | Core navigation & guides | Documentation |
| `/docs/` | 15 | Strategic planning & execution | Documentation |
| `/products/` | 12 | Production code templates | Code + Docs |
| `/templates/` | 2 | Marketing & metrics templates | Documentation |
| `/scripts/` | 2 | Repository scaffolding | Code |
| `/.archive/` | 4 | Outdated documentation | Documentation |

**Total:** 45 files, all organized and documented.

---

## How to Expand This Structure

### Adding a New Product

1. **Create directory:**
   ```bash
   mkdir -p products/new-product-name
   ```

2. **Copy template structure:**
   ```bash
   cp -r products/nightmare-mode/* products/new-product-name/
   ```

3. **Update README:**
   - Edit `products/new-product-name/README.md`
   - Update product description & use case
   - Update timeline & difficulty

4. **Add to priority list:**
   - Update `docs/analysis/PRIORITY_RANKING.md`
   - Score the new product
   - Add to roadmap if applicable

### Adding New Documentation

1. **Choose category:**
   - Execution guide → `docs/guides/`
   - Strategic analysis → `docs/analysis/`
   - Legal doc → `docs/legal/`
   - Roadmap → `docs/roadmaps/`
   - Other → create new subdir

2. **Create file:**
   ```bash
   touch docs/guides/NEW_GUIDE.md
   ```

3. **Update navigation:**
   - Add link to `START_HERE.md`
   - Add link to parent `README.md`
   - Update this `STRUCTURE.md` file

4. **Ensure consistency:**
   - Use UPPERCASE_WITH_UNDERSCORES.md naming
   - Add table of contents (H2 headers)
   - Include read time estimate

---

## Navigation Guide

### If you want to...

| Goal | Start Here | Then Read |
|------|-----------|-----------|
| **Validate an idea** | `START_HERE.md` | `docs/guides/VALIDATION_FIRST.md` |
| **Build a product** | `docs/analysis/PRIORITY_RANKING.md` | `products/nightmare-mode/README.md` |
| **Understand the business** | `PROJECT.md` | `docs/analysis/FINANCIAL_MODEL.md` |
| **Handle legal issues** | `docs/legal/LEGAL_COMPLIANCE.md` | `FAQ.md` |
| **Fix a problem** | `docs/guides/TROUBLESHOOTING.md` | `FAQ.md` |
| **Contribute code** | `CONTRIBUTING.md` | `products/README.md` |
| **Market your product** | `docs/guides/MARKETING_PLAYBOOK.md` | `templates/email/` |

---

## Quick Reference

### Most Important Files
1. `START_HERE.md` - Everyone starts here
2. `docs/guides/VALIDATION_FIRST.md` - Core methodology
3. `PROJECT.md` - Project overview
4. `STRUCTURE.md` - This file (you are here)

### Most Used Directories
1. `/docs/guides/` - Execution playbooks
2. `/products/` - Code templates
3. `/templates/` - Marketing copy

### Most Updated Sections
1. `CHANGELOG.md` - Track changes
2. `docs/analysis/PRIORITY_RANKING.md` - Update scores
3. `/products/*/README.md` - Product-specific docs

---

**Questions? Check `FAQ.md` or `CONTRIBUTING.md`** 📚
