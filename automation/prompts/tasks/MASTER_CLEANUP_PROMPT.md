# Repository Organization & Cleanup Master Prompt

A comprehensive, adaptable prompt for organizing and professionalizing any repository.

---

## Overview

This document provides a reusable prompt template for cleaning up, organizing, and professionalizing repositories of any type. Customize the sections relevant to your project.

---

## Master Prompt Template

Copy and modify this prompt for your specific repository:

```
I need to professionally organize and clean up my [REPOSITORY_TYPE] repository.

## Current Situation
- **Repository:** [PATH/URL]
- **Current Branch:** [BRANCH_NAME]
- **Main Technologies:** [TECH_STACK]
- **Team Size:** [SOLO/SMALL/MEDIUM/LARGE]
- **Current State:** [MESSY/PARTIALLY_ORGANIZED/MOSTLY_ORGANIZED]

## Goals
1. Make the project look professional and production-ready
2. Ensure it's nearly complete and expandable
3. Make it easy for new team members to understand
4. [ADD YOUR SPECIFIC GOALS]

## Scope of Work

### Structure & Organization
- [ ] Audit current directory structure
- [ ] Identify misplaced or orphaned files
- [ ] Detect naming inconsistencies
- [ ] Find duplicate files or content
- [ ] Clean up unnecessary/outdated files
- [ ] Reorganize into standard architecture
- [ ] Create missing organizational files

### Documentation
- [ ] Add README.md for root directory
- [ ] Add README.md for each major subdirectory
- [ ] Create STRUCTURE.md (directory guide)
- [ ] Create PROJECT.md (project overview)
- [ ] Create CONTRIBUTING.md (contribution guidelines)
- [ ] Create .gitignore (if missing)
- [ ] Update START_HERE.md or equivalent
- [ ] Fix all broken internal links

### Code Quality
- [ ] Add test structure if missing
- [ ] Ensure consistent naming conventions
- [ ] Document API endpoints/interfaces
- [ ] Add examples where helpful
- [ ] Identify and document dependencies

### Configuration Files
- [ ] Add/update .gitignore
- [ ] Add .editorconfig (if needed)
- [ ] Add configuration templates (.env.example)
- [ ] Add development setup instructions
- [ ] Add deployment configuration

### Git & Versioning
- [ ] Ensure clean commit history
- [ ] Update CHANGELOG
- [ ] Add proper version numbers
- [ ] Clean up old branches
- [ ] Document git workflow

## Project Type Specifics

### [SELECT: PYTHON / NODE.JS / MONOREPO / HYBRID / OTHER]
- Specific considerations for your tech stack
- Standard directory conventions
- Dependency management approach
- Testing framework expectations

## Expandability Requirements

The project should support:
- [ ] Easy addition of new [MODULES/PACKAGES/SERVICES]
- [ ] Clear template for new [PRODUCTS/FEATURES]
- [ ] Documented extension points
- [ ] Examples of how to extend

## Deliverables

After organization, the project should have:
1. ✅ Clean, standard directory structure
2. ✅ Professional documentation at every level
3. ✅ No orphaned, duplicate, or outdated files
4. ✅ Clear navigation and internal linking
5. ✅ Git best practices in place
6. ✅ Ready for team collaboration
7. ✅ Expandable for future growth

## Timeline & Commits
- Reorganize structure (1-2 commits)
- Add documentation (1 commit per category)
- Final professional polish (1 commit)
- All pushed to branch: [BRANCH_NAME]

Please:
1. Audit the current structure thoroughly
2. Identify all issues and inefficiencies
3. Create a detailed organization plan
4. Execute the plan systematically
5. Verify everything is cohesive and professional
6. Commit with clear messages
```

---

## Quick Reference: Customization Sections

### 1. For Different Repository Types

**Python Projects:**

```
- Add requirements.txt organization
- Add setup.py/pyproject.toml structure
- Include __pycache__ in .gitignore
- Add virtual environment directory structure
```

**Node.js Projects:**

```
- Add package.json organization
- Add node_modules structure
- Include .npm, .yarn in .gitignore
- Add Next.js/React specific directories
```

**Monorepos:**

```
- Organize into multiple workspaces
- Create shared dependencies structure
- Setup workspace scripts
- Document monorepo conventions
```

**Data Science Projects:**

```
- Organize data/ directory structure
- Add notebooks/ with examples
- Include models/ for trained models
- Document data pipeline
```

**Static Site / Documentation:**

```
- Organize content structure
- Create assets directory
- Setup build pipeline structure
- Document publishing process
```

### 2. For Different Team Sizes

**Solo Developer:**

```
Focus on:
- Clear self-documentation
- Expansion templates for future hires
- Easy handoff documentation
- Minimal unnecessary structure
```

**Small Team (2-5):**

```
Focus on:
- Collaboration guidelines
- Clear ownership sections
- Code review process
- Team communication structure
```

**Medium Team (5-20):**

```
Focus on:
- Role-based documentation
- Deployment pipelines
- Monitoring/logging setup
- Cross-team communication
```

**Large Team (20+):**

```
Focus on:
- Governance & standards
- Multi-team organization
- API contracts
- Dependency management
```

### 3. For Different Development Stages

**Early Stage / MVP:**

```
- Keep structure simple
- Focus on core functionality
- Minimal but essential documentation
- Room for future growth
```

**Growth Stage:**

```
- Expand structure for multiple features
- Add testing/CI-CD
- Implement standards
- Prepare for team scaling
```

**Mature / Production:**

```
- Complete documentation
- Strict standards & governance
- Comprehensive testing
- Monitoring & observability
```

---

## Standard Directory Structures by Type

### Modern Python Project

```
project-name/
├── README.md
├── PROJECT.md
├── STRUCTURE.md
├── CONTRIBUTING.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── setup.py / pyproject.toml
├── .env.example
├── src/
│   └── package_name/
│       ├── __init__.py
│       ├── main.py
│       └── modules/
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_*.py
├── docs/
│   ├── guides/
│   ├── api/
│   └── architecture/
├── scripts/
│   ├── setup.sh
│   └── run.sh
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── .github/
    └── workflows/
```

### Modern Node.js/React Project

```
project-name/
├── README.md
├── PROJECT.md
├── STRUCTURE.md
├── CONTRIBUTING.md
├── LICENSE
├── .gitignore
├── .env.example
├── package.json
├── tsconfig.json / jsconfig.json
├── src/
│   ├── index.js/ts
│   ├── components/
│   ├── pages/
│   ├── hooks/
│   ├── utils/
│   └── styles/
├── public/
├── tests/
│   ├── setup.js
│   └── *.test.js
├── docs/
│   ├── guides/
│   ├── api/
│   └── components/
├── scripts/
├── docker/
└── .github/
    └── workflows/
```

### Monorepo Structure

```
monorepo-name/
├── README.md
├── STRUCTURE.md
├── CONTRIBUTING.md
├── package.json (root)
├── lerna.json / pnpm-workspace.yaml
├── .gitignore
├── packages/
│   ├── core/
│   │   ├── README.md
│   │   ├── package.json
│   │   └── src/
│   ├── ui/
│   │   ├── README.md
│   │   ├── package.json
│   │   └── src/
│   ├── api/
│   │   ├── README.md
│   │   ├── package.json
│   │   └── src/
│   └── cli/
│       ├── README.md
│       ├── package.json
│       └── src/
├── shared/
│   ├── types/
│   ├── utils/
│   └── constants/
├── docs/
├── scripts/
└── .github/
```

---

## Documentation Checklist

### Root Level

- [ ] `README.md` - What is this project?
- [ ] `PROJECT.md` - Complete project overview
- [ ] `STRUCTURE.md` - Directory guide
- [ ] `CONTRIBUTING.md` - How to contribute
- [ ] `CHANGELOG.md` - Version history
- [ ] `LICENSE` - Legal information
- [ ] `.gitignore` - What to ignore
- [ ] `.env.example` - Environment template

### Each Major Directory

- [ ] `README.md` - What's in this directory?
- [ ] Navigation guide - How to find things?
- [ ] Examples - Real usage examples
- [ ] Related docs - Links to other sections

### Special Directories

- [ ] `/docs/guides/` - How-to guides
- [ ] `/docs/api/` - API documentation
- [ ] `/scripts/` - What each script does
- [ ] `/tests/` - How to run tests
- [ ] `/docker/` - Docker setup guide

---

## Common Issues & Solutions

### Issue: Outdated/Legacy Files

**Solution:**

```bash
# Identify
find . -name "*.old" -o -name "*.backup" -o -name "*.deprecated"

# Archive
mkdir .archive
mv old_files/* .archive/
git add .archive/
git commit -m "Archive outdated documentation"

# Document
Create .archive/README.md explaining what was archived and why
```

### Issue: Inconsistent Naming

**Solution:**

```
Establish naming conventions:
- Directories: lowercase-with-hyphens
- Files: UPPERCASE_WITH_UNDERSCORES.md (docs)
- Files: lowercase_with_underscores.py (code)
- Branches: feature/description or fix/description

Document in CONTRIBUTING.md
Update all files to follow convention
```

### Issue: Missing Documentation

**Solution:**

```
1. Create section headings in README files
2. Add navigation tables
3. Add "how to..." sections
4. Add examples for each section
5. Link related documents

Template for every README:
- What is this directory?
- What's inside?
- Quick start (if applicable)
- Related documentation
```

### Issue: Scattered Configuration Files

**Solution:**

```
1. Consolidate into /config/ or root
2. Add .env.example template
3. Document each config setting
4. Add setup instructions
5. Ignore actual .env files in .gitignore
```

### Issue: No Clear Expansion Path

**Solution:**

```
Add to PROJECT.md or STRUCTURE.md:
- How to add a new [feature/module/product]
- Template to copy for new items
- Checklist for expansion
- Examples of successful expansion

Create template directories for:
- new-module-template/
- new-component-template/
- etc.
```

---

## Implementation Steps

### Step 1: Audit (30 min)

```bash
1. List all files by directory
2. Identify duplicates (find . -type f -name "*" | sort)
3. Check for outdated content
4. Review current naming patterns
5. Look for orphaned files
```

### Step 2: Plan (30 min)

```
Document:
1. What will change and why
2. New directory structure
3. Files to move/rename/delete
4. Documentation to add
5. Commit strategy
```

### Step 3: Cleanup (1-2 hours)

```
1. Remove/archive outdated files
2. Reorganize directories
3. Rename files for consistency
4. Update internal links
5. Verify no broken references
```

### Step 4: Document (1-2 hours)

```
1. Create PROJECT.md
2. Create STRUCTURE.md
3. Create CONTRIBUTING.md
4. Add README to each directory
5. Update root documentation
```

### Step 5: Polish (30 min)

```
1. Fix all links
2. Add navigation helpers
3. Create examples
4. Add quick-start guides
5. Final review
```

### Step 6: Commit & Push (15 min)

```
1. Add all changes
2. Commit with clear message
3. Push to branch
4. Verify on remote
```

---

## Git Workflow Template

### Commits Suggested

```
1. "Clean up: Remove outdated files and archive legacy content"
2. "Reorganize: Restructure from [old] to standard [new] architecture"
3. "Document: Add comprehensive README and guide files"
4. "Polish: Update links, fix navigation, final review"
```

### Commit Message Template

```
[Type] Brief description (50 chars max)

Detailed explanation (72 chars max per line)
- Specific change 1
- Specific change 2
- Why this change was made

Files affected:
- Moved: old_path/ → new_path/
- Created: new_file.md
- Deleted: old_file.md
- Updated: reference_links_in_docs/
```

---

## Verification Checklist

After organization, verify:

- [ ] No `.old`, `.backup`, `.deprecated` files
- [ ] Consistent naming throughout
- [ ] No broken internal links
- [ ] README in every major directory
- [ ] Clear navigation structure
- [ ] All configuration files organized
- [ ] .gitignore is comprehensive
- [ ] No unnecessary duplicate files
- [ ] Documentation is up-to-date
- [ ] Ready for team collaboration
- [ ] Expandable with clear templates
- [ ] Git history is clean
- [ ] All files are findable

---

## Result: Professional Project Characteristics

✅ **Structure**

- Standard directories
- Clear organization
- Consistent naming
- No orphaned files

✅ **Documentation**

- README at every level
- Clear navigation
- Examples provided
- Internal links work

✅ **Professionalism**

- Modern conventions
- No legacy artifacts
- Well-organized
- Easy to understand

✅ **Expandability**

- Clear template structure
- Documentation for extending
- Examples of how to add
- Organized for growth

✅ **Collaboration**

- Contributing guidelines
- Clear workflows
- Git best practices
- Team-ready

---

## Customization Examples

### Example 1: Python FastAPI Project

```
I need to professionally organize my Python FastAPI project.

## Current Situation
- Repository: /home/user/my-api
- Technologies: Python 3.10, FastAPI, PostgreSQL, Docker
- Team Size: Solo (planning to hire 1-2 devs)
- Current State: Somewhat messy with legacy structure

## Specific Goals
- Migrate from old async/await structure to modern FastAPI
- Organize tests properly with pytest
- Setup Docker for local development
- Create clear API documentation
- Document database migrations with Alembic

## Type: PYTHON

Focus on:
- Moving code from app/ to src/package_name/
- Organizing tests with conftest.py
- Database migration management
- Virtual environment structure
- Dependency management (requirements.txt → pyproject.toml)
```

### Example 2: React Monorepo

```
I need to organize my React monorepo with multiple apps.

## Current Situation
- Repository: /home/user/web-platform
- Technologies: React, TypeScript, Node.js, pnpm workspaces
- Team Size: Small team (4 devs)
- Current State: Growing but unorganized

## Specific Goals
- Setup pnpm workspaces properly
- Organize into clear packages (web, mobile, api, shared)
- Setup shared component library
- Document build and deploy process
- Create clear development guidelines for team

## Type: MONOREPO + NODE.JS

Focus on:
- Multi-package organization
- Shared dependencies
- Build pipeline structure
- Team collaboration setup
- Clear ownership boundaries
```

### Example 3: Data Science Project

```
I need to organize my data science research project.

## Current Situation
- Repository: /home/user/ml-research
- Technologies: Python, Jupyter, PyTorch, Pandas
- Team Size: Solo researcher
- Current State: Notebooks everywhere

## Specific Goals
- Organize notebooks by experiment
- Setup reproducible environment
- Document data pipeline
- Version control for models
- Clear experiment tracking

## Type: PYTHON (Data Science)

Focus on:
- Data directory organization
- Notebooks/ structure
- Models/ for trained weights
- Results/ for outputs
- Reproducibility (requirements.txt, random seeds)
```

---

## Resources & Templates

### Files to Create

**PROJECT.md Template**

```markdown
# Project Name

Brief description of what the project does.

## Quick Facts

- Technology: [tech stack]
- Team size: [size]
- Status: [status]
- Timeline: [timeline]

## Getting Started

[Quick start instructions]

## Documentation

[Link to docs]

## Contributing

[Link to CONTRIBUTING.md]
```

**STRUCTURE.md Template**

```markdown
# Project Structure

## Overview

[Description of directory organization]

## Directories

[Table or list of all major directories]

## How to Navigate

[Guide for finding things]

## Adding New [Features/Modules]

[How to extend the structure]
```

**CONTRIBUTING.md Template**

```markdown
# Contributing Guide

## Getting Started

[Setup instructions]

## Development Workflow

[Branch naming, commit conventions]

## Submitting Changes

[PR process]

## Code Standards

[Style guide, testing requirements]

## Questions?

[Where to ask for help]
```

---

## Final Notes

- **Adapt the prompt** to your specific technology and needs
- **Start small** - don't try to fix everything at once
- **Document as you go** - create guides while reorganizing
- **Commit frequently** - use atomic, meaningful commits
- **Get feedback** - ask teammates if structure makes sense
- **Iterate** - first pass doesn't have to be perfect

---

**Use this prompt as a starting point for your repository organization!** 📚

Modify sections based on your project type and team needs.
