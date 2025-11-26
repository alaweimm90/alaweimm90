# alaweimm90 GitHub Operating System

Enterprise-grade governance system for 55+ repositories across 5 organizations.

## Quick Navigation

- **START_HERE.md** - System overview (15 min read)
- **ARCHITECTURE.md** - System design and architecture
- **docs/guides/QUICK_START.md** - Implementation guide
- **organizations/README.md** - Project portfolio

## System Status

All 3-layer enforcement active:

- Layer 1: Local Pre-commit ✓ (Markdown, Python linting)
- Layer 2: CI/CD Automation ✓ (GitHub Actions workflows)
- Layer 3: Project Governance ✓ (OPA policies, catalog)

## Key Features

- Pre-commit hooks for local validation
- Automated CI/CD pipelines for Python and TypeScript
- Project catalog system with manifest validation
- Governance file templates and synchronization
- 3 valid projects currently tracked

## Getting Started

```bash
# Install pre-commit hooks
pre-commit install

# Run validation
pre-commit run --all-files

# Scan projects
python metaHub/cli/meta.py scan-projects
```

## Repository Structure

```text
GitHub-alaweimm90/
├── START_HERE.md              # Main entry point
├── ARCHITECTURE.md            # System design
├── .github/workflows/         # CI/CD automation
├── .meta/repo.yaml            # Repository metadata
├── docs/                      # Documentation
├── organizations/             # 55+ projects
├── metaHub/                   # Governance tools
│   ├── cli/                   # Command-line tools
│   ├── policies/              # OPA policy rules
│   ├── templates/             # File templates
│   └── linters/               # Linter configs
├── scripts/                   # Utility scripts
├── .pre-commit-config.yaml    # Pre-commit config
├── .gitignore                 # Git ignore rules
└── inventory.json             # Repository manifest
```

## Documentation

- **START_HERE.md** - Complete system overview
- **ARCHITECTURE.md** - Technical architecture
- **docs/guides/** - Implementation guides
- **metaHub/cli/** - Tool documentation

## Support

For questions or issues, see START_HERE.md or ARCHITECTURE.md.

Last updated: November 26, 2025
