# Contributing to metaHub

Thank you for considering contributing to metaHub! This document provides guidelines and instructions for contributing to the central coordination hub for monorepo ecosystems.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [What is metaHub?](#what-is-metahub)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Documentation Standards](#documentation-standards)
- [Adding Scripts](#adding-scripts)
- [Adding Templates](#adding-templates)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the repository maintainer.

## What is metaHub?

metaHub is the **central coordination point** for monorepo ecosystems. It contains:

- **Documentation** - Guides for all organizations and projects
- **Scripts** - Automation tools for setup, compliance, and testing
- **Templates** - Starter templates for new projects and organizations
- **Configuration** - Shared base configurations and standards
- **Tools** - Development utilities for the ecosystem

**metaHub does NOT contain production code.** It's purely for coordination, documentation, and tooling.

## Getting Started

### Prerequisites

- **Git**: For version control
- **Text Editor**: VS Code, Vim, or similar
- **Bash 4+**: For script execution (on Linux, macOS, or WSL)

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/your-org/your-repo.git
cd metaHub

# Verify the directory structure
ls -la

# View documentation
cat README.md
cat CLAUDE.md
```

## Development Workflow

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/organizations.git
cd metaHub
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
# or
git checkout -b docs/documentation-update
```

**Branch naming conventions:**

- `feature/` - New features, documentation, scripts, or templates
- `fix/` - Bug fixes
- `docs/` - Documentation improvements
- `scripts/` - New or improved scripts
- `templates/` - New or updated templates

### 3. Make Your Changes

Follow these guidelines:

- **Documentation**: See [Documentation Standards](#documentation-standards)
- **Scripts**: See [Adding Scripts](#adding-scripts)
- **Templates**: See [Adding Templates](#adding-templates)
- Update CHANGELOG.md for your changes
- Update README.md if adding new features or documentation

### 4. Test Your Changes

For scripts:

```bash
# Make script executable
chmod +x scripts/your-script.sh

# Test the script
./scripts/your-script.sh --help
./scripts/your-script.sh

# Check for bashisms and style
shellcheck scripts/your-script.sh
```

For documentation:

```bash
# Check markdown formatting
# Look for broken links manually
# Ensure all internal links work

# Verify README renders correctly (copy to GitHub preview or use a local renderer)
```

### 5. Commit Your Changes

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```bash
# Documentation changes
git commit -m "docs: update getting-started guide"

# New scripts
git commit -m "scripts: add new compliance checker"

# New templates
git commit -m "templates: add new organization template"

# Bug fixes
git commit -m "fix: correct broken link in README"
```

### 6. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 7. Open a Pull Request

- Go to the original repository on GitHub
- Click "New Pull Request"
- Select your branch
- Fill out the PR template
- Request review

## Documentation Standards

### Markdown Files

- **Format**: GitHub Flavored Markdown
- **Line length**: 120 characters (soft wrap at 80 for readability in editors)
- **Headers**: Use `#` to `###` only (3 levels max per document)
- **Lists**: Use `-` for unordered lists, `1.` for ordered
- **Code blocks**: Use triple backticks with language identifier

### Documentation Structure

#### README.md

- Project/topic overview
- Quick start instructions
- Link to detailed docs in `docs/` directory
- Contact information

#### docs/ Subdirectories

- **getting-started/**: Onboarding guides
  - `overview.md` - What is this?
  - `setup.md` - How to set up?
  - `first-contribution.md` - How to contribute?

- **organizations/**: Organization and project documentation
  - One `.md` file per organization
  - Link back to main monorepo documentation

- **standards/**: Standards and guidelines
  - Repository standards
  - Compliance requirements
  - Testing standards
  - Security guidelines

- **architecture/**: Architecture documentation
  - Design decisions
  - System architecture
  - Project relationships

- **archive/**: Historical documentation (read-only)
  - Old docs for reference only
  - Clearly marked as historical

### Links

- **Internal links**: Use relative paths

  ```markdown
  [Getting Started](./docs/getting-started/overview.md)
  [AlaweinOS](./docs/organizations/AlaweinOS.md)
  ```

- **External links**: Use full URLs
  ```markdown
  [GitHub](https://github.com/your-org)
  ```

## Adding Scripts

### Location

Scripts should be organized by purpose:

```
scripts/
├── setup/         # Environment setup scripts
├── compliance/    # Compliance checking scripts
├── testing/       # Test automation scripts
├── utils/         # General utility scripts
└── validation/    # Validation scripts
```

### Script Template

```bash
#!/bin/bash

set -euo pipefail

# ============================================================================
# Script Name: Short Description
# ============================================================================
#
# Purpose: What does this script do?
# Usage: ./script-name.sh [OPTIONS]
# Author: Your Name (your@email.com)
# Date: YYYY-MM-DD
#
# ============================================================================

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ============================================================================
# Functions
# ============================================================================

print_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    -h, --help       Show this help message
    -v, --verbose    Enable verbose output

Examples:
    $(basename "$0")
    $(basename "$0") --verbose
EOF
}

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            *)
                echo "Unknown option: $1"
                print_help
                exit 1
                ;;
        esac
    done

    echo -e "${GREEN}Script executing...${NC}"
    # Add your script logic here
}

# ============================================================================
# Main Execution
# ============================================================================

main "$@"
```

### Script Requirements

- ✅ Use `#!/bin/bash` shebang
- ✅ Use `set -euo pipefail` for error handling
- ✅ Add help text with `-h` or `--help`
- ✅ Document purpose and usage
- ✅ Use meaningful variable names
- ✅ Add error handling
- ✅ Pass `shellcheck` validation
- ✅ Make executable: `chmod +x script-name.sh`

## Adding Templates

### Location

Templates should be in `templates/`:

```
templates/
├── python-project/      # Python project starter
├── typescript-project/  # TypeScript project starter
├── organization/        # New organization template
└── [other templates]/
```

### Template Structure

Each template should include:

- `README.md` - Template overview
- All mandatory files (LICENSE, .gitignore, CLAUDE.md, etc.)
- Example source code structure
- Example test structure
- Placeholder files showing expected patterns

### Creating a Template

1. Create a directory: `templates/your-template-name/`
2. Add mandatory files
3. Add example structure
4. Document in `templates/README.md`
5. Update root README with template reference

## Pull Request Process

### 1. Before Submitting

- [ ] Documentation is clear and accurate
- [ ] Scripts are tested and pass shellcheck
- [ ] Templates include all mandatory files
- [ ] Links are correct (internal relative, external absolute)
- [ ] CHANGELOG.md is updated
- [ ] README.md is updated (if applicable)

### 2. PR Description

Include:

- **What**: What changes were made?
- **Why**: Why are these changes necessary?
- **Related**: Link to related issues
- **Testing**: How was this tested?

### 3. Review Process

- Address reviewer feedback promptly
- Update code based on suggestions
- Re-request review after changes

### 4. Merge

- PRs require at least 1 approval
- All conversations must be resolved
- Squash and merge to main branch

## Reporting Issues

### Bug Reports

Use the GitHub issue tracker. Include:

- **Description**: Clear description of the bug
- **Location**: Where in metaHub (docs, scripts, templates)?
- **Expected**: What should happen?
- **Actual**: What actually happens?
- **Steps to Reproduce**: How to reproduce?
- **Suggestions**: How might it be fixed?

### Feature Requests

Include:

- **Description**: Clear description of the feature
- **Purpose**: Why is this needed?
- **Location**: Where in metaHub should this go?
- **Examples**: Examples or references

### Documentation Issues

Include:

- **Location**: Which document needs updating?
- **Issue**: What's unclear, incorrect, or missing?
- **Suggestion**: How should it be fixed?

## CHANGELOG Updates

When making changes, update CHANGELOG.md:

```markdown
## [Unreleased]

### Added

- New documentation on X topic
- New script for Y purpose
- New template for Z type of project

### Changed

- Updated documentation on A topic
- Improved script for B purpose

### Fixed

- Corrected link in C document
- Fixed error in D script
```

## Questions?

- **Email**: maintainer@example.com
- **Issues**: GitHub issue tracker
- **Discussions**: GitHub Discussions

## Thank You!

Thank you for contributing to metaHub and helping coordinate excellence across your ecosystem! 🎉
